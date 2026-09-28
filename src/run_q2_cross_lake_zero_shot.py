"""Frozen zero-shot transfer from Fuxian Lake to three external lakes.

The script never trains or tunes a model on the external lakes. It evaluates
three already-frozen three-seed configurations on six dates each for Dianchi
Waihai, Erhai and Xingyun Lake. External Sentinel-2 masks are automatic,
prediction-blind NDWI drafts and therefore produce provisional sensitivity
results, not surveyed-ground-truth accuracy claims.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform
from scipy import ndimage as ndi
from scipy.stats import wilcoxon

import q2_external_reference_drafts as external
import run_round4_fusion_confirmation as fusion
from evaluate_extended_boundary_support import metric as boundary_metric
from run_deep_edge_component_ablation import DeepSingleOperatorDFANet
from run_multidate_ablation_pilot import predict
from run_parameter_matched_conv_control import ParameterMatchedConvDFANet
from run_quick_model_pilot import metrics as area_metrics


ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "preexperiment/results/q2_external/candidate_qc.csv"
DRAFTS = ROOT / "preexperiment/results/q2_external/reference_drafts"
COMPONENT = ROOT / "preexperiment/results/deep_edge_component_ablation"
PRIOR = ROOT / "preexperiment/results/round4_fusion_confirmation"
CONTROL = ROOT / "preexperiment/results/parameter_matched_conv_control"
OUT = ROOT / "preexperiment/results/q2_cross_lake_zero_shot_v04_waihai_target"
REFERENCE_FREEZE = (ROOT / "preexperiment/results/q2_external/screened_reference_freeze/"
                    "external_reference_freeze_manifest.json")
SEEDS = (20260913, 20260929, 20261017)
MODELS = (
    "spatial_prior",
    "deep_prewitt",
    "parameter_matched_ordinary_conv",
)
KEYS = (
    "precision", "recall", "f1", "iou", "boundary_f1_30m",
    "boundary_f1_60m", "mean_symmetric_boundary_distance_m", "hd95_m",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: str) -> str:
    return value.replace("-", "")


def season(date: str) -> str:
    month = int(compact(date)[4:6])
    return "dry" if month in (11, 12, 1, 2, 3, 4) else "wet"


def selected_records() -> list[dict]:
    with QC.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["preliminary_quality_pass"] == "True"]
    records = []
    for row in rows:
        lake, l8_date, s2_date = row["lake"], row["l8_date"], row["s2_date"]
        draft = DRAFTS / lake / l8_date / "S2_water_DRAFT_10m.tif"
        paths = {
            "l8": external.locate(lake, l8_date, s2_date, "L8"),
            "s2": external.locate(lake, l8_date, s2_date, "S2"),
            "qa": external.locate(lake, l8_date, s2_date, "QA_JRC"),
            "draft": draft,
        }
        if not all(path.exists() for path in paths.values()):
            raise FileNotFoundError(f"Incomplete external pair: {lake} {l8_date}")
        records.append({
            "lake": lake, "l8_date": l8_date, "s2_date": s2_date,
            "season": season(l8_date), "paths": paths,
        })
    counts = {lake: sum(row["lake"] == lake for row in records)
              for lake in ("Dianchi_Waihai", "Erhai", "Xingyun")}
    seasonal = {(lake, phase): sum(row["lake"] == lake and row["season"] == phase
                                   for row in records)
                for lake in counts for phase in ("dry", "wet")}
    if counts != {"Dianchi_Waihai": 6, "Erhai": 6, "Xingyun": 6}:
        raise RuntimeError(f"Expected six dates per lake, got {counts}")
    if any(value != 3 for value in seasonal.values()):
        raise RuntimeError(f"Expected three dry and three wet dates per lake, got {seasonal}")
    return records


def read_scene(record: dict) -> dict:
    paths = record["paths"]
    with rasterio.open(paths["l8"]) as src:
        x = src.read(range(1, 7)).astype("float32")
        profile = src.profile.copy()
        l8_transform = src.transform
        crs = src.crs
    with rasterio.open(paths["qa"]) as qa:
        masks = qa.read()
        lake = external.main_component(masks[0] >= 2, qa, record["lake"])
        if record["lake"] == "Dianchi_Waihai":
            # The JRC seed-connected water also contains Caohai through a narrow
            # channel. The named target is Waihai, so use a fixed projected
            # north boundary (about 24.97 N), independent of imagery date and
            # model output, and retain the component containing the Waihai seed.
            rows = np.arange(qa.height)
            northing = qa.transform.f + (rows + 0.5) * qa.transform.e
            lake &= northing[:, None] <= 2763300.0
            labels, count = ndi.label(lake, structure=np.ones((3, 3), dtype=np.uint8))
            seed_x, seed_y = transform("EPSG:4326", qa.crs, [102.7049], [24.8379])
            col, row = (~qa.transform) * (seed_x[0], seed_y[0])
            selected = int(labels[round(row), round(col)])
            if count == 0 or selected == 0:
                raise RuntimeError("Fixed Waihai target mask lost the lake-center seed")
            lake = labels == selected
        lake = ndi.binary_fill_holes(lake)
    interior = ndi.binary_erosion(lake, iterations=10)
    outer = ndi.binary_dilation(lake, iterations=10)
    shore = outer & ~interior
    roi = interior | shore
    joint = masks[3] > 0
    with rasterio.open(paths["draft"]) as src:
        source = src.read(1)
        source = np.where(source == 255, np.nan, source).astype("float32")
        fraction = np.full((profile["height"], profile["width"]), np.nan,
                           dtype="float32")
        reproject(source, fraction, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=l8_transform, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.average)
    truth = fraction >= 0.5
    finite = np.all(np.isfinite(x) & (x > -9000), axis=0)
    valid = finite & joint & np.isfinite(fraction) & roi
    if int((valid & shore).sum()) < 80:
        raise RuntimeError(f"Insufficient valid shoreline: {record['lake']} {record['l8_date']}")
    return {
        "tag": f"{record['lake']}_{record['l8_date']}", "record": record,
        "x_raw": x, "y": truth, "valid": valid, "shore": shore,
        "roi": roi, "interior": interior, "profile": profile,
    }


def normalize(scenes: list[dict], stats: list[dict]) -> None:
    for scene in scenes:
        scene["x"] = np.empty_like(scene["x_raw"], dtype="float32")
        for band, row in enumerate(stats):
            scene["x"][band] = ((np.clip(scene["x_raw"][band], row["p01"], row["p99"])
                                  - row["mean"]) / row["std"])
        scene["x"][~np.isfinite(scene["x"])] = 0
        scene["priors"] = np.stack([scene["interior"], scene["shore"]]).astype("float32")


def evaluate(probability: np.ndarray, scene: dict) -> dict:
    valid, shore = scene["valid"], scene["shore"]
    return {
        **area_metrics(probability, scene["y"], valid, valid & shore),
        **boundary_metric(probability, scene["y"], valid, shore),
        "valid_pixels": int(valid.sum()),
        "valid_shore_pixels": int((valid & shore).sum()),
    }


def apply_frozen_spatial_constraint(probability: np.ndarray, scene: dict) -> np.ndarray:
    """Apply the prespecified deterministic prior used by the final method."""
    constrained = probability.copy()
    constrained[scene["interior"]] = 1.0
    constrained[~scene["roi"]] = 0.0
    return constrained


def write_prediction(path: Path, probability: np.ndarray, scene: dict) -> None:
    profile = scene["profile"].copy()
    profile.update(driver="GTiff", count=1, dtype="float32", nodata=-9999.0,
                   compress="deflate", predictor=3)
    output = np.full(probability.shape, -9999.0, dtype="float32")
    output[scene["valid"]] = probability[scene["valid"]]
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)
        dst.descriptions = ("three_seed_mean_probability",)


def grouped_summary(per_date: dict) -> dict:
    groups: dict[str, list[dict]] = {"all_18_dates": []}
    for lake in ("Dianchi_Waihai", "Erhai", "Xingyun"):
        groups[f"lake:{lake}"] = []
    for phase in ("dry", "wet"):
        groups[f"season:{phase}"] = []
    for row in per_date.values():
        groups["all_18_dates"].append(row)
        groups[f"lake:{row['lake']}"] .append(row)
        groups[f"season:{row['season']}"] .append(row)
    output = {}
    for name, rows in groups.items():
        output[name] = {"n_dates": len(rows)}
        for key in KEYS:
            values = np.asarray([row[key] for row in rows], dtype="float64")
            output[name][key] = {
                "mean": float(values.mean()),
                "date_sd": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
    return output


def paired_effect(candidate: dict, comparator: dict, key: str,
                  lower_is_better: bool = False) -> dict:
    ids = sorted(candidate)
    values = np.asarray([
        (comparator[item][key] - candidate[item][key]) if lower_is_better
        else (candidate[item][key] - comparator[item][key])
        for item in ids
    ], dtype="float64")
    rng = np.random.default_rng(20260927)
    bootstrap = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(1)
    try:
        p_value = float(wilcoxon(values, alternative="two-sided", method="approx").pvalue)
    except ValueError:
        p_value = 1.0
    return {
        "n_dates": len(values), "mean_improvement": float(values.mean()),
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)),
                            float(np.quantile(bootstrap, 0.975))],
        "dates_improved": int((values > 0).sum()),
        "dates_tied": int((values == 0).sum()),
        "dates_worsened": int((values < 0).sum()),
        "wilcoxon_two_sided_p": p_value,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    prediction_dir = OUT / "deep_prewitt_predictions"
    prediction_dir.mkdir()
    freeze = json.loads(REFERENCE_FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "EXTERNAL_REFERENCE_SET_FROZEN_ALL_18_DATES_RETAINED":
        raise RuntimeError("External references are not frozen")
    if not freeze["all_dates_retained"] or freeze["date_count"] != 18:
        raise RuntimeError("Frozen reference set is incomplete")
    records = selected_records()
    scenes = [read_scene(record) for record in records]
    component = json.loads((COMPONENT / "deep_edge_component_ablation_results.json").read_text(encoding="utf-8"))
    stats = component["normalization"]
    normalize(scenes, stats)

    factories = {
        "spatial_prior": fusion.CONFIGS["prior"][0],
        "deep_prewitt": lambda: DeepSingleOperatorDFANet("prewitt"),
        "parameter_matched_ordinary_conv": ParameterMatchedConvDFANet,
    }
    checkpoints = {
        "spatial_prior": lambda seed: PRIOR / f"prior_seed{seed}_best.pt",
        "deep_prewitt": lambda seed: COMPONENT / f"deep_prewitt_only_seed{seed}_best.pt",
        "parameter_matched_ordinary_conv": lambda seed: CONTROL / f"matched_conv_seed{seed}_best.pt",
    }
    frozen = {name: {str(seed): {"path": str(checkpoints[name](seed).relative_to(ROOT)).replace("\\", "/"),
                                  "sha256": sha256(checkpoints[name](seed))}
                     for seed in SEEDS} for name in MODELS}
    input_manifest = [{
        "lake": row["lake"], "l8_date": row["l8_date"],
        "s2_date": row["s2_date"], "season": row["season"],
        "files": {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                           "sha256": sha256(path)}
                  for name, path in row["paths"].items()},
    } for row in records]
    (OUT / "frozen_evaluation_manifest.json").write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "FROZEN_FUXIAN_TO_THREE_LAKE_ZERO_SHOT_EVALUATION",
        "external_training_or_tuning": False,
        "reference_status": "prediction-blind automatic S2 drafts frozen after model-independent visual review; all dates retained",
        "reference_freeze_manifest": str(REFERENCE_FREEZE.relative_to(ROOT)).replace("\\", "/"),
        "reference_freeze_sha256": sha256(REFERENCE_FREEZE),
        "hard_spatial_constraint_applied": True,
        "prior_boundary_definition": "outer shoreline of the named target lake; internal holes filled before the 300 m band is generated; Dianchi Waihai separated from Caohai by fixed northing 2763300 m in EPSG:32648",
        "records": input_manifest, "checkpoints": frozen,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw: dict[str, dict] = {name: {} for name in MODELS}
    averaged: dict[str, dict] = {name: {} for name in MODELS}
    for name in MODELS:
        probabilities = {scene["tag"]: [] for scene in scenes}
        for seed in SEEDS:
            model = factories[name]().to(device)
            state = torch.load(checkpoints[name](seed), map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            raw[name][str(seed)] = {"checkpoint_epoch": int(state["epoch"]), "per_date": {}}
            for scene in scenes:
                probability = apply_frozen_spatial_constraint(predict(model, scene, device), scene)
                probabilities[scene["tag"]].append(probability)
                raw[name][str(seed)]["per_date"][scene["tag"]] = evaluate(probability, scene)
            print(f"completed {name} seed={seed}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for scene in scenes:
            probability = np.mean(probabilities[scene["tag"]], axis=0).astype("float32")
            row = evaluate(probability, scene)
            row.update({"lake": scene["record"]["lake"],
                        "l8_date": scene["record"]["l8_date"],
                        "s2_date": scene["record"]["s2_date"],
                        "season": scene["record"]["season"]})
            averaged[name][scene["tag"]] = row
            if name == "deep_prewitt":
                write_prediction(prediction_dir / f"{scene['tag']}_probability.tif",
                                 probability, scene)

    summary = {name: grouped_summary(averaged[name]) for name in MODELS}
    comparisons = {}
    for comparator in ("spatial_prior", "parameter_matched_ordinary_conv"):
        comparisons[f"deep_prewitt_vs_{comparator}"] = {
            "boundary_f1_30m": paired_effect(averaged["deep_prewitt"], averaged[comparator],
                                              "boundary_f1_30m"),
            "asd_reduction_m": paired_effect(averaged["deep_prewitt"], averaged[comparator],
                                               "mean_symmetric_boundary_distance_m", True),
        }
    payload = {
        "status": "ZERO_SHOT_EVALUATION_COMPLETE_PROVISIONAL_AUTOMATIC_REFERENCES",
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_lake": "Fuxian", "target_lakes": ["Dianchi_Waihai", "Erhai", "Xingyun"],
        "dates_per_lake": 6, "dry_dates_per_lake": 3, "wet_dates_per_lake": 3,
        "external_training_or_tuning": False, "decision_threshold": 0.5,
        "hard_spatial_constraint_applied": True,
        "prior_boundary_definition": "outer shoreline only; internal JRC holes filled; Dianchi Waihai uses fixed model-independent separation from Caohai at EPSG:32648 northing 2763300 m",
        "reference_freeze_manifest_sha256": sha256(REFERENCE_FREEZE),
        "reference_status": "Prediction-blind automatic S2 NDWI drafts; all 18 dates retained after model-independent review",
        "models": list(MODELS), "seeds": list(SEEDS), "normalization": stats,
        "raw_seed_results": raw, "three_seed_ensemble_per_date": averaged,
        "summary": summary, "paired_date_effects": comparisons,
    }
    (OUT / "cross_lake_zero_shot_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "per_date_three_seed_ensemble.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "lake", "l8_date", "s2_date", "season", *KEYS])
        for name in MODELS:
            for row in averaged[name].values():
                writer.writerow([name, row["lake"], row["l8_date"], row["s2_date"], row["season"],
                                 *[row[key] for key in KEYS]])
    with (OUT / "group_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "group", "n_dates", "metric", "mean", "date_sd"])
        for name in MODELS:
            for group, values in summary[name].items():
                for key in KEYS:
                    writer.writerow([name, group, values["n_dates"], key,
                                     values[key]["mean"], values[key]["date_sd"]])
    brief = {
        name: {group: {"IoU": round(values["iou"]["mean"], 4),
                       "BF1_30m": round(values["boundary_f1_30m"]["mean"], 4),
                       "ASD_m": round(values["mean_symmetric_boundary_distance_m"]["mean"], 2)}
               for group, values in summary[name].items()}
        for name in MODELS
    }
    print(json.dumps({"summary": brief, "comparisons": comparisons}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
