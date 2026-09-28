"""Locked evaluation of the six-date manually reviewed shoreline subset.

The program deliberately refuses to run until all six consensus decisions and
reviewed outer-shoreline files exist.  It then rasterizes each reviewed outer
ring on the original 10-m Sentinel-2 grid and repeats inference with the frozen
three-seed checkpoints.  No training, threshold tuning, or date replacement is
performed here.
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
from rasterio.features import rasterize
from rasterio.warp import transform_geom

import run_q2_cross_lake_zero_shot as base
import run_round4_fusion_confirmation as fusion
from run_deep_edge_component_ablation import DeepSingleOperatorDFANet
from run_multidate_ablation_pilot import predict
from run_parameter_matched_conv_control import ParameterMatchedConvDFANet


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "preexperiment/results/q2_external/manual_review_core6"
OUT = ROOT / "preexperiment/results/q2_manual_core6_evaluation"
ALLOWED = {"ACCEPTED", "MODIFIED", "INVALID_SECTOR_ONLY"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review_rows() -> list[dict]:
    with (PACK / "manual_review_form.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    errors = []
    if len(rows) != 6:
        errors.append(f"expected six rows, found {len(rows)}")
    for row in rows:
        scene_id = row.get("scene_id", "UNKNOWN")
        if not row.get("reviewer_A", "").strip() or row.get("reviewer_A_status") not in ALLOWED:
            errors.append(f"{scene_id}: reviewer A name/status incomplete")
        if not row.get("reviewer_B", "").strip() or row.get("reviewer_B_status") not in ALLOWED:
            errors.append(f"{scene_id}: reviewer B name/status incomplete")
        if row.get("consensus_status") not in ALLOWED:
            errors.append(f"{scene_id}: consensus_status incomplete")
        expected = PACK / scene_id / "shoreline_REVIEWED.geojson"
        supplied = row.get("reviewed_file", "").strip()
        if supplied:
            candidate = Path(supplied)
            if not candidate.is_absolute():
                candidate = PACK / candidate
        else:
            candidate = expected
        row["reviewed_path"] = candidate
        if not candidate.exists():
            errors.append(f"{scene_id}: missing {candidate}")
    if errors:
        raise RuntimeError("MANUAL REVIEW GATE CLOSED:\n- " + "\n- ".join(errors))
    return rows


def reviewed_mask(reviewed: Path, template: Path) -> np.ndarray:
    payload = json.loads(reviewed.read_text(encoding="utf-8"))
    candidates = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = geometry.get("coordinates") or []
        if len(coords) < 4:
            continue
        candidates.append((float((feature.get("properties") or {}).get("length_m", len(coords))), coords))
    if len(candidates) != 1:
        raise RuntimeError(f"{reviewed}: retain exactly one target-lake outer LineString; found {len(candidates)}")
    coords = candidates[0][1]
    if max(abs(coords[0][0] - coords[-1][0]), abs(coords[0][1] - coords[-1][1])) > 1e-6:
        raise RuntimeError(f"{reviewed}: reviewed outer shoreline is not closed")
    with rasterio.open(template) as src:
        polygon = {"type": "Polygon", "coordinates": [coords]}
        projected = transform_geom("EPSG:4326", src.crs, polygon, precision=3)
        mask = rasterize([(projected, 1)], out_shape=(src.height, src.width),
                         transform=src.transform, fill=0, dtype="uint8")
    if int(mask.sum()) == 0:
        raise RuntimeError(f"{reviewed}: rasterized reviewed lake is empty")
    return mask.astype(bool)


def main() -> None:
    rows = review_rows()
    if OUT.exists():
        raise FileExistsError(f"Frozen output already exists: {OUT}")
    OUT.mkdir(parents=True)

    record_lookup = {(row["lake"], row["l8_date"]): row for row in base.selected_records()}
    scenes = []
    reviewed_files = []
    for review in rows:
        key = (review["lake"], review["l8_date"])
        if key not in record_lookup:
            raise RuntimeError(f"Review row is outside frozen 18-date set: {key}")
        record = record_lookup[key]
        scene = base.read_scene(record)
        truth10 = reviewed_mask(review["reviewed_path"], record["paths"]["draft"])
        # Reproject the human-reviewed water polygon to the already-defined L8 grid.
        with rasterio.open(record["paths"]["draft"]) as src:
            source_profile = src.profile.copy()
        fraction = np.full(scene["y"].shape, np.nan, dtype="float32")
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
        reproject(truth10.astype("float32"), fraction,
                  src_transform=source_profile["transform"], src_crs=source_profile["crs"],
                  dst_transform=scene["profile"]["transform"], dst_crs=scene["profile"]["crs"],
                  src_nodata=None, dst_nodata=np.nan, resampling=Resampling.average)
        scene["y"] = fraction >= 0.5
        scene["valid"] &= np.isfinite(fraction)
        scenes.append(scene)
        reviewed_files.append({"scene_id": review["scene_id"], "path": str(review["reviewed_path"]),
                               "sha256": sha256(review["reviewed_path"]),
                               "consensus_status": review["consensus_status"]})

    component = json.loads((base.COMPONENT / "deep_edge_component_ablation_results.json").read_text(encoding="utf-8"))
    base.normalize(scenes, component["normalization"])
    factories = {
        "spatial_prior": fusion.CONFIGS["prior"][0],
        "deep_prewitt": lambda: DeepSingleOperatorDFANet("prewitt"),
        "parameter_matched_ordinary_conv": ParameterMatchedConvDFANet,
    }
    checkpoints = {
        "spatial_prior": lambda seed: base.PRIOR / f"prior_seed{seed}_best.pt",
        "deep_prewitt": lambda seed: base.COMPONENT / f"deep_prewitt_only_seed{seed}_best.pt",
        "parameter_matched_ordinary_conv": lambda seed: base.CONTROL / f"matched_conv_seed{seed}_best.pt",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    averaged = {name: {} for name in base.MODELS}
    for name in base.MODELS:
        probs = {scene["tag"]: [] for scene in scenes}
        for seed in base.SEEDS:
            model = factories[name]().to(device)
            state = torch.load(checkpoints[name](seed), map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            for scene in scenes:
                probs[scene["tag"]].append(base.apply_frozen_spatial_constraint(predict(model, scene, device), scene))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for scene in scenes:
            probability = np.mean(probs[scene["tag"]], axis=0).astype("float32")
            result = base.evaluate(probability, scene)
            result.update({"lake": scene["record"]["lake"], "l8_date": scene["record"]["l8_date"],
                           "s2_date": scene["record"]["s2_date"], "season": scene["record"]["season"]})
            averaged[name][scene["tag"]] = result

    summary = {name: base.grouped_summary(averaged[name]) for name in base.MODELS}
    comparisons = {
        f"deep_prewitt_vs_{other}": {
            "boundary_f1_30m": base.paired_effect(averaged["deep_prewitt"], averaged[other], "boundary_f1_30m"),
            "asd_reduction_m": base.paired_effect(averaged["deep_prewitt"], averaged[other],
                                                   "mean_symmetric_boundary_distance_m", True),
        }
        for other in ("spatial_prior", "parameter_matched_ordinary_conv")
    }
    payload = {
        "status": "MANUAL_CORE6_EVALUATION_COMPLETE",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_changed_after_model_review": False,
        "training_or_tuning_on_core6": False,
        "threshold": 0.5,
        "reviewed_files": reviewed_files,
        "models": list(base.MODELS), "seeds": list(base.SEEDS),
        "three_seed_ensemble_per_date": averaged, "summary": summary,
        "paired_date_effects": comparisons,
    }
    (OUT / "manual_core6_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "manual_core6_per_date.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "lake", "l8_date", "s2_date", "season", *base.KEYS])
        for name in base.MODELS:
            for row in averaged[name].values():
                writer.writerow([name, row["lake"], row["l8_date"], row["s2_date"], row["season"],
                                 *[row[key] for key in base.KEYS]])
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
