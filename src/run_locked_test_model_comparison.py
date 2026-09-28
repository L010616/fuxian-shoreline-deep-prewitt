"""One-time four-date test of all frozen comparison checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch

import run_multidate_ablation_pilot as ablation
from run_multidate_ablation_pilot import predict
from run_quick_model_pilot import metrics
from run_standard_model_comparison_pilot import build_models


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "preexperiment/results/fair_comparison_3seed_corrected_labels_v03"
OLD_DRAFTS = ROOT / "preexperiment/results/formal_test_drafts_v02"
NEW_DRAFTS = ROOT / "preexperiment/results/formal_test_drafts_v03_additional"
OLD_REF = OLD_DRAFTS / "manual_review_v01/pilot_test_reference_v01"
NEW_REF = NEW_DRAFTS / "screened_reference_v01"
OUT = ROOT / "preexperiment/results/locked_four_date_model_comparison_v01"
SEEDS = [20260913, 20260929, 20261017]
MODELS = ["proposed_parallel", "unet_resnet18", "deeplabv3plus_resnet18", "segformer_mit_b0"]
DATES = [
    {"l8_date": "2025-05-07", "s2_date": "2025-05-05", "split": "test", "wave": "old"},
    {"l8_date": "2025-12-01", "s2_date": "2025-12-01", "split": "test", "wave": "old"},
    {"l8_date": "2026-01-18", "s2_date": "2026-01-20", "split": "test", "wave": "new"},
    {"l8_date": "2026-02-03", "s2_date": "2026-02-04", "split": "test", "wave": "new"},
]
METRICS = ("precision", "recall", "f1", "iou", "boundary_f1_30m", "boundary_f1_60m", "mean_symmetric_boundary_distance_m")


def load_scenes():
    scenes = []
    for record in DATES:
        ablation.DRAFTS = OLD_DRAFTS if record["wave"] == "old" else NEW_DRAFTS
        scene = ablation.read_scene({k: record[k] for k in ("l8_date", "s2_date", "split")})
        scene["wave"] = record["wave"]
        scenes.append(scene)
    return scenes


def normalize_and_exclude(scenes, normalization):
    for scene in scenes:
        scene["x"] = np.empty_like(scene["x_raw"], dtype="float32")
        for b, stat in enumerate(normalization):
            scene["x"][b] = (np.clip(scene["x_raw"][b], stat["p01"], stat["p99"]) - stat["mean"]) / stat["std"]
        scene["x"][~np.isfinite(scene["x"])] = 0
        scene["priors"] = np.stack([scene["interior"], scene["shore"]]).astype("float32")
        if scene["wave"] == "old":
            path = OLD_REF / scene["tag"] / "pilot_exclusion_mask_L8_30m.tif"
        else:
            path = NEW_REF / scene["tag"] / "exclusion_mask_L8_30m.tif"
        with rasterio.open(path) as src:
            scene["valid"] &= ~src.read(1).astype(bool)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "model_comparison_test_results.json"
    if result_path.exists():
        raise RuntimeError("Locked model comparison already exists; refusing to rerun")
    old_manifest = json.loads((OLD_REF / "pilot_test_reference_manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((NEW_REF / "screened_reference_manifest.json").read_text(encoding="utf-8"))
    if not old_manifest.get("allow_one_time_pilot_test_evaluation") or not new_manifest.get("allow_one_time_evaluation"):
        raise RuntimeError("A reference wave is not frozen for evaluation")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    scenes = load_scenes()
    normalize_and_exclude(scenes, config["normalization"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factories = build_models()
    results = {}
    for name in MODELS:
        results[name] = {}
        for seed in SEEDS:
            model = factories[name]().to(device)
            checkpoint = torch.load(MODEL_DIR / f"{name}_seed{seed}_best.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            per_date = {}
            for scene in scenes:
                probability = predict(model, scene, device)
                raw = metrics(probability, scene["y"], scene["valid"], scene["valid"] & scene["shore"])
                constrained = probability.copy()
                constrained[scene["interior"]] = 1.0
                constrained[scene["roi"] & ~scene["shore"] & ~scene["interior"]] = 0.0
                constrained[~scene["roi"]] = 0.0
                common = metrics(constrained, scene["y"], scene["valid"], scene["valid"] & scene["shore"])
                per_date[scene["record"]["l8_date"]] = {"raw": raw, "common_constraint": common}
            results[name][str(seed)] = {
                "checkpoint_epoch": int(checkpoint["epoch"]), "per_date": per_date,
                "raw_macro": {m: float(np.mean([r["raw"][m] for r in per_date.values()])) for m in METRICS},
                "common_constraint_macro": {m: float(np.mean([r["common_constraint"][m] for r in per_date.values()])) for m in METRICS},
            }
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = {}
    for name in MODELS:
        summary[name] = {}
        for mode in ("raw_macro", "common_constraint_macro"):
            summary[name][mode] = {}
            for metric in METRICS:
                vals = np.asarray([results[name][str(seed)][mode][metric] for seed in SEEDS])
                summary[name][mode][metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)), "values": vals.tolist()}
    output = {
        "status": "LOCKED_FOUR_DATE_MODEL_COMPARISON_OPENED_ONCE",
        "models": MODELS, "seeds": SEEDS, "dates": [r["l8_date"] for r in DATES],
        "decision_threshold": 0.5,
        "evaluation_modes": {"raw": "direct model output", "common_constraint": "identical fixed hard spatial constraint applied to every model output"},
        "training_or_model_selection_after_test": False,
        "reference_status": "Conservatively screened Sentinel-2 references; not independent surveyed ground truth",
        "device": str(device), "results_by_model_and_seed": results, "summary": summary,
    }
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: summary[name]["raw_macro"] for name in MODELS}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
