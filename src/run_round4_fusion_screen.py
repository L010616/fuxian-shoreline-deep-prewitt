"""Validation-only screen of fusion strategy and edge-loss weight."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

import run_multidate_ablation_pilot as ablation
from evaluate_extended_boundary_support import metric as boundary_metric
from fusion_strategies_round4 import factories, segmentation_edge_loss
from run_multidate_ablation_pilot import MultiSceneDataset, predict, seed_all, windows_for
from run_quick_model_pilot import metrics as area_metrics


ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "preexperiment"
OLD_MANIFEST = PRE / "results/pilot_manifest_paired.json"
PROTOCOL = PRE / "results/round4_protocol/round4_split_protocol.json"
OLD_DRAFTS = PRE / "results/formal_trainval_drafts_v02"
NEW_DRAFTS = PRE / "results/round4_references/added_drafts"
NEW_SCREENED = PRE / "results/round4_references/added_screened"
OUT = PRE / "results/round4_fusion_screen"
SEED = 20260913
EDGE_WEIGHTS = (0.0, 0.1, 0.3, 0.5)
MAX_EPOCHS = 25
MIN_EPOCHS = 15
PATIENCE = 5


def load_scenes() -> list[dict]:
    old = [r for r in json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))["records"]
           if r["split"] in {"train", "validation"}]
    new = json.loads(PROTOCOL.read_text(encoding="utf-8"))["added_export_records"]
    scenes = []
    for record in old:
        ablation.DRAFTS = OLD_DRAFTS
        scene = ablation.read_scene(record)
        scene["round4_source"] = "existing"
        scenes.append(scene)
    for record in new:
        ablation.DRAFTS = NEW_DRAFTS
        scene = ablation.read_scene(record)
        with rasterio.open(NEW_SCREENED / scene["tag"] / "exclusion_mask_L8_30m.tif") as src:
            scene["valid"] &= ~src.read(1).astype(bool)
        scene["round4_source"] = "added_screened"
        scenes.append(scene)
    return scenes


def train_epoch(model, loader, device, optimizer, edge_weight):
    model.train(); totals = []
    for x, priors, y, mask in loader:
        x, priors, y, mask = [z.to(device, non_blocking=True) for z in (x, priors, y, mask)]
        loss, _, _ = segmentation_edge_loss(model(x, priors), y, mask, edge_weight)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        totals.append(float(loss.detach()))
    return float(np.mean(totals))


@torch.no_grad()
def validation_loss(model, loader, device, edge_weight):
    model.eval(); totals = []
    for x, priors, y, mask in loader:
        x, priors, y, mask = [z.to(device, non_blocking=True) for z in (x, priors, y, mask)]
        loss, _, _ = segmentation_edge_loss(model(x, priors), y, mask, edge_weight)
        totals.append(float(loss))
    return float(np.mean(totals))


def validate_full_scenes(model, scenes, device):
    rows = {}
    for scene in scenes:
        probability = predict(model, scene, device)
        conventional = area_metrics(probability, scene["y"], scene["valid"], scene["valid"] & scene["shore"])
        boundary = boundary_metric(probability, scene["y"], scene["valid"], scene["shore"])
        rows[scene["record"]["l8_date"]] = {**conventional, **boundary}
    keys = ("iou", "boundary_f1_30m", "boundary_f1_60m", "mean_symmetric_boundary_distance_m", "hd95_m")
    return rows, {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "fusion_screen_results.json"
    if result_path.exists():
        raise RuntimeError("Fusion screen already completed; refusing to rerun")
    final_manifest = json.loads((PRE / "results/round4_protocol/final_blind_six_date_manifest.json").read_text(encoding="utf-8"))
    if final_manifest["status"] != "FINAL_SIX_DATE_TEST_SET_SEALED_BEFORE_FUSION_TRAINING":
        raise RuntimeError("Final test was not sealed before fusion screening")

    seed_all(SEED)
    scenes = load_scenes()
    train = [s for s in scenes if s["record"]["split"] == "train"]
    validation = [s for s in scenes if s["record"]["split"] == "validation"]
    normalization = ablation.normalize_scenes(scenes)
    train_windows = windows_for(train)
    validation_windows = windows_for(validation)
    train_loader = DataLoader(MultiSceneDataset(train, train_windows, augment=True), batch_size=12,
                              shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(MultiSceneDataset(validation, validation_windows), batch_size=12,
                                   shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for fusion_name, factory in factories().items():
        for edge_weight in EDGE_WEIGHTS:
            seed_all(SEED)
            name = f"{fusion_name}_ew{edge_weight:.1f}".replace(".", "p")
            model = factory().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best = float("inf"); best_epoch = 0; stale = 0; history = []
            checkpoint = OUT / f"{name}_best.pt"
            for epoch in range(1, MAX_EPOCHS + 1):
                train_loss = train_epoch(model, train_loader, device, optimizer, edge_weight)
                val_loss = validation_loss(model, validation_loader, device, edge_weight)
                history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
                if val_loss < best - 1e-4:
                    best = val_loss; best_epoch = epoch; stale = 0
                    torch.save({"model": model.state_dict(), "epoch": epoch, "validation_loss": val_loss}, checkpoint)
                else:
                    stale += 1
                if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                    break
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            per_date, macro = validate_full_scenes(model, validation, device)
            row = {"config": name, "fusion": fusion_name, "edge_weight": edge_weight,
                   "best_epoch": best_epoch, "validation_loss": best,
                   "per_date": per_date, "validation_macro": macro, "history": history}
            if fusion_name == "parallel_weighted":
                w = (2 * torch.softmax(model.edges.operator_logits.detach().cpu(), dim=0)).tolist()
                row["learned_operator_weights"] = {"prewitt": w[0], "log": w[1]}
            results.append(row)
            print(name, best_epoch, round(macro["boundary_f1_30m"], 5),
                  round(macro["mean_symmetric_boundary_distance_m"], 2), flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    ranked = sorted(results, key=lambda r: (-r["validation_macro"]["boundary_f1_30m"],
                                            r["validation_macro"]["mean_symmetric_boundary_distance_m"]))
    payload = {
        "status": "VALIDATION_ONLY_FUSION_SCREEN_COMPLETE",
        "test_data_accessed": False,
        "seed": SEED,
        "training_dates": [s["record"]["l8_date"] for s in train],
        "validation_dates": [s["record"]["l8_date"] for s in validation],
        "training_windows": len(train_windows), "validation_windows": len(validation_windows),
        "normalization": normalization,
        "ranking_rule": "maximize validation Boundary F1 at 30 m; break ties by lower mean symmetric boundary distance",
        "top_two_for_three_seed_confirmation": [r["config"] for r in ranked[:2]],
        "ranking": [r["config"] for r in ranked],
        "results": results,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "fusion_screen_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["rank", "config", "fusion", "edge_weight", "best_epoch", "validation_loss",
                  "boundary_f1_30m", "boundary_f1_60m", "asd_m", "hd95_m", "iou"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            m = row["validation_macro"]
            writer.writerow({"rank": rank, "config": row["config"], "fusion": row["fusion"],
                             "edge_weight": row["edge_weight"], "best_epoch": row["best_epoch"],
                             "validation_loss": row["validation_loss"],
                             "boundary_f1_30m": m["boundary_f1_30m"],
                             "boundary_f1_60m": m["boundary_f1_60m"],
                             "asd_m": m["mean_symmetric_boundary_distance_m"],
                             "hd95_m": m["hd95_m"], "iou": m["iou"]})
    print(json.dumps({"top_two": payload["top_two_for_three_seed_confirmation"],
                      "train_windows": len(train_windows), "validation_windows": len(validation_windows)},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
