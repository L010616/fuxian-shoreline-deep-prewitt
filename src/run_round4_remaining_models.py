"""Train remaining round-4 ablation and comparison models on the expanded data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_multidate_ablation_pilot as ablation
import run_round4_fusion_screen as screen
from fusion_strategies_round4 import segmentation_edge_loss
from run_multidate_ablation_pilot import AblationDFANet, MultiSceneDataset, seed_all, windows_for
from run_standard_model_comparison_pilot import build_models


ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "preexperiment"
OUT = PRE / "results/round4_remaining_models"
SEEDS = (20260913, 20260929, 20261017)
MAX_EPOCHS = 40
MIN_EPOCHS = 20
PATIENCE = 8


def model_factories():
    standard = build_models()
    return {
        "dfanet": lambda: AblationDFANet(False, None),
        "prewitt": lambda: AblationDFANet(True, "prewitt"),
        "unet_resnet18": standard["unet_resnet18"],
        "deeplabv3plus_resnet18": standard["deeplabv3plus_resnet18"],
        "segformer_mit_b0": standard["segformer_mit_b0"],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "remaining_model_results.json"
    if result_path.exists():
        raise RuntimeError("Round-4 remaining-model experiment already exists")
    frozen = json.loads((PRE / "results/round4_fusion_confirmation/final_model_spec.json").read_text(encoding="utf-8"))
    if frozen["status"] != "FINAL_MODEL_CONFIG_FROZEN_BEFORE_BLIND_TEST":
        raise RuntimeError("Final configuration is not frozen")
    scenes = screen.load_scenes()
    train = [s for s in scenes if s["record"]["split"] == "train"]
    validation = [s for s in scenes if s["record"]["split"] == "validation"]
    normalization = ablation.normalize_scenes(scenes)
    train_windows = windows_for(train); validation_windows = windows_for(validation)
    validation_loader = DataLoader(MultiSceneDataset(validation, validation_windows), batch_size=12,
                                   shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name, factory in model_factories().items():
        results[name] = {}
        for seed in SEEDS:
            seed_all(seed)
            train_loader = DataLoader(MultiSceneDataset(train, train_windows, augment=True), batch_size=12,
                                      shuffle=True, num_workers=0, pin_memory=True)
            model = factory().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best = float("inf"); best_epoch = 0; stale = 0; history = []
            checkpoint_path = OUT / f"{name}_seed{seed}_best.pt"
            for epoch in range(1, MAX_EPOCHS + 1):
                train_loss = screen.train_epoch(model, train_loader, device, optimizer, 0.0)
                val_loss = screen.validation_loss(model, validation_loader, device, 0.0)
                history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
                if val_loss < best - 1e-4:
                    best = val_loss; best_epoch = epoch; stale = 0
                    torch.save({"model": model.state_dict(), "epoch": epoch,
                                "validation_loss": val_loss, "config": name}, checkpoint_path)
                else:
                    stale += 1
                if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                    break
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            per_date, macro = screen.validate_full_scenes(model, validation, device)
            results[name][str(seed)] = {"best_epoch": best_epoch, "validation_loss": best,
                                       "validation_macro": macro, "per_date": per_date,
                                       "history": history}
            print(name, seed, best_epoch, round(macro["boundary_f1_30m"], 5),
                  round(macro["mean_symmetric_boundary_distance_m"], 2), flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = {}
    for name in results:
        summary[name] = {}
        for key in ("iou", "boundary_f1_30m", "boundary_f1_60m",
                    "mean_symmetric_boundary_distance_m", "hd95_m"):
            vals = np.asarray([results[name][str(seed)]["validation_macro"][key] for seed in SEEDS])
            summary[name][key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                                  "values": vals.tolist()}
    payload = {"status": "ROUND4_REMAINING_MODELS_COMPLETE", "test_data_accessed": False,
               "seeds": SEEDS, "training_windows": len(train_windows),
               "validation_windows": len(validation_windows), "normalization": normalization,
               "results": results, "summary": summary}
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "remaining_models_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["model", "bf1_30m_mean", "bf1_30m_sd", "asd_m_mean", "asd_m_sd", "iou_mean"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name, s in summary.items():
            writer.writerow({"model": name, "bf1_30m_mean": s["boundary_f1_30m"]["mean"],
                             "bf1_30m_sd": s["boundary_f1_30m"]["std"],
                             "asd_m_mean": s["mean_symmetric_boundary_distance_m"]["mean"],
                             "asd_m_sd": s["mean_symmetric_boundary_distance_m"]["std"],
                             "iou_mean": s["iou"]["mean"]})
    print(json.dumps({k: {"bf1": round(v["boundary_f1_30m"]["mean"], 5),
                                 "asd": round(v["mean_symmetric_boundary_distance_m"]["mean"], 2)}
                      for k, v in summary.items()}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
