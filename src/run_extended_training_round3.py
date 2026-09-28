"""Post-review convergence experiment with a larger common training budget.

This run is a robustness analysis prompted by the convergence audit.  It does
not restore the already-opened test set to an untouched state, and it never
uses test metrics for checkpoint selection.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_fair_comparison_3seed as fair
import run_locked_test_model_comparison as locked
from run_multidate_ablation_pilot import (
    AblationDFANet, MANIFEST, MultiSceneDataset, epoch, normalize_scenes,
    read_scene, seed_all, windows_for,
)
from run_standard_model_comparison_pilot import build_models as comparison_models


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preexperiment/results/extended_training_round3"
SEEDS = [20260913, 20260929, 20261017]
MAX_EPOCHS = 50
MIN_EPOCHS = 20
EARLY_STOP_PATIENCE = 8
SCHEDULER_PATIENCE = 3
IMPROVEMENT_EPS = 1e-4


def factories():
    standard = comparison_models()
    return {
        "dfanet": lambda: AblationDFANet(False, None),
        "prior": lambda: AblationDFANet(True, None),
        "log": lambda: AblationDFANet(True, "log"),
        "prewitt": lambda: AblationDFANet(True, "prewitt"),
        "parallel": lambda: AblationDFANet(True, "parallel"),
        "unet_resnet18": standard["unet_resnet18"],
        "deeplabv3plus_resnet18": standard["deeplabv3plus_resnet18"],
        "segformer_mit_b0": standard["segformer_mit_b0"],
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(results, split):
    answer = {}
    for name in factories():
        answer[name] = {}
        for mode in ("raw", "common_constraint"):
            cells = [results[str(seed)][name][split][mode]["macro"] for seed in SEEDS]
            answer[name][mode] = {}
            for metric in cells[0]:
                values = np.asarray([c[metric] for c in cells], dtype=float)
                answer[name][mode][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "values": values.tolist(),
                }
    return answer


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))["records"]
    trainval = [read_scene(r) for r in records if r["split"] in {"train", "validation"}]
    train = [s for s in trainval if s["record"]["split"] == "train"]
    val = [s for s in trainval if s["record"]["split"] == "validation"]
    normalization = normalize_scenes(trainval)
    train_windows, val_windows = windows_for(train), windows_for(val)

    test = locked.load_scenes()
    locked.normalize_and_exclude(test, normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = OUT / "results_by_seed.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    log_path = OUT / "training_log.csv"
    write_header = not log_path.exists()

    protocol = {
        "status": "POST_REVIEW_CONVERGENCE_ROBUSTNESS",
        "reason": "14 of 27 original runs selected their best checkpoint at the 15-epoch budget boundary",
        "seeds": SEEDS,
        "models": list(factories()),
        "max_epochs": MAX_EPOCHS,
        "minimum_epochs_before_early_stop": MIN_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "batch_size": 12,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "checkpoint_selection": "minimum validation loss only",
        "test_metrics_used_for_selection": False,
        "test_status": "Previously opened; used only as a post-review robustness set",
        "training_dates": [s["record"]["l8_date"] for s in train],
        "validation_dates": [s["record"]["l8_date"] for s in val],
        "test_dates": [s["record"]["l8_date"] for s in test],
        "train_patches": len(train_windows),
        "validation_patches": len(val_windows),
        "encoder_weights": None,
        "normalization": normalization,
        "device": str(device),
    }
    write_json(OUT / "protocol.json", protocol)

    with log_path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "seed", "model", "epoch", "train_loss", "val_loss", "learning_rate"
        ])
        if write_header:
            writer.writeheader()
        for seed in SEEDS:
            results.setdefault(str(seed), {})
            for name, factory in factories().items():
                if name in results[str(seed)] and results[str(seed)][name].get("complete"):
                    print(f"skip completed seed={seed} model={name}", flush=True)
                    continue
                seed_all(seed)
                train_loader = DataLoader(
                    MultiSceneDataset(train, train_windows, True), batch_size=12,
                    shuffle=True, num_workers=0, pin_memory=True,
                )
                val_loader = DataLoader(
                    MultiSceneDataset(val, val_windows, False), batch_size=12,
                    shuffle=False, num_workers=0, pin_memory=True,
                )
                model = factory().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=SCHEDULER_PATIENCE, factor=0.5
                )
                best, stale, started = math.inf, 0, time.time()
                checkpoint_path = OUT / f"{name}_seed{seed}_best.pt"
                epochs_run = 0
                for ep in range(1, MAX_EPOCHS + 1):
                    train_loss = epoch(model, train_loader, device, optimizer)
                    val_loss = epoch(model, val_loader, device)
                    scheduler.step(val_loss)
                    lr = optimizer.param_groups[0]["lr"]
                    writer.writerow({"seed": seed, "model": name, "epoch": ep,
                                     "train_loss": train_loss, "val_loss": val_loss,
                                     "learning_rate": lr})
                    stream.flush()
                    epochs_run = ep
                    print(f"seed={seed} {name} epoch={ep:02d} train={train_loss:.5f} val={val_loss:.5f} lr={lr:.2e}", flush=True)
                    if val_loss < best - IMPROVEMENT_EPS:
                        best, stale = val_loss, 0
                        torch.save({"model": model.state_dict(), "epoch": ep,
                                    "val_loss": val_loss}, checkpoint_path)
                    else:
                        stale += 1
                    if ep >= MIN_EPOCHS and stale >= EARLY_STOP_PATIENCE:
                        break
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model"])
                results[str(seed)][name] = {
                    "complete": True,
                    "best_epoch": int(checkpoint["epoch"]),
                    "epochs_run": epochs_run,
                    "best_val_loss": float(checkpoint["val_loss"]),
                    "parameters": int(sum(p.numel() for p in model.parameters())),
                    "training_seconds": time.time() - started,
                    "validation": fair.evaluate(model, val, device),
                    "post_review_test": fair.evaluate(model, test, device),
                }
                write_json(results_path, results)
                print(f"COMPLETE seed={seed} model={name} best_epoch={checkpoint['epoch']} epochs_run={epochs_run}", flush=True)
                del model, optimizer, train_loader, val_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_json(OUT / "validation_summary.json", summarize(results, "validation"))
    write_json(OUT / "post_review_test_summary.json", summarize(results, "post_review_test"))
    convergence = []
    for seed in SEEDS:
        for name in factories():
            r = results[str(seed)][name]
            convergence.append({"seed": seed, "model": name,
                                "best_epoch": r["best_epoch"],
                                "epochs_run": r["epochs_run"],
                                "best_at_epoch_50": r["best_epoch"] == MAX_EPOCHS})
    write_json(OUT / "convergence_summary.json", convergence)
    print("EXTENDED_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
