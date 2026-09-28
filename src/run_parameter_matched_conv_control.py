"""Train and freeze a parameter-matched ordinary-convolution edge control.

This script reads only the existing 12 training and 3 validation dates. It never
opens the post-2019 confirmatory imagery or references. The control replaces the
fixed LoG/Prewitt responses with a learned 3x3 Conv-BN-ReLU projection from the
same green and NIR inputs to four feature maps; every downstream layer is kept
identical to the selected depth-2 edge model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import run_multidate_ablation_pilot as ablation
import run_round4_fusion_screen as screen
from run_edge_depth_validation_screen import DeeperEdgeDFANet, EdgeEncoder
from run_multidate_ablation_pilot import MultiSceneDataset, seed_all, windows_for
from run_quick_model_pilot import ConvBNAct


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preexperiment" / "results" / "parameter_matched_conv_control"
SEEDS = (20260913, 20260929, 20261017)
MAX_EPOCHS = 40
MIN_EPOCHS = 20
PATIENCE = 8


class LearnedOrdinaryEdges(nn.Module):
    """Learn four local feature maps from green and NIR with an ordinary 3x3 conv."""

    def __init__(self):
        super().__init__()
        self.project = ConvBNAct(2, 4, 3, 1)

    def forward(self, x):
        return self.project(torch.cat([x[:, 1:2], x[:, 3:4]], dim=1))


class ParameterMatchedConvDFANet(DeeperEdgeDFANet):
    def __init__(self):
        super().__init__(depth=2)
        self.edge_mode = "learned_ordinary_conv"
        self.edges = LearnedOrdinaryEdges()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "parameter_matched_conv_validation.json"
    if result_path.exists():
        raise RuntimeError("Matched-convolution control already exists; refusing to overwrite")

    scenes = screen.load_scenes()
    train = [s for s in scenes if s["record"]["split"] == "train"]
    val = [s for s in scenes if s["record"]["split"] == "validation"]
    normalization = ablation.normalize_scenes(scenes)
    train_windows, val_windows = windows_for(train), windows_for(val)
    val_loader = DataLoader(
        MultiSceneDataset(val, val_windows), batch_size=12, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_params = sum(p.numel() for p in DeeperEdgeDFANet(2).parameters())
    control_params = sum(p.numel() for p in ParameterMatchedConvDFANet().parameters())
    results = {}

    for seed in SEEDS:
        seed_all(seed)
        train_loader = DataLoader(
            MultiSceneDataset(train, train_windows, augment=True), batch_size=12,
            shuffle=True, num_workers=0, pin_memory=True,
        )
        model = ParameterMatchedConvDFANet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, stale, history = float("inf"), 0, []
        checkpoint_path = OUT / f"matched_conv_seed{seed}_best.pt"
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = screen.train_epoch(model, train_loader, device, optimizer, 0.0)
            validation_loss = screen.validation_loss(model, val_loader, device, 0.0)
            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "validation_loss": validation_loss,
            })
            if validation_loss < best - 1e-4:
                best, stale = validation_loss, 0
                torch.save({
                    "model": model.state_dict(), "epoch": epoch,
                    "architecture": "parameter_matched_ordinary_conv_depth2",
                }, checkpoint_path)
            else:
                stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                break

        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        per_date, macro = screen.validate_full_scenes(model, val, device)
        results[str(seed)] = {
            "best_epoch": int(state["epoch"]),
            "stop_epoch": int(history[-1]["epoch"]),
            "best_validation_loss": float(best),
            "parameters": control_params,
            "per_date": per_date,
            "validation_macro": macro,
            "history": history,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256(checkpoint_path),
        }
        print(
            "matched_conv", seed, "best", state["epoch"], "stop", history[-1]["epoch"],
            "BF1", round(macro["boundary_f1_30m"], 6),
            "ASD", round(macro["mean_symmetric_boundary_distance_m"], 3), flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    keys = ("iou", "boundary_f1_30m", "boundary_f1_60m", "mean_symmetric_boundary_distance_m", "hd95_m")
    summary = {}
    for key in keys:
        values = np.asarray([results[str(s)]["validation_macro"][key] for s in SEEDS])
        summary[key] = {
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "values": values.tolist(),
        }

    payload = {
        "status": "PARAMETER_MATCHED_CONTROL_FROZEN_BEFORE_CONFIRMATORY_REFERENCE_CREATION",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture": (
            "Green and NIR -> learned ordinary 3x3 Conv-BN-ReLU -> four maps; "
            "same four-scale depth-2 residual edge encoder and fusion decoder as LoG-Prewitt model"
        ),
        "selection_data": "12 training dates and 3 validation dates only",
        "post2019_confirmatory_data_opened": False,
        "seeds": list(SEEDS),
        "training_dates": [s["record"]["l8_date"] for s in train],
        "validation_dates": [s["record"]["l8_date"] for s in val],
        "training_windows": len(train_windows),
        "validation_windows": len(val_windows),
        "normalization": normalization,
        "fixed_log_prewitt_parameters": fixed_params,
        "ordinary_conv_parameters": control_params,
        "absolute_parameter_difference": control_params - fixed_params,
        "relative_parameter_difference_percent": 100 * (control_params - fixed_params) / fixed_params,
        "summary": summary,
        "results": results,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "epoch", "train_loss", "validation_loss"])
        for seed in SEEDS:
            for row in results[str(seed)]["history"]:
                writer.writerow([seed, row["epoch"], row["train_loss"], row["validation_loss"]])

    print(json.dumps({
        "parameters": {"fixed": fixed_params, "control": control_params,
                       "difference_percent": payload["relative_parameter_difference_percent"]},
        "summary": summary,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
