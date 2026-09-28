"""Validation-only component ablation for the deep fixed edge branch.

The LoG-only and Prewitt-only variants keep the complete depth-2 trainable
architecture and parameter count of the selected parallel model. Their two
fixed response maps are duplicated with 1/sqrt(2) scaling to preserve the
four-channel interface and response energy. No confirmation imagery is read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import run_multidate_ablation_pilot as ablation
import run_round4_fusion_screen as screen
from run_edge_depth_validation_screen import DeeperEdgeDFANet
from run_multidate_ablation_pilot import FixedEdgeMode, MultiSceneDataset, seed_all, windows_for


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preexperiment/results/deep_edge_component_ablation"
DEPTH_RESULT = ROOT / "preexperiment/results/edge_depth_validation_screen/edge_depth_validation_results.json"
CONTROL_RESULT = ROOT / "preexperiment/results/parameter_matched_conv_control/parameter_matched_conv_validation.json"
PRIOR_RESULT = ROOT / "preexperiment/results/round4_fusion_confirmation/fusion_confirmation_results.json"
SEEDS = (20260913, 20260929, 20261017)
MAX_EPOCHS = 40
MIN_EPOCHS = 20
PATIENCE = 8
KEYS = ("iou", "boundary_f1_30m", "boundary_f1_60m", "mean_symmetric_boundary_distance_m", "hd95_m")


class FourChannelSingleOperator(nn.Module):
    """Hold the selected branch width fixed while isolating one fixed operator."""

    def __init__(self, mode: str):
        super().__init__()
        if mode not in ("log", "prewitt"):
            raise ValueError(mode)
        self.mode = mode
        self.operator = FixedEdgeMode(mode)

    def forward(self, x):
        response = self.operator(x)
        return torch.cat([response, response], dim=1) / math.sqrt(2.0)


class DeepSingleOperatorDFANet(DeeperEdgeDFANet):
    def __init__(self, mode: str):
        super().__init__(depth=2)
        self.edge_mode = f"fixed_{mode}_only_four_channel"
        self.edges = FourChannelSingleOperator(mode)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats_from_runs(runs: dict) -> dict:
    summary = {"parameters": runs[str(SEEDS[0])]["parameters"]}
    for key in KEYS:
        values = np.asarray([runs[str(seed)]["validation_macro"][key] for seed in SEEDS], dtype="float64")
        summary[key] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "values": values.tolist()}
    summary["best_epoch"] = [runs[str(seed)]["best_epoch"] for seed in SEEDS]
    summary["stop_epoch"] = [runs[str(seed)]["stop_epoch"] for seed in SEEDS]
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "deep_edge_component_ablation_results.json"
    if result_path.exists():
        raise RuntimeError("Deep edge component ablation already exists; refusing to overwrite")

    scenes = screen.load_scenes()
    train = [scene for scene in scenes if scene["record"]["split"] == "train"]
    validation = [scene for scene in scenes if scene["record"]["split"] == "validation"]
    normalization = ablation.normalize_scenes(scenes)
    train_windows, validation_windows = windows_for(train), windows_for(validation)
    validation_loader = DataLoader(
        MultiSceneDataset(validation, validation_windows), batch_size=12, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_parameters = sum(parameter.numel() for parameter in DeeperEdgeDFANet(2).parameters())
    runs = {}

    for mode in ("log", "prewitt"):
        name = f"deep_{mode}_only"
        runs[name] = {}
        for seed in SEEDS:
            seed_all(seed)
            train_loader = DataLoader(
                MultiSceneDataset(train, train_windows, augment=True), batch_size=12,
                shuffle=True, num_workers=0, pin_memory=True,
            )
            model = DeepSingleOperatorDFANet(mode).to(device)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            if parameters != target_parameters:
                raise RuntimeError(f"Parameter mismatch for {name}: {parameters} != {target_parameters}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best, stale, history = float("inf"), 0, []
            checkpoint = OUT / f"{name}_seed{seed}_best.pt"
            for epoch in range(1, MAX_EPOCHS + 1):
                training_loss = screen.train_epoch(model, train_loader, device, optimizer, 0.0)
                validation_loss = screen.validation_loss(model, validation_loader, device, 0.0)
                history.append({"epoch": epoch, "train_loss": training_loss, "validation_loss": validation_loss})
                if validation_loss < best - 1e-4:
                    best, stale = validation_loss, 0
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "mode": mode, "depth": 2},
                        checkpoint,
                    )
                else:
                    stale += 1
                if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                    break
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            per_date, macro = screen.validate_full_scenes(model, validation, device)
            runs[name][str(seed)] = {
                "best_epoch": int(state["epoch"]),
                "stop_epoch": int(history[-1]["epoch"]),
                "best_validation_loss": float(best),
                "parameters": parameters,
                "per_date": per_date,
                "validation_macro": macro,
                "history": history,
                "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256(checkpoint),
            }
            print(
                name, seed, "best", state["epoch"], "stop", history[-1]["epoch"],
                "BF1", round(macro["boundary_f1_30m"], 6),
                "ASD", round(macro["mean_symmetric_boundary_distance_m"], 3), flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    depth = json.loads(DEPTH_RESULT.read_text(encoding="utf-8"))
    control = json.loads(CONTROL_RESULT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    if depth["training_dates"] != [s["record"]["l8_date"] for s in train] or depth["validation_dates"] != [s["record"]["l8_date"] for s in validation]:
        raise RuntimeError("Existing deep parallel result used different dates")
    if not np.allclose(
        [[row[k] for k in ("p01", "p99", "mean", "std")] for row in normalization],
        [[row[k] for k in ("p01", "p99", "mean", "std")] for row in depth["normalization"]],
        rtol=0, atol=1e-12,
    ):
        raise RuntimeError("Normalization differs from existing deep parallel result")

    summary = {name: stats_from_runs(values) for name, values in runs.items()}
    existing = {
        "spatial_prior": prior["summary"]["prior"],
        "shallow_parallel": depth["summary"]["edge_depth_0"],
        "deep_parallel": depth["summary"]["edge_depth_2"],
        "parameter_matched_ordinary_conv": control["summary"],
    }
    ranking_values = {
        **{name: values for name, values in summary.items()},
        "deep_parallel": existing["deep_parallel"],
        "parameter_matched_ordinary_conv": existing["parameter_matched_ordinary_conv"],
    }
    ranking = sorted(
        ranking_values,
        key=lambda name: (
            -ranking_values[name]["boundary_f1_30m"]["mean"],
            ranking_values[name]["mean_symmetric_boundary_distance_m"]["mean"],
        ),
    )
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "VALIDATION_ONLY_DEEP_EDGE_COMPONENT_ABLATION_COMPLETE",
        "confirmation_sets_accessed": False,
        "selection_or_interpretation_data": "12 training dates and 3 development validation dates only",
        "single_operator_channel_control": (
            "Two fixed response maps duplicated with 1/sqrt(2) scaling to preserve the selected "
            "four-channel interface, response energy, trainable architecture, and parameter count."
        ),
        "seeds": list(SEEDS),
        "training_dates": [scene["record"]["l8_date"] for scene in train],
        "validation_dates": [scene["record"]["l8_date"] for scene in validation],
        "training_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "normalization": normalization,
        "target_trainable_parameters": target_parameters,
        "new_runs": runs,
        "new_summary": summary,
        "existing_locked_validation_results": existing,
        "component_ranking_rule": "highest validation Boundary F1 at 30 m; tie-break lower ASD",
        "component_ranking": ranking,
        "source_hashes": {
            "deep_parallel": sha256(DEPTH_RESULT),
            "parameter_matched_control": sha256(CONTROL_RESULT),
            "spatial_prior": sha256(PRIOR_RESULT),
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "deep_edge_component_ablation_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["configuration", "parameters", "BF1_30m_mean", "BF1_30m_sd", "ASD_m_mean", "ASD_m_sd", "IoU_mean"])
        display = {**existing, **summary}
        for name in ("spatial_prior", "parameter_matched_ordinary_conv", "deep_log_only", "deep_prewitt_only", "shallow_parallel", "deep_parallel"):
            values = display[name]
            writer.writerow([
                name, values.get("parameters", ""), values["boundary_f1_30m"]["mean"],
                values["boundary_f1_30m"].get("sd", values["boundary_f1_30m"].get("std")),
                values["mean_symmetric_boundary_distance_m"]["mean"],
                values["mean_symmetric_boundary_distance_m"].get("sd", values["mean_symmetric_boundary_distance_m"].get("std")),
                values["iou"]["mean"],
            ])
    print(json.dumps({"ranking": ranking, "new_summary": summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
