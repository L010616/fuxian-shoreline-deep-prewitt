"""Three-seed validation confirmation and final model freeze for round 4."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_multidate_ablation_pilot as ablation
import run_round4_fusion_screen as screen
from fusion_strategies_round4 import WeightedParallelDFANet, segmentation_edge_loss
from run_multidate_ablation_pilot import AblationDFANet, MultiSceneDataset, seed_all, windows_for


ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "preexperiment"
SCREEN = PRE / "results/round4_fusion_screen/fusion_screen_results.json"
OUT = PRE / "results/round4_fusion_confirmation"
SEEDS = (20260913, 20260929, 20261017)
MAX_EPOCHS = 40
MIN_EPOCHS = 20
PATIENCE = 8
CONFIGS = {
    "prior": (lambda: AblationDFANet(True, None), 0.0),
    "log": (lambda: AblationDFANet(True, "log"), 0.0),
    "parallel_concat_ew0p0": (lambda: AblationDFANet(True, "parallel"), 0.0),
    "parallel_concat_ew0p1": (lambda: AblationDFANet(True, "parallel"), 0.1),
    "parallel_weighted_ew0p0": (WeightedParallelDFANet, 0.0),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "fusion_confirmation_results.json"
    if result_path.exists():
        raise RuntimeError("Fusion confirmation already completed; refusing to rerun")
    screened = json.loads(SCREEN.read_text(encoding="utf-8"))
    expected = set(screened["top_two_for_three_seed_confirmation"])
    if not expected.issubset(CONFIGS):
        raise RuntimeError("Confirmation configurations do not contain the screened top two")

    scenes = screen.load_scenes()
    train = [s for s in scenes if s["record"]["split"] == "train"]
    validation = [s for s in scenes if s["record"]["split"] == "validation"]
    normalization = ablation.normalize_scenes(scenes)
    train_windows = windows_for(train); validation_windows = windows_for(validation)
    validation_loader = DataLoader(MultiSceneDataset(validation, validation_windows), batch_size=12,
                                   shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for config, (factory, edge_weight) in CONFIGS.items():
        results[config] = {}
        for seed in SEEDS:
            seed_all(seed)
            train_loader = DataLoader(MultiSceneDataset(train, train_windows, augment=True), batch_size=12,
                                      shuffle=True, num_workers=0, pin_memory=True)
            model = factory().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best = float("inf"); best_epoch = 0; stale = 0; history = []
            checkpoint_path = OUT / f"{config}_seed{seed}_best.pt"
            for epoch in range(1, MAX_EPOCHS + 1):
                train_loss = screen.train_epoch(model, train_loader, device, optimizer, edge_weight)
                val_loss = screen.validation_loss(model, validation_loader, device, edge_weight)
                history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
                if val_loss < best - 1e-4:
                    best = val_loss; best_epoch = epoch; stale = 0
                    torch.save({"model": model.state_dict(), "epoch": epoch,
                                "validation_loss": val_loss, "config": config,
                                "edge_weight": edge_weight}, checkpoint_path)
                else:
                    stale += 1
                if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                    break
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            per_date, macro = screen.validate_full_scenes(model, validation, device)
            item = {"best_epoch": best_epoch, "validation_loss": best,
                    "validation_macro": macro, "per_date": per_date, "history": history}
            if config == "parallel_weighted_ew0p0":
                weights = (2 * torch.softmax(model.edges.operator_logits.detach().cpu(), dim=0)).tolist()
                item["learned_operator_weights"] = {"prewitt": weights[0], "log": weights[1]}
            results[config][str(seed)] = item
            print(config, seed, best_epoch, round(macro["boundary_f1_30m"], 5),
                  round(macro["mean_symmetric_boundary_distance_m"], 2), flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    summary = {}
    for config in CONFIGS:
        summary[config] = {}
        for key in ("iou", "boundary_f1_30m", "boundary_f1_60m",
                    "mean_symmetric_boundary_distance_m", "hd95_m"):
            values = np.asarray([results[config][str(seed)]["validation_macro"][key] for seed in SEEDS])
            summary[config][key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                                    "values": values.tolist()}
    ranking = sorted(CONFIGS, key=lambda name: (-summary[name]["boundary_f1_30m"]["mean"],
                                                summary[name]["mean_symmetric_boundary_distance_m"]["mean"]))
    winner = ranking[0]
    payload = {
        "status": "THREE_SEED_VALIDATION_CONFIRMATION_COMPLETE",
        "test_data_accessed": False,
        "seeds": SEEDS, "training_dates": [s["record"]["l8_date"] for s in train],
        "validation_dates": [s["record"]["l8_date"] for s in validation],
        "training_windows": len(train_windows), "validation_windows": len(validation_windows),
        "normalization": normalization,
        "ranking_rule": "highest three-seed mean validation Boundary F1 at 30 m; tie-break lower ASD",
        "ranking": ranking, "selected_final_config": winner,
        "results": results, "summary": summary,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "fusion_confirmation_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["rank", "config", "bf1_30m_mean", "bf1_30m_sd", "asd_m_mean", "asd_m_sd",
                  "bf1_60m_mean", "iou_mean", "hd95_m_mean"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for rank, config in enumerate(ranking, 1):
            s = summary[config]
            writer.writerow({"rank": rank, "config": config,
                             "bf1_30m_mean": s["boundary_f1_30m"]["mean"],
                             "bf1_30m_sd": s["boundary_f1_30m"]["std"],
                             "asd_m_mean": s["mean_symmetric_boundary_distance_m"]["mean"],
                             "asd_m_sd": s["mean_symmetric_boundary_distance_m"]["std"],
                             "bf1_60m_mean": s["boundary_f1_60m"]["mean"],
                             "iou_mean": s["iou"]["mean"], "hd95_m_mean": s["hd95_m"]["mean"]})
    frozen = {
        "status": "FINAL_MODEL_CONFIG_FROZEN_BEFORE_BLIND_TEST",
        "selected_config": winner,
        "selection_source": str(result_path.relative_to(ROOT)),
        "selection_rule": payload["ranking_rule"],
        "normalization": normalization,
        "decision_threshold": 0.5,
        "seeds": list(SEEDS),
        "test_data_accessed": False,
        "allow_architecture_change_after_blind_test": False,
    }
    (OUT / "final_model_spec.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": winner, "ranking": ranking,
                      "summary": {k: {"bf1": round(v["boundary_f1_30m"]["mean"], 5),
                                             "asd": round(v["mean_symmetric_boundary_distance_m"]["mean"], 2)}
                                  for k, v in summary.items()}}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
