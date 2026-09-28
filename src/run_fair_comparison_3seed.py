"""Three-seed fair comparison with raw and common-constraint evaluation."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from run_quick_model_pilot import metrics
from run_multidate_ablation_pilot import (
    MANIFEST, MultiSceneDataset, epoch, normalize_scenes, predict, read_scene,
    seed_all, windows_for,
)
from run_standard_model_comparison_pilot import build_models


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preexperiment/results/fair_comparison_3seed_corrected_labels_v03"
SEEDS = [20260913, 20260929, 20261017]
EPOCHS = 15


def evaluate(model, val_scenes, device):
    raw, constrained = {}, {}
    for scene in val_scenes:
        probability = predict(model, scene, device)
        date = scene["record"]["l8_date"]
        raw[date] = metrics(
            probability, scene["y"], scene["valid"], scene["valid"] & scene["shore"]
        )
        common = probability.copy()
        common[scene["interior"]] = 1.0
        common[scene["roi"] & ~scene["shore"] & ~scene["interior"]] = 0.0
        common[~scene["roi"]] = 0.0
        constrained[date] = metrics(
            common, scene["y"], scene["valid"], scene["valid"] & scene["shore"]
        )
    keys = ("precision", "recall", "f1", "iou", "boundary_f1_30m",
            "boundary_f1_60m", "mean_symmetric_boundary_distance_m")
    macro = lambda rows: {k: float(np.mean([v[k] for v in rows.values()])) for k in keys}
    return {
        "raw": {"macro": macro(raw), "per_date": raw},
        "common_constraint": {"macro": macro(constrained), "per_date": constrained},
    }


def summarize(all_results):
    summary = {}
    for model in build_models():
        summary[model] = {}
        for mode in ("raw", "common_constraint"):
            rows = [all_results[str(seed)][model][mode]["macro"] for seed in SEEDS]
            summary[model][mode] = {}
            for key in rows[0]:
                values = np.asarray([r[key] for r in rows], dtype=float)
                summary[model][mode][key] = {
                    "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                    "values": values.tolist(),
                }
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))["records"]
    scenes = [read_scene(r) for r in records if r["split"] in {"train", "validation"}]
    train = [s for s in scenes if s["record"]["split"] == "train"]
    val = [s for s in scenes if s["record"]["split"] == "validation"]
    norm = normalize_scenes(scenes); tw, vw = windows_for(train), windows_for(val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results, logs = {}, []
    for seed in SEEDS:
        all_results[str(seed)] = {}
        for name, factory in build_models().items():
            seed_all(seed)
            train_loader = DataLoader(MultiSceneDataset(train, tw, True), batch_size=12,
                                      shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(MultiSceneDataset(val, vw, False), batch_size=12,
                                    shuffle=False, num_workers=0, pin_memory=True)
            model = factory().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            best, patience, started = math.inf, 0, time.time()
            checkpoint_path = OUT / f"{name}_seed{seed}_best.pt"
            for ep in range(1, EPOCHS + 1):
                tr = epoch(model, train_loader, device, optimizer)
                va = epoch(model, val_loader, device)
                scheduler.step(va)
                logs.append({"seed": seed, "model": name, "epoch": ep,
                             "train_loss": tr, "val_loss": va})
                print(f"seed={seed} {name} epoch={ep:02d} train={tr:.4f} val={va:.4f}", flush=True)
                if va < best - 1e-4:
                    best, patience = va, 0
                    torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": va}, checkpoint_path)
                else:
                    patience += 1
                if ep >= 9 and patience >= 4:
                    break
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            result = evaluate(model, val, device)
            result.update({
                "best_epoch": int(checkpoint["epoch"]),
                "best_val_loss": float(checkpoint["val_loss"]),
                "parameters": int(sum(p.numel() for p in model.parameters())),
                "training_seconds": time.time() - started,
            })
            all_results[str(seed)][name] = result
            (OUT / "results_by_seed.json").write_text(
                json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            del model, optimizer, train_loader, val_loader
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    summary = summarize(all_results)
    (OUT / "summary_mean_std.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "training_log.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "model", "epoch", "train_loss", "val_loss"])
        writer.writeheader(); writer.writerows(logs)
    config = {
        "purpose": "three-seed fair comparison on corrected frozen dataset v0.2",
        "not_final_paper_results": True,
        "seeds": SEEDS, "epochs_max": EPOCHS,
        "training_dates": [s["record"]["l8_date"] for s in train],
        "validation_dates": [s["record"]["l8_date"] for s in val],
        "train_patches": len(tw), "validation_patches": len(vw),
        "encoder_weights": None, "smp_version": smp.__version__,
        "torch_version": torch.__version__, "device": str(device),
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "normalization": norm,
        "evaluation_modes": {
            "raw": "direct model probability thresholded at 0.5",
            "common_constraint": "same fixed interior/exterior hard constraint applied to every model",
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = {"proposed_parallel":"完整并联", "unet_resnet18":"U-Net",
              "deeplabv3plus_resnet18":"DeepLabV3+", "segformer_mit_b0":"SegFormer",
              "dfanet":"DFANet", "prior":"边界约束", "log":"边界约束+LoG",
              "prewitt":"边界约束+Prewitt", "parallel":"完整并联",
              "gated_parallel":"门控并联"}
    lines = ["# 三随机种子公平对比（v0.2修正草稿标签）", "",
             "以下为两个独立验证日期的宏平均，再对三个随机种子计算均值±样本标准差。", ""]
    for mode, title in (("raw", "原始输出"), ("common_constraint", "统一空间约束")):
        lines += [f"## {title}", "", "|模型|IoU|Boundary F1 30 m|平均岸线距离/m|", "|---|---:|---:|---:|"]
        for name in build_models():
            m=summary[name][mode]
            lines.append(f"|{labels[name]}|{m['iou']['mean']:.4f}±{m['iou']['std']:.4f}|{m['boundary_f1_30m']['mean']:.4f}±{m['boundary_f1_30m']['std']:.4f}|{m['mean_symmetric_boundary_distance_m']['mean']:.2f}±{m['mean_symmetric_boundary_distance_m']['std']:.2f}|")
        lines.append("")
    lines += ["当前标签为自动草稿，结果用于判断结构和空间约束贡献，不能替代最终人工参考岸线实验。"]
    (OUT / "三随机种子公平对比报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
