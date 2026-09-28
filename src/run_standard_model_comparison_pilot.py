"""One-seed standard-model comparison on frozen draft-label dataset v0.1."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_quick_model_pilot import metrics
from run_multidate_ablation_pilot import (
    AblationDFANet, MANIFEST, MultiSceneDataset, epoch, normalize_scenes,
    predict, read_scene, seed_all, windows_for,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preexperiment/results/standard_model_comparison_pilot"
SEED = 20260913
EPOCHS = 12


class StandardAdapter(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x, priors):
        return self.model(x)


def build_models():
    return {
        "proposed_parallel": lambda: AblationDFANet(True, "parallel"),
        "unet_resnet18": lambda: StandardAdapter(smp.Unet(
            encoder_name="resnet18", encoder_weights=None, in_channels=6, classes=1,
        )),
        "deeplabv3plus_resnet18": lambda: StandardAdapter(smp.DeepLabV3Plus(
            encoder_name="resnet18", encoder_weights=None, in_channels=6, classes=1,
        )),
        "segformer_mit_b0": lambda: StandardAdapter(smp.Segformer(
            encoder_name="mit_b0", encoder_weights=None, in_channels=6, classes=1,
        )),
    }


def main():
    seed_all(SEED); OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))["records"]
    scenes = [read_scene(r) for r in records if r["split"] in {"train", "validation"}]
    train = [s for s in scenes if s["record"]["split"] == "train"]
    val = [s for s in scenes if s["record"]["split"] == "validation"]
    normalize_scenes(scenes); tw, vw = windows_for(train), windows_for(val)
    loaders = {
        "train": DataLoader(MultiSceneDataset(train, tw, True), batch_size=12, shuffle=True, num_workers=0, pin_memory=True),
        "val": DataLoader(MultiSceneDataset(val, vw, False), batch_size=12, shuffle=False, num_workers=0, pin_memory=True),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, logs = {}, []
    for name, factory in build_models().items():
        seed_all(SEED); model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        best, patience, started = math.inf, 0, time.time()
        for ep in range(1, EPOCHS + 1):
            tr = epoch(model, loaders["train"], device, optimizer)
            va = epoch(model, loaders["val"], device)
            scheduler.step(va); logs.append({"model": name, "epoch": ep, "train_loss": tr, "val_loss": va})
            print(f"{name} epoch={ep:02d} train={tr:.4f} val={va:.4f}", flush=True)
            if va < best - 1e-4:
                best, patience = va, 0
                torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": va}, OUT / f"{name}_best.pt")
            else:
                patience += 1
            if ep >= 8 and patience >= 4:
                break
        checkpoint = torch.load(OUT / f"{name}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        per_date = {}
        for scene in val:
            prob = predict(model, scene, device)
            if name == "proposed_parallel":
                prob[scene["interior"]] = 1
                prob[scene["roi"] & ~scene["shore"] & ~scene["interior"]] = 0
                prob[~scene["roi"]] = 0
            per_date[scene["record"]["l8_date"]] = metrics(
                prob, scene["y"], scene["valid"], scene["valid"] & scene["shore"]
            )
        keys = ("precision", "recall", "f1", "iou", "boundary_f1_30m", "boundary_f1_60m", "mean_symmetric_boundary_distance_m")
        macro = {key: float(np.mean([x[key] for x in per_date.values()])) for key in keys}
        results[name] = {
            "best_epoch": int(checkpoint["epoch"]), "best_val_loss": float(checkpoint["val_loss"]),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "training_seconds": time.time() - started, "macro": macro, "per_date": per_date,
        }
        del model, optimizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    with (OUT / "training_log.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "epoch", "train_loss", "val_loss"])
        writer.writeheader(); writer.writerows(logs)
    config = {
        "purpose": "one-seed standard architecture comparison pilot",
        "not_final_paper_results": True, "dataset": "v0.1 automatic draft labels",
        "seed": SEED, "epochs_max": EPOCHS, "encoder_weights": None,
        "training_dates": [s["record"]["l8_date"] for s in train],
        "validation_dates": [s["record"]["l8_date"] for s in val],
        "smp_version": smp.__version__, "torch_version": torch.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_patches": len(tw), "validation_patches": len(vw),
    }
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {"proposed_parallel":"完整并联","unet_resnet18":"U-Net","deeplabv3plus_resnet18":"DeepLabV3+","segformer_mit_b0":"SegFormer"}
    lines = ["# 标准模型对比试跑", "", "模型均从头训练；使用6个训练日期与2个独立验证日期。标签为v0.1自动草稿，本表不能作为最终论文精度。", "", "|模型|IoU|F1|Boundary F1 30 m|Boundary F1 60 m|平均岸线距离/m|参数量|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in build_models():
        m=results[name]["macro"]
        lines.append(f"|{names[name]}|{m['iou']:.4f}|{m['f1']:.4f}|{m['boundary_f1_30m']:.4f}|{m['boundary_f1_60m']:.4f}|{m['mean_symmetric_boundary_distance_m']:.2f}|{results[name]['parameters']}|")
    lines += ["", "正式对比需在最终人工参考岸线、统一训练预算和3个随机种子下重新运行。"]
    (OUT / "标准模型对比试跑报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k:v["macro"] for k,v in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
