"""Run a transparent one-date model feasibility pilot on Fuxian Lake.

This is a technical preexperiment, not the final multi-date paper experiment.
The reference is a same-date Sentinel-2 NDWI proxy, and spatial east/west
sectors are held out to prevent patch-level leakage within the scene.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_erosion, distance_transform_edt, label as component_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "preexperiment"
ANN = PRE / "results/native_20200102/annotation_workspace"
OUT = PRE / "results/quick_model_pilot_20200102"
L8 = ROOT / "Fuxian_L8_native_20200102.tif"
MASKS = ROOT / "Fuxian_masks_native_20200102.tif"
S2_NDWI = ANN / "S2_NDWI_diagnostic.tif"
EXCLUSION = ANN / "ge_corrections/training_exclusion_mask_L8_30m.tif"
S2_THRESHOLD = 0.13653606176376343
SEED = 20260913
PATCH = 64
STRIDE = 32
EPOCHS = 24


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_reference(dst_profile: dict) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(S2_NDWI) as src:
        ndwi = src.read(1)
        proxy = np.full(ndwi.shape, 255, dtype=np.uint8)
        finite = np.isfinite(ndwi)
        proxy[finite] = (ndwi[finite] > S2_THRESHOLD).astype(np.uint8)
        fraction = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=proxy,
            destination=fraction,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=255,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return fraction >= 0.5, np.isfinite(fraction)


def read_data():
    with rasterio.open(L8) as src:
        x = src.read(range(1, 7)).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
    with rasterio.open(MASKS) as src:
        masks = src.read()
    with rasterio.open(EXCLUSION) as src:
        exclusion = src.read(1)

    y, s2_valid = make_reference(profile)
    finite = np.all(np.isfinite(x) & (x > -9000), axis=0)
    shore = masks[0].astype(bool)
    joint = masks[3].astype(bool)
    roi = masks[4].astype(bool)
    valid = finite & joint & s2_valid & (exclusion == 0)
    return x, y, valid, shore, roi, profile, transform


def sector_masks(y: np.ndarray, roi: np.ndarray, transform):
    rows, cols = np.indices(y.shape)
    lake = y & roi
    cy, cx = np.mean(np.argwhere(lake), axis=0)
    # Image row increases southward; invert row for map-like angles.
    angle = np.degrees(np.arctan2(-(rows - cy), cols - cx))
    test = np.abs(angle) <= 35
    val = np.abs(angle) >= 145
    train = ~(test | val)
    center_x, center_y = transform * (float(cx), float(cy))
    return {"train": train, "val": val, "test": test}, {
        "row": float(cy), "col": float(cx), "x": float(center_x), "y": float(center_y)
    }


def build_spatial_prior(roi: np.ndarray, shore: np.ndarray) -> np.ndarray:
    """Identify the lake-side fixed interior without using target-date labels.

    Removing the uncertain shoreline band separates the ROI into a large lake
    interior and smaller exterior-land pieces. The largest component is the
    fixed-water interior supplied by the historical spatial prior.
    """
    labels, count = component_label(roi & ~shore)
    if count == 0:
        raise RuntimeError("Spatial prior has no interior component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def normalize(x: np.ndarray, train_pixels: np.ndarray):
    out = np.empty_like(x, dtype=np.float32)
    stats = []
    for i in range(x.shape[0]):
        vals = x[i][train_pixels]
        lo, hi = np.percentile(vals, [1, 99])
        clipped = np.clip(x[i], lo, hi)
        mean = float(np.mean(clipped[train_pixels]))
        std = float(np.std(clipped[train_pixels]))
        std = max(std, 1e-6)
        out[i] = (clipped - mean) / std
        stats.append({"band": i + 1, "p01": float(lo), "p99": float(hi), "mean": mean, "std": std})
    out[:, ~np.all(np.isfinite(out), axis=0)] = 0
    return out, stats


def build_patch_lists(y, valid, shore, splits):
    lists = {name: [] for name in splits}
    h, w = y.shape
    for row in range(0, h - PATCH + 1, STRIDE):
        for col in range(0, w - PATCH + 1, STRIDE):
            sl = np.s_[row : row + PATCH, col : col + PATCH]
            for name, sector in splits.items():
                m = valid[sl] & shore[sl] & sector[sl]
                n = int(m.sum())
                if n < 180:
                    continue
                positive = int((y[sl] & m).sum())
                frac = positive / n
                if 0.04 <= frac <= 0.96:
                    lists[name].append((row, col))
    return lists


class PatchDataset(Dataset):
    def __init__(self, x, y, valid, shore, roi, sector, windows, augment=False):
        self.x, self.y = x, y
        self.valid, self.shore, self.roi, self.sector = valid, shore, roi, sector
        self.windows = windows
        self.augment = augment

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        row, col = self.windows[idx]
        sl = np.s_[row : row + PATCH, col : col + PATCH]
        x = self.x[:, sl[0], sl[1]].copy()
        y = self.y[sl].astype(np.float32)[None]
        loss_mask = (self.valid[sl] & self.shore[sl] & self.sector[sl]).astype(np.float32)[None]
        priors = np.stack([self.roi[sl], self.shore[sl]]).astype(np.float32)
        if self.augment:
            k = random.randrange(4)
            x, y, loss_mask, priors = [np.rot90(a, k, axes=(-2, -1)).copy() for a in (x, y, loss_mask, priors)]
            if random.random() < 0.5:
                x, y, loss_mask, priors = [a[..., ::-1].copy() for a in (x, y, loss_mask, priors)]
        return tuple(torch.from_numpy(a) for a in (x, priors, y, loss_mask))


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class DSConv(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    def __init__(self, cin, widths):
        super().__init__()
        self.stem = ConvBNAct(cin, widths[0], 3, 2)
        self.s1 = DSConv(widths[0], widths[1], 2)
        self.s2 = DSConv(widths[1], widths[2], 2)
        self.s3 = DSConv(widths[2], widths[3], 2)

    def forward(self, x):
        a = self.stem(x)
        b = self.s1(a)
        c = self.s2(b)
        d = self.s3(c)
        return a, b, c, d


class FixedEdges(nn.Module):
    def __init__(self):
        super().__init__()
        px = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32) / 3
        py = px.t()
        log = torch.tensor([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]], dtype=torch.float32) / 16
        self.register_buffer("px", px[None, None])
        self.register_buffer("py", py[None, None])
        self.register_buffer("log", log[None, None])

    def forward(self, x):
        outputs = []
        for channel in (x[:, 1:2], x[:, 3:4]):
            gx = F.conv2d(channel, self.px, padding=1)
            gy = F.conv2d(channel, self.py, padding=1)
            outputs.append(torch.sqrt(gx.square() + gy.square() + 1e-6))
            outputs.append(torch.abs(F.conv2d(channel, self.log, padding=2)))
        return torch.cat(outputs, dim=1)


class PilotDFANet(nn.Module):
    """Compact depthwise feature-aggregation network for feasibility testing."""
    def __init__(self, enhanced: bool):
        super().__init__()
        self.enhanced = enhanced
        self.semantic = Encoder(8 if enhanced else 6, (24, 48, 96, 128))
        if enhanced:
            self.edges = FixedEdges()
            self.edge_encoder = Encoder(4, (12, 20, 32, 48))
            edge_widths = (12, 20, 32, 48)
        else:
            edge_widths = (0, 0, 0, 0)
        self.d2 = DSConv(128 + edge_widths[3] + 96 + edge_widths[2], 96)
        self.d1 = DSConv(96 + 48 + edge_widths[1], 64)
        self.d0 = DSConv(64 + 24 + edge_widths[0], 48)
        self.head = nn.Conv2d(48, 1, 1)

    def forward(self, x, priors):
        semantic_input = torch.cat([x, priors], dim=1) if self.enhanced else x
        s0, s1, s2, s3 = self.semantic(semantic_input)
        if self.enhanced:
            e0, e1, e2, e3 = self.edge_encoder(self.edges(x))
            z = torch.cat([s3, e3], dim=1)
            z = F.interpolate(z, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            z = self.d2(torch.cat([z, s2, e2], dim=1))
            z = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            z = self.d1(torch.cat([z, s1, e1], dim=1))
            z = F.interpolate(z, size=s0.shape[-2:], mode="bilinear", align_corners=False)
            z = self.d0(torch.cat([z, s0, e0], dim=1))
        else:
            z = F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            z = self.d2(torch.cat([z, s2], dim=1))
            z = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            z = self.d1(torch.cat([z, s1], dim=1))
            z = F.interpolate(z, size=s0.shape[-2:], mode="bilinear", align_corners=False)
            z = self.d0(torch.cat([z, s0], dim=1))
        return F.interpolate(self.head(z), size=x.shape[-2:], mode="bilinear", align_corners=False)


def masked_loss(logits, y, mask):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp_min(1)
    prob = torch.sigmoid(logits)
    inter = (prob * y * mask).sum()
    dice = 1 - (2 * inter + 1) / ((prob * mask).sum() + (y * mask).sum() + 1)
    return bce + dice


def epoch_pass(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses = []
    for x, priors, y, mask in loader:
        x, priors, y, mask = [a.to(device, non_blocking=True) for a in (x, priors, y, mask)]
        with torch.set_grad_enabled(training):
            loss = masked_loss(model(x, priors), y, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def sliding_predict(model, x, priors, device, tile=256, overlap=32):
    model.eval()
    h, w = x.shape[-2:]
    total = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    step = tile - overlap
    rows = list(range(0, max(1, h - tile + 1), step))
    cols = list(range(0, max(1, w - tile + 1), step))
    if rows[-1] != h - tile:
        rows.append(max(0, h - tile))
    if cols[-1] != w - tile:
        cols.append(max(0, w - tile))
    for row in rows:
        for col in cols:
            xx = torch.from_numpy(x[:, row : row + tile, col : col + tile][None]).to(device)
            pp = torch.from_numpy(priors[:, row : row + tile, col : col + tile][None]).to(device)
            prob = torch.sigmoid(model(xx, pp))[0, 0].cpu().numpy()
            total[row : row + tile, col : col + tile] += prob
            count[row : row + tile, col : col + tile] += 1
    return total / np.maximum(count, 1)


def boundary_metrics(pred, truth, valid, tolerance):
    pred_b = pred ^ binary_erosion(pred)
    truth_b = truth ^ binary_erosion(truth)
    core = binary_erosion(valid, iterations=tolerance + 1)
    pred_b &= core
    truth_b &= core
    if not pred_b.any() or not truth_b.any():
        return {"boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0, "assd_pixels": None}
    dt_truth = distance_transform_edt(~truth_b)
    dt_pred = distance_transform_edt(~pred_b)
    precision = float(np.mean(dt_truth[pred_b] <= tolerance))
    recall = float(np.mean(dt_pred[truth_b] <= tolerance))
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    assd = (float(np.mean(dt_truth[pred_b])) + float(np.mean(dt_pred[truth_b]))) / 2
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1, "assd_pixels": assd}


def metrics(prob, truth, region_valid, boundary_valid):
    pred = prob >= 0.5
    p, t = pred[region_valid], truth[region_valid]
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t)); fn = int(np.sum(~p & t)); tn = int(np.sum(~p & ~t))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    result = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "iou": iou}
    for tol in (1, 2):
        bm = boundary_metrics(pred, truth, boundary_valid, tol)
        result[f"boundary_f1_{tol * 30}m"] = bm["boundary_f1"]
        result[f"boundary_precision_{tol * 30}m"] = bm["boundary_precision"]
        result[f"boundary_recall_{tol * 30}m"] = bm["boundary_recall"]
        if tol == 2:
            result["mean_symmetric_boundary_distance_m"] = None if bm["assd_pixels"] is None else bm["assd_pixels"] * 30
    return result


def save_tif(path, array, profile, valid):
    out = np.full(array.shape, 255, dtype=np.uint8)
    out[valid] = array[valid].astype(np.uint8)
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)


def make_qc(path, x_raw, truth, predictions, test_mask):
    valid = np.all(np.isfinite(x_raw) & (x_raw > -9000), axis=0)
    channels = []
    for idx in (2, 1, 0):
        a = x_raw[idx]
        lo, hi = np.percentile(a[valid], [2, 98])
        channels.append((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8))
    rgb = np.dstack(channels)
    panels = []
    for name, pred in [("reference", truth), *predictions.items()]:
        panel = rgb.copy()
        boundary = pred ^ binary_erosion(pred)
        panel[boundary & test_mask] = (255, 40, 40) if name == "reference" else ((255, 220, 0) if name == "base" else (0, 255, 255))
        panel[~test_mask] = (panel[~test_mask] * 0.35).astype(np.uint8)
        panels.append(Image.fromarray(panel))
    canvas = Image.new("RGB", (rgb.shape[1] * len(panels), rgb.shape[0]), "black")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * rgb.shape[1], 0))
    canvas.resize((canvas.width, canvas.height), Image.Resampling.NEAREST).save(path)


def main():
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_raw, y, valid, shore, roi, profile, transform = read_data()
    prior_interior = build_spatial_prior(roi, shore)
    splits, lake_center = sector_masks(y, roi, transform)
    x, norm_stats = normalize(x_raw, valid & shore & splits["train"])
    windows = build_patch_lists(y, valid, shore, splits)
    if min(map(len, windows.values())) == 0:
        raise RuntimeError(f"A spatial split has no eligible patches: { {k: len(v) for k,v in windows.items()} }")
    priors_full = np.stack([prior_interior, shore]).astype(np.float32)

    config = {
        "purpose": "one-date technical feasibility pilot",
        "not_for_paper_claims": True,
        "date": "2020-01-02",
        "seed": SEED,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "reference": "same-date S2 NDWI proxy aggregated to L8 grid",
        "reference_threshold": S2_THRESHOLD,
        "patch_size": PATCH,
        "stride": STRIDE,
        "epochs_max": EPOCHS,
        "split": "north/south train, west validation, east test; pixel masks are disjoint",
        "patch_counts": {k: len(v) for k, v in windows.items()},
        "lake_center": lake_center,
        "spatial_prior": {
            "interior_pixels": int(prior_interior.sum()),
            "uncertain_shore_pixels": int(shore.sum()),
            "rule": "largest connected component of ROI after removing shoreline band",
        },
        "normalization": norm_stats,
    }
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    loaders = {}
    for name in ("train", "val"):
        ds = PatchDataset(x, y, valid, shore, prior_interior, splits[name], windows[name], augment=name == "train")
        loaders[name] = DataLoader(ds, batch_size=16, shuffle=name == "train", num_workers=0, pin_memory=True)

    all_logs = []
    results = {}
    predictions = {}
    for name, enhanced in (("base", False), ("enhanced", True)):
        seed_all(SEED)
        model = PilotDFANet(enhanced).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        best = math.inf
        patience = 0
        start = time.time()
        for epoch in range(1, EPOCHS + 1):
            train_loss = epoch_pass(model, loaders["train"], device, optimizer)
            val_loss = epoch_pass(model, loaders["val"], device)
            scheduler.step(val_loss)
            all_logs.append({"model": name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
            if val_loss < best - 1e-4:
                best = val_loss
                patience = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": config}, OUT / f"{name}_best.pt")
            else:
                patience += 1
            print(f"{name} epoch={epoch:02d} train={train_loss:.4f} val={val_loss:.4f}", flush=True)
            if epoch >= 8 and patience >= 5:
                break
        checkpoint = torch.load(OUT / f"{name}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        prob = sliding_predict(model, x, priors_full, device)
        if enhanced:
            prob[prior_interior] = 1.0
            prob[roi & ~shore & ~prior_interior] = 0.0
            prob[~roi] = 0.0
        test_region = valid & roi & splits["test"]
        test_boundary = valid & shore & splits["test"]
        result = metrics(prob, y, test_region, test_boundary)
        result.update({
            "best_epoch": int(checkpoint["epoch"]),
            "best_val_loss": float(checkpoint["val_loss"]),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "training_seconds": time.time() - start,
        })
        results[name] = result
        predictions[name] = prob >= 0.5
        save_tif(OUT / f"{name}_prediction.tif", predictions[name], profile, valid & roi)

    with (OUT / "training_log.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader(); writer.writerows(all_logs)
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_tif(OUT / "s2_proxy_reference.tif", y, profile, valid & roi)
    make_qc(OUT / "east_test_comparison.png", x_raw, y, predictions, valid & splits["test"])

    delta = {key: results["enhanced"].get(key, 0) - results["base"].get(key, 0) for key in ("iou", "f1", "boundary_f1_30m", "boundary_f1_60m")}
    report = f"""# 快速模型预实验：2020-01-02

## 定位

本实验用于检查模型、标签、训练和边界评价链路能否跑通。参考标签来自同日 Sentinel-2 NDWI，并聚合到 Landsat 8 原生 30 m 网格。由于只有一个日期且参考岸线尚未完成全湖独立人工复核，结果不能作为论文最终精度。

## 空间划分

- 训练：北岸和南岸；验证：西岸；测试：东岸。
- 图块可重叠，但损失和评价像元按空间扇区严格隔离。
- 训练/验证/测试合格图块数：{len(windows['train'])}/{len(windows['val'])}/{len(windows['test'])}。
- 五处 Google Earth 辅助审查区域已从训练和评价中排除。

## 结果

| 模型 | IoU | F1 | Boundary F1 30 m | Boundary F1 60 m | 平均对称边界距离/m | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
|紧凑 DFANet 技术基线|{results['base']['iou']:.4f}|{results['base']['f1']:.4f}|{results['base']['boundary_f1_30m']:.4f}|{results['base']['boundary_f1_60m']:.4f}|{results['base']['mean_symmetric_boundary_distance_m']:.2f}|{results['base']['parameter_count']}|
|空间先验 + LoG/Prewitt 边缘分支|{results['enhanced']['iou']:.4f}|{results['enhanced']['f1']:.4f}|{results['enhanced']['boundary_f1_30m']:.4f}|{results['enhanced']['boundary_f1_60m']:.4f}|{results['enhanced']['mean_symmetric_boundary_distance_m']:.2f}|{results['enhanced']['parameter_count']}|

增强模型相对基线的变化：IoU {delta['iou']:+.4f}，F1 {delta['f1']:+.4f}，Boundary F1 30 m {delta['boundary_f1_30m']:+.4f}，Boundary F1 60 m {delta['boundary_f1_60m']:+.4f}。

## 结论限制

该结果只证明技术链路是否可运行，并用于决定是否进入多日期正式实验。网络是为快速验证编写的紧凑 DFANet 风格实现，正式论文实验必须核对并固定标准 DFANet 结构，在独立日期上重新训练和评价，并报告多个随机种子。
"""
    (OUT / "快速模型预实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps({"results": results, "delta": delta, "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
