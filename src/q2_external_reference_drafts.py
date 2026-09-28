"""Prediction-blind S2 shoreline drafts for preliminarily usable external pairs.

These automatic drafts are annotation aids only. They are never final reference
labels and must not be used for test metrics without independent review.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import approximate_polygon, find_contours


ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "跨湖测试",
           ROOT / "drive-download-20260923T094631Z-1-001",
           ROOT / "drive-download-20260923T095357Z-1-001",
           ROOT / "drive-download-20260923T110142Z-1-001",
           ROOT / "drive-download-20260923T112403Z-1-001",
           ROOT / "drive-download-20260923T124126Z-1-001"]
OUT = ROOT / "preexperiment" / "results" / "q2_external" / "reference_drafts"
QC = ROOT / "preexperiment" / "results" / "q2_external" / "candidate_qc.csv"
SEEDS = {
    "Dianchi_Waihai": (102.7049, 24.8379),
    "Erhai": (100.1960, 25.8000),
    "Xingyun": (102.7820, 24.3370),
}


def locate(lake: str, l8: str, s2: str, kind: str) -> Path:
    name = f"Q2_FAST_{lake}_{l8}_{s2}_{kind}.tif"
    matches = [p for source in SOURCES for p in source.rglob(name)]
    if not matches and kind == "L8":
        matches = [p for source in SOURCES for p in source.rglob(
            f"Q2_FAST_{lake}_{l8}_*_L8.tif")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def main_component(mask: np.ndarray, ds=None, lake: str | None = None) -> np.ndarray:
    labels, n = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if not n:
        raise RuntimeError("No connected water component")
    if ds is not None and lake is not None:
        lon, lat = SEEDS[lake]
        x, y = transform("EPSG:4326", ds.crs, [lon], [lat])
        col, row = (~ds.transform) * (x[0], y[0])
        label = int(labels[round(row), round(col)])
        if label == 0:
            raise RuntimeError(f"{lake}: seed is not on 2019 JRC lake water")
        return labels == label
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def warp_binary(source: np.ndarray, source_ds, target_ds) -> np.ndarray:
    dst = np.zeros((target_ds.height, target_ds.width), dtype=np.uint8)
    reproject(source.astype(np.uint8), dst,
              src_transform=source_ds.transform, src_crs=source_ds.crs,
              dst_transform=target_ds.transform, dst_crs=target_ds.crs,
              resampling=Resampling.nearest)
    return dst > 0


def line_features(water: np.ndarray, valid: np.ndarray, ds,
                  lake: str, l8_date: str, s2_date: str) -> list[dict]:
    features = []
    edge_valid = ndi.binary_erosion(valid, iterations=2)
    for contour in find_contours(water.astype(np.uint8), 0.5):
        if len(contour) < 8:
            continue
        rr = np.clip(np.rint(contour[:, 0]).astype(int), 0, valid.shape[0] - 1)
        cc = np.clip(np.rint(contour[:, 1]).astype(int), 0, valid.shape[1] - 1)
        good = edge_valid[rr, cc]
        start = None
        for i, okay in enumerate(np.r_[good, False]):
            if okay and start is None:
                start = i
            if not okay and start is not None:
                piece = contour[start:i]
                start = None
                if len(piece) < 8:
                    continue
                piece = approximate_polygon(piece, tolerance=0.5)
                xy = np.array([ds.transform * (p[1] + 0.5, p[0] + 0.5)
                               for p in piece])
                length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
                if length < 60:
                    continue
                lon, lat = transform(ds.crs, "EPSG:4326",
                                     xy[:, 0].tolist(), xy[:, 1].tolist())
                features.append({
                    "type": "Feature",
                    "properties": {
                        "lake": lake, "l8_date": l8_date,
                        "s2_date": s2_date, "length_m": round(length, 2),
                        "status": "AUTOMATIC_DRAFT_UNREVIEWED",
                        "origin": "S2_NDWI_OTSU_WITH_FIXED_JRC_SEARCH_ZONE",
                    },
                    "geometry": {"type": "LineString",
                                 "coordinates": list(map(list, zip(lon, lat)))},
                })
    return features


def preview(s2_ds, water: np.ndarray, valid: np.ndarray, path: Path,
            title: str) -> None:
    scale = min(800 / s2_ds.width, 900 / s2_ds.height)
    width = max(1, round(s2_ds.width * scale))
    height = max(1, round(s2_ds.height * scale))
    bands = s2_ds.read([3, 2, 1], out_shape=(3, height, width),
                       resampling=Resampling.bilinear).astype(np.float32)
    rgb = np.clip((bands.transpose(1, 2, 0) - 100) / 3200, 0, 1) ** 0.8
    rgb = (np.nan_to_num(rgb) * 255).astype(np.uint8)
    rgb[~np.asarray(Image.fromarray(valid.astype(np.uint8)).resize(
        (width, height), Image.Resampling.NEAREST), dtype=bool)] = (180, 155, 175)
    edge = water & ~ndi.binary_erosion(water)
    edge = ndi.binary_dilation(edge, iterations=2)
    edge_small = np.asarray(Image.fromarray(edge.astype(np.uint8)).resize(
        (width, height), Image.Resampling.NEAREST), dtype=bool)
    rgb[edge_small] = (80, 220, 220)
    image = Image.fromarray(rgb)
    canvas = Image.new("RGB", (width, height + 48), "white")
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    draw.text((8, 12), title + " | CYAN=UNREVIEWED draft", font=font,
              fill=(25, 25, 40))
    canvas.save(path)


def build(row: dict) -> dict:
    lake, l8_day, s2_day = row["lake"], row["l8_date"], row["s2_date"]
    folder = OUT / lake / l8_day
    folder.mkdir(parents=True, exist_ok=True)
    qa_path = locate(lake, l8_day, s2_day, "QA_JRC")
    s2_path = locate(lake, l8_day, s2_day, "S2")
    with rasterio.open(qa_path) as qa, rasterio.open(s2_path) as s2:
        masks = qa.read()
        prior = main_component(masks[0] >= 2, qa, lake)
        interior_30 = ndi.binary_erosion(prior, iterations=10)
        shore_30 = ndi.binary_dilation(prior, iterations=10) & ~interior_30
        interior = warp_binary(interior_30, qa, s2)
        shore = warp_binary(shore_30, qa, s2)
        joint = warp_binary(masks[3] > 0, qa, s2)
        green = s2.read(2).astype(np.float32)
        nir = s2.read(4).astype(np.float32)
        s2_valid = s2.read(8) > 0
        valid = joint & s2_valid & ((green + nir) > 0)
        ndwi = np.full(green.shape, np.nan, dtype=np.float32)
        np.divide(green - nir, green + nir, out=ndwi,
                  where=valid)
        sample = ndwi[shore & valid & np.isfinite(ndwi)]
        if sample.size < 1000:
            raise RuntimeError(f"Too few valid shoreline samples: {lake} {l8_day}")
        threshold = float(threshold_otsu(sample))
        candidate = interior | (shore & valid & (ndwi > threshold))
        # The target is the component linked to this lake's fixed center seed,
        # never the largest component in a box that may contain another lake.
        labels, n = ndi.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
        lon, lat = SEEDS[lake]
        x, y = transform("EPSG:4326", s2.crs, [lon], [lat])
        col, row = (~s2.transform) * (x[0], y[0])
        label = int(labels[round(row), round(col)])
        if label == 0:
            raise RuntimeError(f"{lake} {l8_day}: seed outside candidate water")
        water = ndi.binary_fill_holes(labels == label)
        features = line_features(water, valid, s2, lake, l8_day, s2_day)
        p = s2.profile.copy()
        p.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        draft = np.where(valid, water.astype(np.uint8), 255).astype(np.uint8)
        with rasterio.open(folder / "S2_water_DRAFT_10m.tif", "w", **p) as dst:
            dst.write(draft, 1)
            dst.descriptions = ("AUTOMATIC_DRAFT_NOT_REFERENCE_TRUTH",)
        preview(s2, water, valid, folder / "review_overview.png",
                f"{lake} {l8_day} / S2 {s2_day}")
    geojson = {"type": "FeatureCollection", "features": features}
    (folder / "shoreline_DRAFT.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False), encoding="utf-8"
    )
    result = {
        "lake": lake, "l8_date": l8_day, "s2_date": s2_day,
        "status": "AUTOMATIC_DRAFT_UNREVIEWED",
        "threshold_ndwi": threshold,
        "threshold_sample_pixels": int(sample.size),
        "draft_segment_count": len(features),
        "draft_length_km": round(sum(f["properties"]["length_m"]
                                     for f in features) / 1000, 2),
        "allow_training": False,
        "allow_model_selection": False,
        "allow_test_evaluation": False,
    }
    (folder / "draft_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with QC.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r["preliminary_quality_pass"] == "True"]
    results = []
    for row in rows:
        result = build(row)
        results.append(result)
        print(result["lake"], result["l8_date"],
              result["draft_segment_count"], result["draft_length_km"],
              flush=True)
    (OUT / "draft_summary.json").write_text(json.dumps({
        "status": "PREDICTION_BLIND_AUTOMATIC_DRAFTS_REQUIRE_HUMAN_REVIEW",
        "model_predictions_opened": False,
        "record_count": len(results),
        "records": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
