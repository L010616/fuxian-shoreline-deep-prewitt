"""Sensitivity of R2 results to plausible manual shoreline placement error.

The exact-date Sentinel-2 grid is 10 m.  We therefore evaluate the frozen
models against inward/outward perturbations of the manually drawn polygon by
one and two 10-m pixels.  This is a sensitivity envelope, not a claim that the
human error follows a specific probability distribution.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scipy import ndimage as ndi

import evaluate_q2_manual_core6 as locked
import evaluate_q2_manual_core6_r2 as r2


ROOT = locked.ROOT
BASE = ROOT / "preexperiment/results/q2_manual_core6_evaluation_R2_from_scratch_v2"
PARENT = ROOT / "preexperiment/results/q2_manual_core6_R2_manual_uncertainty"
OFFSETS = (-20, -10, 0, 10, 20)
METRICS = ("precision", "recall", "f1", "iou", "boundary_f1_30m",
           "boundary_f1_60m", "mean_symmetric_boundary_distance_m", "hd95_m")


def main() -> None:
    PARENT.mkdir(parents=True, exist_ok=True)
    original = locked.reviewed_mask
    for offset in OFFSETS:
        if offset == 0:
            continue
        steps = abs(offset) // 10

        def perturbed(reviewed: Path, template: Path, *, _steps=steps, _offset=offset):
            mask = original(reviewed, template)
            if _offset < 0:
                return ndi.binary_erosion(mask, iterations=_steps)
            return ndi.binary_dilation(mask, iterations=_steps)

        locked.review_rows = r2.r2_rows
        locked.reviewed_mask = perturbed
        locked.OUT = PARENT / f"offset_{offset:+d}m"
        if not locked.OUT.exists():
            print(f"RUN offset {offset:+d} m", flush=True)
            locked.main()

    sources = {0: BASE / "manual_core6_results.json"}
    sources.update({offset: PARENT / f"offset_{offset:+d}m" / "manual_core6_results.json"
                    for offset in OFFSETS if offset})
    payloads = {offset: json.loads(path.read_text(encoding="utf-8"))
                for offset, path in sources.items()}

    summary_rows = []
    for model in locked.base.MODELS:
        for metric in METRICS:
            values = {offset: data["summary"][model]["all_18_dates"][metric]["mean"]
                      for offset, data in payloads.items()}
            summary_rows.append({
                "model": model, "metric": metric, "nominal": values[0],
                "minimum_across_perturbations": min(values.values()),
                "maximum_across_perturbations": max(values.values()),
                **{f"offset_{offset:+d}m": values[offset] for offset in OFFSETS},
            })
    with (PARENT / "metric_sensitivity_envelope.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)

    effect_rows = []
    for comparison in ("deep_prewitt_vs_spatial_prior",
                       "deep_prewitt_vs_parameter_matched_ordinary_conv"):
        for effect in ("boundary_f1_30m", "asd_reduction_m"):
            vals = {offset: payloads[offset]["paired_date_effects"][comparison][effect]
                    for offset in OFFSETS}
            means = {offset: vals[offset]["mean_improvement"] for offset in OFFSETS}
            effect_rows.append({
                "comparison": comparison, "effect": effect,
                "nominal_mean": means[0],
                "minimum_mean_across_perturbations": min(means.values()),
                "maximum_mean_across_perturbations": max(means.values()),
                "all_offsets_same_direction": all(value > 0 for value in means.values()),
                "minimum_dates_improved": min(vals[o]["dates_improved"] for o in OFFSETS),
                **{f"offset_{offset:+d}m": means[offset] for offset in OFFSETS},
            })
    with (PARENT / "paired_effect_sensitivity_envelope.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effect_rows[0]))
        writer.writeheader(); writer.writerows(effect_rows)
    (PARENT / "manual_uncertainty_sensitivity.json").write_text(json.dumps({
        "status": "FROZEN_MODEL_MANUAL_REFERENCE_SENSITIVITY",
        "interpretation": "Deterministic +/-10 m and +/-20 m polygon perturbations; not a probabilistic confidence interval.",
        "offsets_m": list(OFFSETS), "metric_rows": summary_rows,
        "paired_effect_rows": effect_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(effect_rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
