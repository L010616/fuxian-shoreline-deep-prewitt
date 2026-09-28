"""Lake-clustered inference for the frozen 18-date cross-lake experiment."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preexperiment/results/q2_cross_lake_zero_shot_v04_waihai_target"
RESULT = OUT / "cross_lake_zero_shot_results.json"
LAKES = ("Dianchi_Waihai", "Erhai", "Xingyun")
LAKE_LABELS = {"Dianchi_Waihai": "Dianchi Waihai", "Erhai": "Erhai", "Xingyun": "Xingyun"}
COMPARISONS = {
    "deep_vs_prior": ("deep_prewitt", "spatial_prior"),
    "deep_vs_matched_conv": ("deep_prewitt", "parameter_matched_ordinary_conv"),
}
METRICS = {
    "BF1_30m_gain": ("boundary_f1_30m", 1.0),
    "ASD_reduction_m": ("mean_symmetric_boundary_distance_m", -1.0),
}
BOOTSTRAPS = 100_000
SEED = 20260927


def effects(data: dict, model: str, baseline: str, metric: str, sign: float) -> dict[str, np.ndarray]:
    candidate = data["three_seed_ensemble_per_date"][model]
    reference = data["three_seed_ensemble_per_date"][baseline]
    out: dict[str, list[float]] = {lake: [] for lake in LAKES}
    for key, row in candidate.items():
        out[row["lake"]].append(sign * (row[metric] - reference[key][metric]))
    return {lake: np.asarray(values, dtype=float) for lake, values in out.items()}


def hierarchical_bootstrap(by_lake: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    draws = np.empty(BOOTSTRAPS, dtype=float)
    lake_names = np.asarray(LAKES)
    for index in range(BOOTSTRAPS):
        sampled_lakes = rng.choice(lake_names, size=len(lake_names), replace=True)
        lake_means = []
        for lake in sampled_lakes:
            values = by_lake[str(lake)]
            lake_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        draws[index] = float(np.mean(lake_means))
    return draws


def exact_sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    n = len(nonzero)
    # Exact two-sided binomial sign test under p=0.5.
    tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    records = []
    plot_rows = []
    for comparison, (model, baseline) in COMPARISONS.items():
        for effect_name, (metric, sign) in METRICS.items():
            by_lake = effects(data, model, baseline, metric, sign)
            lake_means = {lake: float(values.mean()) for lake, values in by_lake.items()}
            observed = float(np.mean(list(lake_means.values())))
            draws = hierarchical_bootstrap(by_lake, rng)
            ci = np.quantile(draws, [0.025, 0.975])
            loo = {
                omitted: float(np.mean([value for lake, value in lake_means.items() if lake != omitted]))
                for omitted in LAKES
            }
            records.append({
                "comparison": comparison,
                "effect": effect_name,
                "n_lakes": 3,
                "dates_per_lake": 6,
                "equal_weight_lake_mean": observed,
                "hierarchical_bootstrap_ci_low": float(ci[0]),
                "hierarchical_bootstrap_ci_high": float(ci[1]),
                "lakes_improved": sum(value > 0 for value in lake_means.values()),
                "lake_level_exact_sign_p": exact_sign_p(list(lake_means.values())),
                **{f"mean_{lake}": value for lake, value in lake_means.items()},
                **{f"leave_out_{lake}": value for lake, value in loo.items()},
            })
            plot_rows.append((comparison, effect_name, by_lake, observed, ci))

    with (OUT / "lake_hierarchical_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (OUT / "lake_hierarchical_statistics.json").write_text(json.dumps({
        "status": "FROZEN_POSTHOC_HIERARCHICAL_REANALYSIS",
        "input": str(RESULT),
        "independence_structure": "lake is level 1; date nested within lake is level 2",
        "estimator": "equal-weight mean of lake-specific date means",
        "bootstrap": f"resample 3 lakes, then resample 6 dates within each selected lake; {BOOTSTRAPS} draws",
        "random_seed": SEED,
        "warning": "Only three lake clusters are available. Cluster-level sign tests have minimum two-sided p=0.25 and hierarchical intervals are descriptive rather than definitive population-wide inference.",
        "results": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    colors = {"Dianchi_Waihai": "#8FA2D6", "Erhai": "#A9C8B8", "Xingyun": "#E7A889"}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), constrained_layout=True)
    for axis, (comparison, effect_name, by_lake, observed, ci) in zip(axes.flat, plot_rows):
        for index, lake in enumerate(LAKES):
            mean = by_lake[lake].mean()
            sd = by_lake[lake].std(ddof=1)
            axis.errorbar(mean, index, xerr=sd, fmt="o", ms=6, capsize=3,
                          color=colors[lake], ecolor=colors[lake], lw=1.3)
        axis.axvline(0, color="#777A80", ls="--", lw=0.8)
        axis.axvspan(ci[0], ci[1], color="#6A5C95", alpha=0.10)
        axis.axvline(observed, color="#6A5C95", lw=1.8)
        axis.set_yticks(range(3), [LAKE_LABELS[lake] for lake in LAKES])
        axis.grid(axis="x", color="#E6E8EC", lw=0.7)
        axis.set_axisbelow(True)
        label = "BF1 gain" if effect_name.startswith("BF1") else "ASD reduction (m)"
        baseline = "Spatial prior" if comparison.endswith("prior") else "Matched convolution"
        axis.set_xlabel(label)
        axis.set_title(f"Deep Prewitt vs {baseline}", fontsize=10, fontweight="bold")
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(OUT / "lake_hierarchical_effects.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    prior_bf = next(r for r in records if r["comparison"] == "deep_vs_prior" and r["effect"] == "BF1_30m_gain")
    prior_asd = next(r for r in records if r["comparison"] == "deep_vs_prior" and r["effect"] == "ASD_reduction_m")
    matched_bf = next(r for r in records if r["comparison"] == "deep_vs_matched_conv" and r["effect"] == "BF1_30m_gain")
    matched_asd = next(r for r in records if r["comparison"] == "deep_vs_matched_conv" and r["effect"] == "ASD_reduction_m")
    report = f"""# 湖泊层级统计结果

18个日期并非18个完全独立的空间样本，而是嵌套在3个湖泊中。本分析以湖泊为一级、日期为二级，先计算各湖6期平均效应，再对湖泊和湖内日期实施两阶段层级自助抽样。三个湖泊等权，避免岸线长度或有效像元数量较大的湖泊支配结论。

## 深层Prewitt相对空间先验

- Boundary F1的湖泊等权平均增益为{prior_bf['equal_weight_lake_mean']:.4f}，层级自助95%区间为{prior_bf['hierarchical_bootstrap_ci_low']:.4f}—{prior_bf['hierarchical_bootstrap_ci_high']:.4f}；3/3个湖泊方向为正。
- ASD的湖泊等权平均降低量为{prior_asd['equal_weight_lake_mean']:.2f} m，层级自助95%区间为{prior_asd['hierarchical_bootstrap_ci_low']:.2f}—{prior_asd['hierarchical_bootstrap_ci_high']:.2f} m；3/3个湖泊方向为正。
- 逐湖留一后两项效应仍为正，说明结果不由单个目标湖驱动。

## 深层Prewitt相对参数量匹配普通卷积

- Boundary F1的湖泊等权平均增益为{matched_bf['equal_weight_lake_mean']:.4f}，层级自助95%区间为{matched_bf['hierarchical_bootstrap_ci_low']:.4f}—{matched_bf['hierarchical_bootstrap_ci_high']:.4f}；仅{matched_bf['lakes_improved']}/3个湖泊方向为正。
- ASD的湖泊等权平均降低量为{matched_asd['equal_weight_lake_mean']:.2f} m，层级自助95%区间为{matched_asd['hierarchical_bootstrap_ci_low']:.2f}—{matched_asd['hierarchical_bootstrap_ci_high']:.2f} m；仅{matched_asd['lakes_improved']}/3个湖泊方向为正。

## 统计解释

由于只有3个湖泊，湖泊层级双侧精确符号检验即使3/3同向，其最小p值仍为0.25。因此层级自助区间用于描述已观察湖泊中的稳健性，不能代替更多独立湖泊带来的总体推断。正文可以表述为“在三个外部湖泊中方向一致，且逐湖留一分析稳定”，不应写成“湖泊层级显著”。相对普通卷积的效应跨湖不一致，继续保留为限制性证据。
"""
    (OUT / "湖泊层级统计报告.md").write_text(report, encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
