from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "public_backtest"
SWEEP_DIR = ANALYSIS / "target_budget_sweep" / "main_policy"
OUT_DIR = ANALYSIS / "target_budget_sweep" / "main_policy_summary"

SOURCE_FILES = [
    SWEEP_DIR / "swe_lite_main_policy_pair_summary.csv",
    SWEEP_DIR / "swe_verified_main_policy_pair_summary.csv",
    SWEEP_DIR / "appworld_main_policy_pair_summary.csv",
    SWEEP_DIR / "tau_main_policy_pair_summary.csv",
    SWEEP_DIR / "osworld_main_policy_pair_summary.csv",
]

PRIMARY_ALPHA = 0.05
PRIMARY_GAMMA = 0.05
PRIMARY_ETA = 0.25
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260611


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return float("nan")
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    if weights.sum() == 0:
        return float("nan")
    return float((weights * values).sum() / weights.sum())


def read_pair_summary() -> pd.DataFrame:
    frames = []
    missing = [path for path in SOURCE_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sweep outputs: {missing}")
    for path in SOURCE_FILES:
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df["pair_key"] = df.apply(
        lambda row: "|||".join(
            sorted([str(row["tested_system"]), str(row["comparison_system"])])
        ),
        axis=1,
    )
    return df


def aggregate(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "policy_alpha", "budget_cap", "policy_label"]
    for group_keys, group in pair_summary.groupby(keys, dropna=False):
        source, threshold, alpha, budget, policy = group_keys
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "policy_alpha": alpha,
                "budget_cap": budget,
                "policy_label": policy,
                "directed_pairs": int(group["pair_label"].nunique()),
                "positive_all_task_decisions": int(pos["pair_label"].nunique()),
                "negative_all_task_decisions": int(neg["pair_label"].nunique()),
                "conditional_false_accept_rate": weighted_mean(
                    neg, "false_accept_rate"
                ),
                "conditional_false_reject_rate": weighted_mean(
                    pos, "false_reject_rate"
                ),
                "decision_error_rate": weighted_mean(group, "decision_error_rate"),
                "coverage_failure_rate": weighted_mean(group, "coverage_failure_rate"),
                "defer_rate": weighted_mean(group, "defer_rate"),
                "coverage_ok_rate": weighted_mean(group, "coverage_ok_rate"),
                "defer_no_coverage_rate": weighted_mean(
                    group, "defer_no_coverage_rate"
                ),
                "defer_uncertain_rate": weighted_mean(group, "defer_uncertain_rate"),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def target_grid(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in aggregate_df.groupby(
        ["source", "threshold", "policy_alpha", "policy_label"], dropna=False
    ):
        source, threshold, alpha, policy = keys
        ordered = group.sort_values("budget_cap")
        for gamma in [0.0, 0.05, 0.10]:
            for eta in [0.10, 0.25, 0.40]:
                mask = (
                    (ordered["conditional_false_accept_rate"] <= alpha)
                    & (ordered["conditional_false_reject_rate"] <= alpha)
                    & (ordered["coverage_failure_rate"] <= gamma)
                    & (ordered["defer_rate"] <= eta)
                )
                row = ordered[mask].iloc[0] if mask.any() else None
                rows.append(
                    {
                        "source": source,
                        "threshold": threshold,
                        "policy_alpha": alpha,
                        "coverage_target": gamma,
                        "defer_target": eta,
                        "policy_label": policy,
                        "minimum_sufficient_budget": (
                            float(row["budget_cap"]) if row is not None else math.nan
                        ),
                        "defer_rate_at_minimum": (
                            float(row["defer_rate"]) if row is not None else math.nan
                        ),
                        "defer_no_coverage_at_minimum": (
                            float(row["defer_no_coverage_rate"])
                            if row is not None
                            else math.nan
                        ),
                        "defer_uncertain_at_minimum": (
                            float(row["defer_uncertain_rate"])
                            if row is not None
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["source", "threshold", "policy_alpha", "coverage_target", "defer_target"]
    )


def metric_values(group: pd.DataFrame) -> dict[str, float]:
    pos = group[group["truth_accept"]]
    neg = group[~group["truth_accept"]]
    return {
        "conditional_false_accept_rate": weighted_mean(neg, "false_accept_rate"),
        "conditional_false_reject_rate": weighted_mean(pos, "false_reject_rate"),
        "coverage_failure_rate": weighted_mean(group, "coverage_failure_rate"),
        "defer_rate": weighted_mean(group, "defer_rate"),
        "coverage_ok_rate": weighted_mean(group, "coverage_ok_rate"),
        "defer_no_coverage_rate": weighted_mean(group, "defer_no_coverage_rate"),
        "defer_uncertain_rate": weighted_mean(group, "defer_uncertain_rate"),
    }


def pair_bootstrap_intervals(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    subset = pair_summary[pair_summary["policy_alpha"].eq(PRIMARY_ALPHA)].copy()
    keys = ["source", "threshold", "budget_cap", "policy_label"]
    for group_keys, group in subset.groupby(keys, dropna=False):
        source, threshold, budget, policy = group_keys
        point = metric_values(group)
        pair_level = group.drop_duplicates("pair_label").copy()
        n_pairs = len(pair_level)
        sample_idx = rng.integers(0, n_pairs, size=(BOOTSTRAP_REPS, n_pairs))

        weights = pair_level["rows"].to_numpy(dtype=float)
        truth = pair_level["truth_accept"].to_numpy(dtype=bool)
        metric_arrays = {
            "conditional_false_accept_rate": pair_level[
                "false_accept_rate"
            ].to_numpy(dtype=float),
            "conditional_false_reject_rate": pair_level[
                "false_reject_rate"
            ].to_numpy(dtype=float),
            "coverage_failure_rate": pair_level[
                "coverage_failure_rate"
            ].to_numpy(dtype=float),
            "defer_rate": pair_level["defer_rate"].to_numpy(dtype=float),
            "coverage_ok_rate": pair_level["coverage_ok_rate"].to_numpy(dtype=float),
            "defer_no_coverage_rate": pair_level[
                "defer_no_coverage_rate"
            ].to_numpy(dtype=float),
            "defer_uncertain_rate": pair_level[
                "defer_uncertain_rate"
            ].to_numpy(dtype=float),
        }

        bootstrap_values: dict[str, np.ndarray] = {}
        all_sample_weights = weights[sample_idx]
        all_weight_sum = all_sample_weights.sum(axis=1)
        for metric in [
            "coverage_failure_rate",
            "defer_rate",
            "coverage_ok_rate",
            "defer_no_coverage_rate",
            "defer_uncertain_rate",
        ]:
            values = metric_arrays[metric][sample_idx]
            bootstrap_values[metric] = (
                values * all_sample_weights
            ).sum(axis=1) / all_weight_sum

        neg_mask = (~truth)[sample_idx]
        neg_weights = all_sample_weights * neg_mask
        neg_weight_sum = neg_weights.sum(axis=1)
        neg_values = metric_arrays["conditional_false_accept_rate"][sample_idx]
        bootstrap_values["conditional_false_accept_rate"] = np.divide(
            (neg_values * neg_weights).sum(axis=1),
            neg_weight_sum,
            out=np.full(BOOTSTRAP_REPS, np.nan),
            where=neg_weight_sum > 0,
        )

        pos_mask = truth[sample_idx]
        pos_weights = all_sample_weights * pos_mask
        pos_weight_sum = pos_weights.sum(axis=1)
        pos_values = metric_arrays["conditional_false_reject_rate"][sample_idx]
        bootstrap_values["conditional_false_reject_rate"] = np.divide(
            (pos_values * pos_weights).sum(axis=1),
            pos_weight_sum,
            out=np.full(BOOTSTRAP_REPS, np.nan),
            where=pos_weight_sum > 0,
        )

        for metric, arr in bootstrap_values.items():
            rows.append(
                {
                    "source": source,
                    "threshold": threshold,
                    "budget_cap": budget,
                    "policy_label": policy,
                    "metric": metric,
                    "point": point[metric],
                    "ci_low": float(np.nanquantile(arr, 0.025)),
                    "ci_high": float(np.nanquantile(arr, 0.975)),
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    "bootstrap_unit": "directed system pair",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["source", "threshold", "budget_cap", "metric"]
    )


def collapse_orientations(pair_summary: pd.DataFrame) -> pd.DataFrame:
    primary = pair_summary[pair_summary["policy_alpha"].eq(PRIMARY_ALPHA)].copy()
    keep = (
        primary.sort_values("full_delta", ascending=False)
        .groupby(["source", "pair_key"], as_index=False)
        .first()[["source", "pair_key", "tested_system", "comparison_system"]]
    )
    collapsed = primary.merge(
        keep,
        on=["source", "pair_key", "tested_system", "comparison_system"],
        how="inner",
    )
    collapsed_agg = aggregate(collapsed)
    directed_agg = aggregate(primary)
    directed_grid = minimum_budget_table(directed_agg)
    collapsed_grid = minimum_budget_table(collapsed_agg)
    merged = directed_grid.merge(
        collapsed_grid,
        on=["source", "threshold", "policy_alpha", "policy_label"],
        suffixes=("_directed", "_undirected"),
        how="outer",
    )
    return merged.sort_values(["source", "threshold"])


def minimum_budget_table(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "policy_alpha", "policy_label"]
    for group_keys, group in aggregate_df.groupby(keys, dropna=False):
        source, threshold, alpha, policy = group_keys
        ordered = group.sort_values("budget_cap")
        mask = (
            (ordered["conditional_false_accept_rate"] <= alpha)
            & (ordered["conditional_false_reject_rate"] <= alpha)
            & (ordered["coverage_failure_rate"] <= PRIMARY_GAMMA)
            & (ordered["defer_rate"] <= PRIMARY_ETA)
        )
        row = ordered[mask].iloc[0] if mask.any() else None
        first = ordered.iloc[0]
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "policy_alpha": alpha,
                "policy_label": policy,
                "minimum_sufficient_budget": (
                    float(row["budget_cap"]) if row is not None else math.nan
                ),
                "defer_rate_at_minimum": (
                    float(row["defer_rate"]) if row is not None else math.nan
                ),
                "positive_all_task_decisions": int(
                    first["positive_all_task_decisions"]
                ),
                "negative_all_task_decisions": int(
                    first["negative_all_task_decisions"]
                ),
                "pair_count": int(first["directed_pairs"]),
            }
        )
    return pd.DataFrame(rows)


def margin_bin(signed_margin: float) -> str:
    if signed_margin < -0.10:
        return "clear_reject"
    if signed_margin < -0.05:
        return "moderate_reject"
    if signed_margin < 0.05:
        return "near_boundary"
    if signed_margin < 0.10:
        return "moderate_accept"
    return "clear_accept"


def swe_verified_features() -> pd.DataFrame:
    task_rows = pd.read_csv(ANALYSIS / "swebench_smoke" / "swebench_task_rows.csv")
    task_rows = task_rows[task_rows["split"].eq("verified")].copy()
    summary = pd.read_csv(SWEEP_DIR / "swe_verified_main_policy_pair_summary.csv")
    pairs = summary[["tested_system", "comparison_system"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for _, pair in pairs.iterrows():
        tested = task_rows[task_rows["submission"].eq(pair["tested_system"])]
        comparison = task_rows[task_rows["submission"].eq(pair["comparison_system"])]
        merged = tested.merge(
            comparison,
            on="instance_id",
            suffixes=("_tested", "_comparison"),
            how="inner",
        )
        diff = merged["resolved_tested"].astype(float) - merged[
            "resolved_comparison"
        ].astype(float)
        rows.append(
            {
                "tested_system": pair["tested_system"],
                "comparison_system": pair["comparison_system"],
                "pair_label": f'{pair["tested_system"]}__vs__{pair["comparison_system"]}',
                "overlap_tasks_feature": int(len(merged)),
                "repo_count": int(merged["repo_tested"].nunique()),
                "diff_sd": float(diff.std(ddof=0)),
                "tested_success_rate": float(merged["resolved_tested"].mean()),
                "comparison_success_rate": float(merged["resolved_comparison"].mean()),
                "mean_success_rate": float(
                    (
                        merged["resolved_tested"].astype(float)
                        + merged["resolved_comparison"].astype(float)
                    ).mean()
                    / 2
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_correlation(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return float("nan")
    xr = x[valid].rank(method="average")
    yr = y[valid].rank(method="average")
    if float(xr.std(ddof=0)) == 0.0 or float(yr.std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def swe_verified_deferral_diagnosis(pair_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verified = pair_summary[
        pair_summary["source"].eq("SWE-bench Verified")
        & pair_summary["policy_alpha"].eq(PRIMARY_ALPHA)
    ].copy()
    decomp_rows: list[dict[str, object]] = []
    for keys, group in verified.groupby(["threshold", "budget_cap"], dropna=False):
        threshold, budget = keys
        if budget not in {0.25, 0.50, 0.75, 0.95}:
            continue
        vals = metric_values(group)
        vals.update(
            {
                "threshold": threshold,
                "budget_cap": budget,
                "pair_count": int(group["pair_label"].nunique()),
            }
        )
        decomp_rows.append(vals)
    decomp = pd.DataFrame(decomp_rows).sort_values(["threshold", "budget_cap"])

    margin_rows: list[dict[str, object]] = []
    checked = verified[verified["budget_cap"].isin([0.75, 0.95])].copy()
    checked["margin_bin"] = checked["signed_margin"].map(margin_bin)
    for keys, group in checked.groupby(["threshold", "budget_cap", "margin_bin"]):
        threshold, budget, bin_name = keys
        vals = metric_values(group)
        vals.update(
            {
                "threshold": threshold,
                "budget_cap": budget,
                "margin_bin": bin_name,
                "pair_count": int(group["pair_label"].nunique()),
                "mean_abs_margin": float(group["absolute_margin"].mean()),
            }
        )
        margin_rows.append(vals)
    by_margin = pd.DataFrame(margin_rows).sort_values(
        ["threshold", "budget_cap", "margin_bin"]
    )

    features = swe_verified_features()
    at_75 = verified[verified["budget_cap"].eq(0.75)].merge(
        features,
        on=["tested_system", "comparison_system", "pair_label"],
        how="left",
    )
    corr_rows: list[dict[str, object]] = []
    feature_cols = [
        "absolute_margin",
        "repo_count",
        "overlap_tasks_feature",
        "diff_sd",
        "mean_success_rate",
    ]
    for threshold, group in at_75.groupby("threshold"):
        for feature in feature_cols:
            corr_rows.append(
                {
                    "threshold": threshold,
                    "budget_cap": 0.75,
                    "feature": feature,
                    "spearman_with_defer_rate": rank_correlation(
                        group[feature], group["defer_rate"]
                    ),
                }
            )
    correlations = pd.DataFrame(corr_rows).sort_values(["threshold", "feature"])
    return decomp, by_margin, correlations


def pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "None"
    return f"{100 * float(value):.{digits}f}%"


def budget(value: object) -> str:
    if value is None or pd.isna(value):
        return "None"
    return f"{100 * float(value):.0f}%"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display = display.fillna("None")
    columns = [str(col) for col in display.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    fine_min: pd.DataFrame,
    target: pd.DataFrame,
    decomp: pd.DataFrame,
    by_margin: pd.DataFrame,
    correlations: pd.DataFrame,
    intervals: pd.DataFrame,
    orientation: pd.DataFrame,
) -> None:
    primary = fine_min[
        fine_min["policy_alpha"].eq(PRIMARY_ALPHA)
    ].copy()
    primary["threshold"] = primary["threshold"].map(lambda x: f"{100*x:.0f} pp")
    primary["minimum_sufficient_budget"] = primary["minimum_sufficient_budget"].map(
        budget
    )
    primary["defer_rate_at_minimum"] = primary["defer_rate_at_minimum"].map(pct)

    target_short = target[
        target["coverage_target"].eq(PRIMARY_GAMMA)
        & target["defer_target"].isin([0.10, 0.25, 0.40])
    ].copy()
    target_short["threshold"] = target_short["threshold"].map(lambda x: f"{100*x:.0f} pp")
    target_short["policy_alpha"] = target_short["policy_alpha"].map(lambda x: f"{x:.2f}")
    target_short["defer_target"] = target_short["defer_target"].map(pct)
    target_short["minimum_sufficient_budget"] = target_short[
        "minimum_sufficient_budget"
    ].map(budget)

    decomp_display = decomp.copy()
    decomp_display["threshold"] = decomp_display["threshold"].map(lambda x: f"{100*x:.0f} pp")
    decomp_display["budget_cap"] = decomp_display["budget_cap"].map(budget)
    for col in [
        "defer_rate",
        "coverage_ok_rate",
        "defer_no_coverage_rate",
        "defer_uncertain_rate",
        "conditional_false_accept_rate",
        "conditional_false_reject_rate",
    ]:
        decomp_display[col] = decomp_display[col].map(pct)

    margin_display = by_margin.copy()
    margin_display["threshold"] = margin_display["threshold"].map(lambda x: f"{100*x:.0f} pp")
    margin_display["budget_cap"] = margin_display["budget_cap"].map(budget)
    for col in ["defer_rate", "defer_no_coverage_rate", "defer_uncertain_rate"]:
        margin_display[col] = margin_display[col].map(pct)
    margin_display["mean_abs_margin"] = margin_display["mean_abs_margin"].map(
        lambda x: f"{100*x:.1f} pp"
    )

    orientation_display = orientation.copy()
    for col in [
        "minimum_sufficient_budget_directed",
        "minimum_sufficient_budget_undirected",
    ]:
        orientation_display[col] = orientation_display[col].map(budget)
    for col in ["defer_rate_at_minimum_directed", "defer_rate_at_minimum_undirected"]:
        orientation_display[col] = orientation_display[col].map(pct)
    orientation_display["threshold"] = orientation_display["threshold"].map(
        lambda x: f"{100*x:.0f} pp"
    )

    interval_focus = intervals[
        intervals["budget_cap"].isin([0.25, 0.50, 0.75])
        & intervals["metric"].isin(
            [
                "conditional_false_accept_rate",
                "conditional_false_reject_rate",
                "defer_rate",
                "defer_uncertain_rate",
            ]
        )
    ].copy()
    interval_focus["threshold"] = interval_focus["threshold"].map(
        lambda x: f"{100*x:.0f} pp"
    )
    interval_focus["budget_cap"] = interval_focus["budget_cap"].map(budget)
    for col in ["point", "ci_low", "ci_high"]:
        interval_focus[col] = interval_focus[col].map(pct)

    lines = [
        "# Main-Policy Sensitivity and Uncertainty Report",
        "",
        "These diagnostics use completed public task-level records and the paper's coverage-aware main policies. They are empirical finite-task replay diagnostics, not formal stopping guarantees.",
        "",
        "## Fine-Grained Minimum Budgets",
        "",
        "Budgets are swept in 5 percent increments from 5 percent to 95 percent. The table below uses the paper's primary targets, alpha = 0.05, gamma = 0.05, eta = 0.25.",
        "",
        markdown_table(
            primary[
                [
                    "source",
                    "threshold",
                    "policy_label",
                    "minimum_sufficient_budget",
                    "defer_rate_at_minimum",
                    "positive_all_task_decisions",
                    "negative_all_task_decisions",
                ]
            ]
        ),
        "",
        "## Target Grid",
        "",
        "Alpha changes both the bootstrap tail threshold inside the policy and the conditional decision-error target. Gamma has little effect for the coverage-aware policies because they defer before accepting or rejecting without required task-group coverage.",
        "",
        markdown_table(
            target_short[
                [
                    "source",
                    "threshold",
                    "policy_alpha",
                    "defer_target",
                    "minimum_sufficient_budget",
                ]
            ]
        ),
        "",
        "## SWE-bench Verified Deferral Diagnosis",
        "",
        "The decomposition separates deferral caused by missing task-group coverage from deferral after coverage is satisfied but bootstrap evidence is still insufficient.",
        "",
        markdown_table(
            decomp_display[
                [
                    "threshold",
                    "budget_cap",
                    "defer_rate",
                    "coverage_ok_rate",
                    "defer_no_coverage_rate",
                    "defer_uncertain_rate",
                    "conditional_false_accept_rate",
                    "conditional_false_reject_rate",
                ]
            ]
        ),
        "",
        "At 75 percent budget, most remaining deferral is evidence uncertainty after coverage has been satisfied, not missing repository coverage.",
        "",
        markdown_table(
            margin_display[
                [
                    "threshold",
                    "budget_cap",
                    "margin_bin",
                    "pair_count",
                    "mean_abs_margin",
                    "defer_rate",
                    "defer_no_coverage_rate",
                    "defer_uncertain_rate",
                ]
            ]
        ),
        "",
        "Feature correlations are Spearman correlations at 75 percent budget. Negative absolute-margin correlation means pairs closer to the threshold defer more often.",
        "",
        markdown_table(correlations),
        "",
        "## Pair-Level Uncertainty",
        "",
        "Intervals are cluster bootstraps over directed system pairs. They quantify replay variability across pairs and reveal orders; they do not turn the replay into a population guarantee.",
        "",
        markdown_table(
            interval_focus[
                [
                    "source",
                    "threshold",
                    "budget_cap",
                    "metric",
                    "point",
                    "ci_low",
                    "ci_high",
                ]
            ]
        ),
        "",
        "## Orientation Collapse",
        "",
        "The undirected check keeps one orientation per unordered pair, choosing the orientation with the larger all-task delta. It is a dependence check, not the primary estimand.",
        "",
        markdown_table(
            orientation_display[
                [
                    "source",
                    "threshold",
                    "minimum_sufficient_budget_directed",
                    "minimum_sufficient_budget_undirected",
                    "positive_all_task_decisions_directed",
                    "negative_all_task_decisions_directed",
                    "positive_all_task_decisions_undirected",
                    "negative_all_task_decisions_undirected",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "The fine sweep sharpens the main result without changing the story: AppWorld remains sufficient at the lowest tested region, SWE-bench Lite needs a high budget under no-slack coverage, and SWE-bench Verified remains non-actionable under the primary defer target. The strongest new scientific point is the Verified diagnosis: high deferral is mainly an uncertainty issue after repository coverage, especially for near-threshold comparisons. Orientation collapse changes several minimum budgets, so the paper should keep directed system pairs as the estimand and report the collapse as a dependence check.",
        "",
    ]
    (OUT_DIR / "main_policy_sensitivity_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_summary = read_pair_summary()
    aggregate_df = aggregate(pair_summary)
    target = target_grid(aggregate_df)
    fine_min = minimum_budget_table(aggregate_df)
    intervals = pair_bootstrap_intervals(pair_summary)
    orientation = collapse_orientations(pair_summary)
    decomp, by_margin, correlations = swe_verified_deferral_diagnosis(pair_summary)

    pair_summary.to_csv(OUT_DIR / "main_policy_pair_summary.csv", index=False)
    aggregate_df.to_csv(OUT_DIR / "main_policy_aggregate.csv", index=False)
    target.to_csv(OUT_DIR / "main_policy_target_grid.csv", index=False)
    fine_min.to_csv(OUT_DIR / "main_policy_minimum_budgets.csv", index=False)
    intervals.to_csv(OUT_DIR / "pair_level_metric_intervals.csv", index=False)
    orientation.to_csv(OUT_DIR / "orientation_collapse_summary.csv", index=False)
    decomp.to_csv(OUT_DIR / "swe_verified_deferral_decomposition.csv", index=False)
    by_margin.to_csv(OUT_DIR / "swe_verified_deferral_by_margin.csv", index=False)
    correlations.to_csv(
        OUT_DIR / "swe_verified_deferral_feature_correlations.csv", index=False
    )
    metadata = {
        "primary_alpha": PRIMARY_ALPHA,
        "primary_gamma": PRIMARY_GAMMA,
        "primary_eta": PRIMARY_ETA,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "directed system pair",
    }
    (OUT_DIR / "summary_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_report(fine_min, target, decomp, by_margin, correlations, intervals, orientation)
    print(json.dumps({"out_dir": str(OUT_DIR), "rows": len(pair_summary)}, indent=2))


if __name__ == "__main__":
    main()
