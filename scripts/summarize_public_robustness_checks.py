from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "public_backtest"
OUT_DIR = ANALYSIS / "public_robustness_checks"

RISK_TARGET = 0.05
COVERAGE_TARGET = 0.05
DEFER_TARGET = 0.25

POLICY_LABELS = {
    "bootstrap_ci": "bootstrap CI",
    "coverage_bootstrap": "coverage-aware bootstrap",
    "cost_aware_forced": "cost-aware forced",
    "fixed_stratified_forced": "stratified forced",
    "fixed_uniform_forced": "uniform forced",
    "repo_coverage_bootstrap": "repo-coverage bootstrap",
    "repo_stratified_forced": "repo-stratified forced",
    "split_difficulty_coverage_bootstrap": "split-difficulty coverage bootstrap",
    "split_difficulty_stratified_forced": "split-difficulty stratified forced",
    "uniform_forced": "uniform forced",
}


SOURCES = [
    {
        "source": "SWE-bench Lite",
        "summary": ANALYSIS
        / "swebench_lite_replay_coverage_slack0"
        / "swebench_verified_replay_summary.csv",
        "aggregate": ANALYSIS
        / "swebench_lite_replay_coverage_slack0"
        / "swebench_verified_policy_aggregate.csv",
        "min_budget": ANALYSIS
        / "swebench_lite_replay_coverage_slack0"
        / "swebench_verified_min_sufficient_budget.csv",
        "slackened_min_budget": ANALYSIS
        / "swebench_lite_replay_coverage_slack1"
        / "swebench_verified_min_sufficient_budget.csv",
        "main_policy": "repo-coverage bootstrap",
        "id_cols": ("candidate", "reference"),
        "truth_col": "truth_promote",
        "false_accept_col": "false_promotion_rate",
        "false_reject_col": "false_rejection_rate",
        "defer_col": "abstain_rate",
        "positive_col": "positive_reference_decisions",
        "negative_col": "negative_reference_decisions",
    },
    {
        "source": "SWE-bench Verified",
        "summary": ANALYSIS
        / "swebench_verified_replay_coverage_slack0"
        / "swebench_verified_replay_summary.csv",
        "aggregate": ANALYSIS
        / "swebench_verified_replay_coverage_slack0"
        / "swebench_verified_policy_aggregate.csv",
        "min_budget": ANALYSIS
        / "swebench_verified_replay_coverage_slack0"
        / "swebench_verified_min_sufficient_budget.csv",
        "slackened_min_budget": ANALYSIS
        / "swebench_verified_replay_coverage_slack1"
        / "swebench_verified_min_sufficient_budget.csv",
        "main_policy": "repo-coverage bootstrap",
        "id_cols": ("candidate", "reference"),
        "truth_col": "truth_promote",
        "false_accept_col": "false_promotion_rate",
        "false_reject_col": "false_rejection_rate",
        "defer_col": "abstain_rate",
        "positive_col": "positive_reference_decisions",
        "negative_col": "negative_reference_decisions",
    },
    {
        "source": "AppWorld",
        "summary": ANALYSIS
        / "appworld_replay_coverage_slack0"
        / "appworld_replay_summary.csv",
        "aggregate": ANALYSIS
        / "appworld_replay_coverage_slack0"
        / "appworld_policy_aggregate.csv",
        "min_budget": ANALYSIS
        / "appworld_replay_coverage_slack0"
        / "appworld_min_sufficient_budget.csv",
        "slackened_min_budget": ANALYSIS
        / "appworld_replay_coverage_slack1"
        / "appworld_min_sufficient_budget.csv",
        "main_policy": "split-difficulty coverage bootstrap",
        "id_cols": ("tested_system", "comparison_system"),
        "truth_col": "truth_accept",
        "false_accept_col": "false_accept_rate",
        "false_reject_col": "false_reject_rate",
        "defer_col": "defer_rate",
        "positive_col": "positive_full_decisions",
        "negative_col": "negative_full_decisions",
    },
    {
        "source": "tau-bench",
        "summary": ANALYSIS
        / "tau2_pair_grid_failure_coverage_slack0"
        / "public_tau2_replay_summary.csv",
        "aggregate": ANALYSIS
        / "tau2_pair_grid_summary_coverage_slack0"
        / "pair_grid_policy_aggregate.csv",
        "min_budget": ANALYSIS
        / "tau2_pair_grid_summary_coverage_slack0"
        / "pair_grid_min_sufficient_budget.csv",
        "slackened_min_budget": ANALYSIS
        / "tau2_pair_grid_summary_coverage_slack1"
        / "pair_grid_min_sufficient_budget.csv",
        "main_policy": "coverage-aware bootstrap",
        "id_cols": ("candidate", "reference"),
        "truth_col": "truth_promote",
        "false_accept_col": "false_promotion_rate",
        "false_reject_col": "false_rejection_rate",
        "defer_col": "abstain_rate",
        "positive_col": "positive_reference_decisions",
        "negative_col": "negative_reference_decisions",
    },
]


def budget_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "NF"
    if isinstance(value, str):
        return value
    return f"{float(value) * 100:.0f}%"


def pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value) * 100:.{digits}f}%"


def read_main_policy(path: Path, policy_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["policy_label"].eq(policy_label)].copy()


def build_coverage_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in SOURCES:
        primary = read_main_policy(spec["min_budget"], spec["main_policy"])
        slackened = read_main_policy(spec["slackened_min_budget"], spec["main_policy"])
        thresholds = sorted(
            set(primary["threshold"].astype(float)).intersection(
                set(slackened["threshold"].astype(float))
            )
        )
        for threshold in thresholds:
            primary_row = primary[primary["threshold"].eq(threshold)].iloc[0]
            slackened_row = slackened[slackened["threshold"].eq(threshold)].iloc[0]
            rows.append(
                {
                    "source": spec["source"],
                    "threshold": threshold,
                    "main_policy": spec["main_policy"],
                    "no_slack_floor_budget": primary_row.get(
                        "minimum_sufficient_budget"
                    ),
                    "slackened_floor_budget": slackened_row.get(
                        "minimum_sufficient_budget"
                    ),
                    "no_slack_defer_at_minimum": primary_row.get(
                        "abstain_rate_at_minimum",
                        primary_row.get("defer_rate_at_minimum"),
                    ),
                    "slackened_defer_at_minimum": slackened_row.get(
                        "abstain_rate_at_minimum",
                        slackened_row.get("defer_rate_at_minimum"),
                    ),
                }
            )
    return pd.DataFrame(rows)


def normalize_aggregate(spec: dict[str, object]) -> pd.DataFrame:
    df = pd.read_csv(spec["aggregate"])
    df = df[df["policy_label"].eq(spec["main_policy"])].copy()
    if "conditional_false_accept_rate" in df.columns:
        false_accept = "conditional_false_accept_rate"
        false_reject = "conditional_false_reject_rate"
    else:
        false_accept = "conditional_false_promotion_rate"
        false_reject = "conditional_false_rejection_rate"
    if "defer_rate" in df.columns:
        defer_col = "defer_rate"
    else:
        defer_col = "abstain_rate"
    if "positive_full_decisions" in df.columns:
        pos_col = "positive_full_decisions"
        neg_col = "negative_full_decisions"
    else:
        pos_col = "positive_reference_decisions"
        neg_col = "negative_reference_decisions"
    return pd.DataFrame(
        {
            "source": spec["source"],
            "threshold": df["threshold"],
            "budget_cap": df["budget_cap"],
            "main_policy": df["policy_label"],
            "positive_decisions": df[pos_col],
            "negative_decisions": df[neg_col],
            "conditional_false_accept_rate": df[false_accept],
            "conditional_false_reject_rate": df[false_reject],
            "coverage_failure_rate": df["coverage_failure_rate"],
            "defer_rate": df[defer_col],
        }
    )


def build_target_sensitivity() -> pd.DataFrame:
    coverage_targets = [0.0, 0.05, 0.10]
    defer_targets = [0.10, 0.25, 0.40]
    rows: list[dict[str, object]] = []
    for spec in SOURCES:
        agg = normalize_aggregate(spec)
        for threshold, group in agg.groupby("threshold"):
            ordered = group.sort_values("budget_cap")
            for coverage_target in coverage_targets:
                for defer_target in defer_targets:
                    mask = (
                        (
                            ordered[
                                [
                                    "conditional_false_accept_rate",
                                    "conditional_false_reject_rate",
                                ]
                            ].max(axis=1)
                            <= RISK_TARGET
                        )
                        & (ordered["coverage_failure_rate"] <= coverage_target)
                        & (ordered["defer_rate"] <= defer_target)
                    )
                    row = ordered[mask].iloc[0] if mask.any() else None
                    rows.append(
                        {
                            "source": spec["source"],
                            "threshold": float(threshold),
                            "coverage_target": coverage_target,
                            "defer_target": defer_target,
                            "minimum_sufficient_budget": (
                                float(row["budget_cap"]) if row is not None else math.nan
                            ),
                            "observed_defer_rate": (
                                float(row["defer_rate"]) if row is not None else math.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    if weights.sum() == 0:
        return float("nan")
    return float((weights * values).sum() / weights.sum())


def normalize_summary(spec: dict[str, object]) -> pd.DataFrame:
    df = pd.read_csv(spec["summary"])
    policy_label = (
        df["policy_label"]
        if "policy_label" in df.columns
        else df["policy"].map(POLICY_LABELS).fillna(df["policy"])
    )
    left, right = spec["id_cols"]
    out = pd.DataFrame(
        {
            "source": spec["source"],
            "tested": df[left].astype(str),
            "comparison": df[right].astype(str),
            "threshold": df["threshold"].astype(float),
            "budget_cap": df["budget_cap"].astype(float),
            "policy_label": policy_label,
            "truth_accept": df[spec["truth_col"]].astype(bool),
            "full_delta": df["full_delta"].astype(float),
            "rows": df["rows"].astype(float),
            "false_accept_rate": df[spec["false_accept_col"]].astype(float),
            "false_reject_rate": df[spec["false_reject_col"]].astype(float),
            "decision_error_rate": df["decision_error_rate"].astype(float),
            "coverage_failure_rate": df["coverage_failure_rate"].astype(float),
            "defer_rate": df[spec["defer_col"]].astype(float),
        }
    )
    out["pair_key"] = out.apply(
        lambda row: "|||".join(sorted([row["tested"], row["comparison"]])),
        axis=1,
    )
    return out


def summarize_undirected_source(spec: dict[str, object]) -> pd.DataFrame:
    summary = normalize_summary(spec)
    keep_seed = summary[summary["policy_label"].eq(spec["main_policy"])].copy()
    keep_rows = (
        keep_seed.sort_values("full_delta", ascending=False)
        .groupby(["threshold", "pair_key"], as_index=False)
        .first()[["threshold", "pair_key", "tested", "comparison"]]
    )
    collapsed = summary.merge(
        keep_rows,
        on=["threshold", "pair_key", "tested", "comparison"],
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for keys, group in collapsed.groupby(["threshold", "budget_cap", "policy_label"]):
        threshold, budget_cap, policy_label = keys
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        rows.append(
            {
                "source": spec["source"],
                "threshold": threshold,
                "budget_cap": budget_cap,
                "policy_label": policy_label,
                "undirected_pairs": int(group["pair_key"].nunique()),
                "positive_decisions": int(pos["pair_key"].nunique()),
                "negative_decisions": int(neg["pair_key"].nunique()),
                "conditional_false_accept_rate": (
                    weighted_mean(neg, "false_accept_rate")
                    if not neg.empty
                    else float("nan")
                ),
                "conditional_false_reject_rate": (
                    weighted_mean(pos, "false_reject_rate")
                    if not pos.empty
                    else float("nan")
                ),
                "decision_error_rate": weighted_mean(group, "decision_error_rate"),
                "coverage_failure_rate": weighted_mean(group, "coverage_failure_rate"),
                "defer_rate": weighted_mean(group, "defer_rate"),
            }
        )
    return pd.DataFrame(rows)


def build_undirected_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in SOURCES:
        agg = summarize_undirected_source(spec)
        main = agg[agg["policy_label"].eq(spec["main_policy"])].copy()
        for threshold, group in main.groupby("threshold"):
            ordered = group.sort_values("budget_cap")
            mask = (
                (
                    ordered[
                        [
                            "conditional_false_accept_rate",
                            "conditional_false_reject_rate",
                        ]
                    ].max(axis=1)
                    <= RISK_TARGET
                )
                & (ordered["coverage_failure_rate"] <= COVERAGE_TARGET)
                & (ordered["defer_rate"] <= DEFER_TARGET)
            )
            row = ordered[mask].iloc[0] if mask.any() else None
            first = ordered.iloc[0]
            rows.append(
                {
                    "source": spec["source"],
                    "threshold": float(threshold),
                    "main_policy": spec["main_policy"],
                    "undirected_pairs": int(first["undirected_pairs"]),
                    "positive_decisions": int(first["positive_decisions"]),
                    "negative_decisions": int(first["negative_decisions"]),
                    "minimum_sufficient_budget": (
                        float(row["budget_cap"]) if row is not None else math.nan
                    ),
                    "observed_defer_rate": (
                        float(row["defer_rate"]) if row is not None else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if col.endswith("budget"):
            display[col] = display[col].map(budget_label)
        elif col.endswith("rate") or col.endswith("target") or col.endswith("minimum"):
            display[col] = display[col].map(pct)
    display = display.fillna("NA")
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    coverage: pd.DataFrame, target: pd.DataFrame, undirected: pd.DataFrame
) -> None:
    coverage_display = coverage.copy()
    coverage_display["threshold"] = coverage_display["threshold"].map(
        lambda x: f"{x * 100:.0f} pp"
    )
    for col in [
        "no_slack_floor_budget",
        "slackened_floor_budget",
    ]:
        coverage_display[col] = coverage_display[col].map(budget_label)
    for col in [
        "no_slack_defer_at_minimum",
        "slackened_defer_at_minimum",
    ]:
        coverage_display[col] = coverage_display[col].map(pct)

    baseline_target = target[
        target["coverage_target"].eq(COVERAGE_TARGET)
        & target["defer_target"].eq(DEFER_TARGET)
    ].copy()
    baseline_target["threshold"] = baseline_target["threshold"].map(
        lambda x: f"{x * 100:.0f} pp"
    )
    baseline_target["minimum_sufficient_budget"] = baseline_target[
        "minimum_sufficient_budget"
    ].map(budget_label)
    baseline_target["observed_defer_rate"] = baseline_target[
        "observed_defer_rate"
    ].map(pct)

    strict_target = target[
        target["coverage_target"].eq(0.0) & target["defer_target"].eq(0.10)
    ].copy()
    strict_target["threshold"] = strict_target["threshold"].map(
        lambda x: f"{x * 100:.0f} pp"
    )
    strict_target["minimum_sufficient_budget"] = strict_target[
        "minimum_sufficient_budget"
    ].map(budget_label)
    strict_target["observed_defer_rate"] = strict_target[
        "observed_defer_rate"
    ].map(pct)

    undirected_display = undirected.copy()
    undirected_display["threshold"] = undirected_display["threshold"].map(
        lambda x: f"{x * 100:.0f} pp"
    )
    undirected_display["minimum_sufficient_budget"] = undirected_display[
        "minimum_sufficient_budget"
    ].map(budget_label)
    undirected_display["observed_defer_rate"] = undirected_display[
        "observed_defer_rate"
    ].map(pct)

    lines = [
        "# Public Robustness Checks",
        "",
        "These checks use only the public replay summaries. They are empirical finite-task diagnostics, not formal stopping guarantees.",
        "",
        "## Coverage Rule Sensitivity",
        "",
        "The paper's primary rule uses a no-slack proportional floor. The sensitivity column restores the earlier one-task slack. This directly checks whether the reported minimum budget depends on that coverage convention.",
        "",
        markdown_table(
            coverage_display[
                [
                    "source",
                    "threshold",
                    "no_slack_floor_budget",
                    "slackened_floor_budget",
                    "no_slack_defer_at_minimum",
                    "slackened_defer_at_minimum",
                ]
            ]
        ),
        "",
        "## Target Sensitivity",
        "",
        "This re-thresholds observed public aggregate rates. It changes reporting targets, not the bootstrap decisions produced during replay.",
        "",
        "Baseline targets:",
        "",
        markdown_table(
            baseline_target[
                [
                    "source",
                    "threshold",
                    "minimum_sufficient_budget",
                    "observed_defer_rate",
                ]
            ]
        ),
        "",
        "Strict targets, zero coverage failure and 10 percent defer target:",
        "",
        markdown_table(
            strict_target[
                [
                    "source",
                    "threshold",
                    "minimum_sufficient_budget",
                    "observed_defer_rate",
                ]
            ]
        ),
        "",
        "## Undirected Pair Collapse",
        "",
        "Each unordered system pair is kept once, oriented toward the larger all-task delta. This removes mirror comparisons and checks whether the main result relies on duplicated directed pairs.",
        "",
        markdown_table(
            undirected_display[
                [
                    "source",
                    "threshold",
                    "undirected_pairs",
                    "positive_decisions",
                    "negative_decisions",
                    "minimum_sufficient_budget",
                    "observed_defer_rate",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "The coverage-rule check is the most important reviewer-facing result. At 0 pp, SWE-bench Lite is not sufficient under the no-slack floor but reaches 75 percent under the slackened floor, while AppWorld and tau-bench reach 25 percent and SWE-bench Verified remains no-fit. At the positive-margin checks, SWE-bench Lite changes from 75 percent under no-slack coverage to 50 percent under the slackened floor. This means the exact budget should be reported with its threshold and coverage rule. It strengthens the paper's central point that task fraction alone is not a decision rule.",
        "",
        "The undirected collapse does not remove the main pattern, but it changes support counts and should be described as a dependence check rather than a new primary estimate.",
        "",
    ]
    (OUT_DIR / "public_robustness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = build_coverage_sensitivity()
    target = build_target_sensitivity()
    undirected = build_undirected_sensitivity()
    coverage.to_csv(OUT_DIR / "coverage_rule_sensitivity.csv", index=False)
    target.to_csv(OUT_DIR / "target_sensitivity.csv", index=False)
    undirected.to_csv(OUT_DIR / "undirected_pair_collapse.csv", index=False)
    write_report(coverage, target, undirected)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
