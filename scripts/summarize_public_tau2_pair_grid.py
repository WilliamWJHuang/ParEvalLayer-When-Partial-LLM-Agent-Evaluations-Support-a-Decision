from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


POLICY_LABELS = {
    "fixed_uniform_forced": "uniform forced",
    "fixed_stratified_forced": "stratified forced",
    "cost_aware_forced": "cost-aware forced",
    "bootstrap_ci": "bootstrap CI",
    "coverage_bootstrap": "coverage-aware bootstrap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the public tau-bench directed pair-grid replay."
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("analysis") / "public_backtest" / "tau2_pair_grid_failure",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "tau2_pair_grid_summary",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--coverage-target", type=float, default=0.05)
    parser.add_argument("--abstention-target", type=float, default=0.25)
    return parser.parse_args()


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    if weights.sum() == 0:
        return float("nan")
    return float((values * weights).sum() / weights.sum())


def aggregate_by_policy(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in summary.groupby(["threshold", "budget_cap", "policy"], dropna=False):
        threshold, budget_cap, policy = keys
        promote_group = group[group["truth_promote"]]
        reject_group = group[~group["truth_promote"]]
        rows.append(
            {
                "threshold": threshold,
                "budget_cap": budget_cap,
                "policy": policy,
                "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                "comparison_count": int(group["comparison_label"].nunique()),
                "positive_reference_decisions": int(promote_group["comparison_label"].nunique()),
                "negative_reference_decisions": int(reject_group["comparison_label"].nunique()),
                "conditional_false_promotion_rate": (
                    weighted_mean(reject_group, "false_promotion_rate")
                    if not reject_group.empty
                    else float("nan")
                ),
                "conditional_false_rejection_rate": (
                    weighted_mean(promote_group, "false_rejection_rate")
                    if not promote_group.empty
                    else float("nan")
                ),
                "decision_error_rate": weighted_mean(group, "decision_error_rate"),
                "abstain_rate": weighted_mean(group, "abstain_rate"),
                "coverage_failure_rate": weighted_mean(group, "coverage_failure_rate"),
                "candidate_cost_fraction": weighted_mean(
                    group, "mean_candidate_cost_fraction"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold", "budget_cap", "policy"])


def summarize_sufficiency(
    aggregate: pd.DataFrame,
    alpha: float,
    coverage_target: float,
    abstention_target: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in aggregate.groupby(["threshold", "policy"], dropna=False):
        threshold, policy = keys
        ordered = group.sort_values("budget_cap")
        safety_mask = (
            (ordered["conditional_false_promotion_rate"] <= alpha)
            & (ordered["conditional_false_rejection_rate"] <= alpha)
            & (ordered["coverage_failure_rate"] <= coverage_target)
        )
        actionable_mask = safety_mask & (ordered["abstain_rate"] <= abstention_target)
        safety_budget = (
            float(ordered.loc[safety_mask, "budget_cap"].iloc[0])
            if safety_mask.any()
            else None
        )
        actionable_budget = (
            float(ordered.loc[actionable_mask, "budget_cap"].iloc[0])
            if actionable_mask.any()
            else None
        )
        first_actionable = (
            ordered.loc[actionable_mask].iloc[0].to_dict()
            if actionable_mask.any()
            else {}
        )
        rows.append(
            {
                "threshold": threshold,
                "policy": policy,
                "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                "safety_only_budget": safety_budget,
                "minimum_sufficient_budget": actionable_budget,
                "abstain_rate_at_minimum": first_actionable.get("abstain_rate"),
                "positive_reference_decisions": int(
                    ordered["positive_reference_decisions"].iloc[0]
                ),
                "negative_reference_decisions": int(
                    ordered["negative_reference_decisions"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold", "policy"])


def margin_bin(delta: float, threshold: float) -> str:
    margin = delta - threshold
    if margin >= 0.05:
        return "clear_positive"
    if margin > 0:
        return "near_positive"
    if margin >= -0.05:
        return "near_negative"
    return "clear_negative"


def build_truth_table(truth: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in truth.iterrows():
        for threshold in thresholds:
            rows.append(
                {
                    "comparison_label": row["comparison_label"],
                    "candidate_model": row["candidate_model"],
                    "candidate_reasoning_effort": row["candidate_reasoning_effort"],
                    "reference_model": row["reference_model"],
                    "reference_reasoning_effort": row["reference_reasoning_effort"],
                    "full_delta": row["full_delta"],
                    "threshold": threshold,
                    "truth_promote": bool(row["full_delta"] > threshold),
                    "margin_bin": margin_bin(float(row["full_delta"]), threshold),
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold", "full_delta"], ascending=[True, False])


def percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "not found"
    return f"{100 * float(value):.0f} percent"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display = display.fillna("")
    columns = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max([len(columns[idx])] + [len(row[idx]) for row in rows])
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(
        columns[idx].ljust(widths[idx]) for idx in range(len(columns))
    ) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_report(
    out_dir: Path,
    truth_table: pd.DataFrame,
    aggregate: pd.DataFrame,
    sufficient: pd.DataFrame,
    key_rows: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("# Public Tau-Bench Directed Pair-Grid Report\n")
    lines.append("## Status\n")
    lines.append(
        "Completed on the official public tau-bench trajectories already mirrored in the workspace.\n"
    )
    lines.append("## Scope\n")
    lines.append(
        "This is an external decision-sufficiency check, not a second primary benchmark study and not a model leaderboard.\n"
    )
    lines.append("## Data\n")
    lines.append(f"- Directed candidate-reference comparisons: {truth_table['comparison_label'].nunique()}\n")
    lines.append("- Domains: airline, retail, telecom\n")
    lines.append("- Unit: domain by task identifier, with four public trials averaged per task\n")
    lines.append("- Reward handling: infrastructure errors coded as failure in the primary grid\n")
    lines.append("- Banking knowledge excluded because retrieval configuration differs across submissions\n")
    lines.append("## Reference-Decision Mix\n")
    mix = (
        truth_table.groupby(["threshold", "margin_bin"])["comparison_label"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    lines.append(markdown_table(mix))
    lines.append("\n\n## Minimum Sufficient Budgets Across The Directed Grid\n")
    display = sufficient.copy()
    display["safety_only_budget"] = display["safety_only_budget"].map(percent)
    display["minimum_sufficient_budget"] = display["minimum_sufficient_budget"].map(percent)
    display["abstain_rate_at_minimum"] = display["abstain_rate_at_minimum"].map(
        lambda x: "not found" if pd.isna(x) else f"{100 * float(x):.2f} percent"
    )
    lines.append(
        markdown_table(
            display[
                [
                    "threshold",
                    "policy_label",
                    "safety_only_budget",
                    "minimum_sufficient_budget",
                    "abstain_rate_at_minimum",
                    "positive_reference_decisions",
                    "negative_reference_decisions",
                ]
            ]
        )
    )
    lines.append("\n\n## Key 25 Percent Budget Rows\n")
    key_display = key_rows.copy()
    for col in [
        "conditional_false_promotion_rate",
        "conditional_false_rejection_rate",
        "decision_error_rate",
        "abstain_rate",
        "coverage_failure_rate",
        "candidate_cost_fraction",
    ]:
        key_display[col] = key_display[col].map(lambda x: f"{100 * float(x):.2f} percent")
    lines.append(
        markdown_table(
            key_display[
                [
                    "threshold",
                    "policy_label",
                    "conditional_false_promotion_rate",
                    "conditional_false_rejection_rate",
                    "decision_error_rate",
                    "abstain_rate",
                    "coverage_failure_rate",
                    "candidate_cost_fraction",
                ]
            ]
        )
    )
    lines.append("\n\n## Interpretation\n")
    lines.append(
        "Directed comparisons are summarized by threshold, ordering policy, coverage, and unresolved rate.\n"
    )
    lines.append(
        "The result should still be scoped. The grid covers one public benchmark family and three text domains. It broadens the public check from hand-picked comparisons to all directed pairs, but it does not prove population generality.\n"
    )
    lines.append("## Artifacts\n")
    for key in ["summary_table", "truth_table"]:
        if key in metadata:
            lines.append(f"- {key}: `{metadata[key]}`\n")
    lines.append(f"- aggregate table: `{out_dir / 'pair_grid_policy_aggregate.csv'}`\n")
    lines.append(f"- sufficient-budget table: `{out_dir / 'pair_grid_min_sufficient_budget.csv'}`\n")
    (out_dir / "public_tau2_pair_grid_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.grid_dir / "public_tau2_replay_summary.csv")
    truth = pd.read_csv(args.grid_dir / "public_tau2_truth_summary.csv")
    metadata_path = args.grid_dir / "public_tau2_replay_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    thresholds = [
        float(item)
        for item in metadata.get("thresholds", sorted(summary["threshold"].unique()))
    ]
    truth_table = build_truth_table(truth, thresholds)
    aggregate = aggregate_by_policy(summary)
    sufficient = summarize_sufficiency(
        aggregate,
        args.alpha,
        args.coverage_target,
        args.abstention_target,
    )
    key_rows = aggregate[aggregate["budget_cap"].eq(0.25)].copy()

    truth_table.to_csv(args.out_dir / "pair_grid_truth_table.csv", index=False)
    aggregate.to_csv(args.out_dir / "pair_grid_policy_aggregate.csv", index=False)
    sufficient.to_csv(
        args.out_dir / "pair_grid_min_sufficient_budget.csv", index=False
    )
    key_rows.to_csv(args.out_dir / "pair_grid_key_25pct_rows.csv", index=False)
    write_report(args.out_dir, truth_table, aggregate, sufficient, key_rows, metadata)

    print(
        json.dumps(
            {
                "truth_rows": int(len(truth_table)),
                "aggregate_rows": int(len(aggregate)),
                "sufficient_rows": int(len(sufficient)),
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
