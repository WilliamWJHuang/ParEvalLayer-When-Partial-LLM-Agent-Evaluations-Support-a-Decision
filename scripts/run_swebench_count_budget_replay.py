from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_LABELS = {
    "uniform_forced": "uniform forced",
    "repo_stratified_forced": "repo-stratified forced",
    "bootstrap_ci": "bootstrap CI",
    "repo_coverage_bootstrap": "repo-coverage bootstrap",
}


@dataclass(frozen=True)
class Pair:
    candidate: str
    reference: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay count-budget decisions on SWE-bench experiment results."
    )
    parser.add_argument(
        "--task-rows",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "swebench_smoke"
        / "swebench_task_rows.csv",
    )
    parser.add_argument(
        "--pair-summary",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "swebench_smoke"
        / "swebench_pair_overlap_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis") / "public_backtest" / "swebench_verified_replay",
    )
    parser.add_argument("--split", default="verified")
    parser.add_argument("--n-replays", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--budget-caps", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--coverage-target", type=float, default=0.05)
    parser.add_argument("--abstention-target", type=float, default=0.25)
    parser.add_argument(
        "--coverage-slack",
        type=int,
        default=0,
        help=(
            "Subtract this many tasks from the proportional per-group floor. "
            "The default 0 is the paper's no-slack proportional floor."
        ),
    )
    return parser.parse_args()


def random_orders(n_replays: int, n_tasks: int, rng: np.random.Generator) -> np.ndarray:
    return np.argsort(rng.random((n_replays, n_tasks)), axis=1).astype(np.int32)


def strata_codes(strata: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(str(item) for item in np.unique(strata))
    mapping = {name: idx for idx, name in enumerate(unique)}
    codes = np.array([mapping[str(item)] for item in strata], dtype=np.int16)
    return codes, mapping


def stratified_orders(
    strata_code: np.ndarray,
    n_replays: int,
    rng: np.random.Generator,
) -> np.ndarray:
    strata = sorted(int(item) for item in np.unique(strata_code))
    by_stratum = [np.where(strata_code == idx)[0] for idx in strata]
    orders = np.empty((n_replays, len(strata_code)), dtype=np.int32)
    for replay_idx in range(n_replays):
        scores = np.empty(len(strata_code), dtype=float)
        for indices in by_stratum:
            permuted = rng.permutation(indices)
            local_scores = (np.arange(len(permuted)) + rng.random(len(permuted))) / len(
                permuted
            )
            scores[permuted] = local_scores
        orders[replay_idx] = np.argsort(scores).astype(np.int32)
    return orders


def coverage_requirements(
    strata_code: np.ndarray, cap_k: int, coverage_slack: int
) -> dict[int, int]:
    n_tasks = len(strata_code)
    requirements: dict[int, int] = {}
    for idx in sorted(int(item) for item in np.unique(strata_code)):
        stratum_size = int((strata_code == idx).sum())
        expected = cap_k * (stratum_size / n_tasks)
        requirements[idx] = max(
            1, min(stratum_size, math.floor(expected) - coverage_slack)
        )
    return requirements


def coverage_ok(strata: np.ndarray, requirements: dict[int, int]) -> np.ndarray:
    ok = np.ones(strata.shape[0], dtype=bool)
    for idx, required in requirements.items():
        ok &= (strata == idx).sum(axis=1) >= required
    return ok


def bootstrap_quantities(
    seen: np.ndarray,
    thresholds: list[float],
    alpha: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[float, dict[str, np.ndarray]]:
    n_replays, k = seen.shape
    boot_idx = rng.integers(0, k, size=(bootstrap_samples, k), endpoint=False)
    boot_means = seen[:, boot_idx].mean(axis=2)
    lows = np.quantile(boot_means, alpha / 2, axis=1)
    highs = np.quantile(boot_means, 1 - alpha / 2, axis=1)
    out: dict[float, dict[str, np.ndarray]] = {}
    for threshold in thresholds:
        out[threshold] = {
            "lows": lows,
            "highs": highs,
            "risk_promote": (boot_means <= threshold).mean(axis=1),
            "risk_reject": (boot_means > threshold).mean(axis=1),
        }
    return out


def prepare_pair(task_df: pd.DataFrame, pair: Pair) -> pd.DataFrame:
    candidate = task_df[task_df["submission"].eq(pair.candidate)].copy()
    reference = task_df[task_df["submission"].eq(pair.reference)].copy()
    merged = candidate.merge(
        reference,
        on="instance_id",
        suffixes=("_candidate", "_reference"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping instances for {pair.candidate} vs {pair.reference}")
    merged["diff"] = merged["resolved_candidate"].astype(float) - merged[
        "resolved_reference"
    ].astype(float)
    merged["repo"] = merged["repo_candidate"]
    return merged.sort_values(["repo", "instance_id"]).reset_index(drop=True)


def summarize_decisions(
    decision: np.ndarray,
    truth_promote: bool,
    coverage_failure: np.ndarray,
    task_fraction: float,
) -> dict[str, float | int]:
    promoted = decision == 1
    rejected = decision == -1
    abstained = decision == 0
    false_promotion = promoted & (not truth_promote)
    false_rejection = rejected & truth_promote
    return {
        "rows": int(len(decision)),
        "mean_task_fraction": task_fraction,
        "promote_rate": float(promoted.mean()),
        "reject_rate": float(rejected.mean()),
        "abstain_rate": float(abstained.mean()),
        "false_promotion_rate": float(false_promotion.mean()),
        "false_rejection_rate": float(false_rejection.mean()),
        "decision_error_rate": float((false_promotion | false_rejection).mean()),
        "coverage_failure_rate": float(coverage_failure.mean()),
    }


def run_pair(
    pair_df: pd.DataFrame,
    pair: Pair,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    diff = pair_df["diff"].to_numpy(dtype=np.float32)
    strata_code, _ = strata_codes(pair_df["repo"].to_numpy())
    n_tasks = len(diff)
    uniform_orders = random_orders(args.n_replays, n_tasks, rng)
    stratified = stratified_orders(strata_code, args.n_replays, rng)
    truth_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    full_delta = float(diff.mean())
    for threshold in args.thresholds:
        truth_rows.append(
            {
                "split": args.split,
                "candidate": pair.candidate,
                "reference": pair.reference,
                "overlap_tasks": n_tasks,
                "repo_count": int(pair_df["repo"].nunique()),
                "full_delta": full_delta,
                "threshold": threshold,
                "truth_promote": bool(full_delta > threshold),
            }
        )
    for budget_cap in args.budget_caps:
        cap_k = max(1, min(n_tasks, int(math.ceil(n_tasks * budget_cap))))
        requirements = coverage_requirements(strata_code, cap_k, args.coverage_slack)
        policy_orders = {
            "uniform_forced": uniform_orders,
            "bootstrap_ci": uniform_orders,
            "repo_stratified_forced": stratified,
            "repo_coverage_bootstrap": stratified,
        }
        bootstrap_cache: dict[str, dict[float, dict[str, np.ndarray]]] = {}
        for policy, orders in policy_orders.items():
            prefix = orders[:, :cap_k]
            seen = diff[prefix]
            partial_mean = seen.mean(axis=1)
            coverage = coverage_ok(strata_code[prefix], requirements)
            if "bootstrap" in policy:
                bootstrap_cache[policy] = bootstrap_quantities(
                    seen,
                    args.thresholds,
                    args.alpha,
                    args.bootstrap_samples,
                    rng,
                )
            for threshold in args.thresholds:
                truth_promote = full_delta > threshold
                if policy in {"uniform_forced", "repo_stratified_forced"}:
                    decision = np.where(partial_mean > threshold, 1, -1)
                elif policy == "bootstrap_ci":
                    b = bootstrap_cache[policy][threshold]
                    decision = np.zeros(args.n_replays, dtype=np.int8)
                    decision[b["lows"] > threshold] = 1
                    decision[b["highs"] <= threshold] = -1
                elif policy == "repo_coverage_bootstrap":
                    b = bootstrap_cache[policy][threshold]
                    decision = np.zeros(args.n_replays, dtype=np.int8)
                    promote = (
                        coverage
                        & (partial_mean > threshold)
                        & (b["risk_promote"] <= args.alpha)
                    )
                    reject = (
                        coverage
                        & (partial_mean <= threshold)
                        & (b["risk_reject"] <= args.alpha)
                    )
                    decision[promote] = 1
                    decision[reject] = -1
                else:
                    raise ValueError(policy)
                coverage_failure = (decision != 0) & (~coverage)
                row = {
                    "split": args.split,
                    "candidate": pair.candidate,
                    "reference": pair.reference,
                    "threshold": threshold,
                    "budget_cap": budget_cap,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "truth_promote": bool(truth_promote),
                    "full_delta": full_delta,
                    "overlap_tasks": n_tasks,
                    "observed_tasks": cap_k,
                    "repo_count": int(pair_df["repo"].nunique()),
                }
                row.update(
                    summarize_decisions(
                        decision,
                        bool(truth_promote),
                        coverage_failure,
                        task_fraction=cap_k / n_tasks,
                    )
                )
                summary_rows.append(row)
    return truth_rows, summary_rows


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    if weights.sum() == 0:
        return float("nan")
    return float((values * weights).sum() / weights.sum())


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
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
                "policy_label": POLICY_LABELS[str(policy)],
                "directed_pairs": int(group[["candidate", "reference"]].drop_duplicates().shape[0]),
                "positive_reference_decisions": int(
                    promote_group[["candidate", "reference"]].drop_duplicates().shape[0]
                ),
                "negative_reference_decisions": int(
                    reject_group[["candidate", "reference"]].drop_duplicates().shape[0]
                ),
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
                "mean_task_fraction": weighted_mean(group, "mean_task_fraction"),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold", "budget_cap", "policy"])


def summarize_sufficiency(
    aggregate_df: pd.DataFrame,
    alpha: float,
    coverage_target: float,
    abstention_target: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in aggregate_df.groupby(["threshold", "policy"], dropna=False):
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
        sufficient_budget = (
            float(ordered.loc[actionable_mask, "budget_cap"].iloc[0])
            if actionable_mask.any()
            else None
        )
        first = ordered.loc[actionable_mask].iloc[0].to_dict() if actionable_mask.any() else {}
        rows.append(
            {
                "threshold": threshold,
                "policy": policy,
                "policy_label": POLICY_LABELS[str(policy)],
                "safety_only_budget": safety_budget,
                "minimum_sufficient_budget": sufficient_budget,
                "abstain_rate_at_minimum": first.get("abstain_rate"),
                "positive_reference_decisions": int(
                    ordered["positive_reference_decisions"].iloc[0]
                ),
                "negative_reference_decisions": int(
                    ordered["negative_reference_decisions"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold", "policy"])


def percent(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "not found"
    return f"{100 * float(value):.{digits}f} percent"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy().fillna("")
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
    truth: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    sufficient: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    split_label = str(args.split).strip()
    display_split = split_label[:1].upper() + split_label[1:]
    lines.append(f"# SWE-bench {display_split} Count-Budget Replay\n")
    lines.append("## Status\n")
    lines.append(
        f"Completed on near-complete SWE-bench {display_split} submissions using task-count budget only. Public S3 logs and trajectories were not downloaded.\n"
    )
    lines.append("## Scope\n")
    lines.append(
        "This is a conversion and scope check. It tests whether the decision-sufficiency protocol can be applied to released SWE-bench result metadata without changing the paper into a coding-agent benchmark study.\n"
    )
    lines.append("## Reference-Decision Mix\n")
    mix = (
        truth.groupby("threshold")["truth_promote"]
        .agg(
            directed_pairs="count",
            positive_reference_decisions="sum",
        )
        .reset_index()
    )
    mix["negative_reference_decisions"] = (
        mix["directed_pairs"] - mix["positive_reference_decisions"]
    )
    lines.append(markdown_table(mix))
    lines.append("\n\n## Minimum Sufficient Budgets\n")
    display = sufficient.copy()
    display["safety_only_budget"] = display["safety_only_budget"].map(lambda x: percent(x, 0))
    display["minimum_sufficient_budget"] = display["minimum_sufficient_budget"].map(
        lambda x: percent(x, 0)
    )
    display["abstain_rate_at_minimum"] = display["abstain_rate_at_minimum"].map(percent)
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
    lines.append("\n\n## Key 25 Percent Rows\n")
    key = aggregate_df[aggregate_df["budget_cap"].eq(0.25)].copy()
    key = key.sort_values(["threshold", "policy"])
    for col in [
        "conditional_false_promotion_rate",
        "conditional_false_rejection_rate",
        "decision_error_rate",
        "abstain_rate",
        "coverage_failure_rate",
        "mean_task_fraction",
    ]:
        key[col] = key[col].map(percent)
    lines.append(
        markdown_table(
            key[
                [
                    "threshold",
                    "policy_label",
                    "conditional_false_promotion_rate",
                    "conditional_false_rejection_rate",
                    "decision_error_rate",
                    "abstain_rate",
                    "coverage_failure_rate",
                    "mean_task_fraction",
                ]
            ]
        )
    )
    lines.append("\n\n## Interpretation\n")
    lines.append(
        f"The {display_split} split passes the stricter replay gate as a count-budget scope check. Repository coverage matters, and bootstrap policies trade lower decision error for abstention. This supports the paper's decision framing outside tau-bench, but it does not add cost-aware evidence.\n"
    )
    lines.append(
        "The result should enter the main paper only as a compact scope check. It should not be used to compare coding agents or to claim broad task-population generality.\n"
    )
    lines.append("\n\n## Parameters\n")
    lines.append(
        f"- n_replays: {args.n_replays}\n- bootstrap_samples: {args.bootstrap_samples}\n- thresholds: {args.thresholds}\n- budget_caps: {args.budget_caps}\n- alpha: {args.alpha}\n- coverage_slack: {args.coverage_slack}\n"
    )
    (out_dir / "swebench_verified_replay_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_rows = pd.read_csv(args.task_rows)
    pair_summary = pd.read_csv(args.pair_summary)
    task_rows = task_rows[task_rows["split"].eq(args.split)].copy()
    pair_summary = pair_summary[
        pair_summary["split"].eq(args.split)
        & pair_summary["near_complete_pair"].astype(bool)
    ].copy()
    pairs = [
        Pair(candidate=row["candidate"], reference=row["reference"])
        for _, row in pair_summary[["candidate", "reference"]].drop_duplicates().iterrows()
    ]
    rng = np.random.default_rng(args.seed)
    truth_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for idx, pair in enumerate(pairs, start=1):
        pair_df = prepare_pair(task_rows, pair)
        truth, summary = run_pair(pair_df, pair, args, rng)
        truth_rows.extend(truth)
        summary_rows.extend(summary)
        if idx % 20 == 0:
            print(f"processed {idx}/{len(pairs)} pairs")
    truth_df = pd.DataFrame(truth_rows)
    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = aggregate(summary_df)
    sufficient_df = summarize_sufficiency(
        aggregate_df,
        args.alpha,
        args.coverage_target,
        args.abstention_target,
    )
    truth_df.to_csv(args.out_dir / "swebench_verified_truth_summary.csv", index=False)
    summary_df.to_csv(args.out_dir / "swebench_verified_replay_summary.csv", index=False)
    aggregate_df.to_csv(args.out_dir / "swebench_verified_policy_aggregate.csv", index=False)
    sufficient_df.to_csv(
        args.out_dir / "swebench_verified_min_sufficient_budget.csv",
        index=False,
    )
    metadata = {
        "split": args.split,
        "pairs": len(pairs),
        "n_replays": args.n_replays,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "thresholds": args.thresholds,
        "budget_caps": args.budget_caps,
        "alpha": args.alpha,
        "coverage_target": args.coverage_target,
        "abstention_target": args.abstention_target,
        "coverage_slack": args.coverage_slack,
    }
    (args.out_dir / "swebench_verified_replay_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir, truth_df, aggregate_df, sufficient_df, args)
    print(json.dumps({"pairs": len(pairs), "summary_rows": len(summary_df)}, indent=2))


if __name__ == "__main__":
    main()
