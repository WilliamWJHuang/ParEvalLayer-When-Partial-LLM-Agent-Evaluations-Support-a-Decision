from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "public_backtest"
TASK_ROWS = ANALYSIS / "terminal_bench" / "terminal_bench_task_rows.csv"
OUT_DIR = ANALYSIS / "terminal_bench_resource_replay"


@dataclass(frozen=True)
class PairSpec:
    tested: str
    comparison: str
    pair_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench task, token, and time budget replays."
    )
    parser.add_argument("--task-rows", type=Path, default=TASK_ROWS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.00, 0.05, 0.10]
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--budget-caps",
        type=float,
        nargs="+",
        default=[round(x / 100, 2) for x in range(5, 100, 5)],
    )
    parser.add_argument("--n-replays", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--min-overlap-tasks", type=int, default=80)
    parser.add_argument(
        "--budget-metrics",
        nargs="+",
        default=["task_fraction", "tested_tokens", "pair_tokens", "tested_time", "pair_time"],
        choices=["task_fraction", "tested_tokens", "pair_tokens", "tested_time", "pair_time"],
    )
    parser.add_argument(
        "--coverage-mode",
        choices=["inferred_prefix", "none"],
        default="none",
        help=(
            "Terminal-Bench public result JSON does not expose native task domains. "
            "Use none for the main diagnostic and inferred_prefix only as a stress check."
        ),
    )
    parser.add_argument(
        "--min-resource-completeness",
        type=float,
        default=0.95,
        help="Skip a resource metric for a pair if fewer rows have non-missing resources.",
    )
    return parser.parse_args()


def group_codes(groups: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(str(item) for item in np.unique(groups))
    mapping = {name: idx for idx, name in enumerate(unique)}
    return np.array([mapping[str(item)] for item in groups], dtype=np.int16), mapping


def coverage_requirements(group_code: np.ndarray, cap_k: int) -> dict[int, int]:
    n_tasks = len(group_code)
    out: dict[int, int] = {}
    for idx in sorted(int(item) for item in np.unique(group_code)):
        group_size = int((group_code == idx).sum())
        expected = cap_k * (group_size / n_tasks)
        out[idx] = max(1, min(group_size, math.floor(expected)))
    return out


def coverage_ok(groups: np.ndarray, requirements: dict[int, int]) -> bool:
    for idx, required in requirements.items():
        if int((groups == idx).sum()) < required:
            return False
    return True


def random_stratified_orders(
    group_code: np.ndarray, n_replays: int, rng: np.random.Generator
) -> np.ndarray:
    unique = sorted(int(item) for item in np.unique(group_code))
    by_group = [np.where(group_code == idx)[0] for idx in unique]
    orders = np.empty((n_replays, len(group_code)), dtype=np.int32)
    for replay_idx in range(n_replays):
        scores = np.empty(len(group_code), dtype=float)
        for indices in by_group:
            permuted = rng.permutation(indices)
            local = (np.arange(len(permuted)) + rng.random(len(permuted))) / len(
                permuted
            )
            scores[permuted] = local
        orders[replay_idx] = np.argsort(scores).astype(np.int32)
    return orders


def resource_vector(pair_df: pd.DataFrame, metric: str) -> np.ndarray:
    if metric == "task_fraction":
        return np.ones(len(pair_df), dtype=float)
    if metric == "tested_tokens":
        return pair_df["mean_total_tokens_tested"].to_numpy(dtype=float)
    if metric == "pair_tokens":
        return (
            pair_df["mean_total_tokens_tested"].to_numpy(dtype=float)
            + pair_df["mean_total_tokens_comparison"].to_numpy(dtype=float)
        )
    if metric == "tested_time":
        return pair_df["mean_agent_execution_seconds_tested"].to_numpy(dtype=float)
    if metric == "pair_time":
        return (
            pair_df["mean_agent_execution_seconds_tested"].to_numpy(dtype=float)
            + pair_df["mean_agent_execution_seconds_comparison"].to_numpy(dtype=float)
        )
    raise ValueError(metric)


def clean_resource_vector(
    resources: np.ndarray,
    min_completeness: float,
) -> tuple[np.ndarray | None, float]:
    finite = np.isfinite(resources)
    completeness = float(finite.mean()) if len(resources) else 0.0
    if completeness < min_completeness:
        return None, completeness
    if not finite.any():
        return None, completeness
    median = float(np.nanmedian(resources))
    return np.nan_to_num(resources, nan=median), completeness


def prefix_for_budget(
    order: np.ndarray,
    resources: np.ndarray,
    budget_cap: float,
    metric: str,
) -> np.ndarray:
    if metric == "task_fraction":
        cap_k = max(1, min(len(order), int(math.ceil(len(order) * budget_cap))))
        return order[:cap_k]
    clean = resources
    total = float(clean.sum())
    if total <= 0:
        return order[:1]
    limit = budget_cap * total
    cumulative = np.cumsum(clean[order])
    count = int(np.searchsorted(cumulative, limit, side="right"))
    count = max(1, min(len(order), count))
    return order[:count]


def bootstrap_decision(
    values: np.ndarray,
    threshold: float,
    alpha: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[int, float, float, float]:
    mean = float(values.mean())
    sample = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(
        axis=1
    )
    risk_promote = float((sample <= threshold).mean())
    risk_reject = float((sample > threshold).mean())
    decision = 0
    if mean > threshold and risk_promote <= alpha:
        decision = 1
    elif mean <= threshold and risk_reject <= alpha:
        decision = -1
    return decision, mean, risk_promote, risk_reject


def prepare_pairs(task_rows: pd.DataFrame) -> list[PairSpec]:
    systems = sorted(str(item) for item in task_rows["system_id"].unique())
    pairs: list[PairSpec] = []
    for tested in systems:
        for comparison in systems:
            if tested == comparison:
                continue
            pairs.append(
                PairSpec(
                    tested=tested,
                    comparison=comparison,
                    pair_label=f"{tested}__vs__{comparison}",
                )
            )
    return pairs


def prepare_pair(task_rows: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    keep = [
        "task_name",
        "coverage_group",
        "outcome",
        "mean_total_tokens",
        "mean_agent_execution_seconds",
        "trials_observed",
    ]
    tested = task_rows[task_rows["system_id"].eq(pair.tested)][keep].copy()
    comparison = task_rows[task_rows["system_id"].eq(pair.comparison)][keep].copy()
    merged = tested.merge(
        comparison,
        on=["task_name"],
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    merged["coverage_group"] = merged["coverage_group_tested"].astype(str)
    merged["diff"] = merged["outcome_tested"].astype(float) - merged[
        "outcome_comparison"
    ].astype(float)
    return merged.sort_values(["coverage_group", "task_name"]).reset_index(drop=True)


def summarize_pair(
    pair: PairSpec,
    pair_df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    diffs = pair_df["diff"].to_numpy(dtype=np.float32)
    group_code, mapping = group_codes(pair_df["coverage_group"].to_numpy())
    policy_scope = "inferred-prefix"
    if args.coverage_mode == "none":
        group_code, mapping = group_codes(np.array(["all"] * len(pair_df)))
        policy_scope = "no-strata"
    orders = random_stratified_orders(group_code, args.n_replays, rng)
    full_delta = float(diffs.mean())
    rows: list[dict[str, object]] = []
    for metric in args.budget_metrics:
        raw_resources = resource_vector(pair_df, metric)
        resources, resource_completeness = clean_resource_vector(
            raw_resources, args.min_resource_completeness
        )
        if resources is None:
            continue
        total_resource = float(resources.sum())
        for budget_cap in args.budget_caps:
            decisions_by_threshold: dict[float, list[int]] = {
                threshold: [] for threshold in args.thresholds
            }
            coverage_by_replay: list[bool] = []
            observed_tasks: list[int] = []
            observed_resource_fraction: list[float] = []
            for order in orders:
                prefix = prefix_for_budget(order, resources, budget_cap, metric)
                cap_k = len(prefix)
                observed_tasks.append(cap_k)
                observed_resource_fraction.append(
                    float(np.nan_to_num(resources[prefix], nan=0.0).sum() / total_resource)
                    if total_resource > 0
                    else 0.0
                )
                coverage = coverage_ok(
                    group_code[prefix], coverage_requirements(group_code, cap_k)
                )
                coverage_by_replay.append(coverage)
                for threshold in args.thresholds:
                    if coverage:
                        decision, _, _, _ = bootstrap_decision(
                            diffs[prefix],
                            threshold,
                            args.alpha,
                            args.bootstrap_samples,
                            rng,
                        )
                    else:
                        decision = 0
                    decisions_by_threshold[threshold].append(decision)
            coverage_arr = np.array(coverage_by_replay, dtype=bool)
            for threshold in args.thresholds:
                truth_accept = full_delta > threshold
                decision = np.array(decisions_by_threshold[threshold], dtype=np.int8)
                accept = decision == 1
                reject = decision == -1
                defer = decision == 0
                false_accept = accept & (not truth_accept)
                false_reject = reject & truth_accept
                rows.append(
                    {
                        "source": "Terminal-Bench 2.0",
                        "tested_system": pair.tested,
                        "comparison_system": pair.comparison,
                        "pair_label": pair.pair_label,
                        "threshold": threshold,
                        "policy_alpha": args.alpha,
                        "budget_cap": budget_cap,
                        "budget_metric": metric,
                        "policy_label": f"{metric} {policy_scope} bootstrap",
                        "truth_accept": bool(truth_accept),
                        "full_delta": full_delta,
                        "overlap_tasks": int(len(diffs)),
                        "observed_tasks_mean": float(np.mean(observed_tasks)),
                        "observed_tasks_min": int(np.min(observed_tasks)),
                        "observed_tasks_max": int(np.max(observed_tasks)),
                        "observed_resource_fraction_mean": float(
                            np.mean(observed_resource_fraction)
                        ),
                        "group_count": int(len(mapping)),
                        "rows": int(len(decision)),
                        "accept_rate": float(accept.mean()),
                        "reject_rate": float(reject.mean()),
                        "defer_rate": float(defer.mean()),
                        "false_accept_rate": float(false_accept.mean()),
                        "false_reject_rate": float(false_reject.mean()),
                        "decision_error_rate": float((false_accept | false_reject).mean()),
                        "coverage_failure_rate": 0.0,
                        "coverage_ok_rate": float(coverage_arr.mean()),
                        "defer_no_coverage_rate": float((defer & ~coverage_arr).mean()),
                        "defer_uncertain_rate": float((defer & coverage_arr).mean()),
                        "absolute_margin": float(abs(full_delta - threshold)),
                        "signed_margin": float(full_delta - threshold),
                        "tested_trials_min": int(pair_df["trials_observed_tested"].min()),
                        "comparison_trials_min": int(
                            pair_df["trials_observed_comparison"].min()
                        ),
                        "resource_completeness": resource_completeness,
                        "coverage_mode": args.coverage_mode,
                    }
                )
    return rows


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return float("nan")
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    return float((weights * values).sum() / weights.sum())


def aggregate(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "policy_alpha", "budget_cap", "budget_metric", "policy_label"]
    for group_keys, group in pair_rows.groupby(keys, dropna=False):
        source, threshold, alpha, budget, metric, policy = group_keys
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "policy_alpha": alpha,
                "budget_cap": budget,
                "budget_metric": metric,
                "policy_label": policy,
                "directed_pairs": int(group["pair_label"].nunique()),
                "positive_all_task_decisions": int(pos["pair_label"].nunique()),
                "negative_all_task_decisions": int(neg["pair_label"].nunique()),
                "conditional_false_accept_rate": weighted_mean(neg, "false_accept_rate"),
                "conditional_false_reject_rate": weighted_mean(pos, "false_reject_rate"),
                "decision_error_rate": weighted_mean(group, "decision_error_rate"),
                "coverage_failure_rate": weighted_mean(group, "coverage_failure_rate"),
                "defer_rate": weighted_mean(group, "defer_rate"),
                "coverage_ok_rate": weighted_mean(group, "coverage_ok_rate"),
                "defer_no_coverage_rate": weighted_mean(group, "defer_no_coverage_rate"),
                "defer_uncertain_rate": weighted_mean(group, "defer_uncertain_rate"),
                "observed_tasks_mean": weighted_mean(group, "observed_tasks_mean"),
                "observed_resource_fraction_mean": weighted_mean(
                    group, "observed_resource_fraction_mean"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def minimum_budgets(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "policy_alpha", "budget_metric", "policy_label"]
    for group_keys, group in aggregate_df.groupby(keys, dropna=False):
        source, threshold, alpha, metric, policy = group_keys
        ordered = group.sort_values("budget_cap")
        mask = (
            (ordered["conditional_false_accept_rate"] <= alpha)
            & (ordered["conditional_false_reject_rate"] <= alpha)
            & (ordered["coverage_failure_rate"] <= 0.05)
            & (ordered["defer_rate"] <= 0.25)
        )
        row = ordered[mask].iloc[0] if mask.any() else None
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "policy_alpha": alpha,
                "budget_metric": metric,
                "policy_label": policy,
                "minimum_sufficient_budget": float(row["budget_cap"]) if row is not None else math.nan,
                "defer_rate_at_minimum": float(row["defer_rate"]) if row is not None else math.nan,
                "observed_tasks_mean_at_minimum": float(row["observed_tasks_mean"])
                if row is not None
                else math.nan,
                "observed_resource_fraction_mean_at_minimum": float(
                    row["observed_resource_fraction_mean"]
                )
                if row is not None
                else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_rows = pd.read_csv(args.task_rows)
    pairs = prepare_pairs(task_rows)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for idx, pair in enumerate(pairs, start=1):
        pair_df = prepare_pair(task_rows, pair)
        if len(pair_df) < args.min_overlap_tasks:
            continue
        rows.extend(summarize_pair(pair, pair_df, args, rng))
        print(f"processed pair {idx}/{len(pairs)}: {pair.pair_label}", flush=True)
    pair_df = pd.DataFrame(rows)
    aggregate_df = aggregate(pair_df)
    minimum_df = minimum_budgets(aggregate_df)
    pair_df.to_csv(args.out_dir / "terminal_bench_resource_pair_summary.csv", index=False)
    aggregate_df.to_csv(
        args.out_dir / "terminal_bench_resource_aggregate.csv", index=False
    )
    minimum_df.to_csv(
        args.out_dir / "terminal_bench_resource_minimum_budgets.csv", index=False
    )
    usable_pairs = int(pair_df["pair_label"].nunique()) if not pair_df.empty else 0
    output_tasks = (
        int(pair_df["overlap_tasks"].median()) if not pair_df.empty else 0
    )
    coverage_note = (
        "No task-stratum floor was applied; public Terminal-Bench result JSON "
        "does not expose native task-domain labels."
        if args.coverage_mode == "none"
        else "coverage_group is inferred from task-name prefix in the current Terminal-Bench extraction."
    )
    metadata = {
        "task_rows": str(args.task_rows),
        "systems": int(task_rows["system_id"].nunique()),
        "task_names_in_input": int(task_rows["task_name"].nunique()),
        "shared_tasks_after_overlap_gate_median": output_tasks,
        "possible_directed_pairs": len(pairs),
        "usable_directed_pairs": usable_pairs,
        "thresholds": args.thresholds,
        "alpha": args.alpha,
        "budget_caps": args.budget_caps,
        "budget_metrics": args.budget_metrics,
        "coverage_mode": args.coverage_mode,
        "min_resource_completeness": args.min_resource_completeness,
        "n_replays": args.n_replays,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "coverage_group_note": coverage_note,
    }
    (args.out_dir / "terminal_bench_resource_replay_audit.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
