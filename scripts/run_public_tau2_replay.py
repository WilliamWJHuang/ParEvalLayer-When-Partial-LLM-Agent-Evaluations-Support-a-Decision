from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COMPARISONS = [
    (
        "gpt-5-2_sierra_2026-02-26",
        "gpt-5-2-none_sierra_2026-02-26",
        "gpt52_high_vs_none",
    ),
    (
        "claude-opus-4-5_sierra_2026-02-26",
        "gpt-5-2_sierra_2026-02-26",
        "opus45_vs_gpt52_high",
    ),
    (
        "gpt-5-2_sierra_2026-02-26",
        "claude-sonnet-4-5_sierra_2026-02-26",
        "gpt52_high_vs_sonnet45",
    ),
]
POLICIES = [
    "fixed_uniform_forced",
    "fixed_stratified_forced",
    "cost_aware_forced",
    "bootstrap_ci",
    "coverage_bootstrap",
]


@dataclass
class Comparison:
    candidate: str
    reference: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run public tau-bench partial-evaluation replays."
    )
    parser.add_argument(
        "--task-table",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "tau2_ready"
        / "public_tau2_task_rows.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis") / "public_backtest" / "tau2_replay",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="candidate,reference,label. May be passed multiple times.",
    )
    parser.add_argument("--n-replays", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--budget-caps", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument(
        "--cost-mode",
        choices=["candidate", "pairwise"],
        default="candidate",
    )
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


def parse_comparison(text: str) -> Comparison:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Bad comparison spec: {text}")
    return Comparison(candidate=parts[0], reference=parts[1], label=parts[2])


def default_comparisons() -> list[Comparison]:
    return [Comparison(*item) for item in DEFAULT_COMPARISONS]


def random_orders(n_replays: int, n_tasks: int, rng: np.random.Generator) -> np.ndarray:
    return np.argsort(rng.random((n_replays, n_tasks)), axis=1).astype(np.int16)


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
    orders = np.empty((n_replays, len(strata_code)), dtype=np.int16)
    for replay_idx in range(n_replays):
        scores = np.empty(len(strata_code), dtype=float)
        for indices in by_stratum:
            permuted = rng.permutation(indices)
            local_scores = (np.arange(len(permuted)) + rng.random(len(permuted))) / len(
                permuted
            )
            scores[permuted] = local_scores
        orders[replay_idx] = np.argsort(scores).astype(np.int16)
    return orders


def cost_aware_orders(costs: np.ndarray, n_replays: int, rng: np.random.Generator) -> np.ndarray:
    clean_costs = np.nan_to_num(costs, nan=float(np.nanmedian(costs)))
    jitter = rng.random((n_replays, len(clean_costs))) * 1e-9
    return np.argsort(clean_costs[None, :] + jitter, axis=1).astype(np.int16)


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


def bootstrap_stats(
    seen: np.ndarray,
    threshold: float,
    alpha: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_replays = seen.shape[0]
    means = seen.mean(axis=1)
    lows = np.empty(n_replays, dtype=float)
    highs = np.empty(n_replays, dtype=float)
    risk_promote = np.empty(n_replays, dtype=float)
    risk_reject = np.empty(n_replays, dtype=float)
    for idx in range(n_replays):
        values = seen[idx]
        sample = rng.choice(
            values,
            size=(bootstrap_samples, len(values)),
            replace=True,
        ).mean(axis=1)
        lows[idx], highs[idx] = np.quantile(sample, [alpha / 2, 1 - alpha / 2])
        risk_promote[idx] = float((sample <= threshold).mean())
        risk_reject[idx] = float((sample > threshold).mean())
    return means, lows, highs, risk_promote, risk_reject


def prepare_pair(task_df: pd.DataFrame, comparison: Comparison) -> pd.DataFrame | None:
    candidate = task_df[task_df["submission_dir"].eq(comparison.candidate)].copy()
    reference = task_df[task_df["submission_dir"].eq(comparison.reference)].copy()
    if candidate.empty or reference.empty:
        return None
    keep_cols = [
        "domain",
        "task_id",
        "task_key",
        "outcome_mean",
        "cost_usd",
        "model_name",
        "reasoning_effort",
        "tau2_bench_version",
        "user_simulator",
    ]
    merged = candidate[keep_cols].merge(
        reference[keep_cols],
        on=["domain", "task_id", "task_key"],
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    if merged.empty:
        return None
    merged["diff"] = (
        merged["outcome_mean_candidate"] - merged["outcome_mean_reference"]
    )
    merged["candidate_cost_usd"] = merged["cost_usd_candidate"]
    merged["comparison_label"] = comparison.label
    merged["candidate"] = comparison.candidate
    merged["reference"] = comparison.reference
    return merged.sort_values(["domain", "task_id"]).reset_index(drop=True)


def summarize_decisions(
    decision: np.ndarray,
    truth_promote: bool,
    task_fraction: np.ndarray,
    cost_fraction: np.ndarray,
    coverage_failure: np.ndarray,
) -> dict[str, float | int]:
    promoted = decision == 1
    rejected = decision == -1
    abstained = decision == 0
    false_promotion = promoted & (not truth_promote)
    false_rejection = rejected & truth_promote
    decision_error = false_promotion | false_rejection
    promoted_count = int(promoted.sum())
    return {
        "rows": int(len(decision)),
        "mean_task_fraction": float(task_fraction.mean()),
        "mean_candidate_cost_fraction": float(np.nanmean(cost_fraction)),
        "promote_rate": float(promoted.mean()),
        "reject_rate": float(rejected.mean()),
        "abstain_rate": float(abstained.mean()),
        "false_promotion_rate": float(false_promotion.mean()),
        "false_promotion_per_promotion": (
            float(false_promotion.sum() / promoted_count)
            if promoted_count
            else float("nan")
        ),
        "false_rejection_rate": float(false_rejection.mean()),
        "decision_error_rate": float(decision_error.mean()),
        "coverage_failure_rate": float(coverage_failure.mean()),
        "mean_evaluation_cost_fraction": float(np.nanmean(cost_fraction)),
    }


def run_replays_for_pair(
    pair: pd.DataFrame,
    comparison: Comparison,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict[str, float | int | str | bool]], dict[str, float | int | str]]:
    diffs = pair["diff"].to_numpy(dtype=float)
    candidate_costs = pair["cost_usd_candidate"].to_numpy(dtype=float)
    reference_costs = pair["cost_usd_reference"].to_numpy(dtype=float)
    if args.cost_mode == "candidate":
        costs = candidate_costs
    elif args.cost_mode == "pairwise":
        costs = candidate_costs + reference_costs
    else:
        raise ValueError(args.cost_mode)
    missing_cost = int(np.isnan(costs).sum())
    median_cost = float(np.nanmedian(costs)) if not np.isnan(costs).all() else 0.0
    costs_for_replay = np.where(np.isnan(costs), median_cost, costs)
    total_cost = float(costs_for_replay.sum())
    n_tasks = len(pair)
    strata_code, strata_mapping = strata_codes(pair["domain"].to_numpy())
    full_delta = float(diffs.mean())
    truth = {
        "comparison_label": comparison.label,
        "candidate": comparison.candidate,
        "reference": comparison.reference,
        "n_tasks": int(n_tasks),
        "domains": ",".join(sorted(pair["domain"].unique().tolist())),
        "full_delta": full_delta,
        "candidate_model": str(pair["model_name_candidate"].iloc[0]),
        "reference_model": str(pair["model_name_reference"].iloc[0]),
        "candidate_reasoning_effort": str(pair["reasoning_effort_candidate"].iloc[0]),
        "reference_reasoning_effort": str(pair["reasoning_effort_reference"].iloc[0]),
        "candidate_tau2_bench_version": str(pair["tau2_bench_version_candidate"].iloc[0]),
        "reference_tau2_bench_version": str(pair["tau2_bench_version_reference"].iloc[0]),
        "cost_mode": args.cost_mode,
        "missing_replay_cost_tasks": missing_cost,
        "cost_imputation": "median for replay cost ordering and fractions",
    }
    orders = {
        "fixed_uniform_forced": random_orders(args.n_replays, n_tasks, rng),
        "fixed_stratified_forced": stratified_orders(strata_code, args.n_replays, rng),
        "cost_aware_forced": cost_aware_orders(costs_for_replay, args.n_replays, rng),
        "bootstrap_ci": random_orders(args.n_replays, n_tasks, rng),
        "coverage_bootstrap": stratified_orders(strata_code, args.n_replays, rng),
    }
    rows: list[dict[str, float | int | str | bool]] = []
    for threshold in args.thresholds:
        truth_promote = full_delta > threshold
        for budget in args.budget_caps:
            cap_k = max(1, int(math.ceil(n_tasks * budget)))
            requirements = coverage_requirements(strata_code, cap_k, args.coverage_slack)
            for policy in POLICIES:
                selected = orders[policy][:, :cap_k]
                seen = diffs[selected]
                seen_strata = strata_code[selected]
                seen_cost = costs_for_replay[selected]
                sample_delta = seen.mean(axis=1)
                decision = np.zeros(args.n_replays, dtype=np.int8)
                if policy in {
                    "fixed_uniform_forced",
                    "fixed_stratified_forced",
                    "cost_aware_forced",
                }:
                    decision[sample_delta > threshold] = 1
                    decision[sample_delta <= threshold] = -1
                elif policy == "bootstrap_ci":
                    _, low, high, _, _ = bootstrap_stats(
                        seen,
                        threshold,
                        args.alpha,
                        args.bootstrap_samples,
                        rng,
                    )
                    decision[low > threshold] = 1
                    decision[high <= threshold] = -1
                elif policy == "coverage_bootstrap":
                    _, _, _, risk_promote, risk_reject = bootstrap_stats(
                        seen,
                        threshold,
                        args.alpha,
                        args.bootstrap_samples,
                        rng,
                    )
                    ok = coverage_ok(seen_strata, requirements)
                    promote = ok & (sample_delta > threshold) & (
                        risk_promote <= args.alpha
                    )
                    reject = ok & (sample_delta <= threshold) & (
                        risk_reject <= args.alpha
                    )
                    decision[promote] = 1
                    decision[reject] = -1
                else:
                    raise ValueError(policy)
                task_fraction = np.full(args.n_replays, cap_k / n_tasks)
                cost_fraction = (
                    seen_cost.sum(axis=1) / total_cost
                    if total_cost > 0
                    else np.full(args.n_replays, np.nan)
                )
                coverage_failure = ~coverage_ok(seen_strata, requirements)
                summary = summarize_decisions(
                    decision,
                    truth_promote,
                    task_fraction,
                    cost_fraction,
                    coverage_failure,
                )
                rows.append(
                    {
                        "comparison_label": comparison.label,
                        "candidate": comparison.candidate,
                        "reference": comparison.reference,
                        "threshold": float(threshold),
                        "budget_cap": float(budget),
                        "policy": policy,
                        "n_tasks": int(n_tasks),
                        "full_delta": full_delta,
                        "truth_promote": bool(truth_promote),
                        "cost_mode": args.cost_mode,
                        "missing_replay_cost_tasks": missing_cost,
                        **summary,
                    }
                )
    return rows, truth


def main() -> None:
    args = parse_args()
    task_df = pd.read_csv(args.task_table)
    comparisons = (
        [parse_comparison(item) for item in args.comparison]
        if args.comparison
        else default_comparisons()
    )
    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, float | int | str | bool]] = []
    truth_rows: list[dict[str, float | int | str]] = []
    skipped: list[dict[str, str]] = []
    for comparison in comparisons:
        pair = prepare_pair(task_df, comparison)
        if pair is None:
            skipped.append(
                {
                    "comparison_label": comparison.label,
                    "candidate": comparison.candidate,
                    "reference": comparison.reference,
                    "reason": "missing candidate, reference, or common tasks",
                }
            )
            continue
        rows, truth = run_replays_for_pair(pair, comparison, args, rng)
        summary_rows.extend(rows)
        truth_rows.append(truth)

    if not summary_rows:
        raise ValueError(f"No replay rows were produced. Skipped: {skipped}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "public_tau2_replay_summary.csv"
    truth_path = args.out_dir / "public_tau2_truth_summary.csv"
    metadata_path = args.out_dir / "public_tau2_replay_metadata.json"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    metadata = {
        "task_table": str(args.task_table),
        "summary_table": str(summary_path),
        "truth_table": str(truth_path),
        "n_replays": args.n_replays,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "alpha": args.alpha,
        "cost_mode": args.cost_mode,
        "thresholds": args.thresholds,
        "budget_caps": args.budget_caps,
        "coverage_slack": args.coverage_slack,
        "comparisons_requested": [item.__dict__ for item in comparisons],
        "comparisons_skipped": skipped,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
