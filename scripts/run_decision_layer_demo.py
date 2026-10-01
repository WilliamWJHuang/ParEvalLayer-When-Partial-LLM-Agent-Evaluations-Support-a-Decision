from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "public_backtest"
DEFAULT_OUT = ANALYSIS / "decision_layer_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit decision records for one public benchmark comparison from "
            "the budget prefix. The completed record is used only for audit fields."
        )
    )
    parser.add_argument("--source", choices=["tau", "terminal"], default="tau")
    parser.add_argument("--tested", default=None)
    parser.add_argument("--comparison", default=None)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--budget-caps",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 0.95],
    )
    parser.add_argument(
        "--budget-metric",
        choices=[
            "task_fraction",
            "tested_cost_fraction",
            "pair_cost_fraction",
            "tested_time_fraction",
            "pair_time_fraction",
        ],
        default="tested_cost_fraction",
    )
    parser.add_argument(
        "--ordering",
        choices=["coverage_random", "coverage_resource"],
        default="coverage_resource",
    )
    parser.add_argument(
        "--coverage-mode",
        choices=["auto", "source", "none"],
        default="auto",
        help="Use source task strata, no strata, or source-specific default.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def default_systems(source: str) -> tuple[str, str]:
    if source == "tau":
        return "gpt-5-2_sierra_2026-02-26", "gpt-5-2-none_sierra_2026-02-26"
    if source == "terminal":
        task_rows = pd.read_csv(
            ANALYSIS / "terminal_bench" / "terminal_bench_task_rows.csv",
            usecols=["system_id", "task_name"],
        )
        by_system = {
            str(system): set(group["task_name"].astype(str))
            for system, group in task_rows.groupby("system_id")
        }
        systems = sorted(by_system)
        if len(systems) < 2:
            raise ValueError("Terminal-Bench demo requires at least two systems.")
        for tested in systems:
            for comparison in systems:
                if tested == comparison:
                    continue
                if by_system[tested] & by_system[comparison]:
                    return tested, comparison
        raise ValueError("Terminal-Bench demo requires a system pair with shared tasks.")
    raise ValueError(source)


def load_tau_pair(tested: str, comparison: str) -> pd.DataFrame:
    path = ANALYSIS / "tau2_ready" / "public_tau2_task_rows.csv"
    task_rows = pd.read_csv(path)
    keep = [
        "submission_dir",
        "task_key",
        "domain",
        "outcome_mean",
        "cost_usd",
        "agent_cost_usd",
        "duration_seconds",
        "trials_observed",
    ]
    left = task_rows[task_rows["submission_dir"].eq(tested)][keep].copy()
    right = task_rows[task_rows["submission_dir"].eq(comparison)][keep].copy()
    merged = left.merge(
        right,
        on="task_key",
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping tau-bench tasks for {tested} vs {comparison}.")
    merged["group"] = merged["domain_tested"].astype(str)
    merged["outcome_tested"] = merged["outcome_mean_tested"].astype(float)
    merged["outcome_comparison"] = merged["outcome_mean_comparison"].astype(float)
    merged["tested_resource"] = merged["cost_usd_tested"].astype(float)
    merged["comparison_resource"] = merged["cost_usd_comparison"].astype(float)
    merged["tested_time"] = merged["duration_seconds_tested"].astype(float)
    merged["comparison_time"] = merged["duration_seconds_comparison"].astype(float)
    return merged.sort_values(["group", "task_key"]).reset_index(drop=True)


def load_terminal_pair(tested: str, comparison: str) -> pd.DataFrame:
    path = ANALYSIS / "terminal_bench" / "terminal_bench_task_rows.csv"
    task_rows = pd.read_csv(path)
    keep = [
        "system_id",
        "task_name",
        "coverage_group",
        "outcome",
        "mean_agent_execution_seconds",
        "mean_cost_usd",
        "trials_observed",
    ]
    left = task_rows[task_rows["system_id"].eq(tested)][keep].copy()
    right = task_rows[task_rows["system_id"].eq(comparison)][keep].copy()
    merged = left.merge(
        right,
        on="task_name",
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            f"No overlapping Terminal-Bench tasks for {tested} vs {comparison}."
        )
    merged["task_key"] = merged["task_name"]
    merged["group"] = merged["coverage_group_tested"].fillna("terminal").astype(str)
    merged["outcome_tested"] = merged["outcome_tested"].astype(float)
    merged["outcome_comparison"] = merged["outcome_comparison"].astype(float)
    merged["tested_resource"] = merged["mean_cost_usd_tested"].astype(float)
    merged["comparison_resource"] = merged["mean_cost_usd_comparison"].astype(float)
    merged["tested_time"] = merged["mean_agent_execution_seconds_tested"].astype(float)
    merged["comparison_time"] = merged["mean_agent_execution_seconds_comparison"].astype(
        float
    )
    return merged.sort_values(["group", "task_key"]).reset_index(drop=True)


def resource_values(pair_df: pd.DataFrame, metric: str) -> np.ndarray:
    if metric == "task_fraction":
        return np.ones(len(pair_df), dtype=float)
    if metric == "tested_cost_fraction":
        return pair_df["tested_resource"].to_numpy(dtype=float)
    if metric == "pair_cost_fraction":
        return (
            pair_df["tested_resource"].to_numpy(dtype=float)
            + pair_df["comparison_resource"].to_numpy(dtype=float)
        )
    if metric == "tested_time_fraction":
        return pair_df["tested_time"].to_numpy(dtype=float)
    if metric == "pair_time_fraction":
        return (
            pair_df["tested_time"].to_numpy(dtype=float)
            + pair_df["comparison_time"].to_numpy(dtype=float)
        )
    raise ValueError(metric)


def clean_resources(resources: np.ndarray, metric: str) -> tuple[np.ndarray, float]:
    finite = np.isfinite(resources)
    completeness = float(finite.mean()) if len(resources) else 0.0
    if metric == "task_fraction":
        return resources, 1.0
    if not finite.any():
        raise ValueError(f"No finite resource values for metric {metric}.")
    replacement = float(np.nanmedian(resources))
    return np.nan_to_num(resources, nan=replacement), completeness


def rng_for_budget(seed: int, budget_cap: float) -> np.random.Generator:
    cap_key = int(round(float(budget_cap) * 1_000_000))
    return np.random.default_rng(int(seed) + cap_key)


def group_order(
    pair_df: pd.DataFrame,
    resources: np.ndarray,
    ordering: str,
    rng: np.random.Generator,
) -> list[int]:
    groups = sorted(str(item) for item in pair_df["group"].unique())
    remaining: dict[str, list[int]] = {}
    for group in groups:
        idx = pair_df.index[pair_df["group"].eq(group)].to_numpy()
        if ordering == "coverage_random":
            idx = rng.permutation(idx)
        else:
            idx = idx[np.argsort(resources[idx], kind="mergesort")]
        remaining[group] = [int(item) for item in idx]

    observed = {group: 0 for group in groups}
    total = {group: len(remaining[group]) for group in groups}
    order: list[int] = []
    while any(remaining[group] for group in groups):
        best_group = None
        best_score = None
        for group in groups:
            if not remaining[group]:
                continue
            score = observed[group] / total[group]
            tie = resources[remaining[group][0]]
            choice = (score, tie, group)
            if best_score is None or choice < best_score:
                best_score = choice
                best_group = group
        assert best_group is not None
        order.append(remaining[best_group].pop(0))
        observed[best_group] += 1
    return order


def prefix_indices(
    order: list[int],
    resources: np.ndarray,
    budget_cap: float,
    metric: str,
) -> list[int]:
    if metric == "task_fraction":
        cap_k = max(1, min(len(order), int(math.ceil(len(order) * budget_cap))))
        return order[:cap_k]
    total = float(resources.sum())
    if total <= 0:
        return order[:1]
    limit = budget_cap * total
    cumulative = np.cumsum(resources[order])
    count = int(np.searchsorted(cumulative, limit, side="right"))
    count = max(1, min(len(order), count))
    return order[:count]


def coverage_requirements(pair_df: pd.DataFrame, observed_count: int) -> dict[str, int]:
    requirements: dict[str, int] = {}
    n_tasks = len(pair_df)
    for group, group_df in pair_df.groupby("group"):
        group_size = int(len(group_df))
        expected = observed_count * (group_size / n_tasks)
        requirements[str(group)] = max(1, min(group_size, math.floor(expected)))
    return requirements


def coverage_state(pair_df: pd.DataFrame, prefix: list[int]) -> dict[str, object]:
    required = coverage_requirements(pair_df, len(prefix))
    observed_counts = {
        str(group): int(count)
        for group, count in pair_df.iloc[prefix]["group"].value_counts().items()
    }
    missing = [
        group
        for group, required_count in required.items()
        if observed_counts.get(group, 0) < required_count
    ]
    return {
        "required_counts": required,
        "observed_counts": observed_counts,
        "missing_strata": missing,
        "coverage_ok": not missing,
    }


def bootstrap_action(
    values: np.ndarray,
    threshold: float,
    alpha: float,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    mean = float(values.mean())
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    tail_promote = float((draws <= threshold).mean())
    tail_reject = float((draws > threshold).mean())
    if mean > threshold and tail_promote <= alpha:
        action = "promote"
    elif mean <= threshold and tail_reject <= alpha:
        action = "reject"
    else:
        action = "abstain"
    return {
        "action": action,
        "partial_delta": mean,
        "bootstrap_tail_if_promote": tail_promote,
        "bootstrap_tail_if_reject": tail_reject,
    }


def card_for_budget(
    source: str,
    pair_df: pd.DataFrame,
    tested: str,
    comparison: str,
    threshold: float,
    alpha: float,
    budget_cap: float,
    metric: str,
    ordering: str,
    coverage_mode: str,
    prefix: list[int],
    resources: np.ndarray,
    resource_completeness: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    diffs = (
        pair_df["outcome_tested"].to_numpy(dtype=float)
        - pair_df["outcome_comparison"].to_numpy(dtype=float)
    )
    full_delta = float(diffs.mean())
    full_record_decision = "promote" if full_delta > threshold else "reject"
    coverage = coverage_state(pair_df, prefix)
    observed_resource_fraction = (
        float(resources[prefix].sum() / resources.sum())
        if metric != "task_fraction" and resources.sum() > 0
        else float(len(prefix) / len(pair_df))
    )

    if not coverage["coverage_ok"]:
        decision = {
            "action": "abstain",
            "partial_delta": float(diffs[prefix].mean()),
            "bootstrap_tail_if_promote": None,
            "bootstrap_tail_if_reject": None,
        }
        abstention_reason = "missing_required_strata"
    else:
        decision = bootstrap_action(
            diffs[prefix], threshold, alpha, bootstrap_samples, rng
        )
        abstention_reason = (
            "bootstrap_tail_not_below_alpha"
            if decision["action"] == "abstain"
            else None
        )

    action = str(decision["action"])
    matches_full_record = action == full_record_decision if action != "abstain" else None
    return {
        "schema_version": "1.1",
        "source": source,
        "systems": {"tested": tested, "comparison": comparison},
        "threshold": threshold,
        "alpha": alpha,
        "task_selection_policy": {
            "name": ordering,
            "budget_metric": metric,
            "coverage_mode": coverage_mode,
            "prefix_only": True,
        },
        "budget_cap": budget_cap,
        "observed": {
            "tasks": len(prefix),
            "task_fraction": len(prefix) / len(pair_df),
            "resource_fraction": observed_resource_fraction,
            "resource_completeness": resource_completeness,
        },
        "strata_coverage": coverage,
        "decision": {
            "action": action,
            "partial_delta": decision["partial_delta"],
            "bootstrap_tail_if_promote": decision["bootstrap_tail_if_promote"],
            "bootstrap_tail_if_reject": decision["bootstrap_tail_if_reject"],
            "abstention_reason": abstention_reason,
        },
        "audit_only_full_record": {
            "full_delta": full_delta,
            "full_record_decision": full_record_decision,
            "matches_full_record_when_decided": matches_full_record,
        },
    }


def write_summary(cards: list[dict[str, object]], path: Path) -> None:
    lines = ["# Decision-layer demo", ""]
    if not cards:
        lines.append("No cards emitted.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    first = cards[0]
    lines.extend(
        [
            f"Source: `{first['source']}`",
            f"Tested system: `{first['systems']['tested']}`",
            f"Comparison system: `{first['systems']['comparison']}`",
            f"Threshold: {float(first['threshold']) * 100:.0f} pp",
            "",
            "| Budget | Tasks | Resource fraction | Action | Partial delta | Coverage | Audit match |",
            "|---:|---:|---:|---|---:|---|---|",
        ]
    )
    for card in cards:
        action = card["decision"]["action"]
        coverage = "yes" if card["strata_coverage"]["coverage_ok"] else "no"
        match = card["audit_only_full_record"]["matches_full_record_when_decided"]
        if match is None:
            match_label = "n/a"
        else:
            match_label = "yes" if match else "no"
        lines.append(
            "| "
            f"{float(card['budget_cap']) * 100:.0f}% | "
            f"{card['observed']['tasks']} | "
            f"{float(card['observed']['resource_fraction']) * 100:.1f}% | "
            f"{action} | "
            f"{float(card['decision']['partial_delta']) * 100:.1f} pp | "
            f"{coverage} | "
            f"{match_label} |"
        )
    lines.extend(
        [
            "",
            "The action is computed from the observed prefix only. "
            "Completed-record fields are included only to audit agreement with the completed record.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tested, comparison = default_systems(args.source)
    tested = args.tested or tested
    comparison = args.comparison or comparison

    if args.source == "tau":
        pair_df = load_tau_pair(tested, comparison)
        source_label = "tau-bench"
    else:
        pair_df = load_terminal_pair(tested, comparison)
        source_label = "Terminal-Bench 2.0"

    coverage_mode = args.coverage_mode
    if coverage_mode == "auto":
        coverage_mode = "source" if args.source == "tau" else "none"
    if coverage_mode == "none":
        pair_df["group"] = "all"

    resources, completeness = clean_resources(
        resource_values(pair_df, args.budget_metric), args.budget_metric
    )
    rng = np.random.default_rng(args.seed)
    order = group_order(pair_df, resources, args.ordering, rng)

    cards = []
    for budget_cap in sorted(args.budget_caps):
        prefix = prefix_indices(order, resources, budget_cap, args.budget_metric)
        cards.append(
            card_for_budget(
                source_label,
                pair_df,
                tested,
                comparison,
                args.threshold,
                args.alpha,
                budget_cap,
                args.budget_metric,
                args.ordering,
                coverage_mode,
                prefix,
                resources,
                completeness,
                args.bootstrap_samples,
                rng_for_budget(args.seed, budget_cap),
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cards_path = args.out_dir / f"{args.source}_decision_cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, sort_keys=True) + "\n")
    write_summary(cards, args.out_dir / f"{args.source}_decision_layer_demo.md")

    print(
        json.dumps(
            {
                "cards": len(cards),
                "cards_path": str(cards_path),
                "summary_path": str(args.out_dir / f"{args.source}_decision_layer_demo.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
