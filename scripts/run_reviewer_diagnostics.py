from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
ANALYSIS = ROOT / "analysis" / "public_backtest"
OUT_DIR = ANALYSIS / "reviewer_diagnostics"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_main_policy_sensitivity as base  # noqa: E402


PRIMARY_ALPHA = 0.05
PRIMARY_GAMMA = 0.05
PRIMARY_ETA = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reviewer-facing robustness diagnostics for ParEvalLayer."
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["swe_lite", "swe_verified", "appworld", "tau"],
        choices=["swe_lite", "swe_verified", "appworld", "tau", "osworld"],
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260621, 20260622, 20260623]
    )
    parser.add_argument(
        "--bootstrap-levels", type=int, nargs="+", default=[100, 200, 500]
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.00, 0.05, 0.10]
    )
    parser.add_argument(
        "--budget-caps",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.55, 0.75, 0.85, 0.90, 0.95],
    )
    parser.add_argument("--swe-replays", type=int, default=250)
    parser.add_argument("--appworld-replays", type=int, default=250)
    parser.add_argument("--tau-replays", type=int, default=1000)
    parser.add_argument("--osworld-replays", type=int, default=500)
    parser.add_argument(
        "--skip-stability", action="store_true", help="Only run grouping and SWE diagnostics."
    )
    parser.add_argument(
        "--reuse-stability",
        action="store_true",
        help="Reuse existing stability CSV outputs in out-dir instead of recomputing them.",
    )
    parser.add_argument(
        "--skip-grouping", action="store_true", help="Only run seed/bootstrap stability."
    )
    parser.add_argument(
        "--max-pairs-per-source",
        type=int,
        default=None,
        help="Smoke-test limit. Do not use for paper diagnostics.",
    )
    return parser.parse_args()


def percent(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "None"
    return f"{100 * float(value):.{digits}f}%"


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


def load_source(source_key: str):
    if source_key == "swe_lite":
        task_rows, pairs = base.swe_pairs("lite")
        return task_rows, pairs, base.prepare_swe_pair, "SWE-bench Lite", "repo"
    if source_key == "swe_verified":
        task_rows, pairs = base.swe_pairs("verified")
        return task_rows, pairs, base.prepare_swe_pair, "SWE-bench Verified", "repo"
    if source_key == "appworld":
        task_rows, pairs = base.appworld_pairs()
        return task_rows, pairs, base.prepare_appworld_pair, "AppWorld", "stratum"
    if source_key == "tau":
        task_rows, pairs = base.tau_pairs()
        return task_rows, pairs, base.prepare_tau_pair, "tau-bench", "domain"
    if source_key == "osworld":
        task_rows, pairs = base.osworld_pairs()
        return (
            task_rows,
            pairs,
            base.prepare_osworld_pair,
            "OSWorld-Verified",
            "coverage_group",
        )
    raise ValueError(source_key)


def replay_count_for_source(source_key: str, args: argparse.Namespace) -> int:
    if source_key.startswith("swe"):
        return args.swe_replays
    if source_key == "appworld":
        return args.appworld_replays
    if source_key == "tau":
        return args.tau_replays
    if source_key == "osworld":
        return args.osworld_replays
    raise ValueError(source_key)


def summarize_vector(
    decision: np.ndarray,
    truth_accept: bool,
    coverage: np.ndarray,
    full_delta: float,
    threshold: float,
    task_fraction: float,
) -> dict[str, float | int]:
    accept = decision == 1
    reject = decision == -1
    defer = decision == 0
    false_accept = accept & (not truth_accept)
    false_reject = reject & truth_accept
    return {
        "rows": int(len(decision)),
        "mean_task_fraction": float(task_fraction),
        "accept_rate": float(accept.mean()),
        "reject_rate": float(reject.mean()),
        "defer_rate": float(defer.mean()),
        "false_accept_rate": float(false_accept.mean()),
        "false_reject_rate": float(false_reject.mean()),
        "decision_error_rate": float((false_accept | false_reject).mean()),
        "coverage_failure_rate": 0.0,
        "coverage_ok_rate": float(coverage.mean()),
        "defer_no_coverage_rate": float((defer & (~coverage)).mean()),
        "defer_uncertain_rate": float((defer & coverage).mean()),
        "absolute_margin": float(abs(full_delta - threshold)),
    }


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return float("nan")
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((weights * values).sum() / total)


def aggregate_pair_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "source",
        "threshold",
        "budget_cap",
        "seed",
        "bootstrap_samples",
        "policy_label",
    ]
    for key_values, group in pair_rows.groupby(keys, dropna=False):
        source, threshold, budget, seed, bootstrap_samples, policy_label = key_values
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "budget_cap": budget,
                "seed": seed,
                "bootstrap_samples": bootstrap_samples,
                "policy_label": policy_label,
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
                "coverage_ok_rate": weighted_mean(group, "coverage_ok_rate"),
                "defer_rate": weighted_mean(group, "defer_rate"),
                "defer_no_coverage_rate": weighted_mean(
                    group, "defer_no_coverage_rate"
                ),
                "defer_uncertain_rate": weighted_mean(group, "defer_uncertain_rate"),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def minimum_budgets(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "seed", "bootstrap_samples", "policy_label"]
    for key_values, group in aggregate.groupby(keys, dropna=False):
        source, threshold, seed, bootstrap_samples, policy_label = key_values
        ordered = group.sort_values("budget_cap")
        mask = (
            (ordered["conditional_false_accept_rate"] <= PRIMARY_ALPHA)
            & (ordered["conditional_false_reject_rate"] <= PRIMARY_ALPHA)
            & (ordered["coverage_failure_rate"] <= PRIMARY_GAMMA)
            & (ordered["defer_rate"] <= PRIMARY_ETA)
        )
        row = ordered[mask].iloc[0] if mask.any() else None
        first = ordered.iloc[0]
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "seed": seed,
                "bootstrap_samples": bootstrap_samples,
                "policy_label": policy_label,
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
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def minimum_budget_distribution(min_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in min_df.groupby(["source", "threshold", "policy_label"]):
        source, threshold, policy_label = keys
        total = len(group)
        counts: dict[str, int] = {}
        for value in group["minimum_sufficient_budget"]:
            label = "None" if pd.isna(value) else f"{float(value):.2f}"
            counts[label] = counts.get(label, 0) + 1
        def sort_key(item: tuple[str, int]) -> tuple[int, float | str]:
            label, _ = item
            if label == "None":
                return (1, label)
            return (0, float(label))
        for label, count in sorted(counts.items(), key=sort_key):
            rows.append(
                {
                    "source": source,
                    "threshold": threshold,
                    "policy_label": policy_label,
                    "minimum_sufficient_budget": label,
                    "config_count": int(count),
                    "config_probability": float(count / total),
                    "total_configs": int(total),
                }
            )
    return pd.DataFrame(rows)


def metric_intervals_over_configs(aggregate: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "conditional_false_accept_rate",
        "conditional_false_reject_rate",
        "coverage_failure_rate",
        "coverage_ok_rate",
        "defer_rate",
        "defer_no_coverage_rate",
        "defer_uncertain_rate",
    ]
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "budget_cap", "policy_label"]
    for key_values, group in aggregate.groupby(keys, dropna=False):
        source, threshold, budget, policy_label = key_values
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "source": source,
                    "threshold": threshold,
                    "budget_cap": budget,
                    "policy_label": policy_label,
                    "metric": metric,
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                    "q05": float(np.nanquantile(values, 0.05)),
                    "q95": float(np.nanquantile(values, 0.95)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "configs": int(len(values)),
                }
            )
    return pd.DataFrame(rows).sort_values(keys + ["metric"])


def run_stability(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    b_levels = sorted(set(int(item) for item in args.bootstrap_levels))
    max_b = max(b_levels)
    for source_key in args.sources:
        task_rows, pairs, prepare, source_name, _ = load_source(source_key)
        n_replays = replay_count_for_source(source_key, args)
        total_pairs = (
            min(len(pairs), args.max_pairs_per_source)
            if args.max_pairs_per_source is not None
            else len(pairs)
        )
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            for idx, pair in enumerate(pairs[:total_pairs], start=1):
                pair_df = prepare(task_rows, pair)
                diffs = pair_df["diff"].to_numpy(dtype=np.float32)
                group_code, _ = base.group_codes(pair_df[pair.group_column].to_numpy())
                orders = base.random_stratified_orders(group_code, n_replays, rng)
                full_delta = float(diffs.mean())
                for budget_cap in args.budget_caps:
                    cap_k = max(1, min(len(diffs), int(math.ceil(len(diffs) * budget_cap))))
                    prefix = orders[:, :cap_k]
                    seen = diffs[prefix]
                    prefix_groups = group_code[prefix]
                    coverage = base.coverage_ok(
                        prefix_groups, base.coverage_requirements(group_code, cap_k)
                    )
                    boot = base.bootstrap_means(seen, max_b, rng)
                    sample_mean = seen.mean(axis=1)
                    for threshold in args.thresholds:
                        truth_accept = full_delta > threshold
                        for b in b_levels:
                            subset = boot[:, :b]
                            risk_accept = (subset <= threshold).mean(axis=1)
                            risk_reject = (subset > threshold).mean(axis=1)
                            decision = np.zeros(n_replays, dtype=np.int8)
                            accept = (
                                coverage
                                & (sample_mean > threshold)
                                & (risk_accept <= PRIMARY_ALPHA)
                            )
                            reject = (
                                coverage
                                & (sample_mean <= threshold)
                                & (risk_reject <= PRIMARY_ALPHA)
                            )
                            decision[accept] = 1
                            decision[reject] = -1
                            row = {
                                "source": source_name,
                                "source_key": source_key,
                                "seed": seed,
                                "bootstrap_samples": b,
                                "threshold": threshold,
                                "budget_cap": budget_cap,
                                "policy_label": base.main_policy_label(source_name),
                                "tested_system": pair.tested,
                                "comparison_system": pair.comparison,
                                "pair_label": pair.pair_label,
                                "truth_accept": bool(truth_accept),
                                "full_delta": full_delta,
                                "overlap_tasks": int(len(diffs)),
                                "observed_tasks": int(cap_k),
                                "group_count": int(len(np.unique(group_code))),
                            }
                            row.update(
                                summarize_vector(
                                    decision,
                                    truth_accept,
                                    coverage,
                                    full_delta,
                                    threshold,
                                    cap_k / len(diffs),
                                )
                            )
                            rows.append(row)
                if idx % 50 == 0 or idx == total_pairs:
                    print(
                        f"stability {source_key} seed {seed}: {idx}/{total_pairs} pairs",
                        flush=True,
                    )
    pair_rows = pd.DataFrame(rows)
    aggregate = aggregate_pair_rows(pair_rows)
    min_df = minimum_budgets(aggregate)
    return pair_rows, aggregate, min_df


def appworld_grouped_pair(
    task_rows: pd.DataFrame, pair: base.PairSpec, grouping: str
) -> pd.DataFrame:
    df = base.prepare_appworld_pair(task_rows, pair)
    if grouping == "split_difficulty":
        df["group"] = df["stratum"].astype(str)
    elif grouping == "split":
        df["group"] = df["split_tested"].astype(str)
    elif grouping == "difficulty":
        df["group"] = "difficulty_" + df["difficulty_tested"].astype(str)
    elif grouping == "scenario":
        df["group"] = df["scenario_id_tested"].astype(str)
    else:
        raise ValueError(grouping)
    return df


def tau_grouped_pair(
    task_rows: pd.DataFrame, pair: base.PairSpec, grouping: str
) -> pd.DataFrame:
    tested = task_rows[task_rows["submission_dir"].eq(pair.tested)].copy()
    comparison = task_rows[task_rows["submission_dir"].eq(pair.comparison)].copy()
    keep = ["domain", "task_id", "task_key", "outcome_mean", "cost_usd"]
    df = tested[keep].merge(
        comparison[keep],
        on=["domain", "task_id", "task_key"],
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    df["diff"] = df["outcome_mean_tested"] - df["outcome_mean_comparison"]
    if grouping == "domain":
        df["group"] = df["domain"].astype(str)
    elif grouping == "none":
        df["group"] = "all_tasks"
    elif grouping == "domain_cost_tercile":
        cost = df["cost_usd_tested"].astype(float)
        try:
            tercile = pd.qcut(cost.rank(method="first"), 3, labels=["low", "mid", "high"])
        except ValueError:
            tercile = pd.Series(["all_costs"] * len(df), index=df.index)
        df["group"] = df["domain"].astype(str) + "_cost_" + tercile.astype(str)
    else:
        raise ValueError(grouping)
    return df.sort_values(["group", "task_id"]).reset_index(drop=True)


def run_grouping(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    grouping_specs = []
    app_tasks, app_pairs = base.appworld_pairs()
    for grouping in ["split_difficulty", "split", "difficulty", "scenario"]:
        grouping_specs.append(("AppWorld", grouping, app_tasks, app_pairs, appworld_grouped_pair, 250))
    tau_tasks, tau_pairs = base.tau_pairs()
    for grouping in ["domain", "none", "domain_cost_tercile"]:
        grouping_specs.append(("tau-bench", grouping, tau_tasks, tau_pairs, tau_grouped_pair, 1000))

    for source_name, grouping, task_rows, pairs, prepare, n_replays in grouping_specs:
        rng = np.random.default_rng(20260631)
        total_pairs = (
            min(len(pairs), args.max_pairs_per_source)
            if args.max_pairs_per_source is not None
            else len(pairs)
        )
        for idx, pair in enumerate(pairs[:total_pairs], start=1):
            pair_df = prepare(task_rows, pair, grouping)
            diffs = pair_df["diff"].to_numpy(dtype=np.float32)
            group_code, _ = base.group_codes(pair_df["group"].to_numpy())
            orders = base.random_stratified_orders(group_code, n_replays, rng)
            full_delta = float(diffs.mean())
            for budget_cap in [0.25, 0.50, 0.75]:
                cap_k = max(1, min(len(diffs), int(math.ceil(len(diffs) * budget_cap))))
                prefix = orders[:, :cap_k]
                seen = diffs[prefix]
                prefix_groups = group_code[prefix]
                coverage = base.coverage_ok(
                    prefix_groups, base.coverage_requirements(group_code, cap_k)
                )
                boot = base.bootstrap_means(seen, 200, rng)
                sample_mean = seen.mean(axis=1)
                for threshold in args.thresholds:
                    truth_accept = full_delta > threshold
                    risk_accept = (boot <= threshold).mean(axis=1)
                    risk_reject = (boot > threshold).mean(axis=1)
                    decision = np.zeros(n_replays, dtype=np.int8)
                    decision[
                        coverage
                        & (sample_mean > threshold)
                        & (risk_accept <= PRIMARY_ALPHA)
                    ] = 1
                    decision[
                        coverage
                        & (sample_mean <= threshold)
                        & (risk_reject <= PRIMARY_ALPHA)
                    ] = -1
                    row = {
                        "source": source_name,
                        "grouping": grouping,
                        "threshold": threshold,
                        "budget_cap": budget_cap,
                        "seed": 20260631,
                        "bootstrap_samples": 200,
                        "policy_label": f"{grouping} coverage bootstrap",
                        "tested_system": pair.tested,
                        "comparison_system": pair.comparison,
                        "pair_label": pair.pair_label,
                        "truth_accept": bool(truth_accept),
                        "full_delta": full_delta,
                        "group_count": int(len(np.unique(group_code))),
                    }
                    row.update(
                        summarize_vector(
                            decision,
                            truth_accept,
                            coverage,
                            full_delta,
                            threshold,
                            cap_k / len(diffs),
                        )
                    )
                    rows.append(row)
        print(f"grouping {source_name} {grouping}: {total_pairs} pairs", flush=True)

    pair_rows = pd.DataFrame(rows)
    aggregate = aggregate_grouping(pair_rows)
    return pair_rows, aggregate


def aggregate_grouping(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "grouping", "threshold", "budget_cap"]
    for key_values, group in pair_rows.groupby(keys, dropna=False):
        source, grouping, threshold, budget = key_values
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        rows.append(
            {
                "source": source,
                "grouping": grouping,
                "threshold": threshold,
                "budget_cap": budget,
                "directed_pairs": int(group["pair_label"].nunique()),
                "mean_group_count": float(group["group_count"].mean()),
                "conditional_false_accept_rate": weighted_mean(
                    neg, "false_accept_rate"
                ),
                "conditional_false_reject_rate": weighted_mean(
                    pos, "false_reject_rate"
                ),
                "coverage_ok_rate": weighted_mean(group, "coverage_ok_rate"),
                "defer_rate": weighted_mean(group, "defer_rate"),
                "defer_no_coverage_rate": weighted_mean(
                    group, "defer_no_coverage_rate"
                ),
                "defer_uncertain_rate": weighted_mean(group, "defer_uncertain_rate"),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def swe_verified_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows = pd.read_csv(ANALYSIS / "swebench_smoke" / "swebench_task_rows.csv")
    task_rows = task_rows[task_rows["split"].eq("verified")].copy()
    _, pairs = base.swe_pairs("verified")
    rows: list[dict[str, object]] = []
    repo_rows: list[dict[str, object]] = []
    for pair in pairs:
        pair_df = base.prepare_swe_pair(task_rows, pair)
        diff = pair_df["diff"].astype(float)
        repo_delta = pair_df.groupby("repo")["diff"].mean()
        full_delta = float(diff.mean())
        row = {
            "tested_system": pair.tested,
            "comparison_system": pair.comparison,
            "pair_label": pair.pair_label,
            "overlap_tasks": int(len(pair_df)),
            "repo_count": int(pair_df["repo"].nunique()),
            "full_delta": full_delta,
            "diff_sd": float(diff.std(ddof=1)),
            "diff_abs_mean": float(diff.abs().mean()),
            "win_rate": float((diff > 0).mean()),
            "loss_rate": float((diff < 0).mean()),
            "tie_rate": float((diff == 0).mean()),
            "repo_delta_sd": float(repo_delta.std(ddof=1)),
            "repo_delta_min": float(repo_delta.min()),
            "repo_delta_max": float(repo_delta.max()),
        }
        for threshold in [0.0, 0.05, 0.10]:
            row[f"abs_margin_to_{int(threshold * 100)}pp"] = abs(full_delta - threshold)
            row[f"positive_at_{int(threshold * 100)}pp"] = full_delta > threshold
        rows.append(row)
        for repo, value in repo_delta.items():
            repo_rows.append(
                {
                    "tested_system": pair.tested,
                    "comparison_system": pair.comparison,
                    "pair_label": pair.pair_label,
                    "repo": repo,
                    "repo_delta": float(value),
                    "tasks": int((pair_df["repo"] == repo).sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(repo_rows)


def write_report(
    out_dir: Path,
    stability_agg: pd.DataFrame | None,
    min_dist: pd.DataFrame | None,
    metric_intervals: pd.DataFrame | None,
    grouping_agg: pd.DataFrame | None,
    swe_diag: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Reviewer Diagnostics\n")
    lines.append(
        "These diagnostics are empirical finite-task replay checks. They do not create formal stopping guarantees.\n"
    )
    if min_dist is not None and not min_dist.empty:
        lines.append("## Seed and Bootstrap Stability\n")
        primary = min_dist[min_dist["threshold"].eq(0.0)].copy()
        primary["minimum_sufficient_budget"] = primary[
            "minimum_sufficient_budget"
        ].map(lambda x: x if isinstance(x, str) else percent(x, 0))
        primary["config_probability"] = primary["config_probability"].map(
            lambda x: percent(x, 0)
        )
        lines.append(
            markdown_table(
                primary[
                    [
                        "source",
                        "minimum_sufficient_budget",
                        "config_count",
                        "config_probability",
                        "total_configs",
                    ]
                ]
            )
        )
        lines.append(
            "\nThe table reports how often each minimum sufficient budget appears across reveal-order seeds and bootstrap sample levels. A stable row concentrates on one budget or on None.\n"
        )

    if metric_intervals is not None and not metric_intervals.empty:
        lines.append("## Metric Ranges Across Stochastic Configurations\n")
        subset = metric_intervals[
            metric_intervals["threshold"].eq(0.0)
            & metric_intervals["budget_cap"].isin([0.25, 0.75, 0.90])
            & metric_intervals["metric"].isin(
                [
                    "conditional_false_accept_rate",
                    "conditional_false_reject_rate",
                    "defer_rate",
                    "coverage_ok_rate",
                ]
            )
        ].copy()
        for col in ["mean", "q05", "q95", "min", "max"]:
            subset[col] = subset[col].map(lambda x: percent(x, 2))
        lines.append(
            markdown_table(
                subset[
                    [
                        "source",
                        "budget_cap",
                        "metric",
                        "mean",
                        "q05",
                        "q95",
                        "configs",
                    ]
                ].head(80)
            )
        )

    if grouping_agg is not None and not grouping_agg.empty:
        lines.append("\n## Grouping Sensitivity\n")
        group_min = []
        for keys, group in grouping_agg.groupby(["source", "grouping", "threshold"]):
            source, grouping, threshold = keys
            ordered = group.sort_values("budget_cap")
            mask = (
                (ordered["conditional_false_accept_rate"] <= PRIMARY_ALPHA)
                & (ordered["conditional_false_reject_rate"] <= PRIMARY_ALPHA)
                & (ordered["defer_rate"] <= PRIMARY_ETA)
            )
            row = ordered[mask].iloc[0] if mask.any() else None
            group_min.append(
                {
                    "source": source,
                    "grouping": grouping,
                    "threshold": threshold,
                    "mean_group_count": float(ordered["mean_group_count"].iloc[0]),
                    "minimum_main_grid_budget": (
                        float(row["budget_cap"]) if row is not None else math.nan
                    ),
                    "defer_rate_at_minimum": (
                        float(row["defer_rate"]) if row is not None else math.nan
                    ),
                }
            )
        group_min_df = pd.DataFrame(group_min)
        display = group_min_df[group_min_df["threshold"].eq(0.0)].copy()
        display["minimum_main_grid_budget"] = display[
            "minimum_main_grid_budget"
        ].map(lambda x: percent(x, 0))
        display["defer_rate_at_minimum"] = display["defer_rate_at_minimum"].map(
            lambda x: percent(x, 1)
        )
        display["mean_group_count"] = display["mean_group_count"].map(
            lambda x: f"{x:.1f}"
        )
        lines.append(
            markdown_table(
                display[
                    [
                        "source",
                        "grouping",
                        "mean_group_count",
                        "minimum_main_grid_budget",
                        "defer_rate_at_minimum",
                    ]
                ]
            )
        )
        group_min_df.to_csv(out_dir / "grouping_minimum_budgets.csv", index=False)

    lines.append("\n## SWE-bench Verified Full-Delta Diagnostics\n")
    diag_summary = []
    for threshold in [0.0, 0.05, 0.10]:
        margin_col = f"abs_margin_to_{int(threshold * 100)}pp"
        diag_summary.append(
            {
                "threshold": threshold,
                "pairs": len(swe_diag),
                "positive_decisions": int(swe_diag[f"positive_at_{int(threshold * 100)}pp"].sum()),
                "median_abs_margin": float(swe_diag[margin_col].median()),
                "p25_abs_margin": float(swe_diag[margin_col].quantile(0.25)),
                "p75_abs_margin": float(swe_diag[margin_col].quantile(0.75)),
                "median_diff_sd": float(swe_diag["diff_sd"].median()),
                "median_repo_delta_sd": float(swe_diag["repo_delta_sd"].median()),
            }
        )
    diag_display = pd.DataFrame(diag_summary)
    for col in [
        "median_abs_margin",
        "p25_abs_margin",
        "p75_abs_margin",
        "median_diff_sd",
        "median_repo_delta_sd",
    ]:
        diag_display[col] = diag_display[col].map(lambda x: percent(x, 1))
    lines.append(markdown_table(diag_display))
    lines.append(
        "\nSmall margins and heterogeneous repository-level effects explain why many Verified prefixes remain ambiguous even when decision error is low.\n"
    )

    lines.append("\n## Pushback Notes\n")
    lines.append(
        "- Exact paired-permutation tests answer a different null-hypothesis question from this replay. They do not directly estimate whether a partial prefix recovers the completed benchmark decision. An exact multinomial bootstrap over ternary paired differences would remove bootstrap Monte Carlo error, but the seed and bootstrap stability checks tell us whether that extra machinery changes decisions enough to matter.\n"
    )
    lines.append(
        "- Adaptive best-arm or confidence-sequence policies are valid future baselines, but adding them as primary evidence would change the paper from an empirical replay diagnostic into a sequential-testing paper. The current paper should mention this boundary rather than absorb that literature as an unfinished baseline suite.\n"
    )
    lines.append(
        "- A utility function can be written as a stakeholder-specific loss over false accepts, false rejects, coverage failures, deferrals, and cost. The paper's fixed targets are a constrained decision rule: among budgets satisfying risk and deferral constraints, choose the smallest budget. A full utility calibration would require application-specific loss ratios that are not identified by these public logs.\n"
    )
    (out_dir / "reviewer_diagnostics_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stability_pair = stability_agg = min_df = min_dist = metric_intervals = None
    if not args.skip_stability:
        if args.reuse_stability:
            stability_pair_path = args.out_dir / "stability_pair_rows.csv"
            stability_agg_path = args.out_dir / "stability_aggregate.csv"
            min_path = args.out_dir / "stability_minimum_budgets.csv"
            if not (stability_pair_path.exists() and stability_agg_path.exists() and min_path.exists()):
                raise FileNotFoundError(
                    "Cannot reuse stability outputs because one or more expected CSV files are missing."
                )
            stability_pair = pd.read_csv(stability_pair_path)
            stability_agg = pd.read_csv(stability_agg_path)
            min_df = pd.read_csv(min_path)
        else:
            stability_pair, stability_agg, min_df = run_stability(args)
            stability_pair.to_csv(args.out_dir / "stability_pair_rows.csv", index=False)
            stability_agg.to_csv(args.out_dir / "stability_aggregate.csv", index=False)
            min_df.to_csv(args.out_dir / "stability_minimum_budgets.csv", index=False)
        min_dist = minimum_budget_distribution(min_df)
        metric_intervals = metric_intervals_over_configs(stability_agg)
        min_dist.to_csv(args.out_dir / "stability_minimum_budget_distribution.csv", index=False)
        metric_intervals.to_csv(args.out_dir / "stability_metric_intervals.csv", index=False)

    grouping_pair = grouping_agg = None
    if not args.skip_grouping:
        grouping_pair, grouping_agg = run_grouping(args)
        grouping_pair.to_csv(args.out_dir / "grouping_pair_rows.csv", index=False)
        grouping_agg.to_csv(args.out_dir / "grouping_aggregate.csv", index=False)

    swe_diag, swe_repo = swe_verified_diagnostics()
    swe_diag.to_csv(args.out_dir / "swe_verified_delta_diagnostics.csv", index=False)
    swe_repo.to_csv(args.out_dir / "swe_verified_repo_delta_rows.csv", index=False)

    metadata = {
        "seeds": args.seeds,
        "bootstrap_levels": args.bootstrap_levels,
        "thresholds": args.thresholds,
        "budget_caps": args.budget_caps,
        "swe_replays": args.swe_replays,
        "appworld_replays": args.appworld_replays,
        "tau_replays": args.tau_replays,
        "primary_alpha": PRIMARY_ALPHA,
        "primary_gamma": PRIMARY_GAMMA,
        "primary_eta": PRIMARY_ETA,
        "max_pairs_per_source": args.max_pairs_per_source,
    }
    (args.out_dir / "reviewer_diagnostics_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(args.out_dir, stability_agg, min_dist, metric_intervals, grouping_agg, swe_diag)
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
