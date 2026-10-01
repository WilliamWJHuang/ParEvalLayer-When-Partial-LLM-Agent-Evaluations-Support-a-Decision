from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
ANALYSIS = ROOT / "analysis" / "public_backtest"
OUT_DIR = ANALYSIS / "sensitivity_diagnostics"
MAIN_SWEEP = ANALYSIS / "target_budget_sweep" / "main_policy_summary"
REVIEWER_DIAG = ANALYSIS / "reviewer_diagnostics"
ROBUSTNESS = ANALYSIS / "public_robustness_checks"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_main_policy_sensitivity as base  # noqa: E402


PRIMARY_ALPHA = 0.05
PRIMARY_GAMMA = 0.05
PRIMARY_ETA = 0.25


@dataclass(frozen=True)
class SourceBundle:
    key: str
    name: str
    task_rows: pd.DataFrame
    pairs: list[base.PairSpec]
    prepare: object
    replays: int
    bootstrap_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sensitivity diagnostics for budget sufficiency."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["swe_lite", "swe_verified", "appworld", "tau"],
        choices=["swe_lite", "swe_verified", "appworld", "tau", "osworld"],
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.0, 0.05, 0.10]
    )
    parser.add_argument(
        "--budget-caps", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument("--swe-replays", type=int, default=250)
    parser.add_argument("--appworld-replays", type=int, default=250)
    parser.add_argument("--tau-replays", type=int, default=1000)
    parser.add_argument("--osworld-replays", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--tau-cost-sims", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-pairs-per-source",
        type=int,
        default=None,
        help="Smoke-test limit. Do not use for paper diagnostics.",
    )
    return parser.parse_args()


def pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "None"
    return f"{100.0 * float(value):.{digits}f}%"


def budget(value: object) -> str:
    if value is None or pd.isna(value):
        return "NF"
    return f"{100.0 * float(value):.0f}%"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy().fillna("None")
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


def load_source(key: str, args: argparse.Namespace) -> SourceBundle:
    if key == "swe_lite":
        task_rows, pairs = base.swe_pairs("lite")
        return SourceBundle(
            key, "SWE-bench Lite", task_rows, pairs, base.prepare_swe_pair,
            args.swe_replays, args.bootstrap_samples
        )
    if key == "swe_verified":
        task_rows, pairs = base.swe_pairs("verified")
        return SourceBundle(
            key, "SWE-bench Verified", task_rows, pairs, base.prepare_swe_pair,
            args.swe_replays, args.bootstrap_samples
        )
    if key == "appworld":
        task_rows, pairs = base.appworld_pairs()
        return SourceBundle(
            key, "AppWorld", task_rows, pairs, base.prepare_appworld_pair,
            args.appworld_replays, args.bootstrap_samples
        )
    if key == "tau":
        task_rows, pairs = base.tau_pairs()
        return SourceBundle(
            key, "tau-bench", task_rows, pairs, base.prepare_tau_pair,
            args.tau_replays, max(args.bootstrap_samples, 500)
        )
    if key == "osworld":
        task_rows, pairs = base.osworld_pairs()
        return SourceBundle(
            key, "OSWorld-Verified", task_rows, pairs, base.prepare_osworld_pair,
            args.osworld_replays, args.bootstrap_samples
        )
    raise ValueError(key)


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return float("nan")
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((weights * values).sum() / total)


def summarize_decisions(
    decision: np.ndarray,
    truth_accept: bool,
    coverage: np.ndarray,
    task_fraction: float,
    cost_fraction: np.ndarray | None = None,
) -> dict[str, float | int]:
    accept = decision == 1
    reject = decision == -1
    defer = decision == 0
    false_accept = accept & (not truth_accept)
    false_reject = reject & truth_accept
    row: dict[str, float | int] = {
        "rows": int(len(decision)),
        "mean_task_fraction": float(task_fraction),
        "accept_rate": float(accept.mean()),
        "reject_rate": float(reject.mean()),
        "defer_rate": float(defer.mean()),
        "false_accept_rate": float(false_accept.mean()),
        "false_reject_rate": float(false_reject.mean()),
        "decision_error_rate": float((false_accept | false_reject).mean()),
        "coverage_failure_rate": float(((decision != 0) & (~coverage)).mean()),
        "coverage_ok_rate": float(coverage.mean()),
        "defer_no_coverage_rate": float((defer & (~coverage)).mean()),
        "defer_uncertain_rate": float((defer & coverage).mean()),
    }
    if cost_fraction is not None:
        row["mean_cost_fraction"] = float(np.nanmean(cost_fraction))
    return row


def aggregate_pair_rows(
    rows: pd.DataFrame, extra_keys: list[str] | None = None
) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    keys = [*(extra_keys or []), "source", "threshold", "budget_cap", "policy"]
    for values, group in rows.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        value_map = dict(zip(keys, values))
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        row = {
            **value_map,
            "directed_queries": int(group["pair_label"].nunique()),
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
            "defer_no_coverage_rate": weighted_mean(group, "defer_no_coverage_rate"),
            "defer_uncertain_rate": weighted_mean(group, "defer_uncertain_rate"),
        }
        if "mean_cost_fraction" in group.columns:
            row["mean_cost_fraction"] = weighted_mean(group, "mean_cost_fraction")
        out.append(row)
    return pd.DataFrame(out).sort_values(keys)


def minimum_budget(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for values, group in aggregate.groupby(["source", "threshold", "policy"]):
        source, threshold, policy = values
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
                "policy": policy,
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
    return pd.DataFrame(rows).sort_values(["source", "threshold", "policy"])


def allocate_from_weights(
    group_code: np.ndarray,
    cap_k: int,
    weights: dict[int, float],
    minimum: dict[int, int],
) -> dict[int, int]:
    groups = sorted(int(item) for item in np.unique(group_code))
    sizes = {idx: int((group_code == idx).sum()) for idx in groups}
    allocation = {idx: min(sizes[idx], int(minimum.get(idx, 0))) for idx in groups}
    remaining = cap_k - sum(allocation.values())
    if remaining < 0:
        # The requested floor is infeasible. Keep the largest groups first.
        ordered = sorted(groups, key=lambda idx: sizes[idx], reverse=True)
        allocation = {idx: 0 for idx in groups}
        for idx in ordered:
            if sum(allocation.values()) >= cap_k:
                break
            allocation[idx] = 1
        return allocation

    while remaining > 0:
        open_groups = [idx for idx in groups if allocation[idx] < sizes[idx]]
        if not open_groups:
            break
        total_weight = sum(max(weights.get(idx, 0.0), 0.0) for idx in open_groups)
        if total_weight <= 0:
            total_weight = sum(sizes[idx] - allocation[idx] for idx in open_groups)
            raw = {
                idx: remaining * (sizes[idx] - allocation[idx]) / total_weight
                for idx in open_groups
            }
        else:
            raw = {
                idx: remaining * max(weights.get(idx, 0.0), 0.0) / total_weight
                for idx in open_groups
            }
        added = 0
        fractional: list[tuple[float, int]] = []
        for idx in open_groups:
            room = sizes[idx] - allocation[idx]
            inc = min(room, int(math.floor(raw[idx])))
            allocation[idx] += inc
            added += inc
            fractional.append((raw[idx] - math.floor(raw[idx]), idx))
        remaining -= added
        if remaining <= 0:
            break
        for _, idx in sorted(fractional, reverse=True):
            if remaining <= 0:
                break
            if allocation[idx] < sizes[idx]:
                allocation[idx] += 1
                remaining -= 1
    return allocation


def neyman_prefixes(
    diffs: np.ndarray,
    group_code: np.ndarray,
    cap_k: int,
    n_replays: int,
    rng: np.random.Generator,
) -> np.ndarray:
    requirements = base.coverage_requirements(group_code, cap_k)
    groups = sorted(int(item) for item in np.unique(group_code))
    weights: dict[int, float] = {}
    for idx in groups:
        values = diffs[group_code == idx]
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        weights[idx] = len(values) * sd
    if sum(weights.values()) <= 0:
        weights = {idx: float((group_code == idx).sum()) for idx in groups}
    allocation = allocate_from_weights(group_code, cap_k, weights, requirements)
    prefixes = np.empty((n_replays, cap_k), dtype=np.int32)
    by_group = {idx: np.where(group_code == idx)[0] for idx in groups}
    for replay_idx in range(n_replays):
        selected: list[np.ndarray] = []
        for idx in groups:
            count = allocation.get(idx, 0)
            if count <= 0:
                continue
            selected.append(rng.choice(by_group[idx], size=count, replace=False))
        prefix = np.concatenate(selected)
        rng.shuffle(prefix)
        prefixes[replay_idx] = prefix.astype(np.int32)
    return prefixes


def stratified_prefixes(
    group_code: np.ndarray,
    cap_k: int,
    n_replays: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return base.random_stratified_orders(group_code, n_replays, rng)[:, :cap_k]


def serfling_radius(n_tasks: int, observed: int, alpha: float) -> float:
    if observed <= 0:
        return float("inf")
    if n_tasks <= 1:
        return 0.0
    finite_correction = max(0.0, (n_tasks - observed) / (n_tasks - 1))
    return 2.0 * math.sqrt(finite_correction * math.log(2.0 / alpha) / (2.0 * observed))


def run_policy_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for key in args.sources:
        source = load_source(key, args)
        pairs = source.pairs
        if args.max_pairs_per_source is not None:
            pairs = pairs[: args.max_pairs_per_source]
        for idx, pair in enumerate(pairs, start=1):
            pair_df = source.prepare(source.task_rows, pair)
            diffs = pair_df["diff"].to_numpy(dtype=np.float32)
            group_code, _ = base.group_codes(pair_df[pair.group_column].to_numpy())
            full_delta = float(diffs.mean())
            n_tasks = len(diffs)
            for budget_cap in args.budget_caps:
                cap_k = max(1, min(n_tasks, int(math.ceil(n_tasks * budget_cap))))
                prefixes_by_policy = {
                    "coverage bootstrap": stratified_prefixes(
                        group_code, cap_k, source.replays, rng
                    ),
                    "Neyman coverage bootstrap": neyman_prefixes(
                        diffs, group_code, cap_k, source.replays, rng
                    ),
                }
                requirements = base.coverage_requirements(group_code, cap_k)
                radius = serfling_radius(n_tasks, cap_k, PRIMARY_ALPHA)
                for policy, prefixes in prefixes_by_policy.items():
                    seen = diffs[prefixes]
                    prefix_groups = group_code[prefixes]
                    coverage = base.coverage_ok(prefix_groups, requirements)
                    means = seen.mean(axis=1)
                    boot = base.bootstrap_means(seen, source.bootstrap_samples, rng)
                    for threshold in args.thresholds:
                        truth_accept = full_delta > threshold
                        risk_accept = (boot <= threshold).mean(axis=1)
                        risk_reject = (boot > threshold).mean(axis=1)
                        decision = np.zeros(source.replays, dtype=np.int8)
                        accept = (
                            coverage
                            & (means > threshold)
                            & (risk_accept <= PRIMARY_ALPHA)
                        )
                        reject = (
                            coverage
                            & (means <= threshold)
                            & (risk_reject <= PRIMARY_ALPHA)
                        )
                        decision[accept] = 1
                        decision[reject] = -1
                        row = {
                            "source": source.name,
                            "tested_system": pair.tested,
                            "comparison_system": pair.comparison,
                            "pair_label": pair.pair_label,
                            "threshold": threshold,
                            "budget_cap": budget_cap,
                            "policy": policy,
                            "truth_accept": bool(truth_accept),
                            "full_delta": full_delta,
                            "observed_tasks": cap_k,
                            "overlap_tasks": n_tasks,
                        }
                        row.update(
                            summarize_decisions(
                                decision, truth_accept, coverage, cap_k / n_tasks
                            )
                        )
                        rows.append(row)

                prefixes = prefixes_by_policy["coverage bootstrap"]
                seen = diffs[prefixes]
                prefix_groups = group_code[prefixes]
                coverage = base.coverage_ok(prefix_groups, requirements)
                means = seen.mean(axis=1)
                for threshold in args.thresholds:
                    truth_accept = full_delta > threshold
                    decision = np.zeros(source.replays, dtype=np.int8)
                    decision[coverage & ((means - radius) > threshold)] = 1
                    decision[coverage & ((means + radius) <= threshold)] = -1
                    row = {
                        "source": source.name,
                        "tested_system": pair.tested,
                        "comparison_system": pair.comparison,
                        "pair_label": pair.pair_label,
                        "threshold": threshold,
                        "budget_cap": budget_cap,
                        "policy": "Serfling coverage policy",
                        "truth_accept": bool(truth_accept),
                        "full_delta": full_delta,
                        "observed_tasks": cap_k,
                        "overlap_tasks": n_tasks,
                        "serfling_radius": radius,
                    }
                    row.update(
                        summarize_decisions(
                            decision, truth_accept, coverage, cap_k / n_tasks
                        )
                    )
                    rows.append(row)
            if idx % 50 == 0 or idx == len(pairs):
                print(f"{source.name}: {idx}/{len(pairs)} pairs", flush=True)
    pair_rows = pd.DataFrame(rows)
    aggregate = aggregate_pair_rows(pair_rows)
    min_budget = minimum_budget(aggregate)
    return pair_rows, aggregate, min_budget


def load_alpha_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_path = MAIN_SWEEP / "main_policy_aggregate.csv"
    min_path = MAIN_SWEEP / "main_policy_minimum_budgets.csv"
    if not agg_path.exists() or not min_path.exists():
        raise FileNotFoundError("Run run_main_policy_sensitivity.py and summarize_main_policy_sensitivity.py first.")
    aggregate = pd.read_csv(agg_path)
    minimum = pd.read_csv(min_path)
    return aggregate, minimum


def seed_bands() -> pd.DataFrame:
    path = REVIEWER_DIAG / "stability_aggregate.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "bootstrap_samples" in df.columns:
        df = df[df["bootstrap_samples"].eq(200)].copy()
    metrics = [
        "conditional_false_accept_rate",
        "conditional_false_reject_rate",
        "defer_rate",
        "coverage_ok_rate",
    ]
    rows: list[dict[str, object]] = []
    for values, group in df.groupby(["source", "threshold", "budget_cap", "policy_label"]):
        source, threshold, budget_cap, policy = values
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            seed_count = int(len(vals))
            std = float(np.std(vals, ddof=1)) if seed_count > 1 else 0.0
            # Two-sided 95% t multipliers for the small seed counts used here.
            t95 = {
                2: 12.706,
                3: 4.303,
                4: 3.182,
                5: 2.776,
                6: 2.571,
                7: 2.447,
                8: 2.365,
                9: 2.306,
                10: 2.262,
            }.get(seed_count, 1.96)
            half_width = t95 * std / math.sqrt(seed_count) if seed_count > 1 else 0.0
            mean = float(np.mean(vals))
            rows.append(
                {
                    "source": source,
                    "threshold": threshold,
                    "budget_cap": budget_cap,
                    "policy_label": policy,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_low": max(0.0, mean - half_width),
                    "ci95_high": min(1.0, mean + half_width),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "seed_count": seed_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["source", "threshold", "budget_cap", "metric"])


def load_orientation() -> pd.DataFrame:
    path = ROBUSTNESS / "undirected_pair_collapse.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def make_swe_difficulty_groups() -> tuple[pd.DataFrame, list[base.PairSpec]]:
    task_rows = pd.read_csv(ANALYSIS / "swebench_smoke" / "swebench_task_rows.csv")
    scored_frames = []
    for split, frame in task_rows.groupby("split"):
        difficulty = frame.groupby("instance_id")["resolved"].mean().rename("pass_rate")
        temp = frame.merge(difficulty, on="instance_id", how="left")
        rank = temp[["instance_id", "pass_rate"]].drop_duplicates().copy()
        rank["difficulty_proxy"] = pd.qcut(
            rank["pass_rate"].rank(method="first"),
            3,
            labels=["hard", "mid", "easy"],
        ).astype(str)
        scored_frames.append(temp.drop(columns=["pass_rate"]).merge(
            rank[["instance_id", "difficulty_proxy"]], on="instance_id", how="left"
        ))
    task_rows = pd.concat(scored_frames, ignore_index=True)
    pairs: list[base.PairSpec] = []
    for split in ["lite", "verified"]:
        _, split_pairs = base.swe_pairs(split)
        for pair in split_pairs:
            pairs.append(
                base.PairSpec(
                    source="SWE-bench Lite" if split == "lite" else "SWE-bench Verified",
                    tested=pair.tested,
                    comparison=pair.comparison,
                    pair_label=pair.pair_label,
                    group_column="repo_difficulty",
                )
            )
    return task_rows, pairs


def prepare_swe_repo_difficulty(task_rows: pd.DataFrame, pair: base.PairSpec) -> pd.DataFrame:
    tested = task_rows[task_rows["submission"].eq(pair.tested)].copy()
    comparison = task_rows[task_rows["submission"].eq(pair.comparison)].copy()
    merged = tested.merge(
        comparison,
        on="instance_id",
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    merged["diff"] = merged["resolved_tested"].astype(float) - merged[
        "resolved_comparison"
    ].astype(float)
    merged["repo_difficulty"] = (
        merged["repo_tested"].astype(str)
        + "_"
        + merged["difficulty_proxy_tested"].astype(str)
    )
    return merged.sort_values(["repo_difficulty", "instance_id"]).reset_index(drop=True)


def min_abs_requirements(group_code: np.ndarray, cap_k: int, minimum_count: int) -> dict[int, int]:
    groups = sorted(int(item) for item in np.unique(group_code))
    req = {
        idx: min(int((group_code == idx).sum()), minimum_count)
        for idx in groups
    }
    if sum(req.values()) <= cap_k:
        return req
    # Infeasible floor, mark each group as required. The coverage check will fail.
    return req


def coverage_sensitivity(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed + 99)
    bundles = [load_source(key, args) for key in args.sources]
    swe_task_rows, swe_pairs_all = make_swe_difficulty_groups()
    if "swe_lite" in args.sources or "swe_verified" in args.sources:
        selected = []
        for pair in swe_pairs_all:
            if pair.source == "SWE-bench Lite" and "swe_lite" in args.sources:
                selected.append(pair)
            if pair.source == "SWE-bench Verified" and "swe_verified" in args.sources:
                selected.append(pair)
        bundles.append(
            SourceBundle(
                "swe_repo_difficulty",
                "SWE repo x difficulty proxy",
                swe_task_rows,
                selected,
                prepare_swe_repo_difficulty,
                args.swe_replays,
                args.bootstrap_samples,
            )
        )
    floors = ["proportional", "min1", "min2"]
    for bundle in bundles:
        pairs = bundle.pairs
        if args.max_pairs_per_source is not None:
            pairs = pairs[: args.max_pairs_per_source]
        for pair in pairs:
            pair_df = bundle.prepare(bundle.task_rows, pair)
            diffs = pair_df["diff"].to_numpy(dtype=np.float32)
            group_col = pair.group_column
            group_code, _ = base.group_codes(pair_df[group_col].to_numpy())
            full_delta = float(diffs.mean())
            n_tasks = len(diffs)
            for budget_cap in args.budget_caps:
                cap_k = max(1, min(n_tasks, int(math.ceil(n_tasks * budget_cap))))
                prefixes = stratified_prefixes(group_code, cap_k, bundle.replays, rng)
                seen = diffs[prefixes]
                prefix_groups = group_code[prefixes]
                means = seen.mean(axis=1)
                boot = base.bootstrap_means(seen, bundle.bootstrap_samples, rng)
                for floor in floors:
                    if floor == "proportional":
                        req = base.coverage_requirements(group_code, cap_k)
                    elif floor == "min1":
                        req = min_abs_requirements(group_code, cap_k, 1)
                    elif floor == "min2":
                        req = min_abs_requirements(group_code, cap_k, 2)
                    else:
                        raise ValueError(floor)
                    coverage = base.coverage_ok(prefix_groups, req)
                    for threshold in args.thresholds:
                        truth_accept = full_delta > threshold
                        risk_accept = (boot <= threshold).mean(axis=1)
                        risk_reject = (boot > threshold).mean(axis=1)
                        decision = np.zeros(bundle.replays, dtype=np.int8)
                        accept = (
                            coverage
                            & (means > threshold)
                            & (risk_accept <= PRIMARY_ALPHA)
                        )
                        reject = (
                            coverage
                            & (means <= threshold)
                            & (risk_reject <= PRIMARY_ALPHA)
                        )
                        decision[accept] = 1
                        decision[reject] = -1
                        row = {
                            "source": bundle.name,
                            "tested_system": pair.tested,
                            "comparison_system": pair.comparison,
                            "pair_label": pair.pair_label,
                            "grouping": group_col,
                            "coverage_floor": floor,
                            "threshold": threshold,
                            "budget_cap": budget_cap,
                            "policy": "coverage bootstrap",
                            "truth_accept": bool(truth_accept),
                            "full_delta": full_delta,
                            "observed_tasks": cap_k,
                            "overlap_tasks": n_tasks,
                            "group_count": int(len(np.unique(group_code))),
                        }
                        row.update(
                            summarize_decisions(
                                decision, truth_accept, coverage, cap_k / n_tasks
                            )
                        )
                        rows.append(row)
    pair_rows = pd.DataFrame(rows)
    keys = ["source", "grouping", "coverage_floor", "threshold", "budget_cap", "policy"]
    out: list[dict[str, object]] = []
    for values, group in pair_rows.groupby(keys, dropna=False):
        source, grouping, floor, threshold, budget_cap, policy = values
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        out.append(
            {
                "source": source,
                "grouping": grouping,
                "coverage_floor": floor,
                "threshold": threshold,
                "budget_cap": budget_cap,
                "policy": policy,
                "mean_group_count": float(group["group_count"].mean()),
                "conditional_false_accept_rate": weighted_mean(
                    neg, "false_accept_rate"
                ),
                "conditional_false_reject_rate": weighted_mean(
                    pos, "false_reject_rate"
                ),
                "coverage_failure_rate": weighted_mean(group, "coverage_failure_rate"),
                "defer_rate": weighted_mean(group, "defer_rate"),
                "coverage_ok_rate": weighted_mean(group, "coverage_ok_rate"),
            }
        )
    aggregate = pd.DataFrame(out).sort_values(keys)
    return pair_rows, aggregate


def prepare_tau_cost_pair(task_rows: pd.DataFrame, pair: base.PairSpec) -> pd.DataFrame:
    tested = task_rows[task_rows["submission_dir"].eq(pair.tested)].copy()
    comparison = task_rows[task_rows["submission_dir"].eq(pair.comparison)].copy()
    keep = ["domain", "task_id", "task_key", "outcome_mean", "cost_usd"]
    merged = tested[keep].merge(
        comparison[keep],
        on=["domain", "task_id", "task_key"],
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    merged["diff"] = merged["outcome_mean_tested"] - merged["outcome_mean_comparison"]
    merged["cost"] = merged["cost_usd_tested"].astype(float)
    return merged.sort_values(["domain", "task_id"]).reset_index(drop=True)


def cost_prefixes(costs: np.ndarray, n_replays: int, rng: np.random.Generator) -> np.ndarray:
    jitter = rng.random((n_replays, len(costs))) * 1e-9
    return np.argsort(costs[None, :] + jitter, axis=1).astype(np.int32)


def tau_cost_stress(args: argparse.Namespace) -> pd.DataFrame:
    task_rows, pairs = base.tau_pairs()
    rng = np.random.default_rng(args.seed + 123)
    rows: list[dict[str, object]] = []
    for sim_idx in range(args.tau_cost_sims + 1):
        scenario = "observed" if sim_idx == 0 else "within-domain cost resample"
        for pair in pairs:
            pair_df = prepare_tau_cost_pair(task_rows, pair)
            diffs = pair_df["diff"].to_numpy(dtype=np.float32)
            group_code, _ = base.group_codes(pair_df["domain"].to_numpy())
            costs = pair_df["cost"].to_numpy(dtype=float)
            if sim_idx > 0:
                sim_costs = costs.copy()
                for domain in pair_df["domain"].unique():
                    mask = pair_df["domain"].eq(domain).to_numpy()
                    sim_costs[mask] = rng.choice(costs[mask], size=int(mask.sum()), replace=True)
                costs = sim_costs
            costs = np.nan_to_num(costs, nan=float(np.nanmedian(costs)))
            total_cost = float(costs.sum())
            full_delta = float(diffs.mean())
            for budget_cap in args.budget_caps:
                cap_k = max(1, min(len(diffs), int(math.ceil(len(diffs) * budget_cap))))
                orders = cost_prefixes(costs, args.tau_replays, rng)
                prefixes = orders[:, :cap_k]
                seen = diffs[prefixes]
                prefix_groups = group_code[prefixes]
                requirements = base.coverage_requirements(group_code, cap_k)
                coverage = base.coverage_ok(prefix_groups, requirements)
                means = seen.mean(axis=1)
                cost_fraction = costs[prefixes].sum(axis=1) / total_cost
                for threshold in args.thresholds:
                    truth_accept = full_delta > threshold
                    decision = np.where(means > threshold, 1, -1).astype(np.int8)
                    row = {
                        "source": "tau-bench",
                        "scenario": scenario,
                        "simulation_index": sim_idx,
                        "tested_system": pair.tested,
                        "comparison_system": pair.comparison,
                        "pair_label": pair.pair_label,
                        "threshold": threshold,
                        "budget_cap": budget_cap,
                        "policy": "cost-aware forced",
                        "truth_accept": bool(truth_accept),
                        "full_delta": full_delta,
                    }
                    row.update(
                        summarize_decisions(
                            decision,
                            truth_accept,
                            coverage,
                            cap_k / len(diffs),
                            cost_fraction,
                        )
                    )
                    rows.append(row)
    pair_rows = pd.DataFrame(rows)
    agg = aggregate_pair_rows(pair_rows, extra_keys=["scenario", "simulation_index"])
    sim_rows: list[dict[str, object]] = []
    for values, group in agg.groupby(["scenario", "threshold", "budget_cap", "policy"]):
        scenario, threshold, budget_cap, policy = values
        sim_rows.append(
            {
                "scenario": scenario,
                "threshold": threshold,
                "budget_cap": budget_cap,
                "policy": policy,
                "mean_false_accept": float(group["conditional_false_accept_rate"].mean()),
                "mean_false_reject": float(group["conditional_false_reject_rate"].mean()),
                "mean_coverage_failure": float(group["coverage_failure_rate"].mean()),
                "mean_cost_fraction": float(group["mean_cost_fraction"].mean()),
                "q05_coverage_failure": float(group["coverage_failure_rate"].quantile(0.05)),
                "q95_coverage_failure": float(group["coverage_failure_rate"].quantile(0.95)),
                "simulation_rows": int(len(group)),
            }
        )
    return pd.DataFrame(sim_rows).sort_values(["scenario", "threshold", "budget_cap"])


def write_report(
    out_dir: Path,
    alpha_min: pd.DataFrame,
    policy_min: pd.DataFrame,
    seed: pd.DataFrame,
    orientation: pd.DataFrame,
    coverage: pd.DataFrame,
    tau_cost: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Sensitivity Diagnostics")
    lines.append("")
    lines.append("All analyses use completed public task-level records. Percentages are empirical replay rates.")
    lines.append("")

    display = alpha_min[
        alpha_min["policy_alpha"].isin([0.02, 0.05, 0.10])
        & alpha_min["threshold"].isin([0.0, 0.05, 0.10])
    ].copy()
    display["threshold"] = display["threshold"].map(lambda x: f"{100*x:.0f} pp")
    display["policy_alpha"] = display["policy_alpha"].map(lambda x: f"{x:.2f}")
    display["minimum_sufficient_budget"] = display["minimum_sufficient_budget"].map(budget)
    display["defer_rate_at_minimum"] = display["defer_rate_at_minimum"].map(pct)
    lines.extend([
        "## Bootstrap Tail-Mass Operating Curves",
        "",
        "The policy uses the same value for bootstrap tail mass and the FAR/FRR operating target in the main setting, a = alpha = 0.05. The sweep below changes a inside the policy and rechecks the same decision-error, coverage, and defer targets.",
        "",
        markdown_table(display[[
            "source", "threshold", "policy_alpha", "minimum_sufficient_budget",
            "defer_rate_at_minimum"
        ]]),
        "",
    ])

    display = policy_min.copy()
    display["threshold"] = display["threshold"].map(lambda x: f"{100*x:.0f} pp")
    display["minimum_sufficient_budget"] = display["minimum_sufficient_budget"].map(budget)
    display["defer_rate_at_minimum"] = display["defer_rate_at_minimum"].map(pct)
    lines.extend([
        "## Policy Baselines",
        "",
        "Neyman allocation first satisfies the no-slack group floor, then allocates the remaining budget in proportion to N_g S_g using full-record group variances. This oracle-style diagnostic estimates whether variance-aware allocation could lower budgets under the same coverage rule. The Serfling policy uses a finite-population Hoeffding-Serfling radius on the same stratified task subsets.",
        "",
        markdown_table(display[[
            "source", "threshold", "policy", "minimum_sufficient_budget",
            "defer_rate_at_minimum"
        ]]),
        "",
    ])

    if not seed.empty:
        focus = seed[
            seed["threshold"].eq(0.0)
            & seed["budget_cap"].isin([0.25, 0.75, 0.90])
            & seed["metric"].isin([
                "conditional_false_accept_rate",
                "conditional_false_reject_rate",
                "defer_rate",
            ])
        ].copy()
        focus["budget_cap"] = focus["budget_cap"].map(budget)
        for col in ["mean", "ci95_low", "ci95_high", "min", "max"]:
            focus[col] = focus[col].map(pct)
        lines.extend([
            "## Reveal-Order Seed Bands",
            "",
            "These are Monte Carlo seed bands over reveal orders with bootstrap_samples = 200, not population confidence intervals. The 95% intervals use a t interval over the three reveal-order seeds.",
            "",
            markdown_table(focus[[
                "source", "budget_cap", "metric", "mean", "ci95_low", "ci95_high",
                "min", "max", "seed_count"
            ]].head(80)),
            "",
        ])

    if not orientation.empty:
        orient = orientation.copy()
        orient["threshold"] = orient["threshold"].map(lambda x: f"{100*float(x):.0f} pp")
        orient["minimum_sufficient_budget"] = orient["minimum_sufficient_budget"].map(budget)
        lines.extend([
            "## Directed-to-Undirected Collapse",
            "",
            "The collapse keeps one orientation per unordered system pair, oriented toward the larger all-task delta. It changes support counts and some budgets, especially AppWorld and tau-bench positive-margin settings.",
            "",
            markdown_table(orient[[
                "source", "threshold", "undirected_pairs", "positive_decisions",
                "negative_decisions", "minimum_sufficient_budget", "observed_defer_rate"
            ]]),
            "",
        ])

    cov_focus = coverage[
        coverage["threshold"].eq(0.0)
        & coverage["budget_cap"].isin([0.25, 0.50, 0.75])
    ].copy()
    cov_focus["budget_cap"] = cov_focus["budget_cap"].map(budget)
    for col in [
        "conditional_false_accept_rate",
        "conditional_false_reject_rate",
        "defer_rate",
        "coverage_ok_rate",
    ]:
        cov_focus[col] = cov_focus[col].map(pct)
    lines.extend([
        "## Grouping and Coverage-Floor Sensitivity",
        "",
        "SWE repo x difficulty uses a public difficulty proxy from task pass rates across released systems. It is a diagnostic grouping, not a benchmark-provided difficulty label.",
        "",
        markdown_table(cov_focus[[
            "source", "grouping", "coverage_floor", "budget_cap",
            "mean_group_count", "defer_rate", "coverage_ok_rate"
        ]].head(120)),
        "",
    ])

    tau_display = tau_cost[
        tau_cost["budget_cap"].eq(0.25)
        & tau_cost["threshold"].isin([0.0, 0.05, 0.10])
    ].copy()
    for col in [
        "mean_false_accept",
        "mean_false_reject",
        "mean_coverage_failure",
        "mean_cost_fraction",
        "q05_coverage_failure",
        "q95_coverage_failure",
    ]:
        tau_display[col] = tau_display[col].map(pct)
    tau_display["threshold"] = tau_display["threshold"].map(lambda x: f"{100*x:.0f} pp")
    lines.extend([
        "## tau-bench Cost Stress",
        "",
        "The semi-simulated rows resample observed task costs within each domain and rerun cheap-first ordering. Outcomes are not simulated, only cost order is perturbed.",
        "",
        markdown_table(tau_display[[
            "scenario", "threshold", "budget_cap", "mean_cost_fraction",
            "mean_coverage_failure", "q05_coverage_failure", "q95_coverage_failure",
            "mean_false_accept", "mean_false_reject"
        ]]),
        "",
    ])

    (out_dir / "sensitivity_diagnostics_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    alpha_agg, alpha_min = load_alpha_curves()
    alpha_agg.to_csv(args.out_dir / "bootstrap_tail_alpha_operating_curves.csv", index=False)
    alpha_min.to_csv(args.out_dir / "bootstrap_tail_alpha_minimum_budgets.csv", index=False)

    pair_rows, policy_aggregate, policy_min = run_policy_comparison(args)
    pair_rows.to_csv(args.out_dir / "policy_baseline_pair_rows.csv", index=False)
    policy_aggregate.to_csv(args.out_dir / "policy_baseline_aggregate.csv", index=False)
    policy_min.to_csv(args.out_dir / "policy_baseline_minimum_budgets.csv", index=False)

    seed = seed_bands()
    seed.to_csv(args.out_dir / "reveal_order_seed_metric_bands.csv", index=False)

    orientation = load_orientation()
    orientation.to_csv(args.out_dir / "undirected_orientation_collapse.csv", index=False)

    coverage_pair, coverage_agg = coverage_sensitivity(args)
    coverage_pair.to_csv(args.out_dir / "grouping_coverage_pair_rows.csv", index=False)
    coverage_agg.to_csv(args.out_dir / "grouping_coverage_aggregate.csv", index=False)

    tau_cost = tau_cost_stress(args)
    tau_cost.to_csv(args.out_dir / "tau_cost_resampling_stress.csv", index=False)

    metadata = {
        "thresholds": args.thresholds,
        "budget_caps": args.budget_caps,
        "primary_alpha": PRIMARY_ALPHA,
        "primary_gamma": PRIMARY_GAMMA,
        "primary_eta": PRIMARY_ETA,
        "seed": args.seed,
        "swe_replays": args.swe_replays,
        "appworld_replays": args.appworld_replays,
        "tau_replays": args.tau_replays,
        "bootstrap_samples": args.bootstrap_samples,
        "tau_cost_sims": args.tau_cost_sims,
        "sources": args.sources,
    }
    (args.out_dir / "sensitivity_diagnostics_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(args.out_dir, alpha_min, policy_min, seed, orientation, coverage_agg, tau_cost)
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
