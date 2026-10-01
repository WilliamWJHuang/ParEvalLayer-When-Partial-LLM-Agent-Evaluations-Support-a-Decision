from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "public_backtest"
OUT_DIR = ANALYSIS / "target_budget_sweep" / "main_policy"


@dataclass(frozen=True)
class PairSpec:
    source: str
    tested: str
    comparison: str
    pair_label: str
    group_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fine-grained main-policy budget and alpha sensitivity."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["swe_lite", "swe_verified", "appworld", "tau"],
        choices=["swe_lite", "swe_verified", "appworld", "tau", "osworld"],
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.00, 0.05, 0.10],
    )
    parser.add_argument(
        "--budget-caps",
        type=float,
        nargs="+",
        default=[round(x / 100, 2) for x in range(5, 100, 5)],
    )
    parser.add_argument("--swe-replays", type=int, default=500)
    parser.add_argument("--appworld-replays", type=int, default=500)
    parser.add_argument("--tau-replays", type=int, default=2000)
    parser.add_argument("--osworld-replays", type=int, default=500)
    parser.add_argument("--swe-bootstrap-samples", type=int, default=200)
    parser.add_argument("--appworld-bootstrap-samples", type=int, default=200)
    parser.add_argument("--tau-bootstrap-samples", type=int, default=500)
    parser.add_argument("--osworld-bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-pairs-per-source",
        type=int,
        default=None,
        help="Optional smoke-test limit. Leave unset for paper runs.",
    )
    return parser.parse_args()


def random_stratified_orders(
    group_code: np.ndarray,
    n_replays: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = sorted(int(item) for item in np.unique(group_code))
    by_group = [np.where(group_code == idx)[0] for idx in groups]
    orders = np.empty((n_replays, len(group_code)), dtype=np.int32)
    for replay_idx in range(n_replays):
        scores = np.empty(len(group_code), dtype=float)
        for indices in by_group:
            permuted = rng.permutation(indices)
            local_scores = (np.arange(len(permuted)) + rng.random(len(permuted))) / len(
                permuted
            )
            scores[permuted] = local_scores
        orders[replay_idx] = np.argsort(scores).astype(np.int32)
    return orders


def group_codes(groups: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(str(item) for item in np.unique(groups))
    mapping = {name: idx for idx, name in enumerate(unique)}
    codes = np.array([mapping[str(item)] for item in groups], dtype=np.int16)
    return codes, mapping


def coverage_requirements(group_code: np.ndarray, cap_k: int) -> dict[int, int]:
    n_tasks = len(group_code)
    out: dict[int, int] = {}
    for idx in sorted(int(item) for item in np.unique(group_code)):
        group_size = int((group_code == idx).sum())
        expected = cap_k * (group_size / n_tasks)
        out[idx] = max(1, min(group_size, math.floor(expected)))
    return out


def coverage_ok(prefix_groups: np.ndarray, requirements: dict[int, int]) -> np.ndarray:
    ok = np.ones(prefix_groups.shape[0], dtype=bool)
    for idx, required in requirements.items():
        ok &= (prefix_groups == idx).sum(axis=1) >= required
    return ok


def bootstrap_means(
    seen: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    _, k = seen.shape
    probs = np.full(k, 1.0 / k, dtype=np.float64)
    counts = rng.multinomial(k, probs, size=bootstrap_samples).astype(np.float32)
    return (seen.astype(np.float32, copy=False) @ counts.T) / np.float32(k)


def summarize_decision_vector(
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
    defer_no_coverage = defer & (~coverage)
    defer_uncertain = defer & coverage
    return {
        "rows": int(len(decision)),
        "mean_task_fraction": task_fraction,
        "accept_rate": float(accept.mean()),
        "reject_rate": float(reject.mean()),
        "defer_rate": float(defer.mean()),
        "false_accept_rate": float(false_accept.mean()),
        "false_reject_rate": float(false_reject.mean()),
        "decision_error_rate": float((false_accept | false_reject).mean()),
        "coverage_failure_rate": 0.0,
        "coverage_ok_rate": float(coverage.mean()),
        "defer_no_coverage_rate": float(defer_no_coverage.mean()),
        "defer_uncertain_rate": float(defer_uncertain.mean()),
        "absolute_margin": float(abs(full_delta - threshold)),
        "signed_margin": float(full_delta - threshold),
    }


def run_pair(
    pair: PairSpec,
    pair_df: pd.DataFrame,
    alphas: list[float],
    thresholds: list[float],
    budget_caps: list[float],
    n_replays: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    diffs = pair_df["diff"].to_numpy(dtype=np.float32)
    group_code, group_mapping = group_codes(pair_df[pair.group_column].to_numpy())
    orders = random_stratified_orders(group_code, n_replays, rng)
    full_delta = float(diffs.mean())
    out: list[dict[str, object]] = []
    for budget_cap in budget_caps:
        cap_k = max(1, min(len(diffs), int(math.ceil(len(diffs) * budget_cap))))
        prefix = orders[:, :cap_k]
        seen = diffs[prefix]
        prefix_groups = group_code[prefix]
        coverage = coverage_ok(prefix_groups, coverage_requirements(group_code, cap_k))
        boot = bootstrap_means(seen, bootstrap_samples, rng)
        sample_mean = seen.mean(axis=1)
        for threshold in thresholds:
            truth_accept = full_delta > threshold
            risk_accept = (boot <= threshold).mean(axis=1)
            risk_reject = (boot > threshold).mean(axis=1)
            for alpha in alphas:
                decision = np.zeros(n_replays, dtype=np.int8)
                accept = coverage & (sample_mean > threshold) & (risk_accept <= alpha)
                reject = coverage & (sample_mean <= threshold) & (risk_reject <= alpha)
                decision[accept] = 1
                decision[reject] = -1
                row = {
                    "source": pair.source,
                    "tested_system": pair.tested,
                    "comparison_system": pair.comparison,
                    "pair_label": pair.pair_label,
                    "threshold": threshold,
                    "policy_alpha": alpha,
                    "budget_cap": budget_cap,
                    "policy_label": main_policy_label(pair.source),
                    "truth_accept": bool(truth_accept),
                    "full_delta": full_delta,
                    "overlap_tasks": int(len(diffs)),
                    "observed_tasks": int(cap_k),
                    "group_count": int(len(group_mapping)),
                }
                row.update(
                    summarize_decision_vector(
                        decision,
                        truth_accept,
                        coverage,
                        full_delta,
                        threshold,
                        cap_k / len(diffs),
                    )
                )
                out.append(row)
    return out


def main_policy_label(source: str) -> str:
    if source.startswith("SWE-bench"):
        return "repo-coverage bootstrap"
    if source == "AppWorld":
        return "split-difficulty coverage bootstrap"
    if source == "tau-bench":
        return "coverage-aware bootstrap"
    if source == "OSWorld-Verified":
        return "app-coverage bootstrap"
    raise ValueError(source)


def swe_pairs(split: str) -> tuple[pd.DataFrame, list[PairSpec]]:
    task_rows = pd.read_csv(ANALYSIS / "swebench_smoke" / "swebench_task_rows.csv")
    task_rows = task_rows[task_rows["split"].eq(split)].copy()
    pair_summary = pd.read_csv(
        ANALYSIS / "swebench_smoke" / "swebench_pair_overlap_summary.csv"
    )
    pair_summary = pair_summary[
        pair_summary["split"].eq(split)
        & pair_summary["near_complete_pair"].astype(bool)
        & pair_summary["threshold"].eq(0.05)
    ].copy()
    source = "SWE-bench Lite" if split == "lite" else "SWE-bench Verified"
    pairs = [
        PairSpec(
            source=source,
            tested=str(row["candidate"]),
            comparison=str(row["reference"]),
            pair_label=f'{row["candidate"]}__vs__{row["reference"]}',
            group_column="repo",
        )
        for _, row in pair_summary[["candidate", "reference"]]
        .drop_duplicates()
        .iterrows()
    ]
    return task_rows, pairs


def prepare_swe_pair(task_rows: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
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
    merged["repo"] = merged["repo_tested"].astype(str)
    return merged.sort_values(["repo", "instance_id"]).reset_index(drop=True)


def appworld_pairs() -> tuple[pd.DataFrame, list[PairSpec]]:
    task_rows = pd.read_csv(
        ANALYSIS / "appworld_feasibility" / "appworld_task_rows.csv"
    )
    systems = sorted(str(item) for item in task_rows["system_id"].unique())
    pairs = [
        PairSpec(
            source="AppWorld",
            tested=a,
            comparison=b,
            pair_label=f"{a}__vs__{b}",
            group_column="stratum",
        )
        for a in systems
        for b in systems
        if a != b
    ]
    return task_rows, pairs


def prepare_appworld_pair(task_rows: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    tested = task_rows[task_rows["system_id"].eq(pair.tested)].copy()
    comparison = task_rows[task_rows["system_id"].eq(pair.comparison)].copy()
    merged = tested.merge(
        comparison,
        on="task_key",
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    merged["diff"] = merged["success_tested"].astype(float) - merged[
        "success_comparison"
    ].astype(float)
    merged["stratum"] = (
        merged["split_tested"].astype(str)
        + "_difficulty_"
        + merged["difficulty_tested"].astype(str)
    )
    return merged.sort_values(["stratum", "scenario_id_tested", "task_id_tested"]).reset_index(
        drop=True
    )


def tau_pairs() -> tuple[pd.DataFrame, list[PairSpec]]:
    task_rows = pd.read_csv(ANALYSIS / "tau2_ready" / "public_tau2_task_rows.csv")
    systems = sorted(str(item) for item in task_rows["submission_dir"].unique())
    pairs = [
        PairSpec(
            source="tau-bench",
            tested=a,
            comparison=b,
            pair_label=f"{a}__vs__{b}",
            group_column="domain",
        )
        for a in systems
        for b in systems
        if a != b
    ]
    return task_rows, pairs


def prepare_tau_pair(task_rows: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    tested = task_rows[task_rows["submission_dir"].eq(pair.tested)].copy()
    comparison = task_rows[task_rows["submission_dir"].eq(pair.comparison)].copy()
    keep_cols = ["domain", "task_id", "task_key", "outcome_mean"]
    merged = tested[keep_cols].merge(
        comparison[keep_cols],
        on=["domain", "task_id", "task_key"],
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    merged["diff"] = merged["outcome_mean_tested"] - merged["outcome_mean_comparison"]
    return merged.sort_values(["domain", "task_id"]).reset_index(drop=True)


def osworld_pairs() -> tuple[pd.DataFrame, list[PairSpec]]:
    task_rows = pd.read_csv(
        ANALYSIS / "osworld_verified" / "osworld_verified_task_rows.csv"
    )
    systems = sorted(str(item) for item in task_rows["system_id"].unique())
    pairs = [
        PairSpec(
            source="OSWorld-Verified",
            tested=a,
            comparison=b,
            pair_label=f"{a}__vs__{b}",
            group_column="coverage_group",
        )
        for a in systems
        for b in systems
        if a != b
    ]
    return task_rows, pairs


def prepare_osworld_pair(task_rows: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    tested = task_rows[task_rows["system_id"].eq(pair.tested)].copy()
    comparison = task_rows[task_rows["system_id"].eq(pair.comparison)].copy()
    keep_cols = ["task_key", "coverage_group", "outcome"]
    merged = tested[keep_cols].merge(
        comparison[keep_cols],
        on="task_key",
        suffixes=("_tested", "_comparison"),
        how="inner",
    )
    merged["diff"] = merged["outcome_tested"].astype(float) - merged[
        "outcome_comparison"
    ].astype(float)
    merged["coverage_group"] = merged["coverage_group_tested"].astype(str)
    return merged.sort_values(["coverage_group", "task_key"]).reset_index(drop=True)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "policy_alpha", "budget_cap", "policy_label"]
    for group_keys, group in summary.groupby(keys, dropna=False):
        source, threshold, alpha, budget_cap, policy_label = group_keys
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "policy_alpha": alpha,
                "budget_cap": budget_cap,
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
                "defer_rate": weighted_mean(group, "defer_rate"),
                "coverage_ok_rate": weighted_mean(group, "coverage_ok_rate"),
                "defer_no_coverage_rate": weighted_mean(
                    group, "defer_no_coverage_rate"
                ),
                "defer_uncertain_rate": weighted_mean(group, "defer_uncertain_rate"),
                "mean_task_fraction": weighted_mean(group, "mean_task_fraction"),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return float("nan")
    weights = group["rows"].astype(float)
    values = group[column].astype(float)
    if weights.sum() == 0:
        return float("nan")
    return float((weights * values).sum() / weights.sum())


def safe_to_csv(df: pd.DataFrame, path: Path, attempts: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{attempt}")
        try:
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(path)
            return
        except OSError as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(1.0 * attempt)
    if last_error is not None:
        raise last_error


def summarize_target_grid(
    aggregate_df: pd.DataFrame,
    coverage_targets: list[float] | None = None,
    defer_targets: list[float] | None = None,
) -> pd.DataFrame:
    coverage_targets = coverage_targets or [0.0, 0.05, 0.10]
    defer_targets = defer_targets or [0.10, 0.25, 0.40]
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "policy_alpha", "policy_label"]
    for group_keys, group in aggregate_df.groupby(keys, dropna=False):
        source, threshold, alpha, policy_label = group_keys
        ordered = group.sort_values("budget_cap")
        for gamma in coverage_targets:
            for eta in defer_targets:
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
                        "policy_label": policy_label,
                        "minimum_sufficient_budget": (
                            float(row["budget_cap"]) if row is not None else float("nan")
                        ),
                        "defer_rate_at_minimum": (
                            float(row["defer_rate"]) if row is not None else float("nan")
                        ),
                        "coverage_ok_at_minimum": (
                            float(row["coverage_ok_rate"]) if row is not None else float("nan")
                        ),
                        "positive_all_task_decisions": int(
                            ordered["positive_all_task_decisions"].iloc[0]
                        ),
                        "negative_all_task_decisions": int(
                            ordered["negative_all_task_decisions"].iloc[0]
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["source", "threshold", "policy_alpha", "coverage_target", "defer_target"]
    )


def run_source(
    source_key: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if source_key == "swe_lite":
        task_rows, pairs = swe_pairs("lite")
        prepare = prepare_swe_pair
        n_replays = args.swe_replays
        bootstrap_samples = args.swe_bootstrap_samples
    elif source_key == "swe_verified":
        task_rows, pairs = swe_pairs("verified")
        prepare = prepare_swe_pair
        n_replays = args.swe_replays
        bootstrap_samples = args.swe_bootstrap_samples
    elif source_key == "appworld":
        task_rows, pairs = appworld_pairs()
        prepare = prepare_appworld_pair
        n_replays = args.appworld_replays
        bootstrap_samples = args.appworld_bootstrap_samples
    elif source_key == "tau":
        task_rows, pairs = tau_pairs()
        prepare = prepare_tau_pair
        n_replays = args.tau_replays
        bootstrap_samples = args.tau_bootstrap_samples
    elif source_key == "osworld":
        task_rows, pairs = osworld_pairs()
        prepare = prepare_osworld_pair
        n_replays = args.osworld_replays
        bootstrap_samples = args.osworld_bootstrap_samples
    else:
        raise ValueError(source_key)

    rows: list[dict[str, object]] = []
    for idx, pair in enumerate(pairs, start=1):
        if args.max_pairs_per_source is not None and idx > args.max_pairs_per_source:
            break
        pair_df = prepare(task_rows, pair)
        rows.extend(
            run_pair(
                pair,
                pair_df,
                args.alphas,
                args.thresholds,
                args.budget_caps,
                n_replays,
                bootstrap_samples,
                rng,
            )
        )
        total = (
            min(len(pairs), args.max_pairs_per_source)
            if args.max_pairs_per_source is not None
            else len(pairs)
        )
        if idx % 20 == 0 or idx == total:
            print(f"{source_key}: processed {idx}/{total} pairs", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_rows: list[pd.DataFrame] = []
    for source_key in args.sources:
        start = time.perf_counter()
        source_df = run_source(source_key, args, rng)
        source_path = args.out_dir / f"{source_key}_main_policy_pair_summary.csv"
        safe_to_csv(source_df, source_path)
        all_rows.append(source_df)
        elapsed = time.perf_counter() - start
        print(
            f"{source_key}: wrote {len(source_df)} rows in {elapsed:.1f}s",
            flush=True,
        )
    summary = pd.concat(all_rows, ignore_index=True)
    aggregate_df = aggregate(summary)
    target_grid = summarize_target_grid(aggregate_df)
    safe_to_csv(summary, args.out_dir / "main_policy_pair_summary.csv")
    safe_to_csv(aggregate_df, args.out_dir / "main_policy_aggregate.csv")
    safe_to_csv(target_grid, args.out_dir / "main_policy_target_grid.csv")
    metadata = {
        "sources": args.sources,
        "alphas": args.alphas,
        "thresholds": args.thresholds,
        "budget_caps": args.budget_caps,
        "swe_replays": args.swe_replays,
        "appworld_replays": args.appworld_replays,
        "tau_replays": args.tau_replays,
        "swe_bootstrap_samples": args.swe_bootstrap_samples,
        "appworld_bootstrap_samples": args.appworld_bootstrap_samples,
        "tau_bootstrap_samples": args.tau_bootstrap_samples,
        "seed": args.seed,
        "note": "Coverage-aware main policies only; alpha changes both bootstrap tail threshold and conditional error target.",
    }
    (args.out_dir / "main_policy_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pair_summary_rows": int(len(summary)),
                "aggregate_rows": int(len(aggregate_df)),
                "target_grid_rows": int(len(target_grid)),
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
