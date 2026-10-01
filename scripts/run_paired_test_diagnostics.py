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
OUT_DIR = ANALYSIS / "paired_test_diagnostics"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_main_policy_sensitivity as base  # noqa: E402


PRIMARY_ALPHA = 0.05
PRIMARY_GAMMA = 0.05
PRIMARY_ETA = 0.25
Z_ONE_SIDED_95 = 1.6448536269514722


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bootstrap-tail replay decisions with classical paired-test rules."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["swe_lite", "swe_verified", "appworld", "tau"],
        choices=["swe_lite", "swe_verified", "appworld", "tau"],
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.0, 0.05, 0.10]
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
    parser.add_argument("--seed", type=int, default=20260633)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-pairs-per-source",
        type=int,
        default=None,
        help="Optional smoke-test limit. Leave unset for paper diagnostics.",
    )
    return parser.parse_args()


def percent(value: object, digits: int = 1) -> str:
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


def load_source(source_key: str):
    if source_key == "swe_lite":
        task_rows, pairs = base.swe_pairs("lite")
        return task_rows, pairs, base.prepare_swe_pair, "SWE-bench Lite"
    if source_key == "swe_verified":
        task_rows, pairs = base.swe_pairs("verified")
        return task_rows, pairs, base.prepare_swe_pair, "SWE-bench Verified"
    if source_key == "appworld":
        task_rows, pairs = base.appworld_pairs()
        return task_rows, pairs, base.prepare_appworld_pair, "AppWorld"
    if source_key == "tau":
        task_rows, pairs = base.tau_pairs()
        return task_rows, pairs, base.prepare_tau_pair, "tau-bench"
    raise ValueError(source_key)


def replay_count(source_key: str, args: argparse.Namespace) -> int:
    if source_key.startswith("swe"):
        return args.swe_replays
    if source_key == "appworld":
        return args.appworld_replays
    if source_key == "tau":
        return args.tau_replays
    raise ValueError(source_key)


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
        "coverage_failure_rate": float(((decision != 0) & (~coverage)).mean()),
        "coverage_ok_rate": float(coverage.mean()),
        "defer_no_coverage_rate": float((defer & (~coverage)).mean()),
        "defer_uncertain_rate": float((defer & coverage).mean()),
    }


def aggregate_pair_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source", "threshold", "budget_cap", "policy"]
    for values, group in pair_rows.groupby(keys, dropna=False):
        source, threshold, budget_cap, policy = values
        pos = group[group["truth_accept"]]
        neg = group[~group["truth_accept"]]
        rows.append(
            {
                "source": source,
                "threshold": threshold,
                "budget_cap": budget_cap,
                "policy": policy,
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


def paired_normal_decision(
    seen: np.ndarray,
    means: np.ndarray,
    coverage: np.ndarray,
    threshold: float,
) -> np.ndarray:
    decision = np.zeros(seen.shape[0], dtype=np.int8)
    if seen.shape[1] < 2:
        return decision
    sd = seen.std(axis=1, ddof=1)
    se = sd / math.sqrt(seen.shape[1])
    lower = means - Z_ONE_SIDED_95 * se
    upper = means + Z_ONE_SIDED_95 * se
    zero_se = se == 0
    lower[zero_se] = means[zero_se]
    upper[zero_se] = means[zero_se]
    decision[coverage & (lower > threshold)] = 1
    decision[coverage & (upper <= threshold)] = -1
    return decision


def binomial_tail_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    probs = np.empty(n + 1, dtype=np.float64)
    probs[0] = 2.0 ** (-n)
    for k in range(n):
        probs[k + 1] = probs[k] * (n - k) / (k + 1)
    cdf = np.cumsum(probs)
    sf_inclusive = np.cumsum(probs[::-1])[::-1]
    return cdf, sf_inclusive


def exact_mcnemar_decision(seen: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    wins = (seen > 0).sum(axis=1).astype(int)
    losses = (seen < 0).sum(axis=1).astype(int)
    discordant = wins + losses
    p_lower = np.ones(seen.shape[0], dtype=np.float64)
    p_upper = np.ones(seen.shape[0], dtype=np.float64)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in sorted(int(item) for item in np.unique(discordant) if item > 0):
        if n not in cache:
            cache[n] = binomial_tail_arrays(n)
        lower, upper = cache[n]
        mask = discordant == n
        p_lower[mask] = lower[wins[mask]]
        p_upper[mask] = upper[wins[mask]]
    decision = np.zeros(seen.shape[0], dtype=np.int8)
    decision[coverage & (wins > losses) & (p_upper <= PRIMARY_ALPHA)] = 1
    decision[coverage & (wins <= losses) & (p_lower <= PRIMARY_ALPHA)] = -1
    return decision


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for source_key in args.sources:
        task_rows, pairs, prepare, source_name = load_source(source_key)
        if args.max_pairs_per_source is not None:
            pairs = pairs[: args.max_pairs_per_source]
        n_replays = replay_count(source_key, args)
        seed_offsets = {
            "swe_lite": 11,
            "swe_verified": 17,
            "appworld": 23,
            "tau": 29,
        }
        rng = np.random.default_rng(args.seed + seed_offsets[source_key])
        for pair_idx, pair in enumerate(pairs, start=1):
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
                means = seen.mean(axis=1)
                for threshold in args.thresholds:
                    truth_accept = full_delta > threshold
                    policy_rows: list[tuple[str, np.ndarray]] = [
                        (
                            "coverage paired normal test",
                            paired_normal_decision(seen, means, coverage, threshold),
                        )
                    ]
                    if math.isclose(threshold, 0.0):
                        policy_rows.append(
                            (
                                "coverage exact McNemar test",
                                exact_mcnemar_decision(seen, coverage),
                            )
                        )
                    for policy, decision in policy_rows:
                        row = {
                            "source": source_name,
                            "tested_system": pair.tested,
                            "comparison_system": pair.comparison,
                            "pair_label": pair.pair_label,
                            "threshold": threshold,
                            "budget_cap": budget_cap,
                            "policy": policy,
                            "truth_accept": bool(truth_accept),
                            "full_delta": full_delta,
                            "observed_tasks": int(cap_k),
                            "overlap_tasks": int(len(diffs)),
                        }
                        row.update(
                            summarize_decisions(
                                decision, truth_accept, coverage, cap_k / len(diffs)
                            )
                        )
                        rows.append(row)
            if pair_idx % 50 == 0 or pair_idx == len(pairs):
                print(f"{source_name}: {pair_idx}/{len(pairs)} pairs", flush=True)
    pair_rows = pd.DataFrame(rows)
    aggregate = aggregate_pair_rows(pair_rows)
    minimum = minimum_budgets(aggregate)
    return pair_rows, aggregate, minimum


def write_report(out_dir: Path, minimum: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Classical Paired-Test Diagnostics")
    lines.append("")
    lines.append(
        "These diagnostics hold the coverage-aware reveal order fixed and replace the bootstrap-tail decision rule with classical paired-test decision rules."
    )
    lines.append(
        "The exact McNemar rule is reported only for the 0 pp threshold, where the paired binary-disagreement test directly matches the directional comparison."
    )
    lines.append(
        "For 5 pp and 10 pp thresholds, the paired normal rule tests whether the observed paired mean clears the specified threshold."
    )
    lines.append("")

    display = minimum.copy()
    display["threshold"] = display["threshold"].map(lambda x: f"{100*float(x):.0f} pp")
    display["minimum_sufficient_budget"] = display[
        "minimum_sufficient_budget"
    ].map(budget)
    display["defer_rate_at_minimum"] = display["defer_rate_at_minimum"].map(percent)
    lines.append("## Minimum Sufficient Budgets")
    lines.append("")
    lines.append(
        markdown_table(
            display[
                [
                    "source",
                    "threshold",
                    "policy",
                    "minimum_sufficient_budget",
                    "defer_rate_at_minimum",
                    "positive_all_task_decisions",
                    "negative_all_task_decisions",
                ]
            ]
        )
    )
    lines.append("")

    focus = aggregate[
        aggregate["threshold"].eq(0.0)
        & aggregate["budget_cap"].isin([0.25, 0.50, 0.75, 0.90, 0.95])
    ].copy()
    for col in [
        "conditional_false_accept_rate",
        "conditional_false_reject_rate",
        "defer_rate",
        "coverage_ok_rate",
    ]:
        focus[col] = focus[col].map(percent)
    focus["budget_cap"] = focus["budget_cap"].map(budget)
    lines.append("## 0 pp Operating Profiles")
    lines.append("")
    lines.append(
        markdown_table(
            focus[
                [
                    "source",
                    "budget_cap",
                    "policy",
                    "conditional_false_accept_rate",
                    "conditional_false_reject_rate",
                    "defer_rate",
                    "coverage_ok_rate",
                ]
            ].head(120)
        )
    )
    lines.append("")
    (out_dir / "paired_test_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_rows, aggregate, minimum = run(args)
    pair_rows.to_csv(args.out_dir / "paired_test_pair_rows.csv", index=False)
    aggregate.to_csv(args.out_dir / "paired_test_aggregate.csv", index=False)
    minimum.to_csv(args.out_dir / "paired_test_minimum_budgets.csv", index=False)
    metadata = {
        "thresholds": args.thresholds,
        "budget_caps": args.budget_caps,
        "swe_replays": args.swe_replays,
        "appworld_replays": args.appworld_replays,
        "tau_replays": args.tau_replays,
        "seed": args.seed,
        "primary_alpha": PRIMARY_ALPHA,
        "primary_gamma": PRIMARY_GAMMA,
        "primary_eta": PRIMARY_ETA,
        "sources": args.sources,
        "notes": [
            "McNemar rows are only defined for threshold 0 pp.",
            "Paired normal rows use a one-sided 95 percent normal critical value.",
        ],
    }
    (args.out_dir / "paired_test_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(args.out_dir, minimum, aggregate)
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
