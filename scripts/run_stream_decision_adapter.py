from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_decision_layer_demo import (
    ANALYSIS,
    bootstrap_action,
    card_for_budget,
    clean_resources,
    default_systems,
    group_order,
    load_tau_pair,
    load_terminal_pair,
    prefix_indices,
    resource_values,
    rng_for_budget,
)
from stream_decision_schema import (
    validate_event_against_manifest,
    validate_stream_manifest,
    validate_terminal_decision_record,
    validate_trace_record,
)


DEFAULT_OUT = ANALYSIS / "decision_layer_stream_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consume evaluation-pipeline task outcome events and emit decision "
            "records from budget prefixes. Completed-record fields are used "
            "only for optional simulation audit and equivalence checks."
        )
    )
    parser.add_argument("--source", choices=["tau", "terminal", "generic"], default="tau")
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
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stream-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--skip-equivalence-check",
        action="store_true",
        help="Emit stream decisions without comparing them to the batch driver.",
    )
    return parser.parse_args()


def finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_pair_for_source(source: str, tested: str, comparison: str) -> tuple[str, pd.DataFrame]:
    if source == "tau":
        return "tau-bench", load_tau_pair(tested, comparison)
    if source == "terminal":
        return "Terminal-Bench 2.0", load_terminal_pair(tested, comparison)
    raise ValueError("Generic streams must be supplied with --stream-path.")


def resolve_coverage_mode(source: str, coverage_mode: str) -> str:
    if coverage_mode != "auto":
        return coverage_mode
    if source == "generic":
        return "none"
    return "source" if source == "tau" else "none"


def prepare_pair(
    source: str,
    tested: str,
    comparison: str,
    coverage_mode: str,
) -> tuple[str, pd.DataFrame]:
    source_label, pair_df = load_pair_for_source(source, tested, comparison)
    if coverage_mode == "none":
        pair_df = pair_df.copy()
        pair_df["group"] = "all"
    return source_label, pair_df


def metric_resource_from_row(row: dict[str, Any], metric: str) -> float:
    if metric == "task_fraction":
        return 1.0
    if metric == "tested_cost_fraction":
        return float(row["tested_resource"])
    if metric == "pair_cost_fraction":
        return float(row["tested_resource"]) + float(row["comparison_resource"])
    if metric == "tested_time_fraction":
        return float(row["tested_time"])
    if metric == "pair_time_fraction":
        return float(row["tested_time"]) + float(row["comparison_time"])
    raise ValueError(metric)


def group_counts(pair_df: pd.DataFrame) -> dict[str, int]:
    return {
        str(group): int(count)
        for group, count in pair_df["group"].value_counts().sort_index().items()
    }


def write_generated_stream(
    pair_df: pd.DataFrame,
    source_label: str,
    source: str,
    tested: str,
    comparison: str,
    budget_metric: str,
    ordering: str,
    coverage_mode: str,
    seed: int,
    path: Path,
    run_id: str,
) -> tuple[np.ndarray, float, list[int]]:
    resources, completeness = clean_resources(
        resource_values(pair_df, budget_metric), budget_metric
    )
    rng = np.random.default_rng(seed)
    order = group_order(pair_df, resources, ordering, rng)
    denominator = float(len(pair_df) if budget_metric == "task_fraction" else resources.sum())
    if denominator <= 0:
        raise ValueError(f"Non-positive budget denominator for {budget_metric}.")

    manifest = {
        "event_type": "stream_manifest",
        "schema_version": "1.0",
        "source": source_label,
        "source_key": source,
        "run_id": run_id,
        "systems": {"tested": tested, "comparison": comparison},
        "budget_metric": budget_metric,
        "bootstrap_seed": int(seed),
        "budget_denominator": denominator,
        "resource_completeness": completeness,
        "total_tasks": int(len(pair_df)),
        "group_counts": group_counts(pair_df),
        "coverage_mode": coverage_mode,
        "ordering": ordering,
        "note": (
            "Manifest fields describe the planned task-outcome stream and do not "
            "contain task outcomes."
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    event_index = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True) + "\n")
        for pair_position, idx in enumerate(order, start=1):
            row = pair_df.iloc[int(idx)]
            for role, system, suffix in [
                ("tested", tested, "tested"),
                ("comparison", comparison, "comparison"),
            ]:
                event_index += 1
                event = {
                    "event_type": "task_outcome",
                    "schema_version": "1.0",
                    "source": source_label,
                    "run_id": run_id,
                    "event_index": event_index,
                    "pair_position": pair_position,
                    "role": role,
                    "system": system,
                    "task_key": str(row["task_key"]),
                    "group": str(row["group"]),
                    "outcome": finite_or_none(row[f"outcome_{suffix}"]),
                    "resource": {
                        "cost_usd": finite_or_none(row[f"{suffix}_resource"]),
                        "agent_execution_seconds": finite_or_none(row[f"{suffix}_time"]),
                    },
                }
                handle.write(json.dumps(event, sort_keys=True) + "\n")
    return resources, completeness, order


def required_counts_from_manifest(
    manifest: dict[str, Any],
    observed_count: int,
) -> dict[str, int]:
    total_tasks = int(manifest["total_tasks"])
    requirements: dict[str, int] = {}
    for group, raw_count in manifest["group_counts"].items():
        group_size = int(raw_count)
        expected = observed_count * (group_size / total_tasks)
        requirements[str(group)] = max(1, min(group_size, math.floor(expected)))
    return requirements


def coverage_state_from_records(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    required = required_counts_from_manifest(manifest, len(records))
    observed: dict[str, int] = {}
    for record in records:
        group = str(record["group"])
        observed[group] = observed.get(group, 0) + 1
    missing = [
        group
        for group, required_count in required.items()
        if observed.get(group, 0) < required_count
    ]
    return {
        "required_counts": required,
        "observed_counts": observed,
        "missing_strata": missing,
        "coverage_ok": not missing,
    }


def make_stream_card(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    prefix_len: int,
    threshold: float,
    alpha: float,
    budget_cap: float,
    bootstrap_samples: int,
    emission_event_index: int,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    prefix_records = records[:prefix_len]
    diffs = np.array(
        [
            float(record["outcome_tested"]) - float(record["outcome_comparison"])
            for record in prefix_records
        ],
        dtype=float,
    )
    resources = np.array(
        [metric_resource_from_row(record, str(manifest["budget_metric"])) for record in prefix_records],
        dtype=float,
    )
    coverage = coverage_state_from_records(manifest, prefix_records)
    if not coverage["coverage_ok"]:
        decision = {
            "action": "abstain",
            "partial_delta": float(diffs.mean()),
            "bootstrap_tail_if_promote": None,
            "bootstrap_tail_if_reject": None,
        }
        abstention_reason = "missing_required_strata"
    else:
        rng = rng_for_budget(int(manifest["bootstrap_seed"]), budget_cap)
        decision = bootstrap_action(diffs, threshold, alpha, bootstrap_samples, rng)
        abstention_reason = (
            "bootstrap_tail_not_below_alpha"
            if decision["action"] == "abstain"
            else None
        )

    metric = str(manifest["budget_metric"])
    if metric == "task_fraction":
        resource_fraction = float(prefix_len / int(manifest["total_tasks"]))
    else:
        resource_fraction = float(resources.sum() / float(manifest["budget_denominator"]))

    audit_record: dict[str, Any] = {}
    if audit is not None:
        full_record_decision = str(audit["full_record_decision"])
        action = str(decision["action"])
        audit_record = {
            "full_delta": float(audit["full_delta"]),
            "full_record_decision": full_record_decision,
            "matches_full_record_when_decided": (
                action == full_record_decision if action != "abstain" else None
            ),
        }

    return {
        "schema_version": "1.2",
        "source": manifest["source"],
        "systems": manifest["systems"],
        "threshold": threshold,
        "alpha": alpha,
        "task_selection_policy": {
            "name": manifest["ordering"],
            "budget_metric": metric,
            "coverage_mode": manifest["coverage_mode"],
            "prefix_only": True,
            "input": "task_outcome_stream",
        },
        "budget_cap": budget_cap,
        "observed": {
            "tasks": prefix_len,
            "task_fraction": float(prefix_len / int(manifest["total_tasks"])),
            "resource_fraction": resource_fraction,
            "resource_completeness": float(manifest["resource_completeness"]),
        },
        "strata_coverage": coverage,
        "decision": {
            "action": str(decision["action"]),
            "partial_delta": float(decision["partial_delta"]),
            "bootstrap_tail_if_promote": finite_or_none(decision["bootstrap_tail_if_promote"]),
            "bootstrap_tail_if_reject": finite_or_none(decision["bootstrap_tail_if_reject"]),
            "abstention_reason": abstention_reason,
        },
        "event_stream": {
            "run_id": manifest["run_id"],
            "emission_event_index": emission_event_index,
            "paired_tasks_buffered": len(records),
            "budget_denominator": float(manifest["budget_denominator"]),
        },
        "audit_only_full_record": audit_record,
    }


def make_trace_record(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    prefix_len: int,
    threshold: float,
    alpha: float,
    trace_budget_cap: float,
    bootstrap_samples: int,
    terminal: bool,
    terminal_reason: str | None,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    visible_records = records[:prefix_len]
    emission_event_index = int(visible_records[-1]["completion_event_index"])
    record = make_stream_card(
        manifest,
        visible_records,
        prefix_len,
        threshold,
        alpha,
        trace_budget_cap,
        bootstrap_samples,
        emission_event_index,
        audit,
    )
    raw_action = str(record["decision"]["action"])
    raw_reason = record["decision"]["abstention_reason"]
    if not terminal:
        record["decision"]["action"] = "continue"
        record["decision"]["continue_reason"] = raw_reason or "budget_remaining"
        record["decision"]["abstention_reason"] = None
    else:
        record["decision"]["continue_reason"] = None
        if raw_action == "abstain":
            record["decision"]["abstention_reason"] = (
                raw_reason or "budget_cap_reached_without_supported_decision"
            )

    record["schema_version"] = "1.3"
    record["record_type"] = "decision_trace_step"
    record["trace"] = {
        "budget_cap": trace_budget_cap,
        "prefix_index": prefix_len,
        "terminal": terminal,
        "terminal_reason": terminal_reason,
        "would_decide_if_cap_now": raw_action,
    }
    record["task_selection_policy"]["trace"] = "per_paired_task_prefix"
    return record


def make_decision_traces(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    budget_caps: list[float],
    threshold: float,
    alpha: float,
    bootstrap_samples: int,
    audit: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    trace_records: list[dict[str, Any]] = []
    for budget_cap in sorted(budget_caps):
        cap_prefix_len = prefix_len_for_budget(
            records,
            manifest,
            budget_cap,
            stream_end=True,
        )
        if cap_prefix_len is None:
            continue
        for prefix_len in range(1, cap_prefix_len + 1):
            terminal = prefix_len == cap_prefix_len
            terminal_reason = "budget_cap_reached" if terminal else None
            trace_records.append(
                make_trace_record(
                    manifest,
                    records,
                    prefix_len,
                    threshold,
                    alpha,
                    budget_cap,
                    bootstrap_samples,
                    terminal,
                    terminal_reason,
                    audit,
                )
            )
            if terminal:
                break
    return trace_records


def paired_record_from_events(
    task_key: str,
    events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tested_event = events["tested"]
    comparison_event = events["comparison"]
    if tested_event.get("outcome") is None or comparison_event.get("outcome") is None:
        raise ValueError(f"Missing outcome for paired task {task_key}.")
    if str(tested_event["group"]) != str(comparison_event["group"]):
        raise ValueError(f"Group mismatch for paired task {task_key}.")
    return {
        "task_key": task_key,
        "group": str(tested_event["group"]),
        "outcome_tested": float(tested_event["outcome"]),
        "outcome_comparison": float(comparison_event["outcome"]),
        "tested_resource": float(tested_event["resource"].get("cost_usd") or 0.0),
        "comparison_resource": float(comparison_event["resource"].get("cost_usd") or 0.0),
        "tested_time": float(tested_event["resource"].get("agent_execution_seconds") or 0.0),
        "comparison_time": float(comparison_event["resource"].get("agent_execution_seconds") or 0.0),
        "completion_event_index": max(
            int(tested_event["event_index"]),
            int(comparison_event["event_index"]),
        ),
    }


def prefix_len_for_budget(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    budget_cap: float,
    stream_end: bool,
) -> int | None:
    metric = str(manifest["budget_metric"])
    if metric == "task_fraction":
        cap_k = max(
            1,
            min(
                int(manifest["total_tasks"]),
                int(math.ceil(int(manifest["total_tasks"]) * budget_cap)),
            ),
        )
        return cap_k if len(records) >= cap_k else None

    denominator = float(manifest["budget_denominator"])
    limit = budget_cap * denominator
    values = np.array([metric_resource_from_row(row, metric) for row in records], dtype=float)
    cumulative = np.cumsum(values)
    if not len(cumulative):
        return None
    if cumulative[-1] <= limit and not stream_end:
        return None
    count = int(np.searchsorted(cumulative, limit, side="right"))
    count = max(1, min(len(records), count))
    return count


def consume_stream(
    stream_path: Path,
    budget_caps: list[float],
    threshold: float,
    alpha: float,
    bootstrap_samples: int,
    seed: int,
    audit: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: dict[str, Any] | None = None
    pending: dict[str, dict[str, dict[str, Any]]] = {}
    completed: set[str] = set()
    records: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    emitted: set[float] = set()
    last_event_index = 0

    with stream_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{stream_path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"{stream_path}:{line_number}: JSONL rows must be objects."
                )
            if event.get("event_type") == "stream_manifest":
                if manifest is not None or line_number != 1:
                    raise ValueError(
                        "Stream must contain exactly one manifest, as its first record."
                    )
                validate_stream_manifest(event)
                manifest = event
                continue
            if manifest is None:
                raise ValueError("Stream manifest must be the first record.")
            if event.get("event_type") != "task_outcome":
                raise ValueError(
                    f"Unsupported event_type at line {line_number}: "
                    f"{event.get('event_type')!r}."
                )
            validate_event_against_manifest(event, manifest)

            event_index = int(event["event_index"])
            if event_index <= last_event_index:
                raise ValueError(
                    f"Nonmonotone event_index {event_index}; previous was {last_event_index}."
                )
            last_event_index = event_index
            task_key = str(event["task_key"])
            role = str(event["role"])
            if role not in {"tested", "comparison"}:
                raise ValueError(f"Unknown role {role} for task {task_key}.")
            if task_key in completed:
                raise ValueError(f"Duplicate completed task outcome for {task_key}.")
            if event.get("outcome") is None or not math.isfinite(float(event["outcome"])):
                raise ValueError(f"Missing or nonfinite outcome for {task_key}.")
            for resource_name, resource_value in event.get("resource", {}).items():
                if resource_value is None:
                    continue
                resource_value = float(resource_value)
                if not math.isfinite(resource_value) or resource_value < 0:
                    raise ValueError(
                        f"Invalid resource {resource_name}={resource_value} for {task_key}."
                    )
            if role in pending.setdefault(task_key, {}):
                raise ValueError(f"Duplicate {role} event for pending task {task_key}.")
            pending.setdefault(task_key, {})[role] = event
            if {"tested", "comparison"}.issubset(pending[task_key]):
                records.append(paired_record_from_events(task_key, pending.pop(task_key)))
                completed.add(task_key)
                for budget_cap in sorted(budget_caps):
                    if budget_cap in emitted:
                        continue
                    prefix_len = prefix_len_for_budget(
                        records, manifest, budget_cap, stream_end=False
                    )
                    if prefix_len is None:
                        continue
                    cards.append(
                        make_stream_card(
                            manifest,
                            records,
                            prefix_len,
                            threshold,
                            alpha,
                            budget_cap,
                            bootstrap_samples,
                            int(event["event_index"]),
                            audit,
                        )
                    )
                    emitted.add(budget_cap)

    if manifest is None:
        raise ValueError("Stream manifest missing.")
    if pending:
        missing = ", ".join(sorted(pending.keys())[:5])
        raise ValueError(f"Unpaired task outcome events remain, examples: {missing}")
    for budget_cap in sorted(budget_caps):
        if budget_cap in emitted:
            continue
        prefix_len = prefix_len_for_budget(records, manifest, budget_cap, stream_end=True)
        if prefix_len is None:
            raise ValueError(f"Could not emit budget cap {budget_cap}.")
        cards.append(
            make_stream_card(
                manifest,
                records,
                prefix_len,
                threshold,
                alpha,
                budget_cap,
                bootstrap_samples,
                -1,
                audit,
            )
        )
        emitted.add(budget_cap)
    return manifest, records, cards


def batch_cards(
    pair_df: pd.DataFrame,
    source_label: str,
    tested: str,
    comparison: str,
    threshold: float,
    alpha: float,
    budget_caps: list[float],
    budget_metric: str,
    ordering: str,
    coverage_mode: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    resources, completeness = clean_resources(
        resource_values(pair_df, budget_metric), budget_metric
    )
    rng = np.random.default_rng(seed)
    order = group_order(pair_df, resources, ordering, rng)
    cards = []
    for budget_cap in sorted(budget_caps):
        prefix = prefix_indices(order, resources, budget_cap, budget_metric)
        cards.append(
            card_for_budget(
                source_label,
                pair_df,
                tested,
                comparison,
                threshold,
                alpha,
                budget_cap,
                budget_metric,
                ordering,
                coverage_mode,
                prefix,
                resources,
                completeness,
                bootstrap_samples,
                rng_for_budget(seed, budget_cap),
            )
        )
    return cards


def compare_stream_to_batch(
    stream_cards: list[dict[str, Any]],
    reference_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = []
    failures = []
    fields = [
        ("decision.action", lambda card: card["decision"]["action"]),
        ("observed.tasks", lambda card: card["observed"]["tasks"]),
        ("observed.task_fraction", lambda card: card["observed"]["task_fraction"]),
        ("observed.resource_fraction", lambda card: card["observed"]["resource_fraction"]),
        ("strata_coverage.coverage_ok", lambda card: card["strata_coverage"]["coverage_ok"]),
        ("decision.partial_delta", lambda card: card["decision"]["partial_delta"]),
        ("decision.bootstrap_tail_if_promote", lambda card: card["decision"]["bootstrap_tail_if_promote"]),
        ("decision.bootstrap_tail_if_reject", lambda card: card["decision"]["bootstrap_tail_if_reject"]),
    ]
    by_budget = {float(card["budget_cap"]): card for card in reference_cards}
    for stream_card in stream_cards:
        budget = float(stream_card["budget_cap"])
        reference = by_budget.get(budget)
        if reference is None:
            failures.append({"budget_cap": budget, "field": "budget", "reason": "missing"})
            continue
        for field_name, getter in fields:
            left = getter(stream_card)
            right = getter(reference)
            if isinstance(left, float) or isinstance(right, float):
                ok = (
                    left is None
                    and right is None
                    or left is not None
                    and right is not None
                    and math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)
                )
            else:
                ok = left == right
            checks.append(
                {
                    "budget_cap": budget,
                    "field": field_name,
                    "stream": left,
                    "batch": right,
                    "match": ok,
                }
            )
            if not ok:
                failures.append(checks[-1])
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "stream_records": len(stream_cards),
        "batch_records": len(reference_cards),
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_stream_outputs(
    stream_cards: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
) -> None:
    for card in stream_cards:
        validate_terminal_decision_record(card)
    for record in trace_records:
        validate_trace_record(record)


def write_summary(
    path: Path,
    stream_path: Path,
    cards_path: Path,
    trace_path: Path,
    equivalence_path: Path,
    cards: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    equivalence: dict[str, Any] | None,
) -> None:
    lines = [
        "# Stream decision adapter demo",
        "",
        f"Task outcome stream: `{stream_path}`",
        f"Decision records: `{cards_path}`",
        f"Per-prefix decision trace: `{trace_path}`",
        "",
        "| Budget | Tasks | Resource fraction | Action | Coverage | Emission event |",
        "|---:|---:|---:|---|---|---:|",
    ]
    for card in cards:
        coverage = "yes" if card["strata_coverage"]["coverage_ok"] else "no"
        lines.append(
            "| "
            f"{float(card['budget_cap']) * 100:.0f}% | "
            f"{int(card['observed']['tasks'])} | "
            f"{float(card['observed']['resource_fraction']) * 100:.1f}% | "
            f"{card['decision']['action']} | "
            f"{coverage} | "
            f"{int(card['event_stream']['emission_event_index'])} |"
        )
    lines.append("")
    if trace_records:
        actions = sorted({str(record["decision"]["action"]) for record in trace_records})
        terminal_count = sum(1 for record in trace_records if record["trace"]["terminal"])
        lines.extend(
            [
                (
                    "The per-prefix trace emits continue while budget remains and "
                    "emits the terminal promote, reject, or abstain action at the "
                    "configured budget cap."
                ),
                f"Trace records: {len(trace_records)}",
                f"Terminal trace records: {terminal_count}",
                f"Trace actions: {', '.join(actions)}",
                "",
            ]
        )
    if equivalence is not None:
        status = "PASS" if equivalence["passed"] else "FAIL"
        lines.extend(
            [
                f"Batch equivalence: {status}",
                f"Equivalence details: `{equivalence_path}`",
                "",
            ]
        )
    lines.append(
        "The adapter consumes task outcome events and forms paired task differences "
        "only after both tested and comparison outcomes are observed. Completed-record "
        "fields are written only under audit keys and in the equivalence report."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    coverage_mode = resolve_coverage_mode(args.source, args.coverage_mode)
    run_id = args.run_id or f"{args.source}_stream_{args.budget_metric}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stream_path = args.stream_path or args.out_dir / f"{args.source}_task_outcome_stream.jsonl"
    pair_df: pd.DataFrame | None = None
    source_label: str | None = None
    tested: str | None = None
    comparison: str | None = None
    audit: dict[str, Any] | None = None

    if args.source == "generic":
        if args.stream_path is None:
            raise ValueError("Generic mode requires --stream-path.")
        if not args.skip_equivalence_check:
            raise ValueError(
                "Generic mode has no source-specific batch reference; "
                "pass --skip-equivalence-check."
            )
    else:
        tested, comparison = default_systems(args.source)
        tested = args.tested or tested
        comparison = args.comparison or comparison
        source_label, pair_df = prepare_pair(args.source, tested, comparison, coverage_mode)

    if args.stream_path is None:
        assert pair_df is not None
        assert source_label is not None
        assert tested is not None
        assert comparison is not None
        write_generated_stream(
            pair_df,
            source_label,
            args.source,
            tested,
            comparison,
            args.budget_metric,
            args.ordering,
            coverage_mode,
            args.seed,
            stream_path,
            run_id,
        )

    if pair_df is not None:
        diffs = (
            pair_df["outcome_tested"].to_numpy(dtype=float)
            - pair_df["outcome_comparison"].to_numpy(dtype=float)
        )
        full_delta = float(diffs.mean())
        audit = {
            "full_delta": full_delta,
            "full_record_decision": "promote" if full_delta > args.threshold else "reject",
        }

    manifest, paired_records, stream_cards = consume_stream(
        stream_path,
        sorted(args.budget_caps),
        args.threshold,
        args.alpha,
        args.bootstrap_samples,
        args.seed,
        audit,
    )
    trace_records = make_decision_traces(
        manifest,
        paired_records,
        sorted(args.budget_caps),
        args.threshold,
        args.alpha,
        args.bootstrap_samples,
        audit,
    )
    validate_stream_outputs(stream_cards, trace_records)
    cards_path = args.out_dir / f"{args.source}_stream_decision_records.jsonl"
    write_jsonl(stream_cards, cards_path)
    trace_path = args.out_dir / f"{args.source}_stream_decision_trace.jsonl"
    write_jsonl(trace_records, trace_path)

    equivalence = None
    equivalence_path = args.out_dir / f"{args.source}_stream_equivalence.json"
    if not args.skip_equivalence_check:
        assert pair_df is not None
        assert tested is not None
        assert comparison is not None
        assert source_label is not None
        reference_cards = batch_cards(
            pair_df,
            source_label,
            tested,
            comparison,
            args.threshold,
            args.alpha,
            sorted(args.budget_caps),
            args.budget_metric,
            args.ordering,
            coverage_mode,
            args.bootstrap_samples,
            args.seed,
        )
        equivalence = compare_stream_to_batch(stream_cards, reference_cards)
        equivalence_path.write_text(
            json.dumps(equivalence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not equivalence["passed"]:
            raise SystemExit(f"Stream equivalence failed: {equivalence_path}")

    summary_path = args.out_dir / f"{args.source}_stream_decision_adapter.md"
    write_summary(
        summary_path,
        stream_path,
        cards_path,
        trace_path,
        equivalence_path,
        stream_cards,
        trace_records,
        equivalence,
    )
    print(
        json.dumps(
            {
                "source": manifest["source"],
                "paired_tasks": len(paired_records),
                "records": len(stream_cards),
                "trace_records": len(trace_records),
                "stream_path": str(stream_path),
                "records_path": str(cards_path),
                "trace_path": str(trace_path),
                "summary_path": str(summary_path),
                "equivalence_passed": None if equivalence is None else equivalence["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
