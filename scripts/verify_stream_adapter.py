from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from stream_decision_schema import (
    validate_decision_record_file,
    validate_stream_file,
    validate_trace_record_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis" / "public_backtest" / "stream_adapter_verification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify stream decision adapter equivalence and leakage boundaries."
    )
    parser.add_argument("--suite", choices=["all", "tau", "terminal"], default="all")
    parser.add_argument(
        "--budget-grid",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 0.95],
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_config(source: str) -> dict[str, Any]:
    if source == "tau":
        return {
            "source": "tau",
            "budget_metric": "tested_cost_fraction",
            "threshold": "0.05",
            "extra": [],
            "stream_name": "tau_task_outcome_stream.jsonl",
            "records_name": "tau_stream_decision_records.jsonl",
            "trace_name": "tau_stream_decision_trace.jsonl",
            "equivalence_name": "tau_stream_equivalence.json",
        }
    return {
        "source": "terminal",
        "budget_metric": "tested_time_fraction",
        "threshold": "0.00",
        "extra": [
            "--tested",
            "Ante__Gemini-3-Pro-Preview::2025-12-31__22-36-36",
            "--comparison",
            "Ante__Gemini-3.1-Pro-Preview::2026-02-20__19-31-44",
        ],
        "stream_name": "terminal_task_outcome_stream.jsonl",
        "records_name": "terminal_stream_decision_records.jsonl",
        "trace_name": "terminal_stream_decision_trace.jsonl",
        "equivalence_name": "terminal_stream_equivalence.json",
    }


def run_adapter(config: dict[str, Any], budget_grid: list[float], out_dir: Path, seed: int) -> None:
    cmd = [
        sys.executable,
        "scripts/run_stream_decision_adapter.py",
        "--source",
        config["source"],
        "--budget-metric",
        config["budget_metric"],
        "--threshold",
        config["threshold"],
        "--budget-caps",
        *[str(item) for item in budget_grid],
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        *config["extra"],
    ]
    run_command(cmd)


def run_adapter_on_stream(
    config: dict[str, Any],
    stream_path: Path,
    budget_cap: float,
    out_dir: Path,
    seed: int,
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_stream_decision_adapter.py",
        "--source",
        config["source"],
        "--budget-metric",
        config["budget_metric"],
        "--threshold",
        config["threshold"],
        "--budget-caps",
        str(budget_cap),
        "--seed",
        str(seed),
        "--stream-path",
        str(stream_path),
        "--out-dir",
        str(out_dir),
        "--skip-equivalence-check",
        *config["extra"],
    ]
    run_command(cmd)


def run_adapter_expect_failure(
    config: dict[str, Any],
    stream_path: Path,
    budget_cap: float,
    out_dir: Path,
    seed: int,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "scripts/run_stream_decision_adapter.py",
        "--source",
        config["source"],
        "--budget-metric",
        config["budget_metric"],
        "--threshold",
        config["threshold"],
        "--budget-caps",
        str(budget_cap),
        "--seed",
        str(seed),
        "--stream-path",
        str(stream_path),
        "--out-dir",
        str(out_dir),
        "--skip-equivalence-check",
        *config["extra"],
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return proc.returncode != 0, (proc.stderr or proc.stdout).strip()


def mutate_future_events(stream_path: Path, emission_event_index: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with stream_path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            event = json.loads(line)
            if (
                event.get("event_type") == "task_outcome"
                and int(event["event_index"]) > emission_event_index
            ):
                if isinstance(event.get("outcome"), (int, float)):
                    event["outcome"] = 1.0 - float(event["outcome"])
                resource = event.get("resource", {})
                for key in ["cost_usd", "agent_execution_seconds"]:
                    if isinstance(resource.get(key), (int, float)):
                        resource[key] = float(resource[key]) * 10.0 + 123.0
                event["canary_future_mutation"] = "SHOULD_NOT_APPEAR"
            dst.write(json.dumps(event, sort_keys=True) + "\n")


def write_invalid_stream(source_stream: Path, out_path: Path, mode: str) -> None:
    events = read_jsonl(source_stream)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task_indices = [
        idx for idx, event in enumerate(events) if event.get("event_type") == "task_outcome"
    ]
    if len(task_indices) < 2:
        raise ValueError("Need at least two task events for invalid-stream checks.")

    if mode == "nonmonotone_event_index":
        events[task_indices[1]]["event_index"] = events[task_indices[0]]["event_index"]
    elif mode == "negative_resource":
        resource = events[task_indices[0]].setdefault("resource", {})
        key = "cost_usd" if resource.get("cost_usd") is not None else "agent_execution_seconds"
        resource[key] = -1.0
    elif mode == "duplicate_completed_task":
        max_event_index = max(
            int(event["event_index"])
            for event in events
            if event.get("event_type") == "task_outcome"
        )
        duplicate_events = [dict(events[task_indices[0]]), dict(events[task_indices[1]])]
        for offset, event in enumerate(duplicate_events, start=1):
            event["event_index"] = max_event_index + offset
        events.extend(duplicate_events)
    else:
        raise ValueError(mode)

    with out_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def comparable_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "budget_cap": card["budget_cap"],
        "action": card["decision"]["action"],
        "partial_delta": card["decision"]["partial_delta"],
        "bootstrap_tail_if_promote": card["decision"]["bootstrap_tail_if_promote"],
        "bootstrap_tail_if_reject": card["decision"]["bootstrap_tail_if_reject"],
        "tasks": card["observed"]["tasks"],
        "task_fraction": card["observed"]["task_fraction"],
        "resource_fraction": card["observed"]["resource_fraction"],
        "coverage_ok": card["strata_coverage"]["coverage_ok"],
    }


def comparable_trace(records: list[dict[str, Any]], budget_cap: float) -> list[dict[str, Any]]:
    comparable = []
    for record in records:
        if not math.isclose(
            float(record["trace"]["budget_cap"]),
            float(budget_cap),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            continue
        comparable.append(
            {
                "budget_cap": record["trace"]["budget_cap"],
                "prefix_index": record["trace"]["prefix_index"],
                "terminal": record["trace"]["terminal"],
                "terminal_reason": record["trace"]["terminal_reason"],
                "would_decide_if_cap_now": record["trace"]["would_decide_if_cap_now"],
                "action": record["decision"]["action"],
                "continue_reason": record["decision"].get("continue_reason"),
                "abstention_reason": record["decision"].get("abstention_reason"),
                "partial_delta": record["decision"]["partial_delta"],
                "bootstrap_tail_if_promote": record["decision"]["bootstrap_tail_if_promote"],
                "bootstrap_tail_if_reject": record["decision"]["bootstrap_tail_if_reject"],
                "tasks": record["observed"]["tasks"],
                "task_fraction": record["observed"]["task_fraction"],
                "resource_fraction": record["observed"]["resource_fraction"],
                "coverage_ok": record["strata_coverage"]["coverage_ok"],
                "emission_event_index": record["event_stream"]["emission_event_index"],
            }
        )
    return comparable


def values_match(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return (
            left is None
            and right is None
            or left is not None
            and right is not None
            and math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)
        )
    return left == right


def cards_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_cmp = comparable_card(left)
    right_cmp = comparable_card(right)
    return all(values_match(left_cmp[key], right_cmp[key]) for key in left_cmp)


def trace_lists_match(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        if left_row.keys() != right_row.keys():
            return False
        if not all(values_match(left_row[key], right_row[key]) for key in left_row):
            return False
    return True


def flatten_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_paths(child, path))
        return rows
    if isinstance(value, list):
        rows = []
        for idx, child in enumerate(value):
            rows.extend(flatten_paths(child, f"{prefix}[{idx}]"))
        return rows
    return [(prefix, value)]


def metric_resource(row: dict[str, Any], metric: str) -> float:
    if metric == "task_fraction":
        return 1.0
    if metric == "tested_cost_fraction":
        return float(row["tested"]["resource"].get("cost_usd") or 0.0)
    if metric == "pair_cost_fraction":
        return float(row["tested"]["resource"].get("cost_usd") or 0.0) + float(
            row["comparison"]["resource"].get("cost_usd") or 0.0
        )
    if metric == "tested_time_fraction":
        return float(row["tested"]["resource"].get("agent_execution_seconds") or 0.0)
    if metric == "pair_time_fraction":
        return float(row["tested"]["resource"].get("agent_execution_seconds") or 0.0) + float(
            row["comparison"]["resource"].get("agent_execution_seconds") or 0.0
        )
    raise ValueError(metric)


def paired_rows_from_stream(stream_events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = stream_events[0]
    pending: dict[str, dict[str, Any]] = {}
    rows = []
    for event in stream_events[1:]:
        if event.get("event_type") != "task_outcome":
            continue
        task_key = str(event["task_key"])
        role = str(event["role"])
        pending.setdefault(task_key, {})[role] = event
        if {"tested", "comparison"}.issubset(pending[task_key]):
            rows.append(pending.pop(task_key))
    if pending:
        raise ValueError("Unpaired stream rows in accounting check.")
    return manifest, rows


def budget_accounting_rows(
    source: str,
    stream_path: Path,
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stream_events = read_jsonl(stream_path)
    manifest, paired_rows = paired_rows_from_stream(stream_events)
    metric = str(manifest["budget_metric"])
    denominator = float(manifest["budget_denominator"])
    rows = []
    prev_tasks = -1
    prev_resource = -1.0
    for card in cards:
        tasks = int(card["observed"]["tasks"])
        resources = [metric_resource(row, metric) for row in paired_rows[:tasks]]
        expected_fraction = (
            tasks / int(manifest["total_tasks"])
            if metric == "task_fraction"
            else sum(resources) / denominator
        )
        observed_fraction = float(card["observed"]["resource_fraction"])
        rows.append(
            {
                "source": source,
                "budget_cap": card["budget_cap"],
                "tasks": tasks,
                "resource_fraction": observed_fraction,
                "expected_resource_fraction": expected_fraction,
                "resource_match": math.isclose(
                    observed_fraction, expected_fraction, rel_tol=0, abs_tol=1e-12
                ),
                "tasks_nondecreasing": tasks >= prev_tasks,
                "resource_nondecreasing": observed_fraction >= prev_resource,
            }
        )
        prev_tasks = tasks
        prev_resource = observed_fraction
    return rows


def trace_integrity_rows(source: str, trace_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_budget: dict[float, list[dict[str, Any]]] = {}
    for record in trace_records:
        by_budget.setdefault(float(record["trace"]["budget_cap"]), []).append(record)
    for budget_cap, records in sorted(by_budget.items()):
        prefixes = [int(record["trace"]["prefix_index"]) for record in records]
        terminal_positions = [
            idx for idx, record in enumerate(records) if bool(record["trace"]["terminal"])
        ]
        nonterminal_actions = [
            str(record["decision"]["action"])
            for record in records
            if not bool(record["trace"]["terminal"])
        ]
        terminal_action = str(records[-1]["decision"]["action"]) if records else ""
        terminal_reason = str(records[-1]["trace"]["terminal_reason"]) if records else ""
        resource_fractions = [
            float(record["observed"]["resource_fraction"]) for record in records
        ]
        rows.append(
            {
                "source": source,
                "budget_cap": budget_cap,
                "trace_records": len(records),
                "prefixes_increasing": prefixes == sorted(set(prefixes)),
                "one_terminal_at_end": terminal_positions == [len(records) - 1],
                "nonterminal_actions_continue": all(
                    action == "continue" for action in nonterminal_actions
                ),
                "terminal_action_valid": terminal_action in {"promote", "reject", "abstain"},
                "terminal_reason_valid": terminal_reason
                in {"supported_decision", "budget_cap_reached"},
                "resource_nondecreasing": all(
                    later >= earlier
                    for earlier, later in zip(resource_fractions, resource_fractions[1:])
                ),
                "continue_count": sum(
                    1 for record in records if record["decision"]["action"] == "continue"
                ),
            }
        )
    return rows


def leakage_rows(source: str, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    forbidden = {
        "full_delta",
        "full_record_decision",
        "matches_full_record_when_decided",
    }
    rows = []
    for card in cards:
        for path, value in flatten_paths(card):
            key = path.split(".")[-1]
            if key in forbidden and not path.startswith("audit_only_full_record."):
                rows.append(
                    {
                        "source": source,
                        "budget_cap": card["budget_cap"],
                        "path": path,
                        "status": "FAIL",
                        "reason": "completed-record field outside audit_only_full_record",
                    }
                )
            if isinstance(value, str) and "SHOULD_NOT_APPEAR" in value:
                rows.append(
                    {
                        "source": source,
                        "budget_cap": card["budget_cap"],
                        "path": path,
                        "status": "FAIL",
                        "reason": "future mutation canary appeared in output",
                    }
                )
    if not rows:
        rows.append(
            {
                "source": source,
                "budget_cap": "all",
                "path": "",
                "status": "PASS",
                "reason": "completed-record fields confined to audit keys; no canary strings found",
            }
        )
    return rows


def verify_source(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = str(config["source"])
    source_out = args.out_dir / source
    run_adapter(config, args.budget_grid, source_out, args.seed)
    stream_path = source_out / config["stream_name"]
    records_path = source_out / config["records_name"]
    trace_path = source_out / config["trace_name"]
    equivalence_path = source_out / config["equivalence_name"]
    cards = read_jsonl(records_path)
    trace_records = read_jsonl(trace_path)
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    schema_validation = {
        "stream": validate_stream_file(stream_path),
        "records": validate_decision_record_file(records_path),
        "trace": validate_trace_record_file(trace_path),
    }

    prefix_results = []
    for card in cards:
        budget = float(card["budget_cap"])
        emission_event_index = int(card["event_stream"]["emission_event_index"])
        mutated_stream = (
            args.out_dir
            / "prefix_isolation_mutations"
            / source
            / f"budget_{int(round(budget * 10000))}.jsonl"
        )
        mutate_future_events(stream_path, emission_event_index, mutated_stream)
        mutation_out = (
            args.out_dir
            / "prefix_isolation_runs"
            / source
            / f"budget_{int(round(budget * 10000))}"
        )
        run_adapter_on_stream(config, mutated_stream, budget, mutation_out, args.seed)
        mutated_records = read_jsonl(mutation_out / config["records_name"])
        mutated_trace = read_jsonl(mutation_out / config["trace_name"])
        if len(mutated_records) != 1:
            match = False
            reason = f"expected one record, got {len(mutated_records)}"
        else:
            card_match = cards_match(card, mutated_records[0])
            trace_match = trace_lists_match(
                comparable_trace(trace_records, budget),
                comparable_trace(mutated_trace, budget),
            )
            match = card_match and trace_match
            if match:
                reason = "match"
            elif not card_match:
                reason = "budget-cap record changed after future mutation"
            else:
                reason = "decision trace changed after future mutation"
        prefix_results.append(
            {
                "source": source,
                "budget_cap": budget,
                "emission_event_index": emission_event_index,
                "prefix_isolation_passed": match,
                "reason": reason,
            }
        )

    accounting = budget_accounting_rows(source, stream_path, cards)
    trace_integrity = trace_integrity_rows(source, trace_records)
    leakage = leakage_rows(source, cards + trace_records)
    invalid_results = []
    for mode in ["nonmonotone_event_index", "negative_resource", "duplicate_completed_task"]:
        invalid_stream = args.out_dir / "invalid_streams" / source / f"{mode}.jsonl"
        write_invalid_stream(stream_path, invalid_stream, mode)
        failed_closed, message = run_adapter_expect_failure(
            config,
            invalid_stream,
            float(cards[0]["budget_cap"]),
            args.out_dir / "invalid_stream_runs" / source / mode,
            args.seed,
        )
        invalid_results.append(
            {
                "source": source,
                "mode": mode,
                "failed_closed": failed_closed,
                "message": message.splitlines()[-1] if message else "",
            }
        )
    cap_action_set = sorted({str(card["decision"]["action"]) for card in cards})
    trace_action_set = sorted({str(record["decision"]["action"]) for record in trace_records})
    action_set = sorted(set(cap_action_set).union(trace_action_set))
    return {
        "source": source,
        "equivalence": equivalence,
        "schema_validation": schema_validation,
        "prefix_isolation": prefix_results,
        "budget_accounting": accounting,
        "trace_integrity": trace_integrity,
        "leakage": leakage,
        "invalid_streams": invalid_results,
        "actions_observed": action_set,
        "cap_actions_observed": cap_action_set,
        "trace_actions_observed": trace_action_set,
        "paths": {
            "stream": str(stream_path),
            "records": str(records_path),
            "trace": str(trace_path),
            "equivalence": str(equivalence_path),
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = ["tau", "terminal"] if args.suite == "all" else [args.suite]
    results = [verify_source(source_config(source), args) for source in sources]

    matrix_rows = []
    accounting_rows = []
    trace_rows = []
    prefix_rows = []
    leakage_rows_out = []
    invalid_rows = []
    for result in results:
        source = result["source"]
        equivalence_passed = bool(result["equivalence"]["passed"])
        schema_validation_passed = bool(result["schema_validation"])
        prefix_passed = all(row["prefix_isolation_passed"] for row in result["prefix_isolation"])
        accounting_passed = all(
            row["resource_match"]
            and row["tasks_nondecreasing"]
            and row["resource_nondecreasing"]
            for row in result["budget_accounting"]
        )
        trace_passed = bool(result["trace_integrity"]) and all(
            row["prefixes_increasing"]
            and row["one_terminal_at_end"]
            and row["nonterminal_actions_continue"]
            and row["terminal_action_valid"]
            and row["terminal_reason_valid"]
            and row["resource_nondecreasing"]
            for row in result["trace_integrity"]
        ) and any(row["continue_count"] > 0 for row in result["trace_integrity"])
        leakage_passed = all(row["status"] == "PASS" for row in result["leakage"])
        invalid_passed = all(row["failed_closed"] for row in result["invalid_streams"])
        abstain_covered = "abstain" in result["actions_observed"]
        continue_covered = "continue" in result["actions_observed"]
        matrix_rows.append(
            {
                "source": source,
                "schema_validation_passed": schema_validation_passed,
                "equivalence_passed": equivalence_passed,
                "prefix_isolation_passed": prefix_passed,
                "budget_accounting_passed": accounting_passed,
                "trace_integrity_passed": trace_passed,
                "leakage_passed": leakage_passed,
                "invalid_streams_failed_closed": invalid_passed,
                "cap_actions_observed": ",".join(result["cap_actions_observed"]),
                "trace_actions_observed": ",".join(result["trace_actions_observed"]),
                "actions_observed": ",".join(result["actions_observed"]),
                "abstain_covered": abstain_covered,
                "continue_covered": continue_covered,
                "continue_scope_note": "" if continue_covered else "continue absent from trace",
            }
        )
        accounting_rows.extend(result["budget_accounting"])
        trace_rows.extend(result["trace_integrity"])
        prefix_rows.extend(result["prefix_isolation"])
        leakage_rows_out.extend(result["leakage"])
        invalid_rows.extend(result["invalid_streams"])

    write_csv(
        args.out_dir / "verification_matrix.csv",
        matrix_rows,
        [
            "source",
            "schema_validation_passed",
            "equivalence_passed",
            "prefix_isolation_passed",
            "budget_accounting_passed",
            "trace_integrity_passed",
            "leakage_passed",
            "invalid_streams_failed_closed",
            "cap_actions_observed",
            "trace_actions_observed",
            "actions_observed",
            "abstain_covered",
            "continue_covered",
            "continue_scope_note",
        ],
    )
    write_csv(
        args.out_dir / "decision_trace_integrity_report.csv",
        trace_rows,
        [
            "source",
            "budget_cap",
            "trace_records",
            "prefixes_increasing",
            "one_terminal_at_end",
            "nonterminal_actions_continue",
            "terminal_action_valid",
            "terminal_reason_valid",
            "resource_nondecreasing",
            "continue_count",
        ],
    )
    write_csv(
        args.out_dir / "budget_accounting_report.csv",
        accounting_rows,
        [
            "source",
            "budget_cap",
            "tasks",
            "resource_fraction",
            "expected_resource_fraction",
            "resource_match",
            "tasks_nondecreasing",
            "resource_nondecreasing",
        ],
    )
    write_json(args.out_dir / "prefix_isolation_report.json", prefix_rows)
    write_json(args.out_dir / "invalid_stream_checks.json", invalid_rows)
    write_json(
        args.out_dir / "equivalence_diffs.json",
        {result["source"]: result["equivalence"] for result in results},
    )
    write_json(
        args.out_dir / "adapter_verification_audit.json",
        {
            "budget_grid": args.budget_grid,
            "seed": args.seed,
            "sources": results,
            "summary": matrix_rows,
        },
    )
    lines = ["# Stream Adapter Failure and Leakage Report", ""]
    for row in leakage_rows_out:
        lines.append(
            f"- {row['source']} budget {row['budget_cap']}: {row['status']} ({row['reason']})"
        )
    for row in invalid_rows:
        status = "PASS" if row["failed_closed"] else "FAIL"
        lines.append(
            f"- {row['source']} invalid {row['mode']}: {status} ({row['message']})"
        )
    lines.append("")
    lines.append(
        "Continue actions are emitted in the per-prefix decision trace. "
        "Budget-cap records remain terminal records at configured caps."
    )
    (args.out_dir / "failure_leakage_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    failed = [
        row
        for row in matrix_rows
        if not (
            row["equivalence_passed"]
            and row["schema_validation_passed"]
            and row["prefix_isolation_passed"]
            and row["budget_accounting_passed"]
            and row["trace_integrity_passed"]
            and row["leakage_passed"]
            and row["invalid_streams_failed_closed"]
        )
    ]
    print(json.dumps({"rows": matrix_rows, "failed": failed}, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("Stream adapter verification failed.")


if __name__ == "__main__":
    main()
