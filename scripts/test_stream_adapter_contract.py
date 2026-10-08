from __future__ import annotations

import argparse
import json
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
DEFAULT_OUT = ROOT / "analysis" / "public_backtest" / "stream_adapter_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic contract tests for the generic stream decision adapter."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def synthetic_stream_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "event_type": "stream_manifest",
            "schema_version": "1.0",
            "source": "Synthetic Agent Benchmark",
            "source_key": "synthetic",
            "run_id": "synthetic_generic_contract",
            "systems": {"tested": "tested-system", "comparison": "comparison-system"},
            "budget_metric": "task_fraction",
            "bootstrap_seed": 20260613,
            "budget_denominator": 4.0,
            "resource_completeness": 1.0,
            "total_tasks": 4,
            "group_counts": {"domain_a": 2, "domain_b": 2},
            "coverage_mode": "source",
            "ordering": "runner_order",
            "note": "Synthetic stream for generic adapter contract tests.",
        }
    ]
    tasks = [
        ("task_a1", "domain_a", 0.8, 0.6),
        ("task_b1", "domain_b", 0.7, 0.5),
        ("task_a2", "domain_a", 0.9, 0.7),
        ("task_b2", "domain_b", 0.8, 0.6),
    ]
    event_index = 0
    for pair_position, (task_key, group, tested_outcome, comparison_outcome) in enumerate(
        tasks,
        start=1,
    ):
        for role, system, outcome in [
            ("tested", "tested-system", tested_outcome),
            ("comparison", "comparison-system", comparison_outcome),
        ]:
            event_index += 1
            rows.append(
                {
                    "event_type": "task_outcome",
                    "schema_version": "1.0",
                    "source": "Synthetic Agent Benchmark",
                    "run_id": "synthetic_generic_contract",
                    "event_index": event_index,
                    "pair_position": pair_position,
                    "role": role,
                    "system": system,
                    "task_key": task_key,
                    "group": group,
                    "outcome": outcome,
                    "resource": {
                        "cost_usd": 0.01 * pair_position,
                        "agent_execution_seconds": 10.0 * pair_position,
                    },
                }
            )
    return rows


def run_adapter(stream_path: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/run_stream_decision_adapter.py",
        "--source",
        "generic",
        "--stream-path",
        str(stream_path),
        "--threshold",
        "0.05",
        "--budget-caps",
        "0.50",
        "1.00",
        "--skip-equivalence-check",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_adapter_expect_failure(stream_path: Path, out_dir: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "scripts/run_stream_decision_adapter.py",
        "--source",
        "generic",
        "--stream-path",
        str(stream_path),
        "--threshold",
        "0.05",
        "--budget-caps",
        "0.50",
        "--skip-equivalence-check",
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return proc.returncode != 0, (proc.stderr or proc.stdout).strip()


def invalid_stream(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    mutated = [json.loads(json.dumps(row)) for row in rows]
    task_indices = [
        index for index, row in enumerate(mutated) if row.get("event_type") == "task_outcome"
    ]
    if mode == "missing_manifest":
        return mutated[1:]
    if mode == "repeated_manifest":
        return mutated[:1] + [json.loads(json.dumps(mutated[0]))] + mutated[1:]
    if mode == "unrecognized_event":
        mutated.insert(1, {"event_type": "heartbeat"})
        return mutated
    if mode == "wrong_run_id":
        mutated[task_indices[0]]["run_id"] = "another-run"
        return mutated
    if mode == "wrong_system":
        mutated[task_indices[0]]["system"] = "another-system"
        return mutated
    if mode == "wrong_source":
        mutated[task_indices[0]]["source"] = "Another Source"
        return mutated
    if mode == "undeclared_group":
        mutated[task_indices[0]]["group"] = "undeclared"
        mutated[task_indices[1]]["group"] = "undeclared"
        return mutated
    if mode == "same_manifest_systems":
        mutated[0]["systems"]["comparison"] = mutated[0]["systems"]["tested"]
        return mutated
    if mode == "nonmonotone_event_index":
        mutated[task_indices[1]]["event_index"] = mutated[task_indices[0]]["event_index"]
        return mutated
    if mode == "bad_role":
        mutated[task_indices[0]]["role"] = "challenger"
        return mutated
    if mode == "negative_resource":
        mutated[task_indices[0]]["resource"]["cost_usd"] = -1.0
        return mutated
    if mode == "group_mismatch":
        mutated[task_indices[1]]["group"] = "wrong_group"
        return mutated
    if mode == "duplicate_completed_task":
        max_index = max(row["event_index"] for row in mutated if row.get("event_type") == "task_outcome")
        duplicate_pair = [json.loads(json.dumps(mutated[task_indices[0]])), json.loads(json.dumps(mutated[task_indices[1]]))]
        for offset, row in enumerate(duplicate_pair, start=1):
            row["event_index"] = max_index + offset
        return mutated + duplicate_pair
    raise ValueError(mode)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stream_path = args.out_dir / "generic_task_outcome_stream.jsonl"
    rows = synthetic_stream_rows()
    write_jsonl(stream_path, rows)
    stream_stats = validate_stream_file(stream_path)

    run_dir = args.out_dir / "generic_run"
    run_adapter(stream_path, run_dir)
    records_path = run_dir / "generic_stream_decision_records.jsonl"
    trace_path = run_dir / "generic_stream_decision_trace.jsonl"
    record_stats = validate_decision_record_file(records_path)
    trace_stats = validate_trace_record_file(trace_path)

    decision_records = read_jsonl(records_path)
    trace_records = read_jsonl(trace_path)
    invalid_results = []
    for mode in [
        "missing_manifest",
        "repeated_manifest",
        "unrecognized_event",
        "wrong_run_id",
        "wrong_system",
        "wrong_source",
        "undeclared_group",
        "same_manifest_systems",
        "nonmonotone_event_index",
        "bad_role",
        "negative_resource",
        "group_mismatch",
        "duplicate_completed_task",
    ]:
        invalid_path = args.out_dir / "invalid_streams" / f"{mode}.jsonl"
        write_jsonl(invalid_path, invalid_stream(rows, mode))
        failed_closed, message = run_adapter_expect_failure(
            invalid_path,
            args.out_dir / "invalid_runs" / mode,
        )
        invalid_results.append(
            {
                "mode": mode,
                "failed_closed": failed_closed,
                "message": message.splitlines()[-1] if message else "",
            }
        )

    actions = sorted({row["decision"]["action"] for row in decision_records + trace_records})
    report = {
        "stream_path": str(stream_path),
        "records_path": str(records_path),
        "trace_path": str(trace_path),
        "stream_stats": stream_stats,
        "record_stats": record_stats,
        "trace_stats": trace_stats,
        "actions_observed": actions,
        "terminal_record_actions": sorted({row["decision"]["action"] for row in decision_records}),
        "trace_actions": sorted({row["decision"]["action"] for row in trace_records}),
        "invalid_streams": invalid_results,
        "passed": all(row["failed_closed"] for row in invalid_results)
        and "continue" in actions
        and record_stats["records"] == 2
        and trace_stats["records"] == 6,
    }
    report_path = args.out_dir / "generic_contract_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Generic Stream Adapter Contract Report",
        "",
        f"Stream: `{stream_path}`",
        f"Decision records: `{records_path}`",
        f"Trace records: `{trace_path}`",
        "",
        f"Actions observed: {', '.join(actions)}",
        f"Terminal records: {record_stats['records']}",
        f"Trace records: {trace_stats['records']}",
        "",
        "| Invalid stream | Failed closed | Message |",
        "| --- | --- | --- |",
    ]
    for row in invalid_results:
        lines.append(
            f"| {row['mode']} | {row['failed_closed']} | {row['message']} |"
        )
    lines.append("")
    lines.append(f"Overall result: {'PASS' if report['passed'] else 'FAIL'}")
    (args.out_dir / "generic_contract_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("Generic stream adapter contract test failed.")


if __name__ == "__main__":
    main()
