from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from run_decision_layer_demo import ANALYSIS, default_systems
from run_stream_decision_adapter import (
    batch_cards,
    compare_stream_to_batch,
    consume_stream,
    make_decision_traces,
    prepare_pair,
    validate_stream_outputs,
    write_generated_stream,
    write_jsonl,
    write_summary as write_stream_summary,
)
from stream_decision_schema import (
    validate_decision_record_file,
    validate_stream_file,
    validate_trace_record_file,
)


DEFAULT_OUT = ANALYSIS / "runner_replay_pipeline"


@dataclass(frozen=True)
class PipelineCase:
    source: str
    budget_metric: str
    threshold: float
    coverage_mode: str
    tested: str | None = None
    comparison: str | None = None


PIPELINE_CASES = {
    "tau": PipelineCase(
        source="tau",
        budget_metric="tested_cost_fraction",
        threshold=0.05,
        coverage_mode="source",
    ),
    "terminal": PipelineCase(
        source="terminal",
        budget_metric="tested_time_fraction",
        threshold=0.00,
        coverage_mode="none",
        tested="Ante__Gemini-3-Pro-Preview::2025-12-31__22-36-36",
        comparison="Ante__Gemini-3.1-Pro-Preview::2026-02-20__19-31-44",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an end-to-end check of the task-outcome path into the decision layer. "
            "The script turns compact public task tables into task-outcome "
            "streams, validates the streams, emits JSON decision records, "
            "and checks equivalence to the batch prefix driver."
        )
    )
    parser.add_argument("--suite", choices=["all", "tau", "terminal"], default="all")
    parser.add_argument(
        "--budget-caps",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 0.95],
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--ordering",
        choices=["coverage_random", "coverage_resource"],
        default="coverage_resource",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def selected_cases(suite: str) -> list[PipelineCase]:
    if suite == "all":
        return [PIPELINE_CASES["tau"], PIPELINE_CASES["terminal"]]
    return [PIPELINE_CASES[suite]]


def action_set(records: list[dict[str, Any]]) -> str:
    return ", ".join(sorted({str(record["decision"]["action"]) for record in records}))


def full_record_audit(pair_df: Any, threshold: float) -> dict[str, Any]:
    diffs = (
        pair_df["outcome_tested"].to_numpy(dtype=float)
        - pair_df["outcome_comparison"].to_numpy(dtype=float)
    )
    full_delta = float(diffs.mean())
    return {
        "full_delta": full_delta,
        "full_record_decision": "promote" if full_delta > threshold else "reject",
    }


def run_case(
    case: PipelineCase,
    budget_caps: list[float],
    alpha: float,
    ordering: str,
    bootstrap_samples: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    tested, comparison = default_systems(case.source)
    tested = case.tested or tested
    comparison = case.comparison or comparison

    case_dir = out_dir / case.source
    case_dir.mkdir(parents=True, exist_ok=True)
    stream_path = case_dir / f"{case.source}_runner_task_outcome_stream.jsonl"
    records_path = case_dir / f"{case.source}_pipeline_decision_records.jsonl"
    trace_path = case_dir / f"{case.source}_pipeline_decision_trace.jsonl"
    equivalence_path = case_dir / f"{case.source}_pipeline_equivalence.json"
    adapter_summary_path = case_dir / f"{case.source}_pipeline_adapter_summary.md"

    source_label, pair_df = prepare_pair(
        case.source,
        tested,
        comparison,
        case.coverage_mode,
    )
    run_id = f"{case.source}_runner_replay_pipeline"

    write_generated_stream(
        pair_df,
        source_label,
        case.source,
        tested,
        comparison,
        case.budget_metric,
        ordering,
        case.coverage_mode,
        seed,
        stream_path,
        run_id,
    )
    stream_stats = validate_stream_file(stream_path)
    audit = full_record_audit(pair_df, case.threshold)

    manifest, paired_records, decision_records = consume_stream(
        stream_path,
        sorted(budget_caps),
        case.threshold,
        alpha,
        bootstrap_samples,
        seed,
        audit,
    )
    trace_records = make_decision_traces(
        manifest,
        paired_records,
        sorted(budget_caps),
        case.threshold,
        alpha,
        bootstrap_samples,
        audit,
    )
    validate_stream_outputs(decision_records, trace_records)
    write_jsonl(decision_records, records_path)
    write_jsonl(trace_records, trace_path)
    decision_stats = validate_decision_record_file(records_path)
    trace_stats = validate_trace_record_file(trace_path)

    reference_records = batch_cards(
        pair_df,
        source_label,
        tested,
        comparison,
        case.threshold,
        alpha,
        sorted(budget_caps),
        case.budget_metric,
        ordering,
        case.coverage_mode,
        bootstrap_samples,
        seed,
    )
    equivalence = compare_stream_to_batch(decision_records, reference_records)
    equivalence_path.write_text(
        json.dumps(equivalence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not equivalence["passed"]:
        raise SystemExit(f"Pipeline equivalence failed: {equivalence_path}")

    write_stream_summary(
        adapter_summary_path,
        stream_path,
        records_path,
        trace_path,
        equivalence_path,
        decision_records,
        trace_records,
        equivalence,
    )

    budget_rows = []
    for record in decision_records:
        budget_rows.append(
            {
                "budget_cap": float(record["budget_cap"]),
                "tasks": int(record["observed"]["tasks"]),
                "resource_fraction": float(record["observed"]["resource_fraction"]),
                "action": str(record["decision"]["action"]),
                "coverage_ok": bool(record["strata_coverage"]["coverage_ok"]),
                "emission_event_index": int(record["event_stream"]["emission_event_index"]),
            }
        )

    return {
        "source": case.source,
        "source_label": source_label,
        "systems": {"tested": tested, "comparison": comparison},
        "threshold": case.threshold,
        "budget_metric": case.budget_metric,
        "coverage_mode": case.coverage_mode,
        "task_rows": int(len(pair_df)),
        "stream_stats": stream_stats,
        "decision_stats": decision_stats,
        "trace_stats": trace_stats,
        "decision_actions": action_set(decision_records),
        "trace_actions": action_set(trace_records),
        "equivalence_passed": bool(equivalence["passed"]),
        "budget_rows": budget_rows,
        "paths": {
            "stream": str(stream_path),
            "decision_records": str(records_path),
            "trace_records": str(trace_path),
            "equivalence": str(equivalence_path),
            "adapter_summary": str(adapter_summary_path),
        },
    }


def write_pipeline_report(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    report_path = out_dir / "runner_replay_pipeline_report.md"
    lines = [
        "# Task-Outcome Pipeline",
        "",
        "This report exercises the artifact path from compact public benchmark tables "
        "to task-outcome streams, schema validation, decision-record "
        "emission, per-prefix traces, and batch-equivalence checks.",
        "",
        "The pipeline starts after task execution. It does not run agent tasks, "
        "deploy a benchmark service, or validate an online stopping guarantee.",
        "",
        "## Stage Checks",
        "",
        "| Source | Task rows | Stream records | Decision records | Trace records | Decision actions | Trace actions | Equivalence |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['source_label']} | "
            f"{row['task_rows']} | "
            f"{row['stream_stats']['records']} | "
            f"{row['decision_stats']['records']} | "
            f"{row['trace_stats']['records']} | "
            f"{row['decision_actions']} | "
            f"{row['trace_actions']} | "
            f"{'PASS' if row['equivalence_passed'] else 'FAIL'} |"
        )

    lines.extend(["", "## Budget-Cap Outputs", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['source_label']}",
                "",
                "| Budget | Tasks | Resource fraction | Action | Coverage | Emission event |",
                "| ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for budget_row in row["budget_rows"]:
            coverage = "yes" if budget_row["coverage_ok"] else "no"
            lines.append(
                "| "
                f"{budget_row['budget_cap'] * 100:.0f}% | "
                f"{budget_row['tasks']} | "
                f"{budget_row['resource_fraction'] * 100:.1f}% | "
                f"{budget_row['action']} | "
                f"{coverage} | "
                f"{budget_row['emission_event_index']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Output Paths",
            "",
            "| Source | Stream | Decision records | Trace records | Equivalence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        paths = row["paths"]
        lines.append(
            "| "
            f"{row['source_label']} | "
            f"`{paths['stream']}` | "
            f"`{paths['decision_records']}` | "
            f"`{paths['trace_records']}` | "
            f"`{paths['equivalence']}` |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_case(
            case,
            sorted(args.budget_caps),
            args.alpha,
            args.ordering,
            args.bootstrap_samples,
            args.seed,
            args.out_dir,
        )
        for case in selected_cases(args.suite)
    ]
    report_path = write_pipeline_report(rows, args.out_dir)
    json_path = args.out_dir / "runner_replay_pipeline_report.json"
    payload = {
        "passed": all(bool(row["equivalence_passed"]) for row in rows),
        "cases": rows,
        "report_path": str(report_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "cases": [row["source"] for row in rows],
                "report_path": str(report_path),
                "json_path": str(json_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["passed"]:
        raise SystemExit("Task-outcome pipeline failed.")


if __name__ == "__main__":
    main()
