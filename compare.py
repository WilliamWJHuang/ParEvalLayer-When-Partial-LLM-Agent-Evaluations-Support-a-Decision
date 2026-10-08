"""Run the ParEvalLayer decision workflow on a paired wide CSV file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_stream_decision_adapter import consume_stream, make_decision_traces  # noqa: E402
from stream_decision_schema import (  # noqa: E402
    validate_stream_file,
    validate_terminal_decision_record,
    validate_trace_record,
)


REQUIRED_HEADERS = {"task_id", "tested_score", "comparison_score"}
ALLOWED_HEADERS = REQUIRED_HEADERS | {"group"}
DEFAULT_OUTPUT = ROOT / "outputs" / "comparison"


def finite_nonnegative(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def probability(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("must be finite and strictly between 0 and 1")
    return value


def budget_cap(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired tested and comparison scores from a wide CSV file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tested", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--threshold", type=finite_nonnegative, required=True)
    parser.add_argument("--budget-caps", type=budget_cap, nargs="+", default=[1.0])
    parser.add_argument("--alpha", type=probability, default=0.05)
    parser.add_argument("--bootstrap-samples", type=positive_int, default=500)
    parser.add_argument("--seed", type=nonnegative_int, default=20260613)
    parser.add_argument("--source", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_label(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be nonempty.")
    return value.strip()


def load_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        raise ValueError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Input CSV is empty.") from exc
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV header: {exc}") from exc
        if len(header) != len(set(header)):
            duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
            raise ValueError(f"Duplicate CSV headers: {', '.join(duplicates)}")
        missing = sorted(REQUIRED_HEADERS - set(header))
        unknown = sorted(set(header) - ALLOWED_HEADERS)
        if missing:
            raise ValueError(f"Missing required CSV headers: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown CSV headers: {', '.join(unknown)}")
        positions = {name: index for index, name in enumerate(header)}
        rows: list[dict[str, Any]] = []
        seen_tasks: set[str] = set()
        try:
            for line_number, fields in enumerate(reader, start=2):
                if len(fields) != len(header):
                    raise ValueError(
                        f"Malformed CSV row {line_number}: expected {len(header)} fields, "
                        f"found {len(fields)}."
                    )
                task_id = fields[positions["task_id"]].strip()
                group = (
                    fields[positions["group"]].strip() if "group" in positions else "all"
                )
                if not task_id:
                    raise ValueError(f"Row {line_number}: task_id must be nonempty.")
                if not group:
                    raise ValueError(f"Row {line_number}: group must be nonempty.")
                if task_id in seen_tasks:
                    raise ValueError(f"Row {line_number}: duplicate task_id {task_id!r}.")
                seen_tasks.add(task_id)
                scores: dict[str, float] = {}
                for column in ("tested_score", "comparison_score"):
                    raw = fields[positions[column]].strip()
                    if not raw:
                        raise ValueError(f"Row {line_number}: {column} is missing.")
                    try:
                        score = float(raw)
                    except ValueError as exc:
                        raise ValueError(
                            f"Row {line_number}: {column} must be numeric."
                        ) from exc
                    if not math.isfinite(score):
                        raise ValueError(f"Row {line_number}: {column} must be finite.")
                    scores[column] = score
                difference = scores["tested_score"] - scores["comparison_score"]
                if not math.isfinite(difference):
                    raise ValueError(
                        f"Row {line_number}: tested_score minus comparison_score "
                        "must be finite on the shared score scale."
                    )
                rows.append({"task_id": task_id, "group": group, **scores})
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV: {exc}") from exc
    if len(rows) < 2:
        raise ValueError("Input CSV must contain at least two paired task rows.")
    return rows, header


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stream_rows(
    rows: list[dict[str, Any]], source: str, run_id: str, tested: str, comparison: str, seed: int
) -> list[dict[str, Any]]:
    group_counts = dict(Counter(str(row["group"]) for row in rows))
    events: list[dict[str, Any]] = [
        {
            "event_type": "stream_manifest",
            "schema_version": "1.0",
            "source": source,
            "source_key": "csv",
            "run_id": run_id,
            "systems": {"tested": tested, "comparison": comparison},
            "budget_metric": "task_fraction",
            "bootstrap_seed": seed,
            "budget_denominator": float(len(rows)),
            "resource_completeness": 1.0,
            "total_tasks": len(rows),
            "group_counts": group_counts,
            "coverage_mode": "source",
            "ordering": "csv_row_order",
            "note": "Normalized from paired wide CSV; rows retain input order.",
        }
    ]
    event_index = 0
    for pair_position, row in enumerate(rows, start=1):
        for role, system, column in (
            ("tested", tested, "tested_score"),
            ("comparison", comparison, "comparison_score"),
        ):
            event_index += 1
            events.append(
                {
                    "event_type": "task_outcome",
                    "schema_version": "1.0",
                    "source": source,
                    "run_id": run_id,
                    "event_index": event_index,
                    "pair_position": pair_position,
                    "role": role,
                    "system": system,
                    "task_key": row["task_id"],
                    "group": row["group"],
                    "outcome": row[column],
                    "resource": {},
                }
            )
    return events


def decision_reason(card: dict[str, Any]) -> str:
    action = card["decision"]["action"]
    if action == "promote":
        return "supported_above_threshold"
    if action == "reject":
        return "supported_not_above_threshold"
    return str(card["decision"].get("abstention_reason") or "uncertainty")


def dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def write_report(
    path: Path,
    input_path: Path,
    source: str,
    tested: str,
    comparison: str,
    threshold: float,
    cards: list[dict[str, Any]],
) -> None:
    lines = [
        "# Paired comparison report",
        "",
        f"Source: `{source}`",
        f"Input: `{input_path}`",
        f"Comparison: `{tested}` minus `{comparison}` on the input score scale.",
        "Scores are interpreted as higher-is-better. Both columns must use the same "
        "meaningful and comparable score scale; this tool does not normalize them.",
        f"Promotion threshold: `{threshold:g}`.",
        "",
        "Each budget result uses a fixed retrospective prefix in CSV row order. "
        "Promote means the resampling rule supports a mean difference above the threshold. "
        "Reject means the rule supports a difference at or below the threshold, so the tested "
        "system has not established the requested improvement. It does not establish that the "
        "tested system is worse. Abstain reflects unresolved resampling uncertainty or missing "
        "group coverage.",
        "",
        "These are fixed-prefix retrospective results. They provide no live monitoring or "
        "anytime-valid guarantee. Multiple budget caps are reported without multiplicity "
        "adjustment. The bootstrap tail fractions are decision-rule quantities, not p-values "
        "or confidence intervals.",
        "",
        "The input hash, row count, and group counts in `run.json` describe the complete "
        "CSV. Each outcome in the table uses only its stated prefix. Results depend on CSV row "
        "order because each cap selects a fixed row prefix.",
        "",
        "See `run.json` for the input metadata and the alpha, resample count, seed, and other "
        "settings used for this run.",
        "",
        "| Cap | Prefix tasks | Mean difference | Action | Reason |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for card in cards:
        lines.append(
            f"| {card['budget_cap']:.6g} | {card['observed']['tasks']} | "
            f"{card['decision']['partial_delta']:.6g} | {card['decision']['action']} | "
            f"{decision_reason(card)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def populate_output(
    stage: Path,
    rows: list[dict[str, Any]],
    input_path: Path,
    input_sha: str,
    source: str,
    run_id: str,
    tested: str,
    comparison: str,
    threshold: float,
    alpha: float,
    budget_caps: list[float],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    normalized = stream_rows(rows, source, run_id, tested, comparison, seed)
    stream_path = stage / "stream.jsonl"
    write_jsonl(stream_path, normalized)
    validate_stream_file(stream_path)
    manifest, paired, cards = consume_stream(
        stream_path, budget_caps, threshold, alpha, bootstrap_samples, seed, None
    )
    traces = make_decision_traces(
        manifest, paired, budget_caps, threshold, alpha, bootstrap_samples, None
    )
    for card in cards:
        validate_terminal_decision_record(card)
    for trace in traces:
        validate_trace_record(trace)
    write_jsonl(stage / "decisions.jsonl", cards)
    write_jsonl(stage / "trace.jsonl", traces)
    with (stage / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["budget_cap", "observed_tasks", "mean_difference", "action", "reason"],
        )
        writer.writeheader()
        for card in cards:
            writer.writerow(
                {
                    "budget_cap": card["budget_cap"],
                    "observed_tasks": card["observed"]["tasks"],
                    "mean_difference": card["decision"]["partial_delta"],
                    "action": card["decision"]["action"],
                    "reason": decision_reason(card),
                }
            )
    write_report(
        stage / "report.md", input_path, source, tested, comparison, threshold, cards
    )
    run_record = {
        "schema_version": "1.0",
        "input": {
            "path": str(input_path),
            "sha256": input_sha,
            "rows": len(rows),
            "group_counts": dict(Counter(str(row["group"]) for row in rows)),
        },
        "parameters": {
            "source": source,
            "run_id": run_id,
            "tested": tested,
            "comparison": comparison,
            "threshold": threshold,
            "alpha": alpha,
            "budget_caps": budget_caps,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "budget_metric": "task_fraction",
            "ordering": "csv_row_order",
        },
        "dependencies": dependency_versions(),
        "outputs": [
            "stream.jsonl",
            "decisions.jsonl",
            "trace.jsonl",
            "summary.csv",
            "report.md",
            "run.json",
        ],
    }
    (stage / "run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cards


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    rows, _ = load_csv(input_path)
    tested = validate_label(args.tested, "--tested")
    comparison = validate_label(args.comparison, "--comparison")
    if tested == comparison:
        raise ValueError("--tested and --comparison must be distinct.")
    if len(args.budget_caps) != len(set(args.budget_caps)):
        raise ValueError("--budget-caps values must be unique.")
    budget_caps = sorted(args.budget_caps)
    source = validate_label(args.source or input_path.stem, "--source")
    input_sha = sha256_file(input_path)
    run_id = validate_label(args.run_id or f"csv-{input_sha[:12]}", "--run-id")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(
            f"Output directory is not empty: {output_dir}. Choose a new --output-dir."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f"{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        cards = populate_output(
            stage,
            rows,
            input_path,
            input_sha,
            source,
            run_id,
            tested,
            comparison,
            args.threshold,
            args.alpha,
            budget_caps,
            args.bootstrap_samples,
            args.seed,
        )
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print("cap\tn\tmean_difference\taction\treason")
    for card in cards:
        print(
            f"{card['budget_cap']:.6g}\t{card['observed']['tasks']}\t"
            f"{card['decision']['partial_delta']:.6g}\t{card['decision']['action']}\t"
            f"{decision_reason(card)}"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
