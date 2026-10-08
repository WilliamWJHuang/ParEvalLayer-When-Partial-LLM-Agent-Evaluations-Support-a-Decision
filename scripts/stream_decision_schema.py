from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ALLOWED_BUDGET_METRICS = {
    "task_fraction",
    "tested_cost_fraction",
    "pair_cost_fraction",
    "tested_time_fraction",
    "pair_time_fraction",
}
ALLOWED_ROLES = {"tested", "comparison"}
ALLOWED_TERMINAL_ACTIONS = {"promote", "reject", "abstain"}
ALLOWED_TRACE_ACTIONS = {"promote", "reject", "abstain", "continue"}


def _fail(message: str) -> None:
    raise ValueError(message)


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{path} must be an object.")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{path} must be a nonempty string.")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{path} must be a boolean.")
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        _fail(f"{path} must be at least {minimum}.")
    return value


def _require_finite_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{path} must be a finite number.")
    value = float(value)
    if not math.isfinite(value):
        _fail(f"{path} must be finite.")
    return value


def _require_optional_finite_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _require_finite_number(value, path)


def _require_nonnegative_number(value: Any, path: str) -> float:
    value = _require_finite_number(value, path)
    if value < 0:
        _fail(f"{path} must be nonnegative.")
    return value


def _require_probability(value: Any, path: str) -> float:
    value = _require_finite_number(value, path)
    if value < 0 or value > 1:
        _fail(f"{path} must be in [0, 1].")
    return value


def _require_optional_probability(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _require_probability(value, path)


def _require_keys(obj: dict[str, Any], keys: list[str], path: str) -> None:
    for key in keys:
        if key not in obj:
            _fail(f"{path}.{key} is required.")


def validate_stream_manifest(event: dict[str, Any]) -> None:
    _require_mapping(event, "manifest")
    _require_keys(
        event,
        [
            "event_type",
            "source",
            "run_id",
            "systems",
            "budget_metric",
            "bootstrap_seed",
            "budget_denominator",
            "resource_completeness",
            "total_tasks",
            "group_counts",
            "coverage_mode",
            "ordering",
        ],
        "manifest",
    )
    if event["event_type"] != "stream_manifest":
        _fail("manifest.event_type must be stream_manifest.")
    _require_string(event["source"], "manifest.source")
    _require_string(event["run_id"], "manifest.run_id")
    systems = _require_mapping(event["systems"], "manifest.systems")
    tested = _require_string(systems.get("tested"), "manifest.systems.tested")
    comparison = _require_string(systems.get("comparison"), "manifest.systems.comparison")
    if tested == comparison:
        _fail("manifest.systems.tested and manifest.systems.comparison must be distinct.")
    metric = _require_string(event["budget_metric"], "manifest.budget_metric")
    if metric not in ALLOWED_BUDGET_METRICS:
        _fail(f"manifest.budget_metric is unsupported: {metric}.")
    _require_int(event["bootstrap_seed"], "manifest.bootstrap_seed")
    _require_nonnegative_number(event["budget_denominator"], "manifest.budget_denominator")
    if float(event["budget_denominator"]) <= 0:
        _fail("manifest.budget_denominator must be positive.")
    _require_probability(event["resource_completeness"], "manifest.resource_completeness")
    total_tasks = _require_int(event["total_tasks"], "manifest.total_tasks", minimum=1)
    group_counts = _require_mapping(event["group_counts"], "manifest.group_counts")
    counted = 0
    for group, count in group_counts.items():
        _require_string(group, "manifest.group_counts key")
        counted += _require_int(count, f"manifest.group_counts.{group}", minimum=1)
    if counted != total_tasks:
        _fail("manifest.group_counts must sum to manifest.total_tasks.")
    _require_string(event["coverage_mode"], "manifest.coverage_mode")
    _require_string(event["ordering"], "manifest.ordering")


def validate_task_outcome_event(event: dict[str, Any]) -> None:
    _require_mapping(event, "task_outcome")
    _require_keys(
        event,
        [
            "event_type",
            "event_index",
            "run_id",
            "role",
            "system",
            "task_key",
            "group",
            "outcome",
            "resource",
        ],
        "task_outcome",
    )
    if event["event_type"] != "task_outcome":
        _fail("task_outcome.event_type must be task_outcome.")
    _require_int(event["event_index"], "task_outcome.event_index", minimum=1)
    _require_string(event["run_id"], "task_outcome.run_id")
    role = _require_string(event["role"], "task_outcome.role")
    if role not in ALLOWED_ROLES:
        _fail(f"task_outcome.role must be one of {sorted(ALLOWED_ROLES)}.")
    _require_string(event["system"], "task_outcome.system")
    _require_string(event["task_key"], "task_outcome.task_key")
    _require_string(event["group"], "task_outcome.group")
    _require_finite_number(event["outcome"], "task_outcome.outcome")
    resource = _require_mapping(event["resource"], "task_outcome.resource")
    for key in ["cost_usd", "agent_execution_seconds"]:
        if key in resource and resource[key] is not None:
            _require_nonnegative_number(resource[key], f"task_outcome.resource.{key}")


def validate_event_against_manifest(
    event: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Validate task-event identity fields against the stream manifest.

    ``source`` remains optional on task events for compatibility with legacy
    valid streams. When present, it must identify the manifest source.
    """
    validate_task_outcome_event(event)
    if event["run_id"] != manifest["run_id"]:
        _fail("task_outcome.run_id must match manifest.run_id.")
    role = str(event["role"])
    expected_system = manifest["systems"][role]
    if event["system"] != expected_system:
        _fail(f"task_outcome.system must match manifest.systems.{role}.")
    if "source" in event and event["source"] != manifest["source"]:
        _fail("task_outcome.source must match manifest.source when present.")
    if event["group"] not in manifest["group_counts"]:
        _fail("task_outcome.group must be declared in manifest.group_counts.")


def _validate_common_decision_record(record: dict[str, Any], *, trace: bool) -> None:
    path = "trace_record" if trace else "decision_record"
    _require_mapping(record, path)
    _require_keys(
        record,
        [
            "schema_version",
            "source",
            "systems",
            "threshold",
            "alpha",
            "task_selection_policy",
            "budget_cap",
            "observed",
            "strata_coverage",
            "decision",
            "event_stream",
            "audit_only_full_record",
        ],
        path,
    )
    expected_schema = "1.3" if trace else "1.2"
    if record["schema_version"] != expected_schema:
        _fail(f"{path}.schema_version must be {expected_schema}.")
    _require_string(record["source"], f"{path}.source")
    systems = _require_mapping(record["systems"], f"{path}.systems")
    _require_string(systems.get("tested"), f"{path}.systems.tested")
    _require_string(systems.get("comparison"), f"{path}.systems.comparison")
    _require_finite_number(record["threshold"], f"{path}.threshold")
    _require_probability(record["alpha"], f"{path}.alpha")
    _require_nonnegative_number(record["budget_cap"], f"{path}.budget_cap")
    if float(record["budget_cap"]) <= 0:
        _fail(f"{path}.budget_cap must be positive.")

    policy = _require_mapping(record["task_selection_policy"], f"{path}.task_selection_policy")
    _require_string(policy.get("name"), f"{path}.task_selection_policy.name")
    metric = _require_string(
        policy.get("budget_metric"), f"{path}.task_selection_policy.budget_metric"
    )
    if metric not in ALLOWED_BUDGET_METRICS:
        _fail(f"{path}.task_selection_policy.budget_metric is unsupported: {metric}.")
    _require_string(policy.get("coverage_mode"), f"{path}.task_selection_policy.coverage_mode")
    _require_bool(policy.get("prefix_only"), f"{path}.task_selection_policy.prefix_only")

    observed = _require_mapping(record["observed"], f"{path}.observed")
    _require_int(observed.get("tasks"), f"{path}.observed.tasks", minimum=1)
    _require_probability(observed.get("task_fraction"), f"{path}.observed.task_fraction")
    _require_probability(
        observed.get("resource_fraction"), f"{path}.observed.resource_fraction"
    )
    _require_probability(
        observed.get("resource_completeness"), f"{path}.observed.resource_completeness"
    )

    coverage = _require_mapping(record["strata_coverage"], f"{path}.strata_coverage")
    _require_mapping(coverage.get("required_counts"), f"{path}.strata_coverage.required_counts")
    _require_mapping(coverage.get("observed_counts"), f"{path}.strata_coverage.observed_counts")
    if not isinstance(coverage.get("missing_strata"), list):
        _fail(f"{path}.strata_coverage.missing_strata must be an array.")
    _require_bool(coverage.get("coverage_ok"), f"{path}.strata_coverage.coverage_ok")

    decision = _require_mapping(record["decision"], f"{path}.decision")
    action = _require_string(decision.get("action"), f"{path}.decision.action")
    allowed_actions = ALLOWED_TRACE_ACTIONS if trace else ALLOWED_TERMINAL_ACTIONS
    if action not in allowed_actions:
        _fail(f"{path}.decision.action must be one of {sorted(allowed_actions)}.")
    _require_finite_number(decision.get("partial_delta"), f"{path}.decision.partial_delta")
    _require_optional_probability(
        decision.get("bootstrap_tail_if_promote"), f"{path}.decision.bootstrap_tail_if_promote"
    )
    _require_optional_probability(
        decision.get("bootstrap_tail_if_reject"), f"{path}.decision.bootstrap_tail_if_reject"
    )

    event_stream = _require_mapping(record["event_stream"], f"{path}.event_stream")
    _require_string(event_stream.get("run_id"), f"{path}.event_stream.run_id")
    _require_int(
        event_stream.get("paired_tasks_buffered"),
        f"{path}.event_stream.paired_tasks_buffered",
        minimum=1,
    )
    _require_nonnegative_number(
        event_stream.get("budget_denominator"), f"{path}.event_stream.budget_denominator"
    )
    if not isinstance(record["audit_only_full_record"], dict):
        _fail(f"{path}.audit_only_full_record must be an object.")


def validate_terminal_decision_record(record: dict[str, Any]) -> None:
    _validate_common_decision_record(record, trace=False)


def validate_trace_record(record: dict[str, Any]) -> None:
    _validate_common_decision_record(record, trace=True)
    if record.get("record_type") != "decision_trace_step":
        _fail("trace_record.record_type must be decision_trace_step.")
    trace = _require_mapping(record.get("trace"), "trace_record.trace")
    _require_nonnegative_number(trace.get("budget_cap"), "trace_record.trace.budget_cap")
    _require_int(trace.get("prefix_index"), "trace_record.trace.prefix_index", minimum=1)
    _require_bool(trace.get("terminal"), "trace_record.trace.terminal")
    if trace.get("terminal_reason") is not None:
        _require_string(trace.get("terminal_reason"), "trace_record.trace.terminal_reason")
    action = _require_string(
        trace.get("would_decide_if_cap_now"),
        "trace_record.trace.would_decide_if_cap_now",
    )
    if action not in ALLOWED_TERMINAL_ACTIONS:
        _fail("trace_record.trace.would_decide_if_cap_now must be terminal action.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"{path}:{line_number}: invalid JSON: {exc}")
            if not isinstance(row, dict):
                _fail(f"{path}:{line_number}: JSONL rows must be objects.")
            rows.append(row)
    if not rows:
        _fail(f"{path} is empty.")
    return rows


def validate_stream_file(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    validate_stream_manifest(rows[0])
    manifest = rows[0]
    last_index = 0
    task_events = 0
    for row in rows[1:]:
        event_type = row.get("event_type")
        if event_type == "stream_manifest":
            _fail(f"{path}: stream must contain exactly one manifest, as its first record.")
        if event_type != "task_outcome":
            _fail(f"{path}: unsupported event_type: {event_type!r}.")
        validate_event_against_manifest(row, manifest)
        event_index = int(row["event_index"])
        if event_index <= last_index:
            _fail(f"{path}: event_index must be strictly increasing.")
        last_index = event_index
        task_events += 1
    return {"records": len(rows), "task_events": task_events}


def validate_decision_record_file(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    for row in rows:
        validate_terminal_decision_record(row)
    return {"records": len(rows)}


def validate_trace_record_file(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    for row in rows:
        validate_trace_record(row)
    return {"records": len(rows)}
