from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "public_backtest" / "terminal_bench"
REPO = "harborframework/terminal-bench-2-leaderboard"
BASE_PATH = "submissions/terminal-bench/2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Terminal-Bench 2.0 task-level outcome and token/time matrix."
    )
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--base-path", default=BASE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--systems",
        nargs="*",
        default=[],
        help="Optional exact top-level submission names. Default scans sorted submissions.",
    )
    parser.add_argument("--max-systems", type=int, default=8)
    parser.add_argument("--max-jobs-per-system", type=int, default=1)
    parser.add_argument("--min-trials-per-job", type=int, default=350)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def cache_name(text: str) -> str:
    cleaned = safe_name(text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:120]}_{digest}.json"


def quote_path(path: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))


def api_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/api/datasets/{repo}/tree/main/{quote_path(path)}"


def resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{quote_path(path)}"


def fetch_json_url(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "aimlsys-tbench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def cached_json(
    url: str,
    cache_path: Path,
    timeout: float,
    force: bool,
) -> Any:
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    data = fetch_json_url(url, timeout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def list_dir(repo: str, path: str, raw_dir: Path, timeout: float, force: bool) -> list[dict[str, Any]]:
    cache = raw_dir / "trees" / cache_name(path)
    data = cached_json(api_url(repo, path), cache, timeout, force)
    if isinstance(data, dict) and "value" in data:
        data = data["value"]
    if not isinstance(data, list):
        raise ValueError(f"Unexpected tree response for {path}: {type(data)}")
    return data


def read_result(repo: str, path: str, raw_dir: Path, timeout: float, force: bool) -> dict[str, Any]:
    cache = raw_dir / "results" / cache_name(path)
    data = cached_json(resolve_url(repo, path), cache, timeout, force)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected result response for {path}: {type(data)}")
    return data


def parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    clean = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def duration_seconds(block: dict[str, Any] | None) -> float | None:
    if not isinstance(block, dict):
        return None
    start = parse_time(block.get("started_at"))
    end = parse_time(block.get("finished_at"))
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def wall_seconds(row: dict[str, Any]) -> float | None:
    start = parse_time(row.get("started_at"))
    end = parse_time(row.get("finished_at"))
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def infer_group(task_name: str) -> str:
    # Terminal-Bench public results do not expose a task-domain field in result.json.
    # This deterministic prefix is used only for replay diagnostics and is audited as inferred.
    parts = task_name.split("-")
    if len(parts) <= 1:
        return task_name
    return parts[0]


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def trial_to_row(system: str, job: str, trial_path: str, data: dict[str, Any]) -> dict[str, Any]:
    agent_result = data.get("agent_result") or {}
    verifier_result = data.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else {}
    task_name = str(data.get("task_name") or data.get("trial_name", "").split("__")[0])
    reward = numeric((rewards or {}).get("reward"))
    exception = data.get("exception_info") or {}
    input_tokens = numeric(agent_result.get("n_input_tokens"))
    cache_tokens = numeric(agent_result.get("n_cache_tokens"))
    output_tokens = numeric(agent_result.get("n_output_tokens"))
    total_tokens = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0.0) + (output_tokens or 0.0)
    return {
        "source": "Terminal-Bench 2.0",
        "system_id": f"{system}::{job}",
        "system_label": system,
        "job_id": job,
        "trial_name": data.get("trial_name"),
        "trial_path": trial_path,
        "task_name": task_name,
        "coverage_group": infer_group(task_name),
        "outcome": reward,
        "n_input_tokens": input_tokens,
        "n_cache_tokens": cache_tokens,
        "n_output_tokens": output_tokens,
        "n_total_tokens": total_tokens,
        "cost_usd": numeric(agent_result.get("cost_usd")),
        "wall_seconds": wall_seconds(data),
        "environment_setup_seconds": duration_seconds(data.get("environment_setup")),
        "agent_setup_seconds": duration_seconds(data.get("agent_setup")),
        "agent_execution_seconds": duration_seconds(data.get("agent_execution")),
        "verifier_seconds": duration_seconds(data.get("verifier")),
        "exception_type": exception.get("exception_type") if isinstance(exception, dict) else None,
        "model_name": ((data.get("agent_info") or {}).get("model_info") or {}).get("name"),
        "provider": ((data.get("agent_info") or {}).get("model_info") or {}).get("provider"),
        "raw_agent_name": (data.get("agent_info") or {}).get("name"),
    }


def summarize_task_rows(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trial_rows:
        grouped.setdefault((str(row["system_id"]), str(row["task_name"])), []).append(row)
    task_rows: list[dict[str, Any]] = []
    for (system_id, task_name), rows in sorted(grouped.items()):
        values = [numeric(row.get("outcome")) for row in rows]
        outcomes = [value for value in values if value is not None]
        def mean_col(name: str) -> float | None:
            vals = [numeric(row.get(name)) for row in rows]
            vals = [value for value in vals if value is not None]
            return sum(vals) / len(vals) if vals else None

        def sum_col(name: str) -> float | None:
            vals = [numeric(row.get(name)) for row in rows]
            vals = [value for value in vals if value is not None]
            return sum(vals) if vals else None

        first = rows[0]
        task_rows.append(
            {
                "source": "Terminal-Bench 2.0",
                "system_id": system_id,
                "system_label": first.get("system_label"),
                "job_id": first.get("job_id"),
                "task_name": task_name,
                "coverage_group": first.get("coverage_group"),
                "outcome": sum(outcomes) / len(outcomes) if outcomes else None,
                "success_count": sum(outcomes) if outcomes else None,
                "trials_observed": len(rows),
                "outcome_trials_observed": len(outcomes),
                "mean_input_tokens": mean_col("n_input_tokens"),
                "mean_cache_tokens": mean_col("n_cache_tokens"),
                "mean_output_tokens": mean_col("n_output_tokens"),
                "mean_total_tokens": mean_col("n_total_tokens"),
                "sum_total_tokens": sum_col("n_total_tokens"),
                "mean_cost_usd": mean_col("cost_usd"),
                "sum_cost_usd": sum_col("cost_usd"),
                "mean_wall_seconds": mean_col("wall_seconds"),
                "sum_wall_seconds": sum_col("wall_seconds"),
                "mean_agent_execution_seconds": mean_col("agent_execution_seconds"),
                "sum_agent_execution_seconds": sum_col("agent_execution_seconds"),
                "exception_count": sum(1 for row in rows if row.get("exception_type")),
                "model_name": first.get("model_name"),
                "provider": first.get("provider"),
                "raw_agent_name": first.get("raw_agent_name"),
            }
        )
    return task_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raw_dir = args.out_dir / "raw"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    top = list_dir(args.repo, args.base_path, raw_dir, args.timeout, args.force)
    system_dirs = [row for row in top if row.get("type") == "directory"]
    if args.systems:
        wanted = set(args.systems)
        system_dirs = [
            row for row in system_dirs if str(row.get("path", "")).split("/")[-1] in wanted
        ]
    system_dirs = sorted(system_dirs, key=lambda row: str(row.get("path", "")))
    if args.max_systems is not None:
        system_dirs = system_dirs[: args.max_systems]

    audit_rows: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    for system_row in system_dirs:
        system_path = str(system_row["path"])
        system_name = system_path.split("/")[-1]
        children = list_dir(args.repo, system_path, raw_dir, args.timeout, args.force)
        job_dirs = [row for row in children if row.get("type") == "directory"]
        job_dirs = sorted(job_dirs, key=lambda row: str(row.get("path", "")))
        if args.max_jobs_per_system is not None:
            job_dirs = job_dirs[: args.max_jobs_per_system]
        for job_row in job_dirs:
            job_path = str(job_row["path"])
            job_name = job_path.split("/")[-1]
            job_children = list_dir(args.repo, job_path, raw_dir, args.timeout, args.force)
            trial_dirs = [row for row in job_children if row.get("type") == "directory"]
            trial_dirs = sorted(trial_dirs, key=lambda row: str(row.get("path", "")))
            result_path = f"{job_path}/result.json"
            try:
                job_result = read_result(args.repo, result_path, raw_dir, args.timeout, args.force)
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                audit_rows.append(
                    {
                        "system_name": system_name,
                        "job_name": job_name,
                        "status": "job_result_failed",
                        "trial_dirs": len(trial_dirs),
                        "trial_rows": 0,
                        "task_rows": 0,
                        "error": repr(exc),
                    }
                )
                continue
            if len(trial_dirs) < args.min_trials_per_job:
                audit_rows.append(
                    {
                        "system_name": system_name,
                        "job_name": job_name,
                        "status": "too_few_trial_dirs",
                        "trial_dirs": len(trial_dirs),
                        "trial_rows": 0,
                        "task_rows": 0,
                        "error": "",
                    }
                )
                continue

            rows: list[dict[str, Any]] = []
            errors = 0

            def load_trial(row: dict[str, Any]) -> dict[str, Any] | None:
                trial_path = str(row["path"])
                try:
                    data = read_result(
                        args.repo,
                        f"{trial_path}/result.json",
                        raw_dir,
                        args.timeout,
                        args.force,
                    )
                except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
                    return None
                return trial_to_row(system_name, job_name, trial_path, data)

            start = time.time()
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(load_trial, row) for row in trial_dirs]
                for future in as_completed(futures):
                    item = future.result()
                    if item is None:
                        errors += 1
                    else:
                        rows.append(item)
            task_count = len({row["task_name"] for row in rows})
            all_trials.extend(rows)
            audit_rows.append(
                {
                    "system_name": system_name,
                    "job_name": job_name,
                    "status": "ok",
                    "trial_dirs": len(trial_dirs),
                    "trial_rows": len(rows),
                    "task_rows": task_count,
                    "missing_trial_results": errors,
                    "job_n_total_trials": job_result.get("n_total_trials"),
                    "job_started_at": job_result.get("started_at"),
                    "job_finished_at": job_result.get("finished_at"),
                    "seconds": round(time.time() - start, 2),
                    "error": "",
                }
            )
            print(
                f"{system_name}: {len(rows)} trial rows, {task_count} tasks, {errors} missing",
                flush=True,
            )

    task_rows = summarize_task_rows(all_trials)
    write_csv(args.out_dir / "terminal_bench_trial_rows.csv", all_trials)
    write_csv(args.out_dir / "terminal_bench_task_rows.csv", task_rows)
    write_csv(args.out_dir / "terminal_bench_matrix_audit.csv", audit_rows)
    field_audit = {
        "repo": args.repo,
        "base_path": args.base_path,
        "systems_requested": args.systems,
        "max_systems": args.max_systems,
        "trial_rows": len(all_trials),
        "task_rows": len(task_rows),
        "systems": len({row["system_id"] for row in task_rows}),
        "tasks": len({row["task_name"] for row in task_rows}),
        "native_cost_usd_non_null_trials": sum(
            1 for row in all_trials if row.get("cost_usd") not in (None, "")
        ),
        "token_non_null_trials": sum(
            1 for row in all_trials if row.get("n_total_tokens") not in (None, "")
        ),
        "wall_time_non_null_trials": sum(
            1 for row in all_trials if row.get("wall_seconds") not in (None, "")
        ),
        "agent_time_non_null_trials": sum(
            1 for row in all_trials if row.get("agent_execution_seconds") not in (None, "")
        ),
        "coverage_group_source": "inferred from task-name prefix; not a native Terminal-Bench domain label",
    }
    (args.out_dir / "terminal_bench_matrix_audit.json").write_text(
        json.dumps(field_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
