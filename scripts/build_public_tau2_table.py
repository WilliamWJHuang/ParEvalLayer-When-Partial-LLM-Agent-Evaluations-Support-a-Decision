from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


GITHUB_BASE = (
    "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/"
    "web/leaderboard/public/submissions"
)
S3_BASE = "https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions"

DEFAULT_SUBMISSIONS = [
    "gpt-5-2_sierra_2026-02-26",
    "gpt-5-2-none_sierra_2026-02-26",
    "claude-opus-4-5_sierra_2026-02-26",
    "claude-sonnet-4-5_sierra_2026-02-26",
]
DEFAULT_DOMAINS = ["airline", "retail", "telecom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task-level tables from public tau-bench trajectories."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("analysis") / "public_backtest" / "raw_tau2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis") / "public_backtest" / "tau2_ready",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--submissions", nargs="+", default=DEFAULT_SUBMISSIONS)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--github-base", default=GITHUB_BASE)
    parser.add_argument("--s3-base", default=S3_BASE)
    parser.add_argument(
        "--reward-imputation",
        choices=["failure", "drop", "neutral"],
        default="failure",
        help="How to handle missing infrastructure-error rewards.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def maybe_download(url: str, path: Path, should_download: bool) -> None:
    if path.exists():
        return
    if not should_download:
        raise FileNotFoundError(
            f"Required tau-bench input is missing: {path}. "
            "Provide the local file or rerun with --download after reviewing "
            "the upstream source and license terms."
        )
    download(url, path)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def raw_reward_from_simulation(sim: dict[str, Any]) -> float | None:
    reward_info = sim.get("reward_info")
    if isinstance(reward_info, dict):
        reward = reward_info.get("reward")
        if reward is not None:
            return as_float(reward)
    return as_float(sim.get("reward"))


def cost_sum(agent_cost: float | None, user_cost: float | None) -> float | None:
    values = [value for value in [agent_cost, user_cost] if value is not None]
    if not values:
        return None
    return float(sum(values))


def load_submission(
    raw_dir: Path,
    submission_dir: str,
    github_base: str,
    should_download: bool,
) -> dict[str, Any]:
    path = raw_dir / submission_dir / "submission.json"
    url = f"{github_base}/{submission_dir}/submission.json"
    maybe_download(url, path, should_download)
    submission = read_json(path)
    submission["_submission_path"] = str(path)
    submission["_submission_sha256"] = sha256_file(path)
    submission["_submission_url"] = url
    return submission


def load_trajectory(
    raw_dir: Path,
    submission_dir: str,
    domain: str,
    file_name: str,
    s3_base: str,
    should_download: bool,
) -> tuple[dict[str, Any], Path, str]:
    path = raw_dir / submission_dir / "trajectories" / file_name
    url = f"{s3_base}/{submission_dir}/trajectories/{file_name}"
    maybe_download(url, path, should_download)
    return read_json(path), path, url


def simulation_rows_for_file(
    submission_dir: str,
    submission: dict[str, Any],
    domain: str,
    trajectory: dict[str, Any],
    trajectory_path: Path,
    trajectory_url: str,
    reward_imputation: str,
) -> list[dict[str, Any]]:
    tasks = trajectory.get("tasks") or []
    task_ids = {str(task.get("id")) for task in tasks}
    trajectory_sha256 = sha256_file(trajectory_path)
    rows: list[dict[str, Any]] = []
    for sim in trajectory.get("simulations") or []:
        agent_cost = as_float(sim.get("agent_cost"))
        user_cost = as_float(sim.get("user_cost"))
        task_id = str(sim.get("task_id"))
        messages = sim.get("messages")
        raw_reward = raw_reward_from_simulation(sim)
        reward = raw_reward
        reward_imputation_reason = None
        if reward is None and sim.get("termination_reason") == "infrastructure_error":
            if reward_imputation == "drop":
                continue
            if reward_imputation == "failure":
                reward = 0.0
            elif reward_imputation == "neutral":
                reward = 0.5
            else:
                raise ValueError(reward_imputation)
            reward_imputation_reason = f"infrastructure_error_{reward_imputation}"
        success_int = None if reward is None else int(reward >= 1.0)
        rows.append(
            {
                "benchmark": "tau2_public_leaderboard",
                "submission_dir": submission_dir,
                "model_name": submission.get("model_name"),
                "model_organization": submission.get("model_organization"),
                "submitting_organization": submission.get("submitting_organization"),
                "submission_date": submission.get("submission_date"),
                "submission_type": submission.get("submission_type", "standard"),
                "reasoning_effort": submission.get("reasoning_effort"),
                "evaluation_date": (submission.get("methodology") or {}).get(
                    "evaluation_date"
                ),
                "tau2_bench_version": (submission.get("methodology") or {}).get(
                    "tau2_bench_version"
                ),
                "user_simulator": (submission.get("methodology") or {}).get(
                    "user_simulator"
                ),
                "domain": domain,
                "task_id": task_id,
                "task_present": task_id in task_ids,
                "trial": sim.get("trial"),
                "reward": reward,
                "raw_reward": raw_reward,
                "reward_imputed": reward_imputation_reason is not None,
                "reward_imputation_reason": reward_imputation_reason,
                "success_int": success_int,
                "agent_cost_usd": agent_cost,
                "user_cost_usd": user_cost,
                "evaluation_cost_usd": cost_sum(agent_cost, user_cost),
                "duration_seconds": as_float(sim.get("duration")),
                "termination_reason": sim.get("termination_reason"),
                "num_messages": len(messages) if isinstance(messages, list) else None,
                "trajectory_path": str(trajectory_path),
                "trajectory_url": trajectory_url,
                "trajectory_sha256": trajectory_sha256,
            }
        )
    return rows


def build_task_table(sim_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "benchmark",
        "submission_dir",
        "model_name",
        "model_organization",
        "submitting_organization",
        "submission_date",
        "submission_type",
        "reasoning_effort",
        "evaluation_date",
        "tau2_bench_version",
        "user_simulator",
        "domain",
        "task_id",
    ]
    task_df = (
        sim_df.groupby(group_cols, dropna=False)
        .agg(
            outcome_mean=("reward", "mean"),
            reward_mean=("reward", "mean"),
            trials_observed=("trial", "nunique"),
            simulation_rows=("success_int", "size"),
            success_count=("success_int", "sum"),
            agent_cost_usd=("agent_cost_usd", "mean"),
            user_cost_usd=("user_cost_usd", "mean"),
            cost_usd=("evaluation_cost_usd", "mean"),
            duration_seconds=("duration_seconds", "mean"),
            num_messages=("num_messages", "mean"),
            task_present=("task_present", "all"),
        )
        .reset_index()
    )
    task_df["task_key"] = task_df["domain"].astype(str) + ":" + task_df["task_id"].astype(str)
    return task_df


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    submission_summaries: list[dict[str, Any]] = []
    for submission_dir in args.submissions:
        submission = load_submission(
            args.raw_dir, submission_dir, args.github_base, args.download
        )
        files = submission.get("trajectory_files") or {}
        submission_summaries.append(
            {
                "submission_dir": submission_dir,
                "model_name": submission.get("model_name"),
                "reasoning_effort": submission.get("reasoning_effort"),
                "submission_type": submission.get("submission_type", "standard"),
                "trajectories_available": bool(submission.get("trajectories_available")),
                "domains": sorted(files.keys()),
                "submission_sha256": submission["_submission_sha256"],
            }
        )
        for domain in args.domains:
            file_name = files.get(domain)
            if not file_name:
                skipped.append(
                    {
                        "submission_dir": submission_dir,
                        "domain": domain,
                        "reason": "missing trajectory file",
                    }
                )
                continue
            trajectory, trajectory_path, trajectory_url = load_trajectory(
                args.raw_dir,
                submission_dir,
                domain,
                file_name,
                args.s3_base,
                args.download,
            )
            rows.extend(
                simulation_rows_for_file(
                    submission_dir,
                    submission,
                    domain,
                    trajectory,
                    trajectory_path,
                    trajectory_url,
                    args.reward_imputation,
                )
            )

    if not rows:
        raise ValueError("No simulation rows were parsed.")

    sim_df = pd.DataFrame(rows)
    task_df = build_task_table(sim_df)

    sim_path = args.out_dir / "public_tau2_simulation_rows.csv"
    task_path = args.out_dir / "public_tau2_task_rows.csv"
    summary_path = args.out_dir / "public_tau2_summary.json"
    sim_df.to_csv(sim_path, index=False)
    task_df.to_csv(task_path, index=False)

    summary = {
        "simulation_rows": int(len(sim_df)),
        "task_rows": int(len(task_df)),
        "submissions": submission_summaries,
        "domains": sorted(sim_df["domain"].dropna().unique().tolist()),
        "tasks_per_domain": {
            str(domain): int(count)
            for domain, count in task_df.groupby("domain")["task_id"].nunique().items()
        },
        "trial_count_distribution": {
            str(count): int(freq)
            for count, freq in task_df["trials_observed"].value_counts().sort_index().items()
        },
        "missing_reward_rows": int(sim_df["reward"].isna().sum()),
        "imputed_reward_rows": int(sim_df["reward_imputed"].sum()),
        "reward_imputation": args.reward_imputation,
        "missing_cost_rows": int(sim_df["evaluation_cost_usd"].isna().sum()),
        "skipped": skipped,
        "simulation_table": str(sim_path),
        "task_table": str(task_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
