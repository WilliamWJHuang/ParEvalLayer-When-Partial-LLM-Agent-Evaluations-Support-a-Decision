from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


INSTANCE_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+$")
TASK_KEYS = [
    "generated",
    "no_generation",
    "with_logs",
    "install_fail",
    "reset_failed",
    "no_apply",
    "applied",
    "test_errored",
    "test_timeout",
    "resolved",
]
EXPECTED_TASKS = {
    "lite": 300,
    "verified": 500,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a conversion smoke test for SWE-bench experiments results."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "swebench_experiments_repo",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis") / "public_backtest" / "swebench_smoke",
    )
    parser.add_argument("--min-overlap", type=int, default=150)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10])
    return parser.parse_args()


def instance_repo(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def load_results(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_instance_ids(results: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for key in TASK_KEYS:
        value = results.get(key, [])
        if isinstance(value, list):
            ids.update(str(item) for item in value if INSTANCE_RE.match(str(item)))
    return ids


def build_task_rows(repo_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ["lite", "verified"]:
        split_dir = repo_dir / "evaluation" / split
        if not split_dir.exists():
            continue
        for submission_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            results_path = submission_dir / "results" / "results.json"
            if not results_path.exists():
                continue
            results = load_results(results_path)
            instance_ids = collect_instance_ids(results)
            resolved = {
                str(item)
                for item in results.get("resolved", [])
                if INSTANCE_RE.match(str(item))
            }
            generated = {
                str(item)
                for item in results.get("generated", [])
                if INSTANCE_RE.match(str(item))
            }
            with_logs = {
                str(item)
                for item in results.get("with_logs", [])
                if INSTANCE_RE.match(str(item))
            }
            for instance_id in sorted(instance_ids):
                rows.append(
                    {
                        "split": split,
                        "submission": submission_dir.name,
                        "instance_id": instance_id,
                        "repo": instance_repo(instance_id),
                        "resolved": int(instance_id in resolved),
                        "generated": int(instance_id in generated),
                        "with_logs": int(instance_id in with_logs),
                        "results_json": str(results_path.relative_to(repo_dir)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_submissions(task_rows: pd.DataFrame) -> pd.DataFrame:
    if task_rows.empty:
        return pd.DataFrame()
    summary = (
        task_rows.groupby(["split", "submission"], as_index=False)
        .agg(
            task_count=("instance_id", "nunique"),
            repo_count=("repo", "nunique"),
            resolved_count=("resolved", "sum"),
            generated_count=("generated", "sum"),
            with_logs_count=("with_logs", "sum"),
        )
    )
    summary["resolved_rate"] = summary["resolved_count"] / summary["task_count"]
    summary["with_logs_rate"] = summary["with_logs_count"] / summary["task_count"]
    summary["expected_task_count"] = summary["split"].map(EXPECTED_TASKS)
    summary["task_coverage_rate"] = summary["task_count"] / summary["expected_task_count"]
    summary["near_complete"] = summary["task_coverage_rate"] >= 0.95
    return summary.sort_values(["split", "task_count", "submission"], ascending=[True, False, True])


def summarize_pairs(
    task_rows: pd.DataFrame,
    submission_summary: pd.DataFrame,
    min_overlap: int,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    submission_meta = submission_summary.set_index(["split", "submission"]).to_dict("index")
    for split, split_df in task_rows.groupby("split"):
        submissions = sorted(split_df["submission"].unique())
        by_submission = {
            submission: group.set_index("instance_id")
            for submission, group in split_df.groupby("submission")
        }
        for candidate in submissions:
            cand = by_submission[candidate]
            for reference in submissions:
                if candidate == reference:
                    continue
                ref = by_submission[reference]
                overlap = cand.index.intersection(ref.index)
                if len(overlap) < min_overlap:
                    continue
                cand_meta = submission_meta[(split, candidate)]
                ref_meta = submission_meta[(split, reference)]
                c = cand.loc[overlap]
                r = ref.loc[overlap]
                delta = float(c["resolved"].mean() - r["resolved"].mean())
                repo_count = int(pd.Index(c["repo"]).nunique())
                base = {
                    "split": split,
                    "candidate": candidate,
                    "reference": reference,
                    "candidate_task_count": int(cand_meta["task_count"]),
                    "reference_task_count": int(ref_meta["task_count"]),
                    "candidate_near_complete": bool(cand_meta["near_complete"]),
                    "reference_near_complete": bool(ref_meta["near_complete"]),
                    "near_complete_pair": bool(
                        cand_meta["near_complete"] and ref_meta["near_complete"]
                    ),
                    "overlap_tasks": int(len(overlap)),
                    "repo_count": repo_count,
                    "candidate_resolved_rate": float(c["resolved"].mean()),
                    "reference_resolved_rate": float(r["resolved"].mean()),
                    "full_delta": delta,
                    "with_logs_rate_candidate": float(c["with_logs"].mean()),
                    "with_logs_rate_reference": float(r["with_logs"].mean()),
                }
                for threshold in thresholds:
                    rows.append(
                        {
                            **base,
                            "threshold": threshold,
                            "truth_promote": bool(delta >= threshold),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["split", "threshold", "full_delta"],
        ascending=[True, True, False],
    )


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy().fillna("")
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


def write_report(
    out_dir: Path,
    task_rows: pd.DataFrame,
    submission_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    min_overlap: int,
) -> None:
    lines: list[str] = []
    lines.append("# SWE-bench Conversion Smoke Test\n")
    lines.append("## Status\n")
    lines.append(
        "Completed on locally checked-out SWE-bench experiment metadata and `results/results.json` files. S3 logs and trajectories were not downloaded.\n"
    )
    lines.append("## Data Inventory\n")
    inventory = (
        submission_summary.groupby("split", as_index=False)
        .agg(
            submissions=("submission", "nunique"),
            near_complete_submissions=("near_complete", "sum"),
            median_task_count=("task_count", "median"),
            min_task_count=("task_count", "min"),
            max_task_count=("task_count", "max"),
            median_task_coverage=("task_coverage_rate", "median"),
            median_repo_count=("repo_count", "median"),
        )
        if not submission_summary.empty
        else pd.DataFrame()
    )
    lines.append(markdown_table(inventory))
    lines.append("\n\n## Pair Overlap Gate\n")
    complete_pairs = pair_summary[pair_summary["near_complete_pair"]].copy()
    pair_gate = (
        complete_pairs.groupby(["split", "threshold"], as_index=False)
        .agg(
            directed_pairs=("candidate", "count"),
            positive_reference_decisions=("truth_promote", "sum"),
            min_overlap=("overlap_tasks", "min"),
            median_overlap=("overlap_tasks", "median"),
            max_overlap=("overlap_tasks", "max"),
            median_repo_count=("repo_count", "median"),
        )
        if not complete_pairs.empty
        else pd.DataFrame()
    )
    lines.append(markdown_table(pair_gate))
    lines.append("\n\n## Conversion Gate\n")
    split_rows = []
    for split in ["lite", "verified"]:
        split_pairs = pair_summary[pair_summary["split"].eq(split)]
        complete_split_pairs = split_pairs[split_pairs["near_complete_pair"]]
        split_submissions = submission_summary[submission_summary["split"].eq(split)]
        split_tasks = task_rows[task_rows["split"].eq(split)]
        split_rows.append(
            {
                "split": split,
                "task_level_outcomes": bool(not split_tasks.empty),
                "near_complete_submissions": int(split_submissions["near_complete"].sum()),
                "paired_overlap": bool(complete_split_pairs["candidate"].nunique() >= 2),
                "directed_pairs_ge_6": bool(
                    complete_split_pairs[complete_split_pairs["threshold"].eq(0.05)].shape[0]
                    >= 6
                ),
                "task_groups": bool(split_tasks["repo"].nunique() >= 2),
                "budget_proxy": "task count only",
                "main_paper_status": (
                    "conditional pass"
                    if not complete_split_pairs.empty
                    and complete_split_pairs[
                        complete_split_pairs["threshold"].eq(0.05)
                    ].shape[0]
                    >= 6
                    and split_tasks["repo"].nunique() >= 2
                    else "fail"
                ),
            }
        )
    lines.append(markdown_table(pd.DataFrame(split_rows)))
    lines.append("\n\n## Interpretation\n")
    lines.append(
        "SWE-bench can be converted into a count-budget decision-sufficiency smoke test from repository-level `results.json` files alone, but only after filtering to near-complete submissions. Partial result files can otherwise make resolved rates appear artificially high. The repository metadata does not yet support a cost-aware replay without additional logs or trajectories.\n"
    )
    lines.append(
        "The safest next step is a small count-budget replay over one split, framed only as a scope check. It should enter the main paper only if it can be summarized in one paragraph and does not shift the paper toward coding-agent ranking.\n"
    )
    (out_dir / "swebench_conversion_smoke_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_rows = build_task_rows(args.repo_dir)
    submission_summary = summarize_submissions(task_rows)
    pair_summary = summarize_pairs(
        task_rows,
        submission_summary,
        args.min_overlap,
        args.thresholds,
    )
    task_rows.to_csv(args.out_dir / "swebench_task_rows.csv", index=False)
    submission_summary.to_csv(args.out_dir / "swebench_submission_summary.csv", index=False)
    pair_summary.to_csv(args.out_dir / "swebench_pair_overlap_summary.csv", index=False)
    write_report(args.out_dir, task_rows, submission_summary, pair_summary, args.min_overlap)
    print(
        json.dumps(
            {
                "task_rows": int(len(task_rows)),
                "submissions": int(submission_summary.shape[0]),
                "pair_rows": int(pair_summary.shape[0]),
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
