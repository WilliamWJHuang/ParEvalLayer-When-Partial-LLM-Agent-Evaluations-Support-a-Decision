from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DATASET_SUFFIXES = ("test_normal", "test_challenge")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task-level AppWorld feasibility tables from unpacked leaderboard outputs."
    )
    parser.add_argument(
        "--leaderboard-root",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "appworld_leaderboard_repo",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "appworld_feasibility",
    )
    parser.add_argument(
        "--task-rows",
        type=Path,
        default=Path("analysis")
        / "public_backtest"
        / "appworld_feasibility"
        / "appworld_task_rows.csv",
        help="Optional compact task table. If present, rebuild reports from it instead of raw unpacked bundles.",
    )
    parser.add_argument(
        "--force-leaderboard-parse",
        action="store_true",
        help="Ignore --task-rows and parse unpacked leaderboard bundles.",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10])
    return parser.parse_args()


def method_prefix(experiment_name: str) -> tuple[str, str]:
    for suffix in DATASET_SUFFIXES:
        marker = "_" + suffix
        if experiment_name.endswith(marker):
            return experiment_name[: -len(marker)], suffix
    raise ValueError(f"Cannot infer split from experiment name: {experiment_name}")


def task_group(task_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(.+)_([0-9]+)$", task_id)
    if not match:
        return task_id, None
    return match.group(1), int(match.group(2))


def load_leaderboard(path: Path) -> dict[str, dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for entry in entries:
        method = entry.get("method", {}).get("name", "")
        llm = entry.get("llm", {}).get("name", "")
        system_label = f"{method} / {llm}".strip(" /")
        for split in DATASET_SUFFIXES:
            key = f"{entry['id']}_{split}"
            out[key] = {
                "entry_id": entry["id"],
                "method": method,
                "llm": llm,
                "system_label": system_label,
                "date": entry.get("date", ""),
            }
    return out


def collect_rows(root: Path) -> pd.DataFrame:
    outputs_dir = root / "experiments" / "outputs"
    metadata = load_leaderboard(outputs_dir / "_leaderboard.json")
    rows: list[dict[str, object]] = []
    for evaluation_path in sorted(outputs_dir.glob("*/evaluations/*.json")):
        experiment_name = evaluation_path.parents[1].name
        system_prefix, split = method_prefix(experiment_name)
        expected_dataset = split
        if evaluation_path.stem != expected_dataset:
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        individual = evaluation.get("individual", {})
        if not isinstance(individual, dict) or not individual:
            continue
        meta = metadata.get(system_prefix, {})
        for task_id, task_eval in individual.items():
            group, task_number = task_group(task_id)
            rows.append(
                {
                    "system_id": system_prefix,
                    "experiment_name": experiment_name,
                    "entry_id": meta.get("entry_id", ""),
                    "system_label": meta.get("system_label", system_prefix),
                    "method": meta.get("method", ""),
                    "llm": meta.get("llm", ""),
                    "split": split,
                    "task_id": task_id,
                    "task_key": f"{split}:{task_id}",
                    "scenario_id": group,
                    "task_number": task_number,
                    "difficulty": task_eval.get("difficulty"),
                    "success": bool(task_eval.get("success", False)),
                    "num_tests": task_eval.get("num_tests"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["system_id", "split", "task_id"])


def pairwise_summary(task_df: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems = sorted(task_df["system_id"].unique())
    wide = task_df.pivot_table(
        index="task_key",
        columns="system_id",
        values="success",
        aggfunc="first",
    )
    rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    for tested in systems:
        for comparison in systems:
            if tested == comparison:
                continue
            common = wide[[tested, comparison]].dropna()
            if common.empty:
                continue
            tested_success = common[tested].astype(float)
            comparison_success = common[comparison].astype(float)
            delta = float((tested_success - comparison_success).mean())
            tested_rate = float(tested_success.mean())
            comparison_rate = float(comparison_success.mean())
            rows.append(
                {
                    "tested_system": tested,
                    "comparison_system": comparison,
                    "overlap_tasks": int(len(common)),
                    "tested_success_rate": tested_rate,
                    "comparison_success_rate": comparison_rate,
                    "full_delta": delta,
                }
            )
            for threshold in thresholds:
                truth_rows.append(
                    {
                        "tested_system": tested,
                        "comparison_system": comparison,
                        "threshold": threshold,
                        "overlap_tasks": int(len(common)),
                        "full_delta": delta,
                        "positive_label": bool(delta >= threshold),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(truth_rows)


def write_report(
    out_dir: Path,
    task_df: pd.DataFrame,
    system_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# AppWorld Public Feasibility Summary\n")
    lines.append("## Status\n")
    lines.append(
        "Task-level outcomes were parsed from unpacked public AppWorld leaderboard bundles.\n"
    )
    lines.append("## Dataset Shape\n")
    lines.append(f"- systems: {system_df['system_id'].nunique()}\n")
    lines.append(f"- task rows: {len(task_df)}\n")
    lines.append(f"- unique task keys: {task_df['task_key'].nunique()}\n")
    lines.append(f"- directed pairs: {len(pair_df)}\n")
    lines.append("\n## Label Mix\n")
    if truth_df.empty:
        lines.append("No directed-pair labels were available.\n")
    else:
        mix = (
            truth_df.groupby(["threshold", "positive_label"])
            .size()
            .reset_index(name="count")
            .sort_values(["threshold", "positive_label"])
        )
        lines.append(markdown_table(mix))
        lines.append("\n")
    lines.append("\n## Split Coverage\n")
    split_table = (
        task_df.groupby(["split"])["task_id"]
        .nunique()
        .reset_index(name="unique_tasks")
        .sort_values("split")
    )
    lines.append(markdown_table(split_table))
    lines.append("\n")
    lines.append("\n## Interpretation\n")
    lines.append(
        "AppWorld satisfies the main public-data requirement if all systems retain common task coverage: public task-level success labels, stable task identifiers, difficulty labels, and scenario identifiers. It broadens the evidence beyond coding repair benchmarks, but it remains count-budget evidence unless per-task interaction or cost records are extracted and validated separately.\n"
    )
    lines.append("\n## Caution\n")
    lines.append(
        "The leaderboard README warns against posting unencrypted extracted outputs. The paper should publish scripts and aggregate replay summaries, and should avoid releasing raw extracted task reports unless the benchmark maintainers permit it.\n"
    )
    (out_dir / "appworld_feasibility_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.task_rows.exists() and not args.force_leaderboard_parse:
        task_df = pd.read_csv(args.task_rows)
    else:
        task_df = collect_rows(args.leaderboard_root)
    if task_df.empty:
        raise SystemExit("No AppWorld task-level rows found. Did you unpack bundles?")
    system_df = (
        task_df[
            [
                "system_id",
                "entry_id",
                "system_label",
                "method",
                "llm",
                "experiment_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(["system_id", "experiment_name"])
    )
    pair_df, truth_df = pairwise_summary(task_df, args.thresholds)
    task_df.to_csv(args.out_dir / "appworld_task_rows.csv", index=False)
    system_df.to_csv(args.out_dir / "appworld_systems.csv", index=False)
    pair_df.to_csv(args.out_dir / "appworld_pair_summary.csv", index=False)
    truth_df.to_csv(args.out_dir / "appworld_truth_summary.csv", index=False)
    write_report(args.out_dir, task_df, system_df, pair_df, truth_df)
    print(
        json.dumps(
            {
                "systems": int(system_df["system_id"].nunique()),
                "task_rows": int(len(task_df)),
                "unique_task_keys": int(task_df["task_key"].nunique()),
                "directed_pairs": int(len(pair_df)),
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
