from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Comparison:
    candidate: str
    reference: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize how public tau-bench directed-pair all-task decisions "
            "change under infrastructure-error reward handling."
        )
    )
    base = Path("analysis") / "public_backtest"
    parser.add_argument(
        "--metadata",
        type=Path,
        default=base / "tau2_pair_grid_failure" / "public_tau2_replay_metadata.json",
    )
    parser.add_argument(
        "--failure-task-table",
        type=Path,
        default=base / "tau2_ready_failure" / "public_tau2_task_rows.csv",
    )
    parser.add_argument(
        "--neutral-task-table",
        type=Path,
        default=base / "tau2_ready_neutral" / "public_tau2_task_rows.csv",
    )
    parser.add_argument(
        "--drop-task-table",
        type=Path,
        default=base / "tau2_ready_drop" / "public_tau2_task_rows.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base / "tau2_pair_grid_sensitivity",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10])
    return parser.parse_args()


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


def load_comparisons(metadata_path: Path) -> list[Comparison]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [
        Comparison(
            candidate=str(item["candidate"]),
            reference=str(item["reference"]),
            label=str(item["label"]),
        )
        for item in metadata["comparisons_requested"]
    ]


def prepare_pair(task_df: pd.DataFrame, comparison: Comparison) -> pd.DataFrame:
    candidate = task_df[task_df["submission_dir"].eq(comparison.candidate)].copy()
    reference = task_df[task_df["submission_dir"].eq(comparison.reference)].copy()
    keep_cols = [
        "domain",
        "task_id",
        "task_key",
        "outcome_mean",
        "cost_usd",
        "model_name",
        "reasoning_effort",
    ]
    merged = candidate[keep_cols].merge(
        reference[keep_cols],
        on=["domain", "task_id", "task_key"],
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    if merged.empty:
        raise ValueError(f"No overlapping tasks for {comparison.label}")
    merged["diff"] = (
        merged["outcome_mean_candidate"] - merged["outcome_mean_reference"]
    )
    return merged


def margin_bin(delta: float, threshold: float) -> str:
    margin = delta - threshold
    if margin >= 0.05:
        return "clear_positive"
    if margin > 0:
        return "near_positive"
    if margin >= -0.05:
        return "near_negative"
    return "clear_negative"


def summarize_truth(
    task_tables: dict[str, Path],
    comparisons: list[Comparison],
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for imputation, table_path in task_tables.items():
        task_df = pd.read_csv(table_path)
        for comparison in comparisons:
            pair = prepare_pair(task_df, comparison)
            delta = float(pair["diff"].mean())
            base = {
                "imputation": imputation,
                "comparison_label": comparison.label,
                "candidate": comparison.candidate,
                "reference": comparison.reference,
                "candidate_model": pair["model_name_candidate"].iloc[0],
                "candidate_reasoning_effort": pair["reasoning_effort_candidate"].iloc[0],
                "reference_model": pair["model_name_reference"].iloc[0],
                "reference_reasoning_effort": pair["reasoning_effort_reference"].iloc[0],
                "full_delta": delta,
                "task_count": int(len(pair)),
                "candidate_mean": float(pair["outcome_mean_candidate"].mean()),
                "reference_mean": float(pair["outcome_mean_reference"].mean()),
                "candidate_cost_mean": float(pair["cost_usd_candidate"].mean()),
                "reference_cost_mean": float(pair["cost_usd_reference"].mean()),
            }
            for threshold in thresholds:
                rows.append(
                    {
                        **base,
                        "threshold": threshold,
                        "truth_promote": bool(delta > threshold),
                        "margin_bin": margin_bin(delta, threshold),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["threshold", "comparison_label", "imputation"]
    )


def build_flip_table(truth: pd.DataFrame) -> pd.DataFrame:
    pivot = truth.pivot_table(
        index=["threshold", "comparison_label"],
        columns="imputation",
        values=["full_delta", "truth_promote", "margin_bin", "task_count"],
        aggfunc="first",
    )
    pivot.columns = [
        f"{metric}_{imputation}" for metric, imputation in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index()
    for imputation in ["neutral", "drop"]:
        pivot[f"shift_{imputation}_pp"] = (
            pivot[f"full_delta_{imputation}"] - pivot["full_delta_failure"]
        ) * 100.0
        pivot[f"abs_shift_{imputation}_pp"] = pivot[f"shift_{imputation}_pp"].abs()
        pivot[f"flip_{imputation}"] = (
            pivot[f"truth_promote_{imputation}"] != pivot["truth_promote_failure"]
        )
    return pivot.sort_values(["threshold", "comparison_label"]).reset_index(drop=True)


def format_percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f} percent"


def write_report(out_dir: Path, truth: pd.DataFrame, flip: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Public Tau-Bench Reward-Handling Sensitivity Grid\n")
    lines.append("## Status\n")
    lines.append(
        "Completed on the same 12 directed tau-bench public comparisons used in the primary directed pair grid.\n"
    )
    lines.append("## Scope\n")
    lines.append(
        "This check tests whether the completed-suite all-task decisions depend on infrastructure-error reward handling. It does not rerun partial-budget policies.\n"
    )
    lines.append("## All-Task Decision Mix By Reward Handling\n")
    mix = (
        truth.groupby(["threshold", "imputation", "margin_bin"])["comparison_label"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    lines.append(markdown_table(mix))
    lines.append("\n\n## Decision Flips Relative To Failure Imputation\n")
    summary_rows: list[dict[str, object]] = []
    for threshold, group in flip.groupby("threshold"):
        for imputation in ["neutral", "drop"]:
            flip_col = f"flip_{imputation}"
            abs_col = f"abs_shift_{imputation}_pp"
            summary_rows.append(
                {
                    "threshold": threshold,
                    "imputation": imputation,
                    "all_task_decision_flips": int(group[flip_col].sum()),
                    "max_abs_delta_shift_pp": f"{group[abs_col].max():.2f}",
                    "median_abs_delta_shift_pp": f"{group[abs_col].median():.2f}",
                }
            )
    lines.append(markdown_table(pd.DataFrame(summary_rows)))
    lines.append("\n\n## Pairs With Decision Flips\n")
    flip_rows = []
    for _, row in flip.iterrows():
        for imputation in ["neutral", "drop"]:
            if bool(row[f"flip_{imputation}"]):
                flip_rows.append(
                    {
                        "threshold": row["threshold"],
                        "comparison_label": row["comparison_label"],
                        "imputation": imputation,
                        "failure_delta": format_percent(row["full_delta_failure"]),
                        f"{imputation}_delta": format_percent(row[f"full_delta_{imputation}"]),
                        "failure_decision": row["truth_promote_failure"],
                        f"{imputation}_decision": row[f"truth_promote_{imputation}"],
                    }
                )
    if flip_rows:
        lines.append(markdown_table(pd.DataFrame(flip_rows)))
    else:
        lines.append("No all-task decision flips were observed.\n")
    lines.append("\n\n## Interpretation\n")
    lines.append(
        "The primary failure-imputation grid remains a conservative main analysis because infrastructure errors are counted against the submitted system. Neutral and drop handling are sensitivity checks for this choice.\n"
    )
    lines.append(
        "A flip should be treated as a margin-boundary warning, not as evidence that the policy replay is invalid. The paper should avoid using flipped pairs as strong examples of robust acceptance or rejection.\n"
    )
    (out_dir / "public_tau2_reward_sensitivity_grid_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = load_comparisons(args.metadata)
    task_tables = {
        "failure": args.failure_task_table,
        "neutral": args.neutral_task_table,
        "drop": args.drop_task_table,
    }
    truth = summarize_truth(task_tables, comparisons, args.thresholds)
    flip = build_flip_table(truth)
    truth.to_csv(args.out_dir / "reward_sensitivity_truth_table.csv", index=False)
    flip.to_csv(args.out_dir / "reward_sensitivity_flip_table.csv", index=False)
    write_report(args.out_dir, truth, flip)
    summary = {
        "truth_rows": int(len(truth)),
        "flip_rows": int(len(flip)),
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
