from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "outputs" / "figures"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_public_minimum_budgets() -> None:
    policy = {
        "SWE-bench Lite": "repo coverage",
        "SWE-bench Verified": "repo coverage",
        "AppWorld": "difficulty coverage",
        "tau-bench": "domain coverage",
    }
    source_order = [
        "AppWorld",
        "tau-bench",
        "SWE-bench Verified",
        "SWE-bench Lite",
    ]
    main_grid_rows = pd.read_csv(
        ANALYSIS
        / "public_backtest"
        / "target_budget_sweep"
        / "main_policy_summary"
        / "main_policy_minimum_budgets.csv"
    )
    main_grid_rows = main_grid_rows[np.isclose(main_grid_rows["policy_alpha"], 0.05)].copy()
    main_grid_rows["threshold"] = main_grid_rows["threshold"].map(
        lambda value: f"{int(round(float(value) * 100))} pp"
    )
    main_grid_rows["minimum_budget_percent"] = (
        pd.to_numeric(main_grid_rows["minimum_sufficient_budget"], errors="coerce")
        * 100.0
    )

    def budget_value(source: str, threshold: str) -> float:
        match = main_grid_rows[
            (main_grid_rows["source"] == source)
            & (main_grid_rows["threshold"] == threshold)
        ]
        return float(match.iloc[0]["minimum_budget_percent"])

    def cell_style(value: float) -> tuple[str, str, str]:
        if pd.isna(value):
            return "#E6E8EC", "#4B5563", "None"
        if value <= 25:
            return "#CFEAD6", "#174A2A", f"{int(value)}%"
        if value <= 50:
            return "#F5E6A3", "#4B3A03", f"{int(value)}%"
        return "#F2BE8D", "#5B2A0A", f"{int(value)}%"

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.15, 2.65))
        text_color = "#1F2933"
        muted = "#5B6470"
        edge = "#FFFFFF"
        thresholds = ["0 pp", "5 pp", "10 pp"]
        x0 = 2.45
        y0 = 0.25
        cell_w = 1.08
        cell_h = 0.48
        row_gap = 0.12
        header_y = y0 + len(source_order) * (cell_h + row_gap) + 0.13

        ax.set_xlim(0, x0 + len(thresholds) * cell_w + 0.25)
        ax.set_ylim(0, header_y + 0.62)
        ax.axis("off")

        ax.text(
            x0,
            header_y + 0.35,
            "Cell value: minimum budget that matches the completed conclusion",
            ha="left",
            va="center",
            fontsize=7.6,
            color=muted,
        )
        ax.text(
            x0,
            header_y + 0.14,
            "Lower is better; None means no tested budget up to 95% meets all targets",
            ha="left",
            va="center",
            fontsize=7.1,
            color=muted,
        )

        for col_idx, threshold in enumerate(thresholds):
            x = x0 + col_idx * cell_w
            ax.text(
                x + cell_w / 2,
                header_y - 0.18,
                threshold,
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color=text_color,
            )

        for row_idx, source in enumerate(source_order):
            y = y0 + (len(source_order) - 1 - row_idx) * (cell_h + row_gap)
            ax.text(
                0.08,
                y + cell_h * 0.62,
                source,
                ha="left",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                0.08,
                y + cell_h * 0.28,
                policy[source],
                ha="left",
                va="center",
                fontsize=7.0,
                color=muted,
            )
            for col_idx, threshold in enumerate(thresholds):
                value = budget_value(source, threshold)
                face, label_color, label = cell_style(value)
                x = x0 + col_idx * cell_w
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        cell_w - 0.06,
                        cell_h,
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=1.5,
                    )
                )
                ax.text(
                    x + (cell_w - 0.06) / 2,
                    y + cell_h / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    fontweight="bold",
                    color=label_color,
                )

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06)
        save(fig, "public_minimum_sufficient_budget")


def plot_public_abstention_curve_0pp() -> None:
    curves = pd.read_csv(
        ANALYSIS
        / "public_backtest"
        / "target_budget_sweep"
        / "main_policy_summary"
        / "main_policy_aggregate.csv"
    )
    minimum = pd.read_csv(
        ANALYSIS
        / "public_backtest"
        / "target_budget_sweep"
        / "main_policy_summary"
        / "main_policy_minimum_budgets.csv"
    )

    primary_alpha = 0.05
    threshold = 0.0
    sources = [
        "SWE-bench Verified",
        "SWE-bench Lite",
        "tau-bench",
        "AppWorld",
    ]
    colors = {
        "AppWorld": "#009E73",
        "tau-bench": "#CC79A7",
        "SWE-bench Lite": "#0072B2",
        "SWE-bench Verified": "#D55E00",
    }

    curves = curves[
        np.isclose(curves["policy_alpha"], primary_alpha)
        & np.isclose(curves["threshold"], threshold)
    ].copy()
    minimum = minimum[
        np.isclose(minimum["policy_alpha"], primary_alpha)
        & np.isclose(minimum["threshold"], threshold)
    ].copy()
    curves["budget_cap"] = pd.to_numeric(curves["budget_cap"], errors="coerce")
    curves["defer_rate"] = pd.to_numeric(curves["defer_rate"], errors="coerce")
    minimum["minimum_sufficient_budget"] = pd.to_numeric(
        minimum["minimum_sufficient_budget"], errors="coerce"
    )

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.15, 3.15))
        fig.subplots_adjust(top=0.82, bottom=0.16, left=0.09, right=0.86)
        fig.text(
            0.09,
            0.96,
            "When do partial runs meet the unresolved-comparison target?",
            fontsize=9.4,
            fontweight="bold",
            color="#1F2933",
        )
        fig.text(
            0.09,
            0.91,
            "0 pp threshold, 5 pp budget grid. Green band: at most 25% unresolved; open markers: first budgets meeting all targets.",
            fontsize=7.0,
            color="#5B6470",
        )

        ax.axhspan(0, 25, color="#E7F4EA", zorder=0)
        ax.axhline(25, color="#246B3D", linewidth=1.0, linestyle="--")
        ax.text(7, 3.6, "unresolved target", color="#246B3D", fontsize=7.0)
        ax.text(
            52,
            27.8,
            "25% target",
            color="#246B3D",
            fontsize=6.8,
            ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
        )

        for source in sources:
            source_rows = curves[curves["source"] == source].sort_values("budget_cap")
            x = source_rows["budget_cap"].to_numpy() * 100
            y = source_rows["defer_rate"].to_numpy() * 100
            ax.plot(
                x,
                y,
                color=colors[source],
                linewidth=1.8,
                marker="o",
                markersize=3.3,
                zorder=3,
            )

        endpoint_labels = {
            "SWE-bench Lite": (98, 31),
            "SWE-bench Verified": (98, 23),
            "tau-bench": (98, 16),
            "AppWorld": (98, 4.5),
        }
        for source, (label_x, label_y) in endpoint_labels.items():
            ax.text(
                label_x,
                label_y,
                source,
                color=colors[source],
                fontsize=7.5,
                fontweight="bold",
                va="center",
            )

        marker_labels = {
            "AppWorld": -9.0,
            "tau-bench": 8.5,
            "SWE-bench Verified": 9.0,
        }
        for source, label_dy in marker_labels.items():
            source_rows = curves[curves["source"] == source].sort_values("budget_cap")
            min_row = minimum[minimum["source"] == source]
            if min_row.empty or pd.isna(min_row.iloc[0]["minimum_sufficient_budget"]):
                continue
            min_budget = float(min_row.iloc[0]["minimum_sufficient_budget"])
            yrow = source_rows[np.isclose(source_rows["budget_cap"], min_budget)]
            if yrow.empty:
                continue
            marker_x = min_budget * 100
            marker_y = float(yrow.iloc[0]["defer_rate"]) * 100
            ax.scatter(
                [marker_x],
                [marker_y],
                s=78,
                facecolor="white",
                edgecolor=colors[source],
                linewidth=1.9,
                zorder=5,
            )
            ax.text(
                marker_x,
                marker_y + label_dy,
                f"{int(round(marker_x))}%",
                color=colors[source],
                fontsize=8.0,
                fontweight="bold",
                ha="center",
                va="center",
            )

        for source in sources:
            min_row = minimum[minimum["source"].eq(source)]
            if min_row.empty or not pd.isna(
                min_row.iloc[0]["minimum_sufficient_budget"]
            ):
                continue
            source_rows = curves[curves["source"].eq(source)].sort_values("budget_cap")
            tail = source_rows.iloc[-1]
            marker_x = float(tail["budget_cap"]) * 100
            marker_y = float(tail["defer_rate"]) * 100
            ax.scatter([marker_x], [marker_y], s=54, color=colors[source], zorder=5)
            ax.annotate(
                "no sufficient budget by 95%",
                xy=(marker_x, marker_y),
                xytext=(59, 43),
                textcoords="data",
                fontsize=7.0,
                color=colors[source],
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": colors[source],
                    "linewidth": 0.8,
                },
            )

        ax.set_xlabel("Task budget (%)")
        ax.set_ylabel("Unresolved comparisons (%)")
        ax.set_xlim(5, 112)
        ax.set_xticks([5, 15, 25, 50, 75, 95])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#D7DADE", linewidth=0.55, alpha=0.8)
        save(fig, "public_deferral_operating_curve_0pp")


def plot_bootstrap_replicate_stability() -> None:
    path = (
        ANALYSIS
        / "public_backtest"
        / "reviewer_diagnostics"
        / "stability_aggregate.csv"
    )
    if not path.exists():
        return
    df = pd.read_csv(path)
    selected_budgets = {
        "AppWorld": 0.15,
        "tau-bench": 0.25,
        "SWE-bench Verified": 0.90,
        "SWE-bench Lite": 0.95,
    }
    source_order = [
        "AppWorld",
        "tau-bench",
        "SWE-bench Verified",
        "SWE-bench Lite",
    ]
    colors = {
        "AppWorld": "#009E73",
        "tau-bench": "#CC79A7",
        "SWE-bench Lite": "#0072B2",
        "SWE-bench Verified": "#D55E00",
    }
    rows = []
    for source, budget_cap in selected_budgets.items():
        subset = df[
            df["source"].eq(source)
            & np.isclose(df["threshold"], 0.0)
            & np.isclose(df["budget_cap"], budget_cap)
        ].copy()
        if subset.empty:
            continue
        subset["max_conditional_error"] = subset[
            ["conditional_false_accept_rate", "conditional_false_reject_rate"]
        ].max(axis=1)
        for samples, group in subset.groupby("bootstrap_samples"):
            rows.append(
                {
                    "source": source,
                    "budget_cap": budget_cap,
                    "bootstrap_samples": int(samples),
                    "unresolved_rate": float(group["defer_rate"].mean()),
                    "max_conditional_error": float(group["max_conditional_error"].mean()),
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25), sharex=True)
        fig.subplots_adjust(left=0.075, right=0.84, bottom=0.21, top=0.78, wspace=0.28)
        fig.text(
            0.075,
            0.93,
            "Bootstrap replicate count does not change the qualitative decision pattern",
            fontsize=9.0,
            fontweight="bold",
            color="#1F2933",
        )
        fig.text(
            0.075,
            0.84,
            "0 pp threshold; each source is shown at a representative budget from the main decision grid.",
            fontsize=7.2,
            color="#5B6470",
        )
        metrics = [
            ("unresolved_rate", "Unresolved comparisons (%)", 25.0),
            ("max_conditional_error", "Max conditional error (%)", 5.0),
        ]
        handles = []
        labels = []
        for ax, (metric, ylabel, target) in zip(axes, metrics):
            for source in source_order:
                source_rows = plot_df[plot_df["source"].eq(source)].sort_values(
                    "bootstrap_samples"
                )
                if source_rows.empty:
                    continue
                line = ax.plot(
                    source_rows["bootstrap_samples"].to_numpy(),
                    source_rows[metric].to_numpy() * 100.0,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.4,
                    color=colors[source],
                    label=f"{source} ({int(selected_budgets[source] * 100)}%)",
                )[0]
                if ax is axes[0]:
                    handles.append(line)
                    labels.append(f"{source} ({int(selected_budgets[source] * 100)}%)")
            ax.axhline(target, color="#6B7280", linestyle=":", linewidth=1.0)
            ax.set_xlim(80, 540)
            ax.set_xticks([100, 200, 500])
            ax.set_xlabel("Bootstrap replicates")
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
        axes[0].text(
            88,
            25.8,
            "target 25%",
            fontsize=6.7,
            color="#4B5563",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.7},
        )
        axes[1].text(
            88,
            4.72,
            "target 5%",
            fontsize=6.7,
            color="#4B5563",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.7},
        )
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.845, 0.48),
            frameon=False,
            title="Source budget",
            title_fontsize=7.0,
        )
        save(fig, "bootstrap_replicate_stability")


def plot_public_tau_sensitivity() -> None:
    truth = pd.read_csv(
        ANALYSIS
        / "public_backtest"
        / "tau2_pair_grid_sensitivity"
        / "reward_sensitivity_truth_table.csv"
    )
    truth = truth[truth["threshold"] == 0.05].copy()
    order = ["failure", "neutral", "drop"]
    comparisons = [
        "gpt52_high_vs_none",
        "gpt52_high_vs_sonnet45",
        "opus45_vs_gpt52_high",
    ]
    labels = {
        "gpt52_high_vs_none": "GPT-5.2 high vs none",
        "gpt52_high_vs_sonnet45": "GPT-5.2 high vs Sonnet 4.5",
        "opus45_vs_gpt52_high": "Opus 4.5 vs GPT-5.2 high",
    }
    colors = {
        "failure": "#4C78A8",
        "neutral": "#F58518",
        "drop": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(comparisons))
    width = 0.24
    for idx, imputation in enumerate(order):
        vals = []
        for comparison in comparisons:
            row = truth[
                (truth["comparison_label"] == comparison) & (truth["imputation"] == imputation)
            ].iloc[0]
            vals.append(float(row["full_delta"]) * 100)
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width,
            label=imputation,
            color=colors[imputation],
            alpha=0.9,
        )
    ax.axhline(5, color="#333333", linestyle=":", linewidth=1.0)
    ax.axhline(10, color="#777777", linestyle="--", linewidth=1.0)
    ax.text(len(comparisons) - 0.55, 5.25, "5 pp", fontsize=8, color="#333333")
    ax.text(len(comparisons) - 0.55, 10.25, "10 pp", fontsize=8, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in comparisons], rotation=12, ha="right")
    ax.set_ylabel("All-task delta, pp")
    ax.set_title("Public tau-bench reward-handling sensitivity")
    ax.grid(True, axis="y", color="#E5E5E5")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    save(fig, "public_tau2_imputation_sensitivity")


def main() -> None:
    plot_public_minimum_budgets()
    plot_public_abstention_curve_0pp()
    plot_bootstrap_replicate_stability()
    plot_public_tau_sensitivity()


if __name__ == "__main__":
    main()
