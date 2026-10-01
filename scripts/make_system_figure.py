from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


COLORS = {
    "ink": "#1F2933",
    "muted": "#4B5563",
    "line": "#B8C0CC",
    "runner": "#E8EDF2",
    "stream": "#E8F2EE",
    "layer": "#FFF4DE",
    "record": "#F7E7EA",
    "audit": "#F7F7F5",
    "blue": "#4A90D9",
    "teal": "#5BA58B",
    "amber": "#D4A252",
    "rose": "#C75B6A",
    "slate": "#7B8794",
}

FIT_CHECKS: list[tuple[str, plt.Text, tuple[float, float, float, float]]] = []


def register_fit(label: str, artist: plt.Text, bounds: tuple[float, float, float, float]) -> None:
    FIT_CHECKS.append((label, artist, bounds))


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    face: str,
    accent: str,
    title_size: float = 7.1,
    body_size: float = 5.9,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            linewidth=0.75,
            edgecolor=COLORS["line"],
            facecolor=face,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            0.008,
            h,
            linewidth=0,
            facecolor=accent,
            zorder=2,
        )
    )
    title_artist = ax.text(
        x + 0.018,
        y + h - 0.038,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
    )
    body_artist = ax.text(
        x + 0.018,
        y + h - 0.100,
        "\n".join(lines),
        fontsize=body_size,
        color=COLORS["muted"],
        linespacing=1.16,
        ha="left",
        va="top",
    )
    register_fit(f"{title} title", title_artist, (x, y, w, h))
    register_fit(f"{title} body", body_artist, (x, y, w, h))


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.0,
            linewidth=0.9,
            color="#606B78",
            shrinkA=2.0,
            shrinkB=2.0,
            zorder=3,
        )
    )


def check_text_fit(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    pad_px = 2.0
    for label, artist, (x, y, w, h) in FIT_CHECKS:
        bbox = artist.get_window_extent(renderer=renderer)
        left_bottom = ax.transData.transform((x, y))
        right_top = ax.transData.transform((x + w, y + h))
        if (
            bbox.x0 < left_bottom[0] - pad_px
            or bbox.x1 > right_top[0] + pad_px
            or bbox.y0 < left_bottom[1] - pad_px
            or bbox.y1 > right_top[1] + pad_px
        ):
            failures.append(label)
    if failures:
        raise RuntimeError("Figure text overflow: " + ", ".join(failures))


def save_all(fig: plt.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"system_decision_layer.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
        )


def main() -> None:
    FIT_CHECKS.clear()
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "figure.dpi": 340,
            "savefig.dpi": 340,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.25, 2.65))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(
            0.025,
            0.955,
            "From task outcomes to a decision record",
            fontsize=8.4,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="top",
        )

        y = 0.565
        h = 0.285
        boxes = [
            (0.025, 0.205, "1. Record", ["completed tasks", "systems, outcomes", "groups, costs"], COLORS["runner"], COLORS["blue"]),
            (0.285, 0.205, "2. Replay", ["reveal task subset", "check coverage", "track budget"], COLORS["stream"], COLORS["teal"]),
            (0.545, 0.205, "3. Decide", ["positive conclusion", "negative conclusion", "unresolved"], COLORS["layer"], COLORS["amber"]),
            (0.805, 0.165, "4. Report", ["decision errors", "coverage failures", "unresolved comparisons", "sufficient budget"], COLORS["record"], COLORS["rose"]),
        ]
        for x, w, title, lines, face, accent in boxes:
            rounded_box(ax, x, y, w, h, title, lines, face, accent)

        arrow(ax, (0.230, y + h * 0.52), (0.285, y + h * 0.52))
        arrow(ax, (0.490, y + h * 0.52), (0.545, y + h * 0.52))
        arrow(ax, (0.750, y + h * 0.52), (0.805, y + h * 0.52))

        ax.add_patch(
            FancyBboxPatch(
                (0.025, 0.110),
                0.695,
                0.320,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                linewidth=0.75,
                edgecolor=COLORS["line"],
                facecolor=COLORS["audit"],
                zorder=1,
            )
        )
        ax.text(
            0.045,
            0.385,
            "How replay validates the partial decision rule",
            fontsize=7.0,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="top",
        )
        rounded_box(
            ax,
            0.050,
            0.135,
            0.185,
            0.190,
            "Completed data",
            ["All outcomes", "Completed conclusion"],
            "#FFFFFF",
            COLORS["slate"],
            title_size=5.9,
            body_size=5.1,
        )
        rounded_box(
            ax,
            0.285,
            0.135,
            0.180,
            0.190,
            "Hide later outcomes",
            ["Reveal partial run", "At each budget"],
            "#FFFFFF",
            COLORS["amber"],
            title_size=5.9,
            body_size=5.1,
        )
        rounded_box(
            ax,
            0.515,
            0.135,
            0.185,
            0.190,
            "Check judgment",
            ["Compare with", "completed conclusion"],
            "#FFFFFF",
            COLORS["teal"],
            title_size=5.9,
            body_size=5.1,
        )
        arrow(ax, (0.235, 0.230), (0.285, 0.230))
        arrow(ax, (0.465, 0.230), (0.515, 0.230))


        check_text_fit(fig, ax)
        save_all(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
