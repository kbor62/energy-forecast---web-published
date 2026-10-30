"""Shared chart styling and figure saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Up to four series per chart, each with its own colour and dash pattern.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DASHES = [(0, ()), (0, (6, 2)), (0, (1, 1.6)), (0, (5, 1.5, 1, 1.5))]
MAX_SERIES = len(SERIES_COLORS)

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"
OBSERVED = "#3b3a37"


def format_time_axis(ax) -> None:
    """Compact date ticks."""
    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def style_axes(ax, ylabel: str = "NOK/kWh") -> None:
    """Light grid and axes."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_edgecolor(GRID)
    frame.set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(INK)


def save_figure(fig, out_png: str | Path, dpi: int = 150, close: bool = True) -> Path:
    """Write a figure to ``out_png``, creating the folder if needed."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)

    return out_png
