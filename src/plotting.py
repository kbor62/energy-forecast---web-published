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


def fixed_rate_note() -> str:
    """Info text for figures whose NOK values use the fixed config rate."""
    from config import EUR_TO_NOK

    return f"NOK at fixed EUR/NOK {EUR_TO_NOK:.2f} (config.EUR_TO_NOK); models work in EUR"


def save_figure(fig, out_png: str | Path, dpi: int = 150, close: bool = True, note: str | None = None) -> Path:
    """Write a figure to ``out_png``, creating the folder if needed. ``note``: small info
    line under the plot (e.g. the EUR/NOK rate used)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(0.995, 0.0, note, ha="right", va="top", fontsize=8, color=INK_SOFT,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f3", edgecolor=GRID, linewidth=0.6))
    try:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    except OSError as exc:
        # The target can be locked on Windows (open in an image viewer or the Explorer
        # preview pane, or held by a cloud-sync client). Do not fail the pipeline over a
        # figure: write a time-stamped copy next to it instead.
        import datetime as _dt

        alt = out_png.with_name(f"{out_png.stem}_{_dt.datetime.now():%Y%m%d_%H%M%S}{out_png.suffix}")
        print(f"  WARNING: could not write {out_png.name} ({exc.__class__.__name__}: {exc}); "
              f"is it open in another program? Saved as {alt.name}")
        fig.savefig(alt, dpi=dpi, bbox_inches="tight", facecolor="white")
        out_png = alt
    if close:
        plt.close(fig)

    return out_png
