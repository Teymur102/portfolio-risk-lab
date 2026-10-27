"""Static figures (PNG) for the README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def equity_and_drawdown(net: pd.DataFrame, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    ends = {}
    for i, c in enumerate(net.columns):
        w = (1 + net[c]).cumprod()
        a1.plot(w.index, w, lw=1.6, color=SERIES[i], label=c)
        ends[c] = w.iloc[-1]
        a2.plot(net.index, drawdown(net[c]) * 100, lw=1.2, color=SERIES[i])
    # end labels, nudged apart in log space so they never overlap
    order = sorted(ends, key=ends.get)
    pos, last = {}, -np.inf
    for c in order:
        y = max(np.log(ends[c]), last + 0.06)
        pos[c], last = y, y
    for c in order:
        a1.annotate(f"{c} {ends[c]:.2f}x", (net.index[-1], np.exp(pos[c])), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=INK2, va="center")
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_yticks([0.6, 0.8, 1, 1.5, 2, 3, 4])
    a1.set_ylabel("Growth of $1 (log, net of costs)")
    a1.set_title(f"Walk-forward strategies, {net.index[0]:%b %Y} to {net.index[-1]:%b %Y}")
    a1.legend(loc="upper left", ncol=3, fontsize=9)
    a2.set_ylabel("Drawdown (%)")
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.subplots_adjust(right=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def var_full(fc: pd.DataFrame, models: list[str], path, conf_label="99%"):
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 2.3 * len(models)), sharex=True)
    for ax, m, col in zip(axes, models, [SERIES[0], SERIES[6], SERIES[2], SERIES[1]]):
        r = fc["return"] * 100
        v = -fc[f"{m} VaR"] * 100
        exc = r < v
        ax.plot(r.index, r, lw=0.4, color="#9b9a95", label="Daily return")
        ax.plot(v.index, v, lw=1.2, color=col, label=f"-VaR {conf_label}")
        ax.scatter(r.index[exc], r[exc], s=12, color=CRITICAL, marker="x", lw=1, zorder=3,
                   label=f"Exception ({int(exc.sum())})")
        ax.set_title(f"{m}: {int(exc.sum())} exceptions in {len(r)} days "
                     f"(expected {0.01 * len(r):.0f})", loc="left")
        ax.set_ylabel("%")
        ax.legend(loc="lower left", ncol=3, fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"60/40 portfolio: 1-day {conf_label} VaR forecasts vs realised returns",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def var_crises(fc: pd.DataFrame, models: list[str], windows: dict, path):
    fig, axes = plt.subplots(1, len(windows), figsize=(12, 4.2), sharey=True)
    colors = [SERIES[0], SERIES[6], SERIES[2], SERIES[1]]
    for ax, (label, (a, b)) in zip(axes, windows.items()):
        sub = fc.loc[a:b]
        r = sub["return"] * 100
        ax.bar(r.index, r, width=1.0, color="#b9b8b2", label="Daily return")
        for m, c in zip(models, colors):
            v = -sub[f"{m} VaR"] * 100
            n = int((r < v).sum())
            ax.plot(v.index, v, lw=1.6, color=c, label=f"{m} ({n} exc.)")
        ax.set_title(label, loc="left")
        ax.legend(loc="lower left", fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2 if "2008" in label else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axes[0].set_ylabel("Daily return / -VaR 99% (%)")
    fig.suptitle("How fast each 99% VaR model reacts in a crisis (60/40 portfolio)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
