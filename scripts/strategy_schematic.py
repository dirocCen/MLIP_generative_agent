"""Schematic of the five acquisition strategies on a 1-D toy problem (README figure).

    PYTHONPATH=src python scripts/strategy_schematic.py   # -> docs/figures/loop_strategies.png

Toy, not data: candidates sit on a 1-D descriptor axis; the hidden kappa has a broad peak the
Slack estimate reproduces (B-C-N-like) and a narrower, higher peak it misses (BAs-like). Every
panel shows what one strategy has labelled after its first batch and which batch it would pick
next. The toy's parameters and seeds are chosen for legibility; the picks are computed with the
project's own ``Surrogate`` and ``rank``, not drawn by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mga.loop.surrogate import Surrogate, rank  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "loop_strategies.png"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df"
# same slot per strategy as scripts/loop_replay.py
COLOR = {"random": "#eda100", "slack": "#eb6834", "greedy": "#1baf7a", "ucb": "#2a78d6", "slack_ucb": "#e87ba4"}
PANELS = [
    ("random", "random", "uniform at random"),
    ("slack", "Slack ranking", "next in the fixed Slack order"),
    ("greedy", "surrogate (greedy)", "highest predicted μ"),
    ("ucb", "surrogate (UCB)", "highest μ + σ"),
    ("slack_ucb", "Slack start → UCB", "Slack head first, then μ + σ"),
]
N_INIT, BATCH = 5, 3


def truth(x):
    """Hidden log-kappa: broad peak at 0.25 that Slack sees, narrow higher one at 0.82 it misses."""
    return 0.9 * np.exp(-((x - 0.25) / 0.13) ** 2) + 1.25 * np.exp(-((x - 0.82) / 0.10) ** 2) + 0.15


def slack_estimate(x):
    return 0.95 * np.exp(-((x - 0.27) / 0.15) ** 2) + 0.2 + 0.1 * x


def main() -> None:
    rng = np.random.default_rng(3)
    x = np.sort(np.linspace(0.02, 0.98, 36) + rng.uniform(-0.01, 0.01, 36))
    y = truth(x)
    slack = slack_estimate(x)
    X = x[:, None]
    grid = np.linspace(0, 1, 400)

    init_random = np.random.default_rng(1).permutation(len(x))[:N_INIT]
    slack_order = rank("slack", X, slack, rng)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3), dpi=200, sharey=True, facecolor=SURFACE)
    for ax, (key, title, rule) in zip(axes, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        ax.plot(grid, truth(grid), color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        if key in ("slack", "slack_ucb"):
            ax.plot(grid, slack_estimate(grid), color=COLOR["slack"], lw=2, zorder=2)

        labelled = slack_order[:N_INIT] if key in ("slack", "slack_ucb") else init_random
        rest = np.setdiff1d(np.arange(len(x)), labelled)
        if key == "random":
            pick = np.random.default_rng(8).choice(rest, BATCH, replace=False)
        elif key == "slack":
            pick = slack_order[N_INIT:N_INIT + BATCH]
        else:
            sur = Surrogate(0).fit(X[labelled], y[labelled])
            mu, sd = sur.predict(grid[:, None])
            ax.fill_between(grid, mu - sd, mu + sd, color=COLOR[key], alpha=0.18, lw=0, zorder=2)
            ax.plot(grid, mu, color=COLOR[key], lw=2, zorder=3)
            strat = "greedy" if key == "greedy" else "ucb"
            pick = rest[rank(strat, X[rest], slack[rest], rng, sur)[:BATCH]]

        ax.scatter(x, np.full_like(x, -0.08), s=6, color=MUTED, zorder=2, clip_on=False)
        ax.scatter(x[labelled], y[labelled], s=34, color=INK2, edgecolors=SURFACE, linewidths=1.2, zorder=4)
        ax.scatter(x[pick], y[pick], s=110, facecolors="none", edgecolors=INK, linewidths=1.8, zorder=5)

        ax.scatter([0.015], [1.14], s=40, marker="s", color=COLOR[key], transform=ax.transAxes, clip_on=False)
        ax.text(0.05, 1.14, title, transform=ax.transAxes, color=INK, fontsize=10, va="center")
        ax.text(0, 1.03, f"next batch: {rule}", transform=ax.transAxes, color=INK2, fontsize=8, va="bottom")
        ax.set_xlim(-0.01, 1.02)
        ax.set_ylim(-0.12, 1.5)
        ax.set_xticks([0.25, 0.82], ["B–C–N-like", "BAs-like"], color=INK2, fontsize=8)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.set_xlabel("descriptor space (toy, 1-D)", color=INK2, fontsize=8)
    axes[0].set_ylabel("log κ", color=INK2, fontsize=9)

    proxies = [
        plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="true κ (hidden until labelled)"),
        plt.Line2D([], [], color=COLOR["slack"], lw=2, label="Slack estimate (from L0)"),
        plt.Line2D([], [], color=INK2, lw=2, label="GP surrogate μ ± σ (strategy colour)"),
        plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=3, label="candidate"),
        plt.Line2D([], [], color=INK2, marker="o", ls="", ms=6, label=f"labelled by L1 (first {N_INIT})"),
        plt.Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=INK, mew=1.8, label=f"next batch ({BATCH})"),
    ]
    fig.legend(handles=proxies, loc="lower center", ncol=6, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
