"""Replay acquisition strategies on the labelled part of a pool and plot the three loop curves.

Universe = structures of ``--set`` (plus ``--extra-set``s, e.g. earlier campaigns) that are
L0-stable and have an L1 label. Hidden rediscovery targets are matched by reduced formula and
space group of the registry entry (and the R3m-BNC2 control by sid).

Usage:
  MGA_PROJECT_ROOT=$PWD PYTHONPATH=src python scripts/loop_replay.py --set loop.pool.v1.bcn200 \
      --seeds 20 --n-init 10 --batch 10 --rounds 10 --name v1
Writes runs/loop-v1/replay/<name>.csv and docs/figures/loop_replay_<name>.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mga import registry
from mga.loop.data import build_table, labelled
from mga.loop.replay import ReplayConfig, hit_threshold, run
from mga.loop.surrogate import STRATEGIES

ROOT = Path(__file__).resolve().parents[1]

# name -> (reduced formula, accepted space groups)
TARGETS = {
    "diamond": ("C", {227}),
    "c-BN": ("BN", {216}),
    "BAs": ("BAs", {216}),
    "BP": ("BP", {216}),
    "SiC": ("SiC", {216, 186}),
    "R3m-BNC2": ("BC2N", {160}),
}
# validated categorical slots (dataviz reference palette, light mode); fixed per strategy
COLOR = {"ucb": "#2a78d6", "slack": "#eb6834", "greedy": "#1baf7a", "random": "#eda100"}
LABEL = {"ucb": "surrogate (UCB)", "greedy": "surrogate (greedy)", "slack": "Slack ranking", "random": "random"}


def target_sids(sids, control: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in registry.iter_entries(list(sids)):
        for name, (formula, sgs) in TARGETS.items():
            if e["reduced_formula"] == formula and e["spacegroup_number"] in sgs:
                if name == "R3m-BNC2" and control and e["sid"] not in control:
                    continue
                out.setdefault(name, []).append(e["sid"])
    return out


def band(ax, df, x, y, s):
    g = df[df.strategy == s].groupby("round")
    m, lo, hi = g[y].mean(), g[y].quantile(0.25), g[y].quantile(0.75)
    xs = g[x].mean()
    ls = "--" if s == "random" else "-"
    ax.fill_between(xs, lo, hi, color=COLOR[s], alpha=0.15, linewidth=0)
    ax.plot(xs, m, ls, color=COLOR[s], lw=2, label=LABEL[s])
    ax._end_labels = getattr(ax, "_end_labels", []) + [(LABEL[s], xs.iloc[-1], m.iloc[-1])]


def end_labels(ax, min_gap: float = 0.07) -> None:
    """Direct labels at line ends, pushed apart vertically so converging lines stay readable."""
    items = sorted(getattr(ax, "_end_labels", []), key=lambda t: t[2])
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    ys = [t[2] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    ys[-1] = min(ys[-1], hi)  # keep inside the axes: push the stack back down from the top
    for i in range(len(ys) - 2, -1, -1):
        ys[i] = min(ys[i], ys[i + 1] - gap)
    x0, x1 = ax.get_xlim()
    x_lab = max(t[1] for t in items) + 0.02 * (x1 - x0)
    for (text, _x, _y), y_lab in zip(items, ys):
        ax.text(x_lab, y_lab, text, fontsize=8, color="#3d3d3a", va="center", ha="left", clip_on=False)


def plot(df, n_targets: int, thr: float, n_universe: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    strategies = [s for s in ("ucb", "greedy", "slack", "random") if s in set(df.strategy)]
    for s in strategies:
        band(axes[0], df, "gpu_h", "hits", s)
        band(axes[1], df, "n_labelled", "targets_found", s)
    for s in [s for s in strategies if s in ("ucb", "greedy")]:
        d = df[(df.strategy == s) & df.spearman_rest.notna()]
        band(axes[2], d, "n_labelled", "spearman_rest", s)
    axes[0].set(xlabel="L1 GPU·h spent", ylabel=f"hits (κ_L1 ≥ {thr:.0f} W/mK)", title="Hits per GPU·h")
    axes[1].set(xlabel="structures labelled by L1", ylabel=f"hidden targets found (of {n_targets})",
                title="Rediscovery of known high-κ materials")
    axes[2].set(xlabel="structures labelled by L1", ylabel="Spearman ρ on unlabelled rest",
                title="Surrogate accuracy per round")
    for ax in axes:
        ax.grid(color="#e8e7e2", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.25)
        end_labels(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Replay on {n_universe} L1-labelled structures; mean and interquartile band over seeds",
                 fontsize=10, color="#3d3d3a")
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="loop.pool.v1.bcn200")
    ap.add_argument("--extra-set", action="append", default=[])
    ap.add_argument("--extra-sids", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--hit-quantile", type=float, default=0.9)
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    ap.add_argument("--name", default="v1")
    ap.add_argument("--run-id", default="loop-v1")
    a = ap.parse_args()

    sids = list(registry.set_sids(a.set))
    for s in a.extra_set:
        sids += [x for x in registry.set_sids(s) if x not in sids]
    sids += [x for x in a.extra_sids if x not in sids]
    control = (registry.load_set(a.set).get("details") or {}).get("positive_control", [])

    universe = labelled(build_table(sids))
    targets = {k: [s for s in v if s in universe.index] for k, v in target_sids(sids, control).items()}
    targets = {k: v for k, v in targets.items() if v}
    cfg = ReplayConfig(n_init=a.n_init, batch=a.batch, n_rounds=a.rounds,
                       hit_quantile=a.hit_quantile, targets=targets)
    df = run(universe, a.strategies, range(a.seeds), cfg)

    out_dir = ROOT / "runs" / a.run_id / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{a.name}.csv", index=False)
    thr = hit_threshold(universe, a.hit_quantile)
    meta = {"set": a.set, "extra_sets": a.extra_set, "n_universe": len(universe), "hit_threshold": thr,
            "targets": targets, "config": {k: v for k, v in vars(cfg).items() if k != "targets"},
            "seeds": a.seeds}
    (out_dir / f"{a.name}.json").write_text(json.dumps(meta, indent=1))
    plot(df, len(targets), thr, len(universe), ROOT / "docs" / "figures" / f"loop_replay_{a.name}.png")

    last = df[df["round"] == df["round"].max()].groupby("strategy")[["hits", "targets_found", "gpu_h"]].mean()
    print(json.dumps({**meta, "final": last.round(2).to_dict(orient="index")}, indent=1, default=str))


if __name__ == "__main__":
    main()
