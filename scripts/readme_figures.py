"""README figures from the S0 calibration results (run locally after fetch_results).

    PYTHONPATH=src python scripts/readme_figures.py

Reads the kappa_SRME outputs in runs/s0-calibration/outputs/ and the L2 results in the local
result store; writes docs/figures/s0_ksrme.png and docs/figures/s0_ladder.png. Missing results
(jobs still running) are left out of the plot, not filled in.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mga import results  # noqa: E402
from mga.tools.analysis import _submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
RUN = ROOT / "runs" / "s0-calibration"

# categorical slots 1-3 (fixed per model), marker shape as secondary encoding
MODELS = {
    "dpa41_max_oam": ("DPA-4.1-Max-OAM", "#2a78d6", "o"),
    "tece_oam_rra": ("TECE-OAM-RRA-1.0", "#eb6834", "s"),
    "pet_oam_xl": ("PET-OAM-XL (f32)", "#1baf7a", "D"),
}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COMPOUNDS = ["BN", "AlN", "SiC", "BAs"]
LADDER_SIDS = {"c-BN": "BN-4826268f1d", "BAs": "BAs-204a4c990a"}
EXPERIMENT = {"c-BN": 850.0, "BAs": 1300.0}  # README: c-BN ~850, BAs ~1300 W/mK at 300 K


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ksrme_rows() -> dict[str, dict]:
    """Latest kappa_SRME output per model that scored all four S0 compounds."""
    out = {}
    for f in sorted(glob.glob(str(RUN / "outputs" / "*kappa_srme*.json"))):
        rows = {r["name"]: r for r in json.load(open(f))["rows"]}
        rec = json.load(open(RUN / "records" / Path(f).name))
        model = _submission(rec["inputs"]["kappa_step_id"])["model"]
        if model in MODELS and set(COMPOUNDS) <= set(rows):
            out[model] = {"srme": sum(rows[c]["srme"] for c in COMPOUNDS) / len(COMPOUNDS), "rows": rows}
    return out


def fig_ksrme(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK2, linewidth=1)
    width = 0.22
    for i, (model, (label, color, marker)) in enumerate(MODELS.items()):
        if model not in data:
            continue
        rows = data[model]["rows"]
        xs = [k + (i - 1) * width for k in range(len(COMPOUNDS))]
        ys = [100 * (rows[c]["kappa_pred"] / rows[c]["kappa_ref"] - 1) for c in COMPOUNDS]
        ax.vlines(xs, 0, ys, color=color, linewidth=2)
        ax.scatter(xs, ys, s=42, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label=f"{label}  SRME {data[model]['srme']:.3f}")
    ax.set_xticks(range(len(COMPOUNDS)), ["c-BN", "w-AlN", "3C-SiC", "BAs"], color=INK)
    ax.set_ylabel("MLIP vs DFT (phononDB) κ, %", color=INK2, fontsize=9)
    ax.set_title("κ_SRME subset, Wigner-RTA at 300 K", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "s0_ksrme.png", facecolor=SURFACE)


def ladder_values() -> dict[str, dict[str, dict[str, float]]]:
    """{model: {compound: {'L1': RTA, 'L2': LBTE}}} from the L2 results (same FCs as L1)."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(RUN / "submissions" / "*kappa_lbte*.json"))):
        sub = _submission(Path(f).stem)
        for name, sid in LADDER_SIDS.items():
            if sub["model"] in MODELS and results.has_result(sub["kind"], sub["model"], sid, sub["param_hash"]):
                d = results.load_result(sub["kind"], sub["model"], sid, sub["param_hash"])["data"]
                if d.get("ok"):
                    out.setdefault(sub["model"], {})[name] = {"L1": d["kappa_tot_rta_avg"][0], "L2": d["kappa_tot_exact_avg"][0]}
    return out


def fig_ladder(vals: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), dpi=200, facecolor=SURFACE)
    for ax, name in zip(axes, LADDER_SIDS, strict=True):
        _style(ax)
        ax.axhline(EXPERIMENT[name], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"exp. ~{EXPERIMENT[name]:.0f}", (1.12, EXPERIMENT[name]), color=INK2, fontsize=8, va="bottom", ha="right")
        for i, (model, (label, color, marker)) in enumerate(MODELS.items()):
            v = vals.get(model, {}).get(name)
            if not v:
                continue
            dx = (i - 1) * 0.04  # small offset so overlapping models stay visible
            ax.plot([dx, 1 + dx], [v["L1"], v["L2"]], color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.set_xticks([0, 1], ["L1  3ph RTA", "L2  3ph LBTE"], color=INK)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0, None)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ratios = [m[name]["L2"] / m[name]["L1"] for m in vals.values() if name in m]
        if ratios:
            ax.text(0.5, 0.04, f"LBTE/RTA {min(ratios):.2f}-{max(ratios):.2f}", transform=ax.transAxes,
                    color=INK2, fontsize=8, ha="center")
    axes[0].set_ylabel("κ at 300 K, W/mK", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "s0_ladder.png", facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    k = ksrme_rows()
    fig_ksrme(k)
    lad = ladder_values()
    fig_ladder(lad)
    print(json.dumps({"ksrme": {m: round(v["srme"], 4) for m, v in k.items()}, "ladder": lad}, indent=1))
