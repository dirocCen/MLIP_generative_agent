"""Budget ledger and escalation baselines for the multi-fidelity ladder (DESIGN §3.2).

Pure functions on a *ladder table* built from the result cache:

    table[sid] = {"L0": Cell, "L1": Cell, "L2": Cell, "L3": Cell}   (rungs that were computed)
    Cell = {"cost_h": summed over models, "values": per-model numbers, "stable": per-model bools (L0)}

What the number at each rung means:
- L0: a descriptor (default the Slack estimate), committee mean; used for ranking only
- L1: Wigner-RTA kappa, committee mean
- L2: LBTE / RTA ratio (mean over models); estimate = L1 x ratio
- L3: kappa(3+4ph) / kappa(3ph) (FourPhonon); estimate = (L2 estimate, else L1) x ratio

"Truth" for a candidate is the estimate from the highest rung in the cache. That is what the
fair comparison needs: L0-L2 are computed for the whole pool and L3 for a superset of what any
policy selects, so every policy is replayed on the same cache (DESIGN §3.2). L4 (DFT) is only
ever computed for what was actually chosen; it has no counterfactual and is not replayed here.

Cost unit: GPU-hours of a one-GPU datalab job, from the per-structure ``wall_time_s`` each
result records (calculator loading per chunk is not included).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

RUNGS = ("L0", "L1", "L2", "L3")
RUNG_KINDS = {"L0": "mlip_fc2_descriptors", "L1": "mlip_kappa", "L2": "mlip_lbte", "L3": "mlip_fc4"}
DEFAULT_L0_SCORE = "kappa_slack_W_mK"


# --- one result -> its number at that rung --------------------------------------------------
def _first(v):
    return v[0] if isinstance(v, list) and v else v


def _finite(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def rung_value(rung: str, data: dict[str, Any], l0_score: str = DEFAULT_L0_SCORE) -> float | None:
    if not data.get("ok"):
        return None
    if rung == "L0":
        return _finite((data.get("descriptors") or {}).get(l0_score))
    if rung == "L1":
        return _finite(_first(data.get("kappa_tot_avg")))
    if rung == "L2":
        return _finite(_first(data.get("lbte_over_rta")))
    return _finite(data.get("four_phonon_ratio"))


def l0_stable(data: dict[str, Any]) -> bool:
    return bool(data.get("ok")) and not data.get("has_imaginary_modes")


def add_result(table: dict[str, dict[str, Any]], rung: str, sid: str, data: dict[str, Any],
               l0_score: str = DEFAULT_L0_SCORE) -> None:
    cell = table.setdefault(sid, {}).setdefault(rung, {"cost_h": 0.0, "values": [], "stable": [], "n_models": 0})
    cell["cost_h"] += float(data.get("wall_time_s") or 0.0) / 3600.0
    cell["n_models"] += 1
    v = rung_value(rung, data, l0_score)
    if v is not None:
        cell["values"].append(v)
    if rung == "L0":
        cell["stable"].append(l0_stable(data))


# --- scores ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def scores(row: dict[str, Any]) -> dict[str, Any]:
    """Best estimate after each rung for one candidate (None where that rung is missing)."""
    v = {r: _mean(row[r]["values"]) if r in row else None for r in RUNGS}
    s1 = v["L1"]
    s2 = s1 * v["L2"] if s1 is not None and v["L2"] is not None else None
    base3 = s2 if s2 is not None else s1
    s3 = base3 * v["L3"] if base3 is not None and v["L3"] is not None else None
    truth, truth_rung = next(((s, r) for s, r in ((s3, "L3"), (s2, "L2"), (s1, "L1")) if s is not None), (None, None))
    stable_votes = row.get("L0", {}).get("stable", [])
    return {
        "stable": (sum(stable_votes) * 2 > len(stable_votes)) if stable_votes else None,  # committee majority
        "after": {"L0": v["L0"], "L1": s1, "L2": s2 if s2 is not None else s1, "L3": s3 if s3 is not None else base3},
        "truth": truth, "truth_rung": truth_rung,
    }


# --- policies ----------------------------------------------------------------------------
def _cost(table, sid: str, rung: str, fallback: dict[str, float]) -> tuple[float, bool]:
    cell = table.get(sid, {}).get(rung)
    return (cell["cost_h"], True) if cell else (fallback[rung], False)


def median_costs(table) -> dict[str, float]:
    out = {}
    for r in RUNGS:
        c = [row[r]["cost_h"] for row in table.values() if r in row]
        out[r] = float(np.median(c)) if c else 0.0
    return out


def run_policy(table, sc: dict[str, dict[str, Any]], ks: list[int], order, budget_h: float | None,
               hit_threshold: float) -> dict[str, Any]:
    """L0 on the whole pool; then at L1, L2, L3 pick up to k_r of the survivors in the order
    ``order(survivors, rung_before)`` gives, while the budget allows (a candidate that does not
    fit is passed over, a cheaper one further down may still fit)."""
    fallback = median_costs(table)
    pool = list(table)
    spent, n_uncached = 0.0, 0
    for sid in pool:
        c, ok = _cost(table, sid, "L0", fallback)
        spent += c
        n_uncached += not ok
    survivors = [s for s in pool if sc[s]["stable"]]
    chosen: dict[str, list[str]] = {"L0": pool}
    for rung, prev, k in zip(RUNGS[1:], RUNGS[:-1], ks):
        picked = []
        for sid in order(survivors, prev):
            if len(picked) >= k:
                break
            c, ok = _cost(table, sid, rung, fallback)
            if budget_h is not None and spent + c > budget_h + 1e-12:
                continue
            spent += c
            n_uncached += not ok
            picked.append(sid)
        chosen[rung] = picked
        # an L1 failure stops a candidate; an L2 skip (collision matrix too large) does not, the
        # L1 estimate carries on (scores() falls back to it)
        survivors = [s for s in picked if sc[s]["after"][rung] is not None]
    return {"chosen": chosen, **evaluate(chosen, sc, hit_threshold), "spent_h": spent, "n_uncached_cost": n_uncached}


def evaluate(chosen: dict[str, list[str]], sc: dict[str, dict[str, Any]], hit_threshold: float) -> dict[str, Any]:
    hits = {s for s, v in sc.items() if v["truth"] is not None and v["truth"] > hit_threshold}
    final = chosen.get("L3") or []
    found = sorted(hits & set(final))
    return {"n_selected": {r: len(chosen.get(r) or []) for r in RUNGS}, "hits_found": len(found),
            "hits_in_pool": len(hits), "recall": len(found) / len(hits) if hits else None, "found": found}


def fixed_order(sc):
    def order(survivors, prev):
        key = lambda s: (sc[s]["after"][prev] is None, -(sc[s]["after"][prev] or 0.0))  # noqa: E731
        return sorted(survivors, key=key)

    return order


def random_order(rng: np.random.Generator):
    return lambda survivors, _prev: [survivors[i] for i in rng.permutation(len(survivors))]


def ks_for_budget(table, sc, budget_h: float, hit_threshold: float) -> tuple[list[int], float]:
    """Largest common retention fraction f (k_r = ceil(n_stable * f^r)) whose fixed-rule run fits
    the budget without the budget cut-off. Returns (ks, f)."""
    n_stable = sum(bool(v["stable"]) for v in sc.values())
    best = ([0, 0, 0], 0.0)
    for f in np.linspace(0.01, 1.0, 100):
        ks = [max(1, math.ceil(n_stable * f**r)) for r in (1, 2, 3)]
        if run_policy(table, sc, ks, fixed_order(sc), None, hit_threshold)["spent_h"] <= budget_h:
            best = (ks, float(f))
    return best


def spearman_between_rungs(sc) -> dict[str, dict[str, Any]]:
    """Rank agreement of the estimates after consecutive rungs (does a cheap rung order the
    candidates like the next one?)."""
    from scipy.stats import spearmanr

    out = {}
    for a, b in (("L0", "L1"), ("L1", "L2"), ("L2", "L3")):
        pairs = [(v["after"][a], v["after"][b]) for v in sc.values()
                 if v["after"][a] is not None and v["after"][b] is not None]
        if len(pairs) >= 3:
            x, y = zip(*pairs)
            rho = spearmanr(x, y).statistic
            out[f"{a}->{b}"] = {"n": len(pairs), "rho": None if np.isnan(rho) else float(rho)}
        else:
            out[f"{a}->{b}"] = {"n": len(pairs), "rho": None}
    return out
