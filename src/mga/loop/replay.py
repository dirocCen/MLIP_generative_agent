"""Replay acquisition strategies on a fully labelled pool ("compute everything, then replay").

Every structure of the universe already has an L1 label, so a strategy run is a pure
simulation: pick a batch, reveal its labels, retrain, repeat. All strategies of one seed share
the same random initial batch, so differences come from the strategy, not the start.

Per round the curves record: structures labelled, GPU-hours spent on L1 (the measured wall time
of the structures picked), hits (kappa_L1 >= hit threshold), hidden targets found, best kappa so
far, and, for learning strategies, the surrogate's Spearman rho / RMSE (log10) on the still-
unlabelled rest of the universe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mga.loop.data import FEATURES
from mga.loop.surrogate import LEARNING, Surrogate, rank


@dataclass
class ReplayConfig:
    n_init: int = 10
    batch: int = 10
    n_rounds: int = 10
    hit_quantile: float = 0.9
    beta: float = 1.0
    targets: dict[str, list[str]] = field(default_factory=dict)  # name -> sids that count as found


def hit_threshold(universe: pd.DataFrame, q: float) -> float:
    return float(universe["kappa_l1"].quantile(q))


def run_one(universe: pd.DataFrame, strategy: str, seed: int, cfg: ReplayConfig) -> list[dict]:
    X = universe[list(FEATURES)].to_numpy(float)
    y = universe["log_kappa_l1"].to_numpy(float)
    kappa = universe["kappa_l1"].to_numpy(float)
    cost = universe["l1_gpu_h"].fillna(universe["l1_gpu_h"].median()).to_numpy(float)
    slack = universe["kappa_slack"].to_numpy(float)
    sids = universe.index.to_numpy()
    thr = hit_threshold(universe, cfg.hit_quantile)
    target_of = {s: name for name, ss in cfg.targets.items() for s in ss}

    rng = np.random.default_rng(seed)
    n = len(universe)
    init = rng.permutation(n)[: cfg.n_init]
    chosen: list[int] = []
    curve: list[dict] = []

    def record(rnd: int, fit_metrics: dict) -> None:
        idx = np.array(chosen, dtype=int)
        found = sorted({target_of[s] for s in sids[idx] if s in target_of})
        curve.append({
            "strategy": strategy, "seed": seed, "round": rnd,
            "n_labelled": len(idx),
            "gpu_h": float(cost[idx].sum()),
            "hits": int((kappa[idx] >= thr).sum()),
            "targets_found": len(found), "targets": found,
            "best_kappa": float(kappa[idx].max()),
            **fit_metrics,
        })

    # Round 0: the shared random batch for the learning strategies; static strategies
    # simply take the head of their ranking (same budget).
    if strategy in LEARNING or strategy == "random":
        chosen.extend(init.tolist())
    else:
        chosen.extend(rank(strategy, X, slack, rng)[: cfg.n_init].tolist())
    record(0, {})

    static_order = None
    for rnd in range(1, cfg.n_rounds + 1):
        rest = np.setdiff1d(np.arange(n), chosen)
        if len(rest) == 0:
            break
        metrics: dict = {}
        if strategy in LEARNING:
            sur = Surrogate(seed).fit(X[chosen], y[chosen])
            order = rank(strategy, X[rest], slack[rest], rng, sur, cfg.beta)
            mu, _ = sur.predict(X[rest])
            if len(rest) > 2:
                metrics = {
                    "spearman_rest": float(spearmanr(mu, y[rest]).statistic),
                    "rmse_log_rest": float(np.sqrt(np.mean((mu - y[rest]) ** 2))),
                }
            pick = rest[order[: cfg.batch]]
        else:
            if static_order is None:
                static_order = [i for i in (rank(strategy, X, slack, rng) if strategy == "slack"
                                            else rng.permutation(n)) if i not in set(chosen)]
            pick = np.array(static_order[: cfg.batch], dtype=int)
            static_order = static_order[cfg.batch:]
        chosen.extend(pick.tolist())
        record(rnd, metrics)
    return curve


def run(universe: pd.DataFrame, strategies, seeds, cfg: ReplayConfig) -> pd.DataFrame:
    rows = [r for s in strategies for seed in seeds for r in run_one(universe, s, seed, cfg)]
    return pd.DataFrame(rows)
