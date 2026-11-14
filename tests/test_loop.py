"""Replay strategies on a synthetic universe (no result store needed)."""

import numpy as np
import pandas as pd
import pytest

from mga.loop.data import FEATURES
from mga.loop.replay import ReplayConfig, run_one
from mga.loop.surrogate import STRATEGIES


@pytest.fixture


def universe():
    rng = np.random.default_rng(0)
    n = 40
    t = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=list(FEATURES))
    t["log_kappa_l1"] = t[FEATURES[0]] + 0.3 * rng.normal(size=n)
    t["kappa_l1"] = 10 ** t["log_kappa_l1"]
    t["kappa_slack"] = 10 ** t[FEATURES[1]]
    t["l1_gpu_h"] = 1.0
    t.index = [f"s{i}" for i in range(n)]
    return t


@pytest.mark.parametrize("strategy", STRATEGIES)


def test_every_strategy_runs(universe, strategy):
    cfg = ReplayConfig(n_init=5, batch=3, n_rounds=3)
    curve = run_one(universe, strategy, 0, cfg)
    assert [c["n_labelled"] for c in curve] == [5, 8, 11, 14]
    assert curve[-1]["gpu_h"] == 14.0


def test_slack_ucb_starts_from_slack_head(universe):
    cfg = ReplayConfig(n_init=5, batch=3, n_rounds=1)
    slack_head = set(universe["kappa_slack"].nlargest(5).index)
    best_head = universe.loc[list(slack_head), "kappa_l1"].max()
    su = run_one(universe, "slack_ucb", 0, cfg)
    sl = run_one(universe, "slack", 0, cfg)
    # round 0 identical to the Slack ranking, not to the shared random batch of ucb
    assert su[0]["best_kappa"] == sl[0]["best_kappa"] == best_head
    assert su[0]["hits"] == sl[0]["hits"]
    # from round 1 on it learns: the surrogate metrics are recorded, unlike the static ranking
    assert "spearman_rest" in su[1] and "spearman_rest" not in sl[1]
