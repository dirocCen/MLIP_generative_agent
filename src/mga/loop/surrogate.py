"""Gaussian-process surrogate of log10 kappa_L1 from L0 descriptors, and acquisition strategies.

Strategies rank candidates; the loop takes the top ``batch`` of the ranking.

- ``random``  uniform order (seeded): the no-information baseline.
- ``slack``   descending Slack estimate from L0: the old open-loop funnel, which never learns.
- ``greedy``  descending surrogate mean.
- ``ucb``     descending mean + beta * std: explores where the surrogate is unsure.
- ``slack_ucb`` cold start from the head of the Slack ranking instead of a random batch, then
  ``ucb``. The physics prior picks the first batch, the surrogate corrects its blind spots. Once
  labels exist (every real round) it ranks exactly like ``ucb``; only the replay's round 0 differs.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

STRATEGIES = ("random", "slack", "greedy", "ucb", "slack_ucb")
LEARNING = frozenset({"greedy", "ucb", "slack_ucb"})
SLACK_START = frozenset({"slack_ucb"})  # learning strategies whose first batch is the Slack head


class Surrogate:
    """GP with a Matern-5/2 ARD kernel plus white noise on standardized features."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.scaler: StandardScaler | None = None
        self.gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Surrogate:
        self.scaler = StandardScaler().fit(X)
        n_feat = X.shape[1]
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.full(n_feat, 3.0), length_scale_bounds=(0.3, 1e3), nu=2.5)
            + WhiteKernel(0.01, (1e-5, 1.0))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=self.seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self.scaler.transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.gp is not None and self.scaler is not None, "fit first"
        mu, sd = self.gp.predict(self.scaler.transform(X), return_std=True)
        return mu, sd


def rank(
    strategy: str,
    X_cand: np.ndarray,
    slack_cand: np.ndarray,
    rng: np.random.Generator,
    surrogate: Surrogate | None = None,
    beta: float = 1.0,
) -> np.ndarray:
    """Indices of the candidates, best first."""
    if strategy == "random":
        return rng.permutation(len(X_cand))
    if strategy == "slack":
        return np.argsort(-np.nan_to_num(slack_cand, nan=-np.inf), kind="stable")
    if strategy not in LEARNING:
        raise ValueError(f"unknown strategy {strategy!r}; one of {STRATEGIES}")
    assert surrogate is not None
    mu, sd = surrogate.predict(X_cand)
    score = mu if strategy == "greedy" else mu + beta * sd  # ucb and slack_ucb
    return np.argsort(-score, kind="stable")
