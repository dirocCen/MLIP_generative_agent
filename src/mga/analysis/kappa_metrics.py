"""kappa metrics: SRD / SRE / SRME vs phononDB, and model-vs-model disagreement.

SRD and SRME follow matbench-discovery ``metrics/phonons.py`` (commit 2b7bff9, MIT):
- SRD  = 2 (k_pred - k_true) / (k_pred + k_true), SRE = |SRD|, both in [-2, 2] / [0, 2]
- SRME = 2 * sum_modes |mk_pred - mk_true| / sum(weights) / (k_pred + k_true), in [0, 2]
- kappa averages are the mean of the Voigt diagonal (first three components)
- censored to 2: imaginary modes, space group changed in relaxation, failed or invalid output
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from typing import Any

import numpy as np

from mga.config import paths

KAPPA_ERROR_MAX = 2.0


class KappaComparisonError(ValueError):
    pass


def kappa_avg(kappa: np.ndarray) -> np.ndarray:
    return np.asarray(kappa, dtype=float)[..., :3].mean(axis=-1)


def srd(pred: float | None, true: float) -> float:
    if pred is None or not np.isfinite(pred) or pred < 0:
        return -KAPPA_ERROR_MAX
    denom = pred + true
    return 0.0 if denom == 0 else KAPPA_ERROR_MAX * (pred - true) / denom


def srme(mode_pred: np.ndarray, weights: np.ndarray, k_pred: np.ndarray, mode_true: np.ndarray, k_true: np.ndarray) -> np.ndarray:
    """Per-temperature SRME; mode arrays (T, nq, nb), totals (T,)."""
    mode_pred, mode_true = np.asarray(mode_pred, float), np.asarray(mode_true, float)
    k_pred, k_true = np.atleast_1d(np.asarray(k_pred, float)), np.atleast_1d(np.asarray(k_true, float))
    if mode_pred.shape != mode_true.shape:
        raise KappaComparisonError(f"mode kappa shapes differ: {mode_pred.shape} vs {mode_true.shape}")
    if not (np.all(np.isfinite(mode_pred)) and np.all(np.isfinite(k_pred)) and np.sum(weights) > 0):
        return np.full(k_true.shape, KAPPA_ERROR_MAX)
    micro = np.abs(mode_pred - mode_true).sum(axis=tuple(range(1, mode_pred.ndim))) / np.sum(weights)
    denom = k_pred + k_true
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denom == 0, np.where(micro == 0, 0.0, KAPPA_ERROR_MAX), KAPPA_ERROR_MAX * micro / denom)
    valid = np.isfinite(scores) & (scores >= 0) & (scores <= KAPPA_ERROR_MAX)
    return np.where(valid, scores, KAPPA_ERROR_MAX)


# --- phononDB reference ----------------------------------------------------------
@lru_cache(maxsize=1)
def reference_table() -> dict[str, dict[str, Any]]:
    """mp_id -> reference row (kappa_SRME noNAC, k_SRME commit d4cf9c5)."""
    path = paths.reference / "kappa_srme" / "kappas_phononDB_PBE_noNAC.json.gz"
    with gzip.open(path) as fh:
        cols = json.load(fh)
    rows = {}
    for idx in cols["mp_id"]:
        row = {col: cols[col][idx] for col in cols}
        rows[row["mp_id"]] = row
    return rows


def _censor_reason(pred: dict[str, Any], expected_spg: int | None) -> str | None:
    if not pred.get("ok"):
        return f"failed: {pred.get('error')}"
    if pred.get("has_imaginary_modes"):
        return "imaginary_modes"
    relax = pred.get("relax") or {}
    final_spg = relax.get("final_spg_num")
    initial = relax.get("initial_spg_num") or expected_spg
    if final_spg and initial and final_spg != initial:
        return "symmetry_changed"
    if pred.get("conductivity_skipped"):
        return f"skipped: {pred['conductivity_skipped']}"
    return None


def compare_to_reference(pred: dict[str, Any], arrays: dict[str, np.ndarray], mp_id: str) -> dict[str, Any]:
    """One structure: SRD / SRE / SRME vs phononDB (first temperature)."""
    ref = reference_table()[mp_id]
    k_true = float(ref["kappa_TOT_ave"][0])
    out: dict[str, Any] = {"mp_id": mp_id, "name": ref["name"], "kappa_ref": k_true}
    reason = _censor_reason(pred, int(ref["symm.no"]))
    if reason is not None:
        return {**out, "kappa_pred": None, "srd": -KAPPA_ERROR_MAX, "sre": KAPPA_ERROR_MAX, "srme": KAPPA_ERROR_MAX, "censored": reason}
    k_pred = kappa_avg(arrays["kappa_tot_rta"])
    mode_pred = kappa_avg(arrays["mode_kappa_tot_rta"])
    mode_true = np.asarray(ref["mode_kappa_TOT_ave"], float)
    d = srd(float(k_pred[0]), k_true)
    try:
        s = float(srme(mode_pred, arrays["mode_weights"], k_pred, mode_true, np.asarray(ref["kappa_TOT_ave"]))[0])
        note = None
    except KappaComparisonError as exc:  # q-mesh differs from the reference
        s, note = KAPPA_ERROR_MAX, str(exc)
    return {**out, "kappa_pred": float(k_pred[0]), "srd": d, "sre": abs(d), "srme": s, "censored": note}


# --- model vs model -------------------------------------------------------------------
def compare_models(a: tuple[dict, dict], b: tuple[dict, dict]) -> dict[str, Any]:
    """Disagreement between two models on one structure: ((data, arrays), (data, arrays)).

    Mode-resolved comparison needs the same q-points (same mesh and symmetry);
    otherwise only the total-kappa SRD is reported and ``mode_comparable`` is False.
    """
    (da, aa), (db, ab) = a, b
    out: dict[str, Any] = {}
    for tag, d in (("a", da), ("b", db)):
        reason = _censor_reason(d, None)
        out[f"censored_{tag}"] = reason
    if out["censored_a"] or out["censored_b"]:
        out.update(srd=None, srme=None, mode_comparable=False)
        return out
    ka, kb = kappa_avg(aa["kappa_tot_rta"]), kappa_avg(ab["kappa_tot_rta"])
    out["kappa_a"], out["kappa_b"] = float(ka[0]), float(kb[0])
    out["srd"] = srd(float(ka[0]), float(kb[0]))  # sign: a relative to b
    same_q = aa["q_points"].shape == ab["q_points"].shape and np.allclose(aa["q_points"], ab["q_points"], atol=1e-8)
    out["mode_comparable"] = bool(same_q)
    if same_q:
        out["srme"] = float(srme(kappa_avg(aa["mode_kappa_tot_rta"]), aa["mode_weights"], ka, kappa_avg(ab["mode_kappa_tot_rta"]), kb)[0])
        fa, fb = np.asarray(aa["frequencies"]), np.asarray(ab["frequencies"])
        w = np.asarray(aa["mode_weights"], float)[:, None]
        out["freq_mae_thz"] = float((np.abs(fa - fb) * w).sum() / (w.sum() * fa.shape[1]))
        out["freq_max_abs_thz"] = float(np.abs(fa - fb).max())
    else:
        out["srme"] = None
        out["note"] = "q-points differ (different mesh or symmetry); mode-resolved comparison not possible"
    return out
