"""One table per pool: L0 descriptors (features), L1 kappa (label), compute cost, flags.

Results are read from the content-addressed result store, so a structure computed in an
earlier campaign (same model, same protocol hash) is reused without recomputation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from mga import registry
from mga.results import ResultMissing, load_result

L0_KIND, L1_KIND = "mlip_fc2_descriptors", "mlip_kappa"
DEFAULT_MODEL = "dpa41_max_oam"
DEFAULT_L0_HASH = "c7704b907723"
DEFAULT_L1_HASH = "cc4191d4d8d0"

# Descriptors used as surrogate features. Positive, heavy-tailed ones enter as log10.
LOG_FEATURES = (
    "kappa_slack_W_mK",
    "kappa_ctau_1ps_W_mK",
    "kappa_proxy_cv2_over_wjdos",
    "theta_acoustic_K",
    "mass_mean_amu",
    "wjdos_acoustic_cv_mean",
    "gamma_acoustic_rms",
)
LIN_FEATURES = (
    "v_acoustic_rms_km_s",
    "v_rms_km_s",
    "acoustic_max_thz",
    "freq_max_thz",
    "volume_per_atom_A3",
    "ao_gap_ratio",
    "acoustic_cv_fraction",
    "acoustic_branch_spread",
    "gamma_cv_mean",
    "n_atoms_primitive",
)
FEATURES = tuple(f"log_{k}" for k in LOG_FEATURES) + LIN_FEATURES


def _load(kind: str, model: str, sid: str, phash: str) -> dict | None:
    try:
        return load_result(kind, model, sid, phash)
    except ResultMissing:
        return None


def _l1_kappa(doc: dict) -> float | None:
    d = doc["data"]
    if not d.get("ok") or d.get("conductivity_skipped") or not d.get("kappa_tot_avg"):
        return None
    k = d["kappa_tot_avg"][0]
    return float(k) if k and k > 0 else None


def _gpu_h(doc: dict) -> float:
    # data.wall_time_s is per structure; provenance.job_wall_time_s is the whole chunk job.
    return float(doc["data"].get("wall_time_s") or 0.0) / 3600.0


def build_table(
    sids: Iterable[str],
    model: str = DEFAULT_MODEL,
    l0_hash: str = DEFAULT_L0_HASH,
    l1_hash: str = DEFAULT_L1_HASH,
) -> pd.DataFrame:
    """Rows for every sid with an L0 result; ``kappa_l1`` is NaN where L1 is missing or failed.

    Columns: sid, formula, chemsys, l0_ok (no imaginary modes, descriptors present), the
    feature columns of ``FEATURES``, kappa_slack, kappa_l1, log_kappa_l1, l0_gpu_h, l1_gpu_h.
    """
    rows = []
    for e in registry.iter_entries(list(sids)):
        sid = e["sid"]
        l0 = _load(L0_KIND, model, sid, l0_hash)
        if l0 is None:
            continue
        d0 = l0["data"]
        desc = d0.get("descriptors") or {}
        row = {
            "sid": sid,
            "formula": e["reduced_formula"],
            "chemsys": "-".join(sorted(e["elements"])),
            "l0_ok": bool(d0.get("ok") and not d0.get("has_imaginary_modes") and desc),
            "min_frequency_thz": d0.get("min_frequency_thz"),
            "l0_gpu_h": _gpu_h(l0),
            "kappa_slack": desc.get("kappa_slack_W_mK", math.nan),
        }
        for k in LOG_FEATURES:
            v = desc.get(k)
            row[f"log_{k}"] = math.log10(v) if v and v > 0 else math.nan
        for k in LIN_FEATURES:
            row[k] = desc.get(k, math.nan)
        l1 = _load(L1_KIND, model, sid, l1_hash)
        kappa = _l1_kappa(l1) if l1 else None
        row["kappa_l1"] = kappa if kappa is not None else math.nan
        row["log_kappa_l1"] = math.log10(kappa) if kappa else math.nan
        row["l1_gpu_h"] = _gpu_h(l1) if l1 else math.nan
        rows.append(row)
    if not rows:
        raise ValueError("no L0 results for these sids: pass the relaxed / L0 set (relaxation registers new sids)")
    return pd.DataFrame(rows).set_index("sid", drop=False)


def labelled(table: pd.DataFrame) -> pd.DataFrame:
    """L0-stable rows with a finite L1 label and complete features: the replay universe."""
    t = table[table["l0_ok"] & table["kappa_l1"].notna()]
    return t.dropna(subset=list(FEATURES))
