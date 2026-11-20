"""One table per pool: L0 descriptors (features), L1/L2 kappa (label), compute cost, flags.

Results are read from the content-addressed result store, so a structure computed in an
earlier campaign (same model, same protocol hash) is reused without recomputation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import pandas as pd

from mga import registry
from mga.results import ResultMissing, load_result

L0_KIND, L1_KIND, L2_KIND = "mlip_fc2_descriptors", "mlip_kappa", "mlip_lbte"
DEFAULT_MODEL = "dpa41_max_oam"
# L0 = FC2 descriptors with fc2_min_width 10 A (2026-11-12): the 14 A protocol (c7704b907723)
# OOMs on 44 GB GPUs for dense B-C-N; on 31 structures stable under both, Slack Spearman 0.975.
DEFAULT_L0_HASH = "504f29978e8c"
DEFAULT_L1_HASH = "cc4191d4d8d0"
# L1 protocols whose kappa counts as the label, in priority order (first finite value wins).
# FC2 12 A (4187d607eb10, 11-14): kappa within 4 % of 14 A on diamond / c-BN / BAs / BC2N, and
# the dense B-C-N cells that OOMed at 14 A (>= 768 atoms) shrink to 720, which still OOMs on
# A100 80 GB; those run at FC2 10 A (5b38ae92e6ac, 480 atoms, kappa within 6 % of 14 A).
# 9cd3d1dbe642 (11-19): cc4191d4d8d0 with kappa_if_symmetry_broken, for inputs whose symmetry
# FixSymmetry *raised* (slightly distorted MP cells) and the old check skipped as "broken".
L1_HASHES = (DEFAULT_L1_HASH, "4187d607eb10", "5b38ae92e6ac", "9cd3d1dbe642")
# L2 = LBTE on the L1 force constants: the default protocol, and the same with
# kappa_if_symmetry_broken (pairs with 9cd3d1dbe642 so it hits that FC cache).
L2_HASHES = ("134f7ea6f17b", "2ba818643731")

# Label sanity (11-19). A kappa that is not positive, or absurdly large, is a numerical failure
# (RTA diverges when linewidths -> 0; the LBTE pseudo-inverse can turn negative), not physics.
KAPPA_MAX_W_MK = 1e4  # diamond, the record bulk solid, is ~2-3e3
# LBTE / RTA outside this window: one of the two solutions is broken. Measured range on valid
# labels 1.00-1.9 (BAs ~1.9: Normal-process dominance from its large acoustic-optic gap).
LBTE_OVER_RTA = (0.8, 3.0)
LABELS = ("l1", "l2", "l2_or_l1")

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


def _valid_kappa(k) -> bool:
    return k is not None and math.isfinite(k) and 0 < k <= KAPPA_MAX_W_MK


def _l1_kappa(doc: dict) -> float | None:
    d = doc["data"]
    if not d.get("ok") or d.get("conductivity_skipped") or not d.get("kappa_tot_avg"):
        return None
    k = d["kappa_tot_avg"][0]
    return float(k) if _valid_kappa(k) else None


def _l2_kappa(doc: dict) -> tuple[float | None, float | None]:
    """(LBTE kappa or None if invalid, LBTE/RTA ratio) at the first temperature."""
    d = doc["data"]
    if not d.get("ok") or not d.get("kappa_tot_exact_avg") or not d.get("kappa_tot_rta_avg"):
        return None, None
    k, rta = d["kappa_tot_exact_avg"][0], d["kappa_tot_rta_avg"][0]
    ratio = k / rta if rta else math.nan
    ok = _valid_kappa(k) and _valid_kappa(rta) and LBTE_OVER_RTA[0] <= ratio <= LBTE_OVER_RTA[1]
    return (float(k) if ok else None), float(ratio)


def _gpu_h(doc: dict) -> float:
    # data.wall_time_s is per structure; provenance.job_wall_time_s is the whole chunk job.
    return float(doc["data"].get("wall_time_s") or 0.0) / 3600.0


def build_table(
    sids: Iterable[str],
    model: str = DEFAULT_MODEL,
    l0_hash: str = DEFAULT_L0_HASH,
    l1_hash: str | Sequence[str] = L1_HASHES,
    l2_hash: str | Sequence[str] = L2_HASHES,
    label: str = "l2_or_l1",
) -> pd.DataFrame:
    """Rows for every sid with an L0 result; ``kappa_l1`` is NaN where L1 is missing or failed.

    Columns: sid, formula, chemsys, l0_ok (no imaginary modes, descriptors present), the
    feature columns of ``FEATURES``, kappa_slack, kappa_l1, log_kappa_l1, l1_hash (protocol the
    label came from), l0_gpu_h, l1_gpu_h; kappa_l2, l2_hash, lbte_over_rta, l2_invalid (an L2
    result exists but fails the sanity window); and the training label kappa / log_kappa with
    label_source ("l1" | "l2" | None). ``l1_hash`` / ``l2_hash``: one hash or several in
    priority order.

    ``label``: "l1" (RTA, the old behaviour), "l2" (LBTE only), or "l2_or_l1" (LBTE where a
    valid one exists, else L1; a structure whose L2 exists but is invalid gets no label, since
    the failure usually means its linewidths are unreliable for RTA too).
    """
    if label not in LABELS:
        raise ValueError(f"label {label!r}: one of {LABELS}")
    l1_hashes = (l1_hash,) if isinstance(l1_hash, str) else tuple(l1_hash)
    l2_hashes = (l2_hash,) if isinstance(l2_hash, str) else tuple(l2_hash)
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
        kappa, l1, used = None, None, None
        for h in l1_hashes:
            doc = _load(L1_KIND, model, sid, h)
            if doc is None:
                continue
            l1 = l1 or doc  # cost of a failed attempt still counts if no hash gives a label
            if (kappa := _l1_kappa(doc)) is not None:
                l1, used = doc, h
                break
        row["kappa_l1"] = kappa if kappa is not None else math.nan
        row["log_kappa_l1"] = math.log10(kappa) if kappa else math.nan
        row["l1_hash"] = used
        row["l1_gpu_h"] = _gpu_h(l1) if l1 else math.nan
        k2, ratio, used2, seen2 = None, math.nan, None, False
        for h in l2_hashes:
            doc = _load(L2_KIND, model, sid, h)
            if doc is None:
                continue
            seen2 = True
            k2, r = _l2_kappa(doc)
            ratio = r if r is not None else ratio
            if k2 is not None:
                used2 = h
                break
        row["kappa_l2"] = k2 if k2 is not None else math.nan
        row["l2_hash"] = used2
        row["lbte_over_rta"] = ratio
        row["l2_invalid"] = seen2 and k2 is None
        if label == "l1":
            y, src = kappa, "l1"
        elif label == "l2" or k2 is not None or seen2:
            y, src = k2, "l2"
        else:
            y, src = kappa, "l1"
        row["kappa"] = y if y is not None else math.nan
        row["log_kappa"] = math.log10(y) if y is not None else math.nan
        row["label_source"] = src if y is not None else None
        rows.append(row)
    if not rows:
        raise ValueError("no L0 results for these sids: pass the relaxed / L0 set (relaxation registers new sids)")
    return pd.DataFrame(rows).set_index("sid", drop=False)


def labelled(table: pd.DataFrame) -> pd.DataFrame:
    """L0-stable rows with a finite label (``kappa``) and complete features: the replay universe."""
    t = table[table["l0_ok"] & table["kappa"].notna()]
    return t.dropna(subset=list(FEATURES))
