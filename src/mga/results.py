"""Result cache: ``data/results/<kind>/<model>/<key>__<param_hash>.json`` (+ ``.npz``).

Submission tools look here before submitting (no recomputation unless ``force``),
``fetch_results`` writes here, and replay mode reads only from here.
Large arrays (mode kappa, frequencies) go to the ``.npz`` next to the JSON.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from mga.config import paths, utc_now
from mga.utils import check_name, read_json, write_json_atomic

KINDS = (
    "mlip_relax",
    "mlip_kappa",
    "mlip_fc2_descriptors",  # ladder L0
    "mlip_lbte",  # ladder L2
    "mlip_fc4",  # ladder L3
    "mattergen",
    "vasp_relax_static",
    "vasp_phonon_fd",
    "vasp_phono3py_fd",
    "vasp_convergence",
)


class ResultMissing(LookupError):
    pass


def result_path(kind: str, model: str, key: str, phash: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"Unknown result kind {kind!r}")
    check_name(model, "model")
    check_name(key, "result key")
    return paths.results / kind / model / f"{key}__{phash}.json"


def frames_path(kind: str, model: str, key: str, phash: str) -> Path:
    """DFT frames (structure, energy, forces, stress) of a VASP result, as extxyz."""
    return result_path(kind, model, key, phash).with_suffix(".frames.extxyz")


def has_result(kind: str, model: str, key: str, phash: str) -> bool:
    """A stored record exists (success or failure)."""
    return result_path(kind, model, key, phash).exists()


def is_cached(kind: str, model: str, key: str, phash: str) -> bool:
    """A stored record that submission and ingestion should not redo: anything but a raised
    exception (``ok`` false with an ``error``). CUDA OOM, node or I/O trouble are not properties of
    the structure, and ``force`` would rerun the whole set to retry one of them (job 963847); the
    retry's result then replaces the failure on fetch. Physical outcomes (imaginary modes, broken
    symmetry) are stored with ok=True and stay cached."""
    path = result_path(kind, model, key, phash)
    if not path.exists():
        return False
    data = read_json(path).get("data")
    return not (isinstance(data, dict) and data.get("ok") is False and data.get("error"))


def save_result(
    kind: str,
    model: str,
    key: str,
    phash: str,
    data: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    path = result_path(kind, model, key, phash)
    doc = {
        "kind": kind,
        "model": model,
        "key": key,
        "param_hash": phash,
        "saved_at": utc_now(),
        "provenance": provenance or {},
        "data": data,
        "arrays": sorted(arrays) if arrays else [],
    }
    if arrays:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
    write_json_atomic(path, doc)
    return path


def load_result(kind: str, model: str, key: str, phash: str, *, with_arrays: bool = False) -> dict[str, Any]:
    path = result_path(kind, model, key, phash)
    if not path.exists():
        raise ResultMissing(f"No cached {kind} result for model={model} key={key} params={phash}")
    doc = read_json(path)
    if with_arrays and doc.get("arrays"):
        with np.load(path.with_suffix(".npz")) as npz:
            doc["array_data"] = {name: npz[name] for name in npz.files}
    return doc


def iter_results(kind: str, model: str | None = None) -> Iterator[dict[str, Any]]:
    base = paths.results / kind
    pattern = f"{model}/*.json" if model else "*/*.json"
    for path in sorted(base.glob(pattern)):
        yield read_json(path)
