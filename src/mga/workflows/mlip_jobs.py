"""jobflow jobs for MLIP batches (one calculator load per chunk of structures).

Outputs are plain JSON types (lists, not numpy arrays). Every per-structure
``arrays`` field goes to the ``arrays`` additional store (GridFS) via
``@job(arrays="arrays")``, so the main documents stay small.
"""

from __future__ import annotations

import time
from typing import Any

from jobflow import Flow, job


def _atoms(struct):
    """A pymatgen Structure or its dict -> ase Atoms. On the worker it is already a Structure:
    jobflow decodes job inputs with MontyDecoder, which turns any dict carrying @module/@class
    back into the object (job 960873 failed on Structure.from_dict(Structure))."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(struct if isinstance(struct, Structure) else Structure.from_dict(struct))


@job(arrays="arrays")
def mlip_relax_chunk(model: dict[str, Any], items: list[dict[str, Any]], relax_params: dict[str, Any]) -> dict[str, Any]:
    """items: [{"sid": ..., "structure": pymatgen dict}]"""
    import numpy as np

    from mga.workflows.calculators import build_calculator, worker_info
    from mga.workflows.kappa_protocol import relax

    t0 = time.perf_counter()
    calc, calc_info = build_calculator(model, relax_params["dtype"])
    results = []
    for item in items:
        res: dict[str, Any] = {"sid": item["sid"], "ok": False, "error": None, "arrays": {}}
        try:
            relaxed, res["relax"] = relax(_atoms(item["structure"]), calc, relax_params)
            res["relaxed_structure"] = {
                "numbers": relaxed.numbers.tolist(),
                "cell": np.asarray(relaxed.cell).tolist(),
                "positions": relaxed.positions.tolist(),
            }
            res["ok"] = True
        except Exception as exc:  # noqa: BLE001
            res["error"] = f"{type(exc).__name__}: {exc}"
        results.append(res)
    return {"results": results, "calculator": calc_info, "worker_info": worker_info(), "wall_time_s": round(time.perf_counter() - t0, 1)}


@job(arrays="arrays")
def mlip_kappa_chunk(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    relax_params: dict[str, Any],
    kappa_params: dict[str, Any],
) -> dict[str, Any]:
    """items: [{"sid", "structure", "settings": fixed supercells + q-mesh, or None, "fc_dir": optional}]"""
    from mga.workflows.calculators import build_calculator, worker_info
    from mga.workflows.kappa_protocol import run_structure

    t0 = time.perf_counter()
    calc, calc_info = build_calculator(model, kappa_params["dtype"])
    results = []
    for item in items:
        res = run_structure(_atoms(item["structure"]), calc, relax_params, kappa_params, item.get("settings"), item.get("fc_dir"))
        res["sid"] = item["sid"]
        results.append(res)
    return {"results": results, "calculator": calc_info, "worker_info": worker_info(), "wall_time_s": round(time.perf_counter() - t0, 1)}


LADDER_RUNGS = {"fc2_descriptors": "mlip_fc2_descriptors", "lbte": "mlip_lbte", "fc4": "mlip_fc4"}


@job(arrays="arrays")
def mlip_ladder_chunk(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    rung: str,
    relax_params: dict[str, Any],
    kappa_params: dict[str, Any],
    rung_params: dict[str, Any],
) -> dict[str, Any]:
    """One ladder rung (workflows/ladder.py) on a chunk of structures.

    items: [{"sid", "structure", "settings" (the L1 supercells / q-mesh), "fc_dir" (L2), "work_dir" (L3)}].
    The calculator is loaded only when a structure needs it: an L2 chunk whose FCs are all in
    the L1 cache never touches the GPU model."""
    from mga.workflows import ladder
    from mga.workflows.calculators import build_calculator, worker_info

    if rung not in LADDER_RUNGS:
        raise ValueError(f"unknown ladder rung {rung!r}")
    t0 = time.perf_counter()
    calc = calc_info = None

    def calculator():
        nonlocal calc, calc_info
        if calc is None:
            calc, calc_info = build_calculator(model, kappa_params["dtype"])
        return calc

    results = []
    for item in items:
        atoms, settings = _atoms(item["structure"]), item.get("settings")
        if rung == "fc2_descriptors":
            res = ladder.run_descriptors(atoms, calculator(), relax_params, {**kappa_params, **rung_params}, settings)
        elif rung == "lbte":
            cached = ladder.fc_cache_complete(item.get("fc_dir"), settings)
            res = ladder.run_lbte(atoms, None if cached else calculator(), relax_params, kappa_params, rung_params,
                                  settings, item.get("fc_dir"))
        else:
            res = ladder.run_fc4(atoms, calculator(), relax_params, kappa_params, rung_params, settings,
                                 item.get("work_dir") or f"mga_fc4_{item['sid']}")
        res["sid"] = item["sid"]
        results.append(res)
    return {"results": results, "calculator": calc_info, "worker_info": worker_info(), "rung": rung,
            "wall_time_s": round(time.perf_counter() - t0, 1)}


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_relax_flow(model: dict[str, Any], items: list[dict[str, Any]], relax_params: dict[str, Any], name: str) -> Flow:
    jobs = []
    for i, chunk in enumerate(chunked(items, relax_params["chunk_size"])):
        j = mlip_relax_chunk(model, chunk, relax_params)
        j.name = f"{name}-relax-{model['name']}-{i:03d}"
        j.metadata = {"mga_sids": [it["sid"] for it in chunk], "mga_model": model["name"], "mga_kind": "mlip_relax"}
        jobs.append(j)
    return Flow(jobs, name=f"{name}-relax-{model['name']}")


def build_kappa_flow(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    relax_params: dict[str, Any],
    kappa_params: dict[str, Any],
    name: str,
) -> Flow:
    jobs = []
    for i, chunk in enumerate(chunked(items, kappa_params["chunk_size"])):
        j = mlip_kappa_chunk(model, chunk, relax_params, kappa_params)
        j.name = f"{name}-kappa-{model['name']}-{i:03d}"
        j.metadata = {"mga_sids": [it["sid"] for it in chunk], "mga_model": model["name"], "mga_kind": "mlip_kappa"}
        jobs.append(j)
    return Flow(jobs, name=f"{name}-kappa-{model['name']}")


def build_ladder_flow(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    rung: str,
    relax_params: dict[str, Any],
    kappa_params: dict[str, Any],
    rung_params: dict[str, Any],
    name: str,
    chunk_size: int,
) -> Flow:
    jobs = []
    for i, chunk in enumerate(chunked(items, chunk_size)):
        j = mlip_ladder_chunk(model, chunk, rung, relax_params, kappa_params, rung_params)
        j.name = f"{name}-{rung}-{model['name']}-{i:03d}"
        j.metadata = {"mga_sids": [it["sid"] for it in chunk], "mga_model": model["name"], "mga_kind": LADDER_RUNGS[rung]}
        jobs.append(j)
    return Flow(jobs, name=f"{name}-{rung}-{model['name']}")
