"""Analysis tools: job status, fetching results, E_hull, kappa_SRME, model disagreement.

``fetch_results`` has two halves:
- *ingest* (live only): pull finished job outputs from jobflow-remote into ``data/results/``;
- *derive* (live and replay): from the cache only, register child structures and
  create the follow-up sets. Replay therefore runs the same derive step without a cluster.

Everything below the ingest reads only the result cache.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Literal

import numpy as np
from pymatgen.core import Lattice, Structure

from mga import jobs, registry, results
from mga.analysis.kappa_metrics import KAPPA_ERROR_MAX, compare_models, compare_to_reference
from mga.config import get_mode
from mga.models import get_model
from mga.records import iter_records, recorded_step
from mga.tools._common import N_EXAMPLES, ToolInputError, require_decision, top, write_details

LADDER_KINDS = ("mlip_fc2_descriptors", "mlip_lbte", "mlip_fc4")
MLIP_KINDS = ("mlip_relax", "mlip_kappa", *LADDER_KINDS)
VASP_KINDS = ("vasp_relax_static", "vasp_phonon_fd", "vasp_phono3py_fd", "vasp_convergence")
EHULL_BINS = (-np.inf, 0.0, 0.025, 0.05, 0.1, 0.2, np.inf)  # eV/atom
CONV_TOL = 0.05  # heuristic: relative change of kappa between the two largest settings
EV_A3_PER_KBAR = 1e-1 / 160.21766208  # kBar -> GPa -> eV/A^3


def _submission(step_id: str) -> dict[str, Any]:
    try:
        return jobs.load_submission(step_id)
    except FileNotFoundError:
        raise ToolInputError(
            f"No submission map for step {step_id!r}. dry_run steps submit nothing; "
            "pass the step_id returned by a live or replay submit tool."
        ) from None


def _job_uuids(sub: dict[str, Any]) -> list[str]:
    if sub["kind"] in VASP_KINDS:
        return [j["uuid"] for it in sub.get("items", []) for j in it["jobs"]]
    return [j["uuid"] for j in sub.get("jobs", [])]


def _states(sub: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if sub["kind"] in VASP_KINDS:
        return jobs.flow_states([it["flow_uuid"] for it in sub.get("items", [])])
    uuids = _job_uuids(sub)
    return jobs.job_states(uuids) if uuids else {}


def _state_summary(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = top((s["state"] for s in states.values()), 20)
    bad = [s for s in states.values() if s["state"] in jobs.FINAL_BAD]
    final = all(s["state"] == jobs.DONE or s["state"] in jobs.FINAL_BAD for s in states.values())
    return {
        "n_jobs": len(states),
        "states": counts,
        "all_final": final,
        "n_completed": sum(s["state"] == jobs.DONE for s in states.values()),
        "n_failed": len(bad),
        "failed_examples": [{"name": s["name"], "state": s["state"], "error": (s["error"] or "")[-300:], "run_dir": s["run_dir"]}
                            for s in bad[:N_EXAMPLES]],
    }


# --- job status --------------------------------------------------------------------
def job_status(step_id: str) -> dict[str, Any]:
    sub = _submission(step_id)
    if sub.get("mode") == "replay" or not _job_uuids(sub):
        return {"step_id": step_id, "kind": sub["kind"], "n_jobs": 0,
                "note": "nothing was submitted for this step (replay or fully cached); call fetch_results"}
    out = {"step_id": step_id, "kind": sub["kind"], "model": sub.get("model"), **_state_summary(_states(sub))}
    if sub["kind"] in VASP_KINDS:
        out["note"] = "VASP phonon flows add their displacement jobs at run time, so n_jobs grows while they run"
    return out


# --- ingest --------------------------------------------------------------------------
def _plain(obj: Any) -> Any:
    """Job output (pydantic doc, pymatgen objects, numpy) -> plain JSON types."""
    from monty.json import jsanitize

    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return jsanitize(obj, strict=True, enum_values=True)


def _ingest_mlip(sub: dict[str, Any], states: dict[str, dict[str, Any]], step_id: str, force: bool) -> dict[str, int]:
    n_new = n_skip = 0
    for job in sub.get("jobs", []):
        if states.get(job["uuid"], {}).get("state") != jobs.DONE:
            continue
        phash = job.get("param_hash") or sub["param_hash"]
        if not force and all(results.is_cached(sub["kind"], sub["model"], s, phash) for s in job.get("sids", [])):
            n_skip += len(job.get("sids", []))
            continue
        out = jobs.job_output(job["uuid"])
        prov = {"submit_step": sub["step_id"], "fetch_step": step_id, "job_uuid": job["uuid"],
                "calculator": out.get("calculator"), "worker_info": out.get("worker_info"),
                "job_wall_time_s": out.get("wall_time_s")}
        for res in out["results"]:
            if not force and not res.get("ok", True) and results.is_cached(sub["kind"], sub["model"], res["sid"], phash):
                n_skip += 1  # never replace a stored result with a failure (e.g. an old job fetched again)
                continue
            arrays = {k: np.asarray(v) for k, v in (res.pop("arrays", None) or {}).items()}
            results.save_result(sub["kind"], sub["model"], res["sid"], phash, res, arrays, provenance=prov)
            n_new += 1
    return {"n_saved": n_new, "n_already_cached": n_skip}


def _ingest_mattergen(sub: dict[str, Any], states: dict[str, dict[str, Any]], step_id: str, force: bool) -> dict[str, int]:
    key, phash = sub["key"], sub["param_hash"]
    if results.is_cached("mattergen", "mattergen", key, phash) and not force:
        return {"n_saved": 0, "n_already_cached": 1}
    done = [j for j in sub.get("jobs", []) if states.get(j["uuid"], {}).get("state") == jobs.DONE]
    if not done:
        return {"n_saved": 0, "n_already_cached": 0}
    out = jobs.job_output(done[0]["uuid"])
    results.save_result("mattergen", "mattergen", key, phash, out,
                        provenance={"submit_step": sub["step_id"], "fetch_step": step_id, "job_uuid": done[0]["uuid"]})
    return {"n_saved": 1, "n_already_cached": 0}


def _frame(doc: dict[str, Any], name: str, uuid: str):
    """One VASP calculation -> ase.Atoms with energy / forces / stress (ASE sign convention)."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    out = doc["output"]
    atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(out["structure"]))
    stress = None
    if out.get("stress") is not None:  # VASP kBar, positive = compressive -> ASE eV/A^3, positive = tensile
        s = -np.asarray(out["stress"], float) * EV_A3_PER_KBAR
        stress = s[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    atoms.calc = SinglePointCalculator(atoms, energy=out["energy"], forces=np.asarray(out["forces"]), stress=stress)
    atoms.info.update({"config_type": name, "job_uuid": uuid, "dir_name": doc.get("dir_name"),
                       "calc_type": str(doc.get("calc_type")), "vasp_version": doc.get("vasp_version")})
    return atoms


def _vasp_summary_doc(doc: dict[str, Any]) -> dict[str, Any]:
    inp = doc.get("input") or {}
    out = doc.get("output") or {}
    return {
        "energy": out.get("energy"), "energy_per_atom": out.get("energy_per_atom"), "bandgap": out.get("bandgap"),
        "structure": out.get("structure"), "run_type": doc.get("run_type"), "calc_type": doc.get("calc_type"),
        "task_type": doc.get("task_type"), "vasp_version": doc.get("vasp_version"), "dir_name": doc.get("dir_name"),
        "incar": inp.get("incar"), "potcar_spec": inp.get("potcar_spec"), "nkpoints": inp.get("nkpoints"),
        "entry": doc.get("entry"),
    }


def _ingest_vasp(sub: dict[str, Any], step_id: str, force: bool) -> dict[str, Any]:
    from ase.io import write

    kind, phash = sub["kind"], sub["param_hash"]
    n_new = n_skip = 0
    pending, failed = [], []
    for it in sub.get("items", []):
        sid = it["sid"]
        if results.is_cached(kind, "vasp", sid, phash) and not force:
            n_skip += 1
            continue
        states = jobs.flow_states([it["flow_uuid"]])
        st = _state_summary(states)
        if st["n_failed"]:
            failed.append({"sid": sid, **{k: st[k] for k in ("states", "failed_examples")}})
            continue
        if not st["all_final"]:
            pending.append(sid)
            continue
        frames, points = [], []
        for uuid, info in sorted(states.items(), key=lambda kv: kv[1]["end_time"] or ""):
            doc = _plain(jobs.job_output(uuid))
            if isinstance(doc, dict) and isinstance(doc.get("output"), dict) and doc["output"].get("forces") is not None:
                frames.append(_frame(doc, info["name"], uuid))
                points.append({"job": info["name"], "uuid": uuid, **_vasp_summary_doc(doc)})
        data: dict[str, Any] = {"n_frames": len(frames), "calculations": [{k: p[k] for k in ("job", "uuid", "energy_per_atom", "dir_name")} for p in points]}
        arrays: dict[str, np.ndarray] = {}
        if kind == "vasp_relax_static":
            data["final"] = _vasp_summary_doc(_plain(jobs.job_output(it["output_uuid"])))
        elif kind == "vasp_phonon_fd":
            ph = _plain(jobs.job_output(it["output_uuid"]))
            fc = ph.get("force_constants")
            fc = fc.get("force_constants") if isinstance(fc, dict) else fc
            if fc is not None:
                arrays["force_constants"] = np.asarray(fc, float)
            data["phonon"] = {k: ph.get(k) for k in ("has_imaginary_modes", "supercell_matrix", "primitive_matrix",
                                                     "total_dft_energy", "structure", "phonopy_settings", "jobdirs", "uuids")}
        elif kind == "vasp_phono3py_fd":
            ph = _plain(jobs.job_output(it["output_uuid"]))
            arrays.update({k: np.asarray(v) for k, v in (ph.pop("arrays", None) or {}).items()})
            data["phono3py"] = ph
        else:
            data["points"] = points
        results.save_result(kind, "vasp", sid, phash, data, arrays, provenance={
            "submit_step": sub["step_id"], "fetch_step": step_id, "approval_id": sub.get("approval_id"),
            "flow_uuid": it["flow_uuid"], "params": sub.get("params")})
        if frames:
            write(results.frames_path(kind, "vasp", sid, phash), frames, format="extxyz")
        n_new += 1
    return {"n_saved": n_new, "n_already_cached": n_skip, "n_pending": len(pending), "n_failed_flows": len(failed),
            "failed_examples": failed[:N_EXAMPLES]}


# --- derive (cache only) ---------------------------------------------------------------
def _structure_from_result(rs: dict[str, Any]) -> Structure:
    return Structure(Lattice(rs["cell"]), rs["numbers"], rs["positions"], coords_are_cartesian=True)


def _final_set(name: str, sids: list[str], n_missing: int, allow_partial: bool, **kw) -> str | None:
    if n_missing and not allow_partial:
        return None
    full = name if not n_missing else f"{name}.partial{len(sids)}"
    registry.create_set(full, sids, **kw)
    return full


def _derive_relax(sub: dict[str, Any], step_id: str, allow_partial: bool) -> dict[str, Any]:
    model, phash = sub["model"], sub["param_hash"]
    children, failures, missing, unconverged = [], {}, [], []
    for sid in sub["all_sids"]:
        if not results.has_result("mlip_relax", model, sid, phash):
            missing.append(sid)
            continue
        data = results.load_result("mlip_relax", model, sid, phash)["data"]
        if not data.get("ok"):
            failures[sid] = data.get("error")
            continue
        struct = _structure_from_result(data["relaxed_structure"])
        # a relaxed cell that rounds to its own input gets the same ID; it must not become its own parent
        child, _ = registry.register_structure(
            struct,
            origin={"kind": "mlip_relax", "source_sid": sid, "model": model, "param_hash": phash},
            parents=[] if registry.structure_id(struct) == sid else [sid],
            step_id=step_id,
        )
        children.append(child)
        if (data.get("relax") or {}).get("reached_max_steps"):
            unconverged.append(child)
    set_name = sub.get("set_name") or "set"
    created = _final_set(
        f"{set_name}.relaxed.{model}.{phash[:6]}", children, len(missing), allow_partial,
        operation="mlip_relax", parent_sets=[set_name] if sub.get("set_name") else [],
        description=f"{model}-relaxed structures (kappa_SRME relaxation protocol), failures excluded",
        details={"model": model, "param_hash": phash, "n_failed": len(failures), "n_missing": len(missing)},
        step_id=step_id,
    )
    # generated structures can hit max_steps far from a minimum (S1 trial 11-07: 10/46 still at 5-10 GPa
    # after 1000 steps); phonons of those mean nothing, so the next rungs take the converged subset
    converged = None
    if created:
        converged = _final_set(
            f"{created}.converged", [c for c in children if c not in unconverged], 0, allow_partial,
            operation="filter", parent_sets=[created],
            description="relaxations that converged (did not reach max_steps)",
            details={"criterion": "relax.reached_max_steps is false", "unconverged": unconverged}, step_id=step_id,
        )
    return {"relaxed_set": created, "converged_set": converged, "n_unconverged": len(unconverged),
            "n_relaxed": len(children), "n_failed": len(failures), "n_missing": len(missing),
            "failure_reasons": top((str(e).split(":")[0] for e in failures.values()), 10),
            "note": None if created else "some results are missing; the relaxed set is created once all jobs have finished "
            "(or pass allow_partial=true)"}


def _derive_mattergen(sub: dict[str, Any], step_id: str) -> dict[str, Any]:
    key, phash = sub["key"], sub["param_hash"]
    if not results.has_result("mattergen", "mattergen", key, phash):
        return {"generated_set": None, "note": "MatterGen output not fetched yet"}
    data = results.load_result("mattergen", "mattergen", key, phash)["data"]
    sids = []
    for i, d in enumerate(data["structures"]):
        sid, _ = registry.register_structure(
            Structure.from_dict(d),
            origin={"kind": "mattergen", "chemical_system": data["chemical_system"], "key": key, "param_hash": phash, "index": i},
            step_id=step_id,
        )
        sids.append(sid)
    name = f"mattergen.{key}"
    doc = registry.create_set(name, sids, operation="mattergen", description=f"MatterGen samples for {data['chemical_system']}",
                              details={"n_returned": data["n_returned"], "n_requested": data["n_requested"],
                                       "param_hash": phash}, step_id=step_id)
    return {"generated_set": name, "n_returned": data["n_returned"], "n_unique_ids": doc["n"],
            "next": "filter_structures / deduplicate / check_benchmark_leakage on this set"}


def _derive_kappa(sub: dict[str, Any]) -> dict[str, Any]:
    if sub.get("variants"):
        return {"note": "convergence scan: use kappa_convergence with this step_id"}
    model, phash = sub["model"], sub["param_hash"]
    rows = [results.load_result("mlip_kappa", model, s, phash)["data"] for s in sub["all_sids"]
            if results.has_result("mlip_kappa", model, s, phash)]
    kappas = [r["kappa_tot_avg"][0] for r in rows if r.get("kappa_tot_avg")]
    return {
        "n_results": len(rows), "n_missing": len(sub["all_sids"]) - len(rows),
        "n_failed": sum(not r.get("ok") for r in rows),
        "n_imaginary": sum(bool(r.get("has_imaginary_modes")) for r in rows),
        "skipped": top((r["conductivity_skipped"] for r in rows if r.get("conductivity_skipped")), 5),
        "n_with_kappa": len(kappas),
        "kappa_300K_W_mK": {"min": min(kappas), "median": float(np.median(kappas)), "max": max(kappas)} if kappas else None,
    }


def _stats(xs: list[float]) -> dict[str, float] | None:
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return {"n": len(xs), "min": min(xs), "median": float(np.median(xs)), "max": max(xs)} if xs else None


def _first(v):
    return v[0] if isinstance(v, list) and v else v


def _derive_ladder(sub: dict[str, Any]) -> dict[str, Any]:
    kind, model, phash = sub["kind"], sub["model"], sub["param_hash"]
    rows = [results.load_result(kind, model, s, phash)["data"] for s in sub["all_sids"]
            if results.has_result(kind, model, s, phash)]
    out: dict[str, Any] = {
        "n_results": len(rows), "n_missing": len(sub["all_sids"]) - len(rows),
        "n_failed": sum(not r.get("ok") for r in rows),
        "failed_examples": [{"sid": r["sid"], "stage": r.get("stage"), "error": (r.get("error") or "")[-300:]}
                            for r in rows if not r.get("ok")][:N_EXAMPLES],
        "wall_time_s": _stats([r.get("wall_time_s") for r in rows]),
    }
    if kind == "mlip_fc2_descriptors":
        d = [r["descriptors"] for r in rows if r.get("descriptors")]
        out["n_imaginary"] = sum(bool(r.get("has_imaginary_modes")) for r in rows)
        out["descriptors"] = {k: _stats([x.get(k) for x in d]) for k in (
            "kappa_slack_W_mK", "kappa_proxy_cv2_over_wjdos", "kappa_ctau_1ps_W_mK", "theta_acoustic_K",
            "ao_gap_ratio", "acoustic_branch_spread", "gamma_acoustic_rms", "wjdos_cv_mean")}
    elif kind == "mlip_lbte":
        out["n_from_fc_cache"] = sum(r.get("fc_source") == "cache" for r in rows)
        out["n_mesh_reduced"] = sum(bool((r.get("lbte_mesh") or {}).get("reduced")) for r in rows)
        out["skipped"] = top((r["conductivity_skipped"] for r in rows if r.get("conductivity_skipped")), 5)
        out["kappa_lbte_300K_W_mK"] = _stats([_first(r.get("kappa_tot_exact_avg")) for r in rows])
        out["lbte_over_rta"] = _stats([_first(r.get("lbte_over_rta")) for r in rows])
    else:
        out["kappa_fit_phono3py_W_mK"] = _stats([_first(r.get("kappa_tot_fit_avg")) for r in rows])
        out["four_phonon_ratio"] = _stats([r.get("four_phonon_ratio") for r in rows])
        out["n_fourphonon_run"] = sum(r.get("four_phonon_ratio") is not None for r in rows)
        out["roundtrip_max_rel_dev"] = max((max((r.get("shengbte_files") or {}).get("roundtrip_max_rel_dev", {}).values(), default=0.0)
                                            for r in rows), default=None)
        by_order = [(r.get("fit") or {}).get("force_rmse_test_by_order") or {} for r in rows]
        out["held_out_force_rmse_fc234"] = _stats([b.get("4", b.get(4)) for b in by_order])
    return out


def _derive_vasp_relax(sub: dict[str, Any], step_id: str, allow_partial: bool) -> dict[str, Any]:
    phash = sub["param_hash"]
    children, missing = [], []
    for sid in sub["sids"]:
        if not results.has_result("vasp_relax_static", "vasp", sid, phash):
            missing.append(sid)
            continue
        final = results.load_result("vasp_relax_static", "vasp", sid, phash)["data"]["final"]
        struct = Structure.from_dict(final["structure"])
        child, _ = registry.register_structure(
            struct, origin={"kind": "vasp_relax_static", "source_sid": sid, "param_hash": phash},
            parents=[] if registry.structure_id(struct) == sid else [sid], step_id=step_id)
        children.append(child)
    set_name = sub.get("set_name") or "set"
    created = _final_set(f"{set_name}.vasp_relaxed.{phash[:6]}", children, len(missing), allow_partial,
                         operation="vasp_relax_static", description="VASP (MPRelaxSet + PBE_54) relaxed structures",
                         details={"param_hash": phash, "n_missing": len(missing)}, step_id=step_id)
    return {"relaxed_set": created, "n_relaxed": len(children), "n_missing": len(missing)}


def fetch_results(step_id: str, allow_partial: bool = False, force: bool = False) -> dict[str, Any]:
    sub = _submission(step_id)
    kind, mode = sub["kind"], get_mode()
    with recorded_step("fetch_results", inputs={"step_id": step_id, "kind": kind, "allow_partial": allow_partial, "force": force},
                       parent_ids=[step_id]) as rec:
        ingest: dict[str, Any] = {"note": "replay: read from the result cache only"}
        if mode == "live" and sub.get("mode") != "replay":
            if kind in VASP_KINDS:
                ingest = _ingest_vasp(sub, rec.step_id, force)
            elif _job_uuids(sub):
                states = _states(sub)
                ingest = {**(_ingest_mattergen if kind == "mattergen" else _ingest_mlip)(sub, states, rec.step_id, force),
                          "jobs": _state_summary(states)}
            else:
                ingest = {"note": "nothing was submitted (all cached)"}
        elif mode == "dry_run":
            ingest = {"note": "dry_run: nothing fetched from the cluster; derived sets come from the cache only"}
        if kind == "mlip_relax":
            derived = _derive_relax(sub, rec.step_id, allow_partial)
        elif kind == "mattergen":
            derived = _derive_mattergen(sub, rec.step_id)
        elif kind == "mlip_kappa":
            derived = _derive_kappa(sub)
        elif kind in LADDER_KINDS:
            derived = _derive_ladder(sub)
        elif kind == "vasp_relax_static":
            derived = _derive_vasp_relax(sub, rec.step_id, allow_partial)
        else:
            derived = {"n_results": sum(results.has_result(kind, "vasp", s, sub["param_hash"]) for s in sub["sids"]),
                       "frames": "DFT frames (energy/forces/stress) are in data/results/<kind>/vasp/<sid>__<hash>.frames.extxyz"}
        rec.outputs = {"kind": kind, "model": sub.get("model"), "ingest": ingest, "derived": derived}
    return {"step_id": rec.step_id, **rec.outputs}


# --- E_hull -----------------------------------------------------------------------------
def _relax_sub(step_id: str) -> dict[str, Any]:
    sub = _submission(step_id)
    if sub["kind"] != "mlip_relax":
        raise ToolInputError(f"step {step_id} is a {sub['kind']} submission; compute_ehull needs a submit_mlip_relax step")
    return sub


def compute_ehull(
    relax_step_id: str,
    threshold_ev_atom: float = 0.1,
    out_set: str | None = None,
    decided_by: Literal["agent", "human"] | None = None,
    rationale: str | None = None,
    refresh_mp: bool = False,
) -> dict[str, Any]:
    """E_hull of MLIP-relaxed structures vs the MP GGA/GGA+U hull with MP2020 corrections.

    VASP results are not accepted here: they use PBE_54 POTCARs, MP's hull uses the
    legacy PBE set, so a consistent DFT hull would need our own competing phases.
    """
    from mga.analysis.ehull import e_above_hull

    sub = _relax_sub(relax_step_id)
    model = get_model(sub["model"])
    if model.energy_reference != "mp_pbe":
        raise ToolInputError(f"{model.name} has energy_reference={model.energy_reference}; E_hull vs MP is not meaningful")
    if not 0 <= threshold_ev_atom <= 1:
        raise ToolInputError("threshold_ev_atom must be within 0-1 eV/atom")
    if out_set:
        require_decision(decided_by or "", rationale or "")
    phash = sub["param_hash"]
    with recorded_step("compute_ehull", inputs={"relax_step_id": relax_step_id, "threshold_ev_atom": threshold_ev_atom,
                                                "out_set": out_set}, parent_ids=[relax_step_id],
                       decided_by=decided_by, rationale=rationale) as rec:
        items, skipped = [], {}
        for sid in sub["all_sids"]:
            if not results.has_result("mlip_relax", model.name, sid, phash):
                skipped[sid] = "no result"
                continue
            data = results.load_result("mlip_relax", model.name, sid, phash)["data"]
            if not data.get("ok"):
                skipped[sid] = "relaxation failed"
                continue
            items.append((sid, _structure_from_result(data["relaxed_structure"]), float(data["relax"]["energy"])))
        per, mp_meta = e_above_hull(items, refresh_mp=refresh_mp)
        values = np.array([v["e_above_hull"] for v in per.values() if v.get("e_above_hull") is not None])
        hist = np.histogram(values, bins=EHULL_BINS)[0] if len(values) else np.zeros(len(EHULL_BINS) - 1, int)
        labels = ["<0", "0-0.025", "0.025-0.05", "0.05-0.1", "0.1-0.2", ">0.2"]
        passing = [s for s, v in per.items() if v.get("e_above_hull") is not None and v["e_above_hull"] <= threshold_ev_atom]
        if out_set:
            registry.create_set(out_set, passing, operation="ehull_filter", parent_sets=[sub["set_name"]] if sub.get("set_name") else [],
                                description=f"E_hull <= {threshold_ev_atom} eV/atom ({model.name} relaxed, MP2020-corrected MP hull)",
                                details={"relax_step_id": relax_step_id, "threshold_ev_atom": threshold_ev_atom}, step_id=rec.step_id)
        rec.outputs = {
            "model": model.name, "n_structures": len(sub["all_sids"]), "n_with_ehull": int(len(values)),
            "n_skipped": len(skipped), "n_ehull_failed": sum(v.get("e_above_hull") is None for v in per.values()),
            "histogram_ev_atom": dict(zip(labels, hist.tolist())),
            "n_below_threshold": len(passing), "threshold_ev_atom": threshold_ev_atom, "out_set": out_set,
            "most_stable": sorted(({"sid": s, "e_above_hull": v["e_above_hull"]} for s, v in per.items() if v.get("e_above_hull") is not None),
                                  key=lambda r: r["e_above_hull"])[:N_EXAMPLES],
            "mp": mp_meta,
            "note": "E_hull < 0 means the MLIP puts the structure below the MP hull: check for MLIP over-stabilisation before "
                    "calling it a discovery. Input sids (not the relaxed children) are put in out_set.",
        }
        rec.outputs["details_file"] = write_details(rec.step_id, {"per_structure": per, "skipped": skipped})
    return {"step_id": rec.step_id, **rec.outputs}


# --- kappa vs reference ------------------------------------------------------------------
def _kappa_sub(step_id: str) -> dict[str, Any]:
    sub = _submission(step_id)
    if sub["kind"] != "mlip_kappa":
        raise ToolInputError(f"step {step_id} is a {sub['kind']} submission, not submit_mlip_phonon_kappa")
    if sub.get("variants"):
        raise ToolInputError(f"step {step_id} is a convergence scan; use kappa_convergence")
    return sub


def kappa_srme(kappa_step_id: str) -> dict[str, Any]:
    """Port of the matbench-discovery kappa_SRME scoring against phononDB PBE (noNAC)."""
    sub = _kappa_sub(kappa_step_id)
    model, phash = sub["model"], sub["param_hash"]
    with recorded_step("kappa_srme", inputs={"kappa_step_id": kappa_step_id}, parent_ids=[kappa_step_id]) as rec:
        rows, missing, not_bench = [], [], 0
        for sid in sub["all_sids"]:
            mp_id = registry.load_entry(sid)["info"].get("mp_id")
            if not mp_id:
                not_bench += 1
                continue
            if not results.has_result("mlip_kappa", model, sid, phash):
                missing.append({"sid": sid, "mp_id": mp_id})
                continue
            doc = results.load_result("mlip_kappa", model, sid, phash, with_arrays=True)
            row = compare_to_reference(doc["data"], doc.get("array_data", {}), mp_id)
            row["sid"] = sid
            row["settings_source"] = (doc["data"].get("settings") or {}).get("source")
            rows.append(row)
        if not rows and not missing:
            raise ToolInputError("no kappa_SRME benchmark structures (info.mp_id) in this submission")
        srme = np.array([r["srme"] for r in rows])
        sre = np.array([r["sre"] for r in rows])
        n_all = len(rows) + len(missing)
        rec.outputs = {
            "model": model, "param_hash": phash,
            "n_benchmark": n_all, "n_scored": len(rows), "n_missing": len(missing), "n_not_benchmark": not_bench,
            "srme_mean": float(srme.mean()) if len(rows) else None,
            "sre_mean": float(sre.mean()) if len(rows) else None,
            "srme_mean_missing_as_2": float((srme.sum() + KAPPA_ERROR_MAX * len(missing)) / n_all),
            "censored": top((r["censored"].split(":")[0] for r in rows if r.get("censored")), 10),
            "non_reference_settings": sum(r["settings_source"] != "phonondb_pbe" for r in rows),
            "worst": sorted(({k: r[k] for k in ("name", "mp_id", "kappa_ref", "kappa_pred", "srme", "censored")} for r in rows),
                            key=lambda r: -r["srme"])[:N_EXAMPLES],
            "note": ("Official kappa_SRME averages over all 103 entries with failures scored 2; srme_mean_missing_as_2 is the "
                     "comparable number when jobs are missing. Rows whose settings are not phonondb_pbe are not on the reference q-mesh."),
        }
        rec.outputs["details_file"] = write_details(rec.step_id, {"rows": rows, "missing": missing})
    return {"step_id": rec.step_id, **rec.outputs}


# --- model disagreement --------------------------------------------------------------------
def _param_diff(subs: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for a, b in itertools.combinations(subs, 2):
        for block in ("relax_params", "kappa_params"):
            pa, pb = a.get(block) or {}, b.get(block) or {}
            keys |= {f"{block}.{k}" for k in set(pa) | set(pb) if pa.get(k) != pb.get(k) and k != "chunk_size"}
    return sorted(keys)


def model_disagreement(
    kappa_step_ids: list[str],
    top_n: int = 0,
    out_set: str | None = None,
    decided_by: Literal["agent", "human"] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Pairwise kappa disagreement between models on the same structures.

    Mode-resolved SRME needs identical q-points; pairs where they differ are marked
    ``mode_comparable=False`` (total-kappa SRD is still reported), not dropped.
    """
    if not 2 <= len(kappa_step_ids) <= 4:
        raise ToolInputError("pass 2-4 submit_mlip_phonon_kappa step ids (one per model)")
    subs = [_kappa_sub(s) for s in kappa_step_ids]
    models = [s["model"] for s in subs]
    if len(set(models)) != len(models):
        raise ToolInputError(f"each step must be a different model, got {models}")
    if top_n:
        if not out_set:
            raise ToolInputError("top_n needs out_set (the name of the selection set)")
        require_decision(decided_by or "", rationale or "")
    common = [s for s in subs[0]["all_sids"] if all(s in set(x["all_sids"]) for x in subs[1:])]
    with recorded_step("model_disagreement", inputs={"kappa_step_ids": kappa_step_ids, "top_n": top_n, "out_set": out_set},
                       parent_ids=list(kappa_step_ids), decided_by=decided_by, rationale=rationale) as rec:
        per_sid: dict[str, dict[str, Any]] = {}
        pairs: dict[str, list[dict[str, Any]]] = {f"{a}|{b}": [] for a, b in itertools.combinations(models, 2)}
        for sid in common:
            loaded = {}
            for sub in subs:
                if results.has_result("mlip_kappa", sub["model"], sid, sub["param_hash"]):
                    doc = results.load_result("mlip_kappa", sub["model"], sid, sub["param_hash"], with_arrays=True)
                    loaded[sub["model"]] = (doc["data"], doc.get("array_data", {}))
            row: dict[str, Any] = {"kappa": {m: (d.get("kappa_tot_avg") or [None])[0] for m, (d, _) in loaded.items()}, "pairs": {}}
            for (ma, mb) in itertools.combinations(models, 2):
                if ma in loaded and mb in loaded:
                    cmp = compare_models(loaded[ma], loaded[mb])
                    row["pairs"][f"{ma}|{mb}"] = cmp
                    pairs[f"{ma}|{mb}"].append(cmp)
            srds = [abs(c["srd"]) for c in row["pairs"].values() if c.get("srd") is not None]
            row["max_abs_srd"] = max(srds) if srds else None
            ks = [k for k in row["kappa"].values() if k and k > 0]
            row["log10_kappa_std"] = float(np.std(np.log10(ks))) if len(ks) >= 2 else None
            per_sid[sid] = row
        pair_summary = {}
        for name, cmps in pairs.items():
            ok = [c for c in cmps if c.get("srd") is not None]
            modes = [c["srme"] for c in ok if c.get("mode_comparable")]
            pair_summary[name] = {
                "n_compared": len(ok), "n_censored": len(cmps) - len(ok),
                "mean_abs_srd": float(np.mean([abs(c["srd"]) for c in ok])) if ok else None,
                "median_abs_srd": float(np.median([abs(c["srd"]) for c in ok])) if ok else None,
                "mean_srme": float(np.mean(modes)) if modes else None,
                "n_mode_not_comparable": sum(not c.get("mode_comparable") for c in ok),
                "mean_freq_mae_thz": float(np.mean([c["freq_mae_thz"] for c in ok if "freq_mae_thz" in c])) if modes else None,
            }
        ranked = sorted((s for s, r in per_sid.items() if r["max_abs_srd"] is not None), key=lambda s: -per_sid[s]["max_abs_srd"])
        selected = ranked[:top_n] if top_n else []
        if top_n:
            registry.create_set(out_set, selected, operation="model_disagreement_top",
                                description=f"top {top_n} by max pairwise |SRD| of kappa over {models}",
                                details={"kappa_step_ids": kappa_step_ids}, step_id=rec.step_id)
        rec.outputs = {
            "models": models, "n_common_structures": len(common), "param_differences": _param_diff(subs),
            "pairs": pair_summary,
            "most_disagreeing": [{"sid": s, "formula": registry.load_entry(s)["reduced_formula"], "kappa": per_sid[s]["kappa"],
                                  "max_abs_srd": round(per_sid[s]["max_abs_srd"], 3)} for s in ranked[:N_EXAMPLES]],
            "n_censored_in_any_model": sum(r["max_abs_srd"] is None for r in per_sid.values()),
            "out_set": out_set if top_n else None,
            "note": "param_differences lists result-affecting settings that differ between the steps: if not empty, "
                    "disagreement mixes model and parameter effects.",
        }
        rec.outputs["details_file"] = write_details(rec.step_id, per_sid)
    return {"step_id": rec.step_id, **rec.outputs}


# --- convergence ----------------------------------------------------------------------------
def kappa_convergence(conv_step_id: str, tol: float = CONV_TOL) -> dict[str, Any]:
    sub = _submission(conv_step_id)
    if not sub.get("variants"):
        raise ToolInputError(f"step {conv_step_id} is not a submit_mlip_convergence step")
    model, variants = sub["model"], sub["variants"]
    sids = list(dict.fromkeys(s.split(":", 1)[1] for s in sub["all_sids"]))
    with recorded_step("kappa_convergence", inputs={"conv_step_id": conv_step_id, "tol": tol}, parent_ids=[conv_step_id]) as rec:
        table = []
        for sid in sids:
            row: dict[str, Any] = {"sid": sid, "formula": registry.load_entry(sid)["reduced_formula"], "values": []}
            for v in variants:
                ph = v["param_hash"]
                if not results.has_result("mlip_kappa", model, sid, ph):
                    row["values"].append({"value": v["value"], "kappa": None, "status": "missing"})
                    continue
                d = results.load_result("mlip_kappa", model, sid, ph)["data"]
                k = (d.get("kappa_tot_avg") or [None])[0]
                status = "ok" if k is not None else (d.get("conductivity_skipped") or d.get("error") or "no kappa")
                row["values"].append({"value": v["value"], "kappa": k, "status": status,
                                      "fc3_atoms": (d.get("settings") or {}).get("n_atoms_fc3_supercell")})
            ks = [x["kappa"] for x in row["values"]]
            if len(ks) >= 2 and ks[-1] and ks[-2]:
                row["rel_change_last"] = round(abs(ks[-1] - ks[-2]) / ks[-1], 4)
                row["converged"] = row["rel_change_last"] < tol
            else:
                row["rel_change_last"], row["converged"] = None, None
            table.append(row)
        rec.outputs = {
            "model": model, "target": sub["target"], "values": [v["value"] for v in variants], "tol": tol,
            "table": table,
            "n_converged": sum(bool(r["converged"]) for r in table),
            "note": "converged = relative change of kappa between the two largest settings < tol (heuristic 5%). "
                    "One model only; repeat with a second model before changing a default.",
        }
    return {"step_id": rec.step_id, **rec.outputs}


# --- records --------------------------------------------------------------------------------
def query_records(
    run_id: str | None = None,
    tool: str | None = None,
    status: str | None = None,
    contains: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise ToolInputError("limit must be 1-200")
    rows = []
    for rec in iter_records(run_id):
        if tool and rec.tool != tool or status and rec.status != status:
            continue
        if contains and contains not in json.dumps({"i": rec.inputs, "o": rec.outputs}, default=str):
            continue
        rows.append({
            "step_id": rec.step_id, "run_id": rec.run_id, "tool": rec.tool, "status": rec.status, "mode": rec.mode,
            "actor": rec.actor, "decided_by": rec.decided_by, "rationale": (rec.rationale or "")[:200] or None,
            "overrides": sorted(k for k, v in rec.params.items() if v.source == "override"),
            "parent_ids": rec.parent_ids, "error": (rec.error or "")[:200] or None, "created_at": rec.created_at,
        })
    return {"n_matching": len(rows), "records": rows[-limit:], "note": "most recent last; full records are in runs/<run_id>/records/"}


# --- ladder evaluation: budget ledger and escalation baselines (DESIGN §3.2) ----------------------
def _ladder_table(rung_steps: dict[str, list[str]], l0_score: str):
    from mga.analysis import escalation as esc

    if not rung_steps or set(rung_steps) - set(esc.RUNGS):
        raise ToolInputError(f"rung_steps maps rungs {esc.RUNGS} to submit step_ids, got {sorted(rung_steps or {})}")
    table: dict[str, dict[str, Any]] = {}
    per_step = []
    for rung, step_ids in rung_steps.items():
        for step_id in step_ids:
            sub = _submission(step_id)
            if sub["kind"] != esc.RUNG_KINDS[rung]:
                raise ToolInputError(f"step {step_id} is {sub['kind']}; rung {rung} takes {esc.RUNG_KINDS[rung]} submissions")
            if sub.get("variants"):
                raise ToolInputError(f"step {step_id} is a convergence scan, not a ladder rung")
            n, cost = 0, 0.0
            for sid in sub["all_sids"]:
                if results.has_result(sub["kind"], sub["model"], sid, sub["param_hash"]):
                    data = results.load_result(sub["kind"], sub["model"], sid, sub["param_hash"])["data"]
                    esc.add_result(table, rung, sid, data, l0_score)
                    n += 1
                    cost += float(data.get("wall_time_s") or 0.0) / 3600.0
            per_step.append({"rung": rung, "step_id": step_id, "model": sub["model"], "param_hash": sub["param_hash"],
                             "n_structures": len(sub["all_sids"]), "n_results": n, "gpu_h": round(cost, 4),
                             "gpu_h_per_structure": round(cost / n, 5) if n else None})
    return table, per_step


def budget_ledger(rung_steps: dict[str, list[str]], budget_gpu_h: float | None = None) -> dict[str, Any]:
    """What the given ladder submissions cost, per rung and step, in GPU-hours (per-structure wall
    time of one-GPU jobs; calculator loading excluded), against an optional total budget."""
    from mga.analysis import escalation as esc

    table, per_step = _ladder_table(rung_steps, esc.DEFAULT_L0_SCORE)
    by_rung = {}
    for rung in esc.RUNGS:
        cells = [row[rung]["cost_h"] for row in table.values() if rung in row]
        if cells:
            by_rung[rung] = {"n_candidates": len(cells), "gpu_h": round(sum(cells), 4),
                             "gpu_h_per_candidate_median": round(float(np.median(cells)), 5),
                             "gpu_h_per_candidate_max": round(max(cells), 5)}
    total = sum(v["gpu_h"] for v in by_rung.values())
    return {"unit": "GPU-h (one-GPU job, per-structure wall time)", "by_rung": by_rung, "steps": per_step,
            "total_gpu_h": round(total, 4), "budget_gpu_h": budget_gpu_h,
            "remaining_gpu_h": round(budget_gpu_h - total, 4) if budget_gpu_h is not None else None,
            "not_counted": "L4 (DFT) runs on dirona CPU nodes and is not in this unit; queue time; calculator loading"}


def escalation_baseline(
    rung_steps: dict[str, list[str]],
    budget_gpu_h: float,
    hit_threshold_W_mK: float,
    agent_sets: dict[str, str] | None = None,
    ks: list[int] | None = None,
    l0_score: str = "kappa_slack_W_mK",
    n_random: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Replay the fixed rule (top-k by committee mean at every rung) and random selection on the
    cached ladder results under one budget, and score the agent's choices the same way.

    ``rung_steps`` must cover the whole pool at L0 (and L1-L2 for a fair replay); the pool is
    every structure with an L0 result. ``agent_sets`` maps L1/L2/L3 to the sets the agent sent
    to that rung. ``ks`` (k at L1, L2, L3) defaults to the largest common retention fraction
    whose fixed-rule run fits the budget."""
    from mga.analysis import escalation as esc

    if budget_gpu_h <= 0 or not 1 <= n_random <= 10000:
        raise ToolInputError("budget_gpu_h must be > 0 and n_random in 1..10000")
    if ks is not None and (len(ks) != 3 or any(k < 0 for k in ks)):
        raise ToolInputError("ks gives k at L1, L2, L3: three non-negative integers")
    parents = [s for ids in rung_steps.values() for s in ids]
    with recorded_step("escalation_baseline",
                       inputs={"rung_steps": rung_steps, "budget_gpu_h": budget_gpu_h, "hit_threshold_W_mK": hit_threshold_W_mK,
                               "agent_sets": agent_sets, "ks": ks, "l0_score": l0_score, "n_random": n_random, "seed": seed},
                       parent_ids=parents) as rec:
        table, per_step = _ladder_table(rung_steps, l0_score)
        table = {s: row for s, row in table.items() if "L0" in row}
        if not table:
            raise ToolInputError("no L0 results: the pool is every structure with an L0 (fc2_descriptors) result")
        sc = {s: esc.scores(row) for s, row in table.items()}
        f = None
        if ks is None:
            ks, f = esc.ks_for_budget(table, sc, budget_gpu_h, hit_threshold_W_mK)
        fixed = esc.run_policy(table, sc, ks, esc.fixed_order(sc), budget_gpu_h, hit_threshold_W_mK)
        rng = np.random.default_rng(seed)
        rand = [esc.run_policy(table, sc, ks, esc.random_order(rng), budget_gpu_h, hit_threshold_W_mK) for _ in range(n_random)]
        hits_r = np.array([r["hits_found"] for r in rand])
        out: dict[str, Any] = {
            "pool": {"n": len(table), "n_stable": sum(bool(v["stable"]) for v in sc.values()),
                     "n_with_truth": sum(v["truth"] is not None for v in sc.values()),
                     "truth_rung": top((v["truth_rung"] for v in sc.values()), 5),
                     "hits": fixed["hits_in_pool"], "hit_threshold_W_mK": hit_threshold_W_mK},
            "budget_gpu_h": budget_gpu_h, "ks": ks, "retention_fraction": f,
            "median_cost_gpu_h": esc.median_costs(table),
            "fixed_rule": {k: fixed[k] for k in ("n_selected", "hits_found", "recall", "spent_h", "n_uncached_cost")},
            "random": {"n_draws": n_random, "hits_found_mean": float(hits_r.mean()), "hits_found_std": float(hits_r.std()),
                       "recall_mean": float(np.mean([r["recall"] or 0.0 for r in rand])),
                       "spent_h_mean": float(np.mean([r["spent_h"] for r in rand])),
                       "p_random_ge_fixed": float((hits_r >= fixed["hits_found"]).mean())},
            "spearman_between_rungs": esc.spearman_between_rungs(sc),
        }
        if agent_sets:
            chosen = {"L0": list(table), **{r: registry.set_sids(name) for r, name in agent_sets.items()}}
            fb = esc.median_costs(table)
            spent = sum(esc._cost(table, s, r, fb)[0] for r, sids in chosen.items() for s in sids)
            ev = esc.evaluate(chosen, sc, hit_threshold_W_mK)
            out["agent"] = {**{k: ev[k] for k in ("n_selected", "hits_found", "recall")}, "spent_h": spent,
                            "over_budget": spent > budget_gpu_h,
                            "not_in_pool": sorted({s for sids in chosen.values() for s in sids} - set(table))[:N_EXAMPLES]}
        out["note"] = ("truth = estimate from the highest rung in the cache; hits are only counted when a policy took the "
                       "candidate to L3. L4 (DFT) has no counterfactual and is not replayed. If the agent does worse than "
                       "the fixed rule, that is the result (DESIGN §11.4).")
        detail = {"scores": sc, "fixed_chosen": fixed["chosen"], "steps": per_step}
        out["details_file"] = write_details(rec.step_id, detail)
        rec.outputs = out
    return {"step_id": rec.step_id, **rec.outputs}
