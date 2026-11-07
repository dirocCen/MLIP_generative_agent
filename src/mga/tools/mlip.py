"""MLIP and MatterGen tools (submission to datalab through jobflow-remote).

Mode handling is the same for every submit tool (plan §3.1):
- ``dry_run``: build and validate the flow, record it, submit nothing;
- ``live``: skip structures with a cached result, submit the rest, write the
  submission map ``runs/<run>/submissions/<step_id>.json``;
- ``replay``: submit nothing, require every result to be cached.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from mga import jobs, registry, results
from mga.config import get_mode, get_run_id, worker_storage
from mga.models import ModelSpec, all_models, get_model
from mga.params import ParamSpec, describe_spec, resolve_params, result_hash
from mga.records import StepRecord, recorded_step
from mga.specs import (
    VASP_CALC_SPECS,
    ConvergenceTarget,
    GpuResources,
    MatterGenParams,
    MLIPFc2DescriptorParams,
    MLIPFc4Params,
    MLIPLbteParams,
    MLIPPhononKappaParams,
    MLIPRelaxParams,
    StructureMatcherParams,
    VaspResources,
)
from mga.tools._common import ToolInputError, parent_steps, require_decision
from mga.utils import short_hash

SPECS: dict[str, type[ParamSpec]] = {
    "mlip_relax": MLIPRelaxParams,
    "mlip_kappa": MLIPPhononKappaParams,
    "mlip_fc2_descriptors": MLIPFc2DescriptorParams,
    "mlip_lbte": MLIPLbteParams,
    "mlip_fc4": MLIPFc4Params,
    "mattergen": MatterGenParams,
    "structure_matcher": StructureMatcherParams,
    "gpu_resources": GpuResources,
    "vasp_resources": VaspResources,
    **{f"vasp_{k}": v for k, v in VASP_CALC_SPECS.items()},
}
MAX_CONVERGENCE_STRUCTURES = 10
MAX_CONVERGENCE_VALUES = 6


# --- read-only ---------------------------------------------------------------------
def list_models() -> dict[str, Any]:
    return {
        "models": [
            {
                "name": m.name,
                "ready": m.ready,
                "energy_reference": m.energy_reference,
                "committee": m.committee,
                "worker": m.worker,
                "training_data": m.training_data,
                "ksrme_reported": m.ksrme_reported,
                "role": m.role,
                "todo": list(m.todo),
            }
            for m in all_models().values()
        ],
        "note": "models with ready=false still have TODO checkpoints in configs/models.yaml; live submission refuses them",
    }


def describe_parameters(calc: str) -> dict[str, Any]:
    if calc not in SPECS:
        raise ToolInputError(f"Unknown calc {calc!r}; choose from {sorted(SPECS)}")
    return {"calc": calc, "spec": SPECS[calc].__name__, "parameters": describe_spec(SPECS[calc])}


# --- shared submission logic -----------------------------------------------------------
def gpu_qresources(r: GpuResources) -> dict[str, Any]:
    q = {"partition": r.partition, "nodes": 1, "ntasks": 1, "cpus_per_task": r.cpus_per_task, "gres": f"gpu:{r.gpus}", "time": r.time}
    # qtoolkit's slurm header drops unset keys; nodelist / exclude_nodes are its own names
    return q | {k: v for k, v in (("nodelist", r.nodelist), ("exclude_nodes", r.exclude_nodes), ("mem", r.mem)) if v}


def _model(name: str) -> ModelSpec:
    try:
        return get_model(name)
    except KeyError as exc:
        raise ToolInputError(str(exc)) from None


def _split_cached(kind: str, model: str, sids: list[str], phash: str, force: bool) -> tuple[list[str], list[str]]:
    if force:
        return [], list(sids)
    cached = {s for s in sids if results.is_cached(kind, model, s, phash)}
    return [s for s in sids if s in cached], [s for s in sids if s not in cached]


def _flow_summary(flow) -> dict[str, Any]:
    job_list = jobs.flow_jobs(flow)  # nested flows (convergence scans) included
    return {"flow_uuid": flow.uuid, "n_jobs": len(job_list), "jobs": job_list}


def _finish(
    rec: StepRecord,
    *,
    kind: str,
    model: ModelSpec | None,
    worker: str,
    phash: str,
    sids: list[str],
    cached: list[str],
    todo: list[str],
    build_flow,
    qresources: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    mode = get_mode()
    base = {
        "kind": kind,
        "model": model.name if model else None,
        "param_hash": phash,
        "n_structures": len(sids),
        "n_cached": len(cached),
        "n_to_compute": len(todo),
        **extra,
    }
    submission = {"step_id": rec.step_id, "kind": kind, "model": base["model"], "worker": worker, "param_hash": phash,
                  "all_sids": sids, "mode": mode, **extra.get("submission_extra", {})}
    if mode == "replay":
        if todo:
            raise results.ResultMissing(f"replay mode: {len(todo)} results not cached, e.g. {todo[:3]}")
        jobs.save_submission(rec.step_id, rec.run_id, {**submission, "sids": [], "jobs": []})
        rec.status = "replayed"
        rec.outputs = base
        return {"step_id": rec.step_id, "mode": mode, **base, "next": "fetch_results with this step_id (reads the cache)"}

    flow = build_flow(todo) if todo else None
    plan = _flow_summary(flow) if flow else {"n_jobs": 0, "jobs": []}
    if mode == "dry_run":
        rec.status = "dry_run"
        rec.outputs = {**base, "n_jobs": plan["n_jobs"], "worker": worker, "resources": qresources}
        return {"step_id": rec.step_id, "mode": mode, **rec.outputs, "note": "dry_run: nothing was submitted"}

    if model is not None and not model.ready:
        raise ToolInputError(f"Model {model.name} is not ready (TODO in configs/models.yaml): {list(model.todo)}")
    if flow is None:
        jobs.save_submission(rec.step_id, rec.run_id, {**submission, "sids": [], "jobs": []})
        rec.outputs = {**base, "n_jobs": 0, "note": "all results cached; nothing submitted"}
        return {"step_id": rec.step_id, "mode": mode, **rec.outputs, "next": "fetch_results with this step_id (reads the cache)"}
    db_ids = jobs.submit(flow, worker=worker, resources=qresources)
    jobs.save_submission(rec.step_id, rec.run_id, {**submission, "sids": todo, "db_ids": db_ids, **plan})
    rec.status = "submitted"
    rec.outputs = {**base, "n_jobs": plan["n_jobs"], "flow_uuid": plan["flow_uuid"], "worker": worker}
    return {"step_id": rec.step_id, "mode": mode, **rec.outputs, "next": "job_status / fetch_results with this step_id"}


def _items(sids: list[str]) -> list[dict[str, Any]]:
    return [{"sid": s, "structure": registry.load_structure(s).as_dict()} for s in sids]


def _resolve_mlip(m: ModelSpec, spec, override=None, reason=None):
    """resolve_params, then dtype "model_default" -> the model's dtype from configs/models.yaml, so
    the result hash and the record carry the dtype that is actually run."""
    p, prov = resolve_params(spec, override, reason)
    if p.dtype == "model_default":
        p = p.model_copy(update={"dtype": m.dtype})
    return p, prov


def kappa_hash(relax_p: MLIPRelaxParams, kappa_p: MLIPPhononKappaParams) -> str:
    return short_hash([result_hash(relax_p), result_hash(kappa_p)])


def fc_dir(storage: dict[str, str | None], model: str, sid: str, khash: str) -> str | None:
    """Worker-side FC cache of one (model, structure, relax + kappa hash): written by L1, read by L2."""
    root = storage.get("fc_cache")
    return f"{root.rstrip('/')}/{model}/{sid}/{khash}" if root else None


def kappa_items(sids: list[str], kappa_p: MLIPPhononKappaParams) -> list[dict[str, Any]]:
    """Fix supercells and q-mesh per structure before submission (same for every model)."""
    from pymatgen.io.ase import AseAtomsAdaptor

    from mga.workflows.kappa_protocol import length_rule_settings

    p = kappa_p.model_dump()
    items = []
    for sid in sids:
        struct = registry.load_structure(sid)
        ref = registry.load_entry(sid)["info"].get("kappa_settings")
        if ref and kappa_p.use_reference_settings:
            settings = {k: ref[k] for k in ("source", "fc2_supercell", "fc3_supercell", "q_mesh")}
        else:
            unit, settings = length_rule_settings(AseAtomsAdaptor.get_atoms(struct), p)
            settings["source"] = f"{settings['source']}_on_input"
            struct = AseAtomsAdaptor.get_structure(unit)
        n_unit = len(struct)
        settings["n_atoms_fc3_supercell"] = int(round(n_unit * abs(np.linalg.det(settings["fc3_supercell"]))))
        settings["n_atoms_fc2_supercell"] = int(round(n_unit * abs(np.linalg.det(settings["fc2_supercell"]))))
        items.append({"sid": sid, "structure": struct.as_dict(), "settings": settings})
    return items


def _settings_overview(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    fc3 = [it["settings"]["n_atoms_fc3_supercell"] for it in items]
    return {
        "settings_sources": sorted({it["settings"]["source"] for it in items}),
        "fc3_supercell_atoms": {"min": min(fc3), "max": max(fc3)},
        "fc2_supercell_atoms_max": max(it["settings"]["n_atoms_fc2_supercell"] for it in items),
        "examples": [{"sid": it["sid"], **{k: it["settings"][k] for k in ("source", "fc3_supercell", "fc2_supercell", "q_mesh")}} for it in items[:3]],
    }


# --- submit tools ----------------------------------------------------------------------
def submit_mattergen(
    chemical_system: str,
    n_samples: int,
    decided_by: Literal["agent", "human"],
    rationale: str,
    tag: str = "a",
    params_override: dict[str, Any] | None = None,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    from mga.workflows.mattergen_job import mattergen_command, mattergen_generate

    require_decision(decided_by, rationale)
    if not 1 <= n_samples <= 10000:
        raise ToolInputError("n_samples must be between 1 and 10000")
    elements = chemical_system.split("-")
    from pymatgen.core import Element

    for el in elements:
        if not Element.is_valid_symbol(el):
            raise ToolInputError(f"{el!r} in {chemical_system!r} is not an element symbol")
    chemsys = "-".join(sorted(elements))
    p, prov = resolve_params(MatterGenParams, params_override, override_reason)
    r, rprov = resolve_params(GpuResources, resources_override, override_reason)
    key = f"{chemsys}__n{n_samples}__{tag}"
    phash = result_hash(p)
    with recorded_step(
        "submit_mattergen",
        inputs={"chemical_system": chemsys, "n_samples": n_samples, "tag": tag},
        params={**prov, **{f"resources.{k}": v for k, v in rprov.items()}},
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:

        def build(_todo):
            from jobflow import Flow

            j = mattergen_generate(chemsys, n_samples, p.model_dump())
            j.name = f"mattergen-{key}"
            j.metadata = {"mga_sids": [], "mga_kind": "mattergen", "mga_key": key}
            return Flow([j], name=f"mattergen-{key}")

        cached, todo = _split_cached("mattergen", "mattergen", [key], phash, force=False)
        return _finish(
            rec, kind="mattergen", model=None, worker="datalab_mattergen", phash=phash, sids=[key],
            cached=cached, todo=todo, build_flow=build, qresources=gpu_qresources(r),
            extra={"chemical_system": chemsys, "n_samples": n_samples, "key": key,
                   "command_preview": mattergen_command("<run_dir>/mattergen_out", chemsys, n_samples, p.model_dump()),
                   "submission_extra": {"key": key, "chemical_system": chemsys}},
        )


def submit_mlip_relax(
    set_name: str,
    model: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    params_override: dict[str, Any] | None = None,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from mga.workflows.mlip_jobs import build_relax_flow

    require_decision(decided_by, rationale)
    m = _model(model)
    p, prov = _resolve_mlip(m, MLIPRelaxParams, params_override, override_reason)
    r, rprov = resolve_params(GpuResources, resources_override, override_reason)
    phash = result_hash(p)
    sids = registry.set_sids(set_name)
    with recorded_step(
        "submit_mlip_relax",
        inputs={"set_name": set_name, "model": model, "force": force},
        params={**prov, **{f"resources.{k}": v for k, v in rprov.items()}},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        cached, todo = _split_cached("mlip_relax", m.name, sids, phash, force)
        return _finish(
            rec, kind="mlip_relax", model=m, worker=m.worker, phash=phash, sids=sids, cached=cached, todo=todo,
            build_flow=lambda t: build_relax_flow(m.worker_dict(), _items(t), p.model_dump(), set_name),
            qresources=gpu_qresources(r),
            extra={"set_name": set_name, "submission_extra": {"set_name": set_name, "relax_params": p.model_dump()}},
        )


def submit_mlip_phonon_kappa(
    set_name: str,
    model: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    relax_override: dict[str, Any] | None = None,
    kappa_override: dict[str, Any] | None = None,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from mga.workflows.mlip_jobs import build_kappa_flow

    require_decision(decided_by, rationale)
    m = _model(model)
    rp, rprov = _resolve_mlip(m, MLIPRelaxParams, relax_override, override_reason)
    kp, kprov = _resolve_mlip(m, MLIPPhononKappaParams, kappa_override, override_reason)
    res, resprov = resolve_params(GpuResources, resources_override, override_reason)
    phash = kappa_hash(rp, kp)
    sids = registry.set_sids(set_name)
    with recorded_step(
        "submit_mlip_phonon_kappa",
        inputs={"set_name": set_name, "model": model, "force": force},
        params={**{f"relax.{k}": v for k, v in rprov.items()}, **{f"kappa.{k}": v for k, v in kprov.items()},
                **{f"resources.{k}": v for k, v in resprov.items()}},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        cached, todo = _split_cached("mlip_kappa", m.name, sids, phash, force)
        items = kappa_items(todo, kp) if get_mode() != "replay" else []
        storage = worker_storage(m.worker)
        for it in items:  # L2 (submit_mlip_kappa_lbte) reads these back instead of recomputing FC3
            it["fc_dir"] = fc_dir(storage, m.name, it["sid"], phash)
        return _finish(
            rec, kind="mlip_kappa", model=m, worker=m.worker, phash=phash, sids=sids, cached=cached, todo=todo,
            build_flow=lambda _t: build_kappa_flow(m.worker_dict(), items, rp.model_dump(), kp.model_dump(), set_name),
            qresources=gpu_qresources(res),
            extra={"set_name": set_name, "settings": _settings_overview(items),
                   "fc_cache": storage["fc_cache"],
                   "submission_extra": {"set_name": set_name, "relax_params": rp.model_dump(), "kappa_params": kp.model_dump(),
                                        "settings": {it["sid"]: it["settings"] for it in items}}},
        )


# --- multi-fidelity ladder (DESIGN §3.2; workflows/ladder.py) --------------------------------
RUNGS: dict[str, tuple[str, type[ParamSpec]]] = {
    "fc2_descriptors": ("mlip_fc2_descriptors", MLIPFc2DescriptorParams),
    "lbte": ("mlip_lbte", MLIPLbteParams),
    "fc4": ("mlip_fc4", MLIPFc4Params),
}


def rung_hash(relax_p: MLIPRelaxParams, kappa_p: MLIPPhononKappaParams, rung_p: ParamSpec) -> str:
    """Relax + kappa parameters are part of every rung's hash: the rungs reuse the L1 supercells,
    q-mesh, displacement and imaginary-mode gate."""
    return short_hash([result_hash(relax_p), result_hash(kappa_p), result_hash(rung_p)])


def _submit_rung(
    rung: str,
    tool: str,
    set_name: str,
    model: str,
    decided_by: str,
    rationale: str,
    relax_override: dict[str, Any] | None,
    kappa_override: dict[str, Any] | None,
    rung_override: dict[str, Any] | None,
    resources_override: dict[str, Any] | None,
    override_reason: str | None,
    force: bool,
) -> dict[str, Any]:
    from mga.workflows.mlip_jobs import build_ladder_flow

    kind, spec = RUNGS[rung]
    require_decision(decided_by, rationale)
    m = _model(model)
    rp, rprov = _resolve_mlip(m, MLIPRelaxParams, relax_override, override_reason)
    kp, kprov = _resolve_mlip(m, MLIPPhononKappaParams, kappa_override, override_reason)
    lp, lprov = resolve_params(spec, rung_override, override_reason)
    res, resprov = resolve_params(GpuResources, resources_override, override_reason)
    phash, khash = rung_hash(rp, kp, lp), kappa_hash(rp, kp)
    sids = registry.set_sids(set_name)
    storage = worker_storage(m.worker)
    with recorded_step(
        tool,
        inputs={"set_name": set_name, "model": model, "force": force},
        params={**{f"relax.{k}": v for k, v in rprov.items()}, **{f"kappa.{k}": v for k, v in kprov.items()},
                **{f"{rung}.{k}": v for k, v in lprov.items()}, **{f"resources.{k}": v for k, v in resprov.items()}},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        cached, todo = _split_cached(kind, m.name, sids, phash, force)
        items = kappa_items(todo, kp) if get_mode() != "replay" else []
        n_fc_cached = 0
        for it in items:
            if rung == "lbte":
                it["fc_dir"] = fc_dir(storage, m.name, it["sid"], khash)
                # the cache is checked on the worker; here only whether L1 finished for this hash
                n_fc_cached += results.is_cached("mlip_kappa", m.name, it["sid"], khash)
            elif rung == "fc4":
                root = storage["ladder_work"]
                it["work_dir"] = f"{root.rstrip('/')}/fc4/{m.name}/{it['sid']}/{phash}" if root else None
        chunk = lp.chunk_size if rung == "fc4" else kp.chunk_size
        extra: dict[str, Any] = {"set_name": set_name, "rung": rung, "kappa_param_hash": khash,
                                 "settings": _settings_overview(items)}
        if rung == "lbte":
            extra["n_with_l1_result"] = n_fc_cached
            extra["fc_cache"] = storage["fc_cache"]
        if rung == "fc4":
            extra["ladder_work"] = storage["ladder_work"]
            extra["fourphonon_cmd"] = lp.fourphonon_cmd or "$MGA_FOURPHONON on the worker (unset: inputs only)"
        extra["submission_extra"] = {"set_name": set_name, "rung": rung, "relax_params": rp.model_dump(),
                                     "kappa_params": kp.model_dump(), "rung_params": lp.model_dump(),
                                     "kappa_param_hash": khash, "settings": {it["sid"]: it["settings"] for it in items}}
        return _finish(
            rec, kind=kind, model=m, worker=m.worker, phash=phash, sids=sids, cached=cached, todo=todo,
            build_flow=lambda _t: build_ladder_flow(m.worker_dict(), items, rung, rp.model_dump(), kp.model_dump(),
                                                    lp.model_dump(), set_name, chunk),
            qresources=gpu_qresources(res), extra=extra,
        )


def submit_mlip_fc2_descriptors(
    set_name: str,
    model: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    relax_override: dict[str, Any] | None = None,
    kappa_override: dict[str, Any] | None = None,
    descriptor_override: dict[str, Any] | None = None,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """L0: relax -> FC2 (L1 supercell) -> imaginary-mode gate -> harmonic descriptors, weighted
    3-phonon phase space and mode Grueneisen / Slack estimate."""
    return _submit_rung("fc2_descriptors", "submit_mlip_fc2_descriptors", set_name, model, decided_by, rationale,
                        relax_override, kappa_override, descriptor_override, resources_override, override_reason, force)


def submit_mlip_kappa_lbte(
    set_name: str,
    model: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    relax_override: dict[str, Any] | None = None,
    kappa_override: dict[str, Any] | None = None,
    lbte_override: dict[str, Any] | None = None,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """L2: Wigner LBTE on the L1 force constants (same relax + kappa parameters -> same FC cache)."""
    return _submit_rung("lbte", "submit_mlip_kappa_lbte", set_name, model, decided_by, rationale,
                        relax_override, kappa_override, lbte_override, resources_override, override_reason, force)


def submit_mlip_fc4_fourphonon(
    set_name: str,
    model: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    relax_override: dict[str, Any] | None = None,
    kappa_override: dict[str, Any] | None = None,
    fc4_override: dict[str, Any] | None = None,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """L3: FC2/FC3/FC4 by random displacements + symfc, ShengBTE/FourPhonon 3ph and 3+4ph inputs
    (and runs, when a FourPhonon command is configured)."""
    return _submit_rung("fc4", "submit_mlip_fc4_fourphonon", set_name, model, decided_by, rationale,
                        relax_override, kappa_override, fc4_override, resources_override, override_reason, force)


def submit_mlip_convergence(
    set_name: str,
    model: str,
    target: ConvergenceTarget,
    values: list[float],
    decided_by: Literal["agent", "human"],
    rationale: str,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    """Scan one kappa setting on a few structures; each value is an ordinary cached kappa result.

    For the length-rule targets the phononDB settings are switched off
    (``use_reference_settings=False``), otherwise the scan would not change anything.
    """
    from mga.workflows.mlip_jobs import build_kappa_flow

    require_decision(decided_by, rationale)
    m = _model(model)
    sids = registry.set_sids(set_name)
    if len(sids) > MAX_CONVERGENCE_STRUCTURES or not 2 <= len(values) <= MAX_CONVERGENCE_VALUES:
        raise ToolInputError(
            f"convergence scans take <= {MAX_CONVERGENCE_STRUCTURES} structures and 2-{MAX_CONVERGENCE_VALUES} values"
        )
    reason = override_reason or f"convergence scan of {target}"
    rp, rprov = _resolve_mlip(m, MLIPRelaxParams)
    res, resprov = resolve_params(GpuResources, resources_override, override_reason)
    variants = []
    for v in values:
        over = {target: v} if target == "displacement" else {target: v, "use_reference_settings": False}
        kp, _ = _resolve_mlip(m, MLIPPhononKappaParams, over, reason)
        variants.append((v, kp, kappa_hash(rp, kp)))
    with recorded_step(
        "submit_mlip_convergence",
        inputs={"set_name": set_name, "model": model, "target": target, "values": values},
        params={**{f"relax.{k}": v for k, v in rprov.items()}, **{f"resources.{k}": v for k, v in resprov.items()}},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        todo_by_variant = {ph: _split_cached("mlip_kappa", m.name, sids, ph, False)[1] for _, _, ph in variants}
        all_todo = [f"{ph}:{s}" for ph, t in todo_by_variant.items() for s in t]
        cached = [f"{ph}:{s}" for ph in todo_by_variant for s in sids if s not in todo_by_variant[ph]]

        def build(_t):
            from jobflow import Flow

            flows = []
            for v, kp, ph in variants:
                if todo_by_variant[ph]:
                    f = build_kappa_flow(m.worker_dict(), kappa_items(todo_by_variant[ph], kp), rp.model_dump(), kp.model_dump(), f"{set_name}-{target}={v:g}")
                    for j in f.jobs:
                        j.metadata["mga_param_hash"] = ph
                    flows.append(f)
            return Flow(flows, name=f"{set_name}-conv-{target}")

        return _finish(
            rec, kind="mlip_kappa", model=m, worker=m.worker, phash="per-variant", sids=[f"{ph}:{s}" for _, _, ph in variants for s in sids],
            cached=cached, todo=all_todo, build_flow=build, qresources=gpu_qresources(res),
            extra={"set_name": set_name, "target": target,
                   "variants": [{"value": v, "param_hash": ph} for v, _, ph in variants],
                   "submission_extra": {"set_name": set_name, "target": target, "relax_params": rp.model_dump(),
                                        "variants": [{"value": v, "param_hash": ph, "kappa_params": kp.model_dump()} for v, kp, ph in variants]}},
        )

