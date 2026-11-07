"""VASP tools for dirona, with the human approval gate (DESIGN §11.5, plan §4).

``preview_vasp_inputs`` / ``estimate_vasp_cost`` are local and free.
``request_vasp_submission`` freezes the plan and stops; only a human can approve
it (``mga approvals approve <id>`` in their own terminal). ``submit_vasp`` takes
nothing but an approval id and builds the flows from the frozen plan, so what
runs is exactly what was approved.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pymatgen.core import Structure

from mga import approvals, jobs, registry
from mga.config import get_mode, get_run_id, paths
from mga.params import resolve_params, result_hash
from mga.records import recorded_step
from mga.specs import VASP_CALC_SPECS, VaspResources
from mga.tools._common import ToolInputError, parent_steps, rel, require_decision, top, write_details

Calc = Literal["relax_static", "phonon_fd", "phono3py_fd", "convergence"]
WORKER = "dirona_vasp"
EXEC_CONFIG = "vasp_642_intel"
MAX_PREVIEW = 10
MAX_PER_REQUEST = 200  # larger batches need several approvals on purpose


def _params(calc: str, override: dict[str, Any] | None, reason: str | None):
    if calc not in VASP_CALC_SPECS:
        raise ToolInputError(f"calc must be one of {sorted(VASP_CALC_SPECS)}")
    return resolve_params(VASP_CALC_SPECS[calc], override, reason)


def vasp_qresources(r: VaspResources) -> dict[str, Any]:
    return {"partition": r.partition, "nodes": 1, "ntasks": r.ntasks, "cpus_per_task": 1}


def _exec_incar(r: VaspResources) -> dict[str, Any]:
    return {"NCORE": r.ncore, "KPAR": r.kpar}


def _input_sets(calc: str, p: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(label, our input set, plain MP set) for each distinct calculation type of ``calc``."""
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

    from mga.workflows.vasp_flows import relax_set, static_set

    if calc == "relax_static":
        return [("relax", relax_set(p), MPRelaxSet()), ("static", static_set(p), MPStaticSet())]
    if calc == "phono3py_fd":
        spin = {"ISPIN": p["ispin"]}
        return [("relax", relax_set(p, spin), MPRelaxSet()), ("displacement", static_set(p, spin), MPStaticSet())]
    if calc == "phonon_fd":
        spin = {"ISPIN": p["ispin"]}
        return [
            ("relax", relax_set(p, spin), MPRelaxSet()),
            ("static", static_set(p, spin), MPStaticSet()),
            ("displacement", static_set(p, spin), MPStaticSet()),
        ]
    return [("static", static_set(p), MPStaticSet())]


def _incar_diff(ours: dict[str, Any], mp: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(ours) | set(mp))
    return {k: {"mp": mp.get(k), "ours": ours.get(k)} for k in keys if ours.get(k) != mp.get(k)}


def _warnings(struct: Structure, calc: str, p: dict[str, Any]) -> list[str]:
    out = []
    # PAW cores are closed shells (even electron counts), so the parity of the
    # valence electron count equals the parity of sum(Z).
    if sum(site.specie.Z for site in struct) % 2:
        msg = "odd number of electrons"
        if calc in ("phonon_fd", "phono3py_fd") and p["ispin"] == 1:
            msg += ": ISPIN=1 forces a half-filled band; check whether this cell is magnetic or metallic"
        out.append(msg)
    if any(el.is_transition_metal or el.is_rare_earth for el in struct.composition.elements):
        out.append("contains transition-metal / rare-earth elements: check +U and magnetism before trusting ISPIN/MAGMOM")
    return out


def _phonon_cells(struct: Structure, p: dict[str, Any]) -> dict[str, Any]:
    """atomate2's supercell choice and phonopy displacement count.

    In the flow both are decided at run time on the VASP-relaxed cell; here they are
    computed on the input structure, so treat them as an estimate.
    """
    from atomate2.common.utils import get_supercell_matrix  # what the @job get_supercell_size calls
    from phonopy import Phonopy
    from pymatgen.io.phonopy import get_phonopy_structure

    sc = np.asarray(get_supercell_matrix(structure=struct, min_length=p["supercell_min_length"], max_length=None,
                                         prefer_90_degrees=True, allow_orthorhombic=False))
    ph = Phonopy(get_phonopy_structure(struct), sc, primitive_matrix="auto", symprec=p["symprec"])
    ph.generate_displacements(distance=p["displacement"])
    return {"supercell_matrix": sc.astype(int).tolist(), "n_atoms_supercell": len(ph.supercell),
            "n_displacements": len(ph.supercells_with_displacements),
            "supercell": struct * sc}


def _n_kpoints(struct: Structure, gen) -> tuple[int, int]:
    """(mesh points, irreducible points) for the KPOINTS this set writes."""
    import spglib

    kpts = gen.get_input_set(struct, potcar_spec=True)["KPOINTS"]
    mesh = [int(x) for x in kpts.kpts[0]]
    cell = (struct.lattice.matrix, struct.frac_coords, [s.specie.Z for s in struct])
    shift = [0, 0, 0]
    mapping, _ = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift, symprec=1e-3)
    return int(np.prod(mesh)), len(np.unique(mapping))


def _estimate_one(struct: Structure, calc: str, p: dict[str, Any]) -> dict[str, Any]:
    """Relative cost index sum(N_atoms^3 * N_kpts); not calibrated to core-hours yet."""
    from mga.workflows.vasp_flows import static_set

    n = len(struct)
    _, nk_ir = _n_kpoints(struct, static_set(p))
    if calc == "relax_static":
        n_relax = 2 if p["double_relax"] else 1
        return {"nsites": n, "n_calcs": n_relax + 1, "kpts_irreducible": nk_ir, "cost_index": (n_relax * 5 + 1) * n**3 * nk_ir}
    if calc == "convergence":
        n_calc = len(p["encut_values"]) + len(p["reciprocal_density_values"])
        return {"nsites": n, "n_calcs": n_calc, "kpts_irreducible": nk_ir, "cost_index": n_calc * n**3 * nk_ir}
    if calc == "phono3py_fd":
        return _estimate_phono3py(struct, p)
    cells = _phonon_cells(struct, p)
    sc = cells.pop("supercell")
    nk_sc, _ = _n_kpoints(sc, static_set(p))
    n_relax = 2 if p["double_relax"] else 1
    # displaced supercells keep little symmetry: count all mesh points (time reversal halves it)
    disp = cells["n_displacements"] * cells["n_atoms_supercell"] ** 3 * max(1, nk_sc // 2)
    return {"nsites": n, "n_calcs": n_relax + 1 + cells["n_displacements"], "kpts_irreducible": nk_ir,
            "kpts_supercell_mesh": nk_sc, **cells, "cost_index": (n_relax * 5 + 1) * n**3 * nk_ir + disp}


def _phono3py_cells(struct: Structure, p: dict[str, Any]) -> tuple[Structure, dict[str, Any], dict[str, Any]]:
    """Conventional cell, supercells + q-mesh, displacement plan; the flow builder uses the same function."""
    from mga.workflows.phono3py_fd import plan_for_input

    return plan_for_input(struct, p)


def _estimate_phono3py(struct: Structure, p: dict[str, Any]) -> dict[str, Any]:
    from mga.workflows.vasp_flows import static_set

    unit, settings, plan = _phono3py_cells(struct, p)
    n = len(unit)
    _, nk_ir = _n_kpoints(unit, static_set(p))
    n_relax = 2 if p["double_relax"] else 1
    cost = (n_relax * 5) * n**3 * nk_ir
    for order in ("fc3", "fc2"):
        sc = unit * settings[f"{order}_supercell"]
        nk_sc, _ = _n_kpoints(sc, static_set(p))
        cost += plan[f"n_{order}"] * len(sc) ** 3 * max(1, nk_sc // 2)
    return {"nsites": n, "n_calcs": n_relax + plan["n_fc3"] + plan["n_fc2"], "kpts_irreducible": nk_ir,
            "n_atoms_supercell": max(plan["n_atoms_fc3"], plan["n_atoms_fc2"]), "settings": settings,
            "plan": plan, "cost_index": cost}


def _cost_note() -> str:
    return ("cost_index = sum over calculations of N_atoms^3 x N_kpoints (relaxations weighted x5 for ionic steps). "
            "It ranks plans; it is NOT core-hours. Calibrate the factor with the dirona core-count test first.")


# --- tools ---------------------------------------------------------------------------------
def preview_vasp_inputs(
    set_name: str,
    calc: Calc,
    params_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
    max_structures: int = 3,
) -> dict[str, Any]:
    if not 1 <= max_structures <= MAX_PREVIEW:
        raise ToolInputError(f"max_structures must be 1-{MAX_PREVIEW}")
    p, prov = _params(calc, params_override, override_reason)
    pd = p.model_dump()
    sids = registry.set_sids(set_name)[:max_structures]
    with recorded_step(
        "preview_vasp_inputs",
        inputs={"set_name": set_name, "calc": calc, "sids": sids},
        params=prov,
        parent_ids=parent_steps(set_name),
    ) as rec:
        base = paths.run_dir(get_run_id()) / "previews" / rec.step_id
        previews = []
        for sid in sids:
            struct = registry.load_structure(sid)
            one: dict[str, Any] = {"sid": sid, "formula": struct.composition.reduced_formula, "inputs": {}}
            for label, ours, mp in _input_sets(calc, pd):
                vi = ours.get_input_set(struct, potcar_spec=True)
                vi.write_input(base / sid / label)
                mp_vi = mp.get_input_set(struct, potcar_spec=True)
                one["inputs"][label] = {
                    "dir": rel(base / sid / label),
                    "incar_vs_mp": _incar_diff(dict(vi["INCAR"]), dict(mp_vi["INCAR"])),
                    "kpoints": {"ours": str(vi["KPOINTS"]).splitlines()[1:4], "mp": str(mp_vi["KPOINTS"]).splitlines()[1:4]},
                    "potcar": {"functional_ours": pd["potcar_functional"], "functional_mp_default": "PBE (legacy set)",
                               "symbols": vi["POTCAR.spec"].split()},
                }
            if calc == "phonon_fd":
                cells = _phonon_cells(struct, pd)
                cells.pop("supercell")
                one["phonon_cells"] = cells
            if calc == "phono3py_fd":
                _, one["phono3py_settings"], one["displacement_plan"] = _phono3py_cells(struct, pd)
            one["warnings"] = _warnings(struct, calc, pd)
            previews.append(one)
        rec.outputs = {"preview_dir": rel(base), "n_previewed": len(previews), "of_set": len(registry.set_sids(set_name))}
    return {"step_id": rec.step_id, **rec.outputs, "previews": previews}


def estimate_vasp_cost(
    set_name: str,
    calc: Calc,
    params_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    p, prov = _params(calc, params_override, override_reason)
    pd = p.model_dump()
    sids = registry.set_sids(set_name)
    with recorded_step("estimate_vasp_cost", inputs={"set_name": set_name, "calc": calc}, params=prov,
                       parent_ids=parent_steps(set_name)) as rec:
        per = {sid: _estimate_one(registry.load_structure(sid), calc, pd) for sid in sids}
        rec.outputs = _estimate_summary(per)
        rec.outputs["details_file"] = write_details(rec.step_id, per)
    return {"step_id": rec.step_id, **rec.outputs}


def _estimate_summary(per: dict[str, dict[str, Any]]) -> dict[str, Any]:
    costs = sorted(((v["cost_index"], k) for k, v in per.items()), reverse=True)
    return {
        "n_structures": len(per),
        "n_calculations": sum(v["n_calcs"] for v in per.values()),
        "max_atoms_in_any_calc": max((v.get("n_atoms_supercell", v["nsites"]) for v in per.values()), default=0),
        "cost_index_total": int(sum(c for c, _ in costs)),
        "most_expensive": [{"sid": k, "cost_index": int(c), "share": round(c / max(1, sum(x for x, _ in costs)), 3)} for c, k in costs[:5]],
        "note": _cost_note(),
    }


def request_vasp_submission(
    set_name: str,
    calc: Calc,
    decided_by: Literal["agent", "human"],
    rationale: str,
    params_override: dict[str, Any] | None = None,
    resources_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    from mga.tools.structures import leakage_reasons
    from mga.workflows.vasp_flows import FLOW_BUILDERS

    require_decision(decided_by, rationale)
    p, prov = _params(calc, params_override, override_reason)
    r, rprov = resolve_params(VaspResources, resources_override, override_reason)
    pd = p.model_dump()
    sids = registry.set_sids(set_name)
    if not sids:
        raise ToolInputError(f"set {set_name!r} is empty")
    if len(sids) > MAX_PER_REQUEST:
        raise ToolInputError(f"{len(sids)} structures; at most {MAX_PER_REQUEST} per approval. Split the set.")
    with recorded_step(
        "request_vasp_submission",
        inputs={"set_name": set_name, "calc": calc, "n_structures": len(sids)},
        params={**prov, **{f"resources.{k}": v for k, v in rprov.items()}},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        per = {}
        for sid in sids:
            struct = registry.load_structure(sid)
            FLOW_BUILDERS[calc](struct, {**pd, "exec_incar": _exec_incar(r)}, f"check-{sid}")  # fail before approval
            per[sid] = _estimate_one(struct, calc, pd)
        leaks = {s: rs for s in sids if (rs := leakage_reasons(s))}
        overrides = {k: {"value": v.value, "reason": v.reason} for k, v in {**prov, **rprov}.items() if v.source == "override"}
        plan = {
            "calc": calc,
            "set_name": set_name,
            "sids": sids,
            "params": pd,
            "param_hash": result_hash(p),
            "resources": r.model_dump(),
            "worker": WORKER,
            "exec_config": EXEC_CONFIG,
            "requested_in_run": rec.run_id,
        }
        estimate = _estimate_summary(per)
        summary = {
            "calc": calc,
            "set_name": set_name,
            "n_structures": len(sids),
            "formulas": top((registry.load_entry(s)["reduced_formula"] for s in sids), 20),
            "n_calculations": estimate["n_calculations"],
            "max_atoms_in_any_calc": estimate["max_atoms_in_any_calc"],
            "cost_index_total": estimate["cost_index_total"],
            "overrides": overrides,
            "resources": r.model_dump(),
            "benchmark_formula_overlap": len(leaks),
            "decided_by": decided_by,
            "rationale": rationale,
        }
        appr = approvals.request("vasp", plan, summary, rec.step_id)
        rec.status = "pending_approval"
        rec.outputs = {"approval_id": appr.approval_id, "plan_hash": appr.plan_hash, "summary": summary,
                       "details_file": write_details(rec.step_id, {"estimate": per, "leakage": leaks})}
    return {
        "step_id": rec.step_id,
        "status": "pending_approval",
        "approval_id": appr.approval_id,
        "summary": summary,
        "STOP": (
            "Do not call submit_vasp yet. Show this summary to the user and ask them to review it with "
            f"`mga approvals show {appr.approval_id}` and, if they agree, run "
            f"`mga approvals approve {appr.approval_id}` in their own terminal. Then wait for them to tell you."
        ),
    }


def submit_vasp(approval_id: str) -> dict[str, Any]:
    from mga.workflows.vasp_flows import FLOW_BUILDERS

    mode = get_mode()
    appr = approvals.load(approval_id)
    if mode != "live":
        raise ToolInputError(
            f"MGA_MODE={mode}: submit_vasp only runs in live mode; the approval ({appr.status}) was not used"
        )
    jobs.check_available()  # do not spend the approval if jobflow-remote / MongoDB is not usable
    with recorded_step("submit_vasp", inputs={"approval_id": approval_id},
                       parent_ids=[appr.requested_by_step] if appr.requested_by_step else []) as rec:
        rec.decided_by = "human"
        rec.rationale = f"approved by a human ({appr.decided_at}): {appr.decision_note or ''}".strip()
        plan = approvals.consume(approval_id, "vasp", rec.step_id)
        r = VaspResources(**plan["resources"])
        params = {**plan["params"], "exec_incar": _exec_incar(r)}
        items, failed = [], []
        for sid in plan["sids"]:
            try:
                flow = FLOW_BUILDERS[plan["calc"]](registry.load_structure(sid), params, f"{plan['calc']}-{sid}")
                out_ref = getattr(flow, "output", None)
                db_ids = jobs.submit(flow, worker=plan["worker"], resources=vasp_qresources(r), exec_config=plan["exec_config"])
                items.append({
                    "sid": sid,
                    "flow_uuid": flow.uuid,
                    "output_uuid": getattr(out_ref, "uuid", None),
                    "jobs": jobs.flow_jobs(flow),
                    "db_ids": db_ids,
                })
            except Exception as exc:  # noqa: BLE001 - keep what was submitted, report the rest
                failed.append({"sid": sid, "error": f"{type(exc).__name__}: {exc}"})
        jobs.save_submission(rec.step_id, rec.run_id, {
            "step_id": rec.step_id, "kind": f"vasp_{plan['calc']}", "model": "vasp", "approval_id": approval_id,
            "set_name": plan["set_name"], "mode": "live",
            "param_hash": plan["param_hash"], "params": plan["params"], "worker": plan["worker"],
            "sids": [it["sid"] for it in items], "items": items, "failed": failed,
        })
        rec.status = "submitted" if items else "error"
        rec.outputs = {"n_submitted": len(items), "n_failed": len(failed), "failed": failed[:10]}
        if failed and not items:
            rec.error = "no flow could be submitted"
    return {"step_id": rec.step_id, **rec.outputs, "next": "job_status / fetch_results with this step_id"}
