"""records, params, registry, approvals and the approval CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import REPO, si_conventional, si_primitive

from mga import approvals, registry
from mga.approvals import ApprovalError
from mga.params import ParamError, resolve_params, result_hash
from mga.records import iter_records, load_record, recorded_step
from mga.registry import RegistryError
from mga.specs import MLIPPhononKappaParams, VaspResources


# --- records ---------------------------------------------------------------------
def test_record_ok_and_error_are_persisted(project):
    with recorded_step("demo", inputs={"x": 1}, decided_by="agent", rationale="test") as rec:
        rec.outputs = {"y": 2}
    ok = load_record(rec.step_id)
    assert ok.status == "ok" and ok.outputs == {"y": 2} and ok.actor == "agent" and ok.mode == "dry_run"
    assert ok.wall_time_s is not None and "mga" in ok.code_versions

    with pytest.raises(ValueError), recorded_step("demo") as rec2:
        raise ValueError("boom")
    bad = load_record(rec2.step_id, "test")
    assert bad.status == "error" and "boom" in bad.error and "traceback" in bad.outputs
    assert [r.step_id for r in iter_records("test")] == sorted([rec.step_id, rec2.step_id])


# --- params ----------------------------------------------------------------------
def test_params_defaults_and_provenance():
    p, prov = resolve_params(MLIPPhononKappaParams)
    assert p.displacement == 0.03 and p.transport == "MS-SMM19" and p.is_isotope is True
    assert all(v.source == "default" and v.ref for v in prov.values())


@pytest.mark.parametrize(
    "override, reason, match",
    [
        ({"not_a_param": 1}, "x", "Unknown parameter"),
        ({"displacement": 0.5}, "x", "less than or equal"),
        ({"transport": "RTA"}, "x", "Invalid parameter override"),
        ({"displacement": 0.02}, None, "override_reason"),
        ({"displacement": 0.02}, "  ", "override_reason"),
    ],
)
def test_params_rejects_bad_overrides(override, reason, match):
    with pytest.raises(ParamError, match=match):
        resolve_params(MLIPPhononKappaParams, override, reason)


def test_params_override_recorded_and_same_value_needs_no_reason():
    _, prov = resolve_params(MLIPPhononKappaParams, {"displacement": 0.03})  # equals the default
    assert prov["displacement"].source == "default"
    _, prov = resolve_params(MLIPPhononKappaParams, {"displacement": 0.01}, "convergence check")
    assert prov["displacement"].source == "override" and prov["displacement"].reason == "convergence check"


def test_result_hash_ignores_execution_only_params():
    base, _ = resolve_params(MLIPPhononKappaParams)
    chunk, _ = resolve_params(MLIPPhononKappaParams, {"chunk_size": 32}, "faster")
    disp, _ = resolve_params(MLIPPhononKappaParams, {"displacement": 0.01}, "test")
    assert result_hash(base) == result_hash(chunk) != result_hash(disp)


def test_model_default_dtype_resolves_per_model():
    from mga.models import get_model
    from mga.tools.mlip import _resolve_mlip

    pet, dpa = get_model("pet_oam_xl"), get_model("dpa41_max_oam")
    assert pet.dtype == "float32" and dpa.dtype == "float64"
    p_pet, _ = _resolve_mlip(pet, MLIPPhononKappaParams)
    p_dpa, _ = _resolve_mlip(dpa, MLIPPhononKappaParams)
    assert (p_pet.dtype, p_dpa.dtype) == ("float32", "float64")
    assert result_hash(p_pet) != result_hash(p_dpa)  # the hash records the dtype actually run
    forced, _ = _resolve_mlip(pet, MLIPPhononKappaParams, {"dtype": "float64"}, "f64 check")
    assert forced.dtype == "float64" and result_hash(forced) == result_hash(p_dpa)

def test_vasp_resources_divisibility():
    with pytest.raises(ParamError, match="divisible"):
        resolve_params(VaspResources, {"ntasks": 16, "ncore": 3}, "x")


# --- registry ----------------------------------------------------------------------
def test_structure_id_is_translation_and_order_invariant(project):
    s = si_primitive()
    shifted = s.copy()
    shifted.translate_sites(list(range(len(s))), [1, 0, -1], frac_coords=True, to_unit_cell=False)
    reordered = s.copy()
    reordered.reverse()
    assert registry.structure_id(s) == registry.structure_id(shifted) == registry.structure_id(reordered)
    assert registry.structure_id(s) != registry.structure_id(si_conventional())


def test_structure_id_has_no_brackets(project):
    # MatterGen B-C-N sample (11-07): pymatgen's reduced_formula is "B4(CN)3"
    from pymatgen.core import Lattice, Structure

    s = Structure(Lattice.cubic(6.0), ["B"] * 4 + ["C"] * 3 + ["N"] * 3, [[i / 10, i / 7 % 1, i / 3 % 1] for i in range(10)])
    assert s.composition.reduced_formula == "B4(CN)3"
    sid = registry.structure_id(s)
    assert sid.startswith("B4C3N3-")
    registry.register_structure(s, origin={"kind": "file"})
    assert registry.structure_id(si_primitive()).startswith("Si-")


def test_register_is_immutable_and_detects_tampering(project):
    s = si_primitive()
    sid, created = registry.register_structure(s, origin={"kind": "file"})
    assert created and registry.register_structure(s, origin={"kind": "other"}) == (sid, False)
    assert registry.load_entry(sid)["origin"]["kind"] == "file"
    assert registry.load_entry(sid)["spacegroup_number"] == 227

    path = project / "data" / "structures" / f"{sid}.json"
    doc = json.loads(path.read_text())
    doc["structure"]["sites"][1]["abc"] = [0.3, 0.25, 0.25]
    path.write_text(json.dumps(doc))
    with pytest.raises(RegistryError, match="modified"):
        registry.load_structure(sid)


def test_parents_and_ancestors(project):
    a, _ = registry.register_structure(si_primitive(), origin={"kind": "file"})
    b, _ = registry.register_structure(si_conventional(), origin={"kind": "derived"}, parents=[a])
    c_struct = si_conventional()
    c_struct.scale_lattice(c_struct.volume * 1.05)
    c, _ = registry.register_structure(c_struct, origin={"kind": "derived"}, parents=[b])
    assert registry.ancestors(c) == {a, b}
    orphan = si_primitive()
    orphan.scale_lattice(orphan.volume * 0.9)
    with pytest.raises(RegistryError, match="not registered"):
        registry.register_structure(orphan, origin={}, parents=["Si-0000000000"])


def test_sets_are_immutable(project):
    a, _ = registry.register_structure(si_primitive(), origin={"kind": "file"})
    b, _ = registry.register_structure(si_conventional(), origin={"kind": "file"})
    registry.create_set("s1", [a, b, a], operation="test")
    assert registry.set_sids("s1") == [a, b]
    registry.create_set("s1", [a, b], operation="test")  # identical: no-op
    with pytest.raises(RegistryError, match="immutable"):
        registry.create_set("s1", [a], operation="test")
    with pytest.raises(RegistryError, match="not registered"):
        registry.create_set("s2", ["Si-0000000000"], operation="test")
    with pytest.raises(ValueError, match="Invalid set name"):
        registry.create_set("../evil", [a], operation="test")


# --- approvals ---------------------------------------------------------------------
def _request():
    plan = {"calc": "relax_static", "sids": ["a", "b"], "params": {"encut": 520.0}}
    return approvals.request("vasp", plan, {"n_structures": 2, "calc": "relax_static"}, "step-1")


def test_approval_life_cycle(project, monkeypatch):
    appr = _request()
    assert appr.status == "pending" and appr.requested_by_actor == "agent"
    with pytest.raises(ApprovalError, match="pending"):
        approvals.consume(appr.approval_id, "vasp", "step-2")
    with pytest.raises(ApprovalError, match="human"):
        approvals.decide(appr.approval_id, True, "agent tries")

    monkeypatch.setenv("MGA_ACTOR", "human")
    approvals.decide(appr.approval_id, True, "looks fine")
    monkeypatch.setenv("MGA_ACTOR", "agent")
    with pytest.raises(ApprovalError, match="finetune"):
        approvals.consume(appr.approval_id, "finetune", "step-2")
    plan = approvals.consume(appr.approval_id, "vasp", "step-2")
    assert plan["sids"] == ["a", "b"]
    assert approvals.load(appr.approval_id).status == "consumed"
    with pytest.raises(ApprovalError, match="consumed"):
        approvals.consume(appr.approval_id, "vasp", "step-3")


def test_plan_edited_after_approval_is_refused(project, monkeypatch):
    appr = _request()
    monkeypatch.setenv("MGA_ACTOR", "human")
    approvals.decide(appr.approval_id, True, "ok")
    plan_file = project / "approvals" / appr.approval_id / "plan.json"
    plan = json.loads(plan_file.read_text())
    plan["sids"].append("c")
    plan_file.write_text(json.dumps(plan))
    with pytest.raises(ApprovalError, match="changed after approval"):
        approvals.consume(appr.approval_id, "vasp", "step-2")


def test_rejected_cannot_be_consumed(project, monkeypatch):
    appr = _request()
    monkeypatch.setenv("MGA_ACTOR", "human")
    approvals.decide(appr.approval_id, False, "too many structures")
    with pytest.raises(ApprovalError, match="rejected"):
        approvals.consume(appr.approval_id, "vasp", "step-2")
    with pytest.raises(ApprovalError, match="not pending"):
        approvals.decide(appr.approval_id, True, "changed my mind")


def test_approval_id_cannot_escape_directory(project):
    with pytest.raises(ApprovalError, match="Invalid approval id"):
        approvals.load("../../etc")


def _cli(project, *args, actor="human"):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src"), "MGA_PROJECT_ROOT": str(project), "MGA_ACTOR": actor}
    return subprocess.run([sys.executable, "-m", "mga.cli", *args], capture_output=True, text=True, env=env,
                          input=f"{args[-1]}\nok\n", timeout=60)


def test_cli_refuses_without_tty_and_for_agent(project):
    appr = _request()
    out = _cli(project, "approvals", "approve", appr.approval_id)  # stdin/stdout are pipes
    assert out.returncode != 0 and "interactive terminal" in out.stderr
    assert approvals.load(appr.approval_id).status == "pending"

    listed = _cli(project, "approvals", "list")
    assert listed.returncode == 0 and appr.approval_id in listed.stdout
    shown = _cli(project, "approvals", "show", appr.approval_id)
    assert shown.returncode == 0 and "2 structures" in shown.stdout


def test_cli_actor_check(project, monkeypatch):
    from mga import cli

    appr = _request()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("MGA_ACTOR", "agent")
    with pytest.raises(SystemExit, match="MGA_ACTOR=agent"):
        cli.main(["approvals", "approve", appr.approval_id])

    monkeypatch.setenv("MGA_ACTOR", "human")
    answers = iter([appr.approval_id, "checked INCARs"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    cli.main(["approvals", "approve", appr.approval_id])
    assert approvals.load(appr.approval_id).status == "approved"


def test_gpu_resources_nodelist_reaches_slurm_header():
    from qtoolkit.io.slurm import SlurmIO

    from mga.params import resolve_params
    from mga.specs import GpuResources
    from mga.tools.mlip import gpu_qresources

    plain, _ = resolve_params(GpuResources, None, None)
    assert "nodelist" not in gpu_qresources(plain)
    r, _ = resolve_params(GpuResources, {"partition": "GPU-a100", "nodelist": "a-a100-o-1"}, "512-atom FC2 needs 80 GB")
    header = SlurmIO().generate_header(gpu_qresources(r))
    assert "#SBATCH --nodelist=a-a100-o-1" in header and "exclude" not in header


def test_mattergen_ehull_condition_keeps_old_hash():
    import pytest

    from mga.params import ParamError, resolve_params, result_hash
    from mga.specs import MatterGenParams
    from mga.workflows.mattergen_job import mattergen_command

    # adding the optional energy_above_hull field must not invalidate results cached before it existed
    assert result_hash(MatterGenParams()) == "f22beaeba029"
    assert "--properties_to_condition_on={'chemical_system': 'B-N'}" in mattergen_command("o", "B-N", 8, MatterGenParams().model_dump())
    q, _ = resolve_params(MatterGenParams, {"pretrained_name": "chemical_system_energy_above_hull", "energy_above_hull": 0.05}, "test")
    assert "--properties_to_condition_on={'energy_above_hull': 0.05, 'chemical_system': 'B-N'}" in mattergen_command("o", "B-N", 8, q.model_dump())
    for bad in ({"pretrained_name": "chemical_system_energy_above_hull"}, {"energy_above_hull": 0.05}):
        with pytest.raises(ParamError):
            resolve_params(MatterGenParams, bad, "test")
