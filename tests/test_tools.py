"""Tool functions end to end, without a cluster.

``FakeCluster`` replaces ``mga.jobs`` submission and state queries: MLIP flows are
executed in-process with ``jobflow.run_locally`` (EMT calculator) into an in-memory
JobStore, so submit -> job_status -> fetch_results -> derived sets runs for real.
"""

from __future__ import annotations

import json
import shutil

import pytest
from monty.json import MontyDecoder
from conftest import al_fcc, rocksalt, si_conventional, si_primitive, write_json_structures

from mga import approvals, jobs, registry
from mga.approvals import ApprovalError
from mga.tools import analysis, dft, mlip, structures
from mga.tools._common import ToolInputError

A = {"decided_by": "agent", "rationale": "test"}
SMALL_KAPPA = {"fc2_min_width": 8.0, "q_mesh_length": 12.0}


# --- fake cluster ----------------------------------------------------------------------
class FakeCluster:
    def __init__(self, run: bool = True):
        from jobflow import JobStore
        from maggma.stores import MemoryStore

        self.run = run
        self.store = JobStore(MemoryStore(), additional_stores={"arrays": MemoryStore()})
        self.store.connect()
        self.states: dict[str, str] = {}
        self.submitted: list[dict] = []

    def submit(self, flow, *, worker, resources=None, exec_config=None):
        from jobflow.managers.local import run_locally

        listed = jobs.flow_jobs(flow)
        # jobflow-remote stores the flow as JSON and the worker decodes it with MontyDecoder, which
        # turns pymatgen dicts in the inputs back into objects; run_locally alone would skip that (job 960873)
        flow = json.loads(flow.to_json(), cls=MontyDecoder)
        self.submitted.append({"worker": worker, "resources": resources, "exec_config": exec_config, "n_jobs": len(listed)})
        if self.run:
            run_locally(flow, store=self.store, log=False)
        for j in listed:
            done = self.run and self.store.query_one({"uuid": j["uuid"]}) is not None
            self.states[j["uuid"]] = "COMPLETED" if done else ("FAILED" if self.run else "SUBMITTED")
        return [str(i) for i in range(len(listed))]

    def _info(self, uuid):
        return {"uuid": uuid, "state": self.states[uuid], "name": uuid[:8], "worker": "fake", "metadata": {},
                "error": None, "run_dir": None, "start_time": None, "end_time": None}

    def job_states(self, uuids):
        return {u: self._info(u) for u in uuids if u in self.states}

    def flow_states(self, flow_uuids):
        return {u: self._info(u) for u in self.states}

    def job_output(self, uuid):
        return self.store.get_output(uuid, load=True)


@pytest.fixture
def cluster(project, monkeypatch):
    fake = FakeCluster()
    for name in ("submit", "job_states", "flow_states", "job_output"):
        monkeypatch.setattr(jobs, name, getattr(fake, name))
    monkeypatch.setattr(jobs, "check_available", lambda: None)
    monkeypatch.setenv("MGA_MODE", "live")
    return fake


def _import(project, name, structs, kind="file"):
    path = write_json_structures(project / "inputs" / f"{name}.json", structs)
    return structures.import_structures(str(path), name, kind, **A)


# --- structures -------------------------------------------------------------------------
def test_structure_pipeline(project):
    bcc = si_conventional()
    bcc = type(bcc).from_spacegroup("Im-3m", bcc.lattice.__class__.cubic(3.1), ["Si"], [[0, 0, 0]])
    out = _import(project, "mix", [si_primitive(), si_conventional(), si_primitive(), bcc, rocksalt("Na", "Cl", 5.64)])
    assert out["n_read"] == 5 and out["n_in_set"] == 4 and out["n_newly_registered"] == 4

    summary = structures.structure_summary("mix")
    assert summary["reduced_formulas"] == {"Si": 3, "NaCl": 1} and summary["nsites"]["max"] == 8

    filt = structures.filter_structures("mix", "mix.small", max_sites=4, **A)
    assert filt["n_kept"] == 2 and filt["removed_by_criterion"] == {"max_sites": 2}  # conventional Si and NaCl (8 sites)
    with pytest.raises(ToolInputError, match="at least one"):
        structures.filter_structures("mix", "mix.none", **A)
    prim = structures.filter_structures("mix", "mix.prim2", max_primitive_sites=2, **A)
    assert prim["n_kept"] == 4  # conventional Si and rock-salt NaCl reduce to 2-atom primitive cells
    assert structures.filter_structures("mix", "mix.prim1", max_primitive_sites=1, **A)["n_kept"] == 1  # bcc

    base = si_primitive()  # fixed small displacement: P-1 at symprec 1e-3, Fd-3m at 0.1
    rattled = type(base)(base.lattice, ["Si", "Si"], [[0.003, 0.0, -0.002], [0.25, 0.252, 0.249]])
    _import(project, "rattled", [rattled])
    symm = structures.symmetrize_structures("rattled", "rattled.sym", **A)
    assert symm["spacegroups"] == {"227": 1} and symm["n_p1"] == 0
    # input -> output link is in the set details (an identical earlier entry is reused unchanged)
    sym_sid, src = registry.set_sids("rattled.sym")[0], registry.set_sids("rattled")[0]
    assert registry.load_entry(src)["spacegroup_number"] == 2 and registry.load_entry(sym_sid)["spacegroup_number"] == 227
    assert registry.load_set("rattled.sym")["details"]["changes"][src]["sid"] == sym_sid

    merged = structures.merge_sets(["mix.prim1", "mix.small"], "mix.merged", **A)  # bcc is in both
    assert merged["n_merged"] == 2 and merged["n_in"] == {"mix.prim1": 1, "mix.small": 2}
    assert registry.set_sids("mix.merged")[0] == registry.set_sids("mix.prim1")[0]

    dedup = structures.deduplicate("mix", "mix.unique", **A)
    assert dedup["n_unique"] == 3 and dedup["n_removed"] == 1  # primitive and conventional diamond Si match

    leak = structures.check_benchmark_leakage("mix.unique", "mix.clean", **A)
    assert leak["n_flagged"] == 1 and leak["flagged_formulas"] == {"NaCl": 1} and leak["n_benchmark_formulas"] == 70
    clean = registry.set_sids("mix.clean")
    structures.assert_no_leakage(clean)
    with pytest.raises(ToolInputError, match="kappa_SRME"):
        structures.assert_no_leakage(registry.set_sids("mix"))

    names = {s["name"] for s in structures.list_sets()["sets"]}
    assert names == {"mix", "mix.small", "mix.prim2", "mix.prim1", "mix.merged", "rattled", "rattled.sym", "mix.unique", "mix.clean"}


def test_leakage_follows_descendants(project):
    nacl, _ = registry.register_structure(rocksalt("Na", "Cl", 5.64), origin={"kind": "file"})
    # a non-benchmark formula derived from a benchmark structure (e.g. by substitution) is still excluded
    child, _ = registry.register_structure(rocksalt("Cs", "Cl", 7.0), origin={"kind": "derived"}, parents=[nacl])
    assert "CsCl" not in structures.benchmark_formulas()
    reasons = structures.leakage_reasons(child)
    assert any(r.startswith(f"ancestor {nacl}") for r in reasons)


def test_import_kappa_reference_and_reject_outside_paths(project, tmp_path):
    # copied in: data/reference is a symlink here, and project_path refuses paths that resolve outside the root
    ref = project / "inputs" / "phononDB-PBE-structures.extxyz"
    ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(project / "data" / "reference" / "kappa_srme" / ref.name, ref)
    out = structures.import_structures(str(ref), "ksrme103", "kappa_srme_reference", **A)
    assert out["n_read"] == 103 and out["n_in_set"] == 103 and out["n_failed"] == 0
    entry = registry.load_entry(registry.set_sids("ksrme103")[0])
    assert entry["info"]["mp_id"].startswith("mp-") and entry["info"]["kappa_settings"]["source"] == "phonondb_pbe"
    assert len(structures.check_benchmark_leakage("ksrme103", "ksrme103.clean", **A)) and not registry.set_sids("ksrme103.clean")

    outside = write_json_structures(tmp_path / "outside.json", [si_primitive()])
    with pytest.raises(ToolInputError, match="outside the project"):
        structures.import_structures(str(outside), "x", "file", **A)
    with pytest.raises(ToolInputError, match="rationale"):
        structures.import_structures(str(ref), "y", "file", decided_by="agent", rationale=" ")


# --- mlip: dry_run ---------------------------------------------------------------------
def test_mlip_dry_run_submits_nothing(project):
    _import(project, "al", [al_fcc()])
    relax = mlip.submit_mlip_relax("al", "emt_test", **A)
    assert relax["mode"] == "dry_run" and relax["n_jobs"] == 1 and relax["n_to_compute"] == 1
    kappa = mlip.submit_mlip_phonon_kappa("al", "emt_test", kappa_override=SMALL_KAPPA, override_reason="small test", **A)
    assert kappa["settings"]["settings_sources"] == ["length_rule_v2_on_input"]
    assert kappa["settings"]["examples"][0]["fc3_supercell"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    conv = mlip.submit_mlip_convergence("al", "emt_test", "q_mesh_length", [10.0, 12.0], **A)
    assert conv["n_jobs"] == 2 and len(conv["variants"]) == 2
    mg = mlip.submit_mattergen("Si-C", 10, **A)
    assert mg["key"] == "C-Si__n10__a" and mg["command_preview"][0] == "mattergen-generate"
    assert not (project / "runs" / "test" / "submissions").exists()
    with pytest.raises(ToolInputError, match="No submission map"):
        analysis.fetch_results(relax["step_id"])
    with pytest.raises(ToolInputError, match="Unknown model"):
        mlip.submit_mlip_relax("al", "no_such_model", **A)
    with pytest.raises(ToolInputError, match="not an element"):
        mlip.submit_mattergen("Si-Xx", 10, **A)


def test_live_refuses_models_without_checkpoint(project, cluster):
    _import(project, "al", [al_fcc()])
    todo = [m["name"] for m in mlip.list_models()["models"] if not m["ready"]]
    if not todo:
        pytest.skip("every configured model has a checkpoint")
    with pytest.raises(ToolInputError, match="not ready"):
        mlip.submit_mlip_relax("al", todo[0], **A)
    assert cluster.submitted == []


# --- mlip: live with the fake cluster --------------------------------------------------------
@pytest.mark.slow
def test_mlip_relax_and_kappa_end_to_end(project, cluster, monkeypatch):
    _import(project, "al", [al_fcc()])
    relax = mlip.submit_mlip_relax("al", "emt_test", **A)
    assert relax["mode"] == "live" and cluster.submitted[-1]["worker"] == "local"
    status = analysis.job_status(relax["step_id"])
    assert status["all_final"] and status["n_completed"] == 1
    fetched = analysis.fetch_results(relax["step_id"])
    assert fetched["ingest"]["n_saved"] == 1
    relaxed_set = fetched["derived"]["relaxed_set"]
    assert relaxed_set.startswith("al.relaxed.emt_test.") and fetched["derived"]["n_relaxed"] == 1
    child = registry.set_sids(relaxed_set)[0]
    assert registry.load_entry(child)["origin"]["kind"] == "mlip_relax"
    assert fetched["derived"]["converged_set"] == relaxed_set + ".converged" and fetched["derived"]["n_unconverged"] == 0
    assert registry.set_sids(relaxed_set + ".converged") == [child]

    again = mlip.submit_mlip_relax("al", "emt_test", **A)  # cached: nothing submitted
    assert again["n_cached"] == 1 and again["n_jobs"] == 0 and len(cluster.submitted) == 1
    assert analysis.job_status(again["step_id"])["n_jobs"] == 0
    with pytest.raises(ToolInputError, match="energy_reference"):
        analysis.compute_ehull(relax["step_id"], **A)

    kappa = mlip.submit_mlip_phonon_kappa("al", "emt_test", kappa_override=SMALL_KAPPA, override_reason="small test", **A)
    k = analysis.fetch_results(kappa["step_id"])["derived"]
    assert k["n_with_kappa"] == 1 and 1.0 < k["kappa_300K_W_mK"]["median"] < 100.0
    with pytest.raises(ToolInputError, match="no kappa_SRME benchmark"):
        analysis.kappa_srme(kappa["step_id"])

    # a second "model" (EMT again) on the same fixed settings: zero disagreement, mode-comparable
    from mga import models

    orig = models.all_models()
    monkeypatch.setattr(models, "all_models", lambda: {**orig, "emt_b": orig["emt_test"].model_copy(update={"name": "emt_b"})})
    kappa_b = mlip.submit_mlip_phonon_kappa("al", "emt_b", kappa_override=SMALL_KAPPA, override_reason="small test", **A)
    analysis.fetch_results(kappa_b["step_id"])
    dis = analysis.model_disagreement([kappa["step_id"], kappa_b["step_id"]], top_n=1, out_set="al.disagree", **A)
    pair = dis["pairs"]["emt_test|emt_b"]
    assert pair["n_compared"] == 1 and pair["mean_abs_srd"] == 0.0 and pair["mean_srme"] == 0.0
    assert dis["param_differences"] == [] and registry.set_sids("al.disagree") == registry.set_sids("al")
    with pytest.raises(ToolInputError, match="different model"):
        analysis.model_disagreement([kappa["step_id"], kappa["step_id"]])

    # replay: no submission, everything from the cache
    monkeypatch.setenv("MGA_MODE", "replay")
    n_before = len(cluster.submitted)
    rep = mlip.submit_mlip_phonon_kappa("al", "emt_test", kappa_override=SMALL_KAPPA, override_reason="small test", **A)
    assert rep["mode"] == "replay" and len(cluster.submitted) == n_before
    assert analysis.fetch_results(rep["step_id"])["derived"]["n_with_kappa"] == 1
    from mga.results import ResultMissing

    with pytest.raises(ResultMissing):
        mlip.submit_mlip_phonon_kappa("al", "emt_test", **A)  # default (large) settings were never computed


def test_exception_results_are_retried_not_cached(project, cluster):
    # job 963847: a CUDA OOM was stored as the result and the resubmission came back "cached"
    from mga.utils import read_json, write_json_atomic

    _import(project, "al", [al_fcc()])
    first = mlip.submit_mlip_relax("al", "emt_test", **A)
    analysis.fetch_results(first["step_id"])
    (path,) = (project / "data" / "results" / "mlip_relax" / "emt_test").glob("*.json")
    good = read_json(path)
    write_json_atomic(path, {**good, "data": {**good["data"], "ok": False, "error": "RuntimeError: CUDA out of memory"}})

    retry = mlip.submit_mlip_relax("al", "emt_test", **A)
    assert retry["n_cached"] == 0 and retry["n_jobs"] == 1
    assert analysis.fetch_results(retry["step_id"])["ingest"]["n_saved"] == 1
    assert read_json(path)["data"]["ok"] is True
    assert mlip.submit_mlip_relax("al", "emt_test", **A)["n_cached"] == 1


@pytest.mark.slow
def test_convergence_scan(project, cluster):
    _import(project, "al", [al_fcc()])
    conv = mlip.submit_mlip_convergence("al", "emt_test", "q_mesh_length", [10.0, 12.0], **A)
    fetched = analysis.fetch_results(conv["step_id"])
    assert fetched["ingest"]["n_saved"] == 2 and "kappa_convergence" in fetched["derived"]["note"]
    table = analysis.kappa_convergence(conv["step_id"])["table"]
    assert [v["status"] for v in table[0]["values"]] == ["ok", "ok"] and table[0]["rel_change_last"] is not None
    with pytest.raises(ToolInputError, match="convergence scan"):
        analysis.kappa_srme(conv["step_id"])


def test_mattergen_ingest_and_derive(project, monkeypatch):
    fake = FakeCluster(run=False)
    monkeypatch.setattr(jobs, "submit", fake.submit)
    monkeypatch.setattr(jobs, "job_states", lambda uuids: {u: {**fake._info(u), "state": "COMPLETED"} for u in uuids})
    monkeypatch.setenv("MGA_MODE", "live")
    gen = [si_primitive().as_dict(), rocksalt("Si", "C", 4.36).as_dict(), si_primitive().as_dict()]
    monkeypatch.setattr(jobs, "job_output", lambda uuid: {"chemical_system": "C-Si", "n_requested": 3, "n_returned": 3,
                                                          "structures": gen})
    mg = mlip.submit_mattergen("Si-C", 3, **A)
    assert fake.submitted[0]["worker"] == "datalab_mattergen"
    out = analysis.fetch_results(mg["step_id"])["derived"]
    assert out["generated_set"] == "mattergen.C-Si__n3__a" and out["n_unique_ids"] == 2
    entry = registry.load_entry(registry.set_sids(out["generated_set"])[0])
    assert entry["origin"]["kind"] == "mattergen"


# --- dft ------------------------------------------------------------------------------------
@pytest.fixture
def nacl_set(project):
    _import(project, "nacl", [rocksalt("Na", "Cl", 5.64).get_primitive_structure()])
    _import(project, "fe", [type(si_primitive()).from_spacegroup("Im-3m", si_primitive().lattice.__class__.cubic(2.87),
                                                                    ["Fe"], [[0, 0, 0]])])
    return "nacl"


def test_preview_and_estimate(project, nacl_set):
    pv = dft.preview_vasp_inputs("nacl", "phonon_fd")
    one = pv["previews"][0]
    disp = one["inputs"]["displacement"]
    diff = disp["incar_vs_mp"]  # only keys that differ from MPStaticSet
    assert "ENCUT" not in diff and diff["EDIFF"]["ours"] == 1e-8 and diff["ISPIN"] == {"mp": 2, "ours": 1}
    assert diff["MAGMOM"]["ours"] is None and disp["potcar"]["symbols"] == ["Na_pv", "Cl"]
    assert disp["potcar"]["functional_ours"] == "PBE_54"
    incar = (project / disp["dir"] / "INCAR").read_text()
    assert "ENCUT = 520.0" in incar and "LREAL = False" in incar
    assert "GGA" not in incar and "LWAVE = False" in incar  # PBE (no PBEsol GGA tag), no WAVECAR
    assert one["inputs"]["relax"]["incar_vs_mp"]["EDIFFG"]["ours"] == -0.001
    assert one["phonon_cells"]["n_displacements"] == 2 and one["phonon_cells"]["n_atoms_supercell"] == 64
    assert one["warnings"] == []
    fe = dft.preview_vasp_inputs("fe", "phonon_fd")["previews"][0]["warnings"]
    assert any("transition-metal" in w for w in fe)

    est = dft.estimate_vasp_cost("nacl", "phonon_fd")
    assert est["n_structures"] == 1 and est["n_calculations"] == 2 + 1 + 2 and est["cost_index_total"] > 0
    rs = dft.estimate_vasp_cost("nacl", "relax_static")
    assert rs["n_calculations"] == 3 and rs["cost_index_total"] < est["cost_index_total"]
    assert dft.estimate_vasp_cost("nacl", "convergence")["n_calculations"] == 8


def test_phono3py_preview_estimate_request(project, nacl_set):
    one = dft.preview_vasp_inputs("nacl", "phono3py_fd")["previews"][0]
    assert set(one["inputs"]) == {"relax", "displacement"}
    assert one["inputs"]["displacement"]["incar_vs_mp"]["ISPIN"] == {"mp": 2, "ours": 1}
    plan = one["displacement_plan"]
    assert one["phono3py_settings"]["fc3_supercell"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # 8 A on the conventional cell
    assert plan["n_atoms_fc3"] == 64 and plan["fc3_mode"] in ("random", "systematic")
    assert plan["n_fc3"] == min(plan["n_random"]["fc3"], plan["n_systematic"]["fc3"])

    est = dft.estimate_vasp_cost("nacl", "phono3py_fd")
    assert est["n_calculations"] == 2 + plan["n_fc3"] + plan["n_fc2"]
    assert est["max_atoms_in_any_calc"] == plan["n_atoms_fc2"]
    req = dft.request_vasp_submission("nacl", "phono3py_fd", **A)
    frozen = approvals.load_plan(req["approval_id"])
    assert frozen["calc"] == "phono3py_fd" and frozen["params"]["random_seed"] == 1103
    assert req["summary"]["n_calculations"] == est["n_calculations"]


def test_vasp_request_needs_human_approval(project, nacl_set, monkeypatch):
    req = dft.request_vasp_submission("nacl", "phonon_fd", **A)
    aid = req["approval_id"]
    assert req["status"] == "pending_approval" and "STOP" in req and req["summary"]["benchmark_formula_overlap"] == 1
    plan = approvals.load_plan(aid)
    assert plan["worker"] == "dirona_vasp" and plan["params"]["potcar_functional"] == "PBE_54"

    with pytest.raises(ToolInputError, match="live mode"):
        dft.submit_vasp(aid)  # dry_run
    fake = FakeCluster(run=False)
    monkeypatch.setattr(jobs, "submit", fake.submit)
    monkeypatch.setattr(jobs, "check_available", lambda: None)
    monkeypatch.setattr(jobs, "flow_states", fake.flow_states)
    monkeypatch.setenv("MGA_MODE", "live")
    with pytest.raises(ApprovalError, match="pending"):
        dft.submit_vasp(aid)
    assert fake.submitted == [] and approvals.load(aid).status == "pending"

    monkeypatch.setenv("MGA_ACTOR", "human")
    approvals.decide(aid, True, "checked preview")
    monkeypatch.setenv("MGA_ACTOR", "agent")
    sub = dft.submit_vasp(aid)
    assert sub["n_submitted"] == 1 and sub["n_failed"] == 0
    assert fake.submitted[0]["exec_config"] == "vasp_642_intel" and fake.submitted[0]["resources"]["ntasks"] == 16
    assert approvals.load(aid).status == "consumed"
    with pytest.raises(ApprovalError, match="consumed"):
        dft.submit_vasp(aid)
    st = analysis.job_status(sub["step_id"])
    assert st["n_jobs"] >= 3 and "displacement jobs at run time" in st["note"]

    saved = json.loads((project / "runs" / "test" / "submissions" / f"{sub['step_id']}.json").read_text())
    assert saved["approval_id"] == aid and saved["items"][0]["output_uuid"]


def test_vasp_request_validation(project, nacl_set):
    with pytest.raises(ToolInputError, match="rationale"):
        dft.request_vasp_submission("nacl", "relax_static", decided_by="agent", rationale="")
    from mga.params import ParamError

    with pytest.raises(ParamError, match="override_reason"):
        dft.request_vasp_submission("nacl", "relax_static", params_override={"encut": 600.0}, **A)
    with pytest.raises(ParamError):
        dft.request_vasp_submission("nacl", "relax_static", params_override={"potcar_functional": "PBE"}, override_reason="x", **A)
    assert approvals.list_approvals() == []
