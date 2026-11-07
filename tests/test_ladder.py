"""Multi-fidelity ladder (DESIGN §3.2): L0 descriptors, L1 -> L2 force-constant cache, L3 FC4 +
ShengBTE files, and the budget ledger / escalation baselines. EMT stands in for the MLIP."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import al_fcc

from mga.analysis import escalation as esc

A = {"decided_by": "agent", "rationale": "test"}
SMALL_KAPPA = {"fc3_min_width": 6.0, "fc2_min_width": 8.0, "q_mesh_length": 12.0, "use_reference_settings": False}
SMALL_FC4 = {"supercell_min_width": 5.0, "fc3_cutoff": 4.0, "fc4_shells": 1, "min_snapshots": 8}


def _params(**kappa):
    from mga.specs import MLIPPhononKappaParams, MLIPRelaxParams

    return MLIPRelaxParams().model_dump(), MLIPPhononKappaParams(**{**SMALL_KAPPA, **kappa}).model_dump()


# --- pure functions ------------------------------------------------------------------------
def test_slack_formula_sanity():
    from mga.workflows.ladder import slack_kappa

    # diamond-like numbers (M=12 amu, theta_a ~ 1000 K? use the commonly quoted Slack inputs scale):
    # kappa grows as theta^3 and falls as gamma^2 and T
    base = slack_kappa(12.0, 500.0, 5.7, 2, 1.0, 300.0)
    assert base and base > 0
    assert slack_kappa(12.0, 1000.0, 5.7, 2, 1.0, 300.0) == pytest.approx(8 * base)
    assert slack_kappa(12.0, 500.0, 5.7, 2, 1.0, 600.0) == pytest.approx(base / 2)
    assert slack_kappa(12.0, 500.0, 5.7, 2, 2.0, 300.0) < base / 3
    assert slack_kappa(12.0, 500.0, 5.7, 2, None, 300.0) is None


def test_collision_matrix_estimate():
    from mga.workflows.ladder import collision_matrix_gb

    # (n_ir * n_band * 3)^2 doubles, two of them (full + symmetrised work array), per temperature
    assert collision_matrix_gb(100, 6, 1) == pytest.approx(2 * (100 * 6 * 3) ** 2 * 8 / 1e9)
    assert collision_matrix_gb(100, 6, 2) == pytest.approx(2 * collision_matrix_gb(100, 6, 1))


def _cell(cost, values, stable=None):
    return {"cost_h": cost, "values": values, "stable": stable or [], "n_models": max(1, len(values))}


def _table():
    """Six candidates. 'a' looks best at L0 but is mediocre; 'f' is unstable."""
    t = {
        "a": {"L0": _cell(0.1, [90.0], [True]), "L1": _cell(1, [40.0]), "L2": _cell(5, [1.1]), "L3": _cell(20, [0.9])},
        "b": {"L0": _cell(0.1, [50.0], [True]), "L1": _cell(1, [300.0]), "L2": _cell(5, [1.5]), "L3": _cell(20, [0.8])},
        "c": {"L0": _cell(0.1, [40.0], [True]), "L1": _cell(1, [200.0]), "L2": _cell(5, [1.2])},
        "d": {"L0": _cell(0.1, [10.0], [True]), "L1": _cell(1, [20.0])},
        "e": {"L0": _cell(0.1, [5.0], [True, False, True]), "L1": _cell(1, [10.0])},
        "f": {"L0": _cell(0.1, [99.0], [False, False])},
    }
    return t, {s: esc.scores(r) for s, r in t.items()}


def test_scores_chain_the_ratios():
    _, sc = _table()
    assert sc["b"]["after"]["L2"] == pytest.approx(450.0) and sc["b"]["truth"] == pytest.approx(360.0)
    assert sc["b"]["truth_rung"] == "L3"
    assert sc["c"]["truth"] == pytest.approx(240.0) and sc["c"]["truth_rung"] == "L2"
    assert sc["d"]["truth_rung"] == "L1" and sc["d"]["after"]["L2"] == 20.0  # L1 carries on
    assert sc["e"]["stable"] is True and sc["f"]["stable"] is False and sc["f"]["truth"] is None


def test_fixed_and_random_policies_respect_budget():
    t, sc = _table()
    fixed = esc.run_policy(t, sc, [3, 2, 1], esc.fixed_order(sc), None, hit_threshold=100.0)
    assert fixed["chosen"]["L1"] == ["a", "b", "c"]  # by L0 score; 'f' is unstable
    assert fixed["chosen"]["L2"] == ["b", "c"] and fixed["chosen"]["L3"] == ["b"]
    assert fixed["hits_in_pool"] == 2 and fixed["hits_found"] == 1 and fixed["recall"] == 0.5
    assert fixed["spent_h"] == pytest.approx(0.6 + 3 + 10 + 20)
    # a budget cut: L3 no longer fits, nothing reaches it
    cut = esc.run_policy(t, sc, [3, 2, 1], esc.fixed_order(sc), 20.0, hit_threshold=100.0)
    assert cut["spent_h"] <= 20.0 and cut["chosen"]["L3"] == [] and cut["hits_found"] == 0
    ks, f = esc.ks_for_budget(t, sc, 40.0, 100.0)
    assert esc.run_policy(t, sc, ks, esc.fixed_order(sc), None, 100.0)["spent_h"] <= 40.0 and 0 < f <= 1
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = esc.run_policy(t, sc, [3, 2, 1], esc.random_order(rng), 35.0, 100.0)
        assert r["spent_h"] <= 35.0 and "f" not in r["chosen"]["L1"]
    rho = esc.spearman_between_rungs(sc)
    assert rho["L0->L1"]["n"] == 5 and rho["L0->L1"]["rho"] is not None


# --- L0 / L1 -> L2 on EMT ----------------------------------------------------------------------
@pytest.mark.slow
def test_descriptors_and_fc_cache_round_trip(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from mga.specs import MLIPFc2DescriptorParams, MLIPLbteParams
    from mga.workflows import ladder
    from mga.workflows.kappa_protocol import run_structure

    rp, kp = _params()
    al = bulk("Al", "fcc", a=4.05)
    d = ladder.run_descriptors(al.copy(), EMT(), rp, {**kp, **MLIPFc2DescriptorParams().model_dump()})
    assert d["ok"], d.get("traceback")
    x = d["descriptors"]
    assert not d["has_imaginary_modes"] and 100 < x["theta_acoustic_K"] < 600
    assert x["kappa_slack_W_mK"] > 0 and x["gamma_cv_mean"] > 0 and x["wjdos_mean"] > 0

    fc_dir = str(tmp_path / "fc")
    l1 = run_structure(al.copy(), EMT(), rp, kp, None, fc_dir)
    assert l1["ok"] and ladder.fc_cache_complete(fc_dir), l1.get("fc_cache")
    l2 = ladder.run_lbte(al.copy(), None, rp, kp, MLIPLbteParams().model_dump(), None, fc_dir)
    assert l2["ok"], l2.get("traceback")
    assert l2["fc_source"] == "cache" and not l2["lbte_mesh"]["reduced"]
    # same force constants, same mesh: the RTA half of the LBTE run reproduces L1
    assert l2["kappa_tot_rta_avg"][0] == pytest.approx(l1["kappa_tot_avg"][0], rel=1e-6)
    assert l2["lbte_over_rta"][0] > 1.0  # Normal processes are not resistive in the exact solution
    # without cache and without calculator: a clear failure, not a crash
    bad = ladder.run_lbte(al.copy(), None, rp, kp, MLIPLbteParams().model_dump(), None, str(tmp_path / "none"))
    assert not bad["ok"] and "no FC cache" in bad["error"]


# --- L3 on EMT ---------------------------------------------------------------------------------
@pytest.mark.slow
def test_fc4_fit_and_shengbte_files(tmp_path, monkeypatch):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from mga.specs import MLIPFc4Params
    from mga.workflows import ladder

    monkeypatch.delenv("MGA_FOURPHONON", raising=False)
    rp, kp = _params()
    out = ladder.run_fc4(bulk("Al", "fcc", a=4.05), EMT(), rp, kp, MLIPFc4Params(**SMALL_FC4).model_dump(), None,
                         str(tmp_path / "l3"))
    assert out["ok"], out.get("traceback")
    fit = out["fit"]
    rmse = fit["force_rmse_test_by_order"]
    assert rmse[4] < rmse[2] and rmse[4] < 0.1 * fit["force_rms_test"]
    files = out["shengbte_files"]
    assert max(files["roundtrip_max_rel_dev"].values()) < 1e-8
    assert out["four_phonon_ratio"] is None and "inputs written only" in out["fourphonon"]["skipped"]
    assert out["kappa_tot_fit_avg"][0] > 0
    wd = tmp_path / "l3"
    for name in ("FORCE_CONSTANTS_2ND", "FORCE_CONSTANTS_3RD", "FORCE_CONSTANTS_4TH", "3ph/CONTROL", "3_4ph/CONTROL",
                 "3_4ph/FORCE_CONSTANTS_4TH"):
        assert (wd / name).exists(), name
    c34, c3 = ((wd / d / "CONTROL").read_text().replace(" ", "").lower() for d in ("3_4ph", "3ph"))
    assert "four_phonon=.true." in c34 and "num_sample_process_4ph=100000" in c34
    assert "four_phonon=.false." in c3 and "num_sample" not in c3  # FourPhonon refuses sampling without 4ph

    # every R in FORCE_CONSTANTS_4TH is a lattice vector of the primitive cell of the *relaxed*
    # structure (EMT moves a from 4.05 to ~3.99 A)
    from ase import Atoms

    rs_ = out["relaxed_structure"]
    prim = ladder.primitive_atoms(Atoms(numbers=rs_["numbers"], cell=rs_["cell"], positions=rs_["positions"], pbc=True),
                                  kp["symprec"])
    blocks = ladder.read_shengbte_fc(str(wd / "FORCE_CONSTANTS_4TH"), 4)
    inv = np.linalg.inv(np.asarray(prim.cell))
    rs = np.array([r for b in blocks for r in b[0]])
    frac = rs @ inv
    assert len(blocks) == files["n_blocks"][4]
    assert np.abs(frac - np.round(frac)).max() < 1e-4


# --- tools end to end with the fake cluster ------------------------------------------------------------
@pytest.fixture
def cluster(project, monkeypatch, tmp_path):
    from test_tools import FakeCluster

    from mga import jobs

    fake = FakeCluster()
    for name in ("submit", "job_states", "flow_states", "job_output"):
        monkeypatch.setattr(jobs, name, getattr(fake, name))
    monkeypatch.setattr(jobs, "check_available", lambda: None)
    monkeypatch.setenv("MGA_MODE", "live")
    monkeypatch.setenv("MGA_FC_CACHE", str(tmp_path / "fc_cache"))
    monkeypatch.setenv("MGA_LADDER_WORK", str(tmp_path / "ladder"))
    monkeypatch.delenv("MGA_FOURPHONON", raising=False)
    return fake


@pytest.mark.slow
def test_ladder_tools_end_to_end(project, cluster, tmp_path):
    from test_tools import _import

    from mga.tools import analysis, mlip

    _import(project, "al", [al_fcc()])
    over = {"kappa_override": SMALL_KAPPA, "override_reason": "small test"}
    l0 = mlip.submit_mlip_fc2_descriptors("al", "emt_test", **over, **A)
    d0 = analysis.fetch_results(l0["step_id"])["derived"]
    assert d0["n_results"] == 1 and d0["n_failed"] == 0 and d0["descriptors"]["kappa_slack_W_mK"]["n"] == 1

    l1 = mlip.submit_mlip_phonon_kappa("al", "emt_test", **over, **A)
    assert l1["fc_cache"] == str(tmp_path / "fc_cache")
    analysis.fetch_results(l1["step_id"])
    l2 = mlip.submit_mlip_kappa_lbte("al", "emt_test", **over, **A)
    assert l2["n_with_l1_result"] == 1 and l2["kappa_param_hash"] == l1["param_hash"]
    d2 = analysis.fetch_results(l2["step_id"])["derived"]
    assert d2["n_from_fc_cache"] == 1 and d2["lbte_over_rta"]["min"] > 1.0

    l3 = mlip.submit_mlip_fc4_fourphonon("al", "emt_test", **over, fc4_override=SMALL_FC4, **A)
    d3 = analysis.fetch_results(l3["step_id"])["derived"]
    assert d3["n_failed"] == 0 and d3["n_fourphonon_run"] == 0 and d3["roundtrip_max_rel_dev"] < 1e-8
    assert (tmp_path / "ladder" / "fc4" / "emt_test").is_dir()

    steps = {"L0": [l0["step_id"]], "L1": [l1["step_id"]], "L2": [l2["step_id"]], "L3": [l3["step_id"]]}
    ledger = analysis.budget_ledger(steps, budget_gpu_h=1.0)
    assert set(ledger["by_rung"]) == {"L0", "L1", "L2", "L3"} and ledger["remaining_gpu_h"] < 1.0
    base = analysis.escalation_baseline(steps, budget_gpu_h=1.0, hit_threshold_W_mK=1.0, n_random=5,
                                        agent_sets={"L1": "al", "L2": "al", "L3": "al"})
    assert base["pool"]["n"] == 1 and base["fixed_rule"]["hits_found"] == base["agent"]["hits_found"]
    assert base["random"]["n_draws"] == 5

    from mga.tools._common import ToolInputError

    with pytest.raises(ToolInputError, match="rung L2 takes mlip_lbte"):
        analysis.budget_ledger({"L2": [l1["step_id"]]})


def test_parse_shengbte_kappa_fourphonon_format(tmp_path):
    """Rows as FourPhonon writes them (Example-Si/output/T300K/BTE.kappa_tensor, first/last rows)."""
    from mga.workflows.ladder import parse_shengbte_kappa

    d = tmp_path / "T300K"
    d.mkdir()
    diag = lambda k: f"{k:.10E}    0.0E+00    0.0E+00    0.0E+00    {k:.10E}    0.0E+00    0.0E+00    0.0E+00    {k:.10E}"  # noqa: E731
    rows = [(0, 121.4377748), (1, 122.7346035), (4, 123.9808162), (5, 123.9813334)]
    (d / "BTE.kappa_tensor").write_text("".join(f"        {i}    {diag(k)}\n" for i, k in rows))
    out = parse_shengbte_kappa(tmp_path)
    assert out["kappa_rta_avg"] == pytest.approx(121.4377748)
    assert out["kappa_avg"] == pytest.approx(123.9813334) and out["n_iter"] == 5
    assert out["last_rel_change"] == pytest.approx(123.9813334 / 123.9808162 - 1)
    assert parse_shengbte_kappa(tmp_path / "missing") is None


def test_parse_shengbte_kappa_vs_t_fallback(tmp_path):
    """Example-Si-sampling/output ships only BTE.KappaTensorVsT_{CONV,RTA} (first rows copied)."""
    from mga.workflows.ladder import parse_shengbte_kappa

    (tmp_path / "BTE.KappaTensorVsT_CONV").write_text(
        "  300.0   0.13492E+03  -0.16130E-16   0.16835E-16   0.16834E-16   0.13492E+03  -0.23121E-16  -0.16202E-16   0.23121E-16   0.13492E+03     5\n")
    (tmp_path / "BTE.KappaTensorVsT_RTA").write_text(
        "  300.0   0.13282E+03   0.97083E-18   0.11237E-17   0.11237E-17   0.13282E+03  -0.22588E-20   0.97083E-18   0.22588E-20   0.13282E+03\n")
    out = parse_shengbte_kappa(tmp_path)
    assert out["temperature"] == 300.0 and out["n_iter"] == 5
    assert out["kappa_avg"] == pytest.approx(134.92) and out["kappa_rta_avg"] == pytest.approx(132.82)


def test_job_inputs_survive_jobflow_decoding():
    """jobflow hands a job its inputs through MontyDecoder: the pymatgen dicts in ``items``
    arrive as Structure objects (job 960873)."""
    import json

    from monty.json import MontyDecoder, MontyEncoder
    from pymatgen.core import Lattice, Structure

    from mga.workflows.mlip_jobs import _atoms

    s = Structure(Lattice.cubic(3.6), ["B", "N"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    items = [{"sid": "BN-x", "structure": s.as_dict(), "settings": None}]
    decoded = json.loads(json.dumps(items, cls=MontyEncoder), cls=MontyDecoder)
    assert isinstance(decoded[0]["structure"], Structure)
    for struct in (decoded[0]["structure"], items[0]["structure"]):
        assert _atoms(struct).get_chemical_formula() == "BN"


def test_fc4_shell_cutoff_and_sparse_fc4_match_symfc():
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from phonopy import Phonopy
    from symfc import Symfc
    from symfc.solvers.solver_O2O3O4 import FCSolverO2O3O4
    from symfc.utils.utils import SymfcAtoms

    from mga.workflows import ladder

    # zincblende shells at d, 1.633 d, 1.915 d: two shells -> cutoff between 2nd and 3rd for any a
    for a in (3.55, 3.625, 3.70):
        prim = bulk("BN", "zincblende", a=a)
        ph = Phonopy(ladder._phonopy_atoms(prim), supercell_matrix=np.diag([4, 4, 4]), primitive_matrix=None)
        shells, half = ladder.neighbour_shells(ph.supercell, ph.primitive.p2s_map, 0.1)
        d1 = a * np.sqrt(3) / 4
        assert np.isclose(shells[0][0], d1) and np.isclose(shells[1][0], d1 * np.sqrt(8 / 3))
        assert d1 * np.sqrt(8 / 3) < ladder.shell_cutoff(shells, 2, half) < d1 * np.sqrt(11 / 3)

    prim = bulk("Cu", "fcc", a=3.6)
    ph = Phonopy(ladder._phonopy_atoms(prim), supercell_matrix=np.diag([3, 3, 3]), primitive_matrix=None)
    sc = ph.supercell
    s = SymfcAtoms(numbers=sc.numbers, cell=sc.cell, scaled_positions=sc.scaled_positions)
    u = ladder.random_displacements(12, len(sc), 0.03, 0)
    f = np.array([(lambda at: (setattr(at, "calc", EMT()), at.get_forces())[1])(
        __import__("ase").Atoms(sc.symbols, cell=sc.cell, positions=sc.positions + d, pbc=True)) for d in u])
    sf = Symfc(s, cutoff={3: 3.0, 4: 3.0}, log_level=0)
    sf.compute_basis_set(orders=[2, 3, 4])
    solver = FCSolverO2O3O4([sf.basis_set[o] for o in (2, 3, 4)]).solve(u, f)
    dense = solver.compact_fc[2]
    keys, blocks = ladder.sparse_fc4_from_symfc(sf.basis_set[4], solver.coefs[2], len(sc))
    back = np.zeros_like(dense)
    back[tuple(keys.T)] = blocks
    assert np.abs(back - dense).max() < 1e-12 * max(1.0, np.abs(dense).max())
    # sparse and dense FC4 predict the same forces
    fd = ladder.predict_forces({2: solver.compact_fc[0], 4: dense}, u[0])
    fs = ladder.predict_forces({2: solver.compact_fc[0], 4: (keys, blocks)}, u[0])
    assert np.allclose(fd, fs, atol=1e-12)
