"""kappa_SRME metrics (vs the reference file) and the ported protocol on EMT aluminium."""

from __future__ import annotations

import numpy as np
import pytest

from mga.analysis.kappa_metrics import (
    KAPPA_ERROR_MAX,
    compare_models,
    compare_to_reference,
    reference_table,
    srd,
    srme,
)


def _voigt(avg: np.ndarray) -> np.ndarray:
    """(...,) isotropic average -> (..., 6) Voigt with that diagonal."""
    avg = np.asarray(avg, float)
    return np.concatenate([np.repeat(avg[..., None], 3, axis=-1), np.zeros((*avg.shape, 3))], axis=-1)


def _pred_from_reference(row: dict, scale: float = 1.0) -> tuple[dict, dict]:
    arrays = {
        "kappa_tot_rta": _voigt(np.asarray(row["kappa_TOT_ave"]) * scale),
        "mode_kappa_tot_rta": _voigt(np.asarray(row["mode_kappa_TOT_ave"]) * scale),
        "mode_weights": np.asarray(row["weights"], float),
        "q_points": np.asarray(row["qpoints"], float),
        "frequencies": np.asarray(row["frequencies"], float),
    }
    data = {"ok": True, "has_imaginary_modes": False, "conductivity_skipped": None,
            "relax": {"initial_spg_num": row["symm.no"], "final_spg_num": row["symm.no"]}}
    return data, arrays


def test_reference_table_shape():
    table = reference_table()
    assert len(table) == 103
    row = table["mp-1958"] if "mp-1958" in table else next(iter(table.values()))
    nq = len(row["weights"])
    assert np.asarray(row["mode_kappa_TOT_ave"]).shape[:2] == (1, nq)


def test_srd_srme_basics():
    assert srd(10.0, 10.0) == 0.0
    assert srd(None, 10.0) == -KAPPA_ERROR_MAX and srd(-1.0, 10.0) == -KAPPA_ERROR_MAX
    assert srd(30.0, 10.0) == pytest.approx(1.0)
    m = np.ones((1, 4, 3))
    w = np.ones(4)
    assert srme(m, w, np.array([3.0]), m, np.array([3.0]))[0] == 0.0
    assert srme(m * np.nan, w, np.array([3.0]), m, np.array([3.0]))[0] == KAPPA_ERROR_MAX


def test_perfect_prediction_scores_zero_and_censoring():
    mp_id, row = next(iter(reference_table().items()))
    data, arrays = _pred_from_reference(row)
    res = compare_to_reference(data, arrays, mp_id)
    assert res["srd"] == pytest.approx(0.0, abs=1e-12) and res["srme"] == pytest.approx(0.0, abs=1e-12)
    assert res["censored"] is None

    data2, arrays2 = _pred_from_reference(row, scale=2.0)
    res2 = compare_to_reference(data2, arrays2, mp_id)
    assert res2["srd"] == pytest.approx(2 * (2 - 1) / 3) and res2["srme"] > 0

    imag = {**data, "has_imaginary_modes": True}
    assert compare_to_reference(imag, arrays, mp_id)["srme"] == KAPPA_ERROR_MAX
    broken = {**data, "relax": {"initial_spg_num": 225, "final_spg_num": 166}}
    assert compare_to_reference(broken, arrays, mp_id)["censored"] == "symmetry_changed"


def test_wrong_q_mesh_is_flagged_not_crashing():
    mp_id, row = next(iter(reference_table().items()))
    data, arrays = _pred_from_reference(row)
    arrays["mode_kappa_tot_rta"] = arrays["mode_kappa_tot_rta"][:, :-1]
    arrays["mode_weights"] = arrays["mode_weights"][:-1]
    res = compare_to_reference(data, arrays, mp_id)
    assert res["srme"] == KAPPA_ERROR_MAX and "shapes differ" in res["censored"]


def test_compare_models_needs_same_q_points():
    rows = list(reference_table().values())
    a = _pred_from_reference(rows[0])
    b = _pred_from_reference(rows[0], scale=1.5)
    res = compare_models(a, b)
    assert res["mode_comparable"] and res["srme"] > 0 and res["freq_mae_thz"] == 0.0
    other = next(r for r in rows if len(r["weights"]) != len(rows[0]["weights"]))
    res2 = compare_models(a, _pred_from_reference(other))
    assert res2["mode_comparable"] is False and res2["srme"] is None and res2["srd"] is not None


@pytest.mark.slow
def test_protocol_on_emt_aluminium():
    """Full relax -> FC2 -> FC3 -> Wigner kappa with phono3py 3.30.0 (MS-SMM19 output keys)."""
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from mga.specs import MLIPPhononKappaParams, MLIPRelaxParams
    from mga.workflows.kappa_protocol import run_structure

    rp = MLIPRelaxParams().model_dump()
    kp = {**MLIPPhononKappaParams().model_dump(), "fc3_min_width": 6.0, "fc2_min_width": 8.0, "q_mesh_length": 12.0}
    out = run_structure(bulk("Al", "fcc", a=4.05), EMT(), rp, kp, None)
    assert out["ok"], out.get("traceback")
    assert out["relax"]["final_spg_num"] == 225 and not out["has_imaginary_modes"]
    assert out["settings"]["fc3_supercell"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert out["checks"]["mode_sum_max_abs_dev"] < 1e-8
    k = out["kappa_tot_avg"][0]
    assert 1.0 < k < 100.0 and out["kappa_c_avg"][0] < 0.1 * k
    a = {k: np.asarray(v) for k, v in out["arrays"].items()}
    nq = a["mode_weights"].shape[0]
    assert a["mode_kappa_tot_rta"].shape == (1, nq, 3, 6) and int(a["mode_weights"].sum()) == 125

    # the same arrays compared with themselves: zero disagreement
    res = compare_models((out, a), (out, a))
    assert res["srd"] == 0.0 and res["srme"] == 0.0


def test_fixed_settings_are_used_as_given():
    from ase.build import bulk

    from mga.specs import MLIPPhononKappaParams
    from mga.workflows.kappa_protocol import choose_settings, length_rule_settings

    kp = MLIPPhononKappaParams().model_dump()
    atoms = bulk("Al", "fcc", a=4.05)
    unit, fixed = length_rule_settings(atoms, kp)
    assert fixed["source"] == "length_rule_v2" and len(unit) == 4  # conventional cell
    strained = unit.copy()
    strained.set_cell(unit.cell * 1.02, scale_atoms=True)
    _, used = choose_settings(strained, kp, fixed)
    assert used["fc3_supercell"] == fixed["fc3_supercell"] and used["q_mesh"] == fixed["q_mesh"]


def test_minimal_supercell_never_larger_than_diagonal_and_keeps_symmetry():
    from ase.build import bulk

    from mga.workflows.kappa_protocol import (
        _perpendicular_widths,
        diagonal_supercell,
        lattice_width,
        minimal_supercell,
        point_group_rotations,
    )

    # zincblende conventional cell: 8 A -> 128 atoms (bcc-shaped), not the 3x3x3 = 216 of v1
    zb = bulk("BN", "zincblende", a=3.625, cubic=True)
    rots = point_group_rotations(zb, 1e-5)
    m = np.asarray(minimal_supercell(zb.cell, 8.0, rots))
    assert round(np.linalg.det(m)) * len(zb) == 128 and lattice_width(m @ zb.cell) >= 8.0
    minv = np.linalg.inv(m)
    for r in rots:  # the supercell lattice is invariant under the point group
        t = m @ r.T @ minv
        assert np.allclose(t, np.round(t), atol=1e-8)
    # hexagonal cell with c just below the width: any-shape FC2 cell beats diagonal 7x7x2
    hexa = np.array([[2.54, 0, 0], [-1.27, 2.54 * np.sqrt(3) / 2, 0], [0, 0, 12.56]])
    d = diagonal_supercell(hexa, 14.0)
    assert np.isclose(lattice_width(d @ hexa), _perpendicular_widths(d @ hexa).min())
    m2 = np.asarray(minimal_supercell(hexa, 14.0))
    assert lattice_width(m2 @ hexa) >= 14.0 and np.linalg.det(m2) < 0.6 * np.linalg.det(d)
    # cubic at 14 A: identical to the diagonal rule
    assert np.array_equal(np.abs(minimal_supercell(zb.cell, 14.0, rots)), diagonal_supercell(zb.cell, 14.0))


def test_length_rule_q_mesh_respects_body_centred_symmetry():
    """I-centred cells: per-axis length rule gives (n, n, m) on the primitive basis, which the
    4-fold axis does not leave invariant; phono3py rejects it (S1-main b2, I-42d B2AsP)."""
    from ase.build import bulk
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.grid_points import length2mesh

    from mga.specs import MLIPPhononKappaParams
    from mga.workflows.kappa_protocol import _primitive_cell, length_rule_settings, symmetric_q_mesh

    kp = MLIPPhononKappaParams().model_dump()
    bct = bulk("In", "bct", a=3.25, c=4.95)
    unit, fixed = length_rule_settings(bct, kp)
    naive = [int(x) for x in length2mesh(kp["q_mesh_length"], _primitive_cell(unit, kp["symprec"]).cell)]
    assert len(set(naive)) > 1 and len(set(fixed["q_mesh"])) == 1 and max(fixed["q_mesh"]) == max(naive)
    ph3 = Phono3py(PhonopyAtoms(unit.get_chemical_symbols(), cell=np.asarray(unit.cell), positions=unit.positions),
                   supercell_matrix=np.eye(3, dtype=int), primitive_matrix="auto", symprec=kp["symprec"], log_level=0)
    ph3.mesh_numbers = fixed["q_mesh"]  # raises "Grid symmetry is broken" for the naive mesh
    # an already symmetric mesh is left alone (hexagonal)
    hexa = bulk("Mg", "hcp", a=3.21, c=5.21)
    assert symmetric_q_mesh(kp["q_mesh_length"], hexa, kp["symprec"]) == [
        int(x) for x in length2mesh(kp["q_mesh_length"], _primitive_cell(hexa, kp["symprec"]).cell)]
