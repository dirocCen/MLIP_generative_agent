"""DFT FC3 flow (workflows/phono3py_fd.py): plan, flow wiring, site order, and an EMT run
of the force -> FC -> kappa part with VASP replaced by EMT."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import rocksalt
from pymatgen.core import Lattice, Structure

from mga import jobs
from mga.specs import VaspPhono3pyParams
from mga.workflows import phono3py_fd as pf
from mga.workflows import vasp_flows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _zb_sic() -> Structure:
    frac = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]]
    return Structure(Lattice.cubic(4.36), ["Si"] * 4 + ["C"] * 4, frac).get_primitive_structure()


def _cu3au_au_first() -> Structure:
    """L1_2 with Au listed first, so VASP's electronegativity sort (Cu < Au) reorders sites."""
    return Structure(Lattice.cubic(3.75), ["Au", "Cu", "Cu", "Cu"], [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def test_plan_picks_the_cheaper_displacement_scheme():
    p = VaspPhono3pyParams().model_dump()
    _, settings, plan = pf.plan_for_input(_zb_sic(), p)
    assert settings["fc3_supercell"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]] and plan["n_atoms_fc3"] == 64
    # symfc minimum at the 5 A cutoff is 3 snapshots; phono3py's 'auto' rule is x4
    assert plan["symfc_min_snapshots"]["fc3_by_cutoff"]["5.0"] == 3
    assert plan["fc3_mode"] == "random" and plan["n_fc3"] == 12 < plan["n_systematic"]["fc3"]
    # FC2: smallest point-group-keeping cell (fcc-shaped, 432 atoms; the diagonal one had 512), where
    # phonopy needs 4 systematic displacements, so phono3py's random rule (symfc minimum 1 x 2) wins
    assert settings["fc2_supercell"] == [[3, 0, 3], [0, 3, 3], [0, -3, 3]] and plan["n_atoms_fc2"] == 432
    assert plan["fc2_mode"] == "random" and plan["n_fc2"] == 2 < plan["n_systematic"]["fc2"] == 4
    assert plan["learning_curve_snapshots"] == 6

    forced = pf.displacement_plan(*pf.plan_for_input(_zb_sic(), p)[:2], {**p, "displacement_mode": "systematic"})
    assert forced["fc3_mode"] == "systematic" and forced["n_fc3"] == plan["n_systematic"]["fc3"]
    assert forced["learning_curve_snapshots"] is None


def test_generate_is_reproducible_and_checks_the_plan():
    p = VaspPhono3pyParams().model_dump()
    unit, settings, plan = pf.plan_for_input(_zb_sic(), p)
    a, b = pf.generate(unit, settings, plan, p), pf.generate(unit, settings, plan, p)
    assert np.allclose(a.dataset["displacements"], b.dataset["displacements"])
    # only systematic counts can disagree with the plan (random ones follow n_fc2 / n_fc3)
    ps = {**p, "displacement_mode": "systematic"}
    plan_s = pf.displacement_plan(unit, settings, ps)
    with pytest.raises(RuntimeError, match="re-plan"):
        pf.generate(unit, settings, {**plan_s, "n_fc2": plan_s["n_fc2"] + 1}, ps)


def test_sorted_supercells_keep_their_order_in_vasp_inputs():
    from mga.workflows.vasp_flows import static_set

    p = VaspPhono3pyParams().model_dump()
    s = rocksalt("Cl", "Na", 5.64)  # Cl first; VASP sorts Na (EN 0.93) before Cl
    assert s[0].specie.symbol == "Cl"
    unit, settings, plan = pf.plan_for_input(s, p)
    ph3 = pf.generate(unit, settings, plan, p)
    cell = next(c for c in ph3.supercells_with_displacements if c is not None)
    st, perm = pf.sorted_with_permutation(cell)
    poscar = static_set(p).get_input_set(st, potcar_spec=True)["POSCAR"].structure
    assert [x.specie.symbol for x in poscar] == [x.specie.symbol for x in st]
    assert np.allclose(poscar.cart_coords, st.cart_coords)
    assert perm != list(range(len(perm)))
    f = np.arange(len(perm) * 3, dtype=float).reshape(-1, 3)
    assert np.array_equal(pf.unsort_forces(f[perm], perm), f)


def test_phono3py_flow_wiring():
    p = VaspPhono3pyParams().model_dump()
    s = _zb_sic()
    flow = vasp_flows.FLOW_BUILDERS["phono3py_fd"](s, p, "t")
    from test_vasp_flows import _check_common, _incar, _vasp_makers

    makers = dict(_vasp_makers(flow))
    assert set(makers) == {"MP GGA relax 1", "MP GGA relax 2", "run_fc3_displacements:phonon_maker",
                           "run_fc2_displacements:phonon_maker"}
    for maker in makers.values():
        incar = _incar(maker, s)
        _check_common(maker, incar)
        assert incar["ISPIN"] == 1
    names = [j["name"] for j in jobs.flow_jobs(flow)]
    assert "phono3py_displacements" in names and "phono3py_collect" in names
    coll = next(j for j in flow.jobs if getattr(j, "name", "") == "phono3py_collect")
    assert flow.output.uuid == coll.uuid
    assert coll.function_args[2]["n_fc3"] == 12  # the approved plan travels with the flow


def _emt_forces(structures):
    from ase.calculators.emt import EMT
    from pymatgen.io.ase import AseAtomsAdaptor

    out = []
    for st in structures:
        atoms = AseAtomsAdaptor.get_atoms(st)
        atoms.calc = EMT()
        out.append(atoms.get_forces().tolist())
    return out


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["auto", "systematic"])
def test_collect_with_emt_forces(mode, tmp_path, monkeypatch):
    """VASP replaced by EMT. Forces computed on the VASP-sorted supercells and mapped back
    must give the same kappa as forces computed on phonopy's own site order."""
    monkeypatch.chdir(tmp_path)
    p = {**VaspPhono3pyParams().model_dump(), "displacement_mode": mode,
         "fc3_min_width": 6.0, "fc2_min_width": 8.0, "q_mesh_length": 12.0, "fit_cutoffs": (3.0, 4.0)}
    unit, settings, plan = pf.plan_for_input(_cu3au_au_first(), p)
    gen = pf.phono3py_displacements.original(unit, settings, plan, p)
    meta = gen["meta"]
    out = pf.collect(unit, settings, plan, p, _emt_forces(gen["fc3_structures"]), _emt_forces(gen["fc2_structures"]), meta)
    assert out["ok"], out.get("traceback")
    assert not out["has_imaginary_modes"] and out["max_force_drift"] < 1e-6
    assert set(out["files"]) == {"phono3py_params.yaml", "fc2.hdf5", "fc3.hdf5"} and (tmp_path / "fc3.hdf5").exists()
    k = out["kappa_tot_avg"][0]
    assert 0.1 < k < 1000
    expected = {"symfc_cutoff_3", "symfc_cutoff_4"}
    if plan["fc3_mode"] == "systematic":
        assert out["final_variant"] == "traditional_full" and set(out["kappa"]) == expected | {"traditional_full"}
    else:
        assert out["final_variant"] == "symfc_cutoff_4" and set(out["kappa"]) == expected | {"learning_curve"}
        assert out["convergence"]["learning_curve_rel_change"] is not None
    assert out["convergence"]["cutoff_rel_change"] is not None
    a = out["arrays"]
    assert a["mode_kappa_tot_rta"].shape[0] == 1 and int(a["mode_weights"].sum()) == int(np.prod(settings["q_mesh"]))

    # reference: phonopy's own site order, no permutation. Compare force constants, not kappa:
    # on this toy 3x3x3 mesh kappa moves by % for 1e-8 THz frequency noise (degenerate
    # modes in the tetrahedron integration), while the mapping itself is exact.
    import h5py
    from pymatgen.io.phonopy import get_pmg_structure

    ph3 = pf.generate(unit, settings, plan, p)
    raw3 = [get_pmg_structure(c) for c in ph3.supercells_with_displacements if c is not None]
    raw2 = [get_pmg_structure(c) for c in ph3.phonon_supercells_with_displacements]
    ident = {**meta, "perm_fc3": list(range(plan["n_atoms_fc3"])), "perm_fc2": list(range(plan["n_atoms_fc2"]))}
    (tmp_path / "ref").mkdir()
    monkeypatch.chdir(tmp_path / "ref")
    ref = pf.collect(unit, settings, plan, p, _emt_forces(raw3), _emt_forces(raw2), ident)
    assert ref["ok"]
    for name in ("fc2.hdf5", "fc3.hdf5"):
        with h5py.File(tmp_path / name) as x, h5py.File(tmp_path / "ref" / name) as y:
            keys = set(x) - {"version"}
            assert keys == set(y) - {"version"} and any(k.startswith(("fc", "force")) for k in keys)
            for k in keys:
                assert np.allclose(x[k][:], y[k][:], atol=1e-10), (name, k)
