"""Every VASP step of every flow gets the project's PBE inputs (plan §2.3).

The flows are only built, never run: each maker's input-set generator is asked for
the INCAR it would write for NaCl.
"""

from __future__ import annotations

import pytest
from conftest import rocksalt

from mga import jobs
from mga.specs import VaspConvergenceParams, VaspPhononParams, VaspRelaxStaticParams
from mga.workflows import vasp_flows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")  # BadInputSetWarning on the POTCAR override


@pytest.fixture(scope="module")
def nacl():
    return rocksalt("Na", "Cl", 5.64).get_primitive_structure()


def _vasp_makers(flow):
    """(job name, maker) for every job that runs VASP, nested flows and the
    displacement maker passed to ``run_phonon_displacements`` included."""
    from jobflow import Flow, Maker

    def is_vasp(obj):  # OutputReference answers hasattr() for any name, so check the type first
        return isinstance(obj, Maker) and hasattr(obj, "input_set_generator")

    out = []
    for item in flow.jobs:
        if isinstance(item, Flow):
            out.extend(_vasp_makers(item))
            continue
        owner = getattr(item.function, "__self__", None)
        if is_vasp(owner):
            out.append((item.name, owner))
        out.extend((f"{item.name}:{k}", v) for k, v in item.function_kwargs.items() if is_vasp(v))
    return out


def _incar(maker, structure):
    return maker.input_set_generator.get_input_set(structure, potcar_spec=True).incar


def _check_common(maker, incar, encut=520.0):
    assert maker.input_set_generator.user_potcar_functional == "PBE_54"
    assert "GGA" not in incar  # PBE, not atomate2's PBEsol default (GGA = PS)
    assert incar["ENCUT"] == encut and incar["EDIFF"] == 1e-8 and incar["LREAL"] is False
    assert incar["LWAVE"] is False and incar["LCHARG"] is False


def test_upstream_defaults_are_what_we_override(nacl):
    """If this fails, atomate2 changed its defaults; re-check vasp_flows' overrides."""
    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker

    incar = _incar(PhononDisplacementMaker(), nacl)
    assert incar["GGA"].upper() == "PS" and incar["ENCUT"] == 700.0


@pytest.mark.parametrize("double", [True, False])
def test_relax_static_flow(nacl, double):
    p = VaspRelaxStaticParams(double_relax=double).model_dump()
    flow = vasp_flows.relax_static_flow(nacl, p, "t")
    listed = jobs.flow_jobs(flow)  # recurses into the DoubleRelaxMaker sub-flow
    assert len(listed) == (3 if double else 2) and len({j["uuid"] for j in listed}) == len(listed)
    makers = _vasp_makers(flow)
    assert len(makers) == len(listed)
    for name, maker in makers:
        incar = _incar(maker, nacl)
        _check_common(maker, incar)
        assert ("EDIFFG" in incar) == ("relax" in name)


def test_phonon_flow_uses_one_spin_treatment(nacl):
    p = VaspPhononParams().model_dump()
    flow = vasp_flows.phonon_fd_flow(nacl, p, "t")
    makers = dict(_vasp_makers(flow))
    assert set(makers) == {"MP GGA relax 1", "MP GGA relax 2", "MP GGA static", "run_phonon_displacements:phonon_maker"}
    for maker in makers.values():
        incar = _incar(maker, nacl)
        _check_common(maker, incar)
        assert incar["ISPIN"] == 1 and "MAGMOM" not in incar
    names = [j["name"] for j in jobs.flow_jobs(flow)]
    assert "get_supercell_size" in names and "generate_frequencies_eigenvectors" in names
    gen = next(j for j in flow.jobs if getattr(j, "name", "") == "generate_phonon_displacements")
    assert gen.function_kwargs["displacement"] == 0.01 and gen.function_kwargs["symprec"] == 1e-3
    fc = next(j for j in flow.jobs if getattr(j, "name", "") == "generate_frequencies_eigenvectors")
    assert fc.function_kwargs["born"] is None and fc.function_kwargs["store_force_constants"] is True


def test_convergence_flow_scans_one_parameter_at_a_time(nacl):
    p = VaspConvergenceParams().model_dump()
    flow = vasp_flows.convergence_flow(nacl, p, "t")
    makers = _vasp_makers(flow)
    assert len(makers) == len(p["encut_values"]) + len(p["reciprocal_density_values"])
    encuts = [_incar(m, nacl)["ENCUT"] for name, m in makers if "encut=" in name]
    assert encuts == list(p["encut_values"])
    for name, m in makers:
        if "kd=" in name:
            _check_common(m, _incar(m, nacl))
            assert m.input_set_generator.user_kpoints_settings["reciprocal_density"] == int(name.split("kd=")[1])


def test_exec_incar_and_overrides_reach_the_incar(nacl):
    p = {**VaspRelaxStaticParams(encut=600.0).model_dump(), "exec_incar": {"NCORE": 4, "KPAR": 2}}
    incar = vasp_flows.static_set(p).get_input_set(nacl, potcar_spec=True).incar
    assert incar["ENCUT"] == 600.0 and incar["NCORE"] == 4 and incar["KPAR"] == 2
