"""atomate2 VASP flows with MP-compatible PBE inputs (DESIGN §1.4, plan §2.3).

atomate2's own VASP defaults are PBEsol, and pymatgen's MP sets default to the
legacy ``PBE`` POTCAR set; both are overridden here, in one place, so every VASP
calculation of the project uses MPRelaxSet / MPStaticSet + ``PBE_54``.
"""

from __future__ import annotations

from typing import Any

from pymatgen.core import Structure

# Keep outputs small: home quota on dirona is 200 GB (plan §2.1).
_NO_BIG_FILES = {"LWAVE": False, "LCHARG": False, "LAECHG": False, "LVHAR": False, "LORBIT": None}


def incar_overrides(p: dict[str, Any], *, static: bool) -> dict[str, Any]:
    """Result-affecting settings from the params, plus ``p["exec_incar"]`` (NCORE / KPAR,
    execution only, set by the dft tools from ``VaspResources``)."""
    incar = {
        "ENCUT": p["encut"],
        "EDIFF": p["ediff"],
        "PREC": p["prec"],
        "LREAL": p["lreal"],
        "ISMEAR": p["ismear"],
        "SIGMA": p["sigma"],
        "ALGO": p["algo"],
        **_NO_BIG_FILES,
    }
    if not static and "ediffg" in p:
        incar["EDIFFG"] = p["ediffg"]
    incar.update(p.get("exec_incar") or {})
    return incar


def relax_set(p: dict[str, Any], extra_incar: dict[str, Any] | None = None):
    from pymatgen.io.vasp.sets import MPRelaxSet

    return MPRelaxSet(
        user_incar_settings={**incar_overrides(p, static=False), **(extra_incar or {})},
        user_kpoints_settings={"reciprocal_density": p["reciprocal_density"]},
        user_potcar_functional=p["potcar_functional"],
        force_gamma=True,
        inherit_incar=False,
    )


def static_set(p: dict[str, Any], extra_incar: dict[str, Any] | None = None):
    from pymatgen.io.vasp.sets import MPStaticSet

    return MPStaticSet(
        user_incar_settings={**incar_overrides(p, static=True), **(extra_incar or {})},
        user_kpoints_settings={"reciprocal_density": p["reciprocal_density"]},
        user_potcar_functional=p["potcar_functional"],
        reciprocal_density=p["reciprocal_density"],
        force_gamma=True,
        inherit_incar=False,
    )


def mp_default_sets():
    """Plain MP sets, used by preview to list what we changed."""
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

    return MPRelaxSet(), MPStaticSet()


def _relax_maker(p: dict[str, Any], extra_incar: dict[str, Any] | None = None):
    from atomate2.vasp.flows.core import DoubleRelaxMaker
    from atomate2.vasp.jobs.mp import MPGGARelaxMaker

    single = MPGGARelaxMaker(input_set_generator=relax_set(p, extra_incar))
    return DoubleRelaxMaker.from_relax_maker(single) if p.get("double_relax", True) else single


def _static_maker(p: dict[str, Any], extra_incar: dict[str, Any] | None = None, name: str = "MP GGA static"):
    from atomate2.vasp.jobs.mp import MPGGAStaticMaker

    return MPGGAStaticMaker(name=name, input_set_generator=static_set(p, extra_incar))


def relax_static_flow(structure: Structure, p: dict[str, Any], name: str):
    from jobflow import Flow

    relax = _relax_maker(p).make(structure)
    static = _static_maker(p).make(relax.output.structure, prev_dir=relax.output.dir_name)
    return Flow([relax, static], output=static.output, name=name)


def phonon_fd_flow(structure: Structure, p: dict[str, Any], name: str):
    from atomate2.vasp.flows.phonons import PhononMaker
    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker

    spin = {"ISPIN": p["ispin"]}  # same spin treatment in every step of the flow
    disp_maker = PhononDisplacementMaker(input_set_generator=static_set(p, spin))
    maker = PhononMaker(
        name=name,
        displacement=p["displacement"],
        min_length=p["supercell_min_length"],
        symprec=p["symprec"],
        bulk_relax_maker=_relax_maker(p, spin),
        static_energy_maker=_static_maker(p, spin),
        born_maker=None,  # kappa_SRME reference is noNAC (plan §9.1)
        phonon_displacement_maker=disp_maker,
        store_force_constants=True,
        create_thermal_displacements=False,
    )
    return maker.make(structure)


def convergence_flow(structure: Structure, p: dict[str, Any], name: str):
    from jobflow import Flow

    jobs = []
    for encut in p["encut_values"]:
        q = {**p, "encut": encut}
        jobs.append(_static_maker(q, name=f"conv encut={encut:g}").make(structure))
    for kd in p["reciprocal_density_values"]:
        q = {**p, "reciprocal_density": kd}
        jobs.append(_static_maker(q, name=f"conv kd={kd}").make(structure))
    return Flow(jobs, name=name)


def phono3py_fd_flow(structure: Structure, p: dict[str, Any], name: str):
    from mga.workflows.phono3py_fd import phono3py_fd_flow as build

    return build(structure, p, name)


FLOW_BUILDERS = {
    "relax_static": relax_static_flow,
    "phonon_fd": phonon_fd_flow,
    "phono3py_fd": phono3py_fd_flow,
    "convergence": convergence_flow,
}
