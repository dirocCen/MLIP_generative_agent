"""DFT FC2/FC3 by finite displacements and phono3py kappa (plan §9.2, DESIGN §3 L4).

Flow (one structure):

1. VASP (double) relaxation of the conventional cell.
2. ``phono3py_displacements``: phono3py builds the displaced supercells on the
   relaxed cell, exactly as planned at request time.
3. Two ``run_phonon_displacements`` (atomate2): one VASP static per supercell,
   for FC3 and for FC2.
4. ``phono3py_collect``: FC2, stability gate, FC3 fitted several ways, Wigner kappa
   with the kappa_SRME settings (``kappa_protocol.conductivity``).

Displacements
-------------
Per order, the cheaper of:

- *systematic* (phono3py default; one or two atoms displaced by 0.03 Angstrom),
  fitted by the traditional solver;
- *random* (every atom displaced by 0.03 Angstrom in a random direction), fitted
  by symfc. The number of snapshots follows phono3py 3.30.0's own ``"auto"`` rule:
  the symfc minimum (basis size / 3N) x 4 for FC3, at the largest fit cutoff, and
  x 2 for FC2.

For the low-symmetry cells MatterGen produces, random + symfc needs 50-100x fewer
supercells than systematic (plan §9.2). The plan (mode and counts) is computed at
request time on the input structure, so the human approves exact numbers; at run
time the same plan is applied to the VASP-relaxed cell and checked.

Convergence checks recorded with every result
---------------------------------------------
- FC3 cutoff: kappa fitted at every ``fit_cutoffs`` value (relative change).
- Snapshot count (random mode): kappa from the first ``learning_curve_fraction``
  of the snapshots at the largest cutoff.

VASP input sets sort sites by electronegativity; the displaced supercells are
handed over already sorted, and forces are mapped back with the stored permutation.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from jobflow import job

from mga.workflows import kappa_protocol as kp


# --- planning (pure functions; also used locally by the dft tools) ----------------
def _phono3py(unit_cell, settings: dict[str, Any], symprec: float):
    """Phono3py without displacements. ``unit_cell``: ase Atoms or pymatgen Structure."""
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    if hasattr(unit_cell, "get_chemical_symbols"):
        symbols, cell, pos = unit_cell.get_chemical_symbols(), np.asarray(unit_cell.cell), unit_cell.positions
    else:
        symbols = [s.specie.symbol for s in unit_cell]
        cell, pos = unit_cell.lattice.matrix, unit_cell.cart_coords
    ph3 = Phono3py(
        unitcell=PhonopyAtoms(symbols, cell=np.asarray(cell), positions=np.asarray(pos)),
        supercell_matrix=settings["fc3_supercell"],
        phonon_supercell_matrix=settings["fc2_supercell"],
        primitive_matrix="auto",
        symprec=symprec,
    )
    ph3.mesh_numbers = settings["q_mesh"]
    return ph3


def _symfc_min(supercell, symmetry, order: int, cutoff: float | None = None) -> int:
    """Minimum number of random snapshots: ceil(symfc basis size / 3N)."""
    from phonopy.interface.symfc import SymfcFCSolver

    options = {"cutoff": {order: cutoff}} if cutoff is not None else None
    return int(SymfcFCSolver(supercell, symmetry=symmetry, options=options).estimate_numbers_of_supercells(orders=[order])[order])


def _n_systematic_fc3(ph3, distance: float) -> int:
    ph3.generate_displacements(distance=distance)
    ds = ph3.dataset
    return sum(1 + sum(1 for d2 in d1["second_atoms"] if d2.get("included", True)) for d1 in ds["first_atoms"])


def _n_systematic_fc2(ph3, distance: float) -> int:
    ph3.generate_fc2_displacements(distance=distance)
    return len(ph3.phonon_dataset["first_atoms"])


def displacement_plan(unit_cell, settings: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """Mode and number of displaced supercells per order (deterministic)."""
    ph3 = _phono3py(unit_cell, settings, p["symprec"])
    cmax = max(p["fit_cutoffs"])
    min3 = {str(c): _symfc_min(ph3.supercell, ph3.symmetry, 3, c) for c in p["fit_cutoffs"]}
    min2 = _symfc_min(ph3.phonon_supercell, ph3.phonon_supercell_symmetry, 2)
    rnd = {
        "fc3": max(2, math.ceil(min3[str(cmax)] * p["fc3_snapshot_factor"])),
        "fc2": max(2, math.ceil(min2 * p["fc2_snapshot_factor"])),
    }
    sysn = {"fc3": _n_systematic_fc3(ph3, p["displacement"]), "fc2": _n_systematic_fc2(ph3, p["displacement"])}
    plan: dict[str, Any] = {
        "space_group": ph3.symmetry.dataset.number if hasattr(ph3.symmetry.dataset, "number") else ph3.symmetry.dataset["number"],
        "n_atoms_unit": len(ph3.unitcell),
        "n_atoms_primitive": len(ph3.primitive),
        "n_atoms_fc3": len(ph3.supercell),
        "n_atoms_fc2": len(ph3.phonon_supercell),
        "symfc_min_snapshots": {"fc3_by_cutoff": min3, "fc2": min2},
        "n_systematic": sysn,
        "n_random": rnd,
    }
    for order in ("fc3", "fc2"):
        mode = p["displacement_mode"]
        if mode == "auto":
            mode = "random" if rnd[order] < sysn[order] else "systematic"
        plan[f"{order}_mode"] = mode
        plan[f"n_{order}"] = rnd[order] if mode == "random" else sysn[order]
    plan["learning_curve_snapshots"] = _learning_curve_n(plan, p)
    return plan


def _learning_curve_n(plan: dict[str, Any], p: dict[str, Any]) -> int | None:
    """Snapshots in the learning-curve fit, or None if that fit would be underdetermined."""
    if plan["fc3_mode"] != "random":
        return None
    n = int(round(plan["n_fc3"] * p["learning_curve_fraction"]))
    need = plan["symfc_min_snapshots"]["fc3_by_cutoff"][str(max(p["fit_cutoffs"]))]
    return n if n >= need else None


def generate(unit_cell, settings: dict[str, Any], plan: dict[str, Any], p: dict[str, Any]):
    """Phono3py with the planned displacements (same seed -> same displacements)."""
    ph3 = _phono3py(unit_cell, settings, p["symprec"])
    if plan["fc3_mode"] == "random":
        ph3.generate_displacements(distance=p["displacement"], number_of_snapshots=plan["n_fc3"], random_seed=p["random_seed"])
    else:
        ph3.generate_displacements(distance=p["displacement"])
    if plan["fc2_mode"] == "random":
        ph3.generate_fc2_displacements(distance=p["displacement"], number_of_snapshots=plan["n_fc2"], random_seed=p["random_seed"] + 1)
    else:
        ph3.generate_fc2_displacements(distance=p["displacement"])
    n3 = sum(s is not None for s in ph3.supercells_with_displacements)
    n2 = len(ph3.phonon_supercells_with_displacements)
    if (n3, n2) != (plan["n_fc3"], plan["n_fc2"]):
        raise RuntimeError(
            f"relaxed cell gives {n3} FC3 / {n2} FC2 supercells, the approved plan has {plan['n_fc3']} / {plan['n_fc2']} "
            f"(space group {plan['space_group']} at request time). Symmetry changed in the VASP relaxation; "
            "re-plan and request a new approval."
        )
    return ph3


def sorted_with_permutation(cell) -> tuple[Any, list[int]]:
    """phonopy supercell -> (pymatgen Structure in VASP's site order, perm).

    ``sorted[j] = original[perm[j]]``, so forces map back as ``f_orig[perm] = f_sorted``.
    Same stable sort as ``pymatgen.io.vasp.sets.VaspInputSet`` (``get_sorted_structure``).
    """
    from pymatgen.core import Structure
    from pymatgen.io.phonopy import get_pmg_structure

    s = get_pmg_structure(cell)
    perm = sorted(range(len(s)), key=lambda i: s[i])
    return Structure.from_sites([s[i] for i in perm]), perm


def unsort_forces(forces_sorted: list, perm: list[int]) -> np.ndarray:
    f = np.asarray(forces_sorted, dtype=float)
    out = np.empty_like(f)
    out[np.asarray(perm)] = f
    return out


# --- jobs ---------------------------------------------------------------------------
@job(arrays=["fc3_structures", "fc2_structures"])
def phono3py_displacements(structure, settings: dict[str, Any], plan: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    ph3 = generate(structure, settings, plan, p)
    fc3_index = [i for i, s in enumerate(ph3.supercells_with_displacements) if s is not None]
    fc3, perm3 = [], None
    for i in fc3_index:
        st, perm3 = sorted_with_permutation(ph3.supercells_with_displacements[i])
        fc3.append(st)
    fc2, perm2 = [], None
    for cell in ph3.phonon_supercells_with_displacements:
        st, perm2 = sorted_with_permutation(cell)
        fc2.append(st)
    return {
        "fc3_structures": fc3,
        "fc2_structures": fc2,
        "meta": {"fc3_index": fc3_index, "perm_fc3": perm3, "perm_fc2": perm2,
                 "n_fc3_total": len(ph3.supercells_with_displacements)},
    }


def _fc3_variants(plan: dict[str, Any], p: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(name, fit) pairs; the last one is the reported (final) kappa."""
    cuts = sorted(p["fit_cutoffs"])
    out = [(f"symfc_cutoff_{c:g}", {"fc_calculator": "symfc", "fc_calculator_options": f"cutoff = {c}"}) for c in cuts]
    if plan["fc3_mode"] == "systematic":
        out.append(("traditional_full", {"fc_calculator": None, "symmetrize_fc3r": True}))
    elif plan.get("learning_curve_snapshots"):  # random: the largest cutoff (already last) is final
        out.insert(0, ("learning_curve", {"fc_calculator": "symfc", "fc_calculator_options": f"cutoff = {cuts[-1]}",
                                          "n_snapshots": plan["learning_curve_snapshots"]}))
    return out


def _rel(a: float, b: float) -> float | None:
    return None if not b else float(abs(a - b) / abs(b))


def collect(structure, settings: dict[str, Any], plan: dict[str, Any], p: dict[str, Any],
            fc3_forces: list, fc2_forces: list, meta: dict[str, Any], write_files: bool = True) -> dict[str, Any]:
    """Forces -> FC2 -> stability -> FC3 variants -> kappa. Failures are returned, not raised."""
    out: dict[str, Any] = {"ok": False, "stage": "forces", "error": None, "plan": plan, "settings": settings}
    arrays: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    try:
        ph3 = generate(structure, settings, plan, p)
        n3 = len(ph3.supercell)
        f3 = np.zeros((meta["n_fc3_total"], n3, 3))
        for k, i in enumerate(meta["fc3_index"]):
            f3[i] = unsort_forces(fc3_forces[k], meta["perm_fc3"])
        f2 = np.array([unsort_forces(f, meta["perm_fc2"]) for f in fc2_forces])
        out["max_force_drift"] = float(max(np.abs(f3[meta["fc3_index"]].sum(axis=1)).max(), np.abs(f2.sum(axis=1)).max()))
        ph3.forces = f3
        ph3.phonon_forces = f2

        out["stage"] = "fc2"
        if plan["fc2_mode"] == "random":
            ph3.produce_fc2(fc_calculator="symfc")
        else:
            ph3.produce_fc2(symmetrize_fc2=True)
        ph3.init_phph_interaction(symmetrize_fc3q=False)
        ph3.run_phonon_solver()
        freqs, _, _ = ph3.get_phonon_data()
        out["has_imaginary_modes"] = kp.has_imaginary_modes(freqs, p["imaginary_threshold_thz"])
        out["min_frequency_thz"] = float(np.nanmin(freqs))
        arrays["frequencies_bz_grid"] = freqs[ph3.grid.grg2bzg]
        if out["has_imaginary_modes"]:
            out["conductivity_skipped"] = "imaginary_modes"
            out["ok"] = True
            return out

        out["stage"] = "fc3"
        full = {"displacements": np.array(ph3.dataset["displacements"]), "forces": f3} if plan["fc3_mode"] == "random" else None
        kappa: dict[str, Any] = {}
        variants = _fc3_variants(plan, p)
        for name, fit in variants:
            fit = dict(fit)
            n_sub = fit.pop("n_snapshots", None)
            if full is not None:
                n = n_sub or len(full["forces"])
                ph3.dataset = {"displacements": full["displacements"][:n], "forces": full["forces"][:n]}
            ph3.produce_fc3(**fit)
            k_arrays, check = kp.conductivity(ph3, p)
            kappa[name] = {
                "kappa_tot_avg": kp.kappa_avg(k_arrays["kappa_tot_rta"]).tolist(),
                "kappa_p_avg": kp.kappa_avg(k_arrays["kappa_p_rta"]).tolist(),
                "kappa_tot_rta_voigt": np.asarray(k_arrays["kappa_tot_rta"]).tolist(),
                **check,
            }
            final_arrays = k_arrays
        final = variants[-1][0]
        arrays.update(final_arrays)
        out["kappa"] = kappa
        out["final_variant"] = final
        out["kappa_tot_avg"] = kappa[final]["kappa_tot_avg"]
        out["temperatures"] = list(p["temperatures"])
        out["convergence"] = _convergence(kappa, final, p)
        if write_files:
            out["files"] = _write_files(ph3)
        out["ok"], out["stage"] = True, "done"
        out["conductivity_skipped"] = None
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc(limit=5)
    finally:
        out["wall_time_s"] = round(time.perf_counter() - t0, 1)
        out["arrays"] = arrays
    return out


def _convergence(kappa: dict[str, Any], final: str, p: dict[str, Any]) -> dict[str, Any]:
    """Largest relative change of kappa_tot_avg (over temperatures) against the final fit."""
    ref = kappa[final]["kappa_tot_avg"]

    def worst(name: str) -> float | None:
        if name not in kappa:
            return None
        vals = [_rel(a, b) for a, b in zip(kappa[name]["kappa_tot_avg"], ref, strict=True)]
        return None if None in vals else max(vals)

    first = f"symfc_cutoff_{min(p['fit_cutoffs']):g}"
    return {
        "cutoff_rel_change": worst(first) if first != final else None,
        "cutoff_compared": [first, final],
        "learning_curve_rel_change": worst("learning_curve"),
    }


def _write_files(ph3) -> list[str]:
    """phono3py_params.yaml + fc2/fc3 (final fit) in the job directory, for reuse and audit."""
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

    ph3.save("phono3py_params.yaml")
    write_fc2_to_hdf5(ph3.fc2, filename="fc2.hdf5", p2s_map=ph3.phonon_primitive.p2s_map)
    write_fc3_to_hdf5(ph3.fc3, filename="fc3.hdf5", p2s_map=ph3.primitive.p2s_map)
    return ["phono3py_params.yaml", "fc2.hdf5", "fc3.hdf5"]


@job(arrays="arrays")
def phono3py_collect(structure, settings, plan, p, fc3_run: dict[str, Any], fc2_run: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    out = collect(structure, settings, plan, p, fc3_run["forces"], fc2_run["forces"], meta)
    out["vasp_dirs"] = {"fc3": fc3_run.get("dirs"), "fc2": fc2_run.get("dirs")}
    out["vasp_uuids"] = {"fc3": fc3_run.get("uuids"), "fc2": fc2_run.get("uuids")}
    return out


# --- flow builder -------------------------------------------------------------------
def plan_for_input(structure, p: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """(conventional unit cell as pymatgen Structure, supercells + q-mesh, displacement plan).

    Supercells and q-mesh come from ``length_rule_settings`` with the same widths as the
    MLIP kappa tool, so DFT and MLIP kappa share q-points (mode-resolved comparison).
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    unit_atoms, settings = kp.length_rule_settings(AseAtomsAdaptor.get_atoms(structure), p)
    unit = AseAtomsAdaptor.get_structure(unit_atoms)
    return unit, settings, displacement_plan(unit, settings, p)


def phono3py_fd_flow(structure, p: dict[str, Any], name: str):
    from atomate2.common.jobs.phonons import run_phonon_displacements
    from jobflow import Flow

    from mga.workflows.vasp_flows import _relax_maker, _static_maker

    unit, settings, plan = plan_for_input(structure, p)
    spin = {"ISPIN": p["ispin"]}
    relax = _relax_maker(p, spin).make(unit)
    gen = phono3py_displacements(relax.output.structure, settings, plan, p)
    runs = {}
    for order in ("fc3", "fc2"):
        run = run_phonon_displacements(
            gen.output[f"{order}_structures"], relax.output.structure, settings[f"{order}_supercell"],
            phonon_maker=_static_maker(p, spin, name=f"phono3py {order} displacement"),
        )
        run.name = f"run_{order}_displacements"
        runs[order] = run
    coll = phono3py_collect(relax.output.structure, settings, plan, p, runs["fc3"].output, runs["fc2"].output, gen.output["meta"])
    return Flow([relax, gen, runs["fc3"], runs["fc2"], coll], output=coll.output, name=name)
