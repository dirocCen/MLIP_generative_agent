"""kappa_SRME protocol: relax -> FC2 -> stability gate -> FC3 -> Wigner-RTA kappa.

Ported from matbench-discovery (MIT licence), commit 2b7bff9 (2026-10-19):
``matbench_discovery/phonons/{thermal_conductivity,pipeline}.py``,
``matbench_discovery/adapters/standard.py`` and ``matbench_discovery/phonons/__init__.py``,
which in turn adapt k_SRME (Póta, Ahlawat, Csányi, Simoncelli; arXiv:2408.00755).

Differences from the original, all deliberate:
- space groups are detected with spglib instead of moyopy (same ``symprec``);
- the phono3py settings (isotope, boundary mfp, transport type) are parameters
  instead of being hard-coded, with the original values as defaults;
- structures without phononDB metadata get supercells / q-mesh from a length rule
  (``length_rule_settings``), fixed on the input structure before submission;
  phononDB structures use their own settings unchanged.

Requires phono3py==3.30.0: 3.31 removed the MS-SMM19 solver that returns
``mode_kappa_P_RTA`` / ``mode_kappa_C``.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np
from ase import Atoms

NO_TILT_MASK = [True, True, True, False, False, False]
IMAGINARY_FREQ_THRESHOLD = -0.01  # THz


# --- symmetry ---------------------------------------------------------------
def space_group_number(atoms: Atoms, symprec: float) -> int | None:
    import spglib

    cell = (np.asarray(atoms.cell), atoms.get_scaled_positions(), atoms.numbers)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return None
    return int(dataset.number if hasattr(dataset, "number") else dataset["number"])


def conventional_cell(atoms: Atoms, symprec: float = 1e-5) -> Atoms:
    """spglib standardised conventional cell (no idealisation of positions)."""
    import spglib

    cell = (np.asarray(atoms.cell), atoms.get_scaled_positions(), atoms.numbers)
    lattice, positions, numbers = spglib.standardize_cell(cell, to_primitive=False, no_idealize=True, symprec=symprec)
    return Atoms(numbers=numbers, cell=lattice, scaled_positions=positions, pbc=True)


# --- relaxation (adapters/standard.py) -----------------------------------------
def _optimizer(name: str):
    from ase import optimize

    return getattr(optimize, name)


def _cell_filter(name: str):
    if name == "none":
        return None
    from ase import filters

    return getattr(filters, name)


def relax(atoms: Atoms, calculator, p: dict[str, Any]) -> tuple[Atoms, dict[str, Any]]:
    """Single- or two-stage symmetry-constrained relaxation (``MLIPRelaxParams``)."""
    from ase.constraints import FixSymmetry

    atoms = atoms.copy()
    initial_spg = space_group_number(atoms, p["symprec"])
    filter_cls = _cell_filter(p["cell_filter"])
    opt_cls = _optimizer(p["optimizer"])

    def run(target_atoms: Atoms, constrained: bool) -> int:
        target_atoms.calc = calculator
        if constrained:
            target_atoms.set_constraint(FixSymmetry(target_atoms, symprec=p["relax_symprec"]))
        if filter_cls is None:
            target = target_atoms
        elif constrained or p["relaxation_mode"] == "two-stage":
            # the original applies the no-tilt mask in both stages of two-stage mode
            target = filter_cls(target_atoms, mask=NO_TILT_MASK)
        else:
            target = filter_cls(target_atoms)
        opt = opt_cls(target, logfile=None)
        opt.run(fmax=p["fmax"], steps=p["max_steps"])
        return int(opt.get_number_of_steps())

    redirected = False
    if p["relaxation_mode"] == "single-stage":
        n_steps = run(atoms, p["enforce_symmetry"])
        reached_max = n_steps >= p["max_steps"]
    else:
        n1 = run(atoms, p["enforce_symmetry"])
        stage1_spg = space_group_number(atoms, p["symprec"])
        stage1 = atoms.copy()
        atoms.set_constraint()
        n2 = run(atoms, False)
        stage2_spg = space_group_number(atoms, p["symprec"])
        redirected = bool(p["enforce_symmetry"] and stage1_spg != stage2_spg)
        if redirected:
            stage1.calc = calculator
            atoms = stage1
        n_steps = n1 + n2
        reached_max = n1 >= p["max_steps"] or n2 >= p["max_steps"]

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = atoms.get_stress()  # Voigt, eV/A^3
    atoms.calc = None
    atoms.set_constraint()
    final_spg = space_group_number(atoms, p["symprec"])
    info = {
        "initial_spg_num": initial_spg,
        "final_spg_num": final_spg,
        "broken_symmetry": initial_spg != final_spg,
        "redirected_to_symmetry": redirected,
        "n_steps": n_steps,
        "reached_max_steps": reached_max,
        "energy": energy,
        "energy_per_atom": energy / len(atoms),
        "max_force": float(np.linalg.norm(forces, axis=1).max()),
        "max_stress_gpa": float(np.abs(stress).max() * 160.21766208),
    }
    return atoms, info


# --- supercells and q-mesh -------------------------------------------------------
def _perpendicular_widths(cell: np.ndarray) -> np.ndarray:
    volume = abs(np.linalg.det(cell))
    return np.array([volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])


def diagonal_supercell(cell: np.ndarray, min_width: float) -> np.ndarray:
    widths = _perpendicular_widths(np.asarray(cell))
    return np.diag(np.maximum(1, np.ceil(min_width / widths - 1e-8)).astype(int))


def lattice_width(lattice: np.ndarray) -> float:
    """Largest min-over-faces perpendicular width over all bases of the lattice (rows).

    A basis with all three widths >= w exists iff the reciprocal lattice has a basis of vectors
    no longer than 1/w; a Minkowski-reduced reciprocal basis attains the successive minima, so the
    answer is 1 / (its longest vector). For the diagonal supercells of orthogonal or hexagonal
    cells this equals ``_perpendicular_widths(...).min()``."""
    from ase.geometry import minkowski_reduce

    rec, _ = minkowski_reduce(np.linalg.inv(np.asarray(lattice, float)).T)
    return float(1.0 / np.linalg.norm(rec, axis=1).max())


def _hnf_matrices(n: int) -> np.ndarray:
    """All sublattices of index n as integer matrices (rows = supercell vectors in the unit-cell
    basis), one Hermite normal form each: rows (a,0,0), (b,c,0), (d,e,f) transposed from the
    column form with 0 <= b < c, 0 <= d,e < f."""
    out = []
    for a in range(1, n + 1):
        if n % a:
            continue
        for c in range(1, n // a + 1):
            if (n // a) % c:
                continue
            f = n // a // c
            out.extend(((a, b, d), (0, c, e), (0, 0, f)) for b in range(c) for d in range(f) for e in range(f))
    return np.array(out, dtype=np.int64).reshape(-1, 3, 3)


_SHORT_COMBOS = np.array([v for v in np.ndindex(5, 5, 5) if any(x != 2 for x in v)]) - 2  # (-2..2)^3 \ 0


def minimal_supercell(cell: np.ndarray, min_width: float, rotations: np.ndarray | None = None,
                      n_max: int | None = None) -> np.ndarray:
    """Smallest supercell (integer matrix M, supercell = M @ cell) with ``lattice_width >= min_width``.

    ``rotations`` (point-group operations acting on fractional coordinates of ``cell``) restricts
    the search to supercells whose lattice they map onto itself, so the supercell keeps the
    crystal's point group and phonopy keeps its displacement reduction. Among equal-size
    candidates the widest wins; M is returned Minkowski-reduced with det > 0. The diagonal
    supercell is always a candidate, so the result is never larger than ``diagonal_supercell``."""
    from ase.geometry import minkowski_reduce

    cell = np.asarray(cell, float)
    diag = diagonal_supercell(cell, min_width)
    n_diag = int(round(np.linalg.det(diag)))
    n_max = min(n_max or n_diag, n_diag)
    vol = abs(np.linalg.det(cell))
    n_min = max(1, int(np.floor(min_width**3 / vol * (1 - 1e-8))))  # V >= w1 w2 w3
    best = None
    for n in range(n_min, n_max):
        ms = _hnf_matrices(n)
        if rotations is not None:  # M R^T M^-1 integer  <=>  (M R^T adj(M)) % n == 0
            adj = np.rint(np.linalg.inv(ms) * n).astype(np.int64)
            keep = np.ones(len(ms), bool)
            for r in rotations:
                keep &= np.all((ms @ np.asarray(r).T @ adj) % n == 0, axis=(1, 2))
            ms = ms[keep]
        if not len(ms):
            continue
        # necessary condition: every lattice vector is at least min_width long; test short combos
        lat = ms.astype(float) @ cell  # (k, 3, 3)
        lengths = np.linalg.norm(np.einsum("ci,kij->kcj", _SHORT_COMBOS, lat), axis=-1).min(axis=1)
        for m in ms[lengths >= min_width - 1e-8]:
            w = lattice_width(m @ cell)
            if w >= min_width - 1e-8 and (best is None or w > best[1]):
                best = (m, w)
        if best is not None:
            break
    m = diag if best is None else best[0]
    _, op = minkowski_reduce(m @ cell)
    m = np.asarray(op) @ m
    if np.linalg.det(m) < 0:
        m = -m
    return np.rint(m).astype(int)


def point_group_rotations(atoms: Atoms, symprec: float) -> np.ndarray:
    """Distinct rotation parts of the space group, in the fractional basis of ``atoms.cell``."""
    import spglib

    ds = spglib.get_symmetry_dataset((np.asarray(atoms.cell), atoms.get_scaled_positions(), atoms.numbers), symprec=symprec)
    return np.unique(np.asarray(ds.rotations), axis=0)


def _primitive_lattice(unit: Atoms, symprec: float) -> np.ndarray:
    """Lattice (rows) of the primitive cell that ``primitive_matrix="auto"`` gives."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    cell = PhonopyAtoms(unit.get_chemical_symbols(), cell=np.asarray(unit.cell), positions=unit.positions)
    return np.asarray(Phonopy(cell, supercell_matrix=np.eye(3, dtype=int), primitive_matrix="auto", symprec=symprec).primitive.cell)


def length_rule_settings(atoms: Atoms, p: dict[str, Any]) -> tuple[Atoms, dict[str, Any]]:
    """Supercells and q-mesh for structures without phononDB settings.

    The submission tools call this once on the *input* structure, so every model
    (and every rerun) uses the same supercells and q-mesh; mode-resolved
    model-vs-model comparisons need identical q-points.

    v2 (11-07): the smallest supercell of the conventional cell whose lattice width
    (``lattice_width``) reaches the minimum, instead of the diagonal one. FC3 keeps the point
    group (phonopy's displacement count depends on it); FC2 takes any shape, since it needs few
    displacements and its size sets the GPU memory (R3m BC2N: diagonal 1176 atoms OOMed TECE on
    80 GB). Never larger than v1, identical for cubic cells at 14 A. ``p["fc2_supercell_shape"]``
    = "symmetric" keeps the point group for FC2 as well (the VASP path: random FC2 displacements
    in a symmetry-broken cell gave imaginary modes); the MLIP params have no such key -> "any".
    """
    from phonopy.structure.grid_points import length2mesh

    unit = conventional_cell(atoms, p["symprec"])
    rots = point_group_rotations(unit, p["symprec"])
    fc2_shape = p.get("fc2_supercell_shape", "any")
    return unit, {
        "source": "length_rule_v2",
        "fc3_supercell": minimal_supercell(unit.cell, p["fc3_min_width"], rots).tolist(),
        "fc2_supercell": minimal_supercell(unit.cell, p["fc2_min_width"], rots if fc2_shape == "symmetric" else None).tolist(),
        "fc2_supercell_shape": fc2_shape,
        "q_mesh": [int(x) for x in length2mesh(p["q_mesh_length"], _primitive_lattice(unit, p["symprec"]))],
        "fc3_min_width": p["fc3_min_width"],
        "fc2_min_width": p["fc2_min_width"],
        "q_mesh_length": p["q_mesh_length"],
    }


def choose_settings(atoms: Atoms, p: dict[str, Any], settings: dict[str, Any] | None) -> tuple[Atoms, dict[str, Any]]:
    """Return (unit cell for phono3py, settings with their provenance).

    ``settings`` fixed at submission (phononDB or ``length_rule_settings`` on the
    input) are used as given, with the relaxed cell as the unit cell; without
    them the length rule is applied to the relaxed structure.
    """
    if settings:
        return atoms, {
            "source": settings.get("source", "given"),
            "fc3_supercell": np.asarray(settings["fc3_supercell"]).tolist(),
            "fc2_supercell": np.asarray(settings["fc2_supercell"]).tolist(),
            "q_mesh": [int(x) for x in settings["q_mesh"]],
        }
    unit, out = length_rule_settings(atoms, p)
    out["source"] = f"{out['source']}_on_relaxed"
    return unit, out


def init_phono3py(unit: Atoms, settings: dict[str, Any], p: dict[str, Any]):
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    ph3 = Phono3py(
        unitcell=PhonopyAtoms(unit.get_chemical_symbols(), cell=np.asarray(unit.cell), positions=unit.positions),
        supercell_matrix=settings["fc3_supercell"],
        phonon_supercell_matrix=settings["fc2_supercell"],
        primitive_matrix="auto",
        symprec=p["symprec"],
    )
    ph3.mesh_numbers = settings["q_mesh"]
    plusminus = {"auto": "auto", "true": True, "false": False}[p["is_plusminus"]]
    ph3.generate_displacements(distance=p["displacement"], is_plusminus=plusminus)
    settings["n_fc2_displacements"] = len(ph3.phonon_supercells_with_displacements)
    settings["n_fc3_displacements"] = sum(s is not None for s in ph3.supercells_with_displacements)
    settings["n_atoms_fc2"] = len(ph3.phonon_supercell)
    settings["n_atoms_fc3"] = len(ph3.supercell)
    settings["n_atoms_primitive"] = len(ph3.primitive)
    return ph3


# --- forces ------------------------------------------------------------------------
def _to_ase(supercell) -> Atoms:
    return Atoms(supercell.symbols, cell=supercell.cell, positions=supercell.positions, pbc=True)


def _forces(supercells, calculator, n_atoms: int) -> np.ndarray:
    out = np.zeros((len(supercells), n_atoms, 3))
    for i, sc in enumerate(supercells):
        if sc is None:
            continue
        atoms = _to_ase(sc)
        atoms.calc = calculator
        f = np.asarray(atoms.get_forces())
        if f.shape != (n_atoms, 3) or not np.all(np.isfinite(f)):
            raise ValueError(f"invalid forces for displacement {i}: shape {f.shape}")
        out[i] = f
    return out


def fc2_and_frequencies(ph3, calculator) -> np.ndarray:
    ph3.phonon_forces = _forces(ph3.phonon_supercells_with_displacements, calculator, len(ph3.phonon_supercell))
    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_phonon_solver()
    freqs, _eigvecs, _grid = ph3.get_phonon_data()
    return freqs


def has_imaginary_modes(frequencies: np.ndarray, threshold: float = IMAGINARY_FREQ_THRESHOLD) -> bool:
    """matbench_discovery.phonons.check_imaginary_freqs; frequencies[0] is Gamma."""
    return bool(
        np.all(np.isnan(frequencies))
        or np.any(frequencies[0, 3:] < 0)
        or np.any(frequencies[0, :3] < threshold)
        or np.any(frequencies[1:] < 0)
    )


# --- conductivity (thermal_conductivity.calculate_conductivity) ---------------------
def calc_mode_kappa_tot(mode_kappa_p_rta: np.ndarray, mode_kappa_coherence: np.ndarray, heat_capacity: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        mode_kappa_c_per_mode = 2 * (
            (mode_kappa_coherence * heat_capacity[:, :, :, None, None])
            / (heat_capacity[:, :, :, None, None] + heat_capacity[:, :, None, :, None])
        ).sum(axis=2)
    mode_kappa_c_per_mode[np.isnan(mode_kappa_c_per_mode)] = 0
    return mode_kappa_c_per_mode + mode_kappa_p_rta


def conductivity(ph3, p: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_thermal_conductivity(
        temperatures=list(p["temperatures"]),
        is_isotope=p["is_isotope"],
        transport_type=p["transport"],
        boundary_mfp=p["boundary_mfp_um"],
    )
    kappa = ph3.thermal_conductivity
    if kappa is None:
        raise RuntimeError("phono3py returned no thermal conductivity")
    extra, mode_cv = kappa.get_extra_kappa_output(), kappa.mode_heat_capacities
    if extra is None or mode_cv is None:
        raise RuntimeError("phono3py returned no Wigner (MS-SMM19) output; is phono3py==3.30.0 installed?")
    arrays = {
        "kappa_tot_rta": deepcopy(extra["kappa_TOT_RTA"][0]),
        "kappa_p_rta": deepcopy(extra["kappa_P_RTA"][0]),
        "kappa_c": deepcopy(extra["kappa_C"][0]),
        "mode_weights": deepcopy(kappa.grid_weights),
        "q_points": deepcopy(kappa.qpoints),
        "frequencies": deepcopy(kappa.frequencies),
    }
    arrays["mode_kappa_tot_rta"] = calc_mode_kappa_tot(
        deepcopy(extra["mode_kappa_P_RTA"][0]), deepcopy(extra["mode_kappa_C"][0]), deepcopy(mode_cv)
    )
    summed = arrays["mode_kappa_tot_rta"].sum(axis=(1, 2)) / arrays["mode_weights"].sum()
    check = {"mode_sum_max_abs_dev": float(np.abs(summed - arrays["kappa_tot_rta"]).max())}
    return arrays, check


def kappa_avg(kappa: np.ndarray) -> np.ndarray:
    """Voigt (..., 6) -> mean of the diagonal (metrics.phonons.calculate_kappa_avg)."""
    return np.asarray(kappa)[..., :3].mean(axis=-1)


# --- one structure ---------------------------------------------------------------------
def run_structure(
    atoms: Atoms,
    calculator,
    relax_p: dict[str, Any],
    kappa_p: dict[str, Any],
    settings: dict[str, Any] | None = None,
    fc_dir: str | None = None,
) -> dict[str, Any]:
    """The full per-structure flow; failures are returned, not raised.

    ``fc_dir``: write the compact FC2/FC3 there (ladder L2 reuses them). A failed cache write
    is recorded in ``fc_cache`` and does not change the kappa result."""
    out: dict[str, Any] = {"ok": False, "stage": "relax", "error": None}
    arrays: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    try:
        relaxed, out["relax"] = relax(atoms, calculator, relax_p)
        out["stage"] = "force_constants"
        unit, settings = choose_settings(relaxed, kappa_p, settings)
        ph3 = init_phono3py(unit, settings, kappa_p)
        out["settings"] = settings
        freqs = fc2_and_frequencies(ph3, calculator)
        imag = has_imaginary_modes(freqs, kappa_p["imaginary_threshold_thz"])
        out["has_imaginary_modes"] = imag
        out["min_frequency_thz"] = float(np.nanmin(freqs))
        arrays["frequencies_bz_grid"] = freqs[ph3.grid.grg2bzg]
        broken = bool(out["relax"]["broken_symmetry"])
        if imag or (broken and not kappa_p["kappa_if_symmetry_broken"]):
            out["conductivity_skipped"] = "imaginary_modes" if imag else "broken_symmetry"
        else:
            out["stage"] = "fc3"
            ph3.forces = _forces(ph3.supercells_with_displacements, calculator, len(ph3.supercell))
            ph3.produce_fc3(symmetrize_fc3r=True)
            if fc_dir:
                from mga.workflows.ladder import save_fc_cache

                try:
                    out["fc_cache"] = save_fc_cache(ph3, fc_dir, settings)
                except Exception as exc:  # noqa: BLE001
                    out["fc_cache"] = {"fc_dir": fc_dir, "error": f"{type(exc).__name__}: {exc}"}
            out["stage"] = "conductivity"
            kappa_arrays, out["checks"] = conductivity(ph3, kappa_p)
            arrays.update(kappa_arrays)
            out["conductivity_skipped"] = None
            out["temperatures"] = list(kappa_p["temperatures"])
            out["kappa_tot_avg"] = kappa_avg(kappa_arrays["kappa_tot_rta"]).tolist()
            out["kappa_p_avg"] = kappa_avg(kappa_arrays["kappa_p_rta"]).tolist()
            out["kappa_c_avg"] = kappa_avg(kappa_arrays["kappa_c"]).tolist()
        out["ok"] = True
        out["stage"] = "done"
        out["relaxed_structure"] = {
            "numbers": relaxed.numbers.tolist(),
            "cell": np.asarray(relaxed.cell).tolist(),
            "positions": relaxed.positions.tolist(),
        }
    except Exception as exc:  # noqa: BLE001 - every structure failure is recorded
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc(limit=6)
    out["wall_time_s"] = round(time.perf_counter() - t0, 2)
    out["arrays"] = {k: np.asarray(v).tolist() for k, v in arrays.items()}
    return out
