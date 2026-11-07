"""The rungs of the multi-fidelity ladder around the kappa_SRME step (plan §9.3).

    L0  fc2_descriptors      relax + FC2 only: stability gate, harmonic descriptors,
                             weighted phase space, Grueneisen + Slack estimate
    L1  (kappa_protocol)     committee Wigner-RTA kappa; optionally writes its FCs to the FC cache
    L2  lbte_kappa           Wigner kappa from the direct LBTE solution on the cached L1 FCs;
                             RTA on the same mesh, so the ratio kappa_LBTE / kappa_RTA is exact
    L3  fc4_fourphonon       FC2/FC3/FC4 by random displacements + symfc, ShengBTE/FourPhonon
                             input files, optional FourPhonon run (3ph and 3+4ph)
    L4  DFT phono3py_fd      (workflows/phono3py_fd.py)

Every function returns a plain dict and records failures instead of raising, like
``kappa_protocol.run_structure``. The same supercells / q-mesh (``settings``) fixed at
submission are used on every rung, so mode-resolved quantities line up with L1.
"""

from __future__ import annotations

import itertools
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mga.workflows.kappa_protocol import (
    _forces,
    _perpendicular_widths,
    choose_settings,
    conductivity,
    fc2_and_frequencies,
    has_imaginary_modes,
    init_phono3py,
    kappa_avg,
    relax,
)

THZ_TO_EV = 4.135667696e-3
EV_TO_J = 1.602176634e-19
KB_EV = 8.617333262e-5
H_EV_S = 4.135667696e-15
AMU_KG = 1.66053906660e-27
FC_FILES = ("phono3py_params.yaml", "fc2.hdf5", "fc3.hdf5", "mga_fc_meta.json")
SLACK_SOURCE = (
    "Slack / Morelli-Slack form kappa = A M theta_a^3 delta n^(1/3) / (gamma^2 T), "
    "A = 2.43e-6 / (1 - 0.514/gamma + 0.228/gamma^2) with M in amu, delta in Angstrom; "
    "theta_a from the maximum acoustic frequency. A ranking descriptor, not a prediction."
)


def _phonopy_atoms(atoms: Atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(atoms.get_chemical_symbols(), cell=np.asarray(atoms.cell), positions=atoms.positions)


def _structure_dict(atoms: Atoms) -> dict[str, Any]:
    return {"numbers": atoms.numbers.tolist(), "cell": np.asarray(atoms.cell).tolist(), "positions": atoms.positions.tolist()}


def _failure(out: dict[str, Any], exc: Exception) -> None:
    out["error"] = f"{type(exc).__name__}: {exc}"
    out["traceback"] = traceback.format_exc(limit=6)


# ======================================================================================
# L0: harmonic descriptors
# ======================================================================================
def init_phono3py_fc2(unit: Atoms, settings: dict[str, Any], p: dict[str, Any]):
    """Phono3py object with the L1 supercells and q-mesh but only FC2 displacements."""
    from phono3py import Phono3py

    ph3 = Phono3py(
        unitcell=_phonopy_atoms(unit),
        supercell_matrix=settings["fc3_supercell"],
        phonon_supercell_matrix=settings["fc2_supercell"],
        primitive_matrix="auto",
        symprec=p["symprec"],
    )
    ph3.mesh_numbers = settings["q_mesh"]
    plusminus = {"auto": "auto", "true": True, "false": False}[p["is_plusminus"]]
    ph3.generate_fc2_displacements(distance=p["displacement"], is_plusminus=plusminus)
    settings["n_fc2_displacements"] = len(ph3.phonon_supercells_with_displacements)
    settings["n_atoms_fc2"] = len(ph3.phonon_supercell)
    settings["n_atoms_primitive"] = len(ph3.phonon_primitive)
    return ph3


def _phonopy_from_ph3(ph3):
    """Phonopy object on the FC2 supercell of ``ph3`` carrying its FC2."""
    from phonopy import Phonopy

    ph = Phonopy(ph3.unitcell, supercell_matrix=ph3.phonon_supercell_matrix, primitive_matrix=ph3.primitive_matrix,
                 symprec=ph3.symmetry.tolerance)
    ph.force_constants = ph3.fc2
    return ph


def mode_heat_capacity(freqs_thz: np.ndarray, temperature: float) -> np.ndarray:
    """Mode heat capacity in J/K (zero for non-positive frequencies)."""
    from phonopy.phonon.thermal_properties import mode_cv

    f = np.asarray(freqs_thz, float)
    cv = np.zeros_like(f)
    pos = f > 1e-3
    cv[pos] = mode_cv(temperature, f[pos] * THZ_TO_EV) * EV_TO_J
    return cv


def harmonic_descriptors(ph3, temperature: float, with_wjdos: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Descriptors on the irreducible q-points of the L1 mesh.

    - acoustic / optical frequency ranges, a-o gap, optical bandwidth, acoustic branch spread
      (the three lowest branches are taken as acoustic);
    - group velocities and the constant-relaxation-time conductivity with tau = 1 ps;
    - the weighted joint density of states W(q, j) at the mode frequencies (phono3py JointDos
      with temperature, tetrahedron method): the three-phonon phase space weighted by the
      occupations; with a constant interaction strength the scattering rate is proportional to it.
    """
    from phono3py.phonon.grid import get_ir_grid_points
    from phono3py.phonon3.joint_dos import JointDos

    bz = ph3.grid
    jd = JointDos(ph3.phonon_primitive, ph3.phonon_supercell, bz, ph3.fc2, cutoff_frequency=1e-4)
    jd.run_phonon_solver()
    ir, weights, _ = get_ir_grid_points(bz)
    gps = bz.grg2bzg[ir]
    freqs = np.asarray(jd.get_phonons()[0][gps])  # (n_ir, n_band) THz
    qpoints = bz.addresses[gps] @ bz.QDinv.T  # reduced coordinates
    weights = np.asarray(weights, float)

    ph = _phonopy_from_ph3(ph3)
    ph.run_qpoints(qpoints, with_group_velocities=True)
    gv = np.asarray(ph.get_qpoints_dict()["group_velocities"])  # THz * Angstrom
    gv_ms = gv * 1e2  # THz*Angstrom = 1e12 * 1e-10 m/s

    nb = freqs.shape[1]
    n_prim = nb // 3
    cv = mode_heat_capacity(freqs, temperature)
    w = weights[:, None] * np.ones(nb)
    volume_m3 = abs(np.linalg.det(np.asarray(ph3.phonon_primitive.cell))) * 1e-30
    n_q = weights.sum()
    tau = 1e-12
    # isotropic |v|^2 / 3: on the irreducible wedge alone the v_x^2, v_y^2, v_z^2 sums are not
    # symmetry-equivalent (that needs the star of each q), the trace is
    v2_3 = (gv_ms**2).sum(axis=-1) / 3
    k_ctau = float((cv * v2_3 * tau * w).sum() / (volume_m3 * n_q))  # W/m/K

    gamma_mask = np.all(bz.addresses[gps] == 0, axis=1)
    ac = freqs[:, :3]
    out: dict[str, Any] = {
        "temperature": temperature,
        "n_atoms_primitive": n_prim,
        "n_irreducible_q": len(ir),
        "volume_per_atom_A3": volume_m3 * 1e30 / n_prim,
        "mass_mean_amu": float(np.mean(ph3.phonon_primitive.masses)),
        "freq_max_thz": float(freqs.max()),
        "acoustic_max_thz": float(ac.max()),
        "theta_acoustic_K": float(H_EV_S * 1e12 * ac.max() / KB_EV),
        "kappa_ctau_1ps_W_mK": k_ctau,
    }
    # acoustic branch spread: (w_LA - w_TA1) / w_LA averaged over q != Gamma; small = bunched
    nz = ~gamma_mask & (ac[:, 2] > 1e-3)
    out["acoustic_branch_spread"] = float(np.average((ac[nz, 2] - ac[nz, 0]) / ac[nz, 2], weights=weights[nz])) if nz.any() else None
    speed = np.linalg.norm(gv_ms, axis=-1) / 1e3  # km/s
    cva = cv[:, :3] * weights[:, None]
    out["v_acoustic_rms_km_s"] = float(np.sqrt((cva * speed[:, :3] ** 2).sum() / cva.sum())) if cva.sum() > 0 else None
    cvw = cv * w
    out["v_rms_km_s"] = float(np.sqrt((cvw * speed**2).sum() / cvw.sum())) if cvw.sum() > 0 else None
    out["acoustic_cv_fraction"] = float(cva.sum() / cvw.sum()) if cvw.sum() > 0 else None
    if n_prim > 1:
        opt_min = float(freqs[:, 3].min())
        out.update(
            optical_min_thz=opt_min,
            ao_gap_thz=opt_min - out["acoustic_max_thz"],
            ao_gap_ratio=(opt_min - out["acoustic_max_thz"]) / out["acoustic_max_thz"],
            optical_bandwidth_thz=out["freq_max_thz"] - opt_min,
        )
    else:
        out.update(optical_min_thz=None, ao_gap_thz=None, ao_gap_ratio=None, optical_bandwidth_thz=None)

    arrays = {"ir_qpoints": qpoints, "ir_weights": weights, "frequencies": freqs, "group_velocities_thz_A": gv}
    if with_wjdos:
        jd.temperature = temperature
        wj = np.zeros((len(gps), nb, 2))
        for i, gp in enumerate(gps):
            jd.set_grid_point(gp)
            jd.frequency_points = freqs[i]
            jd.run_integration_weights()
            jd.run_jdos()
            wj[i] = jd.joint_dos
        wtot = wj.sum(axis=-1)  # 1/THz, both process classes
        valid = (freqs > 1e-3) & (wtot > 0)
        arrays["wjdos"] = wj
        out["wjdos_mean"] = float(np.average(wtot[valid], weights=w[valid]))
        out["wjdos_cv_mean"] = float(np.average(wtot[valid], weights=cvw[valid])) if cvw[valid].sum() > 0 else None
        va = valid[:, :3]
        out["wjdos_acoustic_cv_mean"] = float(np.average(wtot[:, :3][va], weights=cva[va])) if cva[va].sum() > 0 else None
        # constant-interaction proxy: tau proportional to 1/W -> sum C v^2 / W (relative units, rank at fixed T only)
        out["kappa_proxy_cv2_over_wjdos"] = float(((cvw * v2_3)[valid] / wtot[valid]).sum() / (volume_m3 * n_q) * 1e-12)
    return out, arrays


def _relax_positions(atoms: Atoms, calculator, relax_p: dict[str, Any]) -> int:
    from ase.constraints import FixSymmetry

    from mga.workflows.kappa_protocol import _optimizer

    atoms.calc = calculator
    atoms.set_constraint(FixSymmetry(atoms, symprec=relax_p["relax_symprec"]))
    opt = _optimizer(relax_p["optimizer"])(atoms, logfile=None)
    opt.run(fmax=relax_p["fmax"], steps=relax_p["max_steps"])
    atoms.set_constraint()
    return int(opt.get_number_of_steps())


def gruneisen_descriptors(ph3, unit: Atoms, calculator, p: dict[str, Any], relax_p: dict[str, Any],
                          harmonic: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Mode Grueneisen parameters by finite volume differences (phonopy PhonopyGruneisen):
    FC2 at V(1 +- volume_strain) with internal coordinates relaxed at fixed cell,
    then the Slack estimate from theta_a, the acoustic gamma, M and delta."""
    from phonopy import Phonopy
    from phonopy.api_gruneisen import PhonopyGruneisen

    eps = p["volume_strain"]
    plusminus = {"auto": "auto", "true": True, "false": False}[p["is_plusminus"]]
    phonons, steps = [], []
    for sign in (+1, -1):
        a = unit.copy()
        a.set_cell(np.asarray(a.cell) * (1 + sign * eps) ** (1 / 3), scale_atoms=True)
        steps.append(_relax_positions(a, calculator, relax_p))
        a.calc = None
        ph = Phonopy(_phonopy_atoms(a), supercell_matrix=ph3.phonon_supercell_matrix, primitive_matrix=ph3.primitive_matrix,
                     symprec=ph3.symmetry.tolerance)
        ph.generate_displacements(distance=p["displacement"], is_plusminus=plusminus)
        ph.forces = _forces(ph.supercells_with_displacements, calculator, len(ph.supercell))
        ph.produce_force_constants()
        ph.symmetrize_force_constants()
        phonons.append(ph)
    pg = PhonopyGruneisen(_phonopy_from_ph3(ph3), phonons[0], phonons[1])
    pg.set_mesh(ph3.mesh_numbers, is_gamma_center=True)
    _q, wq, f, _ev, g = pg.get_mesh()
    f, g, wq = np.asarray(f), np.asarray(g), np.asarray(wq, float)
    T = harmonic["temperature"]
    cv = mode_heat_capacity(f, T) * wq[:, None]
    ok = np.isfinite(g) & (f > 1e-2)
    cva, ga = cv[:, :3][ok[:, :3]], g[:, :3][ok[:, :3]]
    gamma_ac = float(np.sqrt((cva * ga**2).sum() / cva.sum())) if cva.sum() > 0 else None
    out = {
        "volume_strain": eps,
        "relax_steps_strained": steps,
        "gamma_cv_mean": float((cv[ok] * g[ok]).sum() / cv[ok].sum()) if cv[ok].sum() > 0 else None,
        "gamma_acoustic_rms": gamma_ac,
        "gamma_abs_max": float(np.abs(g[ok]).max()) if ok.any() else None,
        "slack_source": SLACK_SOURCE,
    }
    out["kappa_slack_W_mK"] = slack_kappa(
        harmonic["mass_mean_amu"], harmonic["theta_acoustic_K"], harmonic["volume_per_atom_A3"],
        harmonic["n_atoms_primitive"], gamma_ac, T)
    return out, {"gruneisen_mesh": g, "gruneisen_mesh_frequencies": f, "gruneisen_mesh_weights": wq}


def slack_kappa(mass_amu: float, theta_a: float, vol_per_atom: float, n: int, gamma: float | None, T: float) -> float | None:
    if not gamma or gamma <= 0:
        return None
    A = 2.43e-6 / (1 - 0.514 / gamma + 0.228 / gamma**2)
    return float(A * mass_amu * theta_a**3 * vol_per_atom ** (1 / 3) * n ** (1 / 3) / (gamma**2 * T))


def run_descriptors(atoms: Atoms, calculator, relax_p: dict[str, Any], p: dict[str, Any],
                    settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """L0 for one structure: relax -> FC2 -> imaginary-mode gate (same test as L1) -> descriptors."""
    out: dict[str, Any] = {"ok": False, "stage": "relax", "error": None}
    arrays: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    try:
        relaxed, out["relax"] = relax(atoms, calculator, relax_p)
        out["stage"] = "fc2"
        unit, settings = choose_settings(relaxed, p, settings)
        ph3 = init_phono3py_fc2(unit, settings, p)
        out["settings"] = settings
        freqs = fc2_and_frequencies(ph3, calculator)
        out["has_imaginary_modes"] = has_imaginary_modes(freqs, p["imaginary_threshold_thz"])
        out["min_frequency_thz"] = float(np.nanmin(freqs))
        out["broken_symmetry"] = bool(out["relax"]["broken_symmetry"])
        if out["has_imaginary_modes"]:
            out["descriptors_skipped"] = "imaginary_modes"
        else:
            out["stage"] = "harmonic"
            out["descriptors"], arrays = harmonic_descriptors(ph3, p["temperature"], p["weighted_phase_space"])
            if p["gruneisen"]:
                out["stage"] = "gruneisen"
                g, garr = gruneisen_descriptors(ph3, unit, calculator, p, relax_p, out["descriptors"])
                out["descriptors"].update(g)
                arrays.update(garr)
            out["descriptors_skipped"] = None
        out["ok"], out["stage"] = True, "done"
        out["relaxed_structure"] = _structure_dict(relaxed)
    except Exception as exc:  # noqa: BLE001
        _failure(out, exc)
    out["wall_time_s"] = round(time.perf_counter() - t0, 2)
    out["arrays"] = {k: np.asarray(v).tolist() for k, v in arrays.items()}
    return out


# ======================================================================================
# FC cache (written by L1, read by L2)
# ======================================================================================
def save_fc_cache(ph3, fc_dir: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compact fc2 / fc3 (rows of the primitive atoms only: what the phonon and ph-ph
    calculations use) + the phono3py structure/displacement yaml. Written to a temporary
    directory first, then renamed, so a reader never sees half a cache."""
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

    final = Path(fc_dir)
    tmp = final.with_name(final.name + f".tmp-{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    p2s = ph3.primitive.p2s_map
    p2s2 = ph3.phonon_primitive.p2s_map
    fc3 = ph3.fc3 if ph3.fc3.shape[0] == len(p2s) else ph3.fc3[p2s]
    fc2 = ph3.fc2 if ph3.fc2.shape[0] == len(p2s2) else ph3.fc2[p2s2]
    write_fc3_to_hdf5(fc3, filename=str(tmp / "fc3.hdf5"), p2s_map=p2s)
    write_fc2_to_hdf5(fc2, filename=str(tmp / "fc2.hdf5"), p2s_map=p2s2)
    ph3.save(str(tmp / "phono3py_params.yaml"), settings={"force_sets": False, "displacements": True, "force_constants": False})
    import json

    # the yaml rounds masses / cell (~1e-7); on a coarse q-mesh that alone moves kappa by several
    # percent (degenerate modes in the tetrahedron integration), so L2 rebuilds from these exact values
    u = ph3.unitcell
    meta = {"q_mesh": [int(x) for x in ph3.mesh_numbers], "settings": settings, "phono3py_version": ph3.version,
            "unitcell": {"symbols": list(u.symbols), "cell": np.asarray(u.cell).tolist(),
                         "scaled_positions": np.asarray(u.scaled_positions).tolist(), "masses": np.asarray(u.masses).tolist()},
            "supercell_matrix": np.asarray(ph3.supercell_matrix).tolist(),
            "phonon_supercell_matrix": np.asarray(ph3.phonon_supercell_matrix).tolist(),
            "primitive_matrix": np.asarray(ph3.primitive_matrix).tolist() if ph3.primitive_matrix is not None else None}
    (tmp / "mga_fc_meta.json").write_text(json.dumps(meta, default=str, indent=1))
    if final.exists():
        import shutil

        shutil.rmtree(final)
    os.replace(tmp, final)
    return {"fc_dir": str(final), "files": list(FC_FILES), "compact": True}


def fc_cache_complete(fc_dir: str | None, settings: dict[str, Any] | None = None) -> bool:
    """All files present and, when ``settings`` are given, written for the same supercells: the
    cache key is the parameter hash, which does not see the supercell rule (length_rule ->
    length_rule_v2 changed the supercells of non-phononDB structures under the same hash)."""
    if not (fc_dir and all((Path(fc_dir) / f).exists() for f in FC_FILES)):
        return False
    if not settings:
        return True
    import json

    cached = json.loads((Path(fc_dir) / "mga_fc_meta.json").read_text()).get("settings") or {}
    return all(np.array_equal(np.asarray(cached.get(k)), np.asarray(settings[k]))
               for k in ("fc2_supercell", "fc3_supercell") if k in settings)


def load_fc_cache(fc_dir: str, symprec: float):
    import json

    import phono3py

    d = Path(fc_dir)
    meta = json.loads((d / "mga_fc_meta.json").read_text())
    if "unitcell" in meta:
        from phono3py import Phono3py
        from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
        from phonopy.structure.atoms import PhonopyAtoms

        u = meta["unitcell"]
        ph3 = Phono3py(
            unitcell=PhonopyAtoms(u["symbols"], cell=u["cell"], scaled_positions=u["scaled_positions"], masses=u["masses"]),
            supercell_matrix=meta["supercell_matrix"], phonon_supercell_matrix=meta["phonon_supercell_matrix"],
            primitive_matrix=meta["primitive_matrix"], symprec=symprec, log_level=0)
        ph3.fc2 = read_fc2_from_hdf5(str(d / "fc2.hdf5"))
        ph3.fc3 = read_fc3_from_hdf5(str(d / "fc3.hdf5"))
        return ph3
    return phono3py.load(str(d / "phono3py_params.yaml"), fc2_filename=str(d / "fc2.hdf5"), fc3_filename=str(d / "fc3.hdf5"),
                         is_compact_fc=True, produce_fc=True, symprec=symprec, log_level=0)


# ======================================================================================
# L2: LBTE
# ======================================================================================
def n_irreducible_q(ph3, mesh) -> int:
    from phono3py.phonon.grid import BZGrid, get_ir_grid_points

    bz = BZGrid(mesh, lattice=ph3.primitive.cell, symmetry_dataset=ph3.primitive_symmetry.dataset, store_dense_gp_map=True)
    return len(get_ir_grid_points(bz)[0])


def collision_matrix_gb(n_ir: int, n_band: int, n_temps: int = 1) -> float:
    """Irreducible collision matrix (n_ir * n_band * 3)^2 doubles per temperature, x2 for the
    pseudo-inverse workspace (phono3py LBTE, is_reducible_collision_matrix=False)."""
    return 2 * n_temps * (n_ir * n_band * 3) ** 2 * 8 / 1e9


def lbte_mesh(ph3, mesh: list[int], max_gb: float, min_scale: float, n_temps: int) -> tuple[list[int], float, int]:
    """Largest mesh <= the L1 mesh (uniform scale, steps of 0.05) whose collision matrix fits."""
    nb = len(ph3.primitive) * 3
    scale = 1.0
    while True:
        m = [max(1, int(round(x * scale))) for x in mesh]
        n_ir = n_irreducible_q(ph3, m)
        gb = collision_matrix_gb(n_ir, nb, n_temps)
        if gb <= max_gb or scale - 0.05 < min_scale - 1e-9:
            return m, gb, n_ir
        scale -= 0.05


def run_lbte(atoms: Atoms, calculator, relax_p: dict[str, Any], kappa_p: dict[str, Any], lp: dict[str, Any],
             settings: dict[str, Any] | None, fc_dir: str | None) -> dict[str, Any]:
    """L2 for one structure. FCs from the L1 cache when present (identical protocol and
    parameter hash), otherwise recomputed exactly as L1 and written to the cache."""
    out: dict[str, Any] = {"ok": False, "stage": "force_constants", "error": None}
    arrays: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    try:
        if fc_cache_complete(fc_dir, settings):
            ph3 = load_fc_cache(fc_dir, kappa_p["symprec"])
            out["fc_source"] = "cache"
        else:
            if calculator is None:
                raise RuntimeError("no FC cache and no calculator")
            relaxed, out["relax"] = relax(atoms, calculator, relax_p)
            unit, settings = choose_settings(relaxed, kappa_p, settings)
            ph3 = init_phono3py(unit, settings, kappa_p)
            freqs = fc2_and_frequencies(ph3, calculator)
            if has_imaginary_modes(freqs, kappa_p["imaginary_threshold_thz"]):
                out.update(ok=True, stage="done", conductivity_skipped="imaginary_modes", fc_source="recomputed")
                return out
            ph3.forces = _forces(ph3.supercells_with_displacements, calculator, len(ph3.supercell))
            ph3.produce_fc3(symmetrize_fc3r=True)
            out["fc_source"] = "recomputed"
            if fc_dir:
                out["fc_cache"] = save_fc_cache(ph3, fc_dir, settings)
        if not (settings or {}).get("q_mesh") and out["fc_source"] == "cache":
            import json

            settings = {**(settings or {}), **json.loads((Path(fc_dir) / "mga_fc_meta.json").read_text())}
        full_mesh = [int(x) for x in settings["q_mesh"]]
        temps = list(lp["temperatures"])
        mesh, gb, n_ir = lbte_mesh(ph3, full_mesh, lp["max_collision_matrix_gb"], lp["min_mesh_scale"], len(temps))
        out["lbte_mesh"] = {"q_mesh_l1": full_mesh, "q_mesh": mesh, "reduced": mesh != full_mesh, "n_irreducible_q": n_ir,
                            "collision_matrix_gb_est": round(gb, 3)}
        if gb > lp["max_collision_matrix_gb"]:
            out.update(ok=True, stage="done", conductivity_skipped=f"collision matrix {gb:.1f} GB > limit even at min_mesh_scale")
            return out
        out["stage"] = "lbte"
        ph3.mesh_numbers = mesh
        ph3.init_phph_interaction(symmetrize_fc3q=False)
        ph3.run_thermal_conductivity(
            temperatures=temps, is_isotope=kappa_p["is_isotope"], transport_type=kappa_p["transport"],
            boundary_mfp=kappa_p["boundary_mfp_um"], is_LBTE=True, pinv_cutoff=lp["pinv_cutoff"],
        )
        k = ph3.thermal_conductivity
        extra = k.get_extra_kappa_output() if k is not None else None
        if not extra or "kappa_TOT_exact" not in extra:
            raise RuntimeError("phono3py returned no Wigner LBTE output (kappa_TOT_exact); is phono3py==3.30.0 installed?")
        get = {name: np.asarray(extra[name][0]) for name in ("kappa_TOT_exact", "kappa_TOT_RTA", "kappa_P_exact", "kappa_P_RTA", "kappa_C")}
        arrays.update({name.lower(): v for name, v in get.items()})
        out["temperatures"] = temps
        for name, v in get.items():
            out[f"{name.lower()}_avg"] = kappa_avg(v).tolist()
        ratio = np.asarray(out["kappa_tot_exact_avg"]) / np.asarray(out["kappa_tot_rta_avg"])
        out["lbte_over_rta"] = ratio.tolist()
        out["conductivity_skipped"] = None
        out["ok"], out["stage"] = True, "done"
    except Exception as exc:  # noqa: BLE001
        _failure(out, exc)
    out["wall_time_s"] = round(time.perf_counter() - t0, 2)
    out["arrays"] = {k: np.asarray(v).tolist() for k, v in arrays.items()}
    return out



# ======================================================================================
# L3: FC2/FC3/FC4 by random displacements + symfc; ShengBTE / FourPhonon input
# ======================================================================================
def primitive_supercell_matrix(prim_cell: np.ndarray, min_width: float) -> list[int]:
    """ShengBTE / FourPhonon need a diagonal supercell (``scell``) of the cell in CONTROL,
    which here is the primitive cell."""
    widths = _perpendicular_widths(np.asarray(prim_cell))
    return [int(x) for x in np.maximum(1, np.ceil(min_width / widths - 1e-8))]


def fc4_memory_gb(n_prim: int, n_supercell: int) -> float:
    """symfc keeps the FC4 compression matrix as a CSR array with one row per compact FC4 element
    (n_prim * N^3 * 81), so its int64 row pointer alone takes as much as a dense compact FC4; the
    solver makes a scaled copy, so the peak is at least twice this (set GpuResources.mem)."""
    return n_prim * n_supercell**3 * 81 * 8 / 1e9


def neighbour_shells(supercell, p2s: np.ndarray, gap: float) -> tuple[list[tuple[float, float]], float]:
    """Shells of minimum-image distances from the primitive atoms, as (shortest, longest) per
    shell: sorted distances split where consecutive ones differ by more than ``gap`` x the
    nearest-neighbour distance. Only distances below half the supercell width are complete, so
    shells are listed up to there; that limit is returned too."""
    from ase.geometry import get_distances

    from mga.workflows.kappa_protocol import lattice_width

    cell = np.asarray(supercell.cell)
    pos = np.asarray(supercell.scaled_positions) @ cell
    _, d = get_distances(pos[np.asarray(p2s)], pos, cell=cell, pbc=True)
    half = lattice_width(cell) / 2
    d = np.sort(d[(d > 1e-3) & (d < half)])
    if not len(d):
        raise ValueError("supercell too small to see the nearest neighbours")
    cuts = np.flatnonzero(np.diff(d) > gap * d[0])
    lo, hi = np.r_[0, cuts + 1], np.r_[cuts, len(d) - 1]
    return [(float(d[a]), float(d[b])) for a, b in zip(lo, hi, strict=True)], float(half)


def shell_cutoff(shells: list[tuple[float, float]], n: int, complete_to: float) -> float:
    """Midway between the n-th shell and the next (or, if the next lies beyond the complete range,
    midway to that range's end): insensitive to the few-percent lattice-constant spread between
    models, unlike a fixed radius (c-BN's 3rd shell sits at 2.997 A)."""
    if len(shells) < n:
        raise ValueError(f"only {len(shells)} complete neighbour shells in the supercell; {n} needed")
    return 0.5 * (shells[n - 1][1] + (shells[n][0] if len(shells) > n else complete_to))


# Sparse compact FC of order n: (keys, blocks), keys (m, n) = [primitive index, supercell atoms...],
# blocks (m, 3, ..., 3). FC4 is only ever held this way (dense: n_prim * N^3 * 81 doubles).
SparseFC = tuple[np.ndarray, np.ndarray]


def dense_to_sparse(fc: np.ndarray, order: int, tol: float = 1e-10) -> SparseFC:
    keys = np.argwhere(np.abs(fc).reshape(*fc.shape[:order], -1).max(axis=-1) > tol)
    return keys, fc[tuple(keys.T)]


def sparse_fc4_from_symfc(basis_o4, coefs: np.ndarray, n_supercell: int, tol: float = 1e-10) -> SparseFC:
    """compact FC4 = (compression matrix @ basis @ coefs) without the dense n_prim * N^3 * 81 vector:
    multiply only the stored entries and sum them per row. Uses symfc's unscaled matrix (1.7.x
    attribute ``_n_a_compression_matrix``) because ``compact_compression_matrix`` copies it."""
    v = np.asarray(basis_o4.blocked_basis_set @ coefs).ravel()
    m = getattr(basis_o4, "_n_a_compression_matrix", None)
    scale = 1.0 / np.sqrt(basis_o4.translation_permutations.shape[0]) if m is not None else 1.0
    if m is None:
        m = basis_o4.compact_compression_matrix
    coo = m.tocoo()
    vals = coo.data * v[coo.col] * scale
    rows, inv = np.unique(np.asarray(coo.row, dtype=np.int64), return_inverse=True)
    vals = np.bincount(inv.ravel(), weights=vals, minlength=len(rows))
    atom_rows, comp = np.divmod(rows, 81)
    keys_u, kinv = np.unique(atom_rows, return_inverse=True)
    blocks = np.zeros((len(keys_u), 81))
    blocks[kinv.ravel(), comp] = vals
    keep = np.abs(blocks).max(axis=1) > tol
    n = n_supercell
    keys = np.stack(np.unravel_index(keys_u[keep], (m.shape[0] // (81 * n**3), n, n, n)), axis=1).astype(np.int64)
    return keys, blocks[keep].reshape(-1, 3, 3, 3, 3)


def primitive_atoms(atoms: Atoms, symprec: float) -> Atoms:
    from phonopy import Phonopy

    prim = Phonopy(_phonopy_atoms(atoms), supercell_matrix=np.eye(3, dtype=int), primitive_matrix="auto", symprec=symprec).primitive
    return Atoms(prim.symbols, cell=prim.cell, scaled_positions=np.mod(prim.scaled_positions, 1.0), pbc=True)


def random_displacements(n_snap: int, n_atoms: int, distance: float, seed: int) -> np.ndarray:
    """Every atom displaced by ``distance`` in a random direction (symfc's usual protocol)."""
    d = np.random.default_rng(seed).normal(size=(n_snap, n_atoms, 3))
    return d * (distance / np.linalg.norm(d, axis=-1, keepdims=True))


def predict_forces(fcs: dict[int, Any], u: np.ndarray) -> np.ndarray:
    """Forces on the primitive-cell atoms (compact FC rows) for one displacement set u (N, 3):
    F = -FC2 u - 1/2 FC3 u u - 1/6 FC4 u u u, contracted one index at a time (no N^3 temporaries).
    FC4 may be dense or a ``SparseFC``."""
    f = -np.einsum("pjab,jb->pa", fcs[2], u)
    if 3 in fcs:
        t = np.tensordot(fcs[3], u, axes=([2, 5], [0, 1]))  # p j a b
        f -= 0.5 * np.einsum("pjab,jb->pa", t, u)
    if 4 in fcs:
        if isinstance(fcs[4], tuple):
            keys, blocks = fcs[4]
            t = np.einsum("nabcd,nb,nc,nd->na", blocks, u[keys[:, 1]], u[keys[:, 2]], u[keys[:, 3]])
            np.add.at(f, keys[:, 0], -t / 6)
        else:
            t = np.tensordot(fcs[4], u, axes=([3, 7], [0, 1]))  # p j k a b c
            t = np.tensordot(t, u, axes=([2, 5], [0, 1]))  # p j a b
            f -= (1 / 6) * np.einsum("pjab,jb->pa", t, u)
    return f


def fit_fc234(prim: Atoms, calculator, p: dict[str, Any]) -> dict[str, Any]:
    """Random displacements on the diagonal primitive supercell -> forces -> symfc fit of
    FC2/FC3/FC4 (compact, eV/Angstrom^n; FC4 sparse). The FC4 cutoff is set by neighbour shells
    (``fc4_shells``). 10 % of the snapshots are held out and predicted."""
    from phonopy import Phonopy
    from symfc import Symfc
    from symfc.solvers.solver_O2O3O4 import FCSolverO2O3O4
    from symfc.utils.utils import SymfcAtoms

    scell = primitive_supercell_matrix(np.asarray(prim.cell), p["supercell_min_width"])
    ph = Phonopy(_phonopy_atoms(prim), supercell_matrix=np.diag(scell), primitive_matrix=None, symprec=p["symprec"])
    sc = ph.supercell
    n_sc = len(sc)
    mem = fc4_memory_gb(len(prim), n_sc)
    if mem > p["max_fc4_gb"]:
        raise MemoryError(f"symfc's FC4 compression matrix would need >= {mem:.1f} GB (> max_fc4_gb={p['max_fc4_gb']}); "
                          f"{len(prim)} atoms/primitive cell, {n_sc}-atom supercell")
    shells, complete_to = neighbour_shells(sc, ph.primitive.p2s_map, p["fc4_shell_gap"])
    sym_sc = SymfcAtoms(numbers=sc.numbers, cell=sc.cell, scaled_positions=sc.scaled_positions)
    cutoff = {3: p["fc3_cutoff"], 4: shell_cutoff(shells, p["fc4_shells"], complete_to)}
    n_basis = {int(k): int(v) for k, v in Symfc(sym_sc, cutoff=cutoff, log_level=0).estimate_basis_size(orders=[2, 3, 4]).items()}
    n_fit = max(p["min_snapshots"], math.ceil(p["snapshot_factor"] * sum(n_basis.values()) / (3 * n_sc)))
    n_test = max(1, n_fit // 10)
    disps = random_displacements(n_fit + n_test, n_sc, p["displacement"], p["random_seed"])
    forces = np.zeros_like(disps)
    for i, d in enumerate(disps):
        a = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions + d, pbc=True)
        a.calc = calculator
        forces[i] = a.get_forces()
    # Symfc.run(orders=[2, 3, 4]) would recover a dense compact FC4: solve here and recover FC4 sparse
    sf = Symfc(sym_sc, displacements=disps[:n_fit], forces=forces[:n_fit], cutoff=cutoff, log_level=0)
    sf.compute_basis_set(orders=[2, 3, 4])
    bases = [sf.basis_set[o] for o in (2, 3, 4)]
    solver = FCSolverO2O3O4(bases, log_level=0).solve(np.ascontiguousarray(disps[:n_fit]), np.ascontiguousarray(forces[:n_fit]),
                                                      batch_size=100)
    c2, c3, c4 = solver.coefs
    fcs: dict[int, Any] = {
        2: np.asarray(bases[0].compact_compression_matrix @ (bases[0].blocked_basis_set @ c2)).reshape(-1, n_sc, 3, 3),
        3: np.asarray(bases[1].compact_compression_matrix @ (bases[1].blocked_basis_set @ c3)).reshape(-1, n_sc, n_sc, 3, 3, 3),
        4: sparse_fc4_from_symfc(bases[2], c4, n_sc),
    }
    p2s = np.asarray(sf.p2s_map)
    test_f = forces[n_fit:][:, p2s]
    err = {o: np.array([predict_forces({k: fcs[k] for k in fcs if k <= o}, u) for u in disps[n_fit:]]) - test_f for o in (2, 3, 4)}
    fit = {
        "supercell_matrix": scell, "n_atoms_supercell": n_sc, "n_basis": n_basis, "n_snapshots_fit": n_fit,
        "n_snapshots_test": n_test, "displacement": p["displacement"], "cutoff_A": cutoff,
        "fc4_cutoff_rule": {"shells": p["fc4_shells"], "shell_gap": p["fc4_shell_gap"],
                            "shells_A": [[round(a, 4), round(b, 4)] for a, b in shells], "complete_below_A": round(complete_to, 3)},
        "fc4_compression_matrix_gb_est": round(mem, 3), "fc4_nonzero_clusters": len(fcs[4][0]),
        "force_rms_test": float(np.sqrt((test_f**2).mean())),
        # held-out force error with the expansion truncated at order 2, 3, 4: the drop from 3 to 4
        # shows how much the quartic term matters at this displacement
        "force_rmse_test_by_order": {o: float(np.sqrt((e**2).mean())) for o, e in err.items()},
    }
    return {"phonopy": ph, "fcs": fcs, "p2s_map": p2s, "fit": fit}


class _SupercellMap:
    """Supercell atoms <-> (primitive atom, primitive lattice vector)."""

    def __init__(self, supercell, primitive):
        self.cell = np.asarray(supercell.cell)
        self.frac = np.asarray(supercell.scaled_positions)
        self.pos = self.frac @ self.cell
        self.s2p = np.asarray(primitive.s2p_map)
        self.p2s = np.asarray(primitive.p2s_map)
        self.p_index = {int(s): i for i, s in enumerate(self.p2s)}
        self.dim = np.round(np.linalg.det(self.cell) / np.linalg.det(np.asarray(primitive.cell))).astype(int)
        self._img: dict[tuple[int, int], list[np.ndarray]] = {}
        self._lookup = {self._key(f): s for s, f in enumerate(self.frac)}

    @staticmethod
    def _key(frac: np.ndarray) -> tuple[int, int, int]:
        f = np.mod(np.round(frac, 5), 1.0)
        return tuple(int(x) for x in np.round(f * 1e5) % 100000)

    def unit_index(self, s: int) -> int:
        """1-based index of the primitive-cell atom that supercell atom s is an image of."""
        return self.p_index[int(self.s2p[s])] + 1

    def images(self, i_s: int, j_s: int) -> list[np.ndarray]:
        """Lattice vectors R (Angstrom) of the cells holding the shortest periodic images of
        atom j seen from atom i (i in the home cell). Ties are all returned."""
        key = (i_s, j_s)
        if key not in self._img:
            base = self.frac[j_s] - self.frac[i_s]
            base -= np.round(base)
            shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
            cands = (base + shifts) @ self.cell
            d = np.linalg.norm(cands, axis=1)
            r_j0 = self.pos[self.p2s[self.p_index[int(self.s2p[j_s])]]]
            r_i = self.pos[i_s]
            self._img[key] = [r_i + v - r_j0 for v in cands[d < d.min() + 1e-5]]
        return self._img[key]

    def atom_at(self, unit_index: int, R: np.ndarray) -> int:
        """Supercell atom at position r(unit atom) + R, folded into the supercell."""
        r = self.pos[self.p2s[unit_index - 1]] + R
        return self._lookup[self._key(r @ np.linalg.inv(self.cell))]


def shengbte_blocks(fc: SparseFC, order: int, supercell, primitive):
    """Sparse compact FC -> list of (R vectors, 1-based unit indices, block), the content of
    FORCE_CONSTANTS_3RD (thirdorder.py) / FORCE_CONSTANTS_4TH (Fourthorder).
    Each partner atom is placed at its shortest image from the first atom; with several equally
    short images the block is split equally over all image combinations (as thirdorder.py does)."""
    m = _SupercellMap(supercell, primitive)
    blocks = []
    for (ip, *partners), fcb in zip(*fc, strict=True):
        i_s = int(m.p2s[ip])
        img = [m.images(i_s, int(s)) for s in partners]
        n_img = math.prod(len(x) for x in img)
        idx = [int(ip) + 1] + [m.unit_index(int(s)) for s in partners]
        for choice in itertools.product(*img):
            blocks.append((list(choice), idx, fcb / n_img))
    return blocks


def fold_shengbte_blocks(blocks, order: int, supercell, primitive) -> SparseFC:
    """Inverse of ``shengbte_blocks``: back onto sparse compact keys (round-trip check)."""
    m = _SupercellMap(supercell, primitive)
    acc: dict[tuple[int, ...], np.ndarray] = {}
    for rs, idx, block in blocks:
        key = (idx[0] - 1, *(m.atom_at(u, R) for u, R in zip(idx[1:], rs, strict=True)))
        acc[key] = acc.get(key, 0) + block
    keys = sorted(acc)
    return np.array(keys, dtype=np.int64).reshape(-1, order), np.array([acc[k] for k in keys]).reshape(-1, *([3] * order))


def sparse_max_rel_dev(a: SparseFC, b: SparseFC) -> float:
    """max |a - b| / max |a| over the union of keys."""
    da = {tuple(k): v for k, v in zip(*a, strict=True)}
    db = {tuple(k): v for k, v in zip(*b, strict=True)}
    zero = np.zeros(a[1].shape[1:])
    dev = max((np.abs(da.get(k, zero) - db.get(k, zero)).max() for k in da.keys() | db.keys()), default=0.0)
    return float(dev / max(np.abs(a[1]).max() if len(a[1]) else 0.0, 1e-30))


def write_shengbte_fc(blocks, order: int, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"{len(blocks)}\n")
        for n, (rs, idx, block) in enumerate(blocks, 1):
            fh.write(f"\n{n}\n")
            for r in rs:
                fh.write(" ".join(f"{x:.10f}" for x in r) + "\n")
            fh.write(" ".join(str(x) for x in idx) + "\n")
            for comp in np.ndindex(*([3] * order)):
                fh.write(" ".join(str(c + 1) for c in comp) + f" {block[comp]:.10e}\n")


def read_shengbte_fc(path: str, order: int):
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    n, pos, out = int(lines[0]), 1, []
    for _ in range(n):
        pos += 1  # block number
        rs = [np.array(lines[pos + i].split(), float) for i in range(order - 1)]
        pos += order - 1
        idx = [int(x) for x in lines[pos].split()]
        pos += 1
        block = np.zeros([3] * order)
        for _c in range(3**order):
            parts = lines[pos].split()
            block[tuple(int(x) - 1 for x in parts[:order])] = float(parts[order])
            pos += 1
        out.append((rs, idx, block))
    return out


def shengbte_control(prim, scell: list[int], ngrid: list[int], p: dict[str, Any], four_phonon: bool) -> str:
    """CONTROL namelist file (ShengBTE manual; FourPhonon adds four_phonon / four_phonon_iteration).
    Lattice vectors in nm with lfactor = 1."""
    symbols = list(dict.fromkeys(prim.symbols))
    types = [symbols.index(s) + 1 for s in prim.symbols]
    cell = np.asarray(prim.cell) * 0.1
    flag = lambda b: ".TRUE." if b else ".FALSE."  # noqa: E731
    lines = [
        "&allocations",
        f"\tnelements={len(symbols)}",
        f"\tnatoms={len(prim)}",
        f"\tngrid(:)={ngrid[0]} {ngrid[1]} {ngrid[2]}",
        "&end",
        "&crystal",
        "\tlfactor=1.0",
        *[f"\tlattvec(:,{i + 1})={v[0]:.10f} {v[1]:.10f} {v[2]:.10f}" for i, v in enumerate(cell)],
        "\telements=" + " ".join(f'"{s}"' for s in symbols),
        "\ttypes=" + " ".join(str(t) for t in types),
        *[f"\tpositions(:,{i + 1})={x[0]:.10f} {x[1]:.10f} {x[2]:.10f}" for i, x in enumerate(np.asarray(prim.scaled_positions))],
        f"\tscell(:)={scell[0]} {scell[1]} {scell[2]}",
        "&end",
        "&parameters",
        f"\tT={p['temperature']}",
        f"\tscalebroad={p['scalebroad']}",
        # sampling only for the 4ph part (3ph stays exact); FourPhonon refuses it with four_phonon=.FALSE.
        *([f"\tnum_sample_process_4ph_phase_space={n4}", f"\tnum_sample_process_4ph={n4}"]
          if four_phonon and (n4 := int(p.get("num_sample_4ph", -1))) > 0 else []),
        "&end",
        "&flags",
        "\tnonanalytic=.FALSE.",
        f"\tisotopes={flag(p['is_isotope'])}",
        f"\tconvergence={flag(p['iterative'])}",
        f"\tfour_phonon={flag(four_phonon)}",
        f"\tfour_phonon_iteration={flag(p['iterative'] and four_phonon)}",
        "&end",
    ]
    return "\n".join(lines) + "\n"


def parse_shengbte_kappa(run_dir: Path) -> dict[str, Any] | None:
    """kappa from T<T>K/BTE.kappa_tensor (FourPhonon's Example-Si output; Example-Si-sampling rerun, job 963770):
    one row per iteration, ``iter k_xx k_xy ... k_zz``; row 0 is RTA, the last row the iterative
    solution (only row 0 without ``convergence``). ``kappa_avg`` is the last row."""
    for f in [run_dir / "BTE.kappa_tensor", *sorted(run_dir.glob("T*K/BTE.kappa_tensor"))]:
        if f.exists():
            rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            if not rows:
                return None
            tensors = [np.array(r[-9:], float).reshape(3, 3) for r in rows]
            avgs = [float(np.trace(t) / 3) for t in tensors]
            return {
                "file": str(f), "kappa_tensor": tensors[-1].tolist(), "kappa_avg": avgs[-1],
                "kappa_rta_avg": avgs[0], "n_iter": int(float(rows[-1][0])) if len(rows[-1]) > 9 else 0,
                # change in the last iteration; ShengBTE stops at eps (default 1e-5) or maxiter (1000)
                "last_rel_change": abs(avgs[-1] / avgs[-2] - 1) if len(avgs) > 1 and avgs[-2] else None,
            }
    # FourPhonon's shipped reference outputs keep only the per-temperature summaries (Example-Si-sampling):
    # ``T k_xx ... k_zz [n_iter]`` per row; first temperature, RTA from the _RTA file
    conv, rta = run_dir / "BTE.KappaTensorVsT_CONV", run_dir / "BTE.KappaTensorVsT_RTA"
    final = conv if conv.exists() else rta
    if final.exists():
        def first_row(f: Path) -> list[str] | None:
            rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            return rows[0] if rows else None

        row, rrow = first_row(final), first_row(rta) if rta.exists() else None
        if row:
            t = np.array(row[1:10], float).reshape(3, 3)
            return {
                "file": str(final), "temperature": float(row[0]), "kappa_tensor": t.tolist(),
                "kappa_avg": float(np.trace(t) / 3),
                "kappa_rta_avg": float(np.trace(np.array(rrow[1:10], float).reshape(3, 3)) / 3) if rrow else None,
                "n_iter": int(float(row[10])) if len(row) > 10 else 0, "last_rel_change": None,
            }
    return None


def _fourphonon_cmd(p: dict[str, Any]) -> str | None:
    return p.get("fourphonon_cmd") or os.environ.get("MGA_FOURPHONON")


def phono3py_from_fit(prim: Atoms, fit: dict[str, Any], kappa_p: dict[str, Any], q_mesh) -> tuple[dict[str, Any], float]:
    """Wigner-RTA kappa (same solver as L1) from the symfc FC2/FC3: separates "different FCs"
    (random 0.03 A + fit vs finite differences) from "different solver" (FourPhonon vs phono3py)."""
    from phono3py import Phono3py

    ph3 = Phono3py(_phonopy_atoms(prim), supercell_matrix=np.diag(fit["fit"]["supercell_matrix"]), primitive_matrix=None,
                   symprec=kappa_p["symprec"])
    if not np.array_equal(np.asarray(ph3.primitive.p2s_map), fit["p2s_map"]):
        raise RuntimeError("phono3py and symfc disagree on the primitive atoms of the supercell")
    ph3.fc2 = fit["fcs"][2]
    ph3.fc3 = fit["fcs"][3]
    ph3.mesh_numbers = q_mesh
    arrays, _ = conductivity(ph3, kappa_p)
    return arrays, float(np.asarray(arrays["frequencies"]).min())


def run_fc4(atoms: Atoms, calculator, relax_p: dict[str, Any], kappa_p: dict[str, Any], p: dict[str, Any],
            settings: dict[str, Any] | None, work_dir: str) -> dict[str, Any]:
    """L3 for one structure: relax (as L1) -> primitive cell -> symfc FC2/FC3/FC4 -> ShengBTE /
    FourPhonon inputs (with a write/fold round-trip check) -> phono3py kappa from the fitted
    FC2/FC3 -> FourPhonon 3ph and 3+4ph runs if a command is configured.

    The number used downstream is the ratio kappa(3+4ph) / kappa(3ph) from one code on one
    q-mesh, applied to the committee L1 kappa; FourPhonon's absolute kappa (particle picture,
    no coherence term) is not mixed with the Wigner kappa."""
    out: dict[str, Any] = {"ok": False, "stage": "relax", "error": None}
    t0 = time.perf_counter()
    try:
        relaxed, out["relax"] = relax(atoms, calculator, relax_p)
        prim = primitive_atoms(relaxed, kappa_p["symprec"])
        q_mesh = (settings or {}).get("q_mesh") or kappa_p["q_mesh_length"]
        out["stage"] = "fit"
        fit = fit_fc234(prim, calculator, {**p, "symprec": kappa_p["symprec"]})
        out["fit"] = fit["fit"]
        ph = fit["phonopy"]
        out["stage"] = "write"
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        from phonopy.file_IO import write_FORCE_CONSTANTS
        from phonopy.harmonic.force_constants import compact_fc_to_full_fc

        write_FORCE_CONSTANTS(compact_fc_to_full_fc(ph.primitive, fit["fcs"][2]), filename=str(wd / "FORCE_CONSTANTS_2ND"))
        roundtrip, n_blocks = {}, {}
        for order, name in ((3, "FORCE_CONSTANTS_3RD"), (4, "FORCE_CONSTANTS_4TH")):
            fc = fit["fcs"][order] if order == 4 else dense_to_sparse(fit["fcs"][order], order)
            blocks = shengbte_blocks(fc, order, ph.supercell, ph.primitive)
            write_shengbte_fc(blocks, order, str(wd / name))
            n_blocks[order] = len(blocks)
            back = fold_shengbte_blocks(read_shengbte_fc(str(wd / name), order), order, ph.supercell, ph.primitive)
            roundtrip[order] = sparse_max_rel_dev(fc, back)
        out["shengbte_files"] = {"dir": str(wd), "n_blocks": n_blocks, "roundtrip_max_rel_dev": roundtrip}
        mesh = q_mesh if not isinstance(q_mesh, float) else [int(x) for x in _length_mesh(prim, q_mesh)]
        ngrid = [int(x) for x in (p.get("q_mesh") or mesh)]
        for tag, four in (("3ph", False), ("3+4ph", True)):
            d = wd / tag.replace("+", "_")
            d.mkdir(exist_ok=True)
            (d / "CONTROL").write_text(shengbte_control(ph.primitive, fit["fit"]["supercell_matrix"], ngrid, p, four))
            for name in ("FORCE_CONSTANTS_2ND", "FORCE_CONSTANTS_3RD") + (("FORCE_CONSTANTS_4TH",) if four else ()):
                link = d / name
                if not link.exists():
                    link.symlink_to(Path("..") / name)
        out["stage"] = "phono3py_check"
        arrays, out["min_frequency_fit_thz"] = phono3py_from_fit(prim, fit, kappa_p, mesh)
        out["temperatures"] = list(kappa_p["temperatures"])
        out["kappa_tot_fit_avg"] = kappa_avg(arrays["kappa_tot_rta"]).tolist()
        out["kappa_p_fit_avg"] = kappa_avg(arrays["kappa_p_rta"]).tolist()
        cmd = _fourphonon_cmd(p)
        out["fourphonon"] = {"cmd": cmd, "ngrid": ngrid, "iterative": p["iterative"], "temperature": p["temperature"]}
        if cmd:
            out["stage"] = "fourphonon"
            import shlex

            for tag in ("3ph", "3_4ph"):
                d = wd / tag
                r = subprocess.run(shlex.split(cmd), cwd=d, capture_output=True, text=True, timeout=p["fourphonon_timeout_s"])
                (d / "stdout.log").write_text(r.stdout)
                (d / "stderr.log").write_text(r.stderr)
                out["fourphonon"][tag] = {"returncode": r.returncode, **(parse_shengbte_kappa(d) or {"kappa_avg": None})}
            k3, k34 = out["fourphonon"]["3ph"]["kappa_avg"], out["fourphonon"]["3_4ph"]["kappa_avg"]
            out["four_phonon_ratio"] = k34 / k3 if k3 and k34 else None
        else:
            out["fourphonon"]["skipped"] = "no FourPhonon command (fourphonon_cmd or $MGA_FOURPHONON): inputs written only"
            out["four_phonon_ratio"] = None
        out["ok"], out["stage"] = True, "done"
        out["relaxed_structure"] = _structure_dict(relaxed)
    except Exception as exc:  # noqa: BLE001
        _failure(out, exc)
    out["wall_time_s"] = round(time.perf_counter() - t0, 2)
    out["arrays"] = {}
    return out


def _length_mesh(prim: Atoms, length: float) -> list[int]:
    """phono3py's length rule: n_i = max(1, round(length * |b_i|)) with b the reciprocal vectors."""
    rec = np.linalg.inv(np.asarray(prim.cell)).T
    return [max(1, int(round(length * np.linalg.norm(b)))) for b in rec]
