"""Install smoke test for one MLIP on a worker (DESIGN §8 item 4; plan §8 item 5).

Builds the calculator through mga's own loader (workflows/calculators.py), then checks
  1. energy / forces / stress on c-BN are finite and forces on the ideal cell vanish,
  2. rotation: E unchanged and F rotates with the structure (PET relies on its symmetrizer),
  3. timing of one force call on 64 / 216 / 512-atom c-BN supercells (input to the budget ledger).
Writes a JSON report. Run on a compute node, never the login node (DESIGN §11).

    python smoke_calculator.py --name tece_oam_rra --loader tace --checkpoint /path/TECE-OAM-RRA-1.0.pt --out report.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from ase.build import bulk
from scipy.spatial.transform import Rotation

from mga.workflows.calculators import build_calculator, worker_info


def _efs(atoms, calc):
    atoms = atoms.copy()
    calc.reset()  # ASE caches by atoms state: an identical copy would return the previous result
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _free() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_gb() -> float | None:
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    except ImportError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    report: dict = {"worker": worker_info(), "ok": False}
    t0 = time.perf_counter()
    calc, info = build_calculator({"name": a.name, "loader": a.loader, "checkpoint": a.checkpoint, "head": a.head}, a.dtype)
    report["load_seconds"] = time.perf_counter() - t0
    report["calculator"] = info

    cbn = bulk("BN", "zincblende", a=3.615)
    e, f, s = _efs(cbn, calc)
    report["cbn_primitive"] = {"energy_per_atom": e / len(cbn), "max_abs_force": float(np.abs(f).max()),
                               "stress_GPa": (s / 1e9 * 1.602176634e-19 / 1e-30).tolist()}

    # rotation test on a rattled 2x2x2 conventional cell (64 atoms): a model that cannot do this
    # cannot do kappa supercells either, so an OOM here is a failed smoke test
    sc = bulk("BN", "zincblende", a=3.615, cubic=True).repeat(2)
    sc.rattle(0.03, seed=1)
    e0, f0, _ = _efs(sc, calc)
    R = Rotation.random(random_state=7).as_matrix()
    rot = sc.copy()
    rot.set_cell(sc.cell.array @ R.T, scale_atoms=False)
    rot.positions = sc.positions @ R.T
    e1, f1, _ = _efs(rot, calc)
    report["rotation"] = {"dE_per_atom_eV": abs(e1 - e0) / len(sc), "max_dF_eV_per_A": float(np.abs(f1 - f0 @ R.T).max()),
                          "max_abs_force_rattled": float(np.abs(f0).max())}
    if hasattr(calc, "base_calculator") or a.loader == "upet":  # PET: same test without the symmetrizer
        base = getattr(calc, "base_calculator", None) or getattr(calc, "calculator", None)
        if base is not None:
            _, g0, _ = _efs(sc, base)
            _, g1, _ = _efs(rot, base)
            report["rotation"]["max_dF_unsymmetrized"] = float(np.abs(g1 - g0 @ R.T).max())

    # timing + peak memory per size; an out-of-memory error is a result (largest usable
    # supercell for this model on this GPU), not a failed install
    timing, mem = {}, {}
    for n in (2, 3, 4):
        cell = bulk("BN", "zincblende", a=3.615, cubic=True).repeat(n)
        cell.rattle(0.01, seed=2)
        key = str(len(cell))
        if "oom_at_n_atoms" in report:  # larger cells will not fit either
            timing[key] = mem[key] = None
            continue
        _free()
        try:
            _efs(cell, calc)  # warm-up (compilation, neighbor-list allocation)
            _sync()
            t = time.perf_counter()
            for _ in range(3):
                _efs(cell, calc)
            _sync()
            timing[key] = (time.perf_counter() - t) / 3
        except Exception as exc:  # noqa: BLE001  (torch.OutOfMemoryError, or a model's own OOM wrapper)
            # deepmd raises its own OutOfMemoryError ("... out-of-memory (OOM) error ...", 11-04)
            text = f"{type(exc).__name__} {exc}".lower().replace("-", " ")
            if "out of memory" not in text and "outofmemory" not in text:
                raise
            timing[key] = None
            report.setdefault("oom_at_n_atoms", int(key))
        mem[key] = _peak_gb()
    report["seconds_per_force_call"] = timing
    report["peak_gpu_mem_gb"] = mem

    report["ok"] = bool(
        np.isfinite(e) and report["cbn_primitive"]["max_abs_force"] < 1e-3
        and report["rotation"]["dE_per_atom_eV"] < 1e-5 and report["rotation"]["max_dF_eV_per_A"] < 1e-3
    )
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
