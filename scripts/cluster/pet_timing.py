"""Where does PET's time go? (11-04: 67 s per force call for 64 atoms on an L40S in float64.)

Times one force call on a rattled 64-atom c-BN cell for float64 / float32, with and without the
rotational symmetrizer, and reports the number of quadrature rotations. Run on a compute node.

    python pet_timing.py --checkpoint /path/pet-oam-xl-v1.0.0.ckpt --out pet_timing_<gpu>.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from ase.build import bulk

from mga.workflows.calculators import build_calculator, worker_info


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(calc, atoms, n=2):
    a = atoms.copy()
    calc.reset()
    a.calc = calc
    a.get_forces()  # warm-up
    _sync()
    t = time.perf_counter()
    for _ in range(n):
        calc.reset()
        a.get_forces()
    _sync()
    return (time.perf_counter() - t) / n, a.get_forces()


def _n_rotations(calc):
    for k, v in vars(calc).items():
        shape = getattr(v, "shape", None)
        if shape is not None and len(shape) == 3 and tuple(shape[1:]) == (3, 3):
            return k, int(shape[0])
        if isinstance(v, (list, tuple)) and v and getattr(v[0], "shape", None) == (3, 3):
            return k, len(v)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cell = bulk("BN", "zincblende", a=3.615, cubic=True).repeat(2)
    cell.rattle(0.01, seed=2)
    rep = {"worker": worker_info(), "n_atoms": len(cell), "runs": {}}
    ref = None
    for dtype in ("float64", "float32"):
        calc, info = build_calculator({"name": "pet_oam_xl", "loader": "upet", "checkpoint": a.checkpoint}, dtype)
        base = getattr(calc, "base_calculator", None) or getattr(calc, "calculator", None)
        r = {"rotation_attr": None, "n_rotations": None}
        r["rotation_attr"], r["n_rotations"] = _n_rotations(calc)
        r["calc_attrs"] = sorted(vars(calc))
        if base is not None:
            r["s_base"], fb = _time(base, cell)
        r["s_symmetrized"], fs = _time(calc, cell, n=1)
        if ref is None:
            ref = fs
        r["max_dF_vs_float64_symmetrized"] = float(np.abs(fs - ref).max())
        rep["runs"][dtype] = r
        print(dtype, json.dumps(r, default=str), flush=True)
        del calc, base
        import torch

        torch.cuda.empty_cache()
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
