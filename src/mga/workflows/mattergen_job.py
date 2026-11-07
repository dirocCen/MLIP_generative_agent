"""MatterGen generation as a jobflow job (runs ``mattergen-generate`` in a subprocess).

TODO verify on datalab: CLI flag names and the output file name below follow the
MatterGen README (``generated_crystals.extxyz``); check them against the
installed version before the first live run.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from jobflow import job


def mattergen_command(out_dir: str, chemical_system: str, n_samples: int, p: dict[str, Any]) -> list[str]:
    batch = min(p["batch_size"], n_samples)
    num_batches = -(-n_samples // batch)
    return [
        "mattergen-generate",
        out_dir,
        f"--pretrained-name={p['pretrained_name']}",
        f"--batch_size={batch}",
        f"--num_batches={num_batches}",
        f"--properties_to_condition_on={{'chemical_system': '{chemical_system}'}}",
        f"--diffusion_guidance_factor={p['diffusion_guidance_factor']}",
        f"--record_trajectories={p['record_trajectories']}",
    ]


@job
def mattergen_generate(chemical_system: str, n_samples: int, params: dict[str, Any]) -> dict[str, Any]:
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor

    from mga.workflows.calculators import worker_info

    out_dir = Path.cwd() / "mattergen_out"
    cmd = mattergen_command(str(out_dir), chemical_system, n_samples, params)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"mattergen-generate failed ({proc.returncode}): {proc.stderr[-2000:]}")
    frames = read(out_dir / "generated_crystals.extxyz", index=":")
    structures = [AseAtomsAdaptor.get_structure(a).as_dict() for a in frames][:n_samples]
    return {
        "chemical_system": chemical_system,
        "command": cmd,
        "n_requested": n_samples,
        "n_returned": len(structures),
        "structures": structures,
        "stdout_tail": proc.stdout[-2000:],
        "worker_info": worker_info(),
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }
