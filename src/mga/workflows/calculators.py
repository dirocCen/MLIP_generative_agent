"""Build ASE calculators from a model spec dict (``ModelSpec.worker_dict()``).

Imports of the MLIP packages are local to each branch: a worker environment only
has the package for its own model.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any

_VERSION_PACKAGES = (
    "mga", "numpy", "ase", "spglib", "phonopy", "phono3py", "torch", "mace-torch",
    "deepmd-kit", "nequip", "upet", "metatomic-torch", "metatomic-ase", "metatrain", "tace", "calorine", "pymatgen", "jobflow", "atomate2", "mattergen",
)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_calculator(model: dict[str, Any], dtype: str = "float64"):
    """Return (calculator, info). ``info`` records what was actually used."""
    loader = model["loader"]
    ckpt = model.get("checkpoint")
    if dtype == "model_default":  # normally resolved at submission (tools/mlip._resolve_mlip)
        dtype = model.get("dtype") or "float64"
    info: dict[str, Any] = {"model": model["name"], "loader": loader, "checkpoint": ckpt}
    if loader != "emt" and (not ckpt or ckpt == "TODO"):
        raise ValueError(f"Model {model['name']!r} has no checkpoint configured (configs/models.yaml)")

    if loader == "emt":
        from ase.calculators.emt import EMT

        calc = EMT()
        info["dtype"] = "float64"
    elif loader == "mace_mp":
        from mace.calculators import mace_mp

        device = _device()
        calc = mace_mp(model=ckpt, default_dtype=dtype, device=device)
        info.update(dtype=dtype, device=device)
    elif loader == "mace":
        from mace.calculators import MACECalculator

        device = _device()
        calc = MACECalculator(model_paths=ckpt, default_dtype=dtype, device=device)
        info.update(dtype=dtype, device=device)
    elif loader == "deepmd":
        calc = _deepmd(model, ckpt, info)
    elif loader == "nequip_compiled":
        from nequip.ase import NequIPCalculator

        device = _device()
        calc = NequIPCalculator.from_compiled_model(compile_path=ckpt, device=device)
        info.update(dtype="model-defined", device=device)
    elif loader == "upet":
        calc = _pet(ckpt, dtype, info)
    elif loader == "tace":
        # API: TACE docs, guide/ase.rst. Eager mode; the leaderboard additionally exports an
        # OpenEquivariance + AOTInductor package, which changes speed, not the model.
        from tace.interface.ase import TACEAseCalc

        device = _device()
        calc = TACEAseCalc(ckpt, device=device)
        info.update(dtype="model-defined (TACE default)", device=device, acceleration="none (eager)")
    elif loader == "calorine_nep":
        from calorine.calculators import CPUNEP

        calc = CPUNEP(ckpt)
        info.update(dtype="model-defined", device="cpu")
    else:
        raise ValueError(f"Unknown loader {loader!r}")
    return calc, info


def _gpu_tag() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return re.sub(r"[^\w.-]+", "-", torch.cuda.get_device_name(0)).strip("-")
    except ImportError:
        pass
    return "cpu"


def _export_once(cmd: list[str], out: str) -> str:
    """Run an export command writing ``{output}`` once; atomic rename so parallel jobs never
    read a half-written file. Returns ``out``."""
    if os.path.exists(out):
        return out
    tmp = f"{out}.tmp-{os.getpid()}{os.path.splitext(out)[1]}"
    # the console script of the running env, even when the env is not activated (batch jobs
    # call $ENV/bin/python directly, so $ENV/bin need not be on PATH)
    exe = os.path.join(os.path.dirname(sys.executable), cmd[0])
    cmd = [exe if os.access(exe, os.X_OK) else (shutil.which(cmd[0]) or cmd[0]), *cmd[1:]]
    subprocess.run([c.replace("{output}", tmp) for c in cmd], check=True)
    os.replace(tmp, out)
    return out


def _deepmd(model: dict[str, Any], ckpt: str, info: dict[str, Any]):
    """DeePMD. A DPA4 training checkpoint (.pt) is frozen once per GPU type to a torch-export
    .pt2 with ``dp --pt freeze`` and loaded with the native neighbor list, as in
    matbench-discovery calculators.py (_deepmd_graph, c70947f)."""
    from deepmd.calculator import DP

    kwargs: dict[str, Any] = {"head": model["head"]} if model.get("head") else {}
    if ckpt.endswith(".pt"):
        frozen = _export_once(["dp", "--pt", "freeze", "-c", ckpt, "-o", "{output}"],
                              f"{ckpt[:-3]}-frozen-{_gpu_tag()}.pt2")
        info.update(frozen=frozen)
        ckpt, kwargs = frozen, {**kwargs, "nlist_backend": "native"}
    calc = DP(ckpt, **kwargs)
    info.update(dtype="model-defined", head=model.get("head"), dp_kwargs=sorted(kwargs))
    return calc


def _pet(ckpt: str, dtype: str, info: dict[str, Any]):
    """PET as on the leaderboard (matbench-discovery calculators.py _pet + PetKappaAdapter,
    c70947f): metatrain .ckpt -> ``mtt export`` -> MetatomicCalculator with conservative forces,
    wrapped in metatomic's rotational symmetrizer (PET is not equivariant by construction)."""
    import torch
    from metatomic.torch import load_atomistic_model

    try:
        from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator
    except ImportError:  # older metatomic: calculator in metatomic.torch
        from metatomic.torch.ase_calculator import MetatomicCalculator
        from metatomic_ase import SymmetrizedCalculator

    device = _device()
    pt = ckpt
    if ckpt.endswith(".ckpt"):
        pt = _export_once(["mtt", "export", ckpt, "-o", "{output}"], ckpt[:-5] + ".pt")
    m = load_atomistic_model(pt)
    m.capabilities().dtype = dtype
    m = m.to(dtype={"float32": torch.float32, "float64": torch.float64}[dtype], device=device)
    base = MetatomicCalculator(m, device=device, non_conservative=False)
    # batch_size only splits the quadrature rotations into GPU batches (same result); the default
    # (all rotations at once) ran out of memory on a 44 GB L40S for a 64-atom cell (11-03).
    # Measured on an 80 GB A100 with a 512-atom cell: float32 fits 4 (job 963761), float64 fits 2
    # but not 4 (jobs 965975 / 963847). Memory goes with rotations x atoms, so the budget is
    # 2048 (float32) / 1024 (float64) rotated atoms per 80 GB, scaled by the card's memory, and the
    # batch is set per structure (FC2 supercells reach ~1200 atoms, e.g. R3m BNC2).
    # MGA_PET_ROTATION_BATCH fixes it instead.
    fixed = os.environ.get("MGA_PET_ROTATION_BATCH")
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device == "cuda" else 80.0
    budget = (1024 if dtype == "float64" else 2048) * gpu_gb / 80.0

    class _Symmetrized(SymmetrizedCalculator):
        def calculate(self, atoms=None, *args, **kwargs):
            if fixed is None and atoms is not None:
                self.batch_size = max(1, int(budget // len(atoms)))
            return super().calculate(atoms, *args, **kwargs)

    calc = _Symmetrized(base, include_inversion=False, batch_size=int(fixed) if fixed else 4)
    info.update(dtype=dtype, device=device, exported=pt, non_conservative=False,
                rotation_batch_size=int(fixed) if fixed else f"floor({budget:.0f} / n_atoms)",
                rotational_symmetrization="metatomic_ase.SymmetrizedCalculator(include_inversion=False, l_max=3 default)")
    return calc


def worker_info() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for pkg in _VERSION_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return {
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
    }
