"""Every test runs in a throw-away project root (MGA_PROJECT_ROOT).

configs/, docs/ and data/reference/ are symlinked from the repository (read-only
inputs); registry, sets, records, results and approvals are written to tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    for rel in ("configs", "docs"):
        (root / rel).symlink_to(REPO / rel)
    (root / "data" / "reference").symlink_to(REPO / "data" / "reference")
    monkeypatch.setenv("MGA_PROJECT_ROOT", str(root))
    monkeypatch.setenv("MGA_MODE", "dry_run")
    monkeypatch.setenv("MGA_ACTOR", "agent")
    monkeypatch.setenv("MGA_RUN_ID", "test")
    monkeypatch.delenv("MP_API_KEY", raising=False)
    return root


def si_primitive() -> Structure:
    a = 5.43
    lat = Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    return Structure(lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def si_conventional() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])


def rocksalt(a: str, b: str, lattice: float) -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(lattice), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def al_fcc() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]]).get_primitive_structure()


def write_json_structures(path: Path, structures: list[Structure]) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"structures": [s.as_dict() for s in structures]}))
    return path
