"""Structure registry and immutable named sets (plan §3.4).

- A structure ID is ``<reduced formula>-<sha1[:10]>`` of a canonical, rounded
  serialisation, so the same file imported twice gets the same ID and a changed
  structure can never silently keep an old ID (``load_structure`` re-checks it).
- Each entry records ``origin`` (where it came from) and ``parents`` (which
  registered structures it was derived from, e.g. MLIP-relaxed from a MatterGen
  sample). The leakage check walks ``parents``.
- Sets are lists of IDs with provenance. Tools pass set names instead of putting
  thousands of IDs into the LLM context. A set name can never be reused for
  different content.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from pymatgen.core import Structure

from mga.config import paths, utc_now
from mga.utils import canonical_json, check_name, read_json, write_json_atomic

_LATTICE_DECIMALS = 4  # Angstrom
_FRAC_DECIMALS = 5


class RegistryError(ValueError):
    pass


def _canonical_payload(structure: Structure) -> dict[str, Any]:
    lattice = np.round(structure.lattice.matrix, _LATTICE_DECIMALS) + 0.0
    sites = []
    for site in structure:
        frac = np.round(np.mod(site.frac_coords, 1.0), _FRAC_DECIMALS) % 1.0 + 0.0
        sites.append((site.species_string, [float(x) for x in frac]))
    sites.sort()
    return {"lattice": lattice.tolist(), "sites": sites}


def structure_id(structure: Structure) -> str:
    digest = hashlib.sha1(canonical_json(_canonical_payload(structure)).encode()).hexdigest()
    formula = structure.composition.reduced_formula
    if "(" in formula:  # pymatgen groups units, e.g. B4(CN)3; ids allow no brackets. Others keep their old id
        formula = structure.composition.reduced_composition.formula.replace(" ", "")
    return f"{formula}-{digest[:10]}"


def _entry_path(sid: str):
    return paths.structures / f"{check_name(sid, 'structure id')}.json"


def _symmetry(structure: Structure) -> dict[str, Any]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-3)
        return {"spacegroup_number": sga.get_space_group_number(), "spacegroup_symbol": sga.get_space_group_symbol(), "symprec": 1e-3}
    except Exception as exc:  # noqa: BLE001 - spglib can fail on odd cells
        return {"spacegroup_number": None, "spacegroup_symbol": None, "symprec": 1e-3, "error": str(exc)}


def register_structure(
    structure: Structure,
    *,
    origin: dict[str, Any],
    parents: Iterable[str] = (),
    info: dict[str, Any] | None = None,
    step_id: str | None = None,
) -> tuple[str, bool]:
    """Register a structure; return (sid, created). Existing entries are never modified."""
    sid = structure_id(structure)
    path = _entry_path(sid)
    if path.exists():
        return sid, False
    parents = sorted(set(parents))
    for parent in parents:
        if not _entry_path(parent).exists():
            raise RegistryError(f"Parent structure {parent!r} is not registered")
    entry = {
        "sid": sid,
        "formula": structure.composition.formula.replace(" ", ""),
        "reduced_formula": structure.composition.reduced_formula,
        "chemsys": structure.composition.chemical_system,
        "elements": sorted(el.symbol for el in structure.composition.elements),
        "nsites": len(structure),
        "volume_per_atom": round(structure.volume / len(structure), 4),
        **_symmetry(structure),
        "origin": origin,
        "parents": parents,
        "info": info or {},
        "registered_by_step": step_id,
        "created_at": utc_now(),
        "structure": structure.as_dict(),
    }
    write_json_atomic(path, entry)
    return sid, True


def load_entry(sid: str) -> dict[str, Any]:
    path = _entry_path(sid)
    if not path.exists():
        raise RegistryError(f"Unknown structure id {sid!r}")
    return read_json(path)


def load_structure(sid: str) -> Structure:
    structure = Structure.from_dict(load_entry(sid)["structure"])
    if structure_id(structure) != sid:
        raise RegistryError(f"Structure file for {sid!r} was modified after registration")
    return structure


def ancestors(sid: str) -> set[str]:
    """All registered ancestors of ``sid`` (not including itself)."""
    seen: set[str] = set()
    stack = list(load_entry(sid)["parents"])
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        stack.extend(load_entry(parent)["parents"])
    return seen


# --- named sets ---------------------------------------------------------------
def _set_path(name: str):
    return paths.sets / f"{check_name(name, 'set name')}.json"


def create_set(
    name: str,
    sids: Iterable[str],
    *,
    operation: str,
    parent_sets: Iterable[str] = (),
    description: str = "",
    details: dict[str, Any] | None = None,
    step_id: str | None = None,
) -> dict[str, Any]:
    """Create an immutable set. Re-creating with identical members is a no-op."""
    members = list(dict.fromkeys(sids))  # keep order, drop repeats
    for sid in members:
        if not _entry_path(sid).exists():
            raise RegistryError(f"Set {name!r}: structure {sid!r} is not registered")
    path = _set_path(name)
    if path.exists():
        existing = read_json(path)
        if existing["sids"] == members:
            return existing
        raise RegistryError(
            f"Set {name!r} already exists with different members ({existing['n']} vs {len(members)}); "
            "sets are immutable, choose a new name"
        )
    doc = {
        "name": name,
        "n": len(members),
        "sids": members,
        "operation": operation,
        "parent_sets": list(parent_sets),
        "description": description,
        "details": details or {},
        "created_by_step": step_id,
        "created_at": utc_now(),
    }
    write_json_atomic(path, doc)
    return doc


def load_set(name: str) -> dict[str, Any]:
    path = _set_path(name)
    if not path.exists():
        raise RegistryError(f"Unknown set {name!r}; existing: {[s['name'] for s in iter_sets()][:50]}")
    return read_json(path)


def set_sids(name: str) -> list[str]:
    return list(load_set(name)["sids"])


def iter_sets() -> Iterator[dict[str, Any]]:
    for path in sorted(paths.sets.glob("*.json")):
        doc = read_json(path)
        yield {k: doc[k] for k in ("name", "n", "operation", "parent_sets", "description", "created_at")}


def iter_entries(sids: Iterable[str]) -> Iterator[dict[str, Any]]:
    for sid in sids:
        yield load_entry(sid)
