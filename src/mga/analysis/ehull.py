"""Energy above the MP convex hull for MLIP energies (DESIGN §1.4, plan §4).

- MP entries: ``get_entries_in_chemsys`` with ``thermo_types=["GGA_GGA+U"]``, i.e.
  MP2020-corrected GGA/GGA+U, cached per chemical system in
  ``data/reference/mp/entries/<chemsys>.json`` with the MP database version.
- MLIP entries get the same MP2020 corrections. The +U values an MP calculation
  would use are read from ``MPRelaxSet`` for that structure, so an MLIP trained
  on MPtrj is corrected as the DFT calculation it imitates. POTCAR checks are off
  (there is no POTCAR behind an MLIP energy).
- Only models with ``energy_reference: mp_pbe`` may be compared with MP.
"""

from __future__ import annotations

import os
from typing import Any

from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from mga.config import get_mode, paths, utc_now
from mga.utils import read_json, write_json_atomic

THERMO_TYPE = "GGA_GGA+U"


class MPUnavailable(RuntimeError):
    pass


def chemsys_key(elements) -> str:
    return "-".join(sorted(set(elements)))


def _rester():
    if not os.environ.get("MP_API_KEY"):
        raise MPUnavailable("MP_API_KEY is not set (docs/mcp_servers_plan.md §8 item 4)")
    from mp_api.client import MPRester

    return MPRester(mute_progress_bars=True)


def _cache_path(kind: str, chemsys: str):
    return paths.reference / "mp" / kind / f"{chemsys}.json"


def _cached(kind: str, chemsys: str, refresh: bool, fetch) -> dict[str, Any]:
    path = _cache_path(kind, chemsys)
    if path.exists() and not refresh:
        return read_json(path)
    if get_mode() == "replay":
        raise MPUnavailable(f"replay mode: no cached MP {kind} for {chemsys}")
    with _rester() as mpr:
        doc = {"chemsys": chemsys, "retrieved_at": utc_now(), "db_version": mpr.get_database_version(), **fetch(mpr)}
    write_json_atomic(path, doc)
    return doc


def mp_entries(chemsys: str, *, refresh: bool = False) -> tuple[list[ComputedStructureEntry], dict[str, Any]]:
    """All MP2020-corrected GGA/GGA+U entries of ``chemsys`` and its subsystems."""

    def fetch(mpr):
        entries = mpr.get_entries_in_chemsys(chemsys, additional_criteria={"thermo_types": [THERMO_TYPE]})
        return {"thermo_type": THERMO_TYPE, "entries": [e.as_dict() for e in entries]}

    doc = _cached("entries", chemsys, refresh, fetch)
    meta = {k: doc[k] for k in ("chemsys", "retrieved_at", "db_version", "thermo_type")}
    return [ComputedStructureEntry.from_dict(d) for d in doc["entries"]], meta


def mp_structures(chemsys: str, *, refresh: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """MP materials with exactly these elements (for novelty), non-deprecated."""

    def fetch(mpr):
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            deprecated=False,
            fields=["material_id", "formula_pretty", "structure", "energy_above_hull", "theoretical"],
        )
        return {
            "materials": [
                {
                    "material_id": str(d.material_id),
                    "formula": d.formula_pretty,
                    "energy_above_hull": d.energy_above_hull,
                    "theoretical": d.theoretical,
                    "structure": d.structure.as_dict(),
                }
                for d in docs
            ]
        }

    doc = _cached("structures", chemsys, refresh, fetch)
    return doc["materials"], {k: doc[k] for k in ("chemsys", "retrieved_at", "db_version")}


def mp_like_entry(structure: Structure, energy: float, entry_id: str) -> ComputedStructureEntry:
    """MLIP total energy as an entry that MP2020 can correct like an MP calculation."""
    from pymatgen.io.vasp.sets import MPRelaxSet

    incar = MPRelaxSet(structure=structure).incar
    params: dict[str, Any] = {"run_type": "GGA", "is_hubbard": False, "hubbards": {}}
    if incar.get("LDAU"):
        symbols = MPRelaxSet(structure=structure).poscar.site_symbols
        params = {"run_type": "GGA+U", "is_hubbard": True, "hubbards": dict(zip(symbols, incar["LDAUU"]))}
    return ComputedStructureEntry(structure, energy, parameters=params, entry_id=entry_id)


def correct(entries: list[ComputedStructureEntry]) -> list[ComputedStructureEntry]:
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    compat = MaterialsProject2020Compatibility(check_potcar=False)
    return compat.process_entries(entries, clean=True, on_error="warn")


def e_above_hull(
    items: list[tuple[str, Structure, float]], *, refresh_mp: bool = False
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """items: (sid, relaxed structure, MLIP total energy). Returns per-sid results and MP metadata."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    by_chemsys: dict[str, list[tuple[str, Structure, float]]] = {}
    for sid, struct, energy in items:
        by_chemsys.setdefault(chemsys_key(el.symbol for el in struct.composition.elements), []).append((sid, struct, energy))

    out: dict[str, dict[str, Any]] = {}
    mp_meta: dict[str, Any] = {}
    for chemsys, group in sorted(by_chemsys.items()):
        ref_entries, mp_meta[chemsys] = mp_entries(chemsys, refresh=refresh_mp)
        pd = PhaseDiagram(ref_entries)
        raw = [mp_like_entry(s, e, sid) for sid, s, e in group]
        corrected = {e.entry_id: e for e in correct(raw)}
        for sid, struct, energy in group:
            entry = corrected.get(sid)
            if entry is None:
                out[sid] = {"e_above_hull": None, "error": "MP2020 compatibility rejected the entry"}
                continue
            decomp, ehull = pd.get_decomp_and_e_above_hull(entry, allow_negative=True, on_error="ignore")
            out[sid] = {
                "e_above_hull": None if ehull is None else round(float(ehull), 6),
                "decomposition": None if decomp is None else {e.composition.reduced_formula: round(float(f), 4) for e, f in decomp.items()},
                "correction_per_atom": round(entry.correction_per_atom, 6),
                "run_type": entry.parameters["run_type"],
                "energy_per_atom_uncorrected": round(energy / len(struct), 6),
                "chemsys": chemsys,
            }
    return out, mp_meta
