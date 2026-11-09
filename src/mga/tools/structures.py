"""Structure tools: import, summarise, filter, deduplicate, novelty, benchmark leakage.

All tools take and produce *named sets* (plan §3.4); per-structure tables are
written to ``runs/<run>/outputs/<step_id>.json`` and only summaries are returned.
"""

from __future__ import annotations

import csv
import hashlib
import io
import statistics
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pymatgen.core import Composition, Structure

from mga import registry
from mga.config import paths
from mga.params import resolve_params
from mga.records import recorded_step
from mga.specs import StructureMatcherParams
from mga.tools._common import (
    N_EXAMPLES,
    ToolInputError,
    parent_steps,
    project_path,
    rel,
    require_decision,
    top,
    write_details,
)

OriginKind = Literal["file", "mattergen", "kappa_srme_reference"]
MAX_IMPORT = 20000


# --- reading ----------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _from_ase(frames) -> list[tuple[Structure, dict[str, Any]]]:
    from pymatgen.io.ase import AseAtomsAdaptor

    return [(AseAtomsAdaptor.get_structure(a), _jsonable(dict(a.info))) for a in frames]


def _read_extxyz_text(text: str):
    from ase.io import read

    return read(io.StringIO(text), index=":", format="extxyz")


def _read_cif_text(text: str) -> list[Structure]:
    from pymatgen.io.cif import CifParser

    return CifParser.from_str(text).parse_structures(primitive=False)


def read_structures(path: Path) -> list[tuple[Structure, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in (".extxyz", ".xyz"):
        return _from_ase(_read_extxyz_text(path.read_text()))
    if suffix == ".cif":
        return [(s, {"cif_block": i}) for i, s in enumerate(_read_cif_text(path.read_text()))]
    if suffix == ".json":
        from mga.utils import read_json

        data = read_json(path)
        items = data["structures"] if isinstance(data, dict) else data
        return [(Structure.from_dict(d), {}) for d in items]
    if suffix == ".zip":  # MatterGen: generated_crystals_cif.zip
        out = []
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                text = zf.read(name).decode()
                if name.lower().endswith(".cif"):
                    out += [(s, {"zip_member": name}) for s in _read_cif_text(text)]
                elif name.lower().endswith((".extxyz", ".xyz")):
                    out += [(s, {**info, "zip_member": name}) for s, info in _from_ase(_read_extxyz_text(text))]
        return out
    raise ToolInputError(f"Unsupported file type {suffix!r}; use .extxyz, .xyz, .cif, .json or .zip")


def _reference_info(info: dict[str, Any]) -> dict[str, Any]:
    """phononDB-PBE extxyz metadata -> registry info with the fixed kappa settings."""
    missing = [k for k in ("mp_id", "fc2_supercell", "fc3_supercell", "q_mesh") if k not in info]
    if missing:
        raise ToolInputError(f"kappa_srme_reference frame lacks {missing}; is this phononDB-PBE-structures.extxyz?")
    return {
        "mp_id": info["mp_id"],
        "name": info.get("name"),
        "benchmark": "kappa_srme",
        "expected_spacegroup_number": info.get("symm.no"),
        "kappa_settings": {
            "source": "phonondb_pbe",
            "fc2_supercell": info["fc2_supercell"],
            "fc3_supercell": info["fc3_supercell"],
            "q_mesh": info["q_mesh"],
            "primitive_matrix": info.get("primitive_matrix"),
        },
    }


# --- tools --------------------------------------------------------------------------
def import_structures(
    path: str,
    set_name: str,
    origin_kind: OriginKind,
    decided_by: Literal["agent", "human"],
    rationale: str,
    description: str = "",
) -> dict[str, Any]:
    require_decision(decided_by, rationale)
    file = project_path(path)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    with recorded_step(
        "import_structures",
        inputs={"path": rel(file), "sha256": digest, "set_name": set_name, "origin_kind": origin_kind},
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        frames = read_structures(file)
        if len(frames) > MAX_IMPORT:
            raise ToolInputError(f"{len(frames)} structures in one file; the limit is {MAX_IMPORT}")
        sids, created, failures, without_settings = [], 0, [], []
        for i, (struct, info) in enumerate(frames):
            try:
                entry_info = _reference_info(info) if origin_kind == "kappa_srme_reference" else {"file_info": info}
                origin = {"kind": origin_kind, "file": rel(file), "file_sha256": digest, "index": i}
                sid, new = registry.register_structure(struct, origin=origin, info=entry_info, step_id=rec.step_id)
            except Exception as exc:  # noqa: BLE001 - report bad frames, keep the rest
                failures.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
                continue
            created += new
            sids.append(sid)
            if origin_kind == "kappa_srme_reference" and not new:
                if "kappa_settings" not in registry.load_entry(sid)["info"]:
                    without_settings.append(sid)
        registry.create_set(
            set_name, sids, operation="import", description=description,
            details={"file": rel(file), "sha256": digest, "origin_kind": origin_kind}, step_id=rec.step_id,
        )
        rec.outputs = {
            "set_name": set_name,
            "n_read": len(frames),
            "n_in_set": len(set(sids)),
            "n_newly_registered": created,
            "n_failed": len(failures),
            "failures": failures[:10],
            "examples": sids[:N_EXAMPLES],
        }
        if without_settings:
            rec.outputs["warning"] = (
                f"{len(without_settings)} structures were registered earlier without phononDB settings "
                "(registry entries are immutable); kappa jobs for them will use the length rule"
            )
    return {"step_id": rec.step_id, **rec.outputs}


def structure_summary(set_name: str, n_examples: int = N_EXAMPLES) -> dict[str, Any]:
    entries = list(registry.iter_entries(registry.set_sids(set_name)))
    nsites = [e["nsites"] for e in entries]
    meta = registry.load_set(set_name)
    return {
        "set": {k: meta[k] for k in ("name", "n", "operation", "parent_sets", "description", "created_at")},
        "reduced_formulas": top(e["reduced_formula"] for e in entries),
        "chemical_systems": top(e["chemsys"] for e in entries),
        "n_elements": top(len(e["elements"]) for e in entries),
        "nsites": {"min": min(nsites), "median": statistics.median(nsites), "max": max(nsites)} if nsites else {},
        "spacegroups": top(e["spacegroup_number"] for e in entries),
        "origins": top(e["origin"].get("kind") for e in entries),
        "examples": [
            {k: e[k] for k in ("sid", "reduced_formula", "nsites", "spacegroup_number", "volume_per_atom")}
            for e in entries[:n_examples]
        ],
    }


def list_sets(limit: int = 50) -> dict[str, Any]:
    sets = list(registry.iter_sets())
    return {"n_sets": len(sets), "sets": sets[-limit:]}


def _min_distance(struct: Structure, cutoff: float = 4.0) -> float:
    _, _, _, dists = struct.get_neighbor_list(r=cutoff)
    dists = dists[dists > 1e-8]
    return float(dists.min()) if len(dists) else float("inf")


PRIMITIVE_SYMPREC = 0.1  # generated cells are not exactly symmetric; 0.1 A is the usual tolerance for them


def _primitive_sites(struct: Structure) -> int:
    """Atoms in the standardized primitive cell (DESIGN §4.1: <= 8); the input size if spglib fails."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        return len(SpacegroupAnalyzer(struct, symprec=PRIMITIVE_SYMPREC).get_primitive_standard_structure())
    except Exception:  # noqa: BLE001 - spglib can fail on odd cells
        return len(struct)


def merge_sets(
    set_names: list[str],
    new_set_name: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
) -> dict[str, Any]:
    """Union of sets in the given order (repeats dropped), e.g. one generated set per chemical system."""
    require_decision(decided_by, rationale)
    if len(set_names) < 2:
        raise ToolInputError("Give at least two sets to merge")
    with recorded_step(
        "merge_sets",
        inputs={"set_names": set_names, "new_set_name": new_set_name},
        parent_ids=parent_steps(*set_names),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        sizes = {name: len(registry.set_sids(name)) for name in set_names}
        sids = [sid for name in set_names for sid in registry.set_sids(name)]
        doc = registry.create_set(new_set_name, sids, operation="merge", parent_sets=set_names, step_id=rec.step_id)
        rec.outputs = {"new_set_name": new_set_name, "n_in": sizes, "n_merged": len(doc["sids"])}
    return {"step_id": rec.step_id, **rec.outputs}


def symmetrize_structures(
    set_name: str,
    new_set_name: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    symprec: float = PRIMITIVE_SYMPREC,
) -> dict[str, Any]:
    """Idealized standard primitive cell at a loose tolerance (S3, between pre-relax and kappa_SRME relax).

    Generated and pre-relaxed cells are nearly symmetric; at the kappa_SRME tolerances they stay P1,
    FixSymmetry then keeps them P1 and FC3 needs ~|G|^2 more displacements. Each output is registered
    with its input as parent.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    require_decision(decided_by, rationale)
    if not 1e-4 <= symprec <= 0.3:
        raise ToolInputError("symprec must be within 1e-4-0.3 Angstrom")
    with recorded_step(
        "symmetrize_structures",
        inputs={"set_name": set_name, "new_set_name": new_set_name, "symprec": symprec},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        out, changes, failed = [], {}, {}
        for sid in registry.set_sids(set_name):
            struct = registry.load_structure(sid)
            try:
                sga = SpacegroupAnalyzer(struct, symprec=symprec)
                sym = sga.get_primitive_standard_structure()
            except Exception as exc:  # noqa: BLE001 - spglib can fail on odd cells
                failed[sid] = str(exc)
                out.append(sid)
                continue
            child, _ = registry.register_structure(
                sym, origin={"kind": "symmetrize", "source_sid": sid, "symprec": symprec},
                parents=[] if registry.structure_id(sym) == sid else [sid], step_id=rec.step_id,
            )
            out.append(child)
            changes[sid] = {"sid": child, "spacegroup": sga.get_space_group_number(), "nsites": [len(struct), len(sym)]}
        registry.create_set(new_set_name, out, operation="symmetrize", parent_sets=[set_name],
                            details={"symprec": symprec, "changes": changes, "failed": failed}, step_id=rec.step_id)
        rec.outputs = {
            "new_set_name": new_set_name,
            "n": len(out),
            "n_failed": len(failed),
            "spacegroups": top(str(c["spacegroup"]) for c in changes.values()),
            "n_p1": sum(c["spacegroup"] == 1 for c in changes.values()),
        }
    return {"step_id": rec.step_id, **rec.outputs}


def filter_structures(
    set_name: str,
    new_set_name: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    min_sites: int | None = None,
    max_sites: int | None = None,
    max_primitive_sites: int | None = None,
    min_elements: int | None = None,
    max_elements: int | None = None,
    elements_allowed: list[str] | None = None,
    elements_required: list[str] | None = None,
    elements_excluded: list[str] | None = None,
    spacegroups_allowed: list[int] | None = None,
    spacegroups_excluded: list[int] | None = None,
    min_distance_angstrom: float | None = None,
    min_volume_per_atom: float | None = None,
    max_volume_per_atom: float | None = None,
) -> dict[str, Any]:
    require_decision(decided_by, rationale)
    criteria = {k: v for k, v in locals().items() if k not in ("set_name", "new_set_name", "decided_by", "rationale") and v is not None}
    if not criteria:
        raise ToolInputError("Give at least one filter criterion")
    with recorded_step(
        "filter_structures",
        inputs={"set_name": set_name, "new_set_name": new_set_name, "criteria": criteria},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        allowed = set(elements_allowed or [])
        kept, removed, why = [], {}, {}
        for entry in registry.iter_entries(registry.set_sids(set_name)):
            els = set(entry["elements"])
            checks = {
                "min_sites": min_sites is None or entry["nsites"] >= min_sites,
                "max_sites": max_sites is None or entry["nsites"] <= max_sites,
                "min_elements": min_elements is None or len(els) >= min_elements,
                "max_elements": max_elements is None or len(els) <= max_elements,
                "elements_allowed": not allowed or els <= allowed,
                "elements_required": not elements_required or set(elements_required) <= els,
                "elements_excluded": not elements_excluded or not (els & set(elements_excluded)),
                "spacegroups_allowed": not spacegroups_allowed or entry["spacegroup_number"] in spacegroups_allowed,
                "spacegroups_excluded": not spacegroups_excluded or entry["spacegroup_number"] not in spacegroups_excluded,
                "min_volume_per_atom": min_volume_per_atom is None or entry["volume_per_atom"] >= min_volume_per_atom,
                "max_volume_per_atom": max_volume_per_atom is None or entry["volume_per_atom"] <= max_volume_per_atom,
            }
            if max_primitive_sites is not None and all(checks.values()):
                checks["max_primitive_sites"] = _primitive_sites(Structure.from_dict(entry["structure"])) <= max_primitive_sites
            if min_distance_angstrom is not None and all(checks.values()):
                checks["min_distance_angstrom"] = _min_distance(Structure.from_dict(entry["structure"])) >= min_distance_angstrom
            failed = [k for k, ok in checks.items() if not ok]
            if failed:
                why[entry["sid"]] = failed
                for k in failed:
                    removed[k] = removed.get(k, 0) + 1
            else:
                kept.append(entry["sid"])
        registry.create_set(
            new_set_name, kept, operation="filter", parent_sets=[set_name],
            details={"criteria": criteria}, step_id=rec.step_id,
        )
        rec.outputs = {
            "new_set_name": new_set_name,
            "n_in": len(kept) + len(why),
            "n_kept": len(kept),
            "removed_by_criterion": removed,
            "details_file": write_details(rec.step_id, {"removed": why}),
        }
    return {"step_id": rec.step_id, **rec.outputs}


def _matcher(p: StructureMatcherParams):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(**p.model_dump())


def deduplicate(
    set_name: str,
    new_set_name: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    matcher_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    """Group by reduced formula, then StructureMatcher within each group; keep the first of each group."""
    require_decision(decided_by, rationale)
    p, prov = resolve_params(StructureMatcherParams, matcher_override, override_reason)
    with recorded_step(
        "deduplicate",
        inputs={"set_name": set_name, "new_set_name": new_set_name},
        params=prov,
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        sm = _matcher(p)
        sids = registry.set_sids(set_name)
        by_formula: dict[str, list[str]] = {}
        for entry in registry.iter_entries(sids):
            by_formula.setdefault(entry["reduced_formula"], []).append(entry["sid"])
        keep, duplicates = [], {}
        for group in by_formula.values():
            if len(group) == 1:
                keep.append(group[0])
                continue
            structs = [registry.load_structure(s) for s in group]
            for cluster in sm.group_structures(structs):
                ids = [group[structs.index(s)] for s in cluster]
                ids.sort(key=sids.index)  # first in set order is kept
                keep.append(ids[0])
                if len(ids) > 1:
                    duplicates[ids[0]] = ids[1:]
        keep.sort(key=sids.index)
        registry.create_set(
            new_set_name, keep, operation="deduplicate", parent_sets=[set_name],
            details={"matcher": p.model_dump(), "duplicates_of_kept": duplicates}, step_id=rec.step_id,
        )
        rec.outputs = {
            "new_set_name": new_set_name,
            "n_in": len(sids),
            "n_unique": len(keep),
            "n_removed": len(sids) - len(keep),
            "largest_groups": dict(sorted(((k, len(v) + 1) for k, v in duplicates.items()), key=lambda kv: -kv[1])[:5]),
        }
    return {"step_id": rec.step_id, **rec.outputs}


def check_novelty(
    set_name: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
    novel_set_name: str | None = None,
    refresh_mp: bool = False,
    matcher_override: dict[str, Any] | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    """Compare with non-deprecated MP materials of the same reduced formula (MP only, not Alexandria/OQMD)."""
    from mga.analysis.ehull import chemsys_key, mp_structures

    require_decision(decided_by, rationale)
    p, prov = resolve_params(StructureMatcherParams, matcher_override, override_reason)
    with recorded_step(
        "check_novelty",
        inputs={"set_name": set_name, "novel_set_name": novel_set_name, "refresh_mp": refresh_mp},
        params=prov,
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        sm = _matcher(p)
        sids = registry.set_sids(set_name)
        mp_cache: dict[str, list[dict[str, Any]]] = {}
        mp_meta: dict[str, Any] = {}
        details, novel = {}, []
        for entry in registry.iter_entries(sids):
            chemsys = chemsys_key(entry["elements"])
            if chemsys not in mp_cache:
                mp_cache[chemsys], mp_meta[chemsys] = mp_structures(chemsys, refresh=refresh_mp)
            struct = registry.load_structure(entry["sid"])
            matches = [
                m["material_id"]
                for m in mp_cache[chemsys]
                if Composition(m["formula"]).reduced_formula == entry["reduced_formula"]
                and sm.fit(struct, Structure.from_dict(m["structure"]))
            ]
            same_formula = sum(Composition(m["formula"]).reduced_formula == entry["reduced_formula"] for m in mp_cache[chemsys])
            details[entry["sid"]] = {"mp_matches": matches, "mp_same_formula": same_formula}
            if not matches:
                novel.append(entry["sid"])
        if novel_set_name:
            registry.create_set(
                novel_set_name, novel, operation="novelty_vs_mp", parent_sets=[set_name],
                details={"matcher": p.model_dump(), "mp": mp_meta}, step_id=rec.step_id,
            )
        rec.outputs = {
            "n_in": len(sids),
            "n_novel_vs_mp": len(novel),
            "n_known": len(sids) - len(novel),
            "n_new_formula": sum(d["mp_same_formula"] == 0 for d in details.values()),
            "novel_set_name": novel_set_name,
            "mp_db_versions": sorted({m["db_version"] for m in mp_meta.values()}),
            "details_file": write_details(rec.step_id, {"per_structure": details, "mp": mp_meta}),
            "note": "novelty is relative to MP only",
        }
    return {"step_id": rec.step_id, **rec.outputs}


# --- benchmark leakage (DESIGN §1.1, §11) --------------------------------------------
@lru_cache(maxsize=1)
def benchmark_formulas() -> frozenset[str]:
    """Reduced formulas of the 103 kappa_SRME compounds (manifest + reference extxyz)."""
    formulas = set()
    with open(paths.root / "docs" / "phonondb_pbe_103_manifest.csv") as fh:
        formulas |= {Composition(row["formula"]).reduced_formula for row in csv.DictReader(fh)}
    from ase.io import read

    for atoms in read(paths.reference / "kappa_srme" / "phononDB-PBE-structures.extxyz", index=":"):
        formulas.add(Composition(atoms.get_chemical_formula()).reduced_formula)
    return frozenset(formulas)


def leakage_reasons(sid: str) -> list[str]:
    bench = benchmark_formulas()
    reasons = []
    for s in [sid, *sorted(registry.ancestors(sid))]:
        entry = registry.load_entry(s)
        tag = "self" if s == sid else f"ancestor {s}"
        if entry["reduced_formula"] in bench:
            reasons.append(f"{tag}: formula {entry['reduced_formula']} is a kappa_SRME test formula")
        if entry["origin"].get("kind") == "kappa_srme_reference" or entry["info"].get("benchmark") == "kappa_srme":
            reasons.append(f"{tag}: imported from the kappa_SRME reference set")
    return reasons


def check_benchmark_leakage(
    set_name: str,
    clean_set_name: str,
    decided_by: Literal["agent", "human"],
    rationale: str,
) -> dict[str, Any]:
    """Conservative: excludes every polymorph of the test formulas and every descendant of them."""
    require_decision(decided_by, rationale)
    with recorded_step(
        "check_benchmark_leakage",
        inputs={"set_name": set_name, "clean_set_name": clean_set_name},
        parent_ids=parent_steps(set_name),
        decided_by=decided_by,
        rationale=rationale,
    ) as rec:
        sids = registry.set_sids(set_name)
        flagged = {sid: r for sid in sids if (r := leakage_reasons(sid))}
        clean = [s for s in sids if s not in flagged]
        registry.create_set(
            clean_set_name, clean, operation="benchmark_leakage_filter", parent_sets=[set_name],
            details={"n_benchmark_formulas": len(benchmark_formulas()), "flagged": flagged}, step_id=rec.step_id,
        )
        rec.outputs = {
            "clean_set_name": clean_set_name,
            "n_in": len(sids),
            "n_flagged": len(flagged),
            "n_clean": len(clean),
            "n_benchmark_formulas": len(benchmark_formulas()),
            "flagged_formulas": top(registry.load_entry(s)["reduced_formula"] for s in flagged),
        }
    return {"step_id": rec.step_id, **rec.outputs}


def assert_no_leakage(sids: list[str]) -> None:
    """Hard guard for anything that becomes training data (fine-tuning, Phase B)."""
    bad = [s for s in sids if leakage_reasons(s)]
    if bad:
        raise ToolInputError(f"{len(bad)} structures overlap the kappa_SRME test set, e.g. {bad[:3]}")


