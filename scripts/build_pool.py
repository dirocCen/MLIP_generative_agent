"""Candidate pool for the closed-loop search: prototype decoration + ordered enumeration.

Three sources, all deterministic (no random structures):

1. ``enum``    ordered decorations of zincblende / wurtzite / rocksalt supercells (<= MAX_ATOMS atoms),
               charge-balanced with fixed oxidation states (the 8-electron rule for tetrahedral networks);
2. ``bcn``     B/C/N on every site of the zincblende / wurtzite supercells with n(B) = n(N)
               (diamond-like networks, where B and N do not have to stay on separate sublattices);
3. ``subst``   same-group element substitution in library prototypes: the project's L0-stable structures
               and cached MP structures with <= MAX_ATOMS primitive atoms. The MP structures themselves
               are added as ``known`` anchors.

Decorations are reduced with the space group of the undecorated supercell before any structure is
built; what is left is deduplicated per reduced formula with StructureMatcher.

Usage:  python scripts/build_pool.py [--out data/structures/pool/loop_pool_v1.extxyz]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spglib
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
MAX_ATOMS = 8
MAX_SPECIES_PER_SUBLATTICE = 2
MAX_ELEMENTS = 4
MIN_DIST = 1.2  # Angstrom, same guard as the PyXtal batch

CATIONS = {"Be": 2, "B": 3, "Al": 3, "Ga": 3, "Si": 4, "C": 4}
ANIONS = {"N": 3, "P": 3, "As": 3, "O": 2, "C": 4}
GROUPS = [("Be",), ("B", "Al", "Ga"), ("C", "Si"), ("N", "P", "As"), ("O",)]
GROUP_OF = {el: g for g in GROUPS for el in g}
PROTOTYPE_MAX_EHULL = 0.3  # eV/atom, MP structures used as prototypes / anchors
# S1-main (without B-Sb, where DPA fails) and S2 chemical systems; candidates stay inside one of them
SYSTEMS = [
    "Al-B-N", "Al-C-N-Si", "As-B-P", "As-B", "B-C-N", "B-C-P", "B-N", "C-N-Si", "C-N",
    "Be-O", "Be-C", "Be-C-N", "Be-N-Si", "B-Be-N", "Al-Be-N", "Ga-N",
]
_SYSTEM_SETS = [frozenset(s.split("-")) for s in SYSTEMS]


def in_scope(elements) -> bool:
    els = frozenset(elements)
    return any(els <= s for s in _SYSTEM_SETS)


# --- parent lattices ----------------------------------------------------------------
def parents() -> dict[str, Structure]:
    a = 1.0
    fcc = Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    zb = Structure(fcc, ["X", "Y"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    rs = Structure(fcc, ["X", "Y"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    c = np.sqrt(8 / 3)
    wz = Structure(
        Lattice.hexagonal(1.0, c), ["X", "X", "Y", "Y"],
        [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.375], [2 / 3, 1 / 3, 0.875]],
    )
    return {"zincblende": zb, "wurtzite": wz, "rocksalt": rs}


def hnfs(n: int):
    for a in range(1, n + 1):
        if n % a:
            continue
        for b in range(1, n // a + 1):
            if (n // a) % b:
                continue
            c = n // a // b
            for d in range(b):
                for e in range(c):
                    for f in range(c):
                        yield [[a, 0, 0], [d, b, 0], [e, f, c]]


def site_permutations(struct: Structure) -> np.ndarray:
    """Permutations of the sites under the space group of the (undecorated) cell."""
    types = [0 if s.specie.symbol == "X" else 1 for s in struct]
    cell = (struct.lattice.matrix, struct.frac_coords, types)
    sym = spglib.get_symmetry(cell, symprec=1e-4)
    frac = struct.frac_coords
    perms = set()
    for rot, trans in zip(sym["rotations"], sym["translations"]):
        new = frac @ rot.T + trans
        diff = new[:, None, :] - frac[None, :, :]
        diff -= np.round(diff)
        dist = np.linalg.norm(diff @ struct.lattice.matrix, axis=2)
        perm = dist.argmin(axis=1)
        if dist[np.arange(len(frac)), perm].max() < 1e-3 and len(set(perm)) == len(frac):
            perms.add(tuple(perm))
    return np.array(sorted(perms))


def canonical(labels: tuple[str, ...], perms: np.ndarray) -> tuple[str, ...]:
    arr = np.array(labels)
    return min(tuple(arr[p]) for p in perms)


# --- compositions -------------------------------------------------------------------
def sublattice_fillings(n: int, palette: dict[str, int]):
    """Multisets of <= MAX_SPECIES_PER_SUBLATTICE species on n sites -> (counts, total charge)."""
    els = sorted(palette)
    for k in range(1, MAX_SPECIES_PER_SUBLATTICE + 1):
        for combo in itertools.combinations(els, k):
            for split in itertools.product(range(1, n + 1), repeat=k):
                if sum(split) == n:
                    yield dict(zip(combo, split)), sum(palette[e] * m for e, m in zip(combo, split))


def neutral_fillings(n: int):
    cats = list(sublattice_fillings(n, CATIONS))
    ans = list(sublattice_fillings(n, ANIONS))
    for cat, qc in cats:
        for an, qa in ans:
            if qc == qa and in_scope(set(cat) | set(an)):
                yield cat, an


def arrangements(counts: dict[str, int]):
    labels = [el for el, m in sorted(counts.items()) for _ in range(m)]
    return sorted(set(itertools.permutations(labels)))


# --- geometry -----------------------------------------------------------------------

_RCOV = {"Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "Al": 1.21, "Si": 1.11, "P": 1.07,
         "Ga": 1.22, "As": 1.19}


def _rcov(el: str) -> float:
    return _RCOV[el]


def rescale_tetrahedral(struct: Structure, rocksalt: bool = False) -> Structure:
    """Volume from the mean covalent bond length (zincblende: V/atom = 1.5396 d^3)."""
    nn = struct.get_all_neighbors(_nn_cut(struct))
    ds = [_rcov(s.specie.symbol) + _rcov(nb.specie.symbol) for s, nbs in zip(struct, nn) for nb in nbs]
    d = float(np.mean(ds))
    v_atom = (2 * d) ** 3 / 8 if rocksalt else 1.5396 * d**3
    out = struct.copy()
    out.scale_lattice(v_atom * len(struct))
    return out


def _nn_cut(struct: Structure) -> float:
    dm = struct.distance_matrix
    return float(dm[dm > 1e-6].min()) * 1.05


# --- sources ------------------------------------------------------------------------
def enumerate_decorations():
    out = []
    plan = {"zincblende": (1, 2, 3, 4), "wurtzite": (1, 2), "rocksalt": (1, 2)}
    for pname, parent in parents().items():
        for idx in plan[pname]:
            if len(parent) * idx > MAX_ATOMS:
                continue
            for h in hnfs(idx):
                cell = parent.copy()
                cell.make_supercell(h)
                perms = site_permutations(cell)
                xs = [i for i, s in enumerate(cell) if s.specie.symbol == "X"]
                ys = [i for i, s in enumerate(cell) if s.specie.symbol == "Y"]
                seen = set()
                for cat, an in neutral_fillings(len(xs)):
                    for ca in arrangements(cat):
                        for aa in arrangements(an):
                            labels = [""] * len(cell)
                            for i, el in zip(xs, ca):
                                labels[i] = el
                            for i, el in zip(ys, aa):
                                labels[i] = el
                            key = canonical(tuple(labels), perms)
                            if key in seen:
                                continue
                            seen.add(key)
                            s = Structure(cell.lattice, list(key), cell.frac_coords)
                            s = rescale_tetrahedral(s, rocksalt=pname == "rocksalt")
                            out.append((s, {"source": "enum", "parent": pname, "hnf": json.dumps(h)}))
                # B/C/N on every site, n(B) = n(N)
                if pname == "rocksalt":
                    continue
                n = len(cell)
                seen = set()
                for nb in range(0, n // 2 + 1):
                    nc = n - 2 * nb
                    if nb == 0 and idx > 1:
                        continue  # pure C: only the primitive parents
                    base = ["B"] * nb + ["N"] * nb + ["C"] * nc
                    for labels in set(itertools.permutations(base)):
                        key = canonical(labels, perms)
                        if key in seen:
                            continue
                        seen.add(key)
                        s = Structure(cell.lattice, list(key), cell.frac_coords)
                        s = rescale_tetrahedral(s)
                        out.append((s, {"source": "bcn", "parent": pname, "hnf": json.dumps(h)}))
    return out


def library_prototypes():
    """(structure, tag) for the project's L0-stable structures and cached MP structures."""
    protos = []
    sets = sorted((ROOT / "data/sets").glob("*l0stable*dpa41_max_oam*.json"))
    sids = []
    for f in sets:
        for sid in json.loads(f.read_text())["sids"]:
            if sid not in sids:
                sids.append(sid)
    for sid in sids:
        entry = json.loads((ROOT / "data/structures" / f"{sid}.json").read_text())
        s = Structure.from_dict(entry["structure"])
        if len(s.composition.elements) >= 2:
            protos.append((s, f"l0stable:{sid}", None))
    for f in sorted((ROOT / "data/reference/mp/entries").glob("*.json")):
        entries = [ComputedStructureEntry.from_dict(e) for e in json.loads(f.read_text())["entries"]]
        if len(f.stem.split("-")) < 2:
            continue
        pd = PhaseDiagram(entries)
        for e in entries:
            if len(e.composition.elements) != len(f.stem.split("-")):
                continue  # subsystems come from their own file
            ehull = pd.get_e_above_hull(e)
            if ehull <= PROTOTYPE_MAX_EHULL:
                protos.append((e.structure, f"mp:{e.entry_id}", round(float(ehull), 4)))
    return protos


def primitive(s: Structure) -> Structure | None:
    try:
        return SpacegroupAnalyzer(s, symprec=0.1).get_primitive_standard_structure()
    except Exception:  # noqa: BLE001
        return None


def substitutions(protos):
    out = []
    for s, tag, ehull in protos:
        p = primitive(s)
        if p is None or len(p) > MAX_ATOMS:
            continue
        els = sorted(el.symbol for el in p.composition.elements)
        if any(el not in GROUP_OF for el in els):
            continue
        if tag.startswith("mp:"):
            out.append((p, {"source": "known", "prototype": tag, "mp_ehull": ehull}))
        v0 = sum(_rcov(sp.symbol) ** 3 for sp in p.species)
        for target in itertools.product(*(GROUP_OF[el] for el in els)):
            if list(target) == els or len(set(target)) < len(target) or not in_scope(target):
                continue
            mapping = dict(zip(els, target))
            new = p.copy()
            new.replace_species(mapping)
            v1 = sum(_rcov(sp.symbol) ** 3 for sp in new.species)
            new.scale_lattice(p.volume * v1 / v0)
            out.append((new, {"source": "subst", "prototype": tag, "map": json.dumps(mapping)}))
    return out


# --- assemble -----------------------------------------------------------------------
def min_dist(s: Structure) -> float:
    dm = s.distance_matrix
    if len(s) == 1:
        return min(s.lattice.abc)
    return float(min(dm[dm > 1e-6].min(), min(s.lattice.abc)))


def fingerprint(s: Structure):
    """Space group + Wyckoff letters per element: only structures that agree here go to the matcher."""
    try:
        ds = SpacegroupAnalyzer(s, symprec=0.01).get_symmetry_dataset()
        wy = tuple(sorted(f"{sp.symbol}{w}" for sp, w in zip(s.species, ds.wyckoffs)))
        return ds.number, wy
    except Exception:  # noqa: BLE001
        return None


def neighbour_signature(s: Structure, cutoff: float = 2.4):
    """Sorted per-site neighbour lists (species, distance) in units of (V/atom)^(1/3).

    Exact for decorations of an ideal parent lattice, where the same structure reached through
    different supercells has identical distances; much cheaper than StructureMatcher.
    """
    unit = (s.volume / len(s)) ** (1 / 3)
    sites = []
    for site, nbs in zip(s, s.get_all_neighbors(cutoff * unit)):
        sites.append((site.specie.symbol, tuple(sorted((nb.specie.symbol, round(nb.nn_distance / unit, 2)) for nb in nbs))))
    return hash(tuple(sorted(sites)))


def dedupe(items):
    seen, unique = set(), []
    for s, info in items:
        if info["source"] in ("enum", "bcn"):
            key = (s.composition.reduced_formula, len(s), neighbour_signature(s))
            if key in seen:
                continue
            seen.add(key)
        unique.append((s, info))
    print(f"after neighbour-signature dedupe of ideal decorations: {len(unique)}", file=sys.stderr, flush=True)
    items = unique
    order = {"known": 0, "subst": 1, "enum": 2, "bcn": 3}
    items = sorted(items, key=lambda it: (order[it[1]["source"]], len(it[0])))
    groups = defaultdict(list)
    for s, info in items:
        groups[(s.composition.reduced_formula, len(s), fingerprint(s))].append((s, info))
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
    kept = []
    for key in sorted(groups, key=str):
        reps = []
        for s, info in groups[key]:
            if not any(matcher.fit(s, r) for r, _ in reps):
                reps.append((s, info))
        kept.extend(reps)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/structures/pool/loop_pool_v1.extxyz")
    args = ap.parse_args()

    raw = enumerate_decorations()
    print(f"enum + bcn decorations (symmetry-reduced per supercell): {len(raw)}", file=sys.stderr)
    protos = library_prototypes()
    sub = substitutions(protos)
    print(f"library prototypes: {len(protos)}; known + substituted: {len(sub)}", file=sys.stderr)

    items = []
    for s, info in raw + sub:
        p = primitive(s)
        if p is None or len(p) > MAX_ATOMS or min_dist(p) < MIN_DIST:
            continue
        if len(p.composition.elements) > MAX_ELEMENTS:
            continue
        items.append((p, info))
    print(f"after primitive-cell / d_min filter: {len(items)}", file=sys.stderr)
    kept = dedupe(items)
    print(f"after StructureMatcher dedupe: {len(kept)}", file=sys.stderr)

    from ase.io import write

    frames = []
    for s, info in kept:
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.info.update({k: v for k, v in info.items() if v is not None})
        frames.append(atoms)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, frames, format="extxyz")

    by_source = Counter(info["source"] for _, info in kept)
    by_n = Counter(len(s) for s, _ in kept)
    by_sys = Counter(s.composition.chemical_system for s, _ in kept)
    print(json.dumps({"out": args.out, "n": len(kept), "by_source": by_source,
                      "by_n_atoms": dict(sorted(by_n.items())), "n_chemsys": len(by_sys),
                      "top_chemsys": dict(by_sys.most_common(25))}, indent=1))


if __name__ == "__main__":
    main()
