"""Pool expansion after an L1/L2 round: children of the best and most promising structures.

Parents (``select_parents``): the top structures by label (L2 LBTE where valid, else L1), the
top unlabelled structures by UCB of the surrogate, at most ``max_per_formula`` per reduced formula
(polymorphs of one formula would otherwise fill the list).

Operators (``children``), all on the parent's MLIP-relaxed primitive cell, one operator per child
(the next round's expansion composes them):

``subst``  isovalent substitution within a valence group (Li; Be Mg; B Al Ga; C Si Ge; N P As Sb Bi;
           O S), volume rescaled by the covalent radii;
``swap``   valence-balanced cross-group swap: two species of equal multiplicity move k groups in
           opposite directions (B+As -> C+Ge, Li+Sb -> Be+Ge, Be+O -> B+N), so the valence-electron
           count, and with it the octet rule of the tetrahedral networks, is kept;
``perm``   species permutation over the parent's sublattices among species of equal multiplicity:
           the same formula in another ordering (the I-site ordering decided stability of the I-II-V
           Nowotny-Juza compounds, 11-19).

Scope (user decision 11-19): light elements with strong bonds plus heavy mass-contrast partners;
every child contains at least one of Be, B, C, N.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# valence group (number of valence s+p electrons) of every element a child may contain
GROUP = {
    "Li": 1,
    "Be": 2, "Mg": 2,
    "B": 3, "Al": 3, "Ga": 3,
    "C": 4, "Si": 4, "Ge": 4,
    "N": 5, "P": 5, "As": 5, "Sb": 5, "Bi": 5,
    "O": 6, "S": 6,
}
ELEMENTS = frozenset(GROUP)
STRONG_BOND = frozenset({"Be", "B", "C", "N"})
# Cordero et al., Dalton Trans. 2008 (sp3 C); the same values as scripts/build_pool.py where shared
RCOV = {"Li": 1.28, "Be": 0.96, "Mg": 1.41, "B": 0.84, "Al": 1.21, "Ga": 1.22, "C": 0.76, "Si": 1.11,
        "Ge": 1.20, "N": 0.71, "P": 1.07, "As": 1.19, "Sb": 1.39, "Bi": 1.48, "O": 0.66, "S": 1.05}
# Pruning after the e1 dry run (11-21: 989 novel children, ~32 GPU-h of L0): children of <= 6
# primitive atoms, and |k| = 1 swaps only (k = 2 turned I-II-V parents into intermetallics such as
# BeAl2 / BeB2, formally 8-electron but not covalent networks).
MAX_ATOMS, MAX_ELEMENTS, MIN_DIST = 6, 4, 1.2
SWAP_STEPS = (1,)  # |k| of the cross-group swap


def in_scope(elements: Iterable[str]) -> bool:
    els = set(elements)
    return els <= ELEMENTS and bool(els & STRONG_BOND) and len(els) <= MAX_ELEMENTS


def primitive(s: Structure) -> Structure | None:
    try:
        return SpacegroupAnalyzer(s, symprec=0.1).get_primitive_standard_structure()
    except Exception:  # noqa: BLE001
        return None


def min_dist(s: Structure) -> float:
    if len(s) == 1:
        return float(min(s.lattice.abc))
    dm = s.distance_matrix
    return float(min(dm[dm > 1e-6].min(), min(s.lattice.abc)))


def _apply(p: Structure, mapping: dict[str, str], rescale: bool) -> Structure:
    new = p.copy()
    new.replace_species(mapping)
    if rescale:
        v0 = sum(RCOV.get(sp.symbol, 1.2) ** 3 for sp in p.species)
        v1 = sum(RCOV[sp.symbol] ** 3 for sp in new.species)
        new.scale_lattice(p.volume * v1 / v0)
    return new


def _by_group(g: int) -> list[str]:
    return sorted(el for el, gg in GROUP.items() if gg == g)


def substitutions(p: Structure):
    els = sorted(el.symbol for el in p.composition.elements)
    if any(el not in GROUP for el in els):
        return
    for target in itertools.product(*(_by_group(GROUP[el]) for el in els)):
        if list(target) == els or len(set(target)) < len(target):
            continue
        yield "subst", dict(zip(els, target))


def swaps(p: Structure):
    comp = p.composition
    els = sorted(el.symbol for el in comp.elements)
    for x, z in itertools.combinations(els, 2):
        if x not in GROUP or z not in GROUP or comp[x] != comp[z]:
            continue
        for k in SWAP_STEPS:
            for sign in (1, -1):
                gx, gz = GROUP[x] + sign * k, GROUP[z] - sign * k
                for y, w in itertools.product(_by_group(gx), _by_group(gz)):
                    yield "swap", {x: y, z: w}


def permutations(p: Structure):
    comp = p.composition
    els = sorted(el.symbol for el in comp.elements)
    for m in sorted({comp[e] for e in els}):
        same = [e for e in els if comp[e] == m]
        for perm in itertools.permutations(same):
            if list(perm) != same:
                yield "perm", dict(zip(same, perm))


OPERATORS = {"subst": substitutions, "swap": swaps, "perm": permutations}


def children(parent: Structure, operators: Iterable[str] = tuple(OPERATORS)):
    """(child primitive structure, info) for every in-scope child that passes the size / distance guards."""
    p = primitive(parent)
    if p is None:
        return
    for op in operators:
        for source, mapping in OPERATORS[op](p):
            new = _apply(p, mapping, rescale=source != "perm")
            if not in_scope(el.symbol for el in new.composition.elements):
                continue
            c = primitive(new)
            if c is None or len(c) > MAX_ATOMS or min_dist(c) < MIN_DIST:
                continue
            yield c, {"source": source, "map": json.dumps(mapping, sort_keys=True)}


def select_parents(
    table: pd.DataFrame,
    n_label: int = 12,
    n_ucb: int = 8,
    beta: float = 1.0,
    max_per_formula: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Top ``n_label`` labelled + top ``n_ucb`` unlabelled (by UCB) L0-stable rows of a
    ``build_table`` table, at most ``max_per_formula`` per formula. Adds ``parent_reason``,
    ``mu_kappa`` and ``ucb``."""
    from mga.loop.data import FEATURES
    from mga.loop.surrogate import Surrogate

    t = table[table["l0_ok"]].dropna(subset=list(FEATURES)).copy()
    train = t[t["kappa"].notna()]
    X = t[list(FEATURES)].to_numpy(float)
    sur = Surrogate(seed).fit(train[list(FEATURES)].to_numpy(float), train["log_kappa"].to_numpy(float))
    mu, sd = sur.predict(X)
    t["mu_kappa"], t["ucb"] = 10 ** mu, mu + beta * sd

    picked: list[tuple[str, str]] = []
    per_formula: dict[str, int] = {}

    def take(rows: pd.DataFrame, n: int, reason: str) -> None:
        k = 0
        for sid, f in zip(rows.index, rows["formula"]):
            if k == n:
                return
            if per_formula.get(f, 0) >= max_per_formula or sid in dict(picked):
                continue
            picked.append((sid, reason))
            per_formula[f] = per_formula.get(f, 0) + 1
            k += 1

    take(train.assign(_k=train["kappa"]).sort_values("_k", ascending=False), n_label, "label")
    unl = t[t["kappa"].isna() & ~t["l2_invalid"]]
    take(unl.sort_values("ucb", ascending=False), n_ucb, "ucb")
    out = t.loc[[s for s, _ in picked]].copy()
    out["parent_reason"] = [r for _, r in picked]
    return out


def mp_novelty(structs: list[Structure], matcher, mp_structures) -> list[dict]:
    """Per structure: MP ids of the same reduced formula that match, and whether MP has the formula at all.

    ``mp_structures(chemsys) -> (materials, meta)`` (``mga.analysis.ehull.mp_structures``)."""
    from pymatgen.core import Composition

    from mga.analysis.ehull import chemsys_key

    cache: dict[str, list] = {}
    out = []
    for s in structs:
        cs = chemsys_key([el.symbol for el in s.composition.elements])
        if cs not in cache:
            cache[cs] = [(Composition(m["formula"]).reduced_formula, m) for m in mp_structures(cs)[0]]
        f = s.composition.reduced_formula
        same = [m for rf, m in cache[cs] if rf == f]
        matches = [m["material_id"] for m in same if matcher.fit(s, Structure.from_dict(m["structure"]))]
        out.append({"mp_matches": matches, "formula_in_mp": bool(same),
                    "mp_min_ehull_formula": min((m["energy_above_hull"] for m in same), default=np.nan)})
    return out
