"""Expansion operators on ideal prototypes (no result store, no MP)."""

import json

from pymatgen.core import Lattice, Structure

from mga.loop import expand


def _zb(a, b, x=4.0):
    return Structure(Lattice([[0, x / 2, x / 2], [x / 2, 0, x / 2], [x / 2, x / 2, 0]]), [a, b],
                     [[0, 0, 0], [0.25, 0.25, 0.25]])


def _heusler(i, ii, v, x=6.0):
    # F-43m I-II-V in the primitive fcc cell (ordering B: V 4a, II 4c, I 4d)
    return Structure(Lattice([[0, x / 2, x / 2], [x / 2, 0, x / 2], [x / 2, x / 2, 0]]), [v, ii, i],
                     [[0, 0, 0], [0.25] * 3, [0.75] * 3])


def _kids(s, ops):
    return [(c.composition.reduced_formula, i) for c, i in expand.children(s, ops)]


def test_scope_needs_strong_bond_element():
    assert expand.in_scope(["Li", "Be", "Sb"]) and expand.in_scope(["B", "As"])
    assert not expand.in_scope(["Li", "Mg", "P"])  # no Be/B/C/N
    assert not expand.in_scope(["Na", "Be", "Sb"])  # Na outside the light-element scope


def test_swap_keeps_valence_electron_count():
    for c, info in expand.children(_zb("B", "As"), ["swap"]):
        m = json.loads(info["map"])
        assert sum(expand.GROUP[k] for k in m) == sum(expand.GROUP[v] for v in m.values())
        assert expand.in_scope(el.symbol for el in c.composition.elements)
    forms = {f for f, _ in _kids(_zb("B", "As"), ["swap"])}
    assert {"BeSe"} - forms == {"BeSe"}  # Se is not in scope
    assert "GeC" in forms and "BeS" in forms  # k = 1: B+As -> C+Ge, Be+S
    assert "BN" not in forms  # B+As -> N+B is a k = 2 swap, pruned 11-21


def test_subst_rescales_volume_and_stays_isovalent():
    p = _zb("B", "As")
    for c, info in expand.children(p, ["subst"]):
        m = json.loads(info["map"])
        assert all(expand.GROUP[k] == expand.GROUP[v] for k, v in m.items())
    kids = dict((f, c) for (f, _), (c, _) in zip(_kids(p, ["subst"]), expand.children(p, ["subst"])))
    assert kids["BN"].volume < p.volume < kids["BSb"].volume


def test_perm_gives_other_orderings_of_the_same_formula():
    p = _heusler("Li", "Be", "Sb")
    kids = list(expand.children(p, ["perm"]))
    assert kids and all(c.composition.reduced_formula == "LiBeSb" for c, _ in kids)
    assert all(abs(c.volume - p.volume) < 1e-6 for c, _ in kids)
