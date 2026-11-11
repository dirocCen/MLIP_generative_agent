"""Grow the pool from the loop's best structures: same-group element substitution, then dedupe.

Parents are the top ``--top`` L0-stable structures of a round's predictions csv, ranked by the L1
label where it exists and by the surrogate mean otherwise. Each parent's relaxed structure goes
through the same substitution as the pool build (``build_pool.substitutions``: same-group swaps,
volume rescaled by covalent radii, kept inside the 16 chemical systems), then the primitive-cell,
size and d_min filters, and dedupe among the children and against the existing pool sets.

The children enter the next round only through relax -> L0 -> surrogate, like every pool member.
Dry run by default; ``--commit`` writes the extxyz and imports it as set ``loop.pool.grow.<round>``.

Usage: MGA_RUN_ID=loop-v1 python scripts/grow_pool.py --round r1 --against loop.pool.v1 [--commit]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from ase.io import write
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from build_pool import MAX_ATOMS, MAX_ELEMENTS, MIN_DIST, dedupe, min_dist, primitive, substitutions
from mga import registry

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--against", action="append", default=["loop.pool.v1"],
                    help="existing pool sets the children must not duplicate")
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()

    run_id = os.environ.get("MGA_RUN_ID", "loop-v1")
    pred = pd.read_csv(ROOT / "runs" / run_id / "rounds" / f"{a.round}.csv")
    pred["score"] = pred["kappa_l1"].fillna(pred["mu_kappa"])
    top = pred.nlargest(a.top, "score")

    protos = [(registry.load_structure(sid), sid, None) for sid in top["sid"]]
    children = []
    for s, info in substitutions(protos):
        if info["source"] != "subst":
            continue
        p = primitive(s)
        if p is None or len(p) > MAX_ATOMS or len(p.composition.elements) > MAX_ELEMENTS or min_dist(p) < MIN_DIST:
            continue
        children.append((p, info))
    children = dedupe(children)

    existing: dict[str, list] = {}
    for name in a.against:
        for e in registry.iter_entries(registry.set_sids(name)):
            existing.setdefault(e["reduced_formula"], []).append(e["sid"])
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
    new = []
    for s, info in children:
        same = existing.get(s.composition.reduced_formula, [])
        if any(matcher.fit(s, registry.load_structure(sid)) for sid in same):
            continue
        new.append((s, info))

    summary = {
        "round": a.round, "parents": top[["sid", "formula", "score"]].round(1).values.tolist(),
        "children_after_filters": len(children), "new_vs_pool": len(new),
        "by_formula": dict(Counter(s.composition.reduced_formula for s, _ in new).most_common()),
    }
    print(json.dumps(summary, indent=1, default=str))
    if not a.commit or not new:
        print("dry run: nothing written (add --commit)" if new else "no new structures")
        return

    out = ROOT / "data" / "structures" / "pool" / f"loop_pool_grow_{a.round}.extxyz"
    frames = []
    for s, info in new:
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.info.update({k: v for k, v in info.items() if v is not None})
        frames.append(atoms)
    write(out, frames, format="extxyz")

    from mga.tools.structures import import_structures

    res = import_structures(
        str(out.relative_to(ROOT)), f"loop.pool.grow.{a.round}", "file", decided_by="agent",
        rationale=f"Pool growth after loop round {a.round}: same-group substitution of the top {a.top} "
                  "structures (L1 label, else surrogate mean), deduplicated against the pool.",
        description=f"Substitution children of the round-{a.round} leaders",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "sids"}, indent=1, default=str))


if __name__ == "__main__":
    main()
