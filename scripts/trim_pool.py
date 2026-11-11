"""Cap the B-C-N ternary part of the loop pool so that one system does not dominate the replay curves.

Deterministic: B-C-N ternaries are grouped by reduced formula, sorted inside each formula by
space group (high first) and then by primitive-cell size (small first), and picked round-robin
across formulas until the cap is reached. The R3m-BNC2 positive control is always kept.
Every other structure of the pool is kept. Writes one record (``trim_pool``) and a new set.

Usage:  MGA_RUN_ID=loop-v1 python scripts/trim_pool.py [--cap 200]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mga import registry
from mga.records import recorded_step
from mga.tools._common import parent_steps

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SET = "loop.pool.v1"
NEW_SET = "loop.pool.v1.bcn200"
TRIMMED = frozenset({"B", "C", "N"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=200)
    cap = ap.parse_args().cap

    ref = AseAtomsAdaptor.get_structure(read(ROOT / "data/reference/s0/R3m-BNC2.extxyz"))
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    sids = registry.set_sids(SOURCE_SET)
    entries = list(registry.iter_entries(sids))
    bcn = [e for e in entries if frozenset(e["elements"]) == TRIMMED]
    control = [
        e["sid"] for e in bcn
        if e["reduced_formula"] == ref.composition.reduced_formula
        and matcher.fit(ref, Structure.from_dict(e["structure"]))
    ]

    by_formula = defaultdict(list)
    for e in bcn:
        by_formula[e["reduced_formula"]].append(e)
    for group in by_formula.values():
        group.sort(key=lambda e: (-e["spacegroup_number"], e["nsites"], e["sid"]))
    keep = list(control)
    queues = [list(by_formula[f]) for f in sorted(by_formula)]
    while len(keep) < cap and any(queues):
        for q in queues:
            while q and q[0]["sid"] in keep:
                q.pop(0)
            if q and len(keep) < cap:
                keep.append(q.pop(0)["sid"])
    keep_set = set(keep)

    with recorded_step(
        "trim_pool",
        inputs={"set_name": SOURCE_SET, "new_set_name": NEW_SET, "cap": cap, "trimmed_system": "B-C-N"},
        parent_ids=parent_steps(SOURCE_SET),
        decided_by="human",
        rationale=(
            "User (11-11): cap the B-C-N ternaries (605 of 1209) at about 200 so that one chemical system "
            "does not dominate the replay comparison; keep the highest-symmetry structures of every formula "
            "(round-robin over formulas) and always keep the R3m-BNC2 positive control."
        ),
    ) as rec:
        new_sids = [s for s in sids if s not in {e["sid"] for e in bcn} or s in keep_set]
        registry.create_set(
            NEW_SET, new_sids, operation="trim", parent_sets=[SOURCE_SET],
            description=f"Loop pool v1 with the B-C-N ternaries capped at {cap}",
            details={"cap": cap, "positive_control": control}, step_id=rec.step_id,
        )
        rec.outputs = {
            "new_set_name": NEW_SET,
            "n_in": len(sids),
            "n_bcn_in": len(bcn),
            "n_bcn_kept": len(keep),
            "n_bcn_formulas": len(by_formula),
            "n_out": len(new_sids),
            "positive_control": control,
        }
    print(json.dumps({"step_id": rec.step_id, **rec.outputs}, indent=1))


if __name__ == "__main__":
    main()
