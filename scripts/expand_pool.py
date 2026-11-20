"""Expand the pool after an L1/L2 round towards novel, high-kappa structures (``mga.loop.expand``).

Two steps around the usual L0 job:

``propose``  parents (label + UCB) -> children (subst / swap / perm) -> dedupe among themselves and
             against every registered structure -> drop children that match an MP structure of the
             same formula (novelty = absent from MP; ``formula_in_mp`` is kept as a flag) ->
             runs/<run>/expand/<tag>.csv. ``--commit`` writes the extxyz and imports set
             ``loop.pool.expand.<tag>``; L0 on it is a separate submission.
``screen``   after L0: phonon-stable children (no imaginary modes) with E_hull against MP recorded,
             not filtered, except below the MP hull by more than ``--ehull-floor`` (an MLIP artefact,
             not a discovery) -> set ``loop.pool.expand.<tag>.screened``. That set goes into
             ``loop_round.py --set`` with the other L0 sets, where Slack/UCB rank it for L1.

Usage:
  export MP_API_KEY=...  # novelty and E_hull query MP
  MGA_RUN_ID=loop-v1 python scripts/expand_pool.py propose --tag e1 [--commit]
  MGA_RUN_ID=loop-v1 python scripts/expand_pool.py screen --tag e1 [--commit]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher

from mga import registry
from mga.loop import expand
from mga.loop.data import DEFAULT_L0_HASH, DEFAULT_MODEL, L0_KIND, build_table
from mga.results import ResultMissing, load_result

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_pool import fingerprint  # noqa: E402

# every L0 pool so far (504f29978e8c protocol); their L1/L2 labels choose the parents
DEFAULT_SETS = [
    "loop.pool.v1.bcn200.l0stable.w10",
    "pool.mass_contrast.v1.l0stable.dpa41_max_oam.504f29",
    "pool.nowotny_juza.v1.l0stable.dpa41_max_oam.504f29",
    "pool.nowotny_juza.v1b.l0stable.dpa41_max_oam.504f29",
]
MATCHER = dict(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)


def _l0(sid: str) -> dict | None:
    try:
        return load_result(L0_KIND, DEFAULT_MODEL, sid, DEFAULT_L0_HASH)["data"]
    except ResultMissing:
        return None


def _relaxed(sid: str):
    from mga.tools.analysis import _structure_from_result

    d = _l0(sid)
    if d and d.get("relaxed_structure"):
        return _structure_from_result(d["relaxed_structure"])
    return registry.load_structure(sid)


def _dedupe(items, matcher):
    groups = defaultdict(list)
    for s, info in items:
        groups[(s.composition.reduced_formula, len(s), fingerprint(s))].append((s, info))
    kept = []
    for key in sorted(groups, key=str):
        reps = []
        for s, info in groups[key]:
            if not any(matcher.fit(s, r) for r, _ in reps):
                reps.append((s, info))
        kept.extend(reps)
    return kept


def _registered_by_formula(formulas: set[str]) -> dict[str, list[str]]:
    """Every registered structure (any set) with one of these reduced formulas."""
    sids = {sid for meta in registry.iter_sets() for sid in registry.set_sids(meta["name"])}
    out: dict[str, list[str]] = defaultdict(list)
    for e in registry.iter_entries(sorted(sids)):
        if e["reduced_formula"] in formulas:
            out[e["reduced_formula"]].append(e["sid"])
    return out


def propose(a) -> None:
    from mga.analysis.ehull import mp_structures

    sids: list[str] = []
    for s in a.set:
        sids += [x for x in registry.set_sids(s) if x not in sids]
    par = expand.select_parents(build_table(sids), a.n_label, a.n_ucb, a.beta, a.max_per_formula, a.seed)
    print(par[["formula", "parent_reason", "kappa", "label_source", "mu_kappa", "kappa_slack"]].round(1).to_string())

    raw = []
    for sid, row in par.iterrows():
        for c, info in expand.children(_relaxed(sid), a.operators):
            info.update(parent=sid, parent_formula=row["formula"], parent_reason=row["parent_reason"],
                        parent_kappa=None if pd.isna(row["kappa"]) else round(float(row["kappa"]), 1),
                        parent_mu_kappa=round(float(row["mu_kappa"]), 1))
            raw.append((c, info))
    matcher = StructureMatcher(**MATCHER)
    kids = _dedupe(raw, matcher)
    print(f"children: {len(raw)} raw, {len(kids)} unique", flush=True)

    known = _registered_by_formula({s.composition.reduced_formula for s, _ in kids})
    kids = [(s, i) for s, i in kids
            if not any(matcher.fit(s, registry.load_structure(x)) for x in known.get(s.composition.reduced_formula, []))]
    print(f"new vs registry: {len(kids)}", flush=True)

    nov = expand.mp_novelty([s for s, _ in kids], matcher, mp_structures)
    rows, new = [], []
    for (s, info), n in zip(kids, nov):
        row = {"formula": s.composition.reduced_formula, "n_atoms": len(s), "spg": fingerprint(s)[0] if fingerprint(s) else None,
               **info, "formula_in_mp": n["formula_in_mp"], "mp_matches": ",".join(n["mp_matches"]),
               "mp_min_ehull_formula": n["mp_min_ehull_formula"]}
        rows.append(row)
        if not n["mp_matches"]:
            info.update(formula_in_mp=n["formula_in_mp"])
            new.append((s, info))
    # at most --per-parent novel children per parent, lightest mean atomic mass first (Slack prior:
    # light atoms + strong bonds; without the cap the 4-element parents give ~200 children each)
    mass = lambda s: s.composition.weight / s.composition.num_atoms  # noqa: E731
    capped, n_par = [], Counter()
    for s, info in sorted(new, key=lambda it: mass(it[0])):
        if n_par[info["parent"]] < a.per_parent:
            capped.append((s, info))
            n_par[info["parent"]] += 1
    n_novel_uncapped, new = len(new), capped
    kept = {id(i) for _, i in new}
    for row, (_, info) in zip(rows, kids):
        row["selected"] = id(info) in kept
    df = pd.DataFrame(rows)
    out = ROOT / "runs" / os.environ.get("MGA_RUN_ID", "loop-v1") / "expand" / f"{a.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = {
        "tag": a.tag, "n_parents": len(par), "parents_by_reason": par["parent_reason"].value_counts().to_dict(),
        "children_raw": len(raw), "new_vs_registry": len(rows), "known_in_mp": int((df["mp_matches"] != "").sum()),
        "novel_uncapped": n_novel_uncapped, "novel": len(new), "novel_new_formula": sum(not i["formula_in_mp"] for _, i in new),
        "novel_by_source": dict(Counter(i["source"] for _, i in new)),
        "novel_by_parent": dict(Counter(i["parent_formula"] for _, i in new).most_common()),
        "novel_n_atoms": dict(sorted(Counter(len(s) for s, _ in new).items())), "csv": str(out.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=1, default=str))
    if not a.commit or not new:
        print("dry run: only the csv written (add --commit)" if new else "no novel children")
        return

    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor

    from mga.tools.structures import import_structures

    xyz = ROOT / "data" / "structures" / "pool" / f"loop_pool_expand_{a.tag}.extxyz"
    frames = []
    for s, info in new:
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.info.update({k: v for k, v in info.items() if v is not None})
        frames.append(atoms)
    write(xyz, frames, format="extxyz")
    res = import_structures(
        str(xyz.relative_to(ROOT)), f"loop.pool.expand.{a.tag}", "file", decided_by=a.decided_by,
        rationale=f"Pool expansion {a.tag}: subst/swap/perm children of {len(par)} parents (top label + UCB), "
                  "light strong-bond scope, deduplicated against the registry, none matching an MP structure.",
        description=f"Expansion {a.tag}: novel (not in MP) children of the loop leaders",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "sids"}, indent=1, default=str))


def screen(a) -> None:
    from mga.analysis.ehull import e_above_hull
    from mga.records import recorded_step
    from mga.tools._common import parent_steps
    from mga.tools.analysis import _structure_from_result

    name = f"loop.pool.expand.{a.tag}"
    rows, items = [], []
    for e in registry.iter_entries(registry.set_sids(name)):
        d = _l0(e["sid"])
        row = {"sid": e["sid"], "formula": e["reduced_formula"], "l0": d is not None and bool(d.get("ok"))}
        if row["l0"]:
            row["stable"] = not d.get("has_imaginary_modes")
            row["min_frequency_thz"] = d.get("min_frequency_thz")
            row["kappa_slack"] = (d.get("descriptors") or {}).get("kappa_slack_W_mK")
            if d.get("relaxed_structure"):
                items.append((e["sid"], _structure_from_result(d["relaxed_structure"]), float(d["relax"]["energy"])))
        rows.append(row)
    t = pd.DataFrame(rows).set_index("sid")
    per, mp_meta = e_above_hull(items)
    t["e_above_hull"] = pd.Series({k: v.get("e_above_hull") for k, v in per.items()}, dtype=float)
    t["artefact"] = t["e_above_hull"] < -a.ehull_floor
    keep = t[t["l0"] & t["stable"].fillna(False).astype(bool) & ~t["artefact"]]
    out = ROOT / "runs" / os.environ.get("MGA_RUN_ID", "loop-v1") / "expand" / f"{a.tag}.screen.csv"
    t.sort_values("kappa_slack", ascending=False).to_csv(out)
    summary = {"tag": a.tag, "n": len(t), "l0_done": int(t["l0"].sum()), "stable": int(t["stable"].fillna(False).sum()),
               "artefact_below_hull": t.index[t["artefact"]].tolist(), "kept": len(keep), "csv": str(out.relative_to(ROOT))}
    print(json.dumps(summary, indent=1, default=str))
    print(keep[["formula", "kappa_slack", "e_above_hull", "min_frequency_thz"]].sort_values("kappa_slack", ascending=False)
          .round(3).to_string())
    if not a.commit:
        print("dry run (add --commit)")
        return
    with recorded_step(
        "expand_screen", inputs={"set_name": name, "ehull_floor": a.ehull_floor}, parent_ids=parent_steps(name),
        decided_by=a.decided_by,
        rationale="Phonon-stable expansion children; E_hull recorded, not filtered (user 11-19), except below-MP-hull artefacts.",
    ) as rec:
        registry.create_set(f"{name}.screened", list(keep.index), operation="filter", parent_sets=[name],
                            description=f"Expansion {a.tag}: L0-stable, not below the MP hull by > {a.ehull_floor} eV/atom",
                            details={**summary, "e_above_hull": t["e_above_hull"].round(4).dropna().to_dict(), "mp": mp_meta},
                            step_id=rec.step_id)
        rec.outputs = {**summary, "new_set_name": f"{name}.screened"}
    print(f"set {name}.screened written")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--set", action="append", default=None, help=f"L0 sets the parents come from (default {DEFAULT_SETS})")
    p.add_argument("--n-label", type=int, default=12)
    p.add_argument("--n-ucb", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--max-per-formula", type=int, default=1)
    p.add_argument("--per-parent", type=int, default=15, help="novel children kept per parent (lightest first)")
    p.add_argument("--operators", nargs="+", default=list(expand.OPERATORS), choices=list(expand.OPERATORS))
    p.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("screen")
    s.add_argument("--ehull-floor", type=float, default=0.1, help="eV/atom below the MP hull = MLIP artefact")
    for x in (p, s):
        x.add_argument("--tag", required=True)
        x.add_argument("--decided-by", default="agent", choices=["agent", "human"])
        x.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    if a.cmd == "propose":
        a.set = a.set or DEFAULT_SETS
        propose(a)
    else:
        screen(a)


if __name__ == "__main__":
    main()
