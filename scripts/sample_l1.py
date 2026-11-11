"""Stratified L1 sample of the L0-stable pool: the labelled universe for the replay comparison.

The sample must not favour any acquisition strategy, so it uses only chemical system and the
Slack tercile inside each system (which guarantees low-kappa negatives as well as high ones).
Quota per chemical system is proportional to sqrt(n) (at least 1). Always included: hidden
rediscovery targets and structures that already have an L1 result (free).

Default is a dry run that prints the plan and the GPU-hour estimate. ``--commit`` writes a
``sample_l1`` record and the set (run only after the user approved the budget).

Usage: MGA_RUN_ID=loop-v1 python scripts/sample_l1.py --set <l0-stable set> [--budget 130] [--commit]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np

from mga import registry
from mga.loop.data import build_table
from mga.records import recorded_step
from mga.tools._common import parent_steps

from loop_replay import TARGETS  # noqa: E402  (same directory)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--budget", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new-set", default="loop.l1.v1")
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()

    t = build_table(registry.set_sids(a.set))
    t = t[t["l0_ok"]].copy()
    entries = {e["sid"]: e for e in registry.iter_entries(list(t.index))}
    t["sg"] = [entries[s]["spacegroup_number"] for s in t.index]

    forced = set(t.index[t["kappa_l1"].notna()])
    for name, (formula, sgs) in TARGETS.items():
        forced |= set(t.index[(t["formula"] == formula) & t["sg"].isin(sgs)])

    rng = np.random.default_rng(a.seed)
    n_sys = t.groupby("chemsys").size()
    weights = np.sqrt(n_sys)
    remaining = max(a.budget - len(forced), 0)
    quota = {cs: max(1, math.floor(remaining * w / weights.sum())) for cs, w in weights.items()}

    picked = set(forced)
    for cs, q in quota.items():
        g = t[(t["chemsys"] == cs) & ~t.index.isin(picked)].sort_values("kappa_slack")
        if g.empty:
            continue
        terciles = np.array_split(g.index.to_numpy(), 3)
        k = 0
        while k < q and any(len(x) for x in terciles):
            for i in (2, 0, 1):  # high, low, middle Slack tercile in turn
                if k < q and len(terciles[i]):
                    j = rng.integers(len(terciles[i]))
                    picked.add(terciles[i][j])
                    terciles[i] = np.delete(terciles[i], j)
                    k += 1

    sample = t.loc[sorted(picked)]
    todo = sample[sample["kappa_l1"].isna()]
    n_atoms = todo["n_atoms_primitive"].fillna(8)
    est = float((0.69 * n_atoms / 8.0).sum())  # measured mean 0.69 GPU-h for the 8-atom cells
    plan = {
        "l0_stable": len(t), "budget": a.budget, "n_sample": len(sample),
        "n_forced": len(forced), "n_already_l1": int(sample["kappa_l1"].notna().sum()),
        "n_to_compute": len(todo), "est_gpu_h": round(est, 1),
        "per_chemsys": dict(Counter(sample["chemsys"]).most_common()),
        "targets_in_sample": sorted({n for n, (f, sgs) in TARGETS.items()
                                     if ((sample["formula"] == f) & sample["sg"].isin(sgs)).any()}),
    }
    print(json.dumps(plan, indent=1))
    if not a.commit:
        print("dry run: nothing written (add --commit after approval)")
        return
    with recorded_step(
        "sample_l1",
        inputs={"set_name": a.set, "new_set_name": a.new_set, "budget": a.budget, "seed": a.seed},
        parent_ids=parent_steps(a.set),
        decided_by="human",
        rationale="Stratified L1 sample (chemical system x Slack tercile, sqrt quota) = the replay universe; "
                  "budget approved by the user.",
    ) as rec:
        registry.create_set(a.new_set, list(sample.index), operation="sample", parent_sets=[a.set],
                            description="Stratified L1 sample of the L0-stable loop pool (replay universe)",
                            details=plan, step_id=rec.step_id)
        rec.outputs = plan


if __name__ == "__main__":
    main()
