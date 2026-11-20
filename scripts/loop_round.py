"""One real round of the loop: fit the surrogate on every L1 label so far, propose the next L1 batch.

Same ``Surrogate`` and ``rank`` as the replay, so the strategy that wins the replay is the one used
here (``slack_ucb`` differs from ``ucb`` only at the replay's cold start, so here it ranks as
``ucb``). A batch takes at most ``--max-per-formula`` structures of one reduced formula (polymorphs
of one formula have near-identical descriptors; without the cap UCB spends a batch on one spot).

Writes runs/<run_id>/rounds/<round>.csv: mu, sd, acquisition score, label (if any) and whether it
was picked, for every L0-stable structure, so the real rounds can be scored later exactly like
the replay. Dry run by default; ``--commit`` writes a ``loop_round`` record and the set
``loop.round.<round>``. Submitting L1 for that set is a separate step that needs the user's approval.

Usage: MGA_RUN_ID=loop-v1 python scripts/loop_round.py --set <L0 set> [--set ...] --round r1 \
    --exclude-set <every set already sent to L1> [--pending-set <sets still running>] [--max-fc2-atoms 360] [--commit]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mga import registry
from mga.loop.data import FEATURES, build_table
from mga.loop.surrogate import STRATEGIES, Surrogate, rank
from mga.records import recorded_step
from mga.tools._common import parent_steps

ROOT = Path(__file__).resolve().parents[1]
# L1 cost (GPU h) ~ COST_A * n_fc3_displacements * n_atoms_fc3**COST_P: least squares in log space
# on the 98 L1 labels of loop.l1.v1 + loop.round.r1 (L40S + A100, 11-15), per-structure ratio
# 0.69-1.41 (10-90 %). Superlinear in atoms because the DPA neighbour search is dense O(N^2)
# (no vesin on the cluster); the linear fit (1.03e-6 * disp * atoms) said 10 h for the 384-atom
# FC3 cells of r1, which took > 24 h. The old rule (0.69 h per 8 primitive atoms) was worse still.
COST_A, COST_P, COST_RANGE = 3.55e-8, 1.757, (0.69, 1.41)
JOB_LIMIT_H = 24.0  # GpuResources.time default; a cell whose 90 % estimate exceeds it needs a longer --time
# FC2 supercell atoms that fit (11-14): <= 360 on L40S 44 GB (>= 399 OOM), <= 672 on A100 80 GB
# (>= 720 OOM); "untested" = larger: run those at a smaller --fc2-min-width (10 A).
FC2_FIT = ((360, "l40s"), (672, "a100"))


def l1_cost(sids: list[str], fc2_min_width: float) -> pd.DataFrame:
    """Supercell sizes, FC3 displacement count, estimated GPU h and target GPU per structure,
    with the same settings the L1 submission will fix (a few s per structure)."""
    from ase import Atoms
    from pymatgen.core import Structure

    from mga.loop.data import DEFAULT_MODEL
    from mga.specs import MLIPPhononKappaParams
    from mga.tools.mlip import _model, _resolve_mlip, kappa_items
    from mga.workflows.kappa_protocol import init_phono3py

    kp, _ = _resolve_mlip(_model(DEFAULT_MODEL), MLIPPhononKappaParams, {"fc2_min_width": fc2_min_width},
                          "loop round L1 protocol")
    rows = []
    for it in kappa_items(sids, kp):
        st, s = dict(it["settings"]), Structure.from_dict(it["structure"])
        init_phono3py(Atoms([str(x.specie) for x in s], cell=s.lattice.matrix, positions=s.cart_coords, pbc=True),
                      st, kp.model_dump())
        gpu = next((g for n, g in FC2_FIT if st["n_atoms_fc2"] <= n), "untested")
        rows.append({"sid": it["sid"], "n_atoms_fc2": st["n_atoms_fc2"], "n_atoms_fc3": st["n_atoms_fc3"],
                     "n_fc3_disp": st["n_fc3_displacements"], "gpu": gpu,
                     "est_gpu_h": COST_A * st["n_fc3_displacements"] * st["n_atoms_fc3"] ** COST_P})
    return pd.DataFrame(rows).set_index("sid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="relaxed / L0 sets (union)")
    ap.add_argument("--round", required=True)
    ap.add_argument("--exclude-set", action="append", default=[],
                    help="sets whose sids are never candidates: L1 already attempted or in flight "
                         "(failed L1 has no label, so without this it would be proposed again)")
    ap.add_argument("--pending-set", action="append", default=[],
                    help="sets still running L1: excluded like --exclude-set, and their unlabelled sids count "
                         "toward --max-per-formula (else a round re-picks the formulas of the one in flight)")
    ap.add_argument("--max-fc2-atoms", type=int, default=None,
                    help="skip candidates whose FC2 supercell is larger (e.g. 360 = fits an L40S)")
    ap.add_argument("--strategy", default="ucb", choices=STRATEGIES)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-per-formula", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fc2-min-width", type=float, default=12.0,
                    help="L1 FC2 width the batch will be run with (cost estimate and GPU placement only)")
    ap.add_argument("--decided-by", default="agent", choices=["agent", "human"])
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()

    sids: list[str] = []
    for s in a.set:
        sids += [x for x in registry.set_sids(s) if x not in sids]
    t = build_table(sids)
    t = t[t["l0_ok"]].dropna(subset=list(FEATURES))
    train = t[t["kappa"].notna()]
    pending = {x for s in a.pending_set for x in registry.set_sids(s)}
    excluded = {x for s in a.exclude_set for x in registry.set_sids(s)} | pending
    # unlabelled and not already shown to give an invalid label (L2 present but failing)
    cand = t[t["kappa"].isna() & t["kappa_l1"].isna() & ~t["l2_invalid"] & ~t.index.isin(excluded)]
    if len(train) < 5:
        raise SystemExit(f"only {len(train)} labelled structures: run the stratified L1 sample first")

    X = lambda d: d[list(FEATURES)].to_numpy(float)  # noqa: E731
    sur = Surrogate(a.seed).fit(X(train), train["log_kappa"].to_numpy(float))
    mu, sd = sur.predict(X(t))
    t = t.assign(mu_log_kappa=mu, sd_log_kappa=sd, mu_kappa=10 ** mu, ucb=mu + a.beta * sd, picked=False)

    order = rank(a.strategy, X(cand), cand["kappa_slack"].to_numpy(float),
                 np.random.default_rng(a.seed), sur, a.beta)
    picked, per_formula, costs, n_too_big = [], {}, [], 0
    for f in t.loc[t.index.isin(pending) & t["kappa"].isna(), "formula"]:
        per_formula[f] = per_formula.get(f, 0) + 1
    for i in order:
        sid, f = cand.index[i], cand["formula"].iloc[i]
        if per_formula.get(f, 0) >= a.max_per_formula:
            continue
        costs.append(l1_cost([sid], a.fc2_min_width))
        if a.max_fc2_atoms and costs[-1]["n_atoms_fc2"].iloc[0] > a.max_fc2_atoms:
            n_too_big += 1
            continue
        picked.append(sid)
        per_formula[f] = per_formula.get(f, 0) + 1
        if len(picked) == a.batch:
            break
    t.loc[picked, "picked"] = True

    run_id = os.environ.get("MGA_RUN_ID", "loop-v1")
    out = ROOT / "runs" / run_id / "rounds" / f"{a.round}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sid", "formula", "chemsys", "kappa_slack", "kappa_l1", "kappa_l2", "kappa", "label_source", "mu_kappa", "mu_log_kappa",
            "sd_log_kappa", "ucb", "picked", "n_atoms_primitive"]
    t[cols].sort_values("ucb", ascending=False).to_csv(out, index=False)

    batch = t.loc[picked].join(pd.concat(costs))
    est = float(batch["est_gpu_h"].sum())
    summary = {
        "round": a.round, "strategy": a.strategy, "n_train": len(train), "train_label_source": train["label_source"].value_counts().to_dict(), "n_candidates": len(cand),
        "n_excluded_unlabelled": int((t["kappa"].isna() & t.index.isin(excluded)).sum()),
        "n_pending": len(pending), "n_skipped_fc2_too_big": n_too_big,
        "batch": len(picked), "fc2_min_width": a.fc2_min_width, "est_l1_gpu_h": round(est, 1),
        "est_l1_gpu_h_range": [round(COST_RANGE[0] * est, 1), round(COST_RANGE[1] * est, 1)],
        "over_job_limit": batch.index[batch["est_gpu_h"] * COST_RANGE[1] > JOB_LIMIT_H].tolist(),
        "gpu_counts": batch["gpu"].value_counts().to_dict(), "predictions": str(out.relative_to(ROOT)),
        "train_best": train.nlargest(3, "kappa")[["formula", "kappa"]].round(0).values.tolist(),
    }
    print(json.dumps(summary, indent=1, default=str))
    print(batch[["formula", "kappa_slack", "mu_kappa", "sd_log_kappa", "n_atoms_fc2", "n_fc3_disp", "gpu",
                 "est_gpu_h"]].round(2).to_string())
    if not a.commit:
        print("dry run: nothing written except the predictions csv (add --commit)")
        return

    new_set = f"loop.round.{a.round}"
    with recorded_step(
        "loop_round",
        inputs={"sets": a.set, "exclude_sets": a.exclude_set, "pending_sets": a.pending_set,
                "max_fc2_atoms": a.max_fc2_atoms, "round": a.round, "strategy": a.strategy, "batch": a.batch,
                "beta": a.beta, "max_per_formula": a.max_per_formula, "seed": a.seed},
        parent_ids=[p for s in a.set for p in parent_steps(s)],
        decided_by=a.decided_by,
        rationale=f"Loop round {a.round}: {a.strategy} on a GP surrogate fitted to {len(train)} L1 labels.",
    ) as rec:
        registry.create_set(new_set, picked, operation="select", parent_sets=list(a.set),
                            description=f"Loop round {a.round}: next L1 batch ({a.strategy})",
                            details=summary, step_id=rec.step_id)
        rec.outputs = {**summary, "new_set_name": new_set, "sids": picked}
    print(f"set {new_set} written (step {rec.step_id})")


if __name__ == "__main__":
    main()
