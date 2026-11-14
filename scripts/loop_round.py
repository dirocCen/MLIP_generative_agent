"""One real round of the loop: fit the surrogate on every L1 label so far, propose the next L1 batch.

Same ``Surrogate`` and ``rank`` as the replay, so the strategy that wins the replay is the one used
here (``slack_ucb`` differs from ``ucb`` only at the replay's cold start, so here it ranks as
``ucb``). A batch takes at most ``--max-per-formula`` structures of one reduced formula (polymorphs
of one formula have near-identical descriptors; without the cap UCB spends a batch on one spot).

Writes runs/<run_id>/rounds/<round>.csv: mu, sd, acquisition score, label (if any) and whether it
was picked, for every L0-stable structure, so the real rounds can be scored later exactly like
the replay. Dry run by default; ``--commit`` writes a ``loop_round`` record and the set
``loop.round.<round>``. Submitting L1 for that set is a separate step that needs the user's approval.

Usage: MGA_RUN_ID=loop-v1 python scripts/loop_round.py --set <L0 set> [--set ...] --round r1 [--commit]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from mga import registry
from mga.loop.data import FEATURES, build_table
from mga.loop.surrogate import STRATEGIES, Surrogate, rank
from mga.records import recorded_step
from mga.tools._common import parent_steps

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="relaxed / L0 sets (union)")
    ap.add_argument("--round", required=True)
    ap.add_argument("--strategy", default="ucb", choices=STRATEGIES)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-per-formula", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decided-by", default="agent", choices=["agent", "human"])
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()

    sids: list[str] = []
    for s in a.set:
        sids += [x for x in registry.set_sids(s) if x not in sids]
    t = build_table(sids)
    t = t[t["l0_ok"]].dropna(subset=list(FEATURES))
    train = t[t["kappa_l1"].notna()]
    cand = t[t["kappa_l1"].isna()]
    if len(train) < 5:
        raise SystemExit(f"only {len(train)} labelled structures: run the stratified L1 sample first")

    X = lambda d: d[list(FEATURES)].to_numpy(float)  # noqa: E731
    sur = Surrogate(a.seed).fit(X(train), train["log_kappa_l1"].to_numpy(float))
    mu, sd = sur.predict(X(t))
    t = t.assign(mu_log_kappa=mu, sd_log_kappa=sd, mu_kappa=10 ** mu, ucb=mu + a.beta * sd, picked=False)

    order = rank(a.strategy, X(cand), cand["kappa_slack"].to_numpy(float),
                 np.random.default_rng(a.seed), sur, a.beta)
    picked, per_formula = [], {}
    for i in order:
        sid, f = cand.index[i], cand["formula"].iloc[i]
        if per_formula.get(f, 0) >= a.max_per_formula:
            continue
        picked.append(sid)
        per_formula[f] = per_formula.get(f, 0) + 1
        if len(picked) == a.batch:
            break
    t.loc[picked, "picked"] = True

    run_id = os.environ.get("MGA_RUN_ID", "loop-v1")
    out = ROOT / "runs" / run_id / "rounds" / f"{a.round}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sid", "formula", "chemsys", "kappa_slack", "kappa_l1", "mu_kappa", "mu_log_kappa",
            "sd_log_kappa", "ucb", "picked", "n_atoms_primitive"]
    t[cols].sort_values("ucb", ascending=False).to_csv(out, index=False)

    batch = t.loc[picked]
    est = float((0.69 * batch["n_atoms_primitive"].fillna(8) / 8.0).sum())
    summary = {
        "round": a.round, "strategy": a.strategy, "n_train": len(train), "n_candidates": len(cand),
        "batch": len(picked), "est_l1_gpu_h": round(est, 1), "predictions": str(out.relative_to(ROOT)),
        "train_best": train.nlargest(3, "kappa_l1")[["formula", "kappa_l1"]].round(0).values.tolist(),
    }
    print(json.dumps(summary, indent=1, default=str))
    print(batch[["formula", "chemsys", "kappa_slack", "mu_kappa", "sd_log_kappa"]].round(2).to_string())
    if not a.commit:
        print("dry run: nothing written except the predictions csv (add --commit)")
        return

    new_set = f"loop.round.{a.round}"
    with recorded_step(
        "loop_round",
        inputs={"sets": a.set, "round": a.round, "strategy": a.strategy, "batch": a.batch,
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
