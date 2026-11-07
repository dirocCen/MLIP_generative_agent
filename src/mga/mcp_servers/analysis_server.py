"""mga-analysis: job status, results, E_hull, kappa_SRME, model disagreement, records."""

from __future__ import annotations

from mga.mcp_servers.common import add, make_server
from mga.tools import analysis as t

server = make_server(
    "mga-analysis",
    "Reads job states from jobflow-remote and everything else from the local result cache. Pass the step_id returned "
    "by a submit tool. Per-structure tables are written to files (details_file); summaries come back here.",
)

add(server, t.job_status, "Job states of one submission (counts, failures with the end of their error message).",
    read_only=True, idempotent=True, open_world=True)
add(server, t.fetch_results,
    "Copy finished job outputs into the result cache, then derive follow-up sets (relaxed structures, MatterGen samples). "
    "Safe to call repeatedly; sets are only created once all jobs have finished unless allow_partial=true.",
    read_only=False, idempotent=True, open_world=True)
add(server, t.compute_ehull,
    "E_hull of MLIP-relaxed structures vs the MP hull with MP2020 corrections (models with an MP-compatible PBE "
    "reference only). Optionally creates a set of the input structures below the threshold.",
    read_only=False, idempotent=True, open_world=True)
add(server, t.kappa_srme,
    "Score a kappa submission on the kappa_SRME benchmark structures against phononDB PBE (port of the "
    "matbench-discovery code). Reports the mean over scored entries and the mean with missing entries counted as 2.",
    read_only=False, idempotent=True)
add(server, t.model_disagreement,
    "Pairwise kappa disagreement (SRD, mode-resolved SRME, frequency MAE) between 2-4 models on the same structures. "
    "Optionally puts the top_n most disagreeing structures in a new set.",
    read_only=False, idempotent=True)
add(server, t.kappa_convergence, "Table of kappa vs the scanned setting for a convergence submission.",
    read_only=False, idempotent=True)
add(server, t.budget_ledger,
    "GPU-hours spent by ladder submissions (rung -> step_ids: L0 fc2_descriptors, L1 kappa, L2 lbte, L3 fc4), per rung "
    "and step, against an optional total budget.",
    read_only=True, idempotent=True)
add(server, t.escalation_baseline,
    "Replay the fixed rule (top-k by committee mean per rung) and random selection on the cached ladder results under "
    "one GPU-hour budget, and score the agent's rung sets the same way: hits found, recall, budget spent, rank "
    "agreement between rungs. Needs L0 for the whole pool.",
    read_only=False, idempotent=True)
add(server, t.query_records, "Search step records by run, tool, status or a substring (e.g. a set name or structure ID).",
    read_only=True, idempotent=True)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
