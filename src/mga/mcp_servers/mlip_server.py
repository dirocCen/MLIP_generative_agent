"""mga-mlip: MatterGen generation and MLIP relaxation / phonon / kappa jobs on datalab."""

from __future__ import annotations

from mga.mcp_servers.common import add, make_server
from mga.tools import mlip as t

server = make_server(
    "mga-mlip",
    "Submits GPU jobs to datalab through jobflow-remote. In dry_run mode nothing is submitted (the flow is built and "
    "recorded); in replay mode only cached results are returned. Results are cached per (structure, model, parameter "
    "hash), so re-submitting the same thing is free. Defaults follow the kappa_SRME protocol; read "
    "describe_parameters before overriding anything. Follow up with job_status / fetch_results (mga-analysis).",
)

add(server, t.list_models,
    "Configured MLIPs: energy reference, committee membership, reported kappa_SRME, and TODOs that block live use.",
    read_only=True, idempotent=True)
add(server, t.describe_parameters,
    "Parameter schema of one calculation type: defaults, bounds and where each default comes from.",
    read_only=True, idempotent=True)
add(server, t.submit_mattergen,
    "Generate structures with MatterGen conditioned on a chemical system (e.g. 'B-N'). fetch_results registers the "
    "samples as the set mattergen.<chemsys>__n<N>__<tag>.",
    read_only=False, open_world=True)
add(server, t.submit_mlip_relax,
    "Relax every structure of a set with one MLIP (kappa_SRME relaxation protocol). fetch_results creates the set "
    "<set>.relaxed.<model>.<hash>.",
    read_only=False, open_world=True)
add(server, t.submit_mlip_phonon_kappa,
    "FC2 -> imaginary-mode check -> FC3 -> Wigner-RTA kappa (300 K) with one MLIP. Supercells and q-mesh are fixed "
    "before submission (phononDB settings for benchmark structures, length rule otherwise), so every model uses the "
    "same settings.",
    read_only=False, open_world=True)
add(server, t.submit_mlip_fc2_descriptors,
    "Ladder L0: FC2 on the L1 supercell -> imaginary-mode gate -> harmonic descriptors (acoustic/optical gap, branch "
    "spread, group velocities), temperature-weighted 3-phonon phase space and Grueneisen/Slack estimate. Cheap; for "
    "ranking and artefact checks, not a kappa.",
    read_only=False, open_world=True)
add(server, t.submit_mlip_kappa_lbte,
    "Ladder L2: Wigner LBTE (direct solution) with the same relax/kappa parameters as an L1 run, reading its force "
    "constants from the FC cache when present. The q-mesh is reduced (and recorded) if the collision matrix would not "
    "fit in memory. Gives the LBTE/RTA ratio.",
    read_only=False, open_world=True)
add(server, t.submit_mlip_fc4_fourphonon,
    "Ladder L3: random displacements + symfc fit of FC2/FC3/FC4, ShengBTE/FourPhonon inputs for 3ph and 3+4ph, and the "
    "runs when a FourPhonon command is configured. Expensive; for a few top candidates. Gives the 4ph correction ratio.",
    read_only=False, open_world=True)
add(server, t.submit_mlip_convergence,
    "Scan one kappa setting (supercell widths, q-mesh length or displacement) on <= 10 structures with 2-6 values. "
    "Analyse with kappa_convergence.",
    read_only=False, open_world=True)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
