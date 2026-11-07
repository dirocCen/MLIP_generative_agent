"""mga-dft: VASP on dirona, behind a human approval gate."""

from __future__ import annotations

from mga.mcp_servers.common import add, make_server
from mga.tools import dft as t

server = make_server(
    "mga-dft",
    "VASP (6.4.2, PBE_54 POTCARs, MP-compatible input sets) on dirona. preview_vasp_inputs and estimate_vasp_cost are "
    "free and local. request_vasp_submission only creates an approval request: after calling it you MUST stop and ask "
    "the user to review and approve it in their own terminal (`mga approvals show/approve <id>`). There is no tool to "
    "approve. Call submit_vasp only after the user tells you they approved it. calc: relax_static; phonon_fd (atomate2 "
    "harmonic phonons); phono3py_fd (DFT FC2+FC3 by finite displacements -- random+symfc or systematic, whichever "
    "needs fewer supercells -- and Wigner kappa with the same supercells and q-mesh as the MLIP kappa tool; the "
    "estimate lists the exact number of displaced supercells); convergence (ENCUT / k-point scans).",
)

add(server, t.preview_vasp_inputs,
    "Write INCAR/KPOINTS/POSCAR/POTCAR.spec for a few structures and list every difference from the plain MP sets; "
    "warns about odd electron counts and transition metals.",
    read_only=False, idempotent=True)
add(server, t.estimate_vasp_cost,
    "Number of calculations, atoms, k-points and an uncalibrated relative cost index for a set.",
    read_only=True, idempotent=True)
add(server, t.request_vasp_submission,
    "Freeze a VASP plan (structures, parameters, resources) and create an approval request. Validates every flow first. "
    "Returns STOP: do not submit until the user has approved it.",
    read_only=False)
add(server, t.submit_vasp,
    "Submit an approved plan (live mode only). Takes only the approval id; the frozen plan is what runs. Each approval "
    "can be used once.",
    read_only=False, destructive=False, open_world=True)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
