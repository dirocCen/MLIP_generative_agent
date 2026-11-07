"""mga-structures: local structure registry, named sets, filtering, novelty and leakage checks."""

from __future__ import annotations

from mga.mcp_servers.common import add, make_server
from mga.tools import structures as t

server = make_server(
    "mga-structures",
    "Local structure registry. Every structure gets a stable ID; tools take and create named sets. "
    "Typical order for generated structures: import_structures -> filter_structures -> deduplicate -> "
    "check_benchmark_leakage -> check_novelty. Nothing here runs on a cluster.",
)

add(server, t.import_structures,
    "Import structures from a file inside the project (extxyz/xyz/cif/json/zip of these), register them and "
    "create a named set. Use origin_kind='kappa_srme_reference' only for the phononDB benchmark file.",
    read_only=False, idempotent=True)
add(server, t.structure_summary,
    "Composition, size and space-group distribution of a set, plus a few examples. Use this instead of reading files.",
    read_only=True, idempotent=True)
add(server, t.list_sets, "List existing named sets (name, size, how they were made).", read_only=True, idempotent=True)
add(server, t.merge_sets,
    "Union of several sets (e.g. one MatterGen set per chemical system) into one new set, in the given order.",
    read_only=False, idempotent=True)
add(server, t.symmetrize_structures,
    "S3: replace each structure by its idealized standard primitive cell at a loose tolerance (default 0.1 A). Use it "
    "between a pre-relax and the kappa_SRME relax of generated structures, which otherwise stay P1.",
    read_only=False, idempotent=True)
add(server, t.filter_structures,
    "Filter a set by atom count (max_primitive_sites counts the standardized primitive cell, symprec 0.1), elements, space group, minimum interatomic distance etc.; creates a new set and records "
    "why each structure was removed.",
    read_only=False, idempotent=True)
add(server, t.deduplicate,
    "Remove duplicates with pymatgen StructureMatcher (grouped by reduced formula first); keeps the first occurrence.",
    read_only=False, idempotent=True)
add(server, t.check_novelty,
    "Compare a set with Materials Project structures of the same chemical systems (needs MP_API_KEY or the local "
    "cache). Only MP is checked: 'novel' here means 'not in MP', not 'not in any database'.",
    read_only=False, idempotent=True, open_world=True)
add(server, t.check_benchmark_leakage,
    "Flag structures whose formula (or any ancestor's) is one of the 70 kappa_SRME test formulas. Run it on every "
    "generated set; flagged structures must not be used for fine-tuning data.",
    read_only=False, idempotent=True)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
