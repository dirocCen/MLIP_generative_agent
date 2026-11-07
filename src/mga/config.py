"""Paths, run mode and actor, all read from the environment at call time.

Environment variables
---------------------
MGA_PROJECT_ROOT  project directory (default: inferred from this file's location)
MGA_MODE          dry_run (default) | live | replay
MGA_ACTOR         agent | human | script; who is invoking the code (recorded, never inferred)
MGA_RUN_ID        groups records of one session (default: adhoc-<UTC date>)
MGA_FC_CACHE      worker-side root of the L1 force-constant cache (overrides configs/cluster.yaml)
MGA_LADDER_WORK   worker-side root of the L3 (FC4 / FourPhonon) work directories (same)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Mode = Literal["dry_run", "live", "replay"]
Actor = Literal["agent", "human", "script", "unspecified"]

MODES: tuple[str, ...] = ("dry_run", "live", "replay")
ACTORS: tuple[str, ...] = ("agent", "human", "script")


def project_root() -> Path:
    env = os.environ.get("MGA_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/mga/config.py -> project root is two levels above the package dir
    return Path(__file__).resolve().parents[2]


def get_mode() -> Mode:
    mode = os.environ.get("MGA_MODE", "dry_run")
    if mode not in MODES:
        raise ValueError(f"MGA_MODE must be one of {MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def get_actor() -> Actor:
    actor = os.environ.get("MGA_ACTOR")
    if actor is None:
        return "unspecified"
    if actor not in ACTORS:
        raise ValueError(f"MGA_ACTOR must be one of {ACTORS}, got {actor!r}")
    return actor  # type: ignore[return-value]


def get_run_id() -> str:
    run_id = os.environ.get("MGA_RUN_ID")
    if run_id:
        return run_id
    return "adhoc-" + datetime.now(timezone.utc).strftime("%Y%m%d")


def worker_storage(worker: str) -> dict[str, str | None]:
    """Directories on the worker's filesystem that outlive one job (paths are used there, not here).

    ``fc_cache``: L1 writes compact FC2/FC3 that L2 reads back; ``ladder_work``: L3 inputs and
    FourPhonon outputs. The cluster is the worker name's prefix (``datalab_dpa`` -> ``datalab``);
    environment variables win (tests, other machines). ``None`` = not configured."""
    import yaml

    cluster = worker.split("_")[0]
    try:
        with open(paths.configs / "cluster.yaml") as fh:
            conf = (yaml.safe_load(fh) or {}).get(cluster) or {}
    except FileNotFoundError:
        conf = {}
    out = {}
    for key, env in (("fc_cache", "MGA_FC_CACHE"), ("ladder_work", "MGA_LADDER_WORK")):
        val = os.environ.get(env) or conf.get(key)
        out[key] = None if val in (None, "", "TODO") else str(val)
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Paths:
    """Project directory layout (DESIGN §9). Resolved lazily so tests can relocate it."""

    @property
    def root(self) -> Path:
        return project_root()

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def structures(self) -> Path:
        return self.data / "structures"

    @property
    def sets(self) -> Path:
        return self.data / "sets"

    @property
    def results(self) -> Path:
        return self.data / "results"

    @property
    def reference(self) -> Path:
        return self.data / "reference"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def approvals(self) -> Path:
        return self.root / "approvals"

    def run_dir(self, run_id: str | None = None) -> Path:
        return self.runs / (run_id or get_run_id())


paths = Paths()
