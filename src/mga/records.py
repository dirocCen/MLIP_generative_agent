"""One JSON record per tool call (DESIGN §6.2).

Records are what the review page and the agent evaluation read. They separate
*who invoked* a step (``actor``) from *who chose* its inputs (``decided_by``),
so reports can state honestly which steps were agent decisions (DESIGN §11.4).
"""

from __future__ import annotations

import platform
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mga import __version__
from mga.config import get_actor, get_mode, get_run_id, paths, utc_now
from mga.utils import read_json, write_json_atomic

Status = Literal["running", "ok", "dry_run", "submitted", "replayed", "pending_approval", "error"]
DecidedBy = Literal["agent", "human"]

_TRACKED_PACKAGES = (
    "pymatgen", "ase", "jobflow", "jobflow-remote", "atomate2", "phonopy", "phono3py",
    "mp-api", "mcp", "pydantic",
)


class ParamProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    source: Literal["default", "override"]
    ref: str | None = None  # where the default comes from
    reason: str | None = None  # why it was overridden


class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    run_id: str
    tool: str
    actor: str
    mode: str
    decided_by: DecidedBy | None = None
    rationale: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    params: dict[str, ParamProvenance] = Field(default_factory=dict)
    code_versions: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    status: Status = "running"
    error: str | None = None
    wall_time_s: float | None = None
    host: str = Field(default_factory=platform.node)
    created_at: str = Field(default_factory=utc_now)


def code_versions() -> dict[str, str]:
    versions = {"mga": __version__, "python": platform.python_version()}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return versions


def new_step_id(tool: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{tool}-{uuid.uuid4().hex[:6]}"


def record_path(step_id: str, run_id: str) -> Path:
    return paths.run_dir(run_id) / "records" / f"{step_id}.json"


def save_record(rec: StepRecord) -> Path:
    path = record_path(rec.step_id, rec.run_id)
    write_json_atomic(path, rec.model_dump(mode="json"))
    return path


@contextmanager
def recorded_step(
    tool: str,
    *,
    inputs: dict[str, Any] | None = None,
    params: dict[str, ParamProvenance] | None = None,
    parent_ids: list[str] | None = None,
    decided_by: DecidedBy | None = None,
    rationale: str | None = None,
) -> Iterator[StepRecord]:
    """Create a record, time the block, and always persist it (also on failure).

    The block sets ``rec.outputs`` and ``rec.status``; if it leaves the status at
    ``running`` the step is marked ``ok``. Exceptions mark it ``error`` and re-raise.
    """
    rec = StepRecord(
        step_id=new_step_id(tool),
        run_id=get_run_id(),
        tool=tool,
        actor=get_actor(),
        mode=get_mode(),
        decided_by=decided_by,
        rationale=rationale,
        parent_ids=list(parent_ids or []),
        params=dict(params or {}),
        code_versions=code_versions(),
        inputs=dict(inputs or {}),
    )
    t0 = time.perf_counter()
    try:
        yield rec
    except BaseException as exc:
        rec.status = "error"
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.outputs.setdefault("traceback", traceback.format_exc(limit=8))
        raise
    else:
        if rec.status == "running":
            rec.status = "ok"
    finally:
        rec.wall_time_s = round(time.perf_counter() - t0, 3)
        save_record(rec)


def load_record(step_id: str, run_id: str | None = None) -> StepRecord:
    if run_id is not None:
        return StepRecord.model_validate(read_json(record_path(step_id, run_id)))
    matches = list(paths.runs.glob(f"*/records/{step_id}.json"))
    if not matches:
        raise FileNotFoundError(f"No record with step_id {step_id!r}")
    return StepRecord.model_validate(read_json(matches[0]))


def iter_records(run_id: str | None = None) -> Iterator[StepRecord]:
    pattern = f"{run_id}/records/*.json" if run_id else "*/records/*.json"
    for path in sorted(paths.runs.glob(pattern)):
        yield StepRecord.model_validate(read_json(path))
