"""Human approval gate for expensive submissions (VASP, fine-tuning; DESIGN §11.5).

Life cycle: ``pending`` -> ``approved`` | ``rejected`` -> ``consumed``.

- ``request`` freezes the full plan (structure IDs + parameters + resources) in
  ``approvals/<id>/plan.json`` and stores its sha256 in ``approval.json``.
- ``approve`` / ``reject`` are only reachable from the ``mga approvals`` CLI, which
  requires an interactive terminal. There is deliberately no MCP tool for them,
  and they refuse to run when ``MGA_ACTOR=agent``.
- ``consume`` re-hashes ``plan.json``; a plan edited after approval is refused.
  An approval can be consumed once.

This is a guard rail against accidental submission, not a security boundary: a
process with shell access to this machine can always edit these files.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mga.config import get_actor, paths, utc_now
from mga.utils import canonical_json, read_json, write_json_atomic

Kind = Literal["vasp", "finetune"]
ApprovalStatus = Literal["pending", "approved", "rejected", "consumed"]


class ApprovalError(RuntimeError):
    pass


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    kind: Kind
    status: ApprovalStatus = "pending"
    plan_hash: str
    summary: dict[str, Any]
    requested_by_step: str | None = None
    requested_by_actor: str
    requested_at: str
    decided_at: str | None = None
    decision_note: str | None = None
    consumed_at: str | None = None
    consumed_by_step: str | None = None


def plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(plan).encode()).hexdigest()


def _dir(approval_id: str):
    if not approval_id.replace("-", "").isalnum():
        raise ApprovalError(f"Invalid approval id {approval_id!r}")
    return paths.approvals / approval_id


def _save(appr: Approval) -> None:
    write_json_atomic(_dir(appr.approval_id) / "approval.json", appr.model_dump(mode="json"))


def request(kind: Kind, plan: dict[str, Any], summary: dict[str, Any], step_id: str | None) -> Approval:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    approval_id = f"{kind}-{stamp}-{secrets.token_hex(3)}"
    appr = Approval(
        approval_id=approval_id,
        kind=kind,
        plan_hash=plan_hash(plan),
        summary=summary,
        requested_by_step=step_id,
        requested_by_actor=get_actor(),
        requested_at=utc_now(),
    )
    write_json_atomic(_dir(approval_id) / "plan.json", plan)
    _save(appr)
    return appr


def load(approval_id: str) -> Approval:
    path = _dir(approval_id) / "approval.json"
    if not path.exists():
        raise ApprovalError(f"Unknown approval {approval_id!r}")
    return Approval.model_validate(read_json(path))


def load_plan(approval_id: str) -> dict[str, Any]:
    return read_json(_dir(approval_id) / "plan.json")


def list_approvals(status: ApprovalStatus | None = None) -> list[Approval]:
    out = []
    for path in sorted(paths.approvals.glob("*/approval.json")):
        appr = Approval.model_validate(read_json(path))
        if status is None or appr.status == status:
            out.append(appr)
    return out


def _require_human() -> None:
    if get_actor() == "agent":
        raise ApprovalError("Approvals can only be decided by a human via `mga approvals` in a terminal")


def decide(approval_id: str, approve: bool, note: str) -> Approval:
    _require_human()
    appr = load(approval_id)
    if appr.status != "pending":
        raise ApprovalError(f"Approval {approval_id} is {appr.status}, not pending")
    if plan_hash(load_plan(approval_id)) != appr.plan_hash:
        raise ApprovalError(f"plan.json of {approval_id} changed after the request; reject it and re-request")
    appr.status = "approved" if approve else "rejected"
    appr.decided_at = utc_now()
    appr.decision_note = note
    _save(appr)
    return appr


def consume(approval_id: str, kind: Kind, step_id: str) -> dict[str, Any]:
    """Check an approval and mark it used; returns the frozen plan."""
    appr = load(approval_id)
    if appr.kind != kind:
        raise ApprovalError(f"Approval {approval_id} is for {appr.kind}, not {kind}")
    if appr.status != "approved":
        raise ApprovalError(
            f"Approval {approval_id} is {appr.status}. A human must run "
            f"`mga approvals approve {approval_id}` in their own terminal first."
        )
    plan = load_plan(approval_id)
    if plan_hash(plan) != appr.plan_hash:
        raise ApprovalError(f"plan.json of {approval_id} changed after approval; refusing to submit")
    appr.status = "consumed"
    appr.consumed_at = utc_now()
    appr.consumed_by_step = step_id
    _save(appr)
    return plan
