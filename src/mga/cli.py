"""``mga`` command line: human approval of VASP / fine-tuning plans, and record listing.

    mga approvals list [--status pending]
    mga approvals show <id> [--plan]
    mga approvals approve <id>      # interactive: type the id again to confirm
    mga approvals reject <id>
    mga records list [--run RUN] [--tool TOOL] [--limit N]

approve / reject need an interactive terminal and refuse to run with MGA_ACTOR=agent.
The project's .claude/settings.json additionally denies them to Claude Code's Bash tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from mga import approvals
from mga.approvals import ApprovalError


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _require_tty() -> None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SystemExit("refused: approve/reject need an interactive terminal (run it yourself, not through an agent)")
    if os.environ.get("MGA_ACTOR", "human") != "human":
        raise SystemExit(f"refused: MGA_ACTOR={os.environ['MGA_ACTOR']}; approvals are decided by a human")
    os.environ["MGA_ACTOR"] = "human"


def _show(appr: approvals.Approval, with_plan: bool) -> None:
    print(f"approval  {appr.approval_id}  [{appr.kind}]  status={appr.status}")
    print(f"requested {appr.requested_at} by {appr.requested_by_actor} (step {appr.requested_by_step})")
    print(f"plan_hash {appr.plan_hash}")
    if appr.decided_at:
        print(f"decided   {appr.decided_at}: {appr.decision_note}")
    if appr.consumed_at:
        print(f"consumed  {appr.consumed_at} by step {appr.consumed_by_step}")
    print("\nsummary:")
    print(_dump(appr.summary))
    if with_plan:
        print("\nplan:")
        print(_dump(approvals.load_plan(appr.approval_id)))
    else:
        plan = approvals.load_plan(appr.approval_id)
        print(f"\n(full plan: {len(plan.get('sids', []))} structures; `mga approvals show {appr.approval_id} --plan`)")


def _decide(approval_id: str, approve: bool) -> None:
    _require_tty()
    appr = approvals.load(approval_id)
    _show(appr, with_plan=False)
    if appr.status != "pending":
        raise SystemExit(f"\n{approval_id} is {appr.status}; nothing to decide")
    verb = "APPROVE" if approve else "REJECT"
    print(f"\nTo {verb}, type the approval id again (anything else aborts):")
    if input("> ").strip() != approval_id:
        raise SystemExit("aborted")
    note = input("note (why; recorded with the decision): ").strip()
    if not approve and not note:
        raise SystemExit("aborted: a rejection needs a reason")
    appr = approvals.decide(approval_id, approve, note or "approved without note")
    print(f"{approval_id}: {appr.status}")
    if approve:
        print(f"Tell the agent: approved {approval_id} (it may now call submit_vasp with this id, once).")


def _records(args) -> None:
    from mga.records import iter_records

    rows = [r for r in iter_records(args.run) if not args.tool or r.tool == args.tool][-args.limit:]
    for r in rows:
        who = f"{r.actor}/{r.decided_by or '-'}"
        print(f"{r.created_at}  {r.status:<16} {who:<14} {r.mode:<8} {r.tool:<26} {r.step_id}")
    if not rows:
        print("no records")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mga")
    sub = ap.add_subparsers(dest="group", required=True)

    a = sub.add_parser("approvals", help="review and decide approval requests")
    asub = a.add_subparsers(dest="cmd", required=True)
    al = asub.add_parser("list")
    al.add_argument("--status", choices=["pending", "approved", "rejected", "consumed"])
    ash = asub.add_parser("show")
    ash.add_argument("approval_id")
    ash.add_argument("--plan", action="store_true", help="print the full frozen plan")
    for name in ("approve", "reject"):
        asub.add_parser(name).add_argument("approval_id")

    r = sub.add_parser("records", help="list step records")
    rsub = r.add_subparsers(dest="cmd", required=True)
    rl = rsub.add_parser("list")
    rl.add_argument("--run")
    rl.add_argument("--tool")
    rl.add_argument("--limit", type=int, default=30)

    args = ap.parse_args(argv)
    try:
        if args.group == "records":
            _records(args)
        elif args.cmd == "list":
            items = approvals.list_approvals(args.status)
            for x in items:
                n = x.summary.get("n_structures", "?")
                print(f"{x.approval_id}  {x.status:<9} {x.kind:<8} n={n:<5} {x.summary.get('calc', '')}  requested {x.requested_at}")
            if not items:
                print("no approvals")
        elif args.cmd == "show":
            _show(approvals.load(args.approval_id), args.plan)
        else:
            _decide(args.approval_id, approve=args.cmd == "approve")
    except ApprovalError as exc:
        raise SystemExit(f"error: {exc}") from None
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\naborted") from None


if __name__ == "__main__":
    main()
