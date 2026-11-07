"""Helpers shared by the tool modules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mga.config import get_run_id, paths
from mga.registry import load_set
from mga.utils import write_json_atomic

N_EXAMPLES = 5


class ToolInputError(ValueError):
    """Bad tool arguments; the message is meant for the agent."""


def project_path(path: str) -> Path:
    """Resolve ``path`` (absolute or relative to the project root); refuse anything outside it."""
    root = paths.root
    p = Path(path).expanduser()
    p = (p if p.is_absolute() else root / p).resolve()
    if root != p and root not in p.parents:
        raise ToolInputError(f"{path!r} is outside the project directory {root}")
    if not p.exists():
        raise ToolInputError(f"{path!r} does not exist")
    return p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(paths.root))
    except ValueError:
        return str(path)


def parent_steps(*set_names: str) -> list[str]:
    """Steps that created the given sets (record ``parent_ids``)."""
    out = []
    for name in set_names:
        step = load_set(name).get("created_by_step")
        if step and step not in out:
            out.append(step)
    return out


def write_details(step_id: str, doc: Any) -> str:
    """Per-structure detail tables go to a file, not into the LLM context."""
    path = paths.run_dir(get_run_id()) / "outputs" / f"{step_id}.json"
    write_json_atomic(path, doc)
    return rel(path)


def top(values: Iterable[Any], n: int = 15) -> dict[str, int]:
    counts = Counter(str(v) for v in values)
    out = dict(counts.most_common(n))
    rest = sum(counts.values()) - sum(out.values())
    if rest:
        out["(other)"] = rest
    return out


def require_decision(decided_by: str, rationale: str) -> None:
    if decided_by not in ("agent", "human"):
        raise ToolInputError("decided_by must be 'agent' (you chose) or 'human' (the user told you)")
    if not rationale or not rationale.strip():
        raise ToolInputError("rationale is required: say why this step and these inputs")
