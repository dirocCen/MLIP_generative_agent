"""Shared MCP server plumbing: error translation and tool annotations.

mcp turns any exception other than ``ToolError`` into an opaque
"Error executing tool <name>", so every tool is wrapped here: expected errors
(bad arguments, unknown sets, missing approvals, cluster not configured) reach
the agent with their message; anything else is reported as an internal error
with its type and message, and the traceback goes to stderr.
"""

from __future__ import annotations

import functools
import sys
import traceback
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mga.analysis.ehull import MPUnavailable
from mga.analysis.kappa_metrics import KappaComparisonError
from mga.approvals import ApprovalError
from mga.config import get_mode
from mga.jobs import JobsUnavailable
from mga.params import ParamError
from mga.registry import RegistryError
from mga.results import ResultMissing
from mga.tools._common import ToolInputError

EXPECTED: tuple[type[BaseException], ...] = (
    ToolInputError, ParamError, RegistryError, ApprovalError, JobsUnavailable, MPUnavailable,
    ResultMissing, KappaComparisonError, FileNotFoundError,
)

COMMON_RULES = """\
Rules for every mga server:
- Pass named sets between tools, never long lists of structure IDs.
- decided_by: 'agent' when you chose the step or its inputs, 'human' only when the user told you to. Say why in rationale.
- Parameter overrides must stay inside the schema and need override_reason; check describe_parameters first.
- Structures from the 70 kappa_SRME test formulas (and their descendants) must never enter training data.
- Heavy computation only runs on cluster compute nodes through these tools, never on login nodes."""


def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def tool(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except EXPECTED as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            raise ToolError(
                f"internal error {type(exc).__name__}: {exc}. The step record (if any) has the traceback; "
                "tell the user instead of retrying with different arguments."
            ) from exc

    return tool


def add(
    server: MCPServer,
    fn: Callable[..., Any],
    description: str,
    *,
    read_only: bool,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = False,
    title: str | None = None,
) -> None:
    server.add_tool(
        _wrap(fn),
        name=fn.__name__,
        title=title or fn.__name__.replace("_", " "),
        description=description,
        annotations=ToolAnnotations(
            title=title or fn.__name__.replace("_", " "),
            read_only_hint=read_only,
            destructive_hint=destructive,
            idempotent_hint=idempotent,
            open_world_hint=open_world,
        ),
        structured_output=True,
    )


def make_server(name: str, instructions: str) -> MCPServer:
    return MCPServer(name, instructions=f"{instructions}\n\nCurrent MGA_MODE: {get_mode()}.\n\n{COMMON_RULES}")
