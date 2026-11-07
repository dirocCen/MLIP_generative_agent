"""The four MCP servers over the real protocol (in-process ``mcp.Client``)."""

from __future__ import annotations

import asyncio
import importlib

import pytest
from conftest import si_primitive, write_json_structures

from mga import approvals

EXPECTED_TOOLS = {
    "structures": {"import_structures", "structure_summary", "list_sets", "merge_sets", "symmetrize_structures", "filter_structures", "deduplicate",
                   "check_novelty", "check_benchmark_leakage"},
    "mlip": {"list_models", "describe_parameters", "submit_mattergen", "submit_mlip_relax", "submit_mlip_phonon_kappa",
             "submit_mlip_convergence", "submit_mlip_fc2_descriptors", "submit_mlip_kappa_lbte",
             "submit_mlip_fc4_fourphonon"},
    "dft": {"preview_vasp_inputs", "estimate_vasp_cost", "request_vasp_submission", "submit_vasp"},
    "analysis": {"job_status", "fetch_results", "compute_ehull", "kappa_srme", "model_disagreement",
                 "kappa_convergence", "query_records", "budget_ledger", "escalation_baseline"},
}

A = {"decided_by": "agent", "rationale": "test"}


def _server(name):
    return importlib.import_module(f"mga.mcp_servers.{name}_server").server


def _session(name, calls):
    """Open one client session and run ``calls(client)``; returns its result."""
    from mcp import Client

    async def run():
        async with Client(_server(name)) as client:
            return await calls(client)

    return asyncio.run(run())


def _text(result):
    return " ".join(getattr(c, "text", "") for c in result.content)


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_tool_lists(project, name):
    async def calls(c):
        return (await c.list_tools()).tools

    tools = _session(name, calls)
    assert {t.name for t in tools} == EXPECTED_TOOLS[name]
    for t in tools:
        assert t.description and t.annotations is not None and t.input_schema["type"] == "object"
        assert not any(w in t.name for w in ("approve", "decide", "reject"))  # approval is human-only (CLI)
        if t.name.startswith(("submit_", "request_", "import_")):
            assert t.annotations.read_only_hint is False


def test_errors_reach_the_agent_with_their_message(project):
    async def calls(c):
        missing = await c.call_tool("structure_summary", {"set_name": "no_such_set"})
        bad_arg = await c.call_tool("filter_structures", {"set_name": "x", "out_set": "y", **A})
        return missing, bad_arg

    missing, bad_arg = _session("structures", calls)
    assert missing.is_error and "RegistryError" in _text(missing) and "no_such_set" in _text(missing)
    assert bad_arg.is_error and "internal error" not in _text(bad_arg)


def test_structured_results_round_trip(project):
    path = write_json_structures(project / "inputs" / "si.json", [si_primitive()])

    async def structures_calls(c):
        imp = await c.call_tool("import_structures", {"path": str(path), "set_name": "si", "origin_kind": "file", **A})
        summ = await c.call_tool("structure_summary", {"set_name": "si"})
        return imp, summ

    imp, summ = _session("structures", structures_calls)
    assert not imp.is_error and imp.structured_content["n_in_set"] == 1
    assert summ.structured_content["reduced_formulas"] == {"Si": 1}

    async def mlip_calls(c):
        models = await c.call_tool("list_models", {})
        dry = await c.call_tool("submit_mlip_relax", {"set_name": "si", "model": "emt_test", **A})
        params = await c.call_tool("describe_parameters", {"calc": "mlip_kappa"})
        wrong = await c.call_tool("describe_parameters", {"kind": "mlip_kappa"})
        return models, dry, params, wrong

    models, dry, params, wrong = _session("mlip", mlip_calls)
    assert "emt_test" in {m["name"] for m in models.structured_content["models"]}
    assert dry.structured_content["mode"] == "dry_run"
    assert not params.is_error, _text(params)
    kp = {d["name"]: d for d in params.structured_content["parameters"]}
    assert kp["displacement"]["default"] == 0.03 and kp["transport"]["default"] == "MS-SMM19"
    assert all(d["source"] for d in kp.values())  # every default says where it comes from
    assert wrong.is_error and "calc" in _text(wrong) and "Field required" in _text(wrong)  # schema errors are readable


def test_vasp_request_over_mcp_stops_for_a_human(project):
    path = write_json_structures(project / "inputs" / "si.json", [si_primitive()])

    async def imp(c):
        return await c.call_tool("import_structures", {"path": str(path), "set_name": "si", "origin_kind": "file", **A})

    _session("structures", imp)

    async def dft_calls(c):
        req = await c.call_tool("request_vasp_submission", {"set_name": "si", "calc": "relax_static", **A})
        sub = await c.call_tool("submit_vasp", {"approval_id": req.structured_content["approval_id"]})
        return req, sub

    req, sub = _session("dft", dft_calls)
    aid = req.structured_content["approval_id"]
    assert req.structured_content["status"] == "pending_approval" and "STOP" in req.structured_content
    assert sub.is_error and approvals.load(aid).status == "pending"
