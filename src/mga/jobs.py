"""Thin wrapper around jobflow-remote (plan §7).

jobflow-remote is imported lazily so that dry-run / replay work on a machine
without MongoDB. Every submission writes ``runs/<run>/submissions/<step_id>.json``
mapping our structure IDs to jobflow UUIDs; ``job_status`` / ``fetch_results``
start from that file.
"""

from __future__ import annotations

from typing import Any

from mga.config import paths
from mga.utils import read_json, write_json_atomic

PROJECT = "mga"


class JobsUnavailable(RuntimeError):
    pass


def _jfr():
    try:
        import jobflow_remote  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise JobsUnavailable("jobflow-remote is not installed in this environment") from exc
    return jobflow_remote


def _controller():
    jfr = _jfr()
    try:
        return jfr.JobController.from_project_name(PROJECT)
    except Exception as exc:  # noqa: BLE001 - config / Mongo errors vary
        raise JobsUnavailable(
            f"jobflow-remote project {PROJECT!r} is not usable ({type(exc).__name__}: {exc}). "
            "See docs/mcp_servers_plan.md §8 (MongoDB + ~/.jfremote/mga.yaml)."
        ) from None


def check_available() -> None:
    """Raise ``JobsUnavailable`` with a readable message if submission cannot work."""
    _controller()


def flow_jobs(flow) -> list[dict[str, Any]]:
    """Every job known at submission time, nested flows included (dynamic jobs added
    later by ``replace``, e.g. phonon displacements, are not in here)."""
    from jobflow import Flow  # noqa: PLC0415

    out = []
    for item in flow.jobs if isinstance(flow, Flow) else [flow]:
        if isinstance(item, Flow):
            out.extend(flow_jobs(item))
        else:
            meta = item.metadata or {}
            entry = {"uuid": item.uuid, "name": item.name, "sids": meta.get("mga_sids"), "kind": meta.get("mga_kind"),
                     "key": meta.get("mga_key"), "param_hash": meta.get("mga_param_hash")}
            out.append({k: v for k, v in entry.items() if v is not None})
    return out


def submit(flow, *, worker: str, resources: dict[str, Any] | None = None, exec_config: str | None = None) -> list[str]:
    jfr = _jfr()
    _controller()  # fail early with a readable message if the project is not configured
    return jfr.submit_flow(flow, worker=worker, project=PROJECT, resources=resources, exec_config=exec_config)


def submission_path(step_id: str, run_id: str):
    return paths.run_dir(run_id) / "submissions" / f"{step_id}.json"


def save_submission(step_id: str, run_id: str, doc: dict[str, Any]) -> None:
    write_json_atomic(submission_path(step_id, run_id), doc)


def load_submission(step_id: str) -> dict[str, Any]:
    matches = list(paths.runs.glob(f"*/submissions/{step_id}.json"))
    if not matches:
        raise FileNotFoundError(f"No submission recorded for step {step_id!r}")
    return read_json(matches[0])


DONE = "COMPLETED"
FINAL_BAD = ("FAILED", "REMOTE_ERROR", "STOPPED", "USER_STOPPED")


def _info(info) -> dict[str, Any]:
    return {
        "uuid": info.uuid,
        "state": info.state.value,
        "name": info.name,
        "worker": info.worker,
        "metadata": info.metadata or {},
        "error": (info.error or "")[-500:] or None,
        "run_dir": info.run_dir,
        "start_time": info.start_time.isoformat() if info.start_time else None,
        "end_time": info.end_time.isoformat() if info.end_time else None,
    }


def job_states(uuids: list[str]) -> dict[str, dict[str, Any]]:
    jc = _controller()
    return {info.uuid: _info(info) for info in jc.get_jobs_info(job_ids=[(u, 1) for u in uuids])}


def flow_states(flow_uuids: list[str]) -> dict[str, dict[str, Any]]:
    """All jobs of the flows, including the ones added at run time (``replace``)."""
    jc = _controller()
    return {info.uuid: _info(info) for info in jc.get_jobs_info(flow_ids=list(flow_uuids))}


def job_output(uuid: str) -> Any:
    jfr = _jfr()
    store = jfr.get_jobstore(project_name=PROJECT)
    store.connect()
    try:
        return store.get_output(uuid, load=True)
    finally:
        store.close()
