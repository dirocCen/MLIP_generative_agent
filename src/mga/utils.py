"""Small helpers shared by local and worker code (no heavy imports)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def short_hash(obj: Any, n: int = 12) -> str:
    data = obj if isinstance(obj, (bytes, bytearray)) else canonical_json(obj).encode()
    return hashlib.sha256(data).hexdigest()[:n]


def _json_default(obj: Any) -> Any:
    # numpy scalars / arrays and Paths show up in outputs; keep this import-free
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON via a temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def check_name(name: str, what: str = "name") -> str:
    if not NAME_RE.match(name):
        raise ValueError(
            f"Invalid {what} {name!r}: use 1-100 chars of letters, digits, '_', '-', '.'"
        )
    return name
