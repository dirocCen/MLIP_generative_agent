"""MLIP registry loaded from ``configs/models.yaml`` (DESIGN §1.4, §8.1)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from mga.config import paths

Loader = Literal["mace_mp", "mace", "deepmd", "nequip_compiled", "upet", "tace", "calorine_nep", "emt"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    loader: Loader
    checkpoint: str | None
    head: str | None = None
    # what dtype="model_default" in the MLIP params resolves to (ignored by loaders whose dtype is
    # fixed by the model file: deepmd, tace, nequip_compiled, calorine_nep)
    dtype: Literal["float64", "float32"] = "float64"
    energy_reference: Literal["mp_pbe", "other"]
    committee: bool
    worker: str
    training_data: str
    ksrme_reported: float | None
    role: str
    todo: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """False while the checkpoint is still a placeholder."""
        return self.loader == "emt" or (bool(self.checkpoint) and self.checkpoint != "TODO")

    def worker_dict(self) -> dict[str, Any]:
        """What the worker needs to build the calculator (no local-only fields)."""
        return {"name": self.name, "loader": self.loader, "checkpoint": self.checkpoint, "head": self.head,
                "dtype": self.dtype}


@lru_cache(maxsize=1)
def _load(path_str: str) -> dict[str, ModelSpec]:
    with open(path_str) as fh:
        raw = yaml.safe_load(fh)["models"]
    return {name: ModelSpec(name=name, **{**cfg, "todo": tuple(cfg.get("todo") or ())}) for name, cfg in raw.items()}


def all_models() -> dict[str, ModelSpec]:
    return _load(str(paths.configs / "models.yaml"))


def get_model(name: str) -> ModelSpec:
    models = all_models()
    if name not in models:
        raise KeyError(f"Unknown model {name!r}; configured: {sorted(models)}")
    return models[name]


def committee() -> list[ModelSpec]:
    return [m for m in all_models().values() if m.committee]
