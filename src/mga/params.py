"""Parameter resolution: defaults with sources, schema-bounded overrides with reasons.

This is layer 1 and layer 3 of DESIGN §4.2. Every default carries a ``source``
string; an override is only accepted if it passes the schema (type + bounds,
unknown names rejected) and comes with an ``override_reason``.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mga.records import ParamProvenance
from mga.utils import short_hash


class ParamError(ValueError):
    """Raised for invalid parameter overrides; the message is shown to the agent."""


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


SpecT = TypeVar("SpecT", bound=ParamSpec)


def P(default: Any, source: str, *, affects_result: bool = True, **kwargs: Any) -> Any:
    """Declare a parameter with its default's source.

    ``affects_result=False`` marks execution-only knobs (chunk sizes, core counts)
    that are excluded from the result-cache hash.
    """
    return Field(
        default,
        json_schema_extra={"source": source, "affects_result": affects_result},
        **kwargs,
    )


def _field_meta(spec_cls: type[ParamSpec], name: str) -> dict[str, Any]:
    extra = spec_cls.model_fields[name].json_schema_extra
    return extra if isinstance(extra, dict) else {}


def resolve_params(
    spec_cls: type[SpecT],
    overrides: dict[str, Any] | None = None,
    override_reason: str | None = None,
) -> tuple[SpecT, dict[str, ParamProvenance]]:
    """Validate overrides against ``spec_cls`` and return (params, provenance)."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(spec_cls.model_fields))
    if unknown:
        raise ParamError(
            f"Unknown parameter(s) {unknown} for {spec_cls.__name__}; "
            f"allowed: {sorted(spec_cls.model_fields)}"
        )
    try:
        params = spec_cls(**overrides)
    except ValidationError as exc:
        msgs = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ParamError(f"Invalid parameter override for {spec_cls.__name__}: {msgs}") from None

    defaults = spec_cls()
    changed = {
        name for name in overrides if getattr(params, name) != getattr(defaults, name)
    }
    if changed and not (override_reason and override_reason.strip()):
        raise ParamError(
            f"Overriding {sorted(changed)} requires a non-empty override_reason "
            "(it is recorded and highlighted on the review page)."
        )

    provenance: dict[str, ParamProvenance] = {}
    dumped = params.model_dump(mode="json")
    for name in spec_cls.model_fields:
        meta = _field_meta(spec_cls, name)
        is_override = name in changed
        provenance[name] = ParamProvenance(
            value=dumped[name],
            source="override" if is_override else "default",
            ref=meta.get("source"),
            reason=override_reason if is_override else None,
        )
    return params, provenance


def result_hash(params: ParamSpec) -> str:
    """Hash of the result-affecting parameters (cache key for data/results/)."""
    spec_cls = type(params)
    dumped = params.model_dump(mode="json")
    relevant = {
        name: value
        for name, value in dumped.items()
        if _field_meta(spec_cls, name).get("affects_result", True)
    }
    return short_hash({"spec": spec_cls.__name__, "params": relevant})


def describe_spec(spec_cls: type[ParamSpec]) -> list[dict[str, Any]]:
    """Human/agent-readable table of a spec: name, default, bounds, source."""
    schema = spec_cls.model_json_schema()
    props = schema.get("properties", {})
    rows = []
    for name, field in spec_cls.model_fields.items():
        prop = props.get(name, {})
        rows.append(
            {
                "name": name,
                "default": spec_cls().model_dump(mode="json")[name],
                "description": field.description,
                "constraints": {
                    k: prop[k]
                    for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum")
                    if k in prop
                }
                or _enum_from_anyof(prop),
                "source": _field_meta(spec_cls, name).get("source"),
                "affects_result": _field_meta(spec_cls, name).get("affects_result", True),
            }
        )
    return rows


def _enum_from_anyof(prop: dict[str, Any]) -> dict[str, Any]:
    if "const" in prop:
        return {"enum": [prop["const"]]}
    options = []
    for sub in prop.get("anyOf", []):
        if "enum" in sub:
            options.extend(sub["enum"])
        elif "const" in sub:
            options.append(sub["const"])
        elif "type" in sub:
            options.append(f"<{sub['type']}>")
    return {"options": options} if options else {}
