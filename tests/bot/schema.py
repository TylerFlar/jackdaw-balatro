"""BalatroBot's gamestate schema (the vendored ``openrpc.json``), with jackdaw's
patches for where the schema disagrees with what BalatroBot v1.5.2 sends.

A small validator for the JSON Schema keywords ``openrpc.json`` uses (``$ref``,
``type``, ``const``, ``oneOf``, ``properties``, ``required``,
``additionalProperties``, ``items``), so the check needs no extra dependency.

Patches (each is a bug in the published schema, not in the view):

* ``State`` lacks ``SMODS_BOOSTER_OPENED`` (Steamodded's pack state, the state
  of every open pack) and ``UNKNOWN`` (``gamestate.lua`` names a state it can't
  find that way, e.g. between ``load``'s run deletion and a new run).
* An empty Lua table is encoded as ``[]`` (rxi ``json.lua``), so every
  ``"type": "object"`` also accepts an empty array: ``used_vouchers`` before a
  voucher, a card's ``modifier`` and ``state`` with nothing set.
* ``modifier.enhancement`` is ``ability.effect`` upper-cased for every card
  (``gamestate.lua``), so jokers and consumables report e.g. ``"MULT"``,
  ``"TYPE MULT"``, ``"ENHANCE"``: any string, not only ``Enhancement``.
* ``CardSet`` lacks ``EDITION`` (``ability.set`` upper-cased; the Edition
  centers are cards too).
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OPENRPC = ROOT / "third_party" / "balatrobot" / "v1.5.2" / "src" / "lua" / "utils" / "openrpc.json"


@cache
def openrpc() -> dict[str, Any]:
    return json.loads(OPENRPC.read_text(encoding="utf-8"))


@cache
def schemas() -> dict[str, Any]:
    """``components.schemas`` with jackdaw's patches applied."""
    s = copy.deepcopy(openrpc()["components"]["schemas"])
    s["State"]["oneOf"] += [{"const": "SMODS_BOOSTER_OPENED"}, {"const": "UNKNOWN"}]
    s["CardSet"]["oneOf"] += [{"const": "EDITION"}]
    s["CardModifier"]["properties"]["enhancement"] = {"type": "string"}
    return s


class SchemaError(AssertionError):
    pass


def _resolve(ref: str) -> dict[str, Any]:
    prefix = "#/components/schemas/"
    assert ref.startswith(prefix), ref
    return schemas()[ref[len(prefix) :]]


def _type_ok(value: Any, ty: str) -> bool:
    if ty == "object":
        return isinstance(value, dict) or value == []  # rxi: an empty table is []
    if ty == "array":
        return isinstance(value, list)
    if ty == "string":
        return isinstance(value, str)
    if ty == "boolean":
        return isinstance(value, bool)
    if ty == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if ty == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ty == "null":
        return value is None
    raise ValueError(f"unsupported type {ty!r}")


_ENUMS: dict[int, frozenset[Any] | None] = {}


def _enum(schema: dict[str, Any]) -> frozenset[Any] | None:
    """The values of a ``oneOf`` made only of ``const`` options (an enum), else None."""
    key = id(schema)
    if key not in _ENUMS:
        options = schema["oneOf"]
        plain = all(set(o) <= {"const", "description"} and "const" in o for o in options)
        _ENUMS[key] = frozenset(o["const"] for o in options) if plain else None
    return _ENUMS[key]


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Raise :class:`SchemaError` where ``value`` doesn't fit ``schema``."""
    if "$ref" in schema:
        validate(value, _resolve(schema["$ref"]), path)
    if "const" in schema and value != schema["const"]:
        raise SchemaError(f"{path}: {value!r} is not {schema['const']!r}")
    enum = _enum(schema) if "oneOf" in schema else None
    if enum is not None:
        if isinstance(value, (dict, list)) or value not in enum:
            raise SchemaError(f"{path}: {value!r} is not one of {sorted(enum)}")
    elif "oneOf" in schema:
        ok = 0
        for option in schema["oneOf"]:
            try:
                validate(value, option, path)
                ok += 1
            except SchemaError:
                pass
        if ok != 1:
            raise SchemaError(f"{path}: {value!r} matches {ok} of the oneOf options")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(value, t) for t in types):
            raise SchemaError(f"{path}: {value!r} is not of type {schema['type']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: missing required {key!r}")
        for key, item in value.items():
            if key in props:
                validate(item, props[key], f"{path}.{key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                validate(item, schema["additionalProperties"], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise SchemaError(f"{path}: unexpected {key!r}")
    elif value == [] and schema.get("required"):
        raise SchemaError(f"{path}: an empty table has no {schema['required'][0]!r}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")


def validate_gamestate(view: dict[str, Any]) -> None:
    validate(view, schemas()["GameState"])
