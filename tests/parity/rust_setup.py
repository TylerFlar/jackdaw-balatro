"""Parity setup ops applied natively in Rust (``Game.apply_setup``,
``crates/jackdaw-core/src/debug.rs``), against the oracle's (``PyEngine.apply``)."""

from __future__ import annotations

import json
from typing import Any

from jackdaw import _core
from tests.parity import canon
from tests.parity.pyengine import PyEngine
from tests.parity.rust_step import plain


def rust_setup(start: dict[str, Any], setup: list[dict[str, Any]]) -> Any:
    """A new Rust run with ``setup`` applied op by op (``_core.Game``)."""
    game = _core.Game.new(start["back"], start["stake"], start["seed"], start.get("challenge"))
    for op in setup:
        game.apply_setup(json.dumps(op))
    return game


def rust_snapshot(game: Any) -> dict[str, Any]:
    return canon.from_json_compatible(json.loads(game.snapshot()))


def setup_diff(start: dict[str, Any], setup: list[dict[str, Any]]) -> list[str]:
    """Where the two engines' states differ after the setup (empty when equal)."""
    py = PyEngine(start, setup)
    expected = plain(py.snapshot())
    game = rust_setup(start, setup)
    if game.state_hash() == canon.digest(expected):
        return []
    return canon.diff(expected, rust_snapshot(game)) or ["digests differ, no field diff"]
