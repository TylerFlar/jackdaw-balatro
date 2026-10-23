"""The Python oracle behind the :class:`tests.parity.harness.Engine` interface."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from jackdaw.bridge.backend import SimBackend
from jackdaw.engine import card as card_module
from jackdaw.engine.actions import GamePhase, get_legal_actions
from jackdaw.engine.challenges import get_challenge
from jackdaw.engine.game import step as engine_step
from jackdaw.engine.run_init import initialize_run
from tests.golden.trajectories import decode_action, encode_action
from tests.parity.py_export import export_state
from tests.parity.py_import import import_state


class PyEngine:
    """One Python oracle run.

    The oracle's ``sort_id`` counter is process-global; each instance keeps
    its own copy and swaps it in around every call, so several runs can be
    interleaved (the round-trip test steps an original and a rebuilt state
    side by side).
    """

    def __init__(self, start: dict[str, Any] | None, setup: list[dict[str, Any]] = ()) -> None:
        self._counter = 0
        self.gs: dict[str, Any] = {}
        if start is None:
            return
        with self._ids():
            challenge = get_challenge(start["challenge"]) if start.get("challenge") else None
            self.gs = initialize_run(
                start["back"], start["stake"], start["seed"], challenge=challenge
            )
            self.gs["phase"] = GamePhase.BLIND_SELECT
            self.gs["blind_on_deck"] = "Small"
        for op in setup:
            self.apply(op)

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> PyEngine:
        eng = cls(None)
        saved = card_module._sort_id_counter
        try:
            eng.gs = import_state(snap)  # sets the global counter from the snapshot
            eng._counter = card_module._sort_id_counter
        finally:
            card_module._sort_id_counter = saved
        return eng

    @contextmanager
    def _ids(self) -> Iterator[None]:
        saved = card_module._sort_id_counter
        card_module._sort_id_counter = self._counter
        try:
            yield
        finally:
            self._counter = card_module._sort_id_counter
            card_module._sort_id_counter = saved

    # -- setup ops ----------------------------------------------------------

    def apply(self, op: dict[str, Any]) -> None:
        kind = op["op"]
        gs = self.gs
        if kind == "force_boss":
            gs["round_resets"]["blind_choices"]["Boss"] = op["key"]
            gs["blind_on_deck"] = "Boss"
        elif kind == "action":
            self.step(op["action"])
        elif kind == "add":
            params = {k: op[k] for k in ("seal", "enhancement", "edition") if k in op}
            backend = SimBackend()
            backend._gs = gs
            with self._ids():
                backend.handle("add", {"key": op["key"], **params})
        elif kind == "set":
            for field, value in op.items():
                if field == "op":
                    continue
                if field == "dollars":
                    gs["dollars"] = value
                elif field == "chips":
                    gs["chips"] = value
                elif field == "hands_left":
                    gs["current_round"]["hands_left"] = value
                    gs["hands_left"] = value
                else:
                    raise ValueError(f"unknown set field {field!r}")
        else:
            raise ValueError(f"unknown setup op {kind!r}")

    # -- Engine protocol ----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._ids():
            return export_state(self.gs)

    def legal_actions(self) -> list[list[Any]]:
        with self._ids():
            return [encode_action(a) for a in get_legal_actions(self.gs)]

    def step(self, action: list[Any]) -> None:
        with self._ids():
            engine_step(self.gs, decode_action(action))
