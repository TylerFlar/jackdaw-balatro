"""Engine-agnostic replay of the recorded engine action tapes.

A :class:`ParitySpec` describes one trajectory declaratively: how to start
the run, the setup operations applied before play, and the recorded action
tape (``tests/fixtures/parity_specs.json``, recorded from the Python engine's
golden trajectories before it was retired).  Any engine that implements
:class:`Engine` can replay it; :class:`RustEngine` is the Rust engine's.

Setup operations (all fields are Lua-native names):

* ``{"op": "force_boss", "key": "bl_psychic"}`` - the ante's boss blind is
  ``key`` and the next blind to play is the boss.
* ``{"op": "action", "action": ["SelectBlind"]}`` - apply one action.
* ``{"op": "add", "key": "j_joker" | "H_A" | "c_fool", "seal": "Red",
  "enhancement": "m_steel", "edition": "foil"}`` - BalatroBot ``add``
  semantics (``SMODS.add_card``): edition poll, ``add_to_deck``, and playing
  cards join the hand in SELECTING_HAND.
* ``{"op": "set", "dollars": 60, "hands_left": 1, "chips": 123, "ante": 5}`` -
  write state directly (BalatroBot ``set`` semantics; ``hands_left`` is the
  current round's hands, ``ante`` is ``round_resets.ante``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jackdaw.engine.data import load
from tests.parity import canon

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SPECS_PATH = FIXTURES / "parity_specs.json"


@dataclass
class ParitySpec:
    name: str
    start: dict[str, Any]
    setup: list[dict[str, Any]]
    tape: list[list[Any]]
    max_steps: int
    check: str | None = None
    check_after: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "setup": self.setup,
            "tape": self.tape,
            "max_steps": self.max_steps,
            "check": self.check,
            "check_after": self.check_after,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ParitySpec:
        return cls(**d)


def load_specs(path: Path = SPECS_PATH) -> list[ParitySpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ParitySpec.from_json(d) for d in data["specs"]]


class Engine(Protocol):
    """What the replay needs from an engine."""

    def snapshot(self) -> dict[str, Any]: ...

    def legal_actions(self) -> list[list[Any]]: ...

    def step(self, action: list[Any]) -> None: ...  # raises on an illegal action


class RustEngine:
    """The Rust engine (:class:`jackdaw.core.Game`) behind the :class:`Engine` interface: a new
    run with the spec's setup ops applied natively (``Game.apply_setup``)."""

    def __init__(self, start: dict[str, Any], setup: list[dict[str, Any]] = ()) -> None:
        from jackdaw.core import Game

        self.game = Game.new(start["back"], start["stake"], start["seed"], start.get("challenge"))
        for op in setup:
            self.game.apply_setup(op)

    def snapshot(self) -> dict[str, Any]:
        return self.game.to_dict()

    def legal_actions(self) -> list[list[Any]]:
        return json.loads(self.game._g.legal_actions())

    def step(self, action: list[Any]) -> None:
        self.game.step(action)


# ---------------------------------------------------------------------------
# Scenario checks, evaluated on snapshots (engine-agnostic)
# ---------------------------------------------------------------------------


def _jokers(snap: dict[str, Any]) -> list[dict[str, Any]]:
    return snap["areas"]["jokers"]


def _score_positive(snap: dict[str, Any]) -> bool:
    last = snap["sim"]["last_score"]
    return last is not None and last["total"] > 0


def _bones_saved(snap: dict[str, Any]) -> bool:
    return snap["phase"] != "game_over" and not any(
        j["center"] == "j_mr_bones" for j in _jokers(snap)
    )


def _debuffed_and_saved(snap: dict[str, Any]) -> bool:
    last = snap["sim"]["last_score"]
    return last is not None and last["debuffed"] and _bones_saved(snap)


def _six_destroyed(snap: dict[str, Any]) -> bool:
    centers = load("centers")
    return any(centers[c["center"]]["set"] == "Spectral" for c in snap["areas"]["consumables"])


CHECKS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "score_positive": _score_positive,
    "bones_saved": _bones_saved,
    "debuffed_and_saved": _debuffed_and_saved,
    "six_destroyed": _six_destroyed,
    "two_jokers": lambda snap: len(_jokers(snap)) >= 2,
}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 10


@dataclass
class Replay:
    digest: str = ""
    records: int = 0
    checkpoints: list[str] = field(default_factory=list)
    actions_taken: int = 0
    checks: dict[str, bool] = field(default_factory=dict)
    record_log: list[tuple[str, Any]] = field(default_factory=list)
    keep_log: bool = False

    def push(self, kind: str, payload: Any) -> None:
        self.digest = canon.chain(self.digest, canon.digest([kind, payload]))
        self.records += 1
        if self.records % CHECKPOINT_EVERY == 0:
            self.checkpoints.append(self.digest[:16])
        if self.keep_log:
            self.record_log.append((kind, payload))

    def as_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "records": self.records,
            "actions": self.actions_taken,
            "checkpoints": self.checkpoints,
            "checks": self.checks,
        }


def replay(
    spec: ParitySpec,
    engine: Engine,
    *,
    keep_log: bool = False,
    on_state: Callable[[int, Engine, dict[str, Any]], None] | None = None,
) -> Replay:
    """Replay ``spec`` on an engine that has already run the spec's setup."""
    res = Replay(keep_log=keep_log)
    res.push("setup", engine.snapshot())
    checked = spec.check is None
    pos = 0
    for _ in range(spec.max_steps):
        snap = engine.snapshot()
        if not checked and res.actions_taken >= spec.check_after:
            res.checks["scenario"] = CHECKS[spec.check](snap)  # type: ignore[index]
            checked = True
        legal = engine.legal_actions()
        res.push("state", {"snapshot": snap, "legal": legal})
        if on_state is not None:
            on_state(pos, engine, snap)
        if snap["phase"] == "game_over" or snap["run"]["won"] or not legal:
            break
        if pos >= len(spec.tape):
            res.push("tape-exhausted", None)
            break
        action = spec.tape[pos]
        pos += 1
        try:
            engine.step(action)
        except Exception:  # behaviour includes refusing an action
            res.push("raised", action)
            break
        res.actions_taken += 1
        res.push("action", action)
    final = engine.snapshot()
    if not checked:
        res.checks["scenario"] = CHECKS[spec.check](final)  # type: ignore[index]
    res.push("final", final)
    return res
