"""A run in the Rust engine, with Python actions and snapshots."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from typing import Any

from jackdaw import _core
from jackdaw.engine import actions as _actions
from jackdaw.engine.actions import GamePhase

IllegalActionError = _core.IllegalActionError
NotPortedError = _core.NotPortedError

_ACTION_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in vars(_actions).values()
    if isinstance(cls, type)
    and dataclasses.is_dataclass(cls)
    and cls.__module__ == _actions.__name__
}


def encode_action(action: Any) -> list[Any]:
    """An action's tape form: ``["PlayHand", [0, 2, 3]]``. A list passes through."""
    if isinstance(action, list):
        return action
    out: list[Any] = [type(action).__name__]
    for f in dataclasses.fields(action):
        v = getattr(action, f.name)
        out.append(list(v) if isinstance(v, tuple) else v)
    return out


def decode_action(entry: list[Any]) -> Any:
    """The action dataclass for a tape entry."""
    cls = _ACTION_TYPES[entry[0]]
    return cls(*(tuple(v) if isinstance(v, list) else v for v in entry[1:]))


def random_seed(rng: random.Random | None = None) -> str:
    """A new seed the way vanilla's ``random_string(8)`` makes one
    (misc_functions.lua:270-277): digits 1-9, and letters without ``O``."""
    r = rng or random.SystemRandom()
    out = []
    for _ in range(8):
        if r.random() > 0.7:
            out.append(chr(r.randint(ord("1"), ord("9"))))
        elif r.random() > 0.45:
            out.append(chr(r.randint(ord("A"), ord("N"))))
        else:
            out.append(chr(r.randint(ord("P"), ord("Z"))))
    return "".join(out)


def _to_json(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"$f": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    return obj


def _from_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        if len(obj) == 1 and "$f" in obj:
            return float(obj["$f"])
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


class Game:
    """A Balatro run in the Rust engine.

    Create one with :meth:`new` or :meth:`from_dict`; advance it with
    :meth:`step`. Every step either completes or raises and leaves the game
    unchanged: :class:`IllegalActionError` for an action the rules refuse,
    :class:`NotPortedError` for game logic the Rust engine doesn't have yet.
    """

    __slots__ = ("_g",)

    def __init__(self, inner: _core.Game) -> None:
        self._g = inner

    @classmethod
    def new(
        cls,
        back: str = "b_red",
        stake: int = 1,
        seed: str | None = None,
        challenge: str | None = None,
    ) -> Game:
        """A new run at the blind-select screen. Without a seed, a random one."""
        return cls(_core.Game.new(back, stake, seed or random_seed(), challenge))

    @classmethod
    def from_dict(cls, snapshot: dict[str, Any]) -> Game:
        """The run a canonical snapshot describes."""
        return cls(_core.Game.from_snapshot(json.dumps(_to_json(snapshot))))

    def to_dict(self) -> dict[str, Any]:
        """The canonical snapshot (``jackdaw.snapshot/1``)."""
        return _from_json(json.loads(self._g.snapshot()))

    def step(self, action: Any) -> None:
        """Take one action (a dataclass from :mod:`jackdaw.engine.actions`, or its
        tape form) and play until the game waits for the next decision."""
        self._g.step(json.dumps(encode_action(action)))

    def clone(self) -> Game:
        """An independent copy of the run."""
        return Game(self._g.clone())

    def state_hash(self) -> str:
        """The SHA-256 hex digest of the canonical snapshot (``tests/parity/canon.py``)."""
        return self._g.state_hash()

    @property
    def phase(self) -> GamePhase:
        return GamePhase(self._g.phase)

    @property
    def seed(self) -> str:
        return self._g.seed

    @property
    def ante(self) -> int:
        return int(self._g.ante)

    @property
    def round(self) -> int:
        return int(self._g.round)

    @property
    def dollars(self) -> int:
        return int(self._g.dollars)

    @property
    def chips(self) -> float:
        return self._g.chips

    @property
    def hands_left(self) -> int:
        return int(self._g.hands_left)

    @property
    def discards_left(self) -> int:
        return int(self._g.discards_left)

    @property
    def won(self) -> bool:
        return self._g.won

    def __repr__(self) -> str:
        return (
            f"Game(seed={self.seed!r}, ante={self.ante}, round={self.round}, "
            f"phase={self.phase.value!r}, dollars={self.dollars})"
        )
