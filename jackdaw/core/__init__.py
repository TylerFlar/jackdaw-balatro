"""The Rust engine behind a Python interface.

:class:`Game` is a run in the Rust core (``jackdaw._core``). It takes the
action dataclasses of :mod:`jackdaw.engine.actions` (or their tape forms) and
reads and writes the canonical snapshot (``docs/snapshot-schema.md``) as its
``to_dict`` / ``from_dict`` form, which is also the save format.
"""

from __future__ import annotations

from jackdaw.core.builder import GameBuilder
from jackdaw.core.game import (
    Game,
    IllegalActionError,
    Legality,
    NotPortedError,
    decode_action,
    encode_action,
    random_seed,
)
from jackdaw.core.views import BlindView, CardView, HandLevel, Play, Score, ScoreEvent

__all__ = [
    "BlindView",
    "CardView",
    "Game",
    "GameBuilder",
    "HandLevel",
    "Play",
    "Score",
    "ScoreEvent",
    "IllegalActionError",
    "Legality",
    "NotPortedError",
    "decode_action",
    "encode_action",
    "random_seed",
]
