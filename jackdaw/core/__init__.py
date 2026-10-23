"""The Rust engine behind a Python interface.

:class:`Game` is a run in the Rust core (``jackdaw._core``). It takes the same
action dataclasses as the Python engine (:mod:`jackdaw.engine.actions`) and
reads and writes the canonical snapshot (``docs/snapshot-schema.md``) as its
``to_dict`` / ``from_dict`` form, which is also the save format.

During the port the Python engine stays the reference; this package replaces
it as ``jackdaw.engine`` once the Rust engine covers the whole game.
"""

from __future__ import annotations

from jackdaw.core.game import (
    Game,
    IllegalActionError,
    Legality,
    NotPortedError,
    decode_action,
    encode_action,
    random_seed,
)

__all__ = [
    "Game",
    "IllegalActionError",
    "Legality",
    "NotPortedError",
    "decode_action",
    "encode_action",
    "random_seed",
]
