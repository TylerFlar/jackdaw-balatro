"""Observation encoder v1 in Rust.

``jackdaw._core`` ports :func:`jackdaw.env.observation.encode_observation`
bit for bit: same arrays, shapes, dtypes and values.  It hands back the
float32 values in one ``bytearray`` (global context, then the hand, joker,
consumable, shop and pack rows) plus the row counts; these helpers view that
buffer as numpy arrays (no copy) and return the same
:class:`~jackdaw.env.observation.Observation` the Python encoder returns.
The arrays are writable, C-contiguous views sharing one buffer per call.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from jackdaw import _core
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
)

if TYPE_CHECKING:
    from jackdaw.core import Game

# Row widths of the entity arrays, in buffer order.
_WIDTHS = (D_PLAYING_CARD, D_JOKER, D_CONSUMABLE, D_SHOP, D_PLAYING_CARD)
_F32 = np.dtype(np.float32)
# Byte sizes of the global context and of one row of each entity array.
_GLOBAL_BYTES = D_GLOBAL * 4
_CARD_BYTES = D_PLAYING_CARD * 4
_JOKER_BYTES = D_JOKER * 4
_CONSUMABLE_BYTES = D_CONSUMABLE * 4
_SHOP_BYTES = D_SHOP * 4

_layout = _core._observation_layout()
if _layout != (D_GLOBAL, list(_WIDTHS)) or _F32.itemsize != 4:
    raise ImportError(f"jackdaw._core observation layout {_layout} differs from the Python encoder")


def observation_from_buffer(data: bytearray | bytes, rows: tuple[int, ...]) -> Observation:
    """Wrap the Rust encoder's ``(data, rows)`` as an :class:`Observation`.

    A ``bytearray`` gives writable arrays (as the Python encoder's are), ``bytes``
    read-only ones.
    """
    n_hand, n_jokers, n_consumables, n_shop, n_pack = rows
    at_hand = _GLOBAL_BYTES
    at_jokers = at_hand + n_hand * _CARD_BYTES
    at_consumables = at_jokers + n_jokers * _JOKER_BYTES
    at_shop = at_consumables + n_consumables * _CONSUMABLE_BYTES
    at_pack = at_shop + n_shop * _SHOP_BYTES
    if len(data) != at_pack + n_pack * _CARD_BYTES:
        raise ValueError(f"observation buffer of {len(data)} bytes does not fit rows {rows}")
    return Observation(
        global_context=np.ndarray((D_GLOBAL,), _F32, data),
        hand_cards=np.ndarray((n_hand, D_PLAYING_CARD), _F32, data, at_hand),
        jokers=np.ndarray((n_jokers, D_JOKER), _F32, data, at_jokers),
        consumables=np.ndarray((n_consumables, D_CONSUMABLE), _F32, data, at_consumables),
        shop_cards=np.ndarray((n_shop, D_SHOP), _F32, data, at_shop),
        pack_cards=np.ndarray((n_pack, D_PLAYING_CARD), _F32, data, at_pack),
    )


def encode_game(game: Game) -> Observation:
    """The observation of a Rust-engine run's current state."""
    return observation_from_buffer(*game.observation_buffer())


def _to_json(obj: Any) -> Any:
    """Snapshot JSON form: non-finite numbers as ``{"$f": ...}``."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"$f": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    return obj


def encode_snapshot(snapshot: dict[str, Any]) -> Observation:
    """The observation of the state a canonical snapshot describes, encoded in Rust."""
    return encode_snapshot_text(json.dumps(_to_json(snapshot)))


def encode_snapshot_text(text: str) -> Observation:
    """:func:`encode_snapshot` for a snapshot already in JSON text."""
    return observation_from_buffer(*_core._encode_observation(text))
