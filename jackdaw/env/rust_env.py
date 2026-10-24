"""The env layer's Rust paths: action masks and factored actions for a Rust run.

``jackdaw._core`` computes the action mask, the factored-action conversions and the
consumables' targeting info (``crates/jackdaw-env``); these helpers wrap the results
as :mod:`jackdaw.env.action_space`'s Python objects for a :class:`jackdaw.core.Game`.
Masks arrive as one ``bytearray`` that the arrays view without copying.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from jackdaw import _core
from jackdaw.core.game import decode_action, encode_action
from jackdaw.env.action_space import NUM_ACTION_TYPES, ActionMask, ActionType, Legality
from jackdaw.env.game_spec import FactoredAction

if TYPE_CHECKING:
    from jackdaw.core import Game

if _core.NUM_ACTION_TYPES != NUM_ACTION_TYPES:
    raise ImportError("jackdaw._core and jackdaw.env disagree on the action types")

_BOOL = np.dtype(np.bool_)
_TYPES = tuple(ActionType)

MaskParts = tuple[bytearray, int, list[tuple[int, int, int]], int, int, int | None]


def action_mask_from_parts(parts: MaskParts) -> ActionMask:
    """Wrap ``_core``'s ``(data, n_hand, entities, min, max, forced)`` as an ActionMask."""
    data, n_hand, entities, min_select, max_select, forced = parts
    return ActionMask(
        type_mask=np.ndarray((NUM_ACTION_TYPES,), _BOOL, data),
        card_mask=np.ndarray((n_hand,), _BOOL, data, NUM_ACTION_TYPES),
        entity_masks={_TYPES[t]: np.ndarray((n,), _BOOL, data, off) for t, off, n in entities},
        max_card_select=max_select,
        min_card_select=min_select,
        forced_card=forced,
    )


def action_mask(game: Game, legality: Legality = Legality.VANILLA) -> ActionMask:
    """The action mask of a Rust run (:func:`~jackdaw.env.action_space.get_action_mask`)."""
    return action_mask_from_parts(game._g.action_mask(Legality(legality).value))


def _to_json(obj: Any) -> Any:
    """Snapshot JSON form: non-finite numbers as ``{"$f": ...}``."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"$f": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    return obj


def action_mask_from_snapshot(
    snapshot: dict[str, Any], legality: Legality = Legality.VANILLA
) -> ActionMask:
    """The action mask of the state a canonical snapshot describes, computed in Rust."""
    text = json.dumps(_to_json(snapshot))
    return action_mask_from_parts(_core._action_mask(text, Legality(legality).value))


def factored_to_engine_action(fa: FactoredAction, game: Game) -> Any:
    """:func:`~jackdaw.env.action_space.factored_to_engine_action` for a Rust run."""
    text = game._g.factored_to_engine(fa.action_type, fa.entity_target, fa.card_target)
    return decode_action(json.loads(text))


def engine_action_to_factored(action: Any) -> FactoredAction:
    """:func:`~jackdaw.env.action_space.engine_action_to_factored`, computed in Rust (the
    conversion reads no state)."""
    t, entity, cards = _core.engine_to_factored(json.dumps(encode_action(action)))
    return FactoredAction(
        action_type=_TYPES[t],
        card_target=tuple(cards) if cards is not None else None,
        entity_target=entity,
    )


def consumable_target_info(game: Game, area: str, index: int) -> tuple[int, int, bool]:
    """:func:`~jackdaw.env.action_space.get_consumable_target_info` for the card at
    ``index`` of ``area`` (``"consumables"`` or ``"pack_cards"``) of a Rust run."""
    return game._g.consumable_target_info(area, index)
