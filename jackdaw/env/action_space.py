"""Factored action space for the RL agent.

Converts between a 23-type factored action representation and the engine's
Action dataclasses.  The factored representation decomposes complex actions
(arbitrary permutations, combinatorial card subsets) into tractable atomic
operations that a policy network can learn.

The 23 action types (21 and 22 were appended, so 0-20 keep their indices):

+-----+-------------------+---------------------------------------------------+
| ID  | Name              | Targets                                           |
+=====+===================+===================================================+
|  0  | PlayHand          | card_target (binary mask over hand, 1-5 cards)    |
|  1  | Discard           | card_target (binary mask over hand, 1-5 cards)    |
|  2  | SelectBlind       | none                                              |
|  3  | SkipBlind         | none                                              |
|  4  | CashOut           | none                                              |
|  5  | Reroll            | none                                              |
|  6  | NextRound         | none                                              |
|  7  | SkipPack          | none                                              |
|  8  | BuyCard           | entity_target (shop card index)                   |
|  9  | SellJoker         | entity_target (joker index)                       |
| 10  | SellConsumable    | entity_target (consumable index)                  |
| 11  | UseConsumable     | entity_target (consumable idx) + card_target      |
| 12  | RedeemVoucher     | entity_target (shop voucher index)                |
| 13  | OpenBooster       | entity_target (shop booster index)                |
| 14  | PickPackCard      | entity_target (pack card index)                   |
| 15  | SwapJokersLeft    | entity_target (joker index to swap left)          |
| 16  | SwapJokersRight   | entity_target (joker index to swap right)         |
| 17  | SwapHandLeft      | entity_target (hand card index to swap left)      |
| 18  | SwapHandRight     | entity_target (hand card index to swap right)     |
| 19  | SortHandRank      | none                                              |
| 20  | SortHandSuit      | none                                              |
| 21  | RerollBoss        | none (Director's Cut / Retcon, $10)               |
| 22  | BuyAndUse         | entity_target (shop card index)                   |
+-----+-------------------+---------------------------------------------------+

The masks, the conversions and the default card selections are computed by the
Rust engine (``jackdaw._core``, wrapped in :mod:`jackdaw.env.rust_env`) for a
:class:`jackdaw.core.Game`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from jackdaw.core.game import Legality
from jackdaw.engine.actions import Action
from jackdaw.env.game_spec import FactoredAction  # noqa: F401 — re-export

if TYPE_CHECKING:
    from jackdaw.core import Game

# ---------------------------------------------------------------------------
# ActionType enum
# ---------------------------------------------------------------------------

NUM_ACTION_TYPES = 23


class ActionType(IntEnum):
    """The 23 factored action types for the RL agent."""

    PlayHand = 0
    Discard = 1
    SelectBlind = 2
    SkipBlind = 3
    CashOut = 4
    Reroll = 5
    NextRound = 6
    SkipPack = 7
    BuyCard = 8
    SellJoker = 9
    SellConsumable = 10
    UseConsumable = 11
    RedeemVoucher = 12
    OpenBooster = 13
    PickPackCard = 14
    SwapJokersLeft = 15
    SwapJokersRight = 16
    SwapHandLeft = 17
    SwapHandRight = 18
    SortHandRank = 19
    SortHandSuit = 20
    RerollBoss = 21
    BuyAndUse = 22


# ---------------------------------------------------------------------------
# ActionMask
# ---------------------------------------------------------------------------


@dataclass
class ActionMask:
    """Masks indicating which actions are currently legal.

    Attributes
    ----------
    type_mask:
        Shape ``(23,)`` bool array — which action types are legal.
    card_mask:
        Shape ``(N_hand,)`` bool array — which hand cards can be selected
        (for PlayHand, Discard, UseConsumable with targeting).
    entity_masks:
        ``{action_type: np.ndarray}`` — for entity-targeted action types
        (8–18, 22), which specific entity indices are legal targets.
    max_card_select:
        Maximum cards that can be selected (5 for play/discard, varies
        for consumable targeting).
    min_card_select:
        Minimum cards required (1 for play/discard, varies for
        consumable targeting).
    forced_card:
        Hand index of Cerulean Bell's forced card, which every PlayHand
        and Discard selection must include (vanilla keeps it highlighted),
        or ``None``.
    """

    type_mask: np.ndarray
    card_mask: np.ndarray
    entity_masks: dict[int, np.ndarray]
    max_card_select: int
    min_card_select: int
    forced_card: int | None = None


# ---------------------------------------------------------------------------
# The Rust engine's masks and conversions
# ---------------------------------------------------------------------------


def get_action_mask(game: Game, legality: Legality = Legality.VANILLA) -> ActionMask:
    """The action mask of a run's current state.

    ``legality`` :attr:`~Legality.BALATROBOT` narrows vanilla's actions to the
    ones BalatroBot's endpoints execute (``docs/balatrobot.md``).
    """
    from jackdaw.env.rust_env import action_mask

    return action_mask(game, legality)


def get_consumable_target_info(game: Game, area: str, index: int) -> tuple[int, int, bool]:
    """``(min_cards, max_cards, needs_targets)`` for the consumable at ``index`` of ``area``
    (``"consumables"`` or ``"pack_cards"``): how many hand cards a policy picks as its
    ``card_target``.  If ``needs_targets`` is False, ``card_target`` should be ``None``.
    Raises ``ValueError`` past the end of the area."""
    from jackdaw.env.rust_env import consumable_target_info

    return consumable_target_info(game, area, index)


def factored_to_engine_action(fa: FactoredAction, game: Game) -> Action:
    """The engine action a factored action stands for in a run's current state.

    A UseConsumable, PickPackCard or BuyAndUse without a ``card_target`` gets the
    default selection the mask judged legal (the lowest-index legal cards).  Raises
    ``ValueError`` when the action lacks a target its type needs.
    """
    from jackdaw.env.rust_env import factored_to_engine_action as convert

    return convert(fa, game)


def engine_action_to_factored(action: Action) -> FactoredAction:
    """The factored form of an engine action (the conversion reads no state).

    Raises ``ValueError`` for an action with no factored form.
    """
    from jackdaw.env.rust_env import engine_action_to_factored as convert

    return convert(action)
