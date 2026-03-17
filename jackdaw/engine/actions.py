"""Player action types and game phase enum.

Every decision point in a Balatro run is represented as a frozen dataclass
action.  The simulator's step function accepts an :data:`Action` and
advances the game state accordingly.

Game phases
-----------
:class:`GamePhase` enumerates the phases where player input is needed.
Each phase permits a specific subset of actions:

+---------------------+------------------------------------------------------+
| Phase               | Valid actions                                         |
+=====================+======================================================+
| BLIND_SELECT        | SelectBlind, SkipBlind                               |
+---------------------+------------------------------------------------------+
| SELECTING_HAND      | PlayHand, Discard, SortHand, UseConsumable,          |
|                     | ReorderJokers                                        |
+---------------------+------------------------------------------------------+
| SHOP                | BuyCard, BuyAndUse, SellCard, UseConsumable,         |
|                     | RedeemVoucher, OpenBooster, Reroll, ReorderJokers,   |
|                     | NextRound                                            |
+---------------------+------------------------------------------------------+
| PACK_OPENING        | PickPackCard, SkipPack                               |
+---------------------+------------------------------------------------------+
| ROUND_EVAL          | CashOut                                              |
+---------------------+------------------------------------------------------+
| GAME_OVER           | *(terminal — no valid actions)*                      |
+---------------------+------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------


class GamePhase(str, Enum):
    """Phases where player input is needed.

    Maps to ``G.STATES`` in the Lua source, filtered to decision points.
    """

    BLIND_SELECT = "blind_select"
    SELECTING_HAND = "selecting_hand"
    ROUND_EVAL = "round_eval"
    SHOP = "shop"
    PACK_OPENING = "pack_opening"
    GAME_OVER = "game_over"


# ---------------------------------------------------------------------------
# Blind selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectBlind:
    """Accept the current blind and start the round."""


@dataclass(frozen=True)
class SkipBlind:
    """Skip the current blind (Small or Big) and collect the tag reward."""


# ---------------------------------------------------------------------------
# Hand play / discard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayHand:
    """Play selected cards from the hand.

    ``card_indices`` are 0-based indices into ``hand.cards``, 1–5 cards.
    """

    card_indices: tuple[int, ...]


@dataclass(frozen=True)
class Discard:
    """Discard selected cards from the hand.

    ``card_indices`` are 0-based indices into ``hand.cards``, 1–5 cards.
    """

    card_indices: tuple[int, ...]


# ---------------------------------------------------------------------------
# Shop actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuyCard:
    """Purchase a card from the shop.

    ``shop_index`` is the 0-based index into the shop card area
    (jokers, tarots, planets, or playing cards depending on slot).
    """

    shop_index: int


@dataclass(frozen=True)
class BuyAndUse:
    """Purchase a consumable and immediately use it.

    Equivalent to buying a tarot/planet/spectral and applying it in one
    step.  ``target_indices`` are hand card indices for targeted effects
    (e.g. The Magician targets up to 2 cards).
    """

    shop_index: int
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SellCard:
    """Sell a card from jokers or consumables for its sell value.

    ``area`` is ``'jokers'`` or ``'consumables'``.
    """

    area: str
    card_index: int


@dataclass(frozen=True)
class UseConsumable:
    """Use a consumable from the consumable area.

    ``target_indices`` are 0-based indices into ``hand.cards`` for
    targeted consumables (tarots that modify specific cards).
    ``None`` for untargeted consumables (planets, most spectrals).
    """

    card_index: int
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RedeemVoucher:
    """Purchase and activate a voucher from the shop."""

    card_index: int


@dataclass(frozen=True)
class OpenBooster:
    """Open a booster pack from the shop."""

    card_index: int


@dataclass(frozen=True)
class Reroll:
    """Reroll the shop (costs reroll_cost dollars)."""


@dataclass(frozen=True)
class NextRound:
    """Leave the shop and proceed to the next round."""


# ---------------------------------------------------------------------------
# Pack opening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickPackCard:
    """Select a card from an opened booster pack."""

    card_index: int


@dataclass(frozen=True)
class SkipPack:
    """Skip remaining picks in a booster pack."""


# ---------------------------------------------------------------------------
# Round evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CashOut:
    """Accept end-of-round earnings and proceed to the shop."""


# ---------------------------------------------------------------------------
# Utility actions (available in multiple phases)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SortHand:
    """Sort the hand by rank or suit.

    ``mode`` is ``'rank'`` or ``'suit'``.
    """

    mode: str


@dataclass(frozen=True)
class ReorderJokers:
    """Reorder jokers by specifying a permutation of indices.

    ``new_order`` is a tuple of 0-based joker indices in the desired
    order (e.g. ``(2, 0, 1)`` moves the third joker to the front).
    """

    new_order: tuple[int, ...]


# ---------------------------------------------------------------------------
# Union type
# ---------------------------------------------------------------------------

Action = (
    PlayHand
    | Discard
    | SelectBlind
    | SkipBlind
    | BuyCard
    | BuyAndUse
    | SellCard
    | UseConsumable
    | RedeemVoucher
    | OpenBooster
    | PickPackCard
    | SkipPack
    | Reroll
    | NextRound
    | CashOut
    | SortHand
    | ReorderJokers
)
