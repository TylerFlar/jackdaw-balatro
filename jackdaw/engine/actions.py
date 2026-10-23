"""Player action types, the game phase enum and the actions' tape form.

Every decision point in a Balatro run is represented as a frozen dataclass
action.  :meth:`jackdaw.core.Game.step` takes an :data:`Action` (or its tape
form, :func:`encode_action`) and advances the run.

Game phases
-----------
:class:`GamePhase` enumerates the phases where player input is needed.
Each phase permits a specific subset of actions:

+---------------------+------------------------------------------------------+
| Phase               | Valid actions                                         |
+=====================+======================================================+
| BLIND_SELECT        | SelectBlind, SkipBlind, RerollBoss, SellCard,        |
|                     | UseConsumable, SwapJokersLeft/Right                  |
+---------------------+------------------------------------------------------+
| SELECTING_HAND      | PlayHand, Discard, SortHand, SellCard,               |
|                     | UseConsumable, SwapHandLeft/Right,                   |
|                     | SwapJokersLeft/Right                                 |
+---------------------+------------------------------------------------------+
| SHOP                | BuyCard, BuyAndUse, SellCard, UseConsumable,         |
|                     | RedeemVoucher, OpenBooster, Reroll,                  |
|                     | SwapJokersLeft/Right, NextRound                      |
+---------------------+------------------------------------------------------+
| PACK_OPENING        | PickPackCard, SkipPack, SellCard, UseConsumable,     |
|                     | SwapHandLeft/Right, SwapJokersLeft/Right             |
+---------------------+------------------------------------------------------+
| ROUND_EVAL          | CashOut, SellCard, UseConsumable,                    |
|                     | SwapJokersLeft/Right                                 |
+---------------------+------------------------------------------------------+
| GAME_OVER           | *(terminal — no valid actions)*                      |
+---------------------+------------------------------------------------------+

A drag reorders the jokers in every phase and the hand whenever it holds
cards (G206): CardArea:align_cards sorts both by position every frame
(cardarea.lua:450-465, 509-529), in a pack's dealt hand too (436-449).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------


class GamePhase(StrEnum):
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


@dataclass(frozen=True)
class RerollBoss:
    """Reroll this ante's boss blind for $10 (Director's Cut / Retcon).

    The blind-select screen's reroll button (button_callbacks.lua:2784-2853):
    Director's Cut allows one reroll per ante, Retcon any number.  Sim-only:
    BalatroBot has no endpoint for it (factored action type 21).
    """


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
    """Buy a consumable from the shop and use it at once.

    The shop's "Buy and Use" button (button_callbacks.lua:77-89,
    2404-2477): needs no consumable slot.  ``target_indices`` is the hand
    selection, as for :class:`UseConsumable` (in the shop the hand is
    empty, so cards that need one are never usable).  Sim-only: BalatroBot
    has no endpoint for it (factored action type 22).
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
    """Select a card from an opened booster pack.

    For Arcana/Spectral packs, ``target_indices`` specifies which dealt
    hand cards the consumable should target (e.g. for The Magician
    enhancing up to 2 cards).  ``None`` for non-targeted picks
    (Celestial, Standard, Buffoon packs).
    """

    card_index: int
    target_indices: tuple[int, ...] | None = None


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
class SwapHandLeft:
    """Swap a hand card one position to the left.

    Swaps ``hand[idx]`` with ``hand[idx - 1]``.  Free action — no cost,
    doesn't consume hands or discards.  Left-to-right order affects which
    card is "first scored" for jokers like Photograph and Hanging Chad.
    """

    idx: int


@dataclass(frozen=True)
class SwapHandRight:
    """Swap a hand card one position to the right.

    Swaps ``hand[idx]`` with ``hand[idx + 1]``.  Free action.
    """

    idx: int


@dataclass(frozen=True)
class SwapJokersLeft:
    """Swap a joker one position to the left.

    Swaps ``jokers[idx]`` with ``jokers[idx - 1]``.
    Left-to-right order affects Phase 9 scoring (additive before
    multiplicative), Blueprint (copies right neighbor), Brainstorm
    (copies leftmost), Ceremonial Dagger (destroys right neighbor).
    """

    idx: int


@dataclass(frozen=True)
class SwapJokersRight:
    """Swap a joker one position to the right.

    Swaps ``jokers[idx]`` with ``jokers[idx + 1]``.
    """

    idx: int


# ---------------------------------------------------------------------------
# Union type
# ---------------------------------------------------------------------------

Action = (
    PlayHand
    | Discard
    | SelectBlind
    | SkipBlind
    | RerollBoss
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
    | SwapHandLeft
    | SwapHandRight
    | SwapJokersLeft
    | SwapJokersRight
)

# Vanilla actions BalatroBot's endpoints cannot express.  The legal actions
# and the env's factored action space (types 21 and 22) offer them; the
# bridge leaves them out (``Game.bot_request`` gives no request for them).
SIM_ONLY_ACTIONS: tuple[type, ...] = (RerollBoss, BuyAndUse)


# ---------------------------------------------------------------------------
# Tape form
# ---------------------------------------------------------------------------

_ACTION_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        PlayHand, Discard, SelectBlind, SkipBlind, RerollBoss, BuyCard, BuyAndUse, SellCard,
        UseConsumable, RedeemVoucher, OpenBooster, PickPackCard, SkipPack, Reroll, NextRound,
        CashOut, SortHand, SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight,
    )
}  # fmt: skip


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
