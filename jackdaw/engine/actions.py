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
| BLIND_SELECT        | SelectBlind, SkipBlind, RerollBoss, SellCard,        |
|                     | UseConsumable                                        |
+---------------------+------------------------------------------------------+
| SELECTING_HAND      | PlayHand, Discard, SortHand, SellCard,               |
|                     | UseConsumable, SwapHandLeft/Right,                   |
|                     | SwapJokersLeft/Right                                 |
+---------------------+------------------------------------------------------+
| SHOP                | BuyCard, BuyAndUse, SellCard, UseConsumable,         |
|                     | RedeemVoucher, OpenBooster, Reroll,                  |
|                     | SwapJokersLeft/Right, NextRound                      |
+---------------------+------------------------------------------------------+
| PACK_OPENING        | PickPackCard, SkipPack, SellCard, UseConsumable      |
+---------------------+------------------------------------------------------+
| ROUND_EVAL          | CashOut, SellCard, UseConsumable                     |
+---------------------+------------------------------------------------------+
| GAME_OVER           | *(terminal — no valid actions)*                      |
+---------------------+------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.card import Card


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

# Vanilla actions BalatroBot's endpoints cannot express.  get_legal_actions
# and the env's factored action space (types 21 and 22) offer them; the
# bridge leaves them out.
SIM_ONLY_ACTIONS: tuple[type, ...] = (RerollBoss, BuyAndUse)


# ---------------------------------------------------------------------------
# Legal action generation
# ---------------------------------------------------------------------------


def get_legal_actions(game_state: dict[str, Any]) -> list[Action]:
    """Return every valid action from the current game state.

    The SELECTING_HAND phase has a combinatorial action space —
    ``PlayHand`` and ``Discard`` each admit up to C(n, 5) card
    subsets.  This function does **not** enumerate every possible
    subset.  Instead it returns *marker* actions with empty
    ``card_indices``, signalling that play/discard is legal.
    The Gym env (M13) will handle subset enumeration or
    parameterised action spaces.

    Parameters
    ----------
    game_state:
        The full runtime state dict.  Must include:

        * ``phase`` (:class:`GamePhase`) — current game phase.
        * ``current_round`` (dict) — ``hands_left``, ``discards_left``,
          ``reroll_cost``, ``free_rerolls``.
        * ``dollars`` (int) — current bank balance.
        * ``bankrupt_at`` (int) — debt floor (-20 with Credit Card).
        * ``round_resets`` (dict) — ``blind_states`` dict.
        * ``blind_on_deck`` (str | None) — ``"Small"``/``"Big"``/``"Boss"``.
        * ``hand`` (list[Card]) — cards in the player's hand.
        * ``jokers`` (list[Card]) — active joker cards.
        * ``consumables`` (list[Card]) — owned consumable cards.
        * ``joker_slots`` (int) — max joker capacity.
        * ``consumable_slots`` (int) — max consumable capacity.
        * ``shop_cards`` (list[Card]) — buyable cards in shop.
        * ``shop_vouchers`` (list[Card]) — buyable vouchers.
        * ``shop_boosters`` (list[Card]) — buyable booster packs.
        * ``pack_cards`` (list[Card]) — cards in an opened pack.
        * ``pack_choices_remaining`` (int) — picks left in pack.

    Returns
    -------
    list[Action]
        All legal actions.  Empty for ``GAME_OVER``.
    """
    phase = game_state.get("phase")
    if phase is None:
        return []

    if isinstance(phase, str):
        phase = GamePhase(phase)

    if phase == GamePhase.GAME_OVER:
        return []
    if phase == GamePhase.BLIND_SELECT:
        return _legal_blind_select(game_state)
    if phase == GamePhase.SELECTING_HAND:
        return _legal_selecting_hand(game_state)
    if phase == GamePhase.ROUND_EVAL:
        return _legal_round_eval(game_state)
    if phase == GamePhase.SHOP:
        return _legal_shop(game_state)
    if phase == GamePhase.PACK_OPENING:
        return _legal_pack_opening(game_state)
    return []


# ---------------------------------------------------------------------------
# Per-phase helpers
# ---------------------------------------------------------------------------


def _legal_blind_select(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    blind_on_deck = gs.get("blind_on_deck", "Small")

    actions.append(SelectBlind())

    # Can skip Small and Big, but not Boss
    if blind_on_deck in ("Small", "Big"):
        actions.append(SkipBlind())

    if can_reroll_boss(gs):
        actions.append(RerollBoss())

    actions.extend(_sell_actions(gs))

    # Consumables usable during blind select
    actions.extend(_usable_consumables(gs))
    return actions


BOSS_REROLL_COST = 10


def can_reroll_boss(gs: dict[str, Any]) -> bool:
    """Whether the blind-select screen's boss reroll button is live.

    The button exists only with Retcon or Director's Cut
    (UI_definitions.lua:1427-1428), and reroll_boss_button enables it when
    ``dollars - bankrupt_at - 10 >= 0`` and (Retcon, or Director's Cut
    with no reroll yet this ante) (button_callbacks.lua:2784-2798).
    """
    used = gs.get("used_vouchers") or {}
    if not (used.get("v_retcon") or used.get("v_directors_cut")):
        return False
    if gs.get("dollars", 0) - gs.get("bankrupt_at", 0) - BOSS_REROLL_COST < 0:
        return False
    rerolled = (gs.get("round_resets") or {}).get("boss_rerolled")
    return bool(used.get("v_retcon")) or not rerolled


def _legal_selecting_hand(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    cr = gs.get("current_round", {})
    hand: list[Card] = gs.get("hand", [])

    # PlayHand: marker action — at least 1 card in hand + hands remaining
    if hand and cr.get("hands_left", 0) > 0:
        actions.append(PlayHand(card_indices=()))

    # Discard: marker action — at least 1 card in hand + discards remaining
    if hand and cr.get("discards_left", 0) > 0:
        actions.append(Discard(card_indices=()))

    # Sort and reorder hand
    if len(hand) > 1:
        actions.append(SortHand(mode="rank"))
        actions.append(SortHand(mode="suit"))
        for i in range(1, len(hand)):
            actions.append(SwapHandLeft(idx=i))
        for i in range(len(hand) - 1):
            actions.append(SwapHandRight(idx=i))

    actions.extend(_sell_actions(gs))

    # Consumables usable during hand selection
    actions.extend(_usable_consumables(gs))

    # Swap jokers
    actions.extend(_joker_swap_actions(gs.get("jokers", [])))

    return actions


def joker_swap_allowed(jokers: list[Card], idx: int, direction: int) -> bool:
    """Whether ``jokers[idx]`` may swap with its neighbour in *direction* (±1).

    A pinned joker never moves: the joker area sorts pinned cards first
    every frame (cardarea.lua:528), so swapping one, or swapping another
    card past it, snaps straight back.  Shared by the executor, the legal
    actions and the env mask.
    """
    other = idx + direction
    if not (0 <= idx < len(jokers) and 0 <= other < len(jokers)):
        return False
    return not (getattr(jokers[idx], "pinned", False) or getattr(jokers[other], "pinned", False))


def _joker_swap_actions(jokers: list[Card]) -> list[Action]:
    actions: list[Action] = [
        SwapJokersLeft(idx=i) for i in range(1, len(jokers)) if joker_swap_allowed(jokers, i, -1)
    ]
    actions.extend(
        SwapJokersRight(idx=i) for i in range(len(jokers) - 1) if joker_swap_allowed(jokers, i, +1)
    )
    return actions


def _legal_round_eval(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = [CashOut()]
    # Card:can_sell_card has no state gate (card.lua:1640-1653): the cash-out
    # screen sells too (G203).
    actions.extend(_sell_actions(gs))
    actions.extend(_usable_consumables(gs))
    return actions


def _legal_shop(gs: dict[str, Any]) -> list[Action]:
    from jackdaw.engine.economy import can_afford, spendable_dollars

    actions: list[Action] = []
    jokers: list[Card] = gs.get("jokers", [])

    # Buy cards from shop
    shop_cards: list[Card] = gs.get("shop_cards", [])
    for i, card in enumerate(shop_cards):
        if not can_afford(gs, card.cost):
            continue
        # Check slot availability
        if not buy_space(card, gs):
            continue  # No room — agent must use/sell a card first
        actions.append(BuyCard(shop_index=i))

    # Buy and use: affordable shop consumables that can be used right now,
    # slots full or not (can_buy_and_use, button_callbacks.lua:77-89).
    from jackdaw.engine.consumables import consumable_block_reason, default_use_targets

    for i, card in enumerate(shop_cards):
        ability = card.ability if isinstance(card.ability, dict) else {}
        # ``ability.consumeable`` is the center's config, a Lua table: present
        # (and truthy) even when empty, as for Judgement or The Soul.
        if "consumeable" not in ability or not can_afford(gs, card.cost):
            continue
        # can_buy_and_use asks can_use_consumeable alone: Card:check_use (Ankh
        # with every joker slot taken) stops the use after the purchase (G138).
        if consumable_block_reason(card, gs, check_use=False) is None:
            actions.append(BuyAndUse(shop_index=i, target_indices=default_use_targets(card, gs)))

    actions.extend(_sell_actions(gs))

    # Use owned consumables
    actions.extend(_usable_consumables(gs))

    # Redeem vouchers (can_redeem: no free-item exemption)
    shop_vouchers: list[Card] = gs.get("shop_vouchers", [])
    for i, voucher in enumerate(shop_vouchers):
        if voucher.cost <= spendable_dollars(gs):
            actions.append(RedeemVoucher(card_index=i))

    # Open boosters
    shop_boosters: list[Card] = gs.get("shop_boosters", [])
    for i, booster in enumerate(shop_boosters):
        if can_afford(gs, booster.cost):
            actions.append(OpenBooster(card_index=i))

    # Reroll
    cr = gs.get("current_round", {})
    reroll_cost = cr.get("reroll_cost", 5)
    free_rerolls = cr.get("free_rerolls", 0)
    if free_rerolls > 0 or can_afford(gs, reroll_cost):
        actions.append(Reroll())

    # Next round (always available in shop)
    actions.append(NextRound())

    # Swap jokers
    actions.extend(_joker_swap_actions(jokers))

    return actions


def _legal_pack_opening(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    pack_cards: list[Card] = gs.get("pack_cards", [])
    remaining: int = gs.get("pack_choices_remaining", 0)

    if remaining > 0:
        from jackdaw.engine.consumables import default_use_targets, pack_pick_block_reason

        # Vanilla greys out a pack card with nowhere to go — a Joker with
        # no free slot (button_callbacks.lua:2112-2113), a creator
        # consumable with no room for what it makes.  This is the gate the
        # PickPackCard executor raises on, and a targeting tarot carries
        # its default selection, so every pick offered can go to step.
        for i, card in enumerate(pack_cards):
            if pack_pick_block_reason(card, gs) is None:
                actions.append(
                    PickPackCard(card_index=i, target_indices=default_use_targets(card, gs))
                )

    if can_skip_booster(gs):
        actions.append(SkipPack())
    actions.extend(_sell_actions(gs))

    # Owned consumables stay usable while a pack is open, without
    # spending a pick (button_callbacks.lua:2269).
    actions.extend(_usable_consumables(gs))
    return actions


def buy_space(card: Card, gs: dict[str, Any]) -> bool:
    """``G.FUNCS.check_for_buy_space(card)`` (button_callbacks.lua:2392-2402).

    Vouchers and playing cards always fit; a Joker or consumable needs its area
    to hold fewer cards than the limit, plus one for a Negative card (G201).
    """
    card_set = card.ability.get("set", "") if isinstance(card.ability, dict) else ""
    bonus = 1 if isinstance(card.edition, dict) and card.edition.get("negative") else 0
    if card_set == "Joker":
        return len(gs.get("jokers", [])) < gs.get("joker_slots", 5) + bonus
    if card_set in ("Tarot", "Planet", "Spectral"):
        return len(gs.get("consumables", [])) < gs.get("consumable_slots", 2) + bonus
    return True


def can_skip_booster(gs: dict[str, Any]) -> bool:
    """``G.FUNCS.can_skip_booster`` (button_callbacks.lua:2132-2141, G202).

    A pack with cards left can be skipped, but an Arcana or Spectral pack only
    once its hand is dealt, or when the hand can hold no cards.
    """
    if not gs.get("pack_cards"):
        return False
    if gs.get("pack_type") in ("Celestial", "Standard", "Buffoon"):
        return True
    return bool(gs.get("hand")) or gs.get("hand_size", 8) <= 0


def _sell_actions(gs: dict[str, Any]) -> list[Action]:
    """SellCard for every non-eternal joker and every consumable.

    Selling is not shop-only: vanilla's Card:can_sell_card (card.lua:1640)
    has no state gate, so the executor accepts it in every phase in
    ``_SELLABLE_PHASES`` (game.py) and these callers mirror that set.
    """
    jokers: list[Card] = gs.get("jokers", [])
    actions: list[Action] = [
        SellCard(area="jokers", card_index=i) for i, joker in enumerate(jokers) if not joker.eternal
    ]
    actions.extend(
        SellCard(area="consumables", card_index=i) for i in range(len(gs.get("consumables", [])))
    )
    return actions


# ---------------------------------------------------------------------------
# Consumable usability helper
# ---------------------------------------------------------------------------


def _usable_consumables(gs: dict[str, Any]) -> list[Action]:
    """Return a UseConsumable action for each owned consumable usable now.

    A consumable that needs a hand selection (The Magician, Death, ...)
    carries the default one from
    :func:`~jackdaw.engine.consumables.default_use_targets`, so every
    action returned can go straight to ``step``; callers may substitute
    any other legal selection.
    """
    consumables: list[Card] = gs.get("consumables", [])
    if not consumables:
        return []

    from jackdaw.engine.consumables import consumable_block_reason, default_use_targets

    return [
        UseConsumable(card_index=i, target_indices=default_use_targets(card, gs))
        for i, card in enumerate(consumables)
        if consumable_block_reason(card, gs) is None
    ]
