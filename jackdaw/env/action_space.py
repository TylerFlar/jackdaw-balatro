"""Factored action space for the RL agent.

Converts between a 21-type factored action representation and the engine's
Action dataclasses.  The factored representation decomposes complex actions
(arbitrary permutations, combinatorial card subsets) into tractable atomic
operations that a policy network can learn.

The 21 action types:

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
+-----+-------------------+---------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from jackdaw.engine.actions import (
    Action,
    GamePhase,
)
from jackdaw.env.game_spec import FactoredAction  # noqa: F401 — re-export
from jackdaw.engine.actions import (
    BuyAndUse as EngineBuyAndUse,
)
from jackdaw.engine.actions import (
    BuyCard as EngineBuyCard,
)
from jackdaw.engine.actions import (
    CashOut as EngineCashOut,
)
from jackdaw.engine.actions import (
    Discard as EngineDiscard,
)
from jackdaw.engine.actions import (
    NextRound as EngineNextRound,
)
from jackdaw.engine.actions import (
    OpenBooster as EngineOpenBooster,
)
from jackdaw.engine.actions import (
    PickPackCard as EnginePickPackCard,
)
from jackdaw.engine.actions import (
    PlayHand as EnginePlayHand,
)
from jackdaw.engine.actions import (
    RedeemVoucher as EngineRedeemVoucher,
)
from jackdaw.engine.actions import (
    ReorderHand as EngineReorderHand,
)
from jackdaw.engine.actions import (
    ReorderJokers as EngineReorderJokers,
)
from jackdaw.engine.actions import (
    Reroll as EngineReroll,
)
from jackdaw.engine.actions import (
    SelectBlind as EngineSelectBlind,
)
from jackdaw.engine.actions import (
    SellCard as EngineSellCard,
)
from jackdaw.engine.actions import (
    SkipBlind as EngineSkipBlind,
)
from jackdaw.engine.actions import (
    SkipPack as EngineSkipPack,
)
from jackdaw.engine.actions import (
    SortHand as EngineSortHand,
)
from jackdaw.engine.actions import (
    UseConsumable as EngineUseConsumable,
)
from jackdaw.engine.consumables import _resolve_consumable_config, can_use_consumable

# ---------------------------------------------------------------------------
# ActionType enum
# ---------------------------------------------------------------------------

NUM_ACTION_TYPES = 21


class ActionType(IntEnum):
    """The 21 factored action types for the RL agent."""

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


# ---------------------------------------------------------------------------
# ActionMask
# ---------------------------------------------------------------------------


@dataclass
class ActionMask:
    """Masks indicating which actions are currently legal.

    Attributes
    ----------
    type_mask:
        Shape ``(21,)`` bool array — which action types are legal.
    card_mask:
        Shape ``(N_hand,)`` bool array — which hand cards can be selected
        (for PlayHand, Discard, UseConsumable with targeting).
    entity_masks:
        ``{action_type: np.ndarray}`` — for entity-targeted action types
        (8–18), which specific entity indices are legal targets.
    max_card_select:
        Maximum cards that can be selected (5 for play/discard, varies
        for consumable targeting).
    min_card_select:
        Minimum cards required (1 for play/discard, varies for
        consumable targeting).
    """

    type_mask: np.ndarray
    card_mask: np.ndarray
    entity_masks: dict[int, np.ndarray]
    max_card_select: int
    min_card_select: int


# ---------------------------------------------------------------------------
# get_action_mask
# ---------------------------------------------------------------------------


def get_action_mask(game_state: dict[str, Any]) -> ActionMask:
    """Build the action mask from a game state dict.

    Parameters
    ----------
    game_state:
        Full engine game state dict (same schema as
        ``get_legal_actions`` expects).

    Returns
    -------
    ActionMask
        The complete mask structure for the current state.
    """
    phase = game_state.get("phase")
    if isinstance(phase, str):
        phase = GamePhase(phase)

    type_mask = np.zeros(NUM_ACTION_TYPES, dtype=bool)
    entity_masks: dict[int, np.ndarray] = {}

    hand = game_state.get("hand", [])
    jokers = game_state.get("jokers", [])
    consumables = game_state.get("consumables", [])
    cr = game_state.get("current_round", {})
    dollars = game_state.get("dollars", 0)

    # Card mask: all hand cards are selectable by default
    card_mask = np.ones(len(hand), dtype=bool) if hand else np.zeros(0, dtype=bool)

    # Default card selection limits (play/discard)
    max_card_select = 5
    min_card_select = 1

    if phase is None or phase == GamePhase.GAME_OVER:
        return ActionMask(type_mask, card_mask, entity_masks, max_card_select, min_card_select)

    # --- BLIND_SELECT ---
    if phase == GamePhase.BLIND_SELECT:
        type_mask[ActionType.SelectBlind] = True
        blind_on_deck = game_state.get("blind_on_deck", "Small")
        if blind_on_deck in ("Small", "Big"):
            type_mask[ActionType.SkipBlind] = True
        _mask_consumables(type_mask, entity_masks, game_state)

    # --- SELECTING_HAND ---
    elif phase == GamePhase.SELECTING_HAND:
        if hand and cr.get("hands_left", 0) > 0:
            type_mask[ActionType.PlayHand] = True
        if hand and cr.get("discards_left", 0) > 0:
            type_mask[ActionType.Discard] = True
        if len(hand) > 1:
            type_mask[ActionType.SortHandRank] = True
            type_mask[ActionType.SortHandSuit] = True
            _mask_hand_swaps(type_mask, entity_masks, hand)
        _mask_joker_swaps(type_mask, entity_masks, jokers)
        _mask_consumables(type_mask, entity_masks, game_state)

    # --- ROUND_EVAL ---
    elif phase == GamePhase.ROUND_EVAL:
        type_mask[ActionType.CashOut] = True
        _mask_consumables(type_mask, entity_masks, game_state)

    # --- SHOP ---
    elif phase == GamePhase.SHOP:
        _mask_shop_buy(type_mask, entity_masks, game_state)
        _mask_sell_jokers(type_mask, entity_masks, jokers)
        _mask_sell_consumables(type_mask, entity_masks, consumables)
        _mask_consumables(type_mask, entity_masks, game_state)
        _mask_vouchers(type_mask, entity_masks, game_state)
        _mask_boosters(type_mask, entity_masks, game_state)

        # Reroll
        reroll_cost = cr.get("reroll_cost", 5)
        free_rerolls = cr.get("free_rerolls", 0)
        if free_rerolls > 0 or dollars >= reroll_cost:
            type_mask[ActionType.Reroll] = True

        type_mask[ActionType.NextRound] = True
        _mask_joker_swaps(type_mask, entity_masks, jokers)

    # --- PACK_OPENING ---
    elif phase == GamePhase.PACK_OPENING:
        pack_cards = game_state.get("pack_cards", [])
        remaining = game_state.get("pack_choices_remaining", 0)
        if remaining > 0 and pack_cards:
            type_mask[ActionType.PickPackCard] = True
            entity_masks[ActionType.PickPackCard] = np.ones(len(pack_cards), dtype=bool)
        type_mask[ActionType.SkipPack] = True

    return ActionMask(type_mask, card_mask, entity_masks, max_card_select, min_card_select)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def _mask_shop_buy(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    gs: dict[str, Any],
) -> None:
    """Set BuyCard mask for affordable shop cards with available slots."""
    shop_cards = gs.get("shop_cards", [])
    if not shop_cards:
        return
    dollars = gs.get("dollars", 0)
    jokers = gs.get("jokers", [])
    joker_slots = gs.get("joker_slots", 5)
    consumables = gs.get("consumables", [])
    consumable_slots = gs.get("consumable_slots", 2)

    mask = np.zeros(len(shop_cards), dtype=bool)
    for i, card in enumerate(shop_cards):
        if card.cost > dollars:
            continue
        card_set = card.ability.get("set", "") if isinstance(card.ability, dict) else ""
        if card_set == "Joker":
            is_negative = isinstance(card.edition, dict) and card.edition.get("negative")
            if len(jokers) >= joker_slots and not is_negative:
                continue
        elif card_set in ("Tarot", "Planet", "Spectral"):
            if len(consumables) >= consumable_slots:
                continue
        mask[i] = True

    if mask.any():
        type_mask[ActionType.BuyCard] = True
        entity_masks[ActionType.BuyCard] = mask


def _mask_sell_jokers(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    jokers: list,
) -> None:
    """Set SellJoker mask for non-eternal jokers."""
    if not jokers:
        return
    mask = np.array([not j.eternal for j in jokers], dtype=bool)
    if mask.any():
        type_mask[ActionType.SellJoker] = True
        entity_masks[ActionType.SellJoker] = mask


def _mask_sell_consumables(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    consumables: list,
) -> None:
    """Set SellConsumable mask — all consumables are sellable."""
    if not consumables:
        return
    type_mask[ActionType.SellConsumable] = True
    entity_masks[ActionType.SellConsumable] = np.ones(len(consumables), dtype=bool)


def _mask_consumables(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    gs: dict[str, Any],
) -> None:
    """Set UseConsumable mask based on can_use_consumable checks."""
    consumables = gs.get("consumables", [])
    if not consumables:
        return
    hand = gs.get("hand", [])
    jokers = gs.get("jokers", [])
    joker_limit = gs.get("joker_slots", 5)
    consumable_limit = gs.get("consumable_slots", 2)

    mask = np.zeros(len(consumables), dtype=bool)
    for i, card in enumerate(consumables):
        if can_use_consumable(
            card,
            hand_cards=hand,
            jokers=jokers,
            consumables=consumables,
            consumable_limit=consumable_limit,
            joker_limit=joker_limit,
        ):
            mask[i] = True

    if mask.any():
        type_mask[ActionType.UseConsumable] = True
        entity_masks[ActionType.UseConsumable] = mask


def _mask_vouchers(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    gs: dict[str, Any],
) -> None:
    """Set RedeemVoucher mask for affordable vouchers."""
    vouchers = gs.get("shop_vouchers", [])
    if not vouchers:
        return
    dollars = gs.get("dollars", 0)
    mask = np.array([v.cost <= dollars for v in vouchers], dtype=bool)
    if mask.any():
        type_mask[ActionType.RedeemVoucher] = True
        entity_masks[ActionType.RedeemVoucher] = mask


def _mask_boosters(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    gs: dict[str, Any],
) -> None:
    """Set OpenBooster mask for affordable boosters."""
    boosters = gs.get("shop_boosters", [])
    if not boosters:
        return
    dollars = gs.get("dollars", 0)
    mask = np.array([b.cost <= dollars for b in boosters], dtype=bool)
    if mask.any():
        type_mask[ActionType.OpenBooster] = True
        entity_masks[ActionType.OpenBooster] = mask


def _mask_joker_swaps(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    jokers: list,
) -> None:
    """Set SwapJokersLeft/Right masks."""
    n = len(jokers)
    if n <= 1:
        return
    # SwapJokersLeft: all except index 0 (can't swap leftward from first position)
    left_mask = np.ones(n, dtype=bool)
    left_mask[0] = False
    type_mask[ActionType.SwapJokersLeft] = True
    entity_masks[ActionType.SwapJokersLeft] = left_mask

    # SwapJokersRight: all except last index
    right_mask = np.ones(n, dtype=bool)
    right_mask[n - 1] = False
    type_mask[ActionType.SwapJokersRight] = True
    entity_masks[ActionType.SwapJokersRight] = right_mask


def _mask_hand_swaps(
    type_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    hand: list,
) -> None:
    """Set SwapHandLeft/Right masks."""
    n = len(hand)
    if n <= 1:
        return
    left_mask = np.ones(n, dtype=bool)
    left_mask[0] = False
    type_mask[ActionType.SwapHandLeft] = True
    entity_masks[ActionType.SwapHandLeft] = left_mask

    right_mask = np.ones(n, dtype=bool)
    right_mask[n - 1] = False
    type_mask[ActionType.SwapHandRight] = True
    entity_masks[ActionType.SwapHandRight] = right_mask


# ---------------------------------------------------------------------------
# Consumable targeting info
# ---------------------------------------------------------------------------


def get_consumable_target_info(
    card: Any,
) -> tuple[int, int, bool]:
    """Return (min_cards, max_cards, needs_targets) for a consumable.

    Used by the policy network to know how many card targets to sample
    after selecting a consumable.

    Parameters
    ----------
    card:
        A Card object with ``center_key`` and ``ability`` attributes.

    Returns
    -------
    tuple[int, int, bool]
        ``(min_cards, max_cards, needs_targets)``.  If ``needs_targets``
        is False, ``card_target`` should be ``None``.
    """
    cfg = _resolve_consumable_config(card)
    max_h = cfg.get("max_highlighted")
    if not max_h:
        return (0, 0, False)
    min_h = cfg.get("min_highlighted", 1)
    mod_num = cfg.get("mod_num", max_h)
    return (min_h, mod_num, True)


# ---------------------------------------------------------------------------
# factored_to_engine_action
# ---------------------------------------------------------------------------


def factored_to_engine_action(
    fa: FactoredAction,
    game_state: dict[str, Any],
) -> Action:
    """Convert a FactoredAction to an engine Action.

    Parameters
    ----------
    fa:
        The factored action from the policy network.
    game_state:
        Current game state dict (needed for swap→permutation conversion).

    Returns
    -------
    Action
        Engine-compatible action dataclass.

    Raises
    ------
    ValueError
        If the action type or targets are invalid.
    """
    at = ActionType(fa.action_type)

    if at == ActionType.PlayHand:
        if not fa.card_target:
            raise ValueError("PlayHand requires card_target")
        return EnginePlayHand(card_indices=fa.card_target)

    if at == ActionType.Discard:
        if not fa.card_target:
            raise ValueError("Discard requires card_target")
        return EngineDiscard(card_indices=fa.card_target)

    if at == ActionType.SelectBlind:
        return EngineSelectBlind()

    if at == ActionType.SkipBlind:
        return EngineSkipBlind()

    if at == ActionType.CashOut:
        return EngineCashOut()

    if at == ActionType.Reroll:
        return EngineReroll()

    if at == ActionType.NextRound:
        return EngineNextRound()

    if at == ActionType.SkipPack:
        return EngineSkipPack()

    if at == ActionType.BuyCard:
        if fa.entity_target is None:
            raise ValueError("BuyCard requires entity_target")
        return EngineBuyCard(shop_index=fa.entity_target)

    if at == ActionType.SellJoker:
        if fa.entity_target is None:
            raise ValueError("SellJoker requires entity_target")
        return EngineSellCard(area="jokers", card_index=fa.entity_target)

    if at == ActionType.SellConsumable:
        if fa.entity_target is None:
            raise ValueError("SellConsumable requires entity_target")
        return EngineSellCard(area="consumables", card_index=fa.entity_target)

    if at == ActionType.UseConsumable:
        if fa.entity_target is None:
            raise ValueError("UseConsumable requires entity_target")
        target_indices = fa.card_target if fa.card_target else None
        return EngineUseConsumable(
            card_index=fa.entity_target,
            target_indices=target_indices,
        )

    if at == ActionType.RedeemVoucher:
        if fa.entity_target is None:
            raise ValueError("RedeemVoucher requires entity_target")
        return EngineRedeemVoucher(card_index=fa.entity_target)

    if at == ActionType.OpenBooster:
        if fa.entity_target is None:
            raise ValueError("OpenBooster requires entity_target")
        return EngineOpenBooster(card_index=fa.entity_target)

    if at == ActionType.PickPackCard:
        if fa.entity_target is None:
            raise ValueError("PickPackCard requires entity_target")
        return EnginePickPackCard(card_index=fa.entity_target)

    if at == ActionType.SwapJokersLeft:
        return _swap_to_reorder_jokers(fa, game_state, direction=-1)

    if at == ActionType.SwapJokersRight:
        return _swap_to_reorder_jokers(fa, game_state, direction=1)

    if at == ActionType.SwapHandLeft:
        return _swap_to_reorder_hand(fa, game_state, direction=-1)

    if at == ActionType.SwapHandRight:
        return _swap_to_reorder_hand(fa, game_state, direction=1)

    if at == ActionType.SortHandRank:
        return EngineSortHand(mode="rank")

    if at == ActionType.SortHandSuit:
        return EngineSortHand(mode="suit")

    raise ValueError(f"Unknown action type: {fa.action_type}")


def _swap_to_reorder_jokers(
    fa: FactoredAction,
    game_state: dict[str, Any],
    direction: int,
) -> EngineReorderJokers:
    """Convert a swap operation to a ReorderJokers permutation.

    ``direction=-1`` swaps index i with i-1 (left).
    ``direction=+1`` swaps index i with i+1 (right).
    """
    if fa.entity_target is None:
        raise ValueError("Swap requires entity_target")
    jokers = game_state.get("jokers", [])
    n = len(jokers)
    idx = fa.entity_target
    swap_with = idx + direction
    if not (0 <= idx < n and 0 <= swap_with < n):
        raise ValueError(f"Invalid swap: index {idx} direction {direction} with {n} jokers")
    order = list(range(n))
    order[idx], order[swap_with] = order[swap_with], order[idx]
    return EngineReorderJokers(new_order=tuple(order))


def _swap_to_reorder_hand(
    fa: FactoredAction,
    game_state: dict[str, Any],
    direction: int,
) -> EngineReorderHand:
    """Convert a swap operation to a ReorderHand permutation."""
    if fa.entity_target is None:
        raise ValueError("Swap requires entity_target")
    hand = game_state.get("hand", [])
    n = len(hand)
    idx = fa.entity_target
    swap_with = idx + direction
    if not (0 <= idx < n and 0 <= swap_with < n):
        raise ValueError(f"Invalid swap: index {idx} direction {direction} with {n} hand cards")
    order = list(range(n))
    order[idx], order[swap_with] = order[swap_with], order[idx]
    return EngineReorderHand(new_order=tuple(order))


# ---------------------------------------------------------------------------
# engine_action_to_factored
# ---------------------------------------------------------------------------


def engine_action_to_factored(
    action: Action,
    game_state: dict[str, Any],
) -> FactoredAction:
    """Convert an engine Action back to a FactoredAction.

    Primarily for logging/debugging.  Note that ReorderJokers and
    ReorderHand with complex permutations cannot be perfectly decomposed
    into a single swap — only adjacent swaps are representable.

    Parameters
    ----------
    action:
        Engine action dataclass.
    game_state:
        Current game state dict (needed for permutation→swap detection).

    Returns
    -------
    FactoredAction
        The factored representation.

    Raises
    ------
    ValueError
        If the action type is not recognized or a permutation cannot
        be represented as a single swap.
    """
    if isinstance(action, EnginePlayHand):
        return FactoredAction(
            action_type=ActionType.PlayHand,
            card_target=action.card_indices if action.card_indices else None,
        )

    if isinstance(action, EngineDiscard):
        return FactoredAction(
            action_type=ActionType.Discard,
            card_target=action.card_indices if action.card_indices else None,
        )

    if isinstance(action, EngineSelectBlind):
        return FactoredAction(action_type=ActionType.SelectBlind)

    if isinstance(action, EngineSkipBlind):
        return FactoredAction(action_type=ActionType.SkipBlind)

    if isinstance(action, EngineCashOut):
        return FactoredAction(action_type=ActionType.CashOut)

    if isinstance(action, EngineReroll):
        return FactoredAction(action_type=ActionType.Reroll)

    if isinstance(action, EngineNextRound):
        return FactoredAction(action_type=ActionType.NextRound)

    if isinstance(action, EngineSkipPack):
        return FactoredAction(action_type=ActionType.SkipPack)

    if isinstance(action, EngineBuyCard):
        return FactoredAction(
            action_type=ActionType.BuyCard,
            entity_target=action.shop_index,
        )

    if isinstance(action, EngineBuyAndUse):
        # Composite action: buy consumable + use immediately.
        # The factored space decomposes this into BuyCard + UseConsumable.
        # Map to BuyCard — the RL agent can UseConsumable in a later step.
        return FactoredAction(
            action_type=ActionType.BuyCard,
            entity_target=action.shop_index,
        )

    if isinstance(action, EngineSellCard):
        if action.area == "jokers":
            return FactoredAction(
                action_type=ActionType.SellJoker,
                entity_target=action.card_index,
            )
        if action.area == "consumables":
            return FactoredAction(
                action_type=ActionType.SellConsumable,
                entity_target=action.card_index,
            )
        raise ValueError(f"Unknown SellCard area: {action.area}")

    if isinstance(action, EngineUseConsumable):
        return FactoredAction(
            action_type=ActionType.UseConsumable,
            entity_target=action.card_index,
            card_target=action.target_indices,
        )

    if isinstance(action, EngineRedeemVoucher):
        return FactoredAction(
            action_type=ActionType.RedeemVoucher,
            entity_target=action.card_index,
        )

    if isinstance(action, EngineOpenBooster):
        return FactoredAction(
            action_type=ActionType.OpenBooster,
            entity_target=action.card_index,
        )

    if isinstance(action, EnginePickPackCard):
        return FactoredAction(
            action_type=ActionType.PickPackCard,
            entity_target=action.card_index,
            card_target=action.target_indices,
        )

    if isinstance(action, EngineSortHand):
        if action.mode == "rank":
            return FactoredAction(action_type=ActionType.SortHandRank)
        return FactoredAction(action_type=ActionType.SortHandSuit)

    if isinstance(action, EngineReorderJokers):
        if not action.new_order:
            # Marker from get_legal_actions — "joker reordering is available".
            # Map to SwapJokersLeft marker (entity_target=None), matching how
            # PlayHand(card_indices=()) → PlayHand(card_target=None).
            return FactoredAction(action_type=ActionType.SwapJokersLeft)
        return _permutation_to_swap(
            action.new_order,
            ActionType.SwapJokersLeft,
            ActionType.SwapJokersRight,
        )

    if isinstance(action, EngineReorderHand):
        if not action.new_order:
            return FactoredAction(action_type=ActionType.SwapHandLeft)
        return _permutation_to_swap(
            action.new_order,
            ActionType.SwapHandLeft,
            ActionType.SwapHandRight,
        )

    raise ValueError(f"Unknown engine action type: {type(action).__name__}")


def _decompose_permutation(perm: tuple[int, ...]) -> tuple[int, int]:
    """Find the first adjacent swap needed to sort a permutation.

    Uses a single pass of bubble sort: find the leftmost position ``i``
    where ``perm[i] > perm[i+1]`` and return ``(i, i+1)``.  If no such
    position exists, scan for any out-of-place element and swap it toward
    its target.

    Returns ``(lower_idx, higher_idx)`` of the swap to perform.
    """
    n = len(perm)
    # Bubble-sort pass: find leftmost inversion
    for i in range(n - 1):
        if perm[i] > perm[i + 1]:
            return (i, i + 1)
    # No inversion in bubble order — find first out-of-place element
    for i in range(n):
        if perm[i] != i:
            # Move it toward its target
            target = perm[i]
            if target > i:
                return (i, i + 1)
            else:
                return (i - 1, i)
    # Identity permutation — shouldn't happen, but return first pair
    return (0, 1)


def _permutation_to_swap(
    new_order: tuple[int, ...],
    left_type: ActionType,
    right_type: ActionType,
) -> FactoredAction:
    """Convert a permutation to the first adjacent swap needed to reach it.

    For single adjacent swaps, returns the exact swap.  For complex
    permutations (multi-element rearrangements), decomposes into the
    first adjacent swap of a bubble-sort pass toward the target order.

    The RL agent achieves arbitrary reorderings through repeated
    single-swap actions, matching the balatrobot ``rearrange`` API.

    Empty-tuple markers are handled by the caller before reaching here.
    """
    n = len(new_order)

    # Find positions that differ from identity
    diffs = [i for i in range(n) if new_order[i] != i]

    if len(diffs) == 0:
        # Identity permutation — no-op, return first valid swap as fallback
        if n >= 2:
            return FactoredAction(action_type=left_type, entity_target=1)
        raise ValueError("Single-element permutation cannot be swapped")

    if len(diffs) == 2:
        i, j = diffs
        if abs(i - j) == 1 and new_order[i] == j and new_order[j] == i:
            # Single adjacent swap — exact conversion
            return FactoredAction(action_type=left_type, entity_target=j)

    # Complex permutation: decompose into first adjacent swap
    lo, hi = _decompose_permutation(new_order)
    # SwapLeft with the higher index (the element at hi moves left)
    return FactoredAction(action_type=left_type, entity_target=hi)
