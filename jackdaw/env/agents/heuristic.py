"""Heuristic agent — port of smart_agent.py adapted to the FactoredAction interface.

This agent gets a PRIVILEGED view via ``info["raw_state"]`` to access the
full engine game_state dict.  It uses game knowledge (hand evaluation, joker
ratings, economy management) rather than learned features.  But it outputs
:class:`FactoredAction` instances, going through the same action space as
the RL agent.

This is intentional: the heuristic baseline validates that the action space
is expressive enough to encode good play, and provides a training target
for imitation learning warm-starts.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from jackdaw.env.action_space import (
    ActionMask,
    ActionType,
    FactoredAction,
)

# ---------------------------------------------------------------------------
# Hand scoring estimate (from smart_agent.py)
# ---------------------------------------------------------------------------

_HAND_BASE_SCORE: dict[str, tuple[int, int]] = {
    "Flush Five": (160, 16),
    "Flush House": (140, 14),
    "Five of a Kind": (120, 12),
    "Straight Flush": (100, 8),
    "Four of a Kind": (60, 7),
    "Full House": (40, 4),
    "Flush": (35, 4),
    "Straight": (30, 4),
    "Three of a Kind": (30, 3),
    "Two Pair": (20, 2),
    "Pair": (10, 2),
    "High Card": (5, 1),
}

_HAND_PRIORITY = [
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
]


def _num(val: Any, default: int | float = 0) -> int | float:
    if isinstance(val, (int, float)):
        return val
    return default


def _estimate_hand_score(
    played_cards: list[Any],
    jokers: list[Any],
    hand_levels: Any,
) -> tuple[str, float]:
    """Estimate score for a hand without RNG. Returns (hand_type, score)."""
    from jackdaw.engine.data.hands import HAND_BASE, HandType
    from jackdaw.engine.hand_eval import evaluate_hand

    result = evaluate_hand(played_cards, jokers)
    hand_type = result.detected_hand

    base_chips, base_mult = _HAND_BASE_SCORE.get(hand_type, (5, 1))
    level = 1
    if hand_levels and hand_type in hand_levels:
        hs = hand_levels[hand_type]
        level = getattr(hs, "level", 1)

    hd = HAND_BASE.get(HandType(hand_type))
    if hd:
        base_chips = hd.chips_at(level)
        base_mult = hd.mult_at(level)

    card_chips = 0
    for c in played_cards:
        if hasattr(c, "get_chip_bonus"):
            card_chips += c.get_chip_bonus()

    total_chips = base_chips + card_chips

    flat_chips = 0
    flat_mult = 0
    x_mult = 1.0

    for j in jokers:
        a = getattr(j, "ability", {})
        if not isinstance(a, dict):
            continue
        if getattr(j, "debuff", False):
            continue
        name = a.get("name", "")
        if name == "Blue Joker":
            flat_chips += _num(a.get("extra")) * 2
        elif name == "Stencil":
            x_mult *= max(1, _num(a.get("x_mult"), 1))
        elif name == "Steel Joker":
            x_mult *= max(1, 1 + _num(a.get("extra")) * 0.2)
        elif name in ("Joker", "Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker"):
            flat_mult += _num(a.get("t_mult")) + _num(a.get("extra"))
        elif name == "Jolly Joker" and hand_type == "Pair":
            flat_mult += _num(a.get("t_mult"), 8)
        elif name == "Zany Joker" and hand_type == "Three of a Kind":
            flat_mult += _num(a.get("t_mult"), 12)
        elif name == "Mad Joker" and hand_type == "Two Pair":
            flat_mult += _num(a.get("t_mult"), 10)
        elif name == "Crazy Joker" and hand_type == "Straight":
            flat_mult += _num(a.get("t_mult"), 12)
        elif name == "Droll Joker" and hand_type == "Flush":
            flat_mult += _num(a.get("t_mult"), 10)
        elif name == "Half Joker" and len(played_cards) <= 3:
            flat_mult += _num(a.get("extra"), 20)
        elif name == "Ride the Bus":
            flat_mult += _num(a.get("extra"))
        elif name == "Blackboard":
            extra = a.get("extra")
            if isinstance(extra, dict):
                x_mult *= _num(extra.get("x_mult"), 1)
        elif name == "The Duo" and hand_type in (
            "Pair",
            "Two Pair",
            "Full House",
            "Three of a Kind",
            "Four of a Kind",
            "Five of a Kind",
        ):
            x_mult *= 2
        elif name == "The Trio" and hand_type in (
            "Three of a Kind",
            "Full House",
            "Four of a Kind",
            "Five of a Kind",
        ):
            x_mult *= 2
        elif name == "The Family" and hand_type in ("Four of a Kind", "Five of a Kind"):
            x_mult *= 2
        elif _num(a.get("t_mult")) > 0:
            flat_mult += _num(a.get("t_mult"))
        elif _num(a.get("t_chips")) > 0:
            flat_chips += _num(a.get("t_chips"))
        if _num(a.get("x_mult")) > 1:
            x_mult *= _num(a.get("x_mult"))

    for j in jokers:
        ed = getattr(j, "edition", None)
        if not ed or not isinstance(ed, dict):
            continue
        if ed.get("foil"):
            flat_chips += 50
        if ed.get("holo"):
            flat_mult += 10
        if ed.get("polychrome"):
            x_mult *= 1.5

    total_mult = base_mult + flat_mult
    total_chips += flat_chips
    score = total_chips * total_mult * x_mult
    return hand_type, score


def _pick_best_hand_scored(
    hand_cards: list[Any],
    jokers: list[Any],
    hand_levels: Any,
) -> tuple[list[int], str, float]:
    """Pick best hand by estimated score. Returns (indices, hand_type, score)."""
    n = len(hand_cards)
    if n == 0:
        return [], "High Card", 0

    best_indices = list(range(min(5, n)))
    best_type = "High Card"
    best_score = 0.0

    for size in (min(5, n), min(4, n), min(3, n), min(2, n), 1):
        if size <= 0 or size > n:
            continue
        for combo in combinations(range(n), size):
            cards = [hand_cards[i] for i in combo]
            hand_type, score = _estimate_hand_score(cards, jokers, hand_levels)
            if score > best_score:
                best_score = score
                best_indices = list(combo)
                best_type = hand_type

    return best_indices, best_type, best_score


# ---------------------------------------------------------------------------
# Joker evaluation
# ---------------------------------------------------------------------------

_GOOD_JOKERS: set[str] = {
    "Steel Joker",
    "Hologram",
    "Obelisk",
    "The Idol",
    "Photograph",
    "Ancient Joker",
    "Loyalty Card",
    "Stencil",
    "Blackboard",
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    "Golden Joker",
    "Delayed Gratification",
    "Cloud 9",
    "Business Card",
    "Faceless Joker",
    "To the Moon",
    "Dusk",
    "Hack",
    "Sock and Buskin",
    "Hanging Chad",
    "Blue Joker",
    "Green Joker",
    "Red Card",
    "Spare Trousers",
    "Ride the Bus",
    "Runner",
    "Ice Cream",
    "Constellation",
    "Swashbuckler",
    "Joker Stencil",
    "Chaos the Clown",
}

_GREAT_JOKERS: set[str] = {
    "Steel Joker",
    "Hologram",
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    "Blackboard",
    "Stencil",
    "Blueprint",
    "Brainstorm",
}


def _joker_value(card: Any) -> int:
    a = getattr(card, "ability", {})
    if not isinstance(a, dict):
        return 10
    name = a.get("name", "")
    if name in _GREAT_JOKERS:
        return 90
    if name in _GOOD_JOKERS:
        return 60
    if a.get("x_mult", 0) > 1:
        return 70
    if a.get("t_mult", 0) >= 4:
        return 40
    if a.get("t_chips", 0) >= 20:
        return 30
    return 15


# ---------------------------------------------------------------------------
# Consumable targeting
# ---------------------------------------------------------------------------


def _pick_consumable_targets(
    card: Any,
    hand: list[Any],
) -> tuple[int, ...] | None:
    """Return target indices for a consumable, or None if no targets needed."""
    from jackdaw.engine.card import _resolve_center

    key = getattr(card, "center_key", "")
    try:
        center = _resolve_center(key)
    except (KeyError, Exception):
        return None

    cfg = center.get("config") or {}
    if isinstance(cfg, list):
        cfg = {}
    max_h = cfg.get("max_highlighted")
    if not max_h or not hand:
        return None

    min_h = cfg.get("min_highlighted", 1)
    n = min(max_h, len(hand))
    if n < min_h:
        return None

    effect = cfg.get("mod_conv") or center.get("effect", "")
    if effect in ("destroy",):
        ranked = sorted(
            range(len(hand)),
            key=lambda i: hand[i].get_chip_bonus() if hasattr(hand[i], "get_chip_bonus") else 0,
        )
        return tuple(sorted(ranked[:n]))

    return tuple(range(n))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card_set(card: Any) -> str:
    a = getattr(card, "ability", None)
    return a.get("set", "") if isinstance(a, dict) else ""


def _find_entity_index(
    at: ActionType,
    entity_masks: dict[int, np.ndarray],
    predicate: Any = None,
) -> int | None:
    """Find the first legal entity index, optionally filtered by predicate."""
    mask = entity_masks.get(at)
    if mask is None:
        return None
    legal = np.nonzero(mask)[0]
    if len(legal) == 0:
        return None
    if predicate is None:
        return int(legal[0])
    for idx in legal:
        if predicate(int(idx)):
            return int(idx)
    return None


# ---------------------------------------------------------------------------
# HeuristicAgent
# ---------------------------------------------------------------------------


class HeuristicAgent:
    """Heuristic baseline ported from smart_agent.py.

    Uses privileged access to ``info["raw_state"]`` for game knowledge,
    but outputs :class:`FactoredAction` through the standard action space.
    """

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        gs = info["raw_state"]
        phase = gs.get("phase")
        if isinstance(phase, str):
            from jackdaw.engine.actions import GamePhase

            phase = GamePhase(phase)

        if phase == GamePhase.BLIND_SELECT:
            return self._blind_select(gs, action_mask)
        if phase == GamePhase.SELECTING_HAND:
            return self._selecting_hand(gs, action_mask)
        if phase == GamePhase.ROUND_EVAL:
            return self._round_eval(gs, action_mask)
        if phase == GamePhase.SHOP:
            return self._shop(gs, action_mask)
        if phase == GamePhase.PACK_OPENING:
            return self._pack_opening(gs, action_mask)

        # Fallback: pick first legal type
        return self._fallback(action_mask)

    # -- Phase handlers -----------------------------------------------------

    def _blind_select(self, gs: dict, mask: ActionMask) -> FactoredAction:
        # Always select blind (smart_agent doesn't skip)
        if mask.type_mask[ActionType.SelectBlind]:
            return FactoredAction(action_type=ActionType.SelectBlind)
        return self._fallback(mask)

    def _selecting_hand(self, gs: dict, mask: ActionMask) -> FactoredAction:
        hand = gs.get("hand", [])
        jokers = gs.get("jokers", [])
        hand_levels = gs.get("hand_levels")
        cr = gs.get("current_round", {})
        hands_left = cr.get("hands_left", 4)
        discards_left = cr.get("discards_left", 0)

        # Use planet consumables immediately
        planet_action = self._use_planet(gs, mask)
        if planet_action is not None:
            return planet_action

        if not hand:
            return self._fallback(mask)

        # Evaluate best hand
        best_indices, best_type, best_score = _pick_best_hand_scored(hand, jokers, hand_levels)

        # Decide: discard or play?
        should_discard = False
        if discards_left > 0 and hands_left > 1 and mask.type_mask[ActionType.Discard]:
            hand_type_rank = _HAND_PRIORITY.index(best_type) if best_type in _HAND_PRIORITY else 11
            if hand_type_rank >= 11:
                should_discard = True
            elif hand_type_rank >= 10 and hands_left >= 3 and discards_left >= 2:
                should_discard = True

        if should_discard:
            best_set = set(best_indices)
            worst = [i for i in range(len(hand)) if i not in best_set]
            if worst:
                n = min(len(worst), 5)
                return FactoredAction(
                    action_type=ActionType.Discard,
                    card_target=tuple(sorted(worst[:n])),
                )

        # Play best hand
        if mask.type_mask[ActionType.PlayHand] and best_indices:
            return FactoredAction(
                action_type=ActionType.PlayHand,
                card_target=tuple(best_indices),
            )

        return self._fallback(mask)

    def _round_eval(self, gs: dict, mask: ActionMask) -> FactoredAction:
        # Use planets before cashing out
        planet_action = self._use_planet(gs, mask)
        if planet_action is not None:
            return planet_action

        if mask.type_mask[ActionType.CashOut]:
            return FactoredAction(action_type=ActionType.CashOut)
        return self._fallback(mask)

    def _shop(self, gs: dict, mask: ActionMask) -> FactoredAction:
        dollars = gs.get("dollars", 0)
        jokers = gs.get("jokers", [])
        joker_slots = gs.get("joker_slots", 5)
        consumables = gs.get("consumables", [])
        consumable_slots = gs.get("consumable_slots", 2)
        shop_cards = gs.get("shop_cards", [])
        cr = gs.get("current_round", {})
        rr = gs.get("round_resets", {})
        ante = rr.get("ante", 1)

        has_joker_room = len(jokers) < joker_slots

        # Economy: interest floor based on ante
        if ante <= 2 and len(jokers) < 3:
            interest_floor = 0
        elif ante <= 3:
            interest_floor = 5
        elif ante <= 5:
            interest_floor = 5
        else:
            interest_floor = 10
        spendable = max(0, dollars - interest_floor)

        reroll_cost = cr.get("reroll_cost", 5)
        free_rerolls = cr.get("free_rerolls", 0)
        need_jokers = has_joker_room and len(jokers) < 4

        # 1. Use owned planets
        planet_action = self._use_planet(gs, mask)
        if planet_action is not None:
            return planet_action

        # 2. Use tarots if consumable slots full
        if len(consumables) >= consumable_slots:
            tarot_action = self._use_tarot(gs, mask)
            if tarot_action is not None:
                return tarot_action

        # 3. Buy jokers
        best_joker_idx, best_joker_val = self._find_best_joker_buy(
            shop_cards, mask, dollars, spendable
        )

        if best_joker_idx is not None and need_jokers:
            return FactoredAction(
                action_type=ActionType.BuyCard,
                entity_target=best_joker_idx,
            )
        if best_joker_idx is not None and best_joker_val >= 25:
            return FactoredAction(
                action_type=ActionType.BuyCard,
                entity_target=best_joker_idx,
            )

        # 4. Reroll to find jokers
        shop_has_joker = any(
            _card_set(shop_cards[i]) == "Joker"
            for i in range(len(shop_cards))
            if i < len(shop_cards)
        )
        if (
            need_jokers
            and not shop_has_joker
            and len(jokers) < 3
            and mask.type_mask[ActionType.Reroll]
        ):
            if free_rerolls > 0 or reroll_cost <= spendable:
                return FactoredAction(action_type=ActionType.Reroll)

        # 5. Redeem vouchers
        if mask.type_mask[ActionType.RedeemVoucher]:
            vouchers = gs.get("shop_vouchers", [])
            voucher_idx = _find_entity_index(
                ActionType.RedeemVoucher,
                mask.entity_masks,
                lambda i: i < len(vouchers) and vouchers[i].cost <= spendable + interest_floor,
            )
            if voucher_idx is not None:
                return FactoredAction(
                    action_type=ActionType.RedeemVoucher,
                    entity_target=voucher_idx,
                )

        # 6. Open booster packs
        if mask.type_mask[ActionType.OpenBooster]:
            boosters = gs.get("shop_boosters", [])
            booster_idx = _find_entity_index(
                ActionType.OpenBooster,
                mask.entity_masks,
                lambda i: i < len(boosters) and boosters[i].cost <= spendable,
            )
            if booster_idx is not None:
                return FactoredAction(
                    action_type=ActionType.OpenBooster,
                    entity_target=booster_idx,
                )

        # 7. Buy planets if we have consumable room
        if mask.type_mask[ActionType.BuyCard] and len(consumables) < consumable_slots:
            planet_idx = _find_entity_index(
                ActionType.BuyCard,
                mask.entity_masks,
                lambda i: (
                    i < len(shop_cards)
                    and _card_set(shop_cards[i]) == "Planet"
                    and shop_cards[i].cost <= spendable
                ),
            )
            if planet_idx is not None:
                return FactoredAction(
                    action_type=ActionType.BuyCard,
                    entity_target=planet_idx,
                )

        # 8. Buy tarots if we have consumable room
        if mask.type_mask[ActionType.BuyCard] and len(consumables) < consumable_slots:
            tarot_idx = _find_entity_index(
                ActionType.BuyCard,
                mask.entity_masks,
                lambda i: (
                    i < len(shop_cards)
                    and _card_set(shop_cards[i]) == "Tarot"
                    and shop_cards[i].cost <= spendable
                ),
            )
            if tarot_idx is not None:
                return FactoredAction(
                    action_type=ActionType.BuyCard,
                    entity_target=tarot_idx,
                )

        # 9. Sell weak jokers for great ones (late game)
        if not has_joker_room and ante >= 3 and mask.type_mask[ActionType.SellJoker]:
            great_available = any(
                i < len(shop_cards)
                and _card_set(shop_cards[i]) == "Joker"
                and _joker_value(shop_cards[i]) >= 80
                for i in range(len(shop_cards))
            )
            if great_available:
                worst_idx = None
                worst_val = 999
                sell_mask = mask.entity_masks.get(ActionType.SellJoker)
                if sell_mask is not None:
                    for idx in np.nonzero(sell_mask)[0]:
                        idx = int(idx)
                        if idx < len(jokers):
                            val = _joker_value(jokers[idx])
                            if val < worst_val:
                                worst_val = val
                                worst_idx = idx
                if worst_idx is not None and worst_val < 50:
                    return FactoredAction(
                        action_type=ActionType.SellJoker,
                        entity_target=worst_idx,
                    )

        # 10. Sell consumables to make room
        if len(consumables) >= consumable_slots and mask.type_mask[ActionType.SellConsumable]:
            sell_idx = _find_entity_index(ActionType.SellConsumable, mask.entity_masks)
            if sell_idx is not None:
                return FactoredAction(
                    action_type=ActionType.SellConsumable,
                    entity_target=sell_idx,
                )

        # 11. Free rerolls
        if free_rerolls > 0 and mask.type_mask[ActionType.Reroll]:
            return FactoredAction(action_type=ActionType.Reroll)

        # 12. Leave shop
        if mask.type_mask[ActionType.NextRound]:
            return FactoredAction(action_type=ActionType.NextRound)

        return self._fallback(mask)

    def _pack_opening(self, gs: dict, mask: ActionMask) -> FactoredAction:
        pack_cards = gs.get("pack_cards", [])
        jokers = gs.get("jokers", [])
        joker_slots = gs.get("joker_slots", 5)
        consumables = gs.get("consumables", [])
        consumable_slots = gs.get("consumable_slots", 2)
        has_joker_room = len(jokers) < joker_slots
        has_consumable_room = len(consumables) < consumable_slots

        if mask.type_mask[ActionType.PickPackCard]:
            pick_mask = mask.entity_masks.get(ActionType.PickPackCard)
            if pick_mask is not None:
                best_idx = None
                best_priority = -1

                for idx in np.nonzero(pick_mask)[0]:
                    idx = int(idx)
                    if idx >= len(pack_cards):
                        continue
                    card = pack_cards[idx]
                    cset = _card_set(card)

                    if cset == "Joker":
                        p = (40 + _joker_value(card)) if has_joker_room else 1
                    elif cset == "Planet":
                        p = 70
                    elif cset == "Spectral":
                        p = 50 if has_consumable_room else 20
                    elif cset == "Tarot":
                        p = 30 if has_consumable_room else 10
                    else:
                        p = 5

                    if p > best_priority:
                        best_priority = p
                        best_idx = idx

                if best_idx is not None:
                    # Check if this card needs targets
                    card = pack_cards[best_idx]
                    targets = _pick_consumable_targets(card, gs.get("hand", []))
                    card_target = targets if targets is not None else None
                    return FactoredAction(
                        action_type=ActionType.PickPackCard,
                        entity_target=best_idx,
                        card_target=card_target,
                    )

        if mask.type_mask[ActionType.SkipPack]:
            return FactoredAction(action_type=ActionType.SkipPack)

        return self._fallback(mask)

    # -- Shared helpers -----------------------------------------------------

    def _use_planet(self, gs: dict, mask: ActionMask) -> FactoredAction | None:
        """Try to use a planet consumable. Returns FactoredAction or None."""
        if not mask.type_mask[ActionType.UseConsumable]:
            return None
        consumables = gs.get("consumables", [])
        use_mask = mask.entity_masks.get(ActionType.UseConsumable)
        if use_mask is None:
            return None

        for idx in np.nonzero(use_mask)[0]:
            idx = int(idx)
            if idx < len(consumables) and _card_set(consumables[idx]) == "Planet":
                return FactoredAction(
                    action_type=ActionType.UseConsumable,
                    entity_target=idx,
                )
        return None

    def _use_tarot(self, gs: dict, mask: ActionMask) -> FactoredAction | None:
        """Try to use a tarot consumable with targets. Returns FactoredAction or None."""
        if not mask.type_mask[ActionType.UseConsumable]:
            return None
        consumables = gs.get("consumables", [])
        hand = gs.get("hand", [])
        use_mask = mask.entity_masks.get(ActionType.UseConsumable)
        if use_mask is None:
            return None

        for idx in np.nonzero(use_mask)[0]:
            idx = int(idx)
            if idx < len(consumables) and _card_set(consumables[idx]) == "Tarot":
                card = consumables[idx]
                targets = _pick_consumable_targets(card, hand)
                return FactoredAction(
                    action_type=ActionType.UseConsumable,
                    entity_target=idx,
                    card_target=targets,
                )
        return None

    def _find_best_joker_buy(
        self,
        shop_cards: list,
        mask: ActionMask,
        dollars: int,
        spendable: int,
    ) -> tuple[int | None, int]:
        """Find the best joker to buy from the shop. Returns (index, value)."""
        if not mask.type_mask[ActionType.BuyCard]:
            return None, 0

        buy_mask = mask.entity_masks.get(ActionType.BuyCard)
        if buy_mask is None:
            return None, 0

        best_idx = None
        best_val = 0

        for idx in np.nonzero(buy_mask)[0]:
            idx = int(idx)
            if idx >= len(shop_cards):
                continue
            card = shop_cards[idx]
            if _card_set(card) == "Joker" and card.cost <= spendable:
                val = _joker_value(card)
                if val > best_val:
                    best_val = val
                    best_idx = idx

        return best_idx, best_val

    def _fallback(self, mask: ActionMask) -> FactoredAction:
        """Pick the first legal action type as a fallback."""
        legal_types = np.nonzero(mask.type_mask)[0]
        if len(legal_types) == 0:
            return FactoredAction(action_type=ActionType.SelectBlind)
        return FactoredAction(action_type=int(legal_types[0]))
