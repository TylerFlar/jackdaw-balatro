"""Smarter validation agent that aims to survive deep into late-game.

Key improvements over the basic validation agent:
- Score-aware hand selection: estimates actual chips×mult, not just hand type rank
- Blind-aware discarding: only discards when estimated score can't beat the blind
- Economy management: preserves $5 interest floors, avoids wasteful spending
- Joker evaluation: prioritizes mult/chip jokers, avoids filling slots with junk
- Consumable targeting: smarter tarot/planet usage based on hand level priorities
- Hand-type focus: concentrates planet upgrades on 1-2 hand types for scaling

It plays a :class:`jackdaw.core.Game`: ``smart_agent(game, game.legal_actions())``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any

from jackdaw.engine.actions import (
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    Reroll,
    SelectBlind,
    SellCard,
    SkipPack,
    UseConsumable,
)
from jackdaw.engine.data import load

if TYPE_CHECKING:
    from jackdaw.core import CardView, Game, HandLevel

# ---------------------------------------------------------------------------
# Hand scoring estimate (fast, no RNG — for decision-making only)
# ---------------------------------------------------------------------------

# Hand type priority order (index 0 = best)
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

_RANK_CHIPS = {"Jack": 10, "Queen": 10, "King": 10, "Ace": 11}

# Jokers with a flat +Mult whatever the hand
_FLAT_MULT = {
    "j_joker",
    "j_greedy_joker",
    "j_lusty_joker",
    "j_wrathful_joker",
    "j_gluttenous_joker",
}
# Jokers whose +Mult needs a hand type
_TYPE_MULT = {
    "j_jolly": ("Pair", 8),
    "j_zany": ("Three of a Kind", 12),
    "j_mad": ("Two Pair", 10),
    "j_crazy": ("Straight", 12),
    "j_droll": ("Flush", 10),
}
# Jokers with X2 Mult when the hand contains a type
_TYPE_XMULT = {
    "j_duo": ("Pair", "Two Pair", "Full House", "Three of a Kind", "Four of a Kind",
              "Five of a Kind"),
    "j_trio": ("Three of a Kind", "Full House", "Four of a Kind", "Five of a Kind"),
    "j_family": ("Four of a Kind", "Five of a Kind"),
}  # fmt: skip


def _num(val: Any, default: float = 0) -> float:
    """Safely extract a numeric value from ability fields."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return val
    return default


def _chip_bonus(card: CardView) -> float:
    """``Card:get_chip_bonus``: the rank's chips (none for a Stone Card) plus the bonuses."""
    if card.debuff:
        return 0
    a = card.ability
    bonus = _num(a.get("bonus")) + _num(a.get("perma_bonus"))
    if card.key == "m_stone" or card.rank is None:
        return bonus
    return _RANK_CHIPS.get(card.rank, _num(int(card.rank) if card.rank.isdigit() else 0)) + bonus


def _estimate_hand_score(
    game: Game,
    positions: Sequence[int],
    hand: Sequence[CardView],
    jokers: Sequence[CardView],
    hand_levels: Mapping[str, HandLevel],
) -> tuple[str, float]:
    """Estimate score for a hand without RNG.  Returns (hand_type, score)."""
    found = game.scoring_cards(positions)
    if found is None:
        return "High Card", 0.0
    hand_type, scoring = found

    # Base chips/mult at the hand's level, plus the scoring cards' chips
    level = hand_levels.get(hand_type)
    base_chips, base_mult = (level.chips, level.mult) if level is not None else (5, 1)
    total_chips = base_chips + sum(_chip_bonus(hand[i]) for i in scoring)

    # Estimate joker contributions (rough)
    flat_chips = 0.0
    flat_mult = 0.0
    x_mult = 1.0
    for j in jokers:
        if j.debuff:
            continue
        a = j.ability
        key = j.key
        if key == "j_blue_joker":
            flat_chips += _num(a.get("extra")) * 2  # rough estimate
        elif key == "j_stencil":
            x_mult *= max(1, _num(a.get("x_mult"), 1))
        elif key == "j_steel_joker":
            x_mult *= max(1, 1 + _num(a.get("extra")) * 0.2)
        elif key in _FLAT_MULT:
            flat_mult += _num(a.get("t_mult")) + _num(a.get("extra"))
        elif key in _TYPE_MULT and hand_type == _TYPE_MULT[key][0]:
            flat_mult += _num(a.get("t_mult"), _TYPE_MULT[key][1])
        elif key == "j_half" and len(positions) <= 3:
            flat_mult += _num(a.get("extra"), 20)
        elif key == "j_ride_the_bus":
            flat_mult += _num(a.get("extra"))
        elif key == "j_blackboard":
            extra = a.get("extra")
            if isinstance(extra, Mapping):
                x_mult *= _num(extra.get("x_mult"), 1)
        elif key in _TYPE_XMULT and hand_type in _TYPE_XMULT[key]:
            x_mult *= 2
        # Generic: if joker has t_mult or t_chips
        elif _num(a.get("t_mult")) > 0:
            flat_mult += _num(a.get("t_mult"))
        elif _num(a.get("t_chips")) > 0:
            flat_chips += _num(a.get("t_chips"))
        # x_mult jokers
        if _num(a.get("x_mult")) > 1:
            x_mult *= _num(a.get("x_mult"))

    # Edition bonuses on jokers
    for j in jokers:
        if j.edition == "foil":
            flat_chips += 50
        elif j.edition == "holo":
            flat_mult += 10
        elif j.edition == "polychrome":
            x_mult *= 1.5

    total_mult = base_mult + flat_mult
    total_chips += flat_chips
    return hand_type, total_chips * total_mult * x_mult


def _forced_card(hand: Sequence[CardView]) -> int | None:
    """Cerulean Bell's forced card, which every play and discard must include."""
    return next((i for i, c in enumerate(hand) if c.ability.get("forced_selection")), None)


def _pick_best_hand_scored(
    game: Game,
    hand: Sequence[CardView],
    jokers: Sequence[CardView],
    hand_levels: Mapping[str, HandLevel],
) -> tuple[list[int], str, float]:
    """Pick best hand by estimated score.  Returns (indices, hand_type, score)."""
    n = len(hand)
    if n == 0:
        return [], "High Card", 0

    forced = _forced_card(hand)
    best_indices = list(range(min(5, n)))
    if forced is not None and forced not in best_indices:
        best_indices = sorted(best_indices[:4] + [forced])
    best_type = "High Card"
    best_score = 0.0

    # Try all 5-card combos, plus smaller sizes (3,4) for types like Three of a Kind
    for size in (min(5, n), min(4, n), min(3, n), min(2, n), 1):
        if size <= 0 or size > n:
            continue
        for combo in combinations(range(n), size):
            if forced is not None and forced not in combo:
                continue
            hand_type, score = _estimate_hand_score(game, combo, hand, jokers, hand_levels)
            if score > best_score:
                best_score = score
                best_indices = list(combo)
                best_type = hand_type

    return best_indices, best_type, best_score


# ---------------------------------------------------------------------------
# Joker evaluation for purchasing decisions
# ---------------------------------------------------------------------------

# Jokers roughly ranked by late-game value
_GOOD_JOKERS: set[str] = {
    # xMult scaling
    "j_steel_joker",
    "j_hologram",
    "j_obelisk",
    "j_idol",
    "j_photograph",
    "j_ancient",
    "j_loyalty_card",
    "j_stencil",
    "j_blackboard",
    # Pair/type boosters
    "j_duo",
    "j_trio",
    "j_family",
    "j_order",
    "j_tribe",
    # Economy
    "j_golden",
    "j_delayed_grat",
    "j_cloud_9",
    "j_business",
    "j_faceless",
    "j_to_the_moon",
    # Retrigger / copy
    "j_dusk",
    "j_hack",
    "j_sock_and_buskin",
    "j_hanging_chad",
    # Chip/mult scaling
    "j_blue_joker",
    "j_green_joker",
    "j_red_card",
    "j_trousers",
    "j_ride_the_bus",
    "j_runner",
    "j_ice_cream",
    "j_constellation",
    "j_swashbuckler",
    # Utility
    "j_chaos",
}

_GREAT_JOKERS: set[str] = {
    "j_steel_joker",
    "j_hologram",
    "j_duo",
    "j_trio",
    "j_family",
    "j_order",
    "j_tribe",
    "j_blackboard",
    "j_stencil",
    "j_blueprint",
    "j_brainstorm",
}


def _card_set(card: CardView) -> str:
    """The card's set (``Joker``, ``Tarot``, ``Planet``...); ``""`` for playing cards."""
    center = load("centers").get(card.key, {})
    return "" if card.front is not None else str(center.get("set") or "")


def _joker_value(card: CardView) -> int:
    """Rate a joker 0-100 for purchase priority."""
    if card.key in _GREAT_JOKERS:
        return 90
    if card.key in _GOOD_JOKERS:
        return 60
    a = card.ability
    # Any joker with x_mult
    if _num(a.get("x_mult")) > 1:
        return 70
    # Any joker with decent flat mult
    if _num(a.get("t_mult")) >= 4:
        return 40
    if _num(a.get("t_chips")) >= 20:
        return 30
    return 15


# ---------------------------------------------------------------------------
# Consumable targeting
# ---------------------------------------------------------------------------


def _pick_consumable_targets(
    game: Game, area: str, index: int, hand_size: int
) -> tuple[int, ...] | None:
    """Return target indices for a consumable, or None if no targets needed."""
    min_h, max_h, needs = game._g.consumable_target_info(area, index)
    if not needs or not hand_size:
        return None
    n = min(max_h, hand_size)
    if n < min_h:
        return None
    # Target the first N cards (simplest valid choice)
    return tuple(range(n))


# ---------------------------------------------------------------------------
# The smart agent
# ---------------------------------------------------------------------------


def smart_agent(game: Game, legal_actions: list[Any]) -> Any:
    """Smart agent targeting deep runs (ante 8+)."""
    from jackdaw.env.game_interface import game_state

    phase = game.phase

    def use_planet() -> Any:
        consumables = game.consumables
        for a in legal_actions:
            if isinstance(a, UseConsumable) and a.card_index < len(consumables):
                if _card_set(consumables[a.card_index]) == "Planet":
                    return a
        return None

    # -- BLIND_SELECT --
    if phase == GamePhase.BLIND_SELECT:
        for a in legal_actions:
            if isinstance(a, SelectBlind):
                return a
        return legal_actions[0]

    # -- SELECTING_HAND --
    if phase == GamePhase.SELECTING_HAND:
        hand = game.hand
        hands_left = game.hands_left
        discards_left = game.discards_left

        # Use planet consumables immediately
        planet = use_planet()
        if planet is not None:
            return planet

        if not hand:
            return legal_actions[0]

        # Evaluate best hand
        best_indices, best_type, _ = _pick_best_hand_scored(
            game, hand, game.jokers, game.hand_levels
        )

        # Decide: discard or play?
        # Only discard when hand is truly bad (High Card) AND we have
        # enough hands+discards to recover.  Being too eager to discard
        # wastes actions and can lose rounds.
        should_discard = False
        if discards_left > 0 and hands_left > 1:
            hand_type_rank = _HAND_PRIORITY.index(best_type) if best_type in _HAND_PRIORITY else 11
            if hand_type_rank >= 11:  # High Card only — discard
                should_discard = True
            elif hand_type_rank >= 10 and hands_left >= 3 and discards_left >= 2:
                # Pair with plenty of room — try for better
                should_discard = True

        if should_discard:
            forced = _forced_card(hand)
            best_set = set(best_indices)
            worst = [i for i in range(len(hand)) if i not in best_set and i != forced]
            if worst:
                chosen = worst[:5] if forced is None else worst[:4] + [forced]
                for a in legal_actions:
                    if isinstance(a, Discard) and not a.card_indices:
                        return Discard(card_indices=tuple(sorted(chosen)))

        # Play best hand
        for a in legal_actions:
            if isinstance(a, PlayHand) and not a.card_indices:
                return PlayHand(card_indices=tuple(best_indices))

        return legal_actions[0]

    # -- ROUND_EVAL --
    if phase == GamePhase.ROUND_EVAL:
        # Use planet consumables before cashing out
        planet = use_planet()
        if planet is not None:
            return planet
        for a in legal_actions:
            if isinstance(a, CashOut):
                return a
        return legal_actions[0]

    # -- SHOP --
    if phase == GamePhase.SHOP:
        state = game_state(game)
        dollars = game.dollars
        jokers = game.jokers
        consumables = game.consumables
        shop_cards = game.shop_jokers
        ante = game.ante

        has_joker_room = len(jokers) < state.joker_slots
        # Economy: in early game spend aggressively on jokers,
        # then preserve $5 interest floors once we have a build.
        if ante <= 2 and len(jokers) < 3:
            interest_floor = 0  # spend everything to build joker slots
        elif ante <= 5:
            interest_floor = 5  # one interest tier
        else:
            interest_floor = 10  # two interest tiers in late game
        spendable = max(0, dollars - interest_floor)

        buy_actions = [a for a in legal_actions if isinstance(a, BuyCard)]
        current_round = game.to_dict()["current_round"]
        reroll_cost = current_round["reroll_cost"]
        free_rerolls = current_round["free_rerolls"]
        need_jokers = has_joker_room and len(jokers) < 4

        # Check if shop has affordable jokers
        shop_has_joker = False
        best_joker_action = None
        best_joker_val = 0
        for a in buy_actions:
            if a.shop_index < len(shop_cards):
                card = shop_cards[a.shop_index]
                if _card_set(card) == "Joker" and card.cost <= dollars:
                    shop_has_joker = True
                    val = _joker_value(card)
                    if val > best_joker_val and card.cost <= spendable:
                        best_joker_val = val
                        best_joker_action = a

        # 1. Use owned planets first (free value)
        planet = use_planet()
        if planet is not None:
            return planet

        # 2. Use owned tarots if consumable slots are full
        if len(consumables) >= state.consumable_slots:
            for a in legal_actions:
                if isinstance(a, UseConsumable) and a.card_index < len(consumables):
                    if _card_set(consumables[a.card_index]) == "Tarot":
                        targets = _pick_consumable_targets(
                            game, "consumables", a.card_index, len(game.hand)
                        )
                        if targets is not None:
                            return UseConsumable(card_index=a.card_index, target_indices=targets)
                        return a

        # 3. Buy jokers — TOP PRIORITY when building roster
        if best_joker_action and need_jokers:
            return best_joker_action
        if best_joker_action and best_joker_val >= 25:
            return best_joker_action

        # 4. Reroll to find jokers when roster is thin
        if need_jokers and not shop_has_joker and len(jokers) < 3:
            if free_rerolls > 0 or reroll_cost <= spendable:
                for a in legal_actions:
                    if isinstance(a, Reroll):
                        return a

        # 5. Redeem vouchers (strong persistent upgrades)
        vouchers = game.shop_vouchers
        for a in legal_actions:
            if isinstance(a, RedeemVoucher) and a.card_index < len(vouchers):
                if vouchers[a.card_index].cost <= spendable + interest_floor:
                    return a

        # 6. Open booster packs (good value, may contain jokers)
        boosters = game.shop_boosters
        for a in legal_actions:
            if isinstance(a, OpenBooster) and a.card_index < len(boosters):
                if boosters[a.card_index].cost <= spendable:
                    return a

        # 7-8. Buy planets, then tarots, if we have consumable slots
        if buy_actions and len(consumables) < state.consumable_slots:
            for wanted in ("Planet", "Tarot"):
                for a in buy_actions:
                    if a.shop_index < len(shop_cards):
                        card = shop_cards[a.shop_index]
                        if _card_set(card) == wanted and card.cost <= spendable:
                            return a

        # 9. Sell weak jokers to make room for better ones (late game)
        if not has_joker_room and ante >= 3:
            great_in_shop = any(
                a.shop_index < len(shop_cards)
                and _card_set(shop_cards[a.shop_index]) == "Joker"
                and _joker_value(shop_cards[a.shop_index]) >= 80
                for a in buy_actions
            )
            if great_in_shop:
                worst_val = 999
                worst_action = None
                for a in legal_actions:
                    if not isinstance(a, SellCard) or a.area != "jokers":
                        continue
                    if a.card_index < len(jokers):
                        val = _joker_value(jokers[a.card_index])
                        if val < worst_val:
                            worst_val = val
                            worst_action = a
                if worst_action and worst_val < 50:
                    return worst_action

        # 10. Sell consumables to make room if needed
        if len(consumables) >= state.consumable_slots:
            for a in legal_actions:
                if isinstance(a, SellCard) and a.area == "consumables":
                    return a

        # 11. Free rerolls
        if free_rerolls > 0:
            for a in legal_actions:
                if isinstance(a, Reroll):
                    return a

        # 12. Leave shop
        for a in legal_actions:
            if isinstance(a, NextRound):
                return a
        return legal_actions[0]

    # -- PACK_OPENING --
    if phase == GamePhase.PACK_OPENING:
        state = game_state(game)
        pack_cards = game.pack_cards
        has_joker_room = len(game.jokers) < state.joker_slots
        has_consumable_room = len(game.consumables) < state.consumable_slots

        best_pick = None
        best_priority = -1

        for a in legal_actions:
            if not isinstance(a, PickPackCard) or a.card_index >= len(pack_cards):
                continue
            card = pack_cards[a.card_index]
            cset = _card_set(card)

            if cset == "Joker":
                p = 40 + _joker_value(card) if has_joker_room else 1  # Skip jokers if full
            elif cset == "Planet":
                p = 70  # Planets are always good (auto-used or leveled)
            elif cset == "Spectral":
                p = 50 if has_consumable_room else 20
            elif cset == "Tarot":
                p = 30 if has_consumable_room else 10
            else:
                p = 5  # Playing cards — rarely worth taking

            if p > best_priority:
                best_priority = p
                best_pick = a

        if best_pick is not None:
            targets = _pick_consumable_targets(
                game, "pack_cards", best_pick.card_index, len(game.hand)
            )
            if targets is not None:
                return PickPackCard(card_index=best_pick.card_index, target_indices=targets)
            return best_pick

        for a in legal_actions:
            if isinstance(a, SkipPack):
                return a
        return legal_actions[0]

    # Fallback
    return legal_actions[0]
