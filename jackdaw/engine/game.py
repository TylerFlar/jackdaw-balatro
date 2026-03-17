"""Game step function — the heart of the simulator.

Applies a player :data:`~jackdaw.engine.actions.Action` to the game state
and advances the phase.  Each handler validates legality, executes the
action, and transitions to the next phase as needed.

Usage::

    from jackdaw.engine.game import step
    from jackdaw.engine.actions import SelectBlind

    game_state = initialize_run("b_red", 1, "SEED")
    game_state["phase"] = GamePhase.BLIND_SELECT
    game_state["blind_on_deck"] = "Small"

    game_state = step(game_state, SelectBlind())
    assert game_state["phase"] == GamePhase.SELECTING_HAND
"""

from __future__ import annotations

import math
from typing import Any

from jackdaw.engine.actions import (
    Action,
    BuyAndUse,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    Reroll,
    RedeemVoucher,
    ReorderJokers,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    UseConsumable,
)


class IllegalActionError(Exception):
    """Raised when an action is not valid in the current game state."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def step(game_state: dict[str, Any], action: Action) -> dict[str, Any]:
    """Apply *action* to *game_state* in-place and return it.

    Dispatches based on action type.  Raises :class:`IllegalActionError`
    if the action is not valid in the current phase.
    """
    match action:
        case SelectBlind():
            return _handle_select_blind(game_state)
        case SkipBlind():
            return _handle_skip_blind(game_state)
        case PlayHand(card_indices=indices):
            return _handle_play_hand(game_state, indices)
        case Discard(card_indices=indices):
            return _handle_discard(game_state, indices)
        case CashOut():
            return _handle_cash_out(game_state)
        case BuyCard(shop_index=idx):
            return _handle_buy_card(game_state, idx)
        case BuyAndUse(shop_index=idx, target_indices=targets):
            return _handle_buy_and_use(game_state, idx, targets)
        case SellCard(area=area, card_index=idx):
            return _handle_sell_card(game_state, area, idx)
        case UseConsumable(card_index=idx, target_indices=targets):
            return _handle_use_consumable(game_state, idx, targets)
        case RedeemVoucher(card_index=idx):
            return _handle_redeem_voucher(game_state, idx)
        case OpenBooster(card_index=idx):
            return _handle_open_booster(game_state, idx)
        case PickPackCard(card_index=idx):
            return _handle_pick_pack_card(game_state, idx)
        case SkipPack():
            return _handle_skip_pack(game_state)
        case Reroll():
            return _handle_reroll(game_state)
        case NextRound():
            return _handle_next_round(game_state)
        case SortHand(mode=mode):
            return _handle_sort_hand(game_state, mode)
        case ReorderJokers(new_order=order):
            return _handle_reorder_jokers(game_state, order)
        case _:
            raise IllegalActionError(f"Unknown action type: {type(action).__name__}")


# ---------------------------------------------------------------------------
# Phase validation helper
# ---------------------------------------------------------------------------


def _require_phase(gs: dict[str, Any], *phases: GamePhase) -> GamePhase:
    """Assert the current phase is one of *phases* and return it."""
    raw = gs.get("phase")
    phase = GamePhase(raw) if isinstance(raw, str) else raw
    if phase not in phases:
        raise IllegalActionError(
            f"Action not valid in phase {phase!r} (expected {phases})"
        )
    return phase


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_select_blind(gs: dict[str, Any]) -> dict[str, Any]:
    """Accept the current blind and start the round.

    Full sequence matching ``game.lua`` select_blind → ``blind.lua``
    ``set_blind`` → ``state_events.lua`` ``new_round``:

    1. Create Blind from blind_choices
    2. Fire joker ``setting_blind`` context (Chicot, Madness, Burglar,
       Marble Joker, Riff-raff, Cartomancer)
    3. Process setting_blind side-effects
    4. Apply boss blind set-time effects (Water, Needle, Manacle,
       Amber Acorn) + debuff playing cards
    5. Call ``start_round`` (reset counters, targeting cards)
    6. Draw hand from deck
    7. Set phase → SELECTING_HAND
    """
    _require_phase(gs, GamePhase.BLIND_SELECT)

    from jackdaw.engine.blind import Blind
    from jackdaw.engine.run_init import start_round

    blind_on_deck = gs.get("blind_on_deck", "Small")
    rr = gs["round_resets"]
    blind_key = rr["blind_choices"].get(blind_on_deck, "bl_small")

    # ------------------------------------------------------------------
    # 1. Create the active Blind
    # ------------------------------------------------------------------
    ante = rr["ante"]
    scaling = gs.get("modifiers", {}).get("scaling", 1)
    ante_scaling = gs["starting_params"].get("ante_scaling", 1.0)
    no_reward = gs.get("modifiers", {}).get("no_blind_reward", {})
    blind = Blind.create(
        blind_key,
        ante,
        scaling=scaling,
        ante_scaling=ante_scaling,
        no_blind_reward=bool(no_reward.get(blind_on_deck)),
    )
    gs["blind"] = blind
    gs["chips"] = 0
    rr["blind_states"][blind_on_deck] = "Current"
    rr["blind"] = blind

    # ------------------------------------------------------------------
    # 2. Fire joker setting_blind context
    # ------------------------------------------------------------------
    jokers: list = gs.get("jokers", [])
    setting_mutations = _fire_setting_blind(gs, jokers, blind)

    # ------------------------------------------------------------------
    # 3. Process setting_blind side-effects
    # ------------------------------------------------------------------
    _apply_setting_blind_mutations(gs, setting_mutations, jokers)

    # ------------------------------------------------------------------
    # 4. Start round (reset counters, targeting cards)
    #    Must run BEFORE boss effects so that The Water/Needle/etc.
    #    can decrement from the freshly-set values.
    # ------------------------------------------------------------------
    start_round(gs)

    # ------------------------------------------------------------------
    # 5. Boss blind set-time effects (blind.lua:157-209)
    #    Fires after new_round in Lua (set_blind is called after new_round).
    # ------------------------------------------------------------------
    if blind.boss and not blind.disabled:
        _apply_boss_blind_effects(gs, blind)

    # Debuff playing cards based on boss blind
    deck: list = gs.get("deck", [])
    pareidolia = any(
        getattr(j, "center_key", None) == "j_pareidolia"
        and not getattr(j, "debuff", False)
        for j in jokers
    )
    for card in deck:
        blind.debuff_card(card, pareidolia=pareidolia)

    # ------------------------------------------------------------------
    # 6. Draw hand from deck
    # ------------------------------------------------------------------
    _draw_hand(gs)
    # Debuff hand cards too (they were drawn from the deck)
    for card in gs.get("hand", []):
        blind.debuff_card(card, pareidolia=pareidolia)

    # ------------------------------------------------------------------
    # 7. Phase → SELECTING_HAND
    # ------------------------------------------------------------------
    gs["phase"] = GamePhase.SELECTING_HAND
    return gs


def _handle_skip_blind(gs: dict[str, Any]) -> dict[str, Any]:
    """Skip the current blind (Small or Big) and advance.

    Full sequence matching ``button_callbacks.lua:2740-2775``:

    1. Validate: not Boss
    2. Increment skips
    3. Award skip tag from ``blind_tags``
    4. Fire tag ``apply('immediate')`` for immediate tags
    5. Fire joker ``skip_blind`` context (Throwback tracking)
    6. Advance ``blind_on_deck``: Small→Big, Big→Boss
    7. Check Double Tag: if active, duplicate the just-awarded tag
    8. Phase stays BLIND_SELECT
    """
    _require_phase(gs, GamePhase.BLIND_SELECT)

    blind_on_deck = gs.get("blind_on_deck", "Small")
    if blind_on_deck not in ("Small", "Big"):
        raise IllegalActionError("Cannot skip Boss blind")

    rr = gs["round_resets"]

    # ------------------------------------------------------------------
    # 1-2. Increment skips
    # ------------------------------------------------------------------
    gs["skips"] = gs.get("skips", 0) + 1
    rr["blind_states"][blind_on_deck] = "Skipped"

    # ------------------------------------------------------------------
    # 3. Award skip tag
    # ------------------------------------------------------------------
    blind_tags = rr.get("blind_tags", {})
    tag_key = blind_tags.get(blind_on_deck)
    awarded_tags: list[dict[str, Any]] = gs.setdefault("awarded_tags", [])

    if tag_key:
        from jackdaw.engine.tags import Tag

        tag = Tag(tag_key)
        tag_result = tag.apply("immediate", gs, rng=gs.get("rng"))

        awarded_tags.append({
            "key": tag_key,
            "result": tag_result,
            "blind": blind_on_deck,
        })

        # Apply immediate tag effects
        if tag_result is not None:
            if tag_result.dollars:
                gs["dollars"] = gs.get("dollars", 0) + tag_result.dollars

    # ------------------------------------------------------------------
    # 4. Fire joker skip_blind context
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    game_snap = GameSnapshot(
        joker_count=len(jokers),
        money=gs.get("dollars", 0),
        skips=gs.get("skips", 0),
    )
    for joker in jokers:
        if not getattr(joker, "debuff", False):
            ctx = JokerContext(
                skip_blind=True,
                jokers=jokers,
                game=game_snap,
            )
            calculate_joker(joker, ctx)

    # ------------------------------------------------------------------
    # 5. Advance blind_on_deck
    # ------------------------------------------------------------------
    if blind_on_deck == "Small":
        gs["blind_on_deck"] = "Big"
        rr["blind_states"]["Big"] = "Select"
    else:
        gs["blind_on_deck"] = "Boss"
        rr["blind_states"]["Boss"] = "Select"

    # ------------------------------------------------------------------
    # 6. Double Tag check
    # ------------------------------------------------------------------
    if tag_key:
        _check_double_tag(gs, tag_key)

    gs["phase"] = GamePhase.BLIND_SELECT
    return gs


def _handle_play_hand(
    gs: dict[str, Any], indices: tuple[int, ...]
) -> dict[str, Any]:
    """Play cards from the hand, score them, and check if blind is beaten."""
    _require_phase(gs, GamePhase.SELECTING_HAND)

    cr = gs["current_round"]
    if cr["hands_left"] <= 0:
        raise IllegalActionError("No hands remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot play more than 5 cards")

    # Extract played cards (sorted descending so removal doesn't shift indices)
    played = [hand[i] for i in indices]
    held = [c for i, c in enumerate(hand) if i not in set(indices)]

    # Score the hand
    from jackdaw.engine.scoring import score_hand

    blind = gs["blind"]
    rng = gs.get("rng")
    jokers = gs.get("jokers", [])
    hand_levels = gs.get("hand_levels")

    result = score_hand(
        played_cards=played,
        held_cards=held,
        jokers=jokers,
        hand_levels=hand_levels,
        blind=blind,
        rng=rng,
        probabilities_normal=gs.get("probabilities", {}).get("normal", 1),
        game_state=gs,
        back_key=gs.get("selected_back_key"),
        blind_chips=blind.chips,
    )

    # Accumulate chips
    total_score = math.floor(result.chips * result.mult)
    gs["chips"] = gs.get("chips", 0) + total_score
    gs["last_score_result"] = result

    # Track hand usage
    cr["hands_left"] -= 1
    cr["hands_played"] += 1
    gs["hands_played"] = gs.get("hands_played", 0) + 1

    if hand_levels is not None:
        hand_levels.record_play(result.hand_type)

    # Remove played cards from hand, add to discard area
    gs["hand"] = held
    played_cards_area = gs.setdefault("played_cards_area", [])
    played_cards_area.extend(played)

    # Check if blind is beaten
    if gs["chips"] >= blind.chips:
        _round_won(gs)
    elif cr["hands_left"] <= 0:
        # No hands left and blind not beaten → game over
        if not result.saved:
            gs["phase"] = GamePhase.GAME_OVER
            gs["won"] = False
        else:
            # Mr. Bones saved — continue with 0 hands (special case)
            _round_won(gs)
    else:
        # More hands available — draw back up and stay in SELECTING_HAND
        _draw_hand(gs)

    return gs


def _handle_discard(
    gs: dict[str, Any], indices: tuple[int, ...]
) -> dict[str, Any]:
    """Discard cards from the hand and draw replacements."""
    _require_phase(gs, GamePhase.SELECTING_HAND)

    cr = gs["current_round"]
    if cr["discards_left"] <= 0:
        raise IllegalActionError("No discards remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot discard more than 5 cards")

    # Discard cost (Golden Needle challenge)
    discard_cost = gs.get("modifiers", {}).get("discard_cost", 0)
    if discard_cost > 0:
        gs["dollars"] = gs.get("dollars", 0) - discard_cost

    # Remove discarded cards
    discarded = [hand[i] for i in indices]
    gs["hand"] = [c for i, c in enumerate(hand) if i not in set(indices)]

    cr["discards_left"] -= 1
    cr["discards_used"] += 1
    gs["unused_discards"] = max(0, gs.get("unused_discards", 0) - 1)

    # Track discards
    discard_pile = gs.setdefault("discard_pile", [])
    discard_pile.extend(discarded)

    # Draw replacements
    _draw_hand(gs)

    return gs


def _handle_cash_out(gs: dict[str, Any]) -> dict[str, Any]:
    """Accept round earnings and proceed to the shop."""
    _require_phase(gs, GamePhase.ROUND_EVAL)

    earnings = gs.get("round_earnings")
    if earnings:
        gs["dollars"] = gs.get("dollars", 0) + earnings.total

    gs["phase"] = GamePhase.SHOP
    return gs


def _handle_buy_card(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase a card from the shop."""
    _require_phase(gs, GamePhase.SHOP)

    shop_cards: list = gs.get("shop_cards", [])
    if idx < 0 or idx >= len(shop_cards):
        raise IllegalActionError(f"Invalid shop index {idx}")

    card = shop_cards[idx]
    if card.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford card")

    gs["dollars"] -= card.cost
    shop_cards.pop(idx)

    # Place card in appropriate area
    card_set = card.ability.get("set", "") if isinstance(card.ability, dict) else ""
    if card_set == "Joker":
        gs.setdefault("jokers", []).append(card)
    elif card_set in ("Tarot", "Planet", "Spectral"):
        gs.setdefault("consumables", []).append(card)
    else:
        # Playing card → add to deck
        gs.setdefault("deck", []).append(card)

    return gs


def _handle_buy_and_use(
    gs: dict[str, Any], idx: int, targets: tuple[int, ...] | None
) -> dict[str, Any]:
    """Buy a consumable and immediately use it."""
    _require_phase(gs, GamePhase.SHOP)

    shop_cards: list = gs.get("shop_cards", [])
    if idx < 0 or idx >= len(shop_cards):
        raise IllegalActionError(f"Invalid shop index {idx}")

    card = shop_cards[idx]
    if card.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford card")

    gs["dollars"] -= card.cost
    shop_cards.pop(idx)

    # Use immediately (consumable effect)
    from jackdaw.engine.consumables import use_consumable

    use_consumable(card, game_state=gs, target_indices=targets)
    return gs


def _handle_sell_card(gs: dict[str, Any], area: str, idx: int) -> dict[str, Any]:
    """Sell a card for its sell value."""
    _require_phase(gs, GamePhase.SHOP)

    cards: list = gs.get(area, [])
    if idx < 0 or idx >= len(cards):
        raise IllegalActionError(f"Invalid {area} index {idx}")

    card = cards[idx]
    if getattr(card, "eternal", False):
        raise IllegalActionError("Cannot sell eternal card")

    gs["dollars"] = gs.get("dollars", 0) + card.sell_cost
    cards.pop(idx)
    return gs


def _handle_use_consumable(
    gs: dict[str, Any], idx: int, targets: tuple[int, ...] | None
) -> dict[str, Any]:
    """Use a consumable from the consumable area."""
    _require_phase(gs, GamePhase.BLIND_SELECT, GamePhase.SELECTING_HAND,
                   GamePhase.ROUND_EVAL, GamePhase.SHOP)

    consumables: list = gs.get("consumables", [])
    if idx < 0 or idx >= len(consumables):
        raise IllegalActionError(f"Invalid consumable index {idx}")

    card = consumables.pop(idx)

    from jackdaw.engine.consumables import use_consumable

    use_consumable(card, game_state=gs, target_indices=targets)
    return gs


def _handle_redeem_voucher(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase and activate a voucher."""
    _require_phase(gs, GamePhase.SHOP)

    vouchers: list = gs.get("shop_vouchers", [])
    if idx < 0 or idx >= len(vouchers):
        raise IllegalActionError(f"Invalid voucher index {idx}")

    card = vouchers[idx]
    if card.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford voucher")

    gs["dollars"] -= card.cost
    vouchers.pop(idx)

    from jackdaw.engine.vouchers import apply_voucher

    gs["used_vouchers"][card.center_key] = True
    apply_voucher(card.center_key, gs)
    return gs


def _handle_open_booster(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Open a booster pack — transition to PACK_OPENING phase."""
    _require_phase(gs, GamePhase.SHOP)

    boosters: list = gs.get("shop_boosters", [])
    if idx < 0 or idx >= len(boosters):
        raise IllegalActionError(f"Invalid booster index {idx}")

    pack = boosters[idx]
    if pack.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford booster")

    gs["dollars"] -= pack.cost
    boosters.pop(idx)

    # Store pack info for PACK_OPENING phase
    # The actual card generation is done by the pack system
    gs["pack_cards"] = gs.get("pack_cards", [])
    gs["pack_choices_remaining"] = gs.get("pack_choices_remaining", 1)
    gs["shop_return_phase"] = GamePhase.SHOP

    gs["phase"] = GamePhase.PACK_OPENING
    return gs


def _handle_pick_pack_card(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Pick a card from an opened booster pack."""
    _require_phase(gs, GamePhase.PACK_OPENING)

    pack_cards: list = gs.get("pack_cards", [])
    remaining = gs.get("pack_choices_remaining", 0)
    if remaining <= 0:
        raise IllegalActionError("No pack choices remaining")
    if idx < 0 or idx >= len(pack_cards):
        raise IllegalActionError(f"Invalid pack card index {idx}")

    card = pack_cards.pop(idx)
    gs["pack_choices_remaining"] = remaining - 1

    # Place card in appropriate area
    card_set = card.ability.get("set", "") if isinstance(card.ability, dict) else ""
    if card_set == "Joker":
        gs.setdefault("jokers", []).append(card)
    elif card_set in ("Tarot", "Planet", "Spectral"):
        gs.setdefault("consumables", []).append(card)
    else:
        gs.setdefault("deck", []).append(card)

    # If no choices remaining, return to shop
    if gs["pack_choices_remaining"] <= 0 or not pack_cards:
        gs["pack_cards"] = []
        gs["phase"] = gs.get("shop_return_phase", GamePhase.SHOP)

    return gs


def _handle_skip_pack(gs: dict[str, Any]) -> dict[str, Any]:
    """Skip remaining picks and return to shop."""
    _require_phase(gs, GamePhase.PACK_OPENING)
    gs["pack_cards"] = []
    gs["pack_choices_remaining"] = 0
    gs["phase"] = gs.get("shop_return_phase", GamePhase.SHOP)
    return gs


def _handle_reroll(gs: dict[str, Any]) -> dict[str, Any]:
    """Reroll the shop."""
    _require_phase(gs, GamePhase.SHOP)

    cr = gs.get("current_round", {})
    free = cr.get("free_rerolls", 0)
    cost = cr.get("reroll_cost", 5)

    if free > 0:
        cr["free_rerolls"] = free - 1
    elif gs.get("dollars", 0) >= cost:
        gs["dollars"] -= cost
    else:
        raise IllegalActionError("Cannot afford reroll")

    # Increment reroll cost for next reroll
    cr["reroll_cost_increase"] = cr.get("reroll_cost_increase", 0) + 1
    cr["reroll_cost"] = gs.get("base_reroll_cost", 5) + cr["reroll_cost_increase"]

    # Actual shop regeneration would happen here
    # (shop card pool generation is handled by the shop module)

    return gs


def _handle_next_round(gs: dict[str, Any]) -> dict[str, Any]:
    """Leave the shop and proceed to the next blind."""
    _require_phase(gs, GamePhase.SHOP)

    rr = gs["round_resets"]
    blind_on_deck = gs.get("blind_on_deck", "Small")

    if blind_on_deck == "Boss" and rr["blind_states"].get("Boss") == "Defeated":
        # Boss defeated → advance ante
        _advance_ante(gs)
        gs["blind_on_deck"] = "Small"
    else:
        # Set next blind to Select
        if blind_on_deck == "Small":
            rr["blind_states"]["Small"] = "Select"
        elif blind_on_deck == "Big":
            rr["blind_states"]["Big"] = "Select"
        else:
            rr["blind_states"]["Boss"] = "Select"

    gs["phase"] = GamePhase.BLIND_SELECT
    return gs


def _handle_sort_hand(gs: dict[str, Any], mode: str) -> dict[str, Any]:
    """Sort the hand by rank or suit."""
    _require_phase(gs, GamePhase.SELECTING_HAND)

    hand: list = gs.get("hand", [])
    if mode == "rank":
        hand.sort(key=lambda c: (
            getattr(c.base, "id", 0) if c.base else 0,
            getattr(c.base, "suit_nominal", 0) if c.base else 0,
        ))
    elif mode == "suit":
        hand.sort(key=lambda c: (
            getattr(c.base, "suit_nominal", 0) if c.base else 0,
            getattr(c.base, "id", 0) if c.base else 0,
        ))
    return gs


def _handle_reorder_jokers(
    gs: dict[str, Any], order: tuple[int, ...]
) -> dict[str, Any]:
    """Reorder jokers by permutation."""
    _require_phase(gs, GamePhase.SELECTING_HAND, GamePhase.SHOP)

    jokers: list = gs.get("jokers", [])
    if not order:
        return gs  # marker action, no-op
    if sorted(order) != list(range(len(jokers))):
        raise IllegalActionError("Invalid joker permutation")

    gs["jokers"] = [jokers[i] for i in order]
    return gs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_hand(gs: dict[str, Any]) -> None:
    """Draw cards from deck to fill the hand up to hand_size."""
    deck: list = gs.get("deck", [])
    hand: list = gs.setdefault("hand", [])
    hand_size: int = gs.get("hand_size", 8)
    to_draw = min(len(deck), hand_size - len(hand))
    for _ in range(to_draw):
        if deck:
            hand.append(deck.pop(0))


def _round_won(gs: dict[str, Any]) -> None:
    """Handle winning a round — transition to ROUND_EVAL."""
    from jackdaw.engine.economy import calculate_round_earnings

    cr = gs["current_round"]
    blind = gs["blind"]
    jokers = gs.get("jokers", [])
    rng = gs.get("rng")

    # Calculate earnings
    earnings = calculate_round_earnings(
        blind=blind,
        hands_left=cr["hands_left"],
        discards_left=cr["discards_left"],
        money=gs.get("dollars", 0),
        jokers=jokers,
        game_state=gs,
        rng=rng,
    )
    gs["round_earnings"] = earnings

    # Process perishable/rental
    from jackdaw.engine.round_lifecycle import process_round_end_cards

    process_round_end_cards(jokers, gs)

    # Return played cards and hand to deck
    deck: list = gs.setdefault("deck", [])
    hand: list = gs.get("hand", [])
    played: list = gs.get("played_cards_area", [])
    discarded: list = gs.get("discard_pile", [])
    deck.extend(hand)
    deck.extend(played)
    deck.extend(discarded)
    gs["hand"] = []
    gs["played_cards_area"] = []
    gs["discard_pile"] = []

    # Track unused discards for tag calculations
    gs["unused_discards"] = cr["discards_left"]

    # Update blind state
    rr = gs["round_resets"]
    blind_on_deck = gs.get("blind_on_deck", "Small")
    rr["blind_states"][blind_on_deck] = "Defeated"

    # Check for game won (Boss blind of final ante)
    if blind_on_deck == "Boss" and rr["ante"] >= gs.get("win_ante", 8):
        gs["won"] = True

    gs["phase"] = GamePhase.ROUND_EVAL


def _advance_ante(gs: dict[str, Any]) -> None:
    """Advance to the next ante after boss is defeated."""
    rr = gs["round_resets"]
    rr["ante"] += 1
    rr["blind_ante"] = rr["ante"]
    rr["blind_states"] = {"Small": "Select", "Big": "Upcoming", "Boss": "Upcoming"}
    rr["boss_rerolled"] = False

    # Generate new boss, tags, voucher for next ante
    from jackdaw.engine.tags import assign_ante_blinds

    rng = gs.get("rng")
    if rng:
        ante_result = assign_ante_blinds(rr["ante"], rng, gs)
        rr["blind_choices"]["Boss"] = ante_result["blind_choices"]["Boss"]
        gs["current_round"]["voucher"] = ante_result["voucher"]


# ---------------------------------------------------------------------------
# setting_blind joker context
# ---------------------------------------------------------------------------


def _fire_setting_blind(
    gs: dict[str, Any],
    jokers: list,
    blind: Any,
) -> list[dict[str, Any]]:
    """Fire ``setting_blind`` on all jokers and collect mutations.

    Returns a list of side-effect dicts from JokerResult.extra.
    """
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    game_snap = GameSnapshot(
        joker_count=len(jokers),
        joker_slots=gs.get("joker_slots", 5),
        money=gs.get("dollars", 0),
    )

    mutations: list[dict[str, Any]] = []
    for joker in jokers:
        if getattr(joker, "debuff", False):
            continue
        ctx = JokerContext(
            setting_blind=True,
            blind=blind,
            jokers=jokers,
            game=game_snap,
        )
        result = calculate_joker(joker, ctx)
        if result and result.extra:
            mutations.append(result.extra)

    return mutations


def _apply_setting_blind_mutations(
    gs: dict[str, Any],
    mutations: list[dict[str, Any]],
    jokers: list,
) -> None:
    """Process side-effects from setting_blind jokers."""
    rng = gs.get("rng")

    for mut in mutations:
        # Chicot / Luchador: disable blind
        if mut.get("disable_blind"):
            blind = gs.get("blind")
            if blind:
                blind.disabled = True
                # Un-debuff all playing cards
                for card in gs.get("deck", []):
                    card.debuff = False

        # Madness: destroy random joker (not self)
        if mut.get("destroy_random_joker") and len(jokers) > 1:
            if rng:
                # Pick a random non-self joker to destroy
                import random

                candidates = [j for j in jokers if j is not jokers[0]]
                if candidates:
                    seed_val = rng.seed("madness")
                    target, _ = rng.element(candidates, seed_val)
                    jokers.remove(target)

        # Burglar: set hands / remove discards
        if "set_hands" in mut:
            cr = gs.get("current_round", {})
            cr["hands_left"] = cr.get("hands_left", 0) + mut["set_hands"]
        if "set_discards" in mut:
            cr = gs.get("current_round", {})
            cr["discards_left"] = mut["set_discards"]

        # Marble Joker / Certificate / Riff-raff / Cartomancer: create cards
        if "create" in mut:
            create = mut["create"]
            ctype = create.get("type", "")
            if ctype == "playing_card":
                # Marble Joker: add Stone Card; Certificate: add card with seal
                deck: list = gs.setdefault("deck", [])
                from jackdaw.engine.card import Card

                c = Card()
                enhancement = create.get("enhancement")
                if enhancement:
                    c.ability = {"effect": enhancement, "set": "Enhanced"}
                if create.get("seal"):
                    c.seal = "Gold"  # Certificate default
                deck.append(c)
            elif ctype == "Joker":
                # Riff-raff: create Common jokers
                count = create.get("count", 1)
                for _ in range(count):
                    from jackdaw.engine.card import Card as _Card

                    j = _Card(center_key="j_joker")
                    j.ability = {"set": "Joker", "effect": "", "name": "Joker"}
                    jokers.append(j)
            elif ctype == "Tarot":
                # Cartomancer: create Tarot
                consumables: list = gs.setdefault("consumables", [])
                from jackdaw.engine.card import Card as _Card2

                t = _Card2(center_key="c_fool")
                t.ability = {"set": "Tarot", "effect": ""}
                consumables.append(t)


# ---------------------------------------------------------------------------
# Boss blind set-time effects
# ---------------------------------------------------------------------------


def _apply_boss_blind_effects(gs: dict[str, Any], blind: Any) -> None:
    """Apply boss blind effects at set-time (blind.lua:157-209).

    These are one-time mutations that happen when the blind is set,
    before the round starts.
    """
    cr = gs.get("current_round", {})
    name = getattr(blind, "name", "")

    # The Water: remove all discards
    if name == "The Water":
        current_discards = cr.get("discards_left", 0)
        blind.discards_sub = current_discards
        cr["discards_left"] = 0

    # The Needle: reduce to 1 hand
    elif name == "The Needle":
        rr = gs.get("round_resets", {})
        current_hands = rr.get("hands", 4)
        blind.hands_sub = current_hands - 1
        cr["hands_left"] = max(1, cr.get("hands_left", current_hands) - blind.hands_sub)

    # The Manacle: -1 hand size
    elif name == "The Manacle":
        gs["hand_size"] = gs.get("hand_size", 8) - 1

    # Amber Acorn: shuffle jokers (flip + randomize order)
    elif name == "Amber Acorn":
        jokers: list = gs.get("jokers", [])
        if jokers:
            for j in jokers:
                j.facing = "back"
            rng = gs.get("rng")
            if rng and len(jokers) > 1:
                seed_val = rng.seed("aajk")
                rng.shuffle(jokers, seed_val)

    # The Eye: reset hand tracking
    elif name == "The Eye":
        blind.hands = {ht: False for ht in [
            "Flush Five", "Flush House", "Five of a Kind",
            "Straight Flush", "Four of a Kind", "Full House",
            "Flush", "Straight", "Three of a Kind",
            "Two Pair", "Pair", "High Card",
        ]}

    # The Mouth: reset only_hand
    elif name == "The Mouth":
        blind.only_hand = False


# ---------------------------------------------------------------------------
# Double Tag check
# ---------------------------------------------------------------------------


def _check_double_tag(gs: dict[str, Any], awarded_tag_key: str) -> None:
    """If player has a Double Tag active, duplicate the just-awarded tag."""
    tags: list = gs.get("tags", [])
    if not tags:
        return

    # Check if any active tag is tag_double
    from jackdaw.engine.tags import Tag

    for i, tag_entry in enumerate(tags):
        tag_key = tag_entry if isinstance(tag_entry, str) else getattr(tag_entry, "key", "")
        if tag_key == "tag_double" and awarded_tag_key != "tag_double":
            # Fire the duplicate
            dup_tag = Tag(awarded_tag_key)
            dup_result = dup_tag.apply("immediate", gs, rng=gs.get("rng"))

            awarded_tags: list = gs.setdefault("awarded_tags", [])
            awarded_tags.append({
                "key": awarded_tag_key,
                "result": dup_result,
                "blind": "double",
            })

            if dup_result and dup_result.dollars:
                gs["dollars"] = gs.get("dollars", 0) + dup_result.dollars

            # Remove the Double Tag (consumed)
            tags.pop(i)
            break
