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
    RedeemVoucher,
    Reroll,
    RerollBoss,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
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
    result = _dispatch(game_state, action)
    # The joker area re-sorts pinned cards to the left every frame
    # (cardarea.lua:528), and Luxury Tax re-checks the hand size against
    # the dollars every frame (cardarea.lua:245).
    _align_pinned_jokers(result.get("jokers") or [])
    _sync_hand_size_to_dollars(result)
    run_pending_reprice(result)
    run_pending_blind_redebuff(result)
    # Game:update checks SELECTING_HAND every frame: an emptied hand is
    # redrawn, and a round out of cards ends (game.lua:2544-2550, 3056-3069).
    if result.get("phase") == GamePhase.SELECTING_HAND:
        _update_selecting_hand(result)
    # No per-step reprice: Lua re-runs set_cost only where it calls it
    # (shop.shop_set_cost lists the points; G204).
    return result


def _dispatch(game_state: dict[str, Any], action: Action) -> dict[str, Any]:
    match action:
        case SelectBlind():
            return _handle_select_blind(game_state)
        case SkipBlind():
            return _handle_skip_blind(game_state)
        case RerollBoss():
            return _handle_reroll_boss(game_state)
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
        case PickPackCard(card_index=idx, target_indices=targets):
            return _handle_pick_pack_card(game_state, idx, targets)
        case SkipPack():
            return _handle_skip_pack(game_state)
        case Reroll():
            return _handle_reroll(game_state)
        case NextRound():
            return _handle_next_round(game_state)
        case SortHand(mode=mode):
            return _handle_sort_hand(game_state, mode)
        case SwapHandLeft(idx=idx):
            return _handle_swap_hand(game_state, idx, -1)
        case SwapHandRight(idx=idx):
            return _handle_swap_hand(game_state, idx, +1)
        case SwapJokersLeft(idx=idx):
            return _handle_swap_jokers(game_state, idx, -1)
        case SwapJokersRight(idx=idx):
            return _handle_swap_jokers(game_state, idx, +1)
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
        raise IllegalActionError(f"Action not valid in phase {phase!r} (expected {phases})")
    return phase


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_select_blind(gs: dict[str, Any]) -> dict[str, Any]:
    """Accept the current blind and start the round.

    Full sequence matching ``game.lua`` select_blind → ``blind.lua``
    ``set_blind`` → ``state_events.lua`` ``new_round``:

    1. Create Blind from blind_choices
    2. Call ``start_round`` (reset counters, targeting cards) and set_blind's
       debuff pass
    3. Apply boss blind set-time effects (Water, Needle, Manacle,
       Amber Acorn's flip)
    4. Fire joker ``setting_blind`` context (Chicot, Madness, Burglar,
       Marble Joker, Riff-raff, Cartomancer) and process its events in
       Lua's order
    5. Shuffle the deck, disable the boss for each Chicot
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
    # set_blind records G.GAME.last_blind.boss (blind.lua:97-99).
    gs["last_blind_was_boss"] = bool(blind.boss)
    rr["blind_states"][blind_on_deck] = "Current"
    rr["blind"] = blind
    # ease_round(1) fires in the select-blind callback
    # (button_callbacks.lua:2533), not at blind defeat.
    gs["round"] = gs.get("round", 0) + 1

    # ------------------------------------------------------------------
    # 2. Start round (reset counters, targeting cards)
    #    Vanilla new_round() resets hands/discards (state_events.lua:296-297)
    #    BEFORE the joker setting_blind loop (line 336) — Burglar et al.
    #    mutate the freshly-reset counters.  Boss effects (The Water /
    #    Needle) then decrement from the post-joker values.
    # ------------------------------------------------------------------
    start_round(gs)

    # set_blind's debuff pass over every playing card and joker
    # (blind.lua:207-213) runs inside set_blind, before the jokers'
    # setting_blind context: a card a setting_blind effect creates (Marble
    # Joker's stone) is not checked here.
    _set_blind_debuff_pass(gs)

    # ------------------------------------------------------------------
    # 3. Boss blind set-time effects (blind.lua:157-205): set_blind runs
    #    before the jokers' setting_blind context (state_events.lua:333-337).
    #    The Water and The Needle count what they take now, and their eases
    #    are events queued ahead of every joker's: Burglar sees no discards
    #    left, and a later disable gives back the full count (G97).
    # ------------------------------------------------------------------
    jokers: list = gs.get("jokers", [])
    # Amber Acorn's set_blind flips the board face down BEFORE the jokers'
    # setting_blind context (blind.lua:190-194; state_events.lua:333-337),
    # so jokers that setting_blind effects create stay face up.  Its three
    # shuffles are queued events that run after those effects.
    amber_shuffles = blind.name == "Amber Acorn" and len(jokers) > 1
    if blind.name == "Amber Acorn":
        for j in jokers:
            j.facing = "back"
    if blind.boss and not blind.disabled:
        _apply_boss_blind_effects(gs, blind)

    # ------------------------------------------------------------------
    # 4. Fire joker setting_blind context, then its events in Lua's order
    #    (Amber Acorn's three 'aajk' shuffles among them)
    # ------------------------------------------------------------------
    setting_mutations = _fire_setting_blind(gs, jokers, blind)
    touched, sliced_hand_size = _apply_setting_blind_mutations(
        gs, setting_mutations, jokers, amber_shuffles=amber_shuffles
    )

    # ------------------------------------------------------------------
    # 5. Per-round deck shuffle (state_events.lua:344)
    #    Fires AFTER set_blind and joker setting_blind context,
    #    BEFORE draw_to_hand.  Key: 'nr' + str(ante).
    # ------------------------------------------------------------------
    rng = gs.get("rng")
    if rng:
        deck_list: list = gs.get("deck", [])
        nr_seed = rng.seed("nr" + str(ante))
        rng.shuffle(deck_list, nr_seed)

    # pending_deck_bottom stays in the snapshot schema but is always empty:
    # Marble Joker's stones now join the deck before the shuffle (G55).
    pending_bottom = gs.pop("pending_deck_bottom", None)
    if pending_bottom:
        gs.setdefault("deck", [])[:0] = pending_bottom

    # Chicot: its setting_blind wraps G.GAME.blind:disable() in two nested
    # events (card.lua:2492-2500), so the disable runs after the set-time
    # effects above (applied in full) and after the deck shuffle, before the
    # deal — once per Chicot that fired.
    for _ in range(sum(1 for m in setting_mutations if m.get("disable_blind"))):
        disable_blind(gs)

    # A joker or consumable added or removed by those events queued a
    # set_blind(nil, true) re-debuff (card.lua:641, 698) that runs after
    # every card they made, so Marble's Stone gets the boss's debuff too
    # (Verdant Leaf; The Plant with Pareidolia).  Without one it keeps the
    # state it was made with.
    if touched:
        reapply_blind_debuffs(gs)

    # ------------------------------------------------------------------
    # 6b. round_start_bonus tags (Juggle: +3 hand size for this round).
    #     Vanilla fires these at DRAW_TO_HAND, before the initial draw
    #     (game.lua:3215, tag.lua:334), tracking the delta in
    #     round_resets.temp_handsize and reverting it at round end.
    # ------------------------------------------------------------------
    from jackdaw.engine.tags import Tag

    for entry in gs.get("awarded_tags", []):
        if entry.get("rsb_fired"):
            continue
        rsb = Tag(entry.get("key", "")).apply("round_start_bonus", gs, rng=rng)
        if rsb is not None and rsb.hand_size_delta:
            change_hand_size(gs, rsb.hand_size_delta)
            # run_init seeds temp_handsize as None — `or 0`, not a default.
            rr["temp_handsize"] = (rr.get("temp_handsize") or 0) + rsb.hand_size_delta
            entry["rsb_fired"] = True

    # ------------------------------------------------------------------
    # 7-9. Draw to hand: the deal (debuffs were set by set_blind's pass; a
    #      draw never re-checks them), the first_hand_drawn context
    #      (Certificate), then SELECTING_HAND and the boss's drawn_to_hand
    #      (Cerulean Bell, Crimson Heart) — game.lua:3208-3245.
    # ------------------------------------------------------------------
    _draw_to_hand(gs)

    # A sliced joker's G.hand:change_size (remove_from_deck, card.lua:648-650)
    # is queued by its removal event, itself queued by the slicing joker's
    # event: it runs after the deal counted its cards and while the hand is
    # still empty, so it neither shrinks the deal nor draws (G105).
    if sliced_hand_size:
        gs["hand_size"] = gs.get("hand_size", 0) + sliced_hand_size
    return gs


def _fire_new_blind_choice_tags(gs: dict[str, Any]) -> None:
    """Fire new_blind_choice context for any awarded tags that need it.

    In Lua, new_blind_choice tags fire when entering the blind select screen
    after a skip.  Pack-creating tags (buffoon, charm, ethereal, meteor,
    standard) open a pack for the player to pick from.  The boss tag rerolls
    the boss blind.

    Since the engine has no interactive player, pack-creating tags populate
    ``gs["pack_cards"]`` and set the phase to PACK_OPENING so the caller
    (or agent) can pick or skip.
    """
    from jackdaw.engine.tags import Tag

    awarded: list[dict] = gs.get("awarded_tags", [])
    rng = gs.get("rng")

    for entry in awarded:
        tag_key = entry.get("key", "")
        # Only fire tags that haven't been processed for new_blind_choice yet
        if entry.get("nbc_fired"):
            continue

        tag = Tag(tag_key)
        result = tag.apply("new_blind_choice", gs, rng=rng)
        entry["nbc_fired"] = True

        if result is None:
            continue

        if result.reroll_boss:
            # Boss Tag: G.from_boss_tag + G.FUNCS.reroll_boss (tag.lua:284-300).
            # The loop goes on to the next pending tag, as reroll_boss's own
            # new_blind_choice pass would.
            _reroll_boss(gs, from_tag=True)

        if result.create_pack:
            _open_tag_pack(gs, result.create_pack)
            # Every pass stops at the first tag that fires (`then break end`):
            # the next pack tag stays held until end_consumeable's own pass
            # (button_callbacks.lua:2617-2619), so one pack is open at a time
            # (G106).
            return


def _reroll_boss(gs: dict[str, Any], *, from_tag: bool) -> None:
    """``G.FUNCS.reroll_boss`` (button_callbacks.lua:2800-2853).

    Marks the ante rerolled (Director's Cut's once-per-ante gate) even when
    a Boss Tag does it, charges $10 unless it is the Boss Tag
    (``G.from_boss_tag``), and draws a new boss from the ante's pool — bans
    and the win ante included (get_new_boss, common_events.lua:2338-2383).
    The rebuilt boss UI finds its Orbital hand set, so nothing else rolls.
    """
    from jackdaw.engine.actions import BOSS_REROLL_COST
    from jackdaw.engine.blind import get_new_boss

    rr = gs["round_resets"]
    rr["boss_rerolled"] = True
    if not from_tag:
        gs["dollars"] = gs.get("dollars", 0) - BOSS_REROLL_COST
    rr.setdefault("blind_choices", {})["Boss"] = get_new_boss(
        rr.get("ante", 1),
        gs.setdefault("bosses_used", {}),
        gs.get("rng"),
        win_ante=gs.get("win_ante", 8),
        banned_keys=gs.get("banned_keys"),
    )


def _handle_reroll_boss(gs: dict[str, Any]) -> dict[str, Any]:
    """Reroll the boss blind with Director's Cut or Retcon (G.FUNCS.reroll_boss).

    After the new boss, reroll_boss's event gives pending new_blind_choice
    tags their turn (button_callbacks.lua:2847-2849).
    """
    _require_phase(gs, GamePhase.BLIND_SELECT)
    from jackdaw.engine.actions import can_reroll_boss

    if not can_reroll_boss(gs):
        raise IllegalActionError("Boss reroll not available")
    _reroll_boss(gs, from_tag=False)
    _fire_new_blind_choice_tags(gs)
    return gs


def _open_tag_pack(gs: dict[str, Any], pack_key: str, force: bool = False) -> None:
    """Open a pack from a tag reward, populating pack_cards.

    The caller (or agent) must then pick from the pack or skip it.
    For the engine-only path (no interactive player), we store the pack
    state so that get_legal_actions returns PickPackCard/SkipPack options.

    A pack tag makes a free booster Card and opens it through
    ``G.FUNCS.use_card`` (tag.lua:209-282), i.e. the same ``Card:open`` as
    a shop pack: see :func:`_open_pack`.
    """
    # A pack is already open: queue this one, to open (and roll its
    # contents) when the current one closes.  The tag passes never get here
    # since G106: a second pack tag stays held until end_consumeable's pass.
    if not force and (gs.get("phase") == GamePhase.PACK_OPENING or gs.get("pack_cards")):
        gs.setdefault("pending_tag_packs", []).append(pack_key)
        return

    # Save current phase and switch to pack opening
    gs["shop_return_phase"] = gs.get("phase", GamePhase.BLIND_SELECT)
    # The tag's booster is a Card (tag.lua:213-216): it takes the next sort_id
    # before the pack's contents do.
    from jackdaw.engine.card import _next_sort_id

    _next_sort_id()
    _open_pack(gs, pack_key)


def _open_pack(gs: dict[str, Any], pack_key: str) -> None:
    """``Card:open`` (card.lua:1681-1811) for a shop or tag pack, once paid for.

    Card:open queues one event (after 0.4s) that, in order: explodes the
    booster, queues the contents' creation as a non-blocking event
    ``explode_time`` later (card.lua:1725), runs every joker's
    ``open_booster`` context (card.lua:1796-1798) and, under Inflation,
    raises ``G.GAME.inflation`` (card.lua:1800-1807).  ``explode()`` queues
    a non-blockable but blocking event that holds the queue for
    ``1.5 * explode_time`` (card.lua:2002-2012), so the contents arrive
    first and Hallucination's Tarot, made by the blocking event its context
    queues (card.lua:2339-2348), after them: the pack's cards are in
    used_jokers when the Tarot is drawn (G266, which reverses G44).  An
    Arcana/Spectral pack then deals the hand its picks target
    (update_arcana_pack / update_spectral_pack).
    """
    from jackdaw.engine.data.prototypes import BOOSTERS
    from jackdaw.engine.packs import generate_pack_cards

    # open_booster (Hallucination's 'halu' roll), then the inflation; the
    # Tarot itself comes from the queued event, after the contents.
    mutations = _fire_shop_joker_context(gs, open_booster=True)
    _inflate(gs)

    rng = gs.get("rng")
    ante = gs.get("round_resets", {}).get("ante", 1)
    if pack_key in BOOSTERS:
        # G.GAME.pack_size = self.ability.extra (card.lua:1691-1705).
        gs["pack_size"] = BOOSTERS[pack_key].config["extra"]
    if rng and pack_key in BOOSTERS:
        cards, choose = generate_pack_cards(pack_key, rng, ante, gs)
        gs["pack_cards"] = cards
        gs["pack_choices_remaining"] = choose
        gs["pack_type"] = BOOSTERS[pack_key].kind
    else:
        gs["pack_cards"] = []
        gs["pack_choices_remaining"] = 1
        gs["pack_type"] = "Unknown"

    # Hallucination gates its roll on consumable room counting
    # G.GAME.consumeable_buffer (card.lua:2336-2338, handler side); each hit
    # reserves a slot and its event makes the Tarot and zeroes the buffer
    # (card.lua:2339-2347).
    for mut in mutations:
        desc = mut.get("create")
        if desc:
            _resolve_create_descriptors(gs, [desc])
    gs["consumeable_buffer"] = 0

    # For Arcana/Spectral packs: deal hand from deck for targeting
    if gs["pack_type"] in ("Arcana", "Spectral"):
        _deal_pack_hand(gs)

    gs["phase"] = GamePhase.PACK_OPENING


def _deal_pack_hand(gs: dict[str, Any]) -> None:
    """Deal the hand an Arcana or Spectral pack's picks target.

    update_arcana_pack / update_spectral_pack (game.lua:3374/3425) call
    draw_from_deck_to_hand: up to the hand size, each card drawn from the
    END of the deck through draw_card_from, so X-ray Vision's roll applies
    (cardarea.lua:601-605).  Between rounds the blind is the reset one, so
    no boss flip applies.  The dealt cards go back to the deck when the
    pack closes (_close_pack).
    """
    _sync_hand_size_to_dollars(gs)
    deck: list = gs.get("deck", [])
    hand: list = gs.get("hand", [])
    to_deal = min(len(deck), gs.get("hand_size", 8) - len(hand))
    pack_hand: list = []
    for _ in range(to_deal):
        card = deck.pop()
        card.facing = "back" if _x_ray_flip(gs) else "front"
        pack_hand.append(card)
    gs["pack_hand"] = pack_hand
    combined_hand = hand + pack_hand
    _sort_hand(gs, combined_hand)
    gs["hand"] = combined_hand


def _award_tag(
    gs: dict[str, Any], tag_key: str, source: str, orbital_hand: str | None = None
) -> None:
    """Grant one tag, mirroring live ``add_tag`` (UI_definitions.lua:1252).

    Tags live in ``awarded_tags`` as ``{"key", "result", "blind"}`` records;
    ``gs["tags"]`` is a phantom key nothing reads.  Every grant runs the
    ``immediate`` hook and applies its result, so callers never rebuild the
    record shape by hand.  ``source`` labels the provenance ("Small"/"Big"
    for skips, otherwise the granting card).  *orbital_hand* is the tag's
    ``ability.orbital_hand`` (the Orbital Tag levels it).

    Callers own the ``_check_double_tag`` call: the skip path defers it until
    after ``blind_on_deck`` advances (button_callbacks.lua:2740-2775).
    """
    from jackdaw.engine.tags import Tag

    tag = Tag(tag_key, orbital_hand=orbital_hand)
    tag_result = tag.apply("immediate", gs, rng=gs.get("rng"))
    gs.setdefault("awarded_tags", []).append(
        {"key": tag_key, "result": tag_result, "blind": source}
    )
    if tag_result is not None:
        _apply_tag_result(gs, tag_result)


def _apply_tag_result(gs: dict[str, Any], result: Any) -> None:
    """Apply a TagResult's effects to the game state.

    Handles all TagResult fields that produce immediate side-effects.
    """
    if result.dollars:
        gs["dollars"] = gs.get("dollars", 0) + result.dollars

    if result.create_jokers:
        from jackdaw.engine.card_factory import create_card

        jokers = gs.get("jokers", [])
        joker_slots = gs.get("joker_slots", 5)
        rng = gs.get("rng")
        ante = gs.get("round_resets", {}).get("ante", 1)
        for _ in range(result.create_jokers):
            if len(jokers) >= joker_slots:
                break
            # Top-up Tag: create_card('Joker', G.jokers, nil, 0, nil, nil,
            # nil, 'top') — forced Common, append 'top' (tag.lua:138)
            card = create_card(
                "Joker",
                rng,
                ante,
                area="",
                soulable=False,
                forced_rarity=1,
                append="top",
                game_state=gs,
            )
            jokers.append(card)
            card.add_to_deck(gs)

    if result.level_up is not None:
        hand_type, levels = result.level_up
        hand_levels = gs.get("hand_levels")
        if hand_levels is not None:
            for _ in range(levels):
                hand_levels.level_up(hand_type)


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
    # Keep the key present even when this blind carries no tag (shape-stable
    # for get_state/load_state); _award_tag setdefaults it again when it does.
    gs.setdefault("awarded_tags", [])

    # The skip tag was made with the blind-select screen and took this
    # blind's Orbital hand then (tag.lua:107-110).
    from jackdaw.engine.tags import ensure_orbital_choices

    orbital_hand = ensure_orbital_choices(gs)[blind_on_deck]
    if tag_key:
        _award_tag(gs, tag_key, source=blind_on_deck, orbital_hand=orbital_hand)

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
        _check_double_tag(gs, tag_key, orbital_hand=orbital_hand)

    # ------------------------------------------------------------------
    # 7. Fire new_blind_choice tags from awarded (deferred) tags
    # ------------------------------------------------------------------
    # In Lua, new_blind_choice tags fire when the blind select screen
    # appears after a skip.  Tags like tag_buffoon, tag_charm, etc.
    # open packs; tag_boss rerolls the boss blind.
    _fire_new_blind_choice_tags(gs)

    # Only return to BLIND_SELECT if a tag didn't open a pack
    if gs.get("phase") != GamePhase.PACK_OPENING:
        gs["phase"] = GamePhase.BLIND_SELECT
    return gs


def _handle_play_hand(gs: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    """Play cards from the hand, score them, and check if blind is beaten.

    Full sequence matching ``state_events.lua`` play_cards_from_highlighted
    → evaluate_play:

    1. Validate indices
    2. Move cards from hand to play area (preserve index order)
    3. Decrement hands_left, increment hands_played
    4. Update per-card stats (times_played, played_this_ante)
    5. Fire ``Blind:press_play`` (The Hook, The Tooth)
    6. Call ``score_hand`` (full 14-phase pipeline)
    7. Process scoring side-effects (dollars, card destruction,
       joker removal)
    8. Move surviving played cards to discard pile
    9. Record hand type in hand_levels
    10. Determine next phase: won / continue / game over
    11. If continuing: draw cards, re-debuff for boss
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    cr = gs["current_round"]
    if cr["hands_left"] <= 0:
        raise IllegalActionError("No hands remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot play more than 5 cards")
    if any(i < 0 or i >= len(hand) for i in indices):
        raise IllegalActionError("Card index out of range")
    _require_forced_card(hand, indices)
    _clear_forced_selection(gs)

    # ------------------------------------------------------------------
    # 2. Move cards from hand to play area
    # ------------------------------------------------------------------
    # Hand order, not click order: play_cards_from_highlighted sorts
    # G.hand.highlighted by T.x (the hand position) before drawing the
    # cards into G.play (state_events.lua:463), and evaluate_play scores
    # them in play-area order (state_events.lua:600).
    idx_set = set(indices)
    played = [hand[i] for i in sorted(idx_set)]
    held = [c for i, c in enumerate(hand) if i not in idx_set]
    gs["hand"] = held
    # G.play: the played cards stay in G.playing_cards while they are in
    # play, so the passes over every playing card see them, e.g. the boss
    # re-check that a joker or consumable's add_to_deck/remove_from_deck
    # queues (set_blind(nil, true), blind.lua:207-210; G80).
    resting_play = gs.get("played_cards_area")
    gs["played_cards_area"] = played
    # Played cards are revealed (face-down draws from The House / The
    # Wheel / The Mark / The Fish flip up when they leave the hand).
    for c in played:
        c.facing = "front"

    # ------------------------------------------------------------------
    # 3. Decrement hands_left
    #
    # hands_left drops at the play press (vanilla ease_hands_played
    # early in play_cards_from_highlighted), but BOTH hands_played
    # counters increment in the event queued AFTER evaluate_play
    # (state_events.lua:517-27) — every scoring context sees
    # PRE-increment values: Loyalty Card's run-wide window
    # (card.lua:3633), DNA / Sixth Sense's current_round == 0 checks
    # (card.lua:3501/2604).  The increments moved below score_hand
    # (bug #58, LSHACSAC: sim's Loyalty x4 fired a play late).
    # ------------------------------------------------------------------
    cr["hands_left"] -= 1

    # ------------------------------------------------------------------
    # 4. Per-card stats
    # ------------------------------------------------------------------
    for card in played:
        base = getattr(card, "base", None)
        if base is not None:
            base.times_played = getattr(base, "times_played", 0) + 1
        ability = getattr(card, "ability", None)
        if isinstance(ability, dict):
            ability["played_this_ante"] = True

    # ------------------------------------------------------------------
    # 5. Blind:press_play (blind.lua:464)
    # ------------------------------------------------------------------
    blind = gs["blind"]
    rng = gs.get("rng")
    # triggered is PER-PLAY state: vanilla clears it at every play
    # (state_events.lua:455) and Matador reads it during joker_main.
    blind.triggered = False
    _press_play(gs, blind, played, rng)

    # ------------------------------------------------------------------
    # 6. Score the hand (full 14-phase pipeline)
    # ------------------------------------------------------------------
    from jackdaw.engine.scoring import score_hand

    jokers = gs.get("jokers", [])
    hand_levels = gs.get("hand_levels")

    # Populate snapshot values that score_hand reads from game_state.
    # These are stored in nested structures but score_hand expects them
    # at the top level.
    gs["hands_left"] = cr.get("hands_left", 0)
    gs["current_round_hands_played"] = cr.get("hands_played", 0)
    gs["discards_left"] = cr.get("discards_left", 0)
    gs["discards_used"] = cr.get("discards_used", 0)
    gs["money"] = gs.get("dollars", 0)
    gs["deck_cards_remaining"] = len(gs.get("deck", []))

    # Card tallies for jokers that reference full-deck counts
    all_cards = gs.get("deck", []) + gs.get("hand", []) + gs.get("discard_pile", []) + played
    gs["playing_cards_count"] = len(all_cards)
    gs["stone_tally"] = sum(1 for c in all_cards if getattr(c, "center_key", None) == "m_stone")
    gs["steel_tally"] = sum(1 for c in all_cards if getattr(c, "center_key", None) == "m_steel")
    gs["enhanced_card_count"] = sum(
        1 for c in all_cards if getattr(c, "center_key", "") not in ("", "c_base")
    )

    # Targeting card values (from current_round)
    gs["mail_card_id"] = cr.get("mail_card", {}).get("id")
    gs["idol_card"] = cr.get("idol_card")
    gs["ancient_suit"] = cr.get("ancient_card", {}).get("suit")

    # Consumable usage tally
    usage = gs.get("consumable_usage_total", {})
    gs["consumable_usage_tarot"] = usage.get("tarot", 0)

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
        consumables=gs.get("consumables", []),
    )

    # hands_played counters increment AFTER evaluate_play completes
    # (state_events.lua:523-24) — see the step-3 comment above.
    cr["hands_played"] += 1
    gs["hands_played"] = gs.get("hands_played", 0) + 1
    gs["current_round_hands_played"] = cr["hands_played"]

    # Double or Nothing: every card of the scoring hand is debuffed for the
    # rest of the run (state_events.lua:1077-1084 — past the debuffed-hand
    # branch, so a blocked hand counts too); Card:update applies
    # perma_debuff on the next frame (card.lua:4158).
    if (gs.get("modifiers") or {}).get("debuff_played_cards"):
        for c in result.scoring_cards:
            c.ability["perma_debuff"] = True
            c.debuff = True

    # ------------------------------------------------------------------
    # 7. Process scoring side-effects
    # ------------------------------------------------------------------
    # Accumulate chips.  G.GAME.chips is a double in vanilla: add as doubles
    # so sums past 2**53 round exactly like Lua's, and a total past the
    # largest double reads as inf ("naneinf").
    gs["chips"] = float(gs.get("chips", 0)) + float(result.total)
    gs["last_score_result"] = result

    # Dollars from scoring (Gold Seal, Lucky Card, joker economy)
    if result.dollars_earned:
        gs["dollars"] = gs.get("dollars", 0) + result.dollars_earned

    # Joker self-destruction (Ice Cream, Popcorn, etc.).  A Mr. Bones that
    # saves the run dissolves only after end_round's joker loop (below).
    saver = _bones_saver(result)
    for removed in result.jokers_removed:
        if removed in jokers and removed is not saver:
            jokers.remove(removed)
            removed.remove_from_deck(gs)
            _release_used_key(gs, removed)

    # Playing card destruction (Glass shatter, etc.)
    destroyed_set = set(id(c) for c in result.cards_destroyed)
    played = [c for c in played if id(c) not in destroyed_set]
    _playing_cards_removed(gs, result.cards_destroyed)

    # Joker card creation (Vagabond, 8-Ball, Superposition, etc.): the
    # events queued during evaluate_play make the cards, each zeroing
    # G.GAME.consumeable_buffer (card.lua:3118, 3753, 3777, 3797, 2614).
    if result.joker_creates:
        _resolve_create_descriptors(
            gs, [{"type": c.get("type", "Tarot"), **c} for c in result.joker_creates]
        )
    gs["consumeable_buffer"] = 0

    # (The Ox takes the money inside Blind.debuff_hand, before the hand
    # scores — blind.lua:560-569.)

    # ------------------------------------------------------------------
    # 8. Move surviving played cards to discard pile
    #
    # In Lua, draw_from_play_to_discard (state_events.lua:522, 1088-1096)
    # moves played cards to the discard area after scoring.
    # ------------------------------------------------------------------
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(played)
    # G.play is empty again; the key keeps its resting form (absent until a
    # round ends, then an empty list).
    if resting_play is None:
        del gs["played_cards_area"]
    else:
        gs["played_cards_area"] = resting_play

    # ------------------------------------------------------------------
    # 9. Hand-type play counts are recorded INSIDE score_hand (Phase 4),
    #    matching Lua's evaluate_play which increments played /
    #    played_this_round before joker scoring (Supernova and Card Sharp
    #    read the count mid-score, current play included).  Recording here
    #    again double-counted every play (found by lockstep diff vs live).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 10. Determine next phase
    # ------------------------------------------------------------------
    if blind.beaten_by(gs["chips"]):
        _round_won(gs)
    elif cr["hands_left"] <= 0:
        if not result.saved:
            # Vanilla end_round fires joker end_of_round effects for
            # LOSSES too (state_events.lua:96-110, game_over flag in the
            # context): Turtle Bean decays, rentals charge, perishables
            # tick as you die.  Found by lockstep: hand size differed at
            # GAME_OVER.  end_round runs out of SELECTING_HAND (vanilla's
            # NEW_ROUND), so a hand-size change there draws nothing.
            gs["phase"] = GamePhase.GAME_OVER
            _joker_end_of_round_effects(gs, game_over=True)
            _end_round_win_check(gs)
            # G.GAME.won is never cleared: losing in endless mode after a
            # win keeps it (state_events.lua:111-122).
        else:
            _round_won(gs)
            _dissolve_saver(gs, saver)
    else:
        # ------------------------------------------------------------------
        # 11. More hands — draw to hand and stay in SELECTING_HAND
        #     (update_hand_played -> DRAW_TO_HAND, game.lua:3196-3200).
        #     The Serpent caps the draw at 3; The Fish flips only the newly
        #     drawn cards, per card through Blind.stay_flipped
        #     (blind.lua:617) — the old whole-hand flip here hid cards the
        #     live game left face up (found by lockstep).  No debuff
        #     re-check: a draw never runs debuff_card.
        # ------------------------------------------------------------------
        _draw_to_hand(gs)

    return gs


def _parse_highlighted(gs: dict[str, Any], cards: list) -> None:
    """``CardArea:parse_highlighted`` (cardarea.lua:162-185) for the selected
    cards: the blind's ``debuff_hand`` in check mode, which only sets
    ``blind.triggered`` (blind.lua:519-570)."""
    blind = gs.get("blind")
    if blind is None or not cards:
        return
    from jackdaw.engine.hand_eval import evaluate_hand

    result = evaluate_hand(cards, jokers=gs.get("jokers", []))
    if result.detected_hand == "NULL":
        return
    blind.debuff_hand(cards, result.poker_hands, result.detected_hand, check=True, game_state=gs)


def forced_card_index(hand: list) -> int | None:
    """Index of Cerulean Bell's forced card in *hand*, or None.

    Every play and discard must include it (see ``_require_forced_card``);
    the env action masks read it from here so they offer only those.
    """
    for i, c in enumerate(hand):
        ability = getattr(c, "ability", None)
        if isinstance(ability, dict) and ability.get("forced_selection"):
            return i
    return None


def _require_forced_card(hand: list, indices: tuple[int, ...]) -> None:
    """Cerulean Bell's forced card cannot be deselected in vanilla — a
    play/discard without it is UI-impossible (blind.lua:572-87: the
    forced card is auto-highlighted every deal).  Bug #67 (ES7L222Z):
    the sim rolled the SAME forced card as live but let the policy
    play around it; live's endpoint force-included it and cap-dropped
    the 5th requested card (full house became trips)."""
    forced = forced_card_index(hand)
    if forced is not None and forced not in indices:
        raise IllegalActionError("Forced card (Cerulean Bell) must be in the selection")


def _clear_forced_selection(gs: dict[str, Any]) -> None:
    """Drop Cerulean Bell's forced flag from every playing card.

    Vanilla clears it on all of ``G.playing_cards`` before each play and
    discard (state_events.lua:384, 459) and at round end (:279); the
    redraw's ``drawn_to_hand`` then forces a new card.  Left set, the
    played or discarded card stayed forced when it was dealt again in a
    later round, whatever the blind.

    The key stays, as ``None``: ``Card.set_ability`` keeps it on every card.
    """
    from jackdaw.engine.card_factory import _PLAYING_CARD_AREAS

    for area in _PLAYING_CARD_AREAS:
        for card in gs.get(area) or ():
            if card.ability.get("forced_selection"):
                card.ability["forced_selection"] = None


def _handle_discard(gs: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    """Discard highlighted cards, fire joker contexts, draw replacements.

    Full sequence matching ``state_events.lua:379-448``:

    1. Validate
    2. Sort discarded cards (left-to-right by index)
    3. Fire joker ``pre_discard`` context (Burnt Joker)
    4. Per-card: fire seal effects (Purple Seal → Tarot) + joker ``discard``
       context with ``other_card`` + ``full_hand``
    5. Process side-effects (dollars, card destruction, joker mutations)
    6. Discard cost (Golden Needle challenge)
    7. Decrement discards_left, increment discards_used
    8. Move surviving cards to discard pile
    9. Draw replacements from deck
    10. The Serpent: draw only 3 if not first action
    11. Re-debuff drawn cards for boss blind
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    cr = gs["current_round"]
    if cr["discards_left"] <= 0:
        raise IllegalActionError("No discards remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot discard more than 5 cards")
    if any(i < 0 or i >= len(hand) for i in indices):
        raise IllegalActionError("Card index out of range")
    _require_forced_card(hand, indices)
    _clear_forced_selection(gs)

    # ------------------------------------------------------------------
    # 2. Extract discarded cards in sorted order
    # ------------------------------------------------------------------
    idx_set = set(indices)
    discarded = [hand[i] for i in sorted(indices)]
    # Highlighting the cards ran CardArea:parse_highlighted (cardarea.lua:162-185),
    # whose check-mode debuff_hand leaves blind.triggered set for this hand.
    _parse_highlighted(gs, discarded)
    gs["hand"] = [c for i, c in enumerate(hand) if i not in idx_set]
    # Discarded cards flip face up on leaving the hand (see play handler).
    for c in discarded:
        c.facing = "front"

    # ------------------------------------------------------------------
    # 3. Fire joker pre_discard context (Burnt Joker: level up hand)
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import JokerContext, calculate_joker, effect_flags

    jokers: list = gs.get("jokers", [])
    rng = gs.get("rng")
    game_snap = _build_discard_snapshot(gs, jokers)

    # is_face / is_suit read Pareidolia / Smeared (card.lua:964, 4064):
    # Faceless Joker, Castle.
    flags = effect_flags(jokers)
    pre_discard_effects: list = []
    for joker in jokers:
        if getattr(joker, "debuff", False):
            continue
        ctx = JokerContext(
            pre_discard=True,
            full_hand=discarded,
            jokers=jokers,
            rng=rng,
            game=game_snap,
            **flags,
        )
        result = calculate_joker(joker, ctx)
        if result:
            pre_discard_effects.append(result)

    # Burnt Joker (and each Blueprint/Brainstorm copy): level up the hand
    # get_poker_hand_info finds in the highlighted cards (card.lua:2750-2753);
    # detection honours Four Fingers / Shortcut / Smeared like vanilla's.
    for eff in pre_discard_effects:
        if eff.level_up:
            hand_levels = gs.get("hand_levels")
            if hand_levels is not None:
                from jackdaw.engine.hand_eval import evaluate_hand

                det = evaluate_hand(discarded, jokers=jokers)
                if det.detected_hand and det.detected_hand != "NULL":
                    hand_levels.level_up(det.detected_hand)

    # ------------------------------------------------------------------
    # 4. Per-card: seal effects + joker discard context
    # ------------------------------------------------------------------
    dollars_earned = 0
    destroyed: list = []
    jokers_to_remove: list = []
    purple_tarots: list = []

    for card in discarded:
        # Seal: Purple Seal → random Tarot, append '8ba' (card.lua:2253-2267).
        # calculate_seal returns at once for a debuffed card (card.lua:2243).
        _purple_seal(gs, card, purple_tarots)

        # Fire joker discard context per card
        card_destroyed = False
        for joker in jokers:
            if getattr(joker, "debuff", False):
                continue
            ctx = JokerContext(
                discard=True,
                other_card=card,
                full_hand=discarded,
                jokers=jokers,
                rng=rng,
                game=game_snap,
                **flags,
            )
            result = calculate_joker(joker, ctx)
            if result:
                dollars_earned += result.dollars
                if result.remove:
                    # Trading Card: destroy the discarded card
                    if result.extra and result.extra.get("destroy"):
                        card_destroyed = True
                    else:
                        # Ramen: self-destruct
                        if joker not in jokers_to_remove:
                            jokers_to_remove.append(joker)

        if card_destroyed:
            # Destroyed on discard, a Glass Card shatters on the spot
            # (state_events.lua:413) — Glass Joker counts it.
            if card.ability.get("name") == "Glass Card":
                card.shattered = True
            destroyed.append(card)

    # ------------------------------------------------------------------
    # 5. Process side-effects (the Purple Seal Tarots are made by events
    #    queued during the loop, card.lua:2256-2265)
    # ------------------------------------------------------------------
    if purple_tarots:
        _resolve_create_descriptors(gs, purple_tarots)
    if dollars_earned:
        gs["dollars"] = gs.get("dollars", 0) + dollars_earned

    for joker in jokers_to_remove:
        if joker in jokers:
            jokers.remove(joker)
            joker.remove_from_deck(gs)
            _release_used_key(gs, joker)

    # ------------------------------------------------------------------
    # 6. Discard cost (Golden Needle challenge)
    # ------------------------------------------------------------------
    discard_cost = gs.get("modifiers", {}).get("discard_cost", 0)
    if discard_cost > 0:
        gs["dollars"] = gs.get("dollars", 0) - discard_cost

    # ------------------------------------------------------------------
    # 7. Decrement discards_left, increment discards_used
    # ------------------------------------------------------------------
    cr["discards_left"] -= 1
    cr["discards_used"] += 1

    # ------------------------------------------------------------------
    # 8. Move surviving cards to discard pile
    # ------------------------------------------------------------------
    # remove_playing_cards joker notify (state_events.lua:426)
    _notify_cards_destroyed(gs, destroyed)

    surviving = [c for c in discarded if c not in destroyed]
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(surviving)

    # A Purple Seal's Tarot or an eaten Ramen came or went: its add_to_deck /
    # remove_from_deck queued the boss debuff pass (card.lua:641, 698), which
    # runs after the queued draws to the discard pile and before the draw to
    # hand, so it covers the discarded cards too (a Wild Card under The Club,
    # G263).
    if purple_tarots or jokers_to_remove:
        reapply_blind_debuffs(gs)

    # Track stat
    gs["round_scores"] = gs.get("round_scores", {})
    gs["round_scores"]["cards_discarded"] = gs["round_scores"].get("cards_discarded", 0) + len(
        discarded
    )

    # ------------------------------------------------------------------
    # 9-11. Draw to hand (DRAW_TO_HAND, state_events.lua:438): the
    #       replacements (The Serpent caps them at 3), then the boss's
    #       drawn_to_hand.  No debuff re-check: a draw never runs
    #       debuff_card.
    # ------------------------------------------------------------------
    _draw_to_hand(gs)

    return gs


def _build_discard_snapshot(gs: dict[str, Any], jokers: list) -> Any:
    """Build a GameSnapshot for discard context."""
    from jackdaw.engine.jokers import GameSnapshot

    cr = gs.get("current_round", {})
    return GameSnapshot(
        joker_count=len(jokers),
        joker_slots=gs.get("joker_slots", 5),
        money=gs.get("dollars", 0),
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        discards_used=cr.get("discards_used", 0),
        mail_card_id=cr.get("mail_card", {}).get("id"),
        castle_card_suit=cr.get("castle_card", {}).get("suit"),
        skips=gs.get("skips", 0),
    )


def _handle_cash_out(gs: dict[str, Any]) -> dict[str, Any]:
    """Accept round earnings and proceed to the shop.

    1. Shuffle deck (button_callbacks.lua:2918)
    2. Apply round earnings to dollars
    3. Track previous_round.dollars
    4. Populate shop (jokers, voucher, boosters)
    5. Phase → SHOP
    """
    _require_phase(gs, GamePhase.ROUND_EVAL)

    # The Coupon and D6 Tags' once-per-shop guards reset at cash-out
    # (button_callbacks.lua:2932-2933: shop_free = nil, shop_d6ed = nil).
    gs.pop("shop_free", None)
    gs.pop("shop_d6ed", None)

    rng = gs.get("rng")

    # Shuffle deck at cash-out (button_callbacks.lua:2918)
    # G.deck:shuffle('cashout'..G.GAME.round_resets.ante)
    if rng:
        deck: list = gs.get("deck", [])
        ante = gs.get("round_resets", {}).get("ante", 1)
        cashout_seed = rng.seed("cashout" + str(ante))
        rng.shuffle(deck, cashout_seed)

    earnings = gs.get("round_earnings")
    if earnings:
        gs["dollars"] = gs.get("dollars", 0) + earnings.total
    # cash_out's event zeroes the shop's purchase count
    # (button_callbacks.lua:2928), and ease_chips(0) the round's score (:2948).
    gs["current_round"]["jokers_purchased"] = 0
    # The same event refills the hands and discards for the next round
    # (button_callbacks.lua:2929-2930); new_round sets them again.
    rr = gs["round_resets"]
    bonus = gs.get("round_bonus", {})
    gs["current_round"]["discards_left"] = max(0, rr["discards"] + bonus.get("discards", 0))
    gs["current_round"]["hands_left"] = max(1, rr["hands"] + bonus.get("next_hands", 0))
    gs["chips"] = 0
    _reset_blinds_after_boss(gs)

    gs["previous_round"] = {"dollars": gs.get("dollars", 0)}

    # Populate shop
    _populate_shop(gs)

    gs["phase"] = GamePhase.SHOP
    return gs


def _release_used_key(gs: dict[str, Any], card: Any) -> None:
    """Free a removed card's center key for future pool draws.

    card.lua:4739-4749 (Card:remove): used_jokers[key] is cleared unless
    another copy of the same center is still in play (joker board or
    consumable slots). Without this, pools exhaust permanently — e.g.
    3-4 celestial packs mark all 12 planets and every later draw
    collapses to the empty-pool fallback (all-Pluto packs)."""
    key = getattr(card, "center_key", None)
    if not key:
        return
    for c in gs.get("jokers", []) + gs.get("consumables", []):
        if c is not card and getattr(c, "center_key", None) == key:
            return
    used = gs.get("used_jokers")
    if used:
        used.pop(key, None)


def _handle_buy_card(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase a card from the shop.

    After buying:
    - Joker: add to jokers area, mark in used_jokers
    - Consumable: add to consumables area
    - Playing card: add to deck, fire ``playing_card_added`` joker context
    - Fire ``buying_card`` on all jokers
    """
    _require_phase(gs, GamePhase.SHOP)

    shop_cards: list = gs.get("shop_cards", [])
    if idx < 0 or idx >= len(shop_cards):
        raise IllegalActionError(f"Invalid shop index {idx}")

    from jackdaw.engine.economy import can_afford

    card = shop_cards[idx]
    if not can_afford(gs, card.cost):
        raise IllegalActionError("Cannot afford card")

    # check_for_buy_space (button_callbacks.lua:2392-2402): a Negative card
    # raises the area's limit by one, so an over-full board takes none (G201, G264).
    from jackdaw.engine.actions import buy_space

    card_set = _get_card_set(card)
    if not buy_space(card, gs):
        raise IllegalActionError(
            "No room for joker" if card_set == "Joker" else "No room for consumable"
        )

    gs["dollars"] -= card.cost
    shop_cards.pop(idx)
    # Only a joker counts (button_callbacks.lua:2445-2448).
    if card_set == "Joker":
        gs["current_round"]["jokers_purchased"] = (
            gs.get("current_round", {}).get("jokers_purchased", 0) + 1
        )

    # Place card in appropriate area
    added_playing_card = False
    if card_set == "Joker":
        # The key was marked when the card was made (card.lua:349-354); buying
        # doesn't mark it again, so a mark cleared since stays cleared.
        gs.setdefault("jokers", []).append(card)
        card.add_to_deck(gs)
    elif card_set in ("Tarot", "Planet", "Spectral"):
        gs.setdefault("consumables", []).append(card)
        card.add_to_deck(gs)
    else:
        # c1:add_to_deck() marks a playing card added too (button_callbacks.lua:
        # 2418; card.lua:568-575), which its removal reads (G205).
        card.add_to_deck(gs)
        # G.deck:emplace (button_callbacks.lua:2427): bottom of the deck.
        _emplace_in_deck(gs, card)
        added_playing_card = True

    # Astronomer joining the board runs the all-cards set_cost pass
    # (dump card.lua:786-793) — restores couponed owned costs too.
    if getattr(card, "center_key", "") == "j_astronomer":
        _all_cards_set_cost_pass(gs)

    # Fire buying_card joker context
    _fire_shop_joker_context(gs, buying_card=True)

    # Fire playing_card_added if a playing card was bought
    # (button_callbacks.lua:2429)
    if added_playing_card:
        _playing_card_joker_effects(gs, [card])

    _inflate(gs)
    return gs


def _handle_buy_and_use(
    gs: dict[str, Any], idx: int, targets: tuple[int, ...] | None
) -> dict[str, Any]:
    """Buy a shop consumable and use it straight away (the "Buy and Use" button).

    can_buy_and_use (button_callbacks.lua:77-89) needs the card affordable
    (``cost <= dollars - bankrupt_at`` or free) and usable
    (Card:can_use_consumeable, in the shop's state).  buy_from_shop with
    the 'buy_and_use' id (:2404-2477) skips check_for_buy_space, never
    emplaces the card in G.consumeables, runs the buying_card context and
    the Inflation increment, charges the cost and hands the card to
    use_card, which uses it with no source area; the card then dissolves.
    """
    _require_phase(gs, GamePhase.SHOP)

    from jackdaw.engine.consumables import consumable_block_reason
    from jackdaw.engine.economy import can_afford

    shop_cards: list = gs.get("shop_cards", [])
    if idx < 0 or idx >= len(shop_cards):
        raise IllegalActionError(f"Invalid shop index {idx}")
    card = shop_cards[idx]
    if "consumeable" not in card.ability:
        raise IllegalActionError("Only consumables can be bought and used")
    if not can_afford(gs, card.cost):
        raise IllegalActionError("Cannot afford card")
    if (reason := consumable_block_reason(card, gs, targets, check_use=False)) is not None:
        raise IllegalActionError(reason)

    gs["dollars"] -= card.cost
    shop_cards.pop(idx)
    _fire_shop_joker_context(gs, buying_card=True)
    _inflate(gs)
    from jackdaw.engine.consumables import ankh_check_use

    if card.center_key == "c_ankh" and ankh_check_use(
        gs.get("jokers", []), gs.get("joker_slots", 5)
    ):
        # use_card's check_use returns before anything happens
        # (button_callbacks.lua:2163-2169): the paid card is in no area and
        # never removed, so it is gone, its used_jokers mark kept (G138).
        return gs
    _use_consumable_card(gs, card, targets)
    _release_used_key(gs, card)
    return gs


# States in which vanilla's sell button is live. ROUND_EVAL is excluded
# deliberately: our engine models the cash-out as one atomic step, so
# there is no window there for the player to act.
_SELLABLE_PHASES = frozenset(
    {
        GamePhase.BLIND_SELECT,
        GamePhase.SELECTING_HAND,
        GamePhase.ROUND_EVAL,
        GamePhase.SHOP,
        GamePhase.PACK_OPENING,
    }
)


def _handle_sell_card(gs: dict[str, Any], area: str, idx: int) -> dict[str, Any]:
    """Sell a card for its sell value.

    After selling:
    - Fire ``selling_card`` on all jokers (Campfire +xMult)
    - If joker sold itself: fire ``selling_self``
    """
    # Selling is NOT shop-only. Card:can_sell_card (card.lua:1640) has no
    # state gate at all -- its blockers are cards mid-scoring, a locked
    # controller and STOP_USE -- and vanilla explicitly COMMENTED OUT the
    # blind-select restriction that used to be there. Both the joker and
    # the consumable areas qualify: can_sell_card requires
    # area.config.type == 'joker', and game.lua:2239 gives the
    # consumables area exactly that type.
    #
    # Restricting this to SHOP removed a real tactic from every agent --
    # selling a joker mid-blind to fire selling_self, to dump a
    # perishable before it expires, or to free a slot during a pack.
    if gs.get("phase") not in _SELLABLE_PHASES:
        raise IllegalActionError(f"Cannot sell in phase {gs.get('phase')}")
    if gs.get("STOP_USE", 0) > 0:
        raise IllegalActionError("Cannot sell while STOP_USE is set")

    cards: list = gs.get(area, [])
    if idx < 0 or idx >= len(cards):
        raise IllegalActionError(f"Invalid {area} index {idx}")

    card = cards[idx]
    if getattr(card, "eternal", False):
        raise IllegalActionError("Cannot sell eternal card")

    # Live Card:sell_card dispatches selling_self before the sold card is
    # removed (card.lua:1599). Apply the returned mutation at that point too.
    selling_self_mutations = _fire_selling_self(gs, card)
    if _apply_selling_self_mutations(gs, card, selling_self_mutations):
        _sell_into_round_end(gs, card)
        return gs

    gs["dollars"] = gs.get("dollars", 0) + card.sell_cost
    cards.pop(idx)
    card.remove_from_deck(gs)
    _release_used_key(gs, card)

    # Selling any Joker under Verdant Leaf disables it (card.lua:1616-1621;
    # no disabled check there, and none needed: a second disable of
    # Verdant Leaf changes nothing).
    blind = gs.get("blind")
    if card.ability.get("set") == "Joker" and blind is not None and blind.name == "Verdant Leaf":
        disable_blind(gs)

    # Astronomer leaving the board runs the all-cards set_cost pass
    # (dump card.lua:850) — shop planet prices un-zero, couponed owned
    # costs restore.
    if getattr(card, "center_key", "") == "j_astronomer":
        _all_cards_set_cost_pass(gs)

    # Fire selling_card joker context (Campfire +xMult per card sold)
    _fire_shop_joker_context(gs, selling_card=True)

    return gs


def _fire_selling_self(gs: dict[str, Any], card: Any) -> list[dict[str, Any]]:
    """Dispatch ``selling_self`` for the sold card while it remains owned."""
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    cr = gs.get("current_round", {})
    result = calculate_joker(
        card,
        JokerContext(
            selling_self=True,
            blind=gs.get("blind"),
            jokers=jokers,
            game=GameSnapshot(
                joker_count=len(jokers),
                joker_slots=gs.get("joker_slots", 5),
                money=gs.get("dollars", 0),
                hands_played=cr.get("hands_played", 0),
                discards_used=cr.get("discards_used", 0),
            ),
            rng=gs.get("rng"),
        ),
    )
    if result and result.extra:
        return [result.extra]
    return []


def _sell_into_round_end(gs: dict[str, Any], card: Any) -> None:
    """A sale whose selling_self disabled a boss the score now beats (G161).

    Blind:disable only queues the beaten-boss check (blind.lua:397-406), and
    Card:sell_card queues the payout and the dissolve (card.lua:1601-1637);
    G.FUNCS.sell_card's selling_card loop runs before any of them
    (button_callbacks.lua:2318-2326).  The check's NEW_ROUND state queues
    end_round (game.lua:3248-3256), which runs while the sold card is still
    held (its dissolve removes it on a later timer), before the payout's
    own ease_dollars event; evaluate_round's interest counts the payout.
    Nothing in end_round reads the money, so it is paid first here.
    """
    _fire_shop_joker_context(gs, selling_card=True, exclude=card)
    gs["dollars"] = gs.get("dollars", 0) + card.sell_cost
    _end_round(gs)
    for area in ("jokers", "consumables"):
        held: list = gs.get(area, [])
        if any(c is card for c in held):
            held.remove(card)
            card.remove_from_deck(gs)
            _release_used_key(gs, card)


def _apply_selling_self_mutations(
    gs: dict[str, Any],
    sold_card: Any,
    mutations: list[dict[str, Any]],
) -> bool:
    """Apply Luchador, Invisible Joker, and Diet Cola sell effects.

    True when Luchador disabled a boss the score now beats: the round ends
    once the sale is done (see :func:`_sell_into_round_end`).
    """
    round_over = False
    for mutation in mutations:
        if mutation.get("disable_blind"):
            round_over = disable_blind(gs, end_round=False) or round_over

        if mutation.get("duplicate_random_joker"):
            jokers: list = gs.get("jokers", [])
            candidates = [joker for joker in jokers if joker is not sold_card]
            rng = gs.get("rng")
            if candidates and len(jokers) <= gs.get("joker_slots", 5) and rng is not None:
                from jackdaw.engine.card_factory import copy_card

                chosen, _ = rng.element(candidates, rng.seed("invisible"))
                # copy_card(chosen, nil, nil, nil, chosen.edition and
                # chosen.edition.negative) (card.lua:2384): a new Card, and a
                # Negative joker's copy loses the edition.
                duplicate = copy_card(
                    chosen,
                    strip_edition=bool(chosen.edition and chosen.edition.get("negative")),
                    game_state=gs,
                )
                if "invis_rounds" in duplicate.ability:
                    duplicate.ability["invis_rounds"] = 0
                duplicate.add_to_deck(gs)
                jokers.append(duplicate)

        create = mutation.get("create", {})
        if create.get("type") == "Tag":
            # Diet Cola: live wraps add_tag(Tag('tag_double')) in an event
            # (card.lua:2361), but the tag is held with no immediate effect,
            # so the deferral is unobservable and is not modelled.
            tag_key = create["key"]
            _award_tag(gs, tag_key, source=getattr(sold_card, "center_key", "sell"))
            # Live add_tag fires tag_add on every held tag; a held Double
            # never duplicates another Double (tag.lua:320), so this is inert
            # for Diet Cola itself and guards any future tag-creating joker.
            _check_double_tag(gs, tag_key)
    return round_over


def disable_blind(gs: dict[str, Any], *, end_round: bool = True) -> bool:
    """``G.GAME.blind:disable()`` (blind.lua:356-415).

    Called by Chicot (its setting_blind, deferred past the round's deck
    shuffle, card.lua:2492-2500; and its add_to_deck mid-round,
    card.lua:596-600), Luchador's selling_self (card.lua:2355-2359) and any
    joker sold under Verdant Leaf (card.lua:1616-1621).  Like vanilla it
    has no guard: a second call applies the restorations again.

    With ``end_round=False`` a boss the score now beats does not end the
    round here; the return value says it should, once the caller's own work
    is done (the queued check runs after it, blind.lua:397-406).
    """
    blind = gs.get("blind")
    if blind is None:
        return False
    # The flag, The Wall's /2 and Violet Vessel's /3 (blind.lua:357, 377-380,
    # 393-396).
    blind.disable()
    name = blind.name
    cr = gs.setdefault("current_round", {})
    hand: list = gs.setdefault("hand", [])

    for joker in gs.get("jokers", ()):  # blind.lua:358-360
        joker.facing = "front"
    if name == "The Water" and blind.discards_sub:  # ease_discard (blind.lua:361-363)
        cr["discards_left"] = cr.get("discards_left", 0) + blind.discards_sub
    if name in ("The Wheel", "The House", "The Mark", "The Fish"):  # blind.lua:364-373
        for card in hand:
            card.facing = "front"
    if name == "The Needle" and blind.hands_sub:  # ease_hands_played (blind.lua:374-376)
        cr["hands_left"] = cr.get("hands_left", 0) + blind.hands_sub
    if name == "Cerulean Bell":  # blind.lua:381-385
        _clear_forced_selection(gs)
    game_over = False
    if name == "The Manacle":
        # change_size(1) (a mid-round draw of its own, cardarea.lua:100-105)
        # then draw_from_deck_to_hand(1) (blind.lua:386-390), whose game-over
        # test runs at once — against the limit the queued change_size has
        # not raised yet.
        game_over = gs.get("hand_size", 0) <= 0 and not hand
        change_hand_size(gs, 1)
        if not game_over:
            _draw_from_deck_to_hand(gs, 1)

    # The debuff pass (blind.lua:407-412): disabled, the blind debuffs
    # nothing, so every playing card and joker is freed (a perished joker
    # excepted) and a freed joker's passives come back.
    reapply_blind_debuffs(gs)
    for joker in list(gs.get("jokers", ())):
        blind.debuff_card(joker, is_joker_area=True, game_state=gs)

    if game_over:
        gs["phase"] = GamePhase.GAME_OVER
        return False
    # Queued check (blind.lua:397-406): a disabled boss whose target the
    # score already meets ends the round (G.STATE = NEW_ROUND).
    beaten = (
        blind.boss
        and gs.get("phase") == GamePhase.SELECTING_HAND
        and blind.beaten_by(gs.get("chips", 0))
    )
    if beaten and end_round:
        _end_round(gs)
        return False
    return bool(beaten)


def _handle_use_consumable(
    gs: dict[str, Any], idx: int, targets: tuple[int, ...] | None
) -> dict[str, Any]:
    """Use a consumable from the player's consumable slots.

    Consumables can be used in BLIND_SELECT, SELECTING_HAND, ROUND_EVAL,
    SHOP and PACK_OPENING phases.  The phase does NOT change after use,
    and during a pack no pick is spent: vanilla's use_card just restores
    the open pack when the used card came from G.consumeables
    (button_callbacks.lua:2269).  An Arcana/Spectral pack's dealt hand is
    the target hand.

    Sequence:
    1. Validate phase and index
    2. Pop card from consumables
    3. Use via ``_use_consumable_card`` (builds ConsumableContext,
       calls handler, applies ConsumableResult mutations, tracks
       last_tarot_planet, fires the ``using_consumeable`` joker context)
    """
    _require_phase(
        gs,
        GamePhase.BLIND_SELECT,
        GamePhase.SELECTING_HAND,
        GamePhase.ROUND_EVAL,
        GamePhase.SHOP,
        GamePhase.PACK_OPENING,
    )

    consumables: list = gs.get("consumables", [])
    if idx < 0 or idx >= len(consumables):
        raise IllegalActionError(f"Invalid consumable index {idx}")

    card = consumables[idx]

    # Validate can_use before consuming (matches Balatro's can_use_consumeable)
    from jackdaw.engine.consumables import can_use_consumable, highlighted_cards

    hand: list = gs.get("hand", [])
    highlighted = highlighted_cards(hand, targets)

    if not can_use_consumable(
        card,
        highlighted=highlighted,
        hand_cards=hand,
        jokers=gs.get("jokers", []),
        consumables=consumables,
        joker_limit=gs.get("joker_slots", 5),
        consumable_limit=gs.get("consumable_slots", 2),
        game_state=gs,
    ):
        consumable_name = card.ability.get("name", card.center_key)
        raise IllegalActionError(f"Consumable {consumable_name!r} cannot be used at this time")

    consumables.pop(idx)
    _use_consumable_card(gs, card, targets)
    # The used card dissolves in use_card's last events, after the use's
    # own: Card:remove's remove_from_deck (a Negative's slot, the boss
    # re-check of set_blind(nil, true), card.lua:645-698) and the release of
    # its no-repeat key (card.lua:4741-4748) come after them (G140).  So a
    # used Emperor stays excluded from its own Tarot pool draw (live-verified:
    # LSUNHM8G — sim released early and self-recreated c_emperor where live
    # drew c_justice).
    card.remove_from_deck(gs)
    _release_used_key(gs, card)

    # Phase does NOT change — returns to whatever it was
    return gs


def _handle_redeem_voucher(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase and activate a voucher."""
    _require_phase(gs, GamePhase.SHOP)

    vouchers: list = gs.get("shop_vouchers", [])
    if idx < 0 or idx >= len(vouchers):
        raise IllegalActionError(f"Invalid voucher index {idx}")

    from jackdaw.engine.economy import spendable_dollars

    card = vouchers[idx]
    # can_redeem has no free-item exemption (button_callbacks.lua:97).
    if card.cost > spendable_dollars(gs):
        raise IllegalActionError("Cannot afford voucher")

    gs["dollars"] -= card.cost
    vouchers.pop(idx)
    # The redeemed card dissolves, and with it its used_jokers mark
    # (Card:remove, card.lua:4741-4748).
    _release_used_key(gs, card)

    from jackdaw.engine.vouchers import apply_voucher

    gs["used_vouchers"][card.center_key] = True
    _slots_before = gs.get("shop", {}).get("joker_max", 2)
    apply_voucher(card.center_key, gs)
    # Inflation goes up in Card:redeem itself (card.lua:1858), before the
    # queued Overstock refill below creates its card.
    _inflate(gs)

    # ONLY Clearance Sale / Liquidation run vanilla's ALL-cards set_cost
    # pass (card.lua:1917-1923) — that pass also restores couponed OWNED
    # cards to their real costs (live-verified both ways: LS6G1CWJ's
    # Clearance Sale flipped two $0 tag jokers to $2/$3, while
    # LS1Z615Y's non-price voucher left a foil-tag $0 joker untouched).
    if card.center_key in ("v_clearance_sale", "v_liquidation"):
        # G.I.CARD: every card, playing cards included. The coupon zeroes a
        # price only in the shop areas (card.lua:383); the flag stays.
        _all_cards_set_cost_pass(gs)

    # Overstock / Overstock Plus trigger a FULL shop refill to the new
    # limit (purchase-emptied slots refill too — live rolled two cards
    # when one slot was empty); non-slot vouchers refill nothing.
    # All three behaviors lockstep-confirmed on live.  Each new slot is a
    # create_card_for_shop (change_shop_size, common_events.lua:1112-1115):
    # store_joker_create tags, Illusion, then the edition tags.
    _new_max = gs.get("shop", {}).get("joker_max", 2)
    if _new_max > _slots_before:
        from jackdaw.engine.shop import fill_shop_slots

        _sync_played_hand_types(gs)
        fill_shop_slots(gs, _new_max - len(gs.get("shop_cards", [])))
        _fire_shop_tags(gs, rerolled=True)

    # Clear the voucher slot so the next shop doesn't re-offer it.
    # Matches card.lua:1850: G.GAME.current_round.voucher = nil
    gs.get("current_round", {})["voucher"] = None

    return gs


def _handle_open_booster(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Open a booster pack from the shop and transition to PACK_OPENING.

    Pays (Card:open's ease_dollars, card.lua:1711-1716), removes the pack
    from the shop and opens it (:func:`_open_pack`: open_booster jokers,
    inflation, contents, dealt hand).
    """
    _require_phase(gs, GamePhase.SHOP)

    boosters: list = gs.get("shop_boosters", [])
    if idx < 0 or idx >= len(boosters):
        raise IllegalActionError(f"Invalid booster index {idx}")

    from jackdaw.engine.economy import can_afford

    pack = boosters[idx]
    if not can_afford(gs, pack.cost):
        raise IllegalActionError("Cannot afford booster")

    gs["dollars"] -= pack.cost
    boosters.pop(idx)
    _release_used_key(gs, pack)
    gs["shop_return_phase"] = GamePhase.SHOP
    _open_pack(gs, pack.center_key)
    return gs


def _handle_pick_pack_card(
    gs: dict[str, Any],
    idx: int,
    targets: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Pick a card from an opened booster pack.

    Matching ``button_callbacks.lua:2155-2247`` use_card:

    - **Consumable** (Arcana/Spectral/Celestial): use immediately via
      ``use_consumeable``.  For Arcana/Spectral, ``targets`` specifies
      which dealt hand cards the consumable should target.  Planets
      are used without targets (level up hand type).
    - **Playing card** (Standard pack): added to deck.  Fires
      ``playing_card_added`` joker context (Hologram).
    - **Joker** (Buffoon pack): added to joker slots.  Marks in
      ``used_jokers``.

    When ``pack_choices_remaining`` hits 0 or pack is empty, the pack
    closes: remaining cards are removed, dealt hand cards (if any)
    return to deck, and phase restores to SHOP.
    """
    from jackdaw.engine.consumables import pack_pick_block_reason

    _require_phase(gs, GamePhase.PACK_OPENING)

    pack_cards: list = gs.get("pack_cards", [])
    remaining = gs.get("pack_choices_remaining", 0)
    if remaining <= 0:
        raise IllegalActionError("No pack choices remaining")
    if idx < 0 or idx >= len(pack_cards):
        raise IllegalActionError(f"Invalid pack card index {idx}")

    # Targeting consumables demand their highlight count — the live game
    # rejects e.g. a Chariot pick with no target ("requires exactly 1
    # target card(s)"); the sim used to accept it as a silent no-op and
    # burn the pick (found by lockstep).  Same bounds as
    # can_use_consumable (min_highlighted..mod_num).
    _pick = pack_cards[idx]
    _ability = getattr(_pick, "ability", None) or {}
    if _ability.get("set") in ("Tarot", "Spectral"):
        from jackdaw.engine.card import _resolve_center

        try:
            _cfg = _resolve_center(_pick.center_key).get("config") or {}
        except Exception:
            _cfg = {}
        if isinstance(_cfg, dict) and _cfg.get("max_highlighted"):
            _max_h = _cfg["max_highlighted"]
            _min_h = _cfg.get("min_highlighted", 1)
            _mod_num = _cfg.get("mod_num", _max_h)
            _n = len(targets or ())
            if not (_min_h <= _n <= _mod_num):
                raise IllegalActionError(
                    f"{_pick.center_key} requires between {_min_h} and "
                    f"{_mod_num} target card(s); provided {_n}"
                )

    card = pack_cards[idx]
    card_set = _get_card_set(card)

    # Space check — jokers need a slot (Negative exempt); creator
    # consumables need somewhere to put what they make, gated at USE time
    # in vanilla (can_use_consumeable, card.lua:1550-1563), The Fool also
    # needing a tarot/planet to copy.  The same helper drives the action
    # mask, so a masked-legal pick never raises.
    # NOTE: the smods booster UI skips this gate — live created a 6th
    # joker on a 5-slot board (LSBVJSQL) — but the sim stays
    # vanilla-faithful and the lockstep policy vetoes the pick instead.
    if (_blocked := pack_pick_block_reason(card, gs, targets)) is not None:
        raise IllegalActionError(_blocked)

    pack_cards.pop(idx)
    gs["pack_choices_remaining"] = remaining - 1

    if card_set in ("Tarot", "Planet", "Spectral"):
        # Consumable: use immediately (Arcana/Spectral/Celestial pack);
        # this notifies the jokers (using_consumeable) like a tray use.
        _use_consumable_card(gs, card, targets)
        _release_used_key(gs, card)

    elif card_set == "Joker":
        # Buffoon pack: add to joker slots
        gs.setdefault("jokers", []).append(card)
        gs.setdefault("used_jokers", {})[card.center_key] = True
        card.add_to_deck(gs)
        if getattr(card, "center_key", "") == "j_astronomer":
            _all_cards_set_cost_pass(gs)

    else:
        # Standard pack: playing card → G.deck:emplace, the bottom of the
        # deck (button_callbacks.lua:2222-2224).
        _emplace_in_deck(gs, card)
        # playing_card_joker_effects({card}) (button_callbacks.lua:2228)
        _playing_card_joker_effects(gs, [card])
        # card:add_to_deck() (button_callbacks.lua:2229) marks it added (G205).
        card.add_to_deck(gs)

    # Check if pack should close
    if gs["pack_choices_remaining"] <= 0 or not pack_cards:
        _close_pack(gs)

    return gs


def _handle_skip_pack(gs: dict[str, Any]) -> dict[str, Any]:
    """Skip remaining pack cards.

    Fires ``skipping_booster`` on all jokers (Red Card +mult per skip),
    then closes the pack.
    """
    _require_phase(gs, GamePhase.PACK_OPENING)
    from jackdaw.engine.actions import can_skip_booster

    if not can_skip_booster(gs):
        # The skip button waits for an Arcana or Spectral pack's hand (G202).
        raise IllegalActionError("The pack can't be skipped before its hand is dealt")

    # Fire skipping_booster joker context (Red Card +mult)
    _fire_shop_joker_context(gs, skipping_booster=True)

    _close_pack(gs)
    return gs


def _handle_reroll(gs: dict[str, Any]) -> dict[str, Any]:
    """Reroll the shop.

    After rerolling:
    - Fire ``reroll_shop`` on all jokers (Flash Card +mult)
    - Track times_rerolled stat
    """
    _require_phase(gs, GamePhase.SHOP)

    from jackdaw.engine.economy import can_afford

    cr = gs.get("current_round", {})
    free = cr.get("free_rerolls", 0)
    cost = cr.get("reroll_cost", 5)

    if free > 0:
        cr["free_rerolls"] = free - 1
    elif can_afford(gs, cost):
        gs["dollars"] -= cost
    else:
        raise IllegalActionError("Cannot afford reroll")

    # Only PAID rerolls climb the cost ladder — vanilla's free-reroll
    # branch calls calculate_reroll_cost(true) with skip_increment
    # (button_callbacks.lua:2855; live-confirmed: cost stayed 6 after a
    # Chaos free reroll).  Recalc via the shared helper so temp costs
    # (D6 Tag) and remaining free rerolls are honored.
    if free <= 0:
        cr["reroll_cost_increase"] = cr.get("reroll_cost_increase", 0) + 1
    from jackdaw.engine.shop import calculate_reroll_cost

    calculate_reroll_cost(gs)

    # Track stat
    gs.setdefault("round_scores", {})
    gs["round_scores"]["times_rerolled"] = gs["round_scores"].get("times_rerolled", 0) + 1

    # Regenerate shop joker cards
    _reroll_shop_cards(gs)

    # Fire reroll_shop joker context (Flash Card +mult)
    _fire_shop_joker_context(gs, reroll_shop=True)

    return gs


def _handle_next_round(gs: dict[str, Any]) -> dict[str, Any]:
    """Leave the shop and proceed to the next blind.

    Before leaving:
    - Fire ``ending_shop`` on all jokers (Perkeo copies consumable)
    - Process Perkeo side-effects
    """
    _require_phase(gs, GamePhase.SHOP)

    # Fire ending_shop joker context (Perkeo)
    mutations = _fire_shop_joker_context(gs, ending_shop=True)
    _apply_shop_mutations(gs, mutations)

    # Clear shop areas (unsold cards dissolve -> keys released)
    for c in gs.get("shop_cards", []) + gs.get("shop_vouchers", []) + gs.get("shop_boosters", []):
        _release_used_key(gs, c)
    gs["shop_cards"] = []
    gs["shop_vouchers"] = []
    gs["shop_boosters"] = []

    # Building the blind-select screen picks the blind on deck from the
    # states and marks it Select (UI_definitions.lua:1441-1444, 1489).
    rr = gs["round_resets"]
    states = rr["blind_states"]
    done = ("Defeated", "Skipped", "Hide")
    if states.get("Small") not in done:
        blind_on_deck = "Small"
    elif states.get("Big") not in done:
        blind_on_deck = "Big"
    else:
        blind_on_deck = "Boss"
    gs["blind_on_deck"] = blind_on_deck
    states[blind_on_deck] = "Select"

    gs["phase"] = GamePhase.BLIND_SELECT
    # The blind-select screen is built again (game.lua:3258-3298), rolling
    # this ante's Orbital hands if it has none yet.
    from jackdaw.engine.tags import ensure_orbital_choices

    ensure_orbital_choices(gs)
    return gs


# SortHand.mode -> the CardArea:sort method its button uses
# (button_callbacks.lua:36-48).
_SORT_BUTTON_MODES = {"rank": "desc", "suit": "suit desc"}


def _handle_sort_hand(gs: dict[str, Any], mode: str) -> dict[str, Any]:
    """Press a sort button: ``G.hand:sort('desc')`` (rank) or
    ``G.hand:sort('suit desc')`` (suit).  CardArea:sort stores the method in
    ``G.hand.config.sort`` (cardarea.lua:578), so every later auto-sort of the
    hand uses it too."""
    _require_phase(gs, GamePhase.SELECTING_HAND)

    method = _SORT_BUTTON_MODES.get(mode)
    if method is None:
        raise IllegalActionError(f"Unknown sort mode {mode!r}")
    gs["hand_sort"] = method
    _sort_hand(gs, gs.get("hand", []))
    return gs


def _handle_swap_hand(gs: dict[str, Any], idx: int, direction: int) -> dict[str, Any]:
    """Swap a hand card with its neighbor.

    *direction* is ``-1`` (left) or ``+1`` (right).
    Free action — no cost, doesn't consume hands or discards.
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    hand: list = gs.get("hand", [])
    other = idx + direction
    if not (0 <= idx < len(hand) and 0 <= other < len(hand)):
        raise IllegalActionError("Swap index out of range")

    hand[idx], hand[other] = hand[other], hand[idx]
    return gs


def _handle_swap_jokers(gs: dict[str, Any], idx: int, direction: int) -> dict[str, Any]:
    """Swap a joker with its neighbor.

    *direction* is ``-1`` (left) or ``+1`` (right).
    """
    _require_phase(gs, GamePhase.SELECTING_HAND, GamePhase.SHOP)

    from jackdaw.engine.actions import joker_swap_allowed

    jokers: list = gs.get("jokers", [])
    other = idx + direction
    if not (0 <= idx < len(jokers) and 0 <= other < len(jokers)):
        raise IllegalActionError("Swap index out of range")
    if not joker_swap_allowed(jokers, idx, direction):
        raise IllegalActionError("A pinned joker cannot be moved")

    jokers[idx], jokers[other] = jokers[other], jokers[idx]
    return gs


def _align_pinned_jokers(jokers: list) -> None:
    """Sort pinned jokers to the left, as the joker area does every frame.

    cardarea.lua:528 orders G.jokers by ``T.x - 100*(pinned and sort_id or
    0)``: pinned cards first (the newest pin leftmost), the rest in their
    current order.  Only something that reorders the area (Amber Acorn's
    shuffle) or copies a pinned joker can disturb it.
    """
    if any(getattr(j, "pinned", False) for j in jokers):
        jokers.sort(key=lambda j: -j.sort_id if j.pinned else 0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emplace_in_deck(gs: dict[str, Any], card: Any) -> None:
    """``G.deck:emplace(card)``: a 'deck' CardArea inserts at index 1
    (cardarea.lua:33-34) — the bottom, drawn last.  The deck list's index 0
    is the bottom; draws pop from the end."""
    gs.setdefault("deck", []).insert(0, card)


def _sort_hand(gs: dict[str, Any], hand: list) -> None:
    """``G.hand:sort()``: sort the hand in place with its stored mode.

    CardArea:sort with no argument uses ``G.hand.config.sort``
    (cardarea.lua:577-590): ``'desc'`` (the default) sorts by
    ``get_nominal()`` descending, ``'suit desc'`` by ``get_nominal('suit')``
    descending.  The keys combine rank, suit tiebreaks, face nominal and a
    unique micro-value, so every card gets a distinct position.

    Vanilla sorts the hand when cards are drawn into it
    (``draw_card(..., sort=true)``, common_events.lua:418-420), after a
    hand-size draw (cardarea.lua:104) and after Certificate's card
    (card.lua:2474) — not when cards are created into it.
    """
    nominal_mod = "suit" if gs.get("hand_sort", "desc") == "suit desc" else None
    hand.sort(key=lambda c: c.get_nominal(nominal_mod), reverse=True)


def _draw_hand(gs: dict[str, Any], count: int | None = None) -> None:
    """Draw cards from deck to fill the hand up to hand_size.

    Cards are drawn from the END of the deck list (top of the visual
    stack), matching Lua's ``draw_card(G.deck, G.hand, ...)`` which
    pops from the last position.  *count* overrides the number drawn
    (The Serpent's 3, state_events.lua:363-368).

    After drawing, the hand is sorted with its stored mode (matching Lua's
    ``draw_from_deck_to_hand`` which passes ``sort=true`` to ``draw_card``,
    triggering ``CardArea:sort()``).  Its ``hand_space`` is
    ``min(#deck, limit - #hand)`` (state_events.lua:362): with nothing to
    draw there is no draw_card call, so no sort either.
    """
    _sync_hand_size_to_dollars(gs)
    hand: list = gs.setdefault("hand", [])
    _draw_cards(gs, min(len(gs.get("deck", [])), gs.get("hand_size", 8) - len(hand)))


def _draw_from_deck_to_hand(gs: dict[str, Any], count: int | None = None) -> bool:
    """``G.FUNCS.draw_from_deck_to_hand(e)`` (state_events.lua:355-377).

    With a hand limit of 0 or less and an empty hand the game is over
    (outside a pack; the engine's packs deal their own hand) — returns True.
    Otherwise it draws *count* cards, by default ``min(#deck, limit - #hand)``;
    under an active Serpent, once the round's first play or discard is done,
    at most 3.
    """
    _sync_hand_size_to_dollars(gs)
    hand: list = gs.setdefault("hand", [])
    hand_size = gs.get("hand_size", 8)
    if hand_size <= 0 and not hand:
        gs["phase"] = GamePhase.GAME_OVER
        return True
    deck: list = gs.get("deck", [])
    hand_space = count if count is not None else min(len(deck), hand_size - len(hand))
    blind = gs.get("blind")
    cr = gs.get("current_round", {})
    if (
        blind is not None
        and blind.name == "The Serpent"
        and not blind.disabled
        and (cr.get("hands_played", 0) > 0 or cr.get("discards_used", 0) > 0)
    ):
        hand_space = min(len(deck), 3)
    _draw_cards(gs, hand_space)
    return False


def _fire_first_hand_drawn(gs: dict[str, Any]) -> None:
    """The first_hand_drawn joker context (game.lua:3226-3231): Certificate
    deals a sealed card into the hand ('cert_fr' front, 'certsl' seal,
    card.lua:2462-2479; live-verified: LSI5EB7T dealt 8 + a red-seal H_K as
    the 9th card) and runs ``G.GAME.blind:debuff_card`` on it."""
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    snap = GameSnapshot(joker_count=len(jokers), money=gs.get("dollars", 0))
    mutations: list[dict[str, Any]] = []
    for joker in jokers:
        if joker.debuff:
            continue
        res = calculate_joker(joker, JokerContext(first_hand_drawn=True, jokers=jokers, game=snap))
        if res and res.extra:
            mutations.append(dict(res.extra))
            # Certificate calls playing_card_joker_effects({true}) at calc
            # time; the card itself is made in an event (card.lua:2464-2481).
            if (res.extra.get("create") or {}).get("key") == "cert":
                _playing_card_joker_effects(gs, [True])
    if mutations:
        before = list(gs.get("hand", []))
        _apply_setting_blind_mutations(gs, mutations, jokers)
        for c in gs.get("hand", []):
            if all(c is not b for b in before):
                blind_debuff_card(gs, c)


def _blind_drawn_to_hand(gs: dict[str, Any]) -> None:
    """``G.GAME.blind:drawn_to_hand()`` — for every blind, so the prepped
    flag clears after each draw (blind.lua:572-603)."""
    blind = gs.get("blind")
    if blind is None:
        return
    hand: list = gs.get("hand", [])
    dth = blind.drawn_to_hand(
        hand_cards=hand,
        joker_cards=gs.get("jokers", []),
        rng=gs.get("rng"),
        game_state=gs,
    )
    idx = dth.get("forced_card_index")
    if idx is not None and 0 <= idx < len(hand):
        hand[idx].ability["forced_selection"] = True


def _draw_to_hand(gs: dict[str, Any]) -> None:
    """Game:update_draw_to_hand (game.lua:3208-3245), after the
    round_start_bonus tags: draw from the deck (which may end the game),
    fire first_hand_drawn on the round's first draw, then — in a later
    event — enter SELECTING_HAND and call Blind:drawn_to_hand.  Certificate's
    card is therefore in the hand when Cerulean Bell picks its forced card.
    """
    if _draw_from_deck_to_hand(gs):
        return
    cr = gs.get("current_round", {})
    if cr.get("hands_played", 0) == 0 and cr.get("discards_used", 0) == 0:
        _fire_first_hand_drawn(gs)
    gs["phase"] = GamePhase.SELECTING_HAND
    _blind_drawn_to_hand(gs)


def _update_selecting_hand(gs: dict[str, Any]) -> None:
    """The SELECTING_HAND branch of Game:update.  An empty hand with cards
    left in the deck goes back to DRAW_TO_HAND (game.lua:2544-2548); with the
    deck (and the play area) empty too, update_selecting_hand calls
    end_round (game.lua:3056-3069)."""
    if gs.get("phase") != GamePhase.SELECTING_HAND or gs.get("hand"):
        return
    if gs.get("deck"):
        _draw_to_hand(gs)
    else:
        # Vanilla calls end_round() from every frame of that state, queueing
        # one end_round event per frame until the first one leaves it; how
        # many run depends on the frame rate.  The engine runs it once.
        _end_round(gs)


def _mr_bones_saver(gs: dict[str, Any]) -> Any:
    """The joker whose game_over end_of_round pass saves the run (Mr. Bones
    at 25% of the target, card.lua:3047-3055), or None."""
    import math

    from jackdaw.engine.jokers import JokerContext, calculate_joker

    blind_chips = gs["blind"].chips
    if not 0 < blind_chips < math.inf:  # NaN / inf targets: no ratio
        return None
    ratio = gs.get("chips", 0) / blind_chips
    jokers: list = gs.get("jokers", [])
    for joker in jokers:
        if joker.debuff:
            continue
        ctx = JokerContext(jokers=jokers, rng=gs.get("rng"), blind=gs["blind"])
        ctx.game_over = True  # type: ignore[attr-defined]
        ctx.chips_ratio = ratio  # type: ignore[attr-defined]
        res = calculate_joker(joker, ctx)
        if res and res.saved:
            return joker
    return None


def _bones_saver(result: Any) -> Any:
    """The Mr. Bones whose game_over check saved this play's round (score_hand
    lists it among the jokers to remove), or None."""
    if not result.saved:
        return None
    return next((j for j in result.jokers_removed if j.center_key == "j_mr_bones"), None)


def _dissolve_saver(gs: dict[str, Any], saver: Any) -> None:
    """Mr. Bones' save queues the event that dissolves it (card.lua:3049-3057);
    it runs after end_round's joker loop, which still charges its rent and
    ticks its perishable sticker (state_events.lua:99-110, G101)."""
    jokers: list = gs.get("jokers", [])
    if saver is not None and saver in jokers:
        jokers.remove(saver)
        saver.remove_from_deck(gs)
        _release_used_key(gs, saver)


def _end_round(gs: dict[str, Any]) -> None:
    """end_round() for a round left without a play (state_events.lua:87-120):
    won if the score already meets the target, saved by Mr. Bones, else lost
    — the end_of_round pass runs either way."""
    if gs["blind"].beaten_by(gs.get("chips", 0)):
        _round_won(gs)
        return
    saver = _mr_bones_saver(gs)
    if saver is not None:
        _round_won(gs)
        _dissolve_saver(gs, saver)
        return
    gs["phase"] = GamePhase.GAME_OVER
    _joker_end_of_round_effects(gs, game_over=True)
    _end_round_win_check(gs)
    # G.GAME.won is never cleared (state_events.lua:111-122).


def _end_round_win_check(gs: dict[str, Any]) -> None:
    """end_round's win check (state_events.lua:111-114): the boss of exactly
    ``win_ante`` sets ``G.GAME.won``.  It runs after the jokers' end_of_round
    pass and before the game-over test, so losing that boss marks the run
    won too (G260); a boss of a later ante (endless) doesn't."""
    if gs.get("blind_on_deck") == "Boss" and gs["round_resets"]["ante"] == gs.get("win_ante", 8):
        gs["won"] = True


def _boss_flags(gs: dict[str, Any]) -> tuple[bool, bool]:
    """(Pareidolia, Smeared Joker) for debuff checks: ``find_joker`` skips
    debuffed jokers (misc_functions.lua:903-917)."""
    pareidolia = smeared = False
    for j in gs.get("jokers", ()):
        if not j.debuff:
            key = j.center_key
            pareidolia = pareidolia or key == "j_pareidolia"
            smeared = smeared or key == "j_smeared"
    return pareidolia, smeared


def blind_debuff_card(gs: dict[str, Any], card: Any) -> None:
    """``G.GAME.blind:debuff_card(card)`` (blind.lua:624-652) on one card."""
    blind = gs.get("blind")
    if blind is None:
        return
    pareidolia, smeared = _boss_flags(gs)
    blind.debuff_card(
        card,
        is_joker_area=any(j is card for j in gs.get("jokers", ())),
        pareidolia=pareidolia,
        smeared=smeared,
        game_state=gs,
    )


def reapply_blind_debuffs(gs: dict[str, Any]) -> None:
    """``G.GAME.blind:set_blind(nil, true)``: the reset re-runs debuff_card
    over every playing card (blind.lua:207-210) and leaves the jokers alone
    (blind.lua:211-213); none of set_blind's set-time effects run.  Queued
    by every joker or consumable's add_to_deck / remove_from_deck
    (card.lua:641, 698), and also set_blind's own pass at round start."""
    blind = gs.get("blind")
    if blind is None:
        return
    from jackdaw.engine.card_factory import _PLAYING_CARD_AREAS

    pareidolia, smeared = _boss_flags(gs)
    for area in (*_PLAYING_CARD_AREAS, "pending_deck_bottom"):
        for card in gs.get(area) or ():
            blind.debuff_card(card, pareidolia=pareidolia, smeared=smeared)


def _set_blind_debuff_pass(gs: dict[str, Any]) -> None:
    """set_blind's own debuff pass (not a reset, blind.lua:207-213): every
    playing card, then every joker — which un-debuffs a joker unless the
    blind is an active Crimson Heart (blind.lua:647-649)."""
    reapply_blind_debuffs(gs)
    blind = gs.get("blind")
    if blind is None:
        return
    for joker in list(gs.get("jokers", ())):
        blind.debuff_card(joker, is_joker_area=True, game_state=gs)


def change_hand_size(gs: dict[str, Any], delta: int) -> None:
    """``G.hand:change_size(delta)`` (cardarea.lua:94-111).

    The limit moves by *delta* (``gs["hand_size"]`` is the unclamped
    ``real_card_limit``).  When it grows to more than 1 while the hand holds
    cards in SELECTING_HAND, *delta* cards are drawn from the deck at once
    and the hand is sorted.  Vanilla also draws in DRAW_TO_HAND, which the
    engine folds into its draw steps; every hand-size change that happens
    there finds the hand empty.  Round-end processing runs outside
    SELECTING_HAND, as vanilla's end_round runs in NEW_ROUND.
    """
    if not delta:
        return
    gs["hand_size"] = gs.get("hand_size", 0) + delta
    # While a consumable is used G.STATE is PLAY_TAROT (button_callbacks.lua:
    # 2181-2186) until use_card's last event restores it, and every
    # change_size event the use queues (a joker made, copied or destroyed)
    # runs before that: the hand grows without a draw (G133).
    if (
        delta > 0
        and gs["hand_size"] > 1
        and gs.get("hand")
        and gs.get("phase") == GamePhase.SELECTING_HAND
        and not gs.get(_PLAY_TAROT)
    ):
        _draw_cards(gs, delta)


def _draw_cards(gs: dict[str, Any], count: int) -> None:
    """*count* ``draw_card(G.deck, G.hand, ...)`` calls: each takes the top
    card of the deck if there is one, through ``draw_card_from``'s boss
    face-down check (cardarea.lua:592-610), and the hand is sorted after
    (common_events.lua:418-420) — even when the deck had run dry."""
    if count <= 0:
        return
    deck: list = gs.get("deck", [])
    hand: list = gs.setdefault("hand", [])
    to_draw = min(len(deck), count)

    # Boss face-down draws (Blind:stay_flipped, blind.lua:605-622): The
    # Wheel (seeded 'wheel' roll per drawn card), The House (first hand of
    # the round), The Mark (face cards).  The Fish flips on redraw via its
    # prepped flag (handled at the redraw sites).  Evaluated per card in
    # draw order BEFORE the hand sort, so The Wheel's stream consumption
    # matches Lua exactly.
    blind = gs.get("blind")
    check_flip = (
        blind is not None and getattr(blind, "boss", None) and not getattr(blind, "disabled", False)
    )
    if check_flip:
        pareidolia = any(
            getattr(j, "center_key", None) == "j_pareidolia" and not getattr(j, "debuff", False)
            for j in gs.get("jokers", [])
        )

    for _ in range(to_draw):
        if deck:
            card = deck.pop()
            flip = bool(check_flip) and blind.stay_flipped(
                card,
                rng=gs.get("rng"),
                probabilities_normal=gs.get("probabilities", {}).get("normal", 1.0),
                hands_played=gs.get("current_round", {}).get("hands_played", 0),
                discards_used=gs.get("current_round", {}).get("discards_used", 0),
                pareidolia=pareidolia,
            )
            flip = _x_ray_flip(gs) or flip
            # CardArea:emplace (cardarea.lua:38) turns a face-down card face
            # up unless it stays flipped.
            card.facing = "back" if flip else "front"
            hand.append(card)
    # draw_card(..., sort=true) sorts with the hand's stored mode
    _sort_hand(gs, hand)


def _x_ray_flip(gs: dict[str, Any]) -> bool:
    """X-ray Vision's roll for one card drawn into the hand.

    cardarea.lua:601-605 (and common_events.lua:403-407 for a named card):
    after the boss's stay_flipped check, under ``modifiers.flipped_cards``
    every card drawn to G.hand rolls ``'flipped_card'`` and stays face down
    below ``1/flipped_cards``.
    """
    n = (gs.get("modifiers") or {}).get("flipped_cards")
    return bool(n) and gs["rng"].random("flipped_card") < 1 / n


def _sync_hand_size_to_dollars(gs: dict[str, Any]) -> None:
    """Luxury Tax: -1 hand size per ``minus_hand_size_per_X_dollar`` dollars.

    cardarea.lua:245-251: G.hand's update compares ``floor(dollars/X)``
    with ``config.last_poll_size`` (nil = 0) every frame and changes the
    hand size by the difference.  The sim checks after every action and
    before every draw, which is where vanilla's frames would have caught up.
    """
    x = (gs.get("modifiers") or {}).get("minus_hand_size_per_X_dollar")
    if not x:
        return
    poll = math.floor(gs.get("dollars", 0) / x)
    last = gs.get("hand_last_poll_size", 0)
    if poll != last:
        gs["hand_size"] = gs.get("hand_size", 8) + (last - poll)
        gs["hand_last_poll_size"] = poll


def _joker_end_of_round_effects(gs: dict[str, Any], *, game_over: bool = False) -> dict[str, Any]:
    """Fire joker end_of_round context + perishable/rental processing.

    Vanilla's end_round (state_events.lua:96-110) runs this for EVERY
    round outcome — wins AND losses (game_over is a flag inside the
    context; Turtle Bean decays, Egg gains, perishables tick as you die).
    Dollars returned here are only ever banked via the cash-out screen,
    which a lost run never reaches.

    ``game_over``: the run is lost.  The game pauses in the same frame
    (update_game_over, game.lua:2601) and never runs the events end_round
    queued (G265): the rent (ease_dollars) is never charged, the jokers that
    destroy themselves stay, and the hand-size and discard changes (Turtle
    Bean's decay, a perished Juggler or Drunkard: change_size and
    ease_discard are events, cardarea.lua:94-111, common_events.lua:112-131)
    never happen.  The synchronous effects remain: the perishable ticks and
    debuffs, Turtle Bean's own h_size, the extinction flag, Egg, Gift Card.
    """
    from jackdaw.engine.jokers import (
        GameSnapshot,
        JokerContext,
        calc_dollar_bonus,
        calculate_joker,
        count_planets_used,
    )
    from jackdaw.engine.round_lifecycle import calculate_perishable, calculate_rental

    cr = gs.get("current_round", {})
    jokers = gs.get("jokers", [])
    rng = gs.get("rng")
    # What only end_round's events change: kept when the run is lost (G265).
    held = (gs.get("dollars", 0), gs.get("hand_size", 0), cr.get("discards_left", 0))

    # Cards have not yet returned to the deck here, so the full owned set
    # spans all four piles (matches vanilla's G.playing_cards).
    _all_owned = (
        gs.get("deck", [])
        + gs.get("hand", [])
        + gs.get("discard_pile", [])
        + gs.get("played_cards_area", [])
    )
    game_snap = GameSnapshot(
        money=gs.get("dollars", 0),
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        # Delayed Gratification pays ONLY when no discard was used this
        # round (card.lua:1675).
        discards_used=cr.get("discards_used", 0),
        # Cloud 9 pays per 9 in the full deck; get_id() == 9 excludes
        # Stone cards, matching Card:update's tally (card.lua:4192).
        nine_tally=sum(1 for c in _all_owned if c.get_id() == 9),
        joker_count=len(jokers),
        # Gros Michel / Cavendish roll against G.GAME.probabilities.normal
        # (card.lua:3020), which Oops! All 6s doubles.
        probabilities_normal=gs.get("probabilities", {}).get("normal", 1),
        planets_used=count_planets_used(gs),
    )
    # end_round's joker loop (state_events.lua:99-110): each joker in turn
    # gets its end_of_round context, then calculate_rental, then
    # calculate_perishable.  A joker that destroys itself is only removed
    # by a later event, so it still pays its rent in the same pass; one
    # that perishes is debuffed through set_debuff and loses its passives.
    ctx = JokerContext(
        end_of_round=True,
        jokers=jokers,
        rng=rng,
        game=game_snap,
        hand_levels=gs.get("hand_levels"),
        # Rocket's boss bump reads the just-finished blind
        # (card.lua:2896); fires on losses too (bug #15 semantics).
        blind=gs.get("blind"),
    )
    removed: list = []
    mutations: list[dict[str, Any]] = []
    for joker in list(jokers):
        if not joker.debuff:
            result = calculate_joker(joker, ctx)
            if result:
                if result.remove:
                    removed.append(joker)
                if result.extra:
                    mutations.append(result.extra)
        calculate_rental(joker, gs)
        calculate_perishable(joker, gs)
    if game_over:
        removed = []
    eor: dict[str, Any] = {"jokers_removed": removed, "mutations": mutations}

    for removed_joker in eor.get("jokers_removed", []):
        if removed_joker in jokers:
            jokers.remove(removed_joker)
            removed_joker.remove_from_deck(gs)
            _release_used_key(gs, removed_joker)
    # End-of-round joker mutations (Turtle Bean's per-round hand-size decay)
    for mut in eor.get("mutations", []):
        if mut.get("hand_size_delta"):
            change_hand_size(gs, mut["hand_size_delta"])
        # Gros Michel's self-destruct marks the species extinct
        # (card.lua:3037): its no_pool_flag drops it from the Joker pool
        # and Cavendish's yes_pool_flag admits Cavendish.
        if mut.get("pool_flag"):
            gs.setdefault("pool_flags", {})[mut["pool_flag"]] = True
        # Egg's round-end bump runs self:set_cost() (card.lua:2985-2988) —
        # a set_cost trigger that restores a couponed buy cost too (the
        # coupon zero applies only in shop areas, card.lua:383).  The
        # card keeps ability.couponed: Lua never clears it (G99).
        for c in mut.get("set_cost_cards", []):
            if hasattr(c, "set_cost"):
                _reprice(gs, c)
        if mut.get("gift_card_bump"):
            # Gift Card (card.lua:2993-3010): +extra_value to every joker
            # AND consumable, each followed by set_cost — which also
            # restores a couponed card's real buy cost (the coupon zero
            # applies only in shop areas, card.lua:383); couponed stays (G99).
            # set_cost reads Astronomer for a held Planet (card.lua:380).
            bump = mut["gift_card_bump"]
            for c in gs.get("jokers", []) + gs.get("consumables", []):
                c.ability["extra_value"] = c.ability.get("extra_value", 0) + bump
                if hasattr(c, "set_cost"):
                    _reprice(gs, c)

    if game_over:
        gs["dollars"], gs["hand_size"], cr["discards_left"] = held

    # calculate_dollar_bonus runs later, in evaluate_round
    # (state_events.lua:1175-1182): after the perishable ticks — a joker
    # that perished this round is debuffed and pays nothing
    # (card.lua:1655-1656) — and after the self-destructions.  Dollars are
    # NOT banked here: vanilla pays the whole round total in one
    # ease_dollars at the cash-out press, with interest computed on
    # pre-payout money; the value flows to calculate_round_earnings.
    eor["dollars_earned"] = sum(calc_dollar_bonus(j, game_snap) for j in jokers)
    return eor


def _round_won(gs: dict[str, Any]) -> None:
    """Handle winning a round — transition to ROUND_EVAL.

    Full sequence matching ``state_events.lua:87-120``:

    1. Fire joker ``end_of_round`` context (economy + scaling)
    2. Process perishable/rental (round_lifecycle)
    3. Held-card effects: Blue Seal planet for the last hand played
       (Gold Seal has NO held effect — it pays on play+score only)
    4. Return all cards to deck (hand + played + discard)
    5. Un-debuff all playing cards (blind debuffs don't persist)
    6. Track unused discards (for Garbage Tag)
    7. Mark blind as Defeated
    8. Advance blind progression (Small→Big, Big→Boss)
    9. Boss beaten: check win condition, advance ante
    10. Calculate round earnings
    11. Phase → ROUND_EVAL
    """
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.economy import calculate_round_earnings

    cr = gs["current_round"]
    blind = gs["blind"]
    jokers = gs.get("jokers", [])
    rng = gs.get("rng")
    # end_round runs in NEW_ROUND, not SELECTING_HAND: a hand-size change
    # from here on (a perishing Merry Andy, The Manacle's restore) draws
    # nothing (cardarea.lua:100).
    gs["phase"] = GamePhase.ROUND_EVAL

    # end_round adds the round's discards left to unused_discards
    # (state_events.lua:124) before the ease_discard events its joker pass
    # queued have run: a perishing Drunkard or Merry Andy's remove_from_deck
    # takes its discard away in an event (card.lua:686-693, G262).
    discards_left = cr.get("discards_left", 0)

    # ------------------------------------------------------------------
    # 1-2. Joker end_of_round effects + perishable/rental — shared with
    # the LOSS path (vanilla's end_round fires these for every outcome,
    # state_events.lua:96-110).
    # ------------------------------------------------------------------
    eor = _joker_end_of_round_effects(gs)

    # ------------------------------------------------------------------
    # 3. Held-card end-of-round effects (card.lua:1033-65): only
    #    h_dollars (Gold CARD enhancement) and Blue Seal planets.  Gold
    #    SEAL has NO held effect — it pays $3 when played+scoring only
    #    (get_p_dollars, card.lua:1071-73); the old +$3-per-held-gold-
    #    seal block here was invented (live-verified LSGLNPN9: sim +$3
    #    for a Certificate gold-seal Ace held at round end).
    #    Blue Seal creates the planet for the LAST hand played
    #    (G.GAME.last_hand_played, card.lua:1047-53) — NOT most-played —
    #    via the descriptor path so the forced-key create registers in
    #    used_jokers and room-gates like vanilla's 'blusl' create_card.
    # ------------------------------------------------------------------
    hand: list = gs.get("hand", [])
    _held_cards_end_of_round(gs, hand, jokers)

    # ------------------------------------------------------------------
    # 4. Return all cards to deck
    #
    # Lua sequence (state_events.lua:237-250):
    #   a) draw_from_hand_to_discard — hand cards removed first-first,
    #      appended at end of discard
    #   b) draw_from_discard_to_deck — discard cards popped LAST-first
    #      (remove_card on discard type takes #cards), then INSERTED AT
    #      FRONT of deck (emplace on deck type does table.insert(1))
    #
    # Net effect: [old_discard, hand] is prepended to deck front in
    # original order (pop-last + insert-at-front cancel out).
    # ------------------------------------------------------------------
    deck: list = gs.setdefault("deck", [])
    played: list = gs.get("played_cards_area", [])
    discarded: list = gs.get("discard_pile", [])

    # Step a: hand → discard end (forward order)
    discarded.extend(hand)
    # Any leftover played cards (shouldn't normally exist post-scoring)
    discarded.extend(played)
    # Step b: discard → deck FRONT (pop-last + insert-at-front = original order at front)
    deck[:0] = discarded

    gs["hand"] = []
    gs["played_cards_area"] = []
    gs["discard_pile"] = []

    # ------------------------------------------------------------------
    # 5. Flip everything face up (The Fish/House/Wheel/Mark flips and
    #    Amber Acorn's joker flip, blind.lua:338-340, last only for the
    #    blind).  Cerulean Bell's forced flag goes too
    #    (state_events.lua:279).  The debuffs go at the defeat, below.
    # ------------------------------------------------------------------
    for card in deck:
        card.facing = "front"
    for j in jokers:
        j.facing = "front"
    _clear_forced_selection(gs)

    # ------------------------------------------------------------------
    # 6. Track unused discards / hands played (for Garbage/Handy Tags)
    #    These are run-level cumulative totals matching Lua:
    #    - G.GAME.unused_discards += current_round.discards_left  (state_events.lua:124)
    #    - G.GAME.hands_played += 1 per hand  (state_events.lua:523, tracked at line 522)
    # ------------------------------------------------------------------
    gs["unused_discards"] = gs.get("unused_discards", 0) + discards_left

    # 6b. After a boss, end_round re-tallies the most played hand — the one
    #     The Ox reads until the next boss falls (state_events.lua:129-138).
    if blind.get_type() == "Boss" and gs.get("hand_levels") is not None:
        from jackdaw.engine.hand_levels import most_played_poker_hand

        cr["most_played_poker_hand"] = most_played_poker_hand(gs["hand_levels"])

    # ------------------------------------------------------------------
    # 7. Mark blind as Defeated
    # ------------------------------------------------------------------
    rr = gs["round_resets"]
    # Revert Juggle Tag's one-round hand-size bonus
    # (state_events.lua:270 — temp_handsize cleared at round end).
    if rr.get("temp_handsize"):
        change_hand_size(gs, -rr.pop("temp_handsize"))
    # end_round's last event marks the blind Defeated (state_events.lua:258-264);
    # blind_on_deck moves on only when the blind-select screen is built
    # (UI_definitions.lua:1441, _handle_next_round).
    blind_on_deck = gs.get("blind_on_deck", "Small")
    rr["blind_states"][blind_on_deck] = "Defeated"

    # ------------------------------------------------------------------
    # 8-9. A boss falls: the ante goes up (end_round's ease_ante,
    #      state_events.lua:248) and the next ante's voucher is rolled
    #      (state_events.lua:263); the tags, boss and blind states follow
    #      at cash-out.
    # ------------------------------------------------------------------
    if blind_on_deck == "Boss":
        # Boss beaten: end_round's win check, then the ante goes up.
        _end_round_win_check(gs)
        # Clear per-ante play tracking (The Pillar): vanilla nils
        # played_this_ante on every playing card at boss defeat only
        # (state_events.lua:266).
        for card in deck:
            ability = getattr(card, "ability", None)
            if isinstance(ability, dict):
                ability.pop("played_this_ante", None)
        # Typecast (state_events.lua:240-247), before the ante goes up: at
        # the boss of set_eternal_ante every joker turns Eternal (if it can,
        # card.lua:506), and at set_joker_slots_ante the slots drop to 0.
        mods = gs.get("modifiers") or {}
        if mods.get("set_eternal_ante") == rr["ante"]:
            from jackdaw.engine.card_factory import set_eternal_checked

            for j in jokers:
                set_eternal_checked(j)
        if mods.get("set_joker_slots_ante") == rr["ante"]:
            gs["joker_slots"] = 0
        _advance_ante(gs)

    # The rest of end_round's last event (state_events.lua:271-276): the D6
    # Tag's reroll price lapses, and the Idol / Mail-In Rebate / Ancient Joker
    # / Castle targets are re-rolled for the next round (with the new ante
    # after a boss).
    if rr.get("temp_reroll_cost") is not None:
        rr["temp_reroll_cost"] = None
        from jackdaw.engine.run_init import _calculate_reroll_cost

        _calculate_reroll_cost(gs, skip_increment=True)
    if rng:
        from jackdaw.engine.round_lifecycle import reset_round_targets

        reset_round_targets(rng, rr["ante"], gs)

    # ------------------------------------------------------------------
    # 10. Calculate round earnings (for cash-out screen)
    # ------------------------------------------------------------------
    earnings = calculate_round_earnings(
        blind=blind,
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        money=gs.get("dollars", 0),
        jokers=jokers,
        game_state=gs,
        rng=rng,
        joker_dollars=eor.get("dollars_earned", 0),
    )
    # evaluate_round's tag rows (state_events.lua:1180-1188): the Investment
    # Tag pays after a boss (tag.lua:115-129) — part of the cash-out total.
    if gs.get("last_blind_was_boss"):
        from jackdaw.engine.tags import Tag

        for entry in gs.get("awarded_tags", []):
            if entry.get("eval_fired"):
                continue
            result = Tag(entry.get("key", "")).apply("eval", gs, rng=rng, last_blind_is_boss=True)
            if result is not None and result.dollars:
                earnings.total += result.dollars
                entry["eval_fired"] = True
    gs["round_earnings"] = earnings

    # evaluate_round runs the back's 'eval' trigger (state_events.lua:1163):
    # the Anaglyph Deck queues add_tag(Tag('tag_double')) after a boss
    # (back.lua:111-120; last_blind is the blind just beaten).
    from jackdaw.engine.back import Back

    back_key = gs.get("selected_back_key") or "b_red"
    back_effect = Back(back_key).trigger_effect("eval", boss_defeated=blind_on_deck == "Boss")
    if back_effect and back_effect.get("create_tag"):
        tag_key = back_effect["create_tag"]
        _award_tag(gs, tag_key, source=back_key)
        _check_double_tag(gs, tag_key)

    # ------------------------------------------------------------------
    # 10b. Blind:defeat (blind.lua:276-344), run by evaluate_round after
    #      its dollar rows (state_events.lua:1148-1155).  The Manacle
    #      gives its hand size back; the queued set_blind(nil, nil, true)
    #      leaves the empty blind for the cash-out screen and the shop,
    #      and its debuff pass frees every playing card and joker
    #      (Crimson Heart's victim included) — a perished joker stays
    #      debuffed (card.lua:527-533).
    # ------------------------------------------------------------------
    if getattr(blind, "name", "") == "The Manacle" and not getattr(blind, "disabled", False):
        change_hand_size(gs, 1)
    gs["blind"] = Blind.empty()
    # set_blind(nil) clears G.GAME.last_blind.boss (blind.lua:97-99).
    gs["last_blind_was_boss"] = False
    _set_blind_debuff_pass(gs)

    # ------------------------------------------------------------------
    # 11. Phase → ROUND_EVAL
    # ------------------------------------------------------------------
    gs["phase"] = GamePhase.ROUND_EVAL


def _end_of_round_effect(
    gs: dict[str, Any], card: Any, planet_key: str | None, planets: list[dict[str, Any]]
) -> dict[str, Any]:
    """Card:get_end_of_round_effect (card.lua:1033-1065).

    A debuffed card gives ``{}``.  A Gold Card gives its ``h_dollars``.  A
    Blue Seal with consumable room (counting G.GAME.consumeable_buffer)
    reserves a slot and queues the Planet for G.GAME.last_hand_played
    (append 'blusl'); its event makes the card and zeroes the buffer.
    """
    if getattr(card, "debuff", False):
        return {}
    ret: dict[str, Any] = {}
    h_dollars = card.ability.get("h_dollars", 0)
    if h_dollars > 0:
        ret["h_dollars"] = h_dollars
    if getattr(card, "seal", None) == "Blue" and _consumable_room(gs):
        gs["consumeable_buffer"] = gs.get("consumeable_buffer", 0) + 1
        if planet_key:
            planets.append({"type": "Planet", "forced_key": planet_key, "seed": "blusl"})
        ret["effect"] = True
    return ret


def _held_cards_end_of_round(gs: dict[str, Any], hand: list, jokers: list) -> None:
    """end_round's held-card loop (state_events.lua:171-233).

    Every repetition recomputes get_end_of_round_effect; after the first
    pass, a card that produced something is repeated by its Red Seal
    (``eval_card`` repetition_only, card.lua:2244-2251) and by Mime and
    its Blueprint/Brainstorm copies (card.lua:2877-2886).  Gold Card
    dollars are eased as they come (before cash-out's interest); the Blue
    Seal Planets are made by the queued events afterwards.  No joker has an
    individual end-of-round effect on held cards (card.lua:2875-2876).
    """
    from jackdaw.engine.consumables import _PLANET_HAND
    from jackdaw.engine.jokers import JokerContext, calculate_joker, has_card_effects

    last_hand = gs.get("last_hand_played")
    planet_key = next((pk for pk, ht in _PLANET_HAND.items() if ht == last_hand), None)
    planets: list[dict[str, Any]] = []
    for card in hand:
        reps: list[Any] = [1]
        rep_index = 0
        while rep_index < len(reps):
            effects = [_end_of_round_effect(gs, card, planet_key, planets)]
            if rep_index == 0 and has_card_effects(effects):
                seal = card.calculate_seal(repetition=True)
                if seal and seal.get("repetitions"):
                    reps.extend([seal] * seal["repetitions"])
                ctx = JokerContext(
                    repetition=True,
                    cardarea="hand",
                    other_card=card,
                    card_effects=effects,
                    jokers=jokers,
                    game_state=gs,
                )
                for joker in jokers:
                    if getattr(joker, "debuff", False):
                        continue
                    result = calculate_joker(joker, ctx)
                    if result and result.repetitions > 0:
                        reps.extend([result] * result.repetitions)
            for effect in effects:
                if effect.get("h_dollars"):
                    gs["dollars"] = gs.get("dollars", 0) + effect["h_dollars"]
            rep_index += 1
    if planets:
        _resolve_create_descriptors(gs, planets)
    gs["consumeable_buffer"] = 0


def _advance_ante(gs: dict[str, Any]) -> None:
    """A boss fell: the ante goes up (``ease_ante(1)``, state_events.lua:248)
    and the next ante's voucher is rolled (state_events.lua:263)."""
    rr = gs["round_resets"]
    rr["ante"] += 1
    from jackdaw.engine.tags import roll_next_voucher

    rng = gs.get("rng")
    if rng:
        gs["current_round"]["voucher"] = roll_next_voucher(rr["ante"], rng, gs)


def _reset_blinds_after_boss(gs: dict[str, Any]) -> None:
    """``cash_out`` after a boss (button_callbacks.lua:2949-2953): the new
    ante's blind tags, then ``reset_blinds`` (common_events.lua:2326-2336) —
    every blind Upcoming, the Small blind on deck and a new boss."""
    rr = gs["round_resets"]
    if rr["blind_states"].get("Boss") != "Defeated":
        return
    rr["blind_ante"] = rr["ante"]
    from jackdaw.engine.tags import roll_ante_blinds

    rng = gs.get("rng")
    if rng:
        rr["blind_choices"]["Boss"] = roll_ante_blinds(rr["ante"], rng, gs)
    rr["blind_states"] = {"Small": "Upcoming", "Big": "Upcoming", "Boss": "Upcoming"}
    gs["blind_on_deck"] = "Small"
    rr["boss_rerolled"] = False


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

    # new_round's loop (state_events.lua:335-337) over the board as it is
    # now: every destruction/creation below is a queued event, so the
    # decisions see the same G.jokers plus the getting_sliced marks and
    # G.GAME.joker_buffer / consumeable_buffer left by earlier jokers.
    mutations: list[dict[str, Any]] = []
    for joker in list(jokers):
        if getattr(joker, "debuff", False):
            continue
        ctx = JokerContext(
            setting_blind=True,
            blind=blind,
            jokers=jokers,
            rng=gs.get("rng"),
            game=game_snap,
            game_state=gs,
        )
        result = calculate_joker(joker, ctx)
        if result and result.extra:
            entry = dict(result.extra)
            # Madness must exclude ITSELF from its destroy pool
            # (card.lua: v ~= self) — record who fired the mutation.
            entry["_source_joker"] = joker
            mutations.append(entry)
            # Marble Joker calls playing_card_joker_effects({true}) at calc
            # time; the Stone card is made in an event (card.lua:2580-2600).
            if (entry.get("create") or {}).get("key") == "marble":
                _playing_card_joker_effects(gs, [True])

    return mutations


def _apply_setting_blind_mutations(
    gs: dict[str, Any],
    mutations: list[dict[str, Any]],
    jokers: list,
    *,
    amber_shuffles: bool = False,
) -> tuple[bool, int]:
    """Process the side-effects of the setting_blind (or first_hand_drawn)
    jokers in Lua's event order.

    The jokers decide at calc time and queue events, in board order
    (state_events.lua:335-337); an event's own ``add_event`` calls land
    behind everything already queued.  So the jokers' own events run first
    (Burglar, Riff-raff's jokers, Marble's Stone, Certificate's card), and
    the events those queue run after them all: Amber Acorn's three shuffles
    first (set_blind queued their parent before any joker's, blind.lua:195-
    204), then, in board order, the sliced jokers' removals (start_dissolve
    from Madness's and the Dagger's events, card.lua:2513-2516, 2570-2576)
    and Cartomancer's Tarot (card.lua:2547-2559).  The new cards take their
    sort ids in that order, and the shuffles still see the sliced jokers.

    Returns whether a joker or consumable was added or removed: each queues
    ``G.GAME.blind:set_blind(nil, true)`` (card.lua:641, 698), which runs
    after every card made here (Marble's Stone included); and the hand-size
    change the sliced jokers' removals queue (``G.hand:change_size`` from
    remove_from_deck, card.lua:648-650): it lands after the round's deal is
    counted, so the caller applies it after the deal (G105).
    """
    rng = gs.get("rng")
    # Cards dissolve over several frames (Card:remove runs after the
    # creation events of this batch), so the destroyed jokers' keys stay in
    # used_jokers while Riff-raff's create rolls.
    released: list = []
    touched = False
    hand_size_change = 0

    for mut in mutations:
        # (Chicot's disable_blind is deferred by _handle_select_blind.)

        # Burglar: set hands / remove discards
        if "set_hands" in mut:
            cr = gs.get("current_round", {})
            cr["hands_left"] = cr.get("hands_left", 0) + mut["set_hands"]
        if "set_discards" in mut:
            cr = gs.get("current_round", {})
            cr["discards_left"] = mut["set_discards"]

        # Marble Joker / Certificate / Riff-raff: create cards (Cartomancer's
        # Tarot comes from a nested event, below)
        if "create" in mut:
            create = mut["create"]
            ctype = create.get("type", "")
            if ctype == "playing_card":
                # Marble Joker (card.lua:2580): front via 'marb_fr', Stone
                # center, joins the playing cards. Certificate (card.lua:2462):
                # front via 'cert_fr' emplaced into the HAND, seal via
                # 'certsl' (>0.75 Red, >0.5 Blue, >0.25 Gold, else Purple).
                # The old code appended a Card with NO front (base=None),
                # which corrupted the playing-card pool and crashed the
                # round-end targeting rolls (mail.base.id).
                from jackdaw.engine.card_factory import (
                    RANK_LETTER,
                    SUIT_LETTER,
                    create_playing_card,
                )

                ckey = create.get("key", "")
                stream = {"cert": "cert_fr", "marble": "marb_fr"}.get(ckey)
                if rng is not None and stream is not None:
                    p_cards = {
                        f"{sl}_{rl}": (suit, rank)
                        for sl, suit in SUIT_LETTER.items()
                        for rl, rank in RANK_LETTER.items()
                    }
                    (c_suit, c_rank), _ = rng.element(p_cards, rng.seed(stream))
                    seal = None
                    if create.get("seal"):
                        roll = rng.random(rng.seed("certsl"))
                        if roll > 0.75:
                            seal = "Red"
                        elif roll > 0.5:
                            seal = "Blue"
                        elif roll > 0.25:
                            seal = "Gold"
                        else:
                            seal = "Purple"
                    c = create_playing_card(
                        c_suit,
                        c_rank,
                        create.get("enhancement", "c_base"),
                        seal=seal,
                        hands_played=gs.get("hands_played", 0),
                        inflation=gs.get("inflation", 0),
                        discount_percent=gs.get("discount_percent", 0),
                    )
                    # Card:init ends with set_cost (card.lua:72), and
                    # Certificate's set_seal repeats it (card.lua:498).
                    _reprice(gs, c)
                    if ckey == "cert":
                        # create_playing_card(..., G.hand) — Certificate
                        # deals into the hand (first_hand_drawn is in-round)
                        gs.setdefault("hand", []).append(c)
                        # Certificate's event ends with G.hand:sort() (card.lua:2474)
                        _sort_hand(gs, gs.get("hand", []))
                    elif ckey == "marble":
                        # Marble's events (create in G.play, then draw_card
                        # to G.deck) are queued during the setting_blind
                        # loop, ahead of new_round's 'nr' shuffle event
                        # (card.lua:2580-2601, state_events.lua:335-349): the
                        # Stone joins the deck now and is shuffled in (G55).
                        # Steamodded queues the draw inside the creation
                        # event, after the shuffle (smods-diffs.md S01).
                        _emplace_in_deck(gs, c)
                    else:
                        _emplace_in_deck(gs, c)
            elif ctype == "Joker":
                # Riff-raff ('rif', Common): its event makes the count it
                # reserved at calc time with no further room check
                # (card.lua:2532-2542) — a sliced joker still holds its slot.
                from jackdaw.engine.card_factory import resolve_create_descriptor

                ante = gs.get("round_resets", {}).get("ante", 1)
                for _ in range(create.get("count", 1)):
                    card = resolve_create_descriptor({**create, "count": 1}, rng, ante, gs)
                    if card is not None:
                        jokers.append(card)
                        card.add_to_deck(gs)
                        touched = True

    # Amber Acorn's three G.jokers:shuffle('aajk') events (blind.lua:195-204)
    # run on the board as the jokers' own events left it, sliced jokers
    # included; a disable does not cancel them.
    if amber_shuffles and rng is not None:
        for _ in range(3):
            rng.shuffle(jokers, rng.seed("aajk"))

    for mut in mutations:
        # Madness / Ceremonial Dagger: the victim was picked and marked
        # getting_sliced when the joker was evaluated (card.lua:2503-2521,
        # 2561-2579); start_dissolve, called from the joker's event, queues
        # its removal.  (Live-verified: LS5EUNSF's Madness pool, LSL9ZZUW's
        # Dagger victim.)
        victim = mut.get("destroy_joker")
        if victim is not None and victim in jokers:
            jokers.remove(victim)
            size = gs.get("hand_size", 0)
            victim.remove_from_deck(gs)
            hand_size_change += gs.get("hand_size", 0) - size
            gs["hand_size"] = size
            released.append(victim)
            touched = True

        # Cartomancer ('car'): its event queues the event that makes the Tarot.
        create = mut.get("create") or {}
        if create.get("type") in ("Tarot", "Planet", "Spectral"):
            if _resolve_create_descriptors(gs, [create]):
                touched = True

    # The creation events end with G.GAME.joker_buffer = 0 (card.lua:2539,
    # 2571) and consumeable_buffer = 0; the dissolved jokers leave play.
    gs["joker_buffer"] = 0
    gs["consumeable_buffer"] = 0
    for victim in released:
        _release_used_key(gs, victim)
    return touched, hand_size_change


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
        change_hand_size(gs, -1)

    # Amber Acorn's flip and shuffles are applied around the setting_blind
    # context in _handle_select_blind (blind.lua:190-205).

    # The Eye: reset hand tracking
    elif name == "The Eye":
        blind.hands = {
            ht: False
            for ht in [
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
        }

    # The Mouth: reset only_hand
    elif name == "The Mouth":
        blind.only_hand = None

    # The House / The Mark: flip cards face-down (blind.lua:200-203)
    # Cards are flipped at draw_to_hand time, not set_blind time.
    # We handle this in _draw_hand context by checking blind.name.


# ---------------------------------------------------------------------------
# Shop joker context helpers
# ---------------------------------------------------------------------------


#: ``gs`` key set while a consumable is being used: vanilla's ``G.STATE`` is
#: ``PLAY_TAROT`` then (button_callbacks.lua:2181-2186), outside a pack.
_PLAY_TAROT = "_play_tarot"


def _use_consumable_card(
    gs: dict[str, Any],
    card: Any,
    targets: tuple[int, ...] | None = None,
) -> None:
    """Build a ConsumableContext and use a consumable card.

    Bridges the game_state dict with the ``use_consumable(card, ctx)`` API.
    Shared by tray uses and pack picks, which both go through
    ``G.FUNCS.use_card`` in vanilla.

    1. Build ConsumableContext from game_state + target_indices
    2. Call handler → ConsumableResult
    3. Apply all result mutations (incl. last_tarot_planet)
    4. Fire the ``using_consumeable`` joker context

    ``G.STATE`` is ``PLAY_TAROT`` throughout (``gs[_PLAY_TAROT]``): a hand
    size grown meanwhile draws nothing (:func:`change_hand_size`).
    """
    gs[_PLAY_TAROT] = True
    try:
        _use_consumable_card_now(gs, card, targets)
    finally:
        gs.pop(_PLAY_TAROT, None)


def _use_consumable_card_now(
    gs: dict[str, Any],
    card: Any,
    targets: tuple[int, ...] | None,
) -> None:
    from jackdaw.engine.consumables import ConsumableContext, highlighted_cards, use_consumable

    hand: list = gs.get("hand", [])
    highlighted = highlighted_cards(hand, targets)

    ctx = ConsumableContext(
        card=card,
        highlighted=highlighted or None,
        hand_cards=hand or None,
        jokers=gs.get("jokers") or None,
        consumables=gs.get("consumables") or None,
        playing_cards=gs.get("deck") or None,
        rng=gs.get("rng"),
        game_state=gs,
    )
    result = use_consumable(card, ctx)
    if result is not None:
        _apply_consumable_result(gs, result, card)

    # Every use notifies the jokers right after use_consumeable
    # (button_callbacks.lua:2219-2221), from the tray or a pack alike:
    # Constellation counts Planets, Glass Joker the Glass Cards The Hanged
    # Man was used on.
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    game_snap = GameSnapshot(
        joker_count=len(jokers),
        money=gs.get("dollars", 0),
    )
    for joker in list(jokers):
        if getattr(joker, "debuff", False):
            continue
        jctx = JokerContext(
            using_consumeable=True,
            consumeable=card,
            highlighted=highlighted,
            jokers=jokers,
            game=game_snap,
        )
        calculate_joker(joker, jctx)


def _apply_consumable_result(
    gs: dict[str, Any],
    result: Any,
    card: Any = None,
) -> None:
    """Apply a ConsumableResult's mutations to game_state.

    Handles all 14+ mutation types from ConsumableResult.
    """
    # Track last_tarot_planet for The Fool
    card_key = getattr(card, "center_key", None)
    if card_key:
        card_set = _get_card_set(card) if card else ""
        if card_set in ("Tarot", "Planet"):
            gs["last_tarot_planet"] = card_key

    # ---- Card modifications ----
    # A non-initial set_ability / set_base ends with
    # G.GAME.blind:debuff_card(self) (card.lua:143, 365), and change_suit
    # runs it again (card.lua:561): the boss debuff is re-checked at once.

    # a. Enhancement: set_ability(G.P_CENTERS[mod_conv]) stamps
    #    hands_played_at_create = G.GAME.hands_played like every set_ability
    #    (card.lua:337, 1141-1144; G130).
    if getattr(result, "enhance", None):
        for target, enh_key in result.enhance:
            if hasattr(target, "set_ability"):
                target.set_ability(enh_key, hands_played=gs.get("hands_played", 0))
                blind_debuff_card(gs, target)

    # b. Suit changes
    if getattr(result, "change_suit", None):
        for target, suit in result.change_suit:
            if hasattr(target, "change_suit"):
                target.change_suit(suit)
                blind_debuff_card(gs, target)

    # c. Rank changes
    if getattr(result, "change_rank", None):
        for target, delta in result.change_rank:
            if hasattr(target, "change_rank"):
                target.change_rank(delta)
                blind_debuff_card(gs, target)

    # d. Copy card (Death): copy_card(rightmost, target) (card.lua:1117)
    #    overwrites the target in place — center, base, every ability entry
    #    (perma_bonus, played_this_ante, ...), edition, seal and debuff.
    if getattr(result, "copy_card", None):
        from jackdaw.engine.card_factory import copy_card

        source, target = result.copy_card
        # copy_card ends with new_card.debuff = other.debuff
        # (common_events.lua:2177), after the set_ability / set_base re-checks.
        copy_card(source, target, game_state=gs)

    # e. Destroy playing cards
    if getattr(result, "destroy", None):
        deck: list = gs.get("deck", [])
        hand: list = gs.get("hand", [])
        _removed: list = []
        for destroyed in result.destroy:
            if destroyed in hand:
                hand.remove(destroyed)
                _removed.append(destroyed)
            elif destroyed in deck:
                deck.remove(destroyed)
                _removed.append(destroyed)
        # remove_playing_cards joker notify (card.lua:1370)
        _notify_cards_destroyed(gs, _removed)

    # f. Add seal
    if getattr(result, "add_seal", None):
        for target, seal_type in result.add_seal:
            target.seal = seal_type

    # m. Destroy jokers (Hex, Ankh).  Ankh's dissolving event blocks for
    #    0.75s, longer than the 0.735s dissolve, so the jokers are removed
    #    (Card:remove: their passives end) before the copy is made
    #    (card.lua:1435-1453; G136): ease_discard stops at 0, so the order
    #    shows in the discards left.
    if getattr(result, "destroy_jokers", None):
        jokers: list = gs.get("jokers", [])
        for j in result.destroy_jokers:
            if j in jokers:
                jokers.remove(j)
                j.remove_from_deck(gs)
                _release_used_key(gs, j)

    # g. Create cards (High Priestess, Emperor, Judgement, etc.)
    if getattr(result, "create", None):
        created = _resolve_create_descriptors(gs, result.create)
        # Familiar / Grim / Incantation: playing_card_joker_effects with the
        # new cards, in the creation event (card.lua:1311-1338).
        new_playing_cards = [c for c in created if c.ability.get("set") in ("Default", "Enhanced")]
        if new_playing_cards:
            _playing_card_joker_effects(gs, new_playing_cards)

    # ---- Economy ----

    # h. Dollars
    if getattr(result, "dollars", 0):
        gs["dollars"] = gs.get("dollars", 0) + result.dollars

    # i. Money set (Wraith → set to 0)
    if getattr(result, "money_set", None) is not None:
        gs["dollars"] = result.money_set

    # ---- Hand levels ----

    # j. Level up (Planet cards)
    if getattr(result, "level_up", None):
        hand_levels = gs.get("hand_levels")
        if hand_levels:
            for ht, amount in result.level_up:
                hand_levels.level_up(ht, amount)

    # ---- Deck mutation ----

    # k. Add playing cards to deck
    if getattr(result, "add_to_deck", None):
        from jackdaw.engine.card import Card as _Card
        from jackdaw.engine.card_factory import copy_card

        added: list = []
        for card_spec in result.add_to_deck:
            # Cryptid: copy an existing card — copies are emplaced into the
            # HAND, not the draw pile (card.lua:1206-1213: copy_card +
            # G.hand:emplace), each a new Card with a fresh sort_id.
            copy_source = card_spec.get("copy_of")
            if copy_source is not None:
                new_card = copy_card(copy_source, game_state=gs)
                # _card:add_to_deck() (card.lua:1209) marks the copy added (G205).
                new_card.add_to_deck(gs)
                added.append(new_card)
                # Emplaced at the right end; nothing sorts the hand until the
                # next draw (cardarea.lua:32-37).  In an Arcana or Spectral
                # pack that is the dealt hand, which returns to the deck when
                # the pack closes (G132).
                gs.setdefault("hand", []).append(new_card)
                continue
            new_card = _Card()
            if "suit" in card_spec and "rank" in card_spec:
                new_card.set_base(
                    card_spec.get("key", ""),
                    card_spec["suit"],
                    card_spec["rank"],
                )
            if "enhancement" in card_spec:
                new_card.set_ability(card_spec["enhancement"])
            _emplace_in_deck(gs, new_card)
            added.append(new_card)
        # Cryptid: playing_card_joker_effects(new_cards) (card.lua:1217)
        if added:
            _playing_card_joker_effects(gs, added)

    # ---- Joker effects ----

    # l. Add edition (Wheel of Fortune, Aura, Ectoplasm, Hex): all call
    #    card:set_edition(edition, true) (card.lua:1197, 1486), which builds
    #    the edition table with its scoring value (card.lua:390-418) and
    #    ends in set_cost (card.lua:461) — the edition surcharge reprices it.
    if getattr(result, "add_edition", None):
        ae = result.add_edition
        target = ae.get("target")
        edition = ae.get("edition")
        if target and edition:
            was_negative = bool(target.edition and target.edition.get("negative"))
            target.set_edition(edition)
            # Going Negative while already added to the deck raises the
            # owning area's limit (card.lua:405-415).  A debuffed joker is out
            # of the deck (set_debuff -> remove_from_deck(true), card.lua:526-538),
            # so Ectoplasm on it adds no slot (G139).
            if edition.get("negative") and not was_negative and target.added_to_deck:
                if target in gs.get("jokers", []):
                    gs["joker_slots"] = gs.get("joker_slots", 5) + 1
                elif target in gs.get("consumables", []):
                    gs["consumable_slots"] = gs.get("consumable_slots", 2) + 1
            _reprice(gs, target)

    # ---- Game state ----

    # n. Hand size modification (Ectoplasm -1, Ouija -1)
    if getattr(result, "hand_size_mod", 0):
        change_hand_size(gs, result.hand_size_mod)


def _resolve_create_descriptors(gs: dict[str, Any], descriptors: list[dict[str, Any]]) -> list[Any]:
    """Resolve card creation descriptors from ConsumableResult.create.

    Each descriptor is ``{'type': ..., 'count': ..., 'seed': ...,
    'forced_key': ...}``.  Creates the actual Card objects and adds
    them to the appropriate area; returns the cards placed.

    Delegates to :func:`~jackdaw.engine.card_factory.resolve_create_descriptor`
    for actual card creation.
    """
    from jackdaw.engine.card_factory import resolve_create_descriptor

    rng = gs.get("rng")
    ante = gs.get("round_resets", {}).get("ante", 1)
    # Planet pool softlock filtering needs current played-hand counts
    # (High Priestess / Blue Seal create Planets mid-round).
    _sync_played_hand_types(gs)
    consumables: list = gs.setdefault("consumables", [])
    consumable_limit = gs.get("consumable_slots", 2)
    jokers: list = gs.setdefault("jokers", [])
    joker_slots = gs.get("joker_slots", 5)
    placed: list = []

    for desc in descriptors:
        count = desc.get("count", 1)

        for _ in range(count):
            # Room check BEFORE resolving: vanilla gates create_card on
            # area room in the caller's condition chain, so at full
            # slots NO pool/edition streams are consumed.  Resolving
            # then dropping desynced the 'Tarotvag' stream
            # (live-verified: LSEEC4W8 — sim burned a roll at full
            # slots and created c_lovers where live created the
            # c_wheel_of_fortune the sim had wasted).
            desc_type = desc.get("type", "")
            # The Fool's 'Tarot_Planet' too: its event checks the room before
            # create_card (card.lua:1375-1377; G142).
            if desc_type in ("Tarot", "Planet", "Spectral", "Tarot_Planet"):
                if len(consumables) >= gs.get("consumable_slots", 2):
                    continue
            elif desc_type == "Joker":
                if len(jokers) >= gs.get("joker_slots", 5):
                    continue

            card = resolve_create_descriptor(desc, rng, ante, gs)
            if card is None:
                continue

            card_set = card.ability.get("set", "")
            if card_set == "Joker":
                negative = card.edition and card.edition.get("negative")
                # Re-read live: a Negative added this loop raises the cap.
                joker_slots = gs.get("joker_slots", 5)
                if len(jokers) < joker_slots + (1 if negative else 0):
                    jokers.append(card)
                    card.add_to_deck(gs)
                    placed.append(card)
            elif card_set in ("Tarot", "Planet", "Spectral"):
                negative = card.edition and card.edition.get("negative")
                consumable_limit = gs.get("consumable_slots", 2)
                if len(consumables) < consumable_limit + (1 if negative else 0):
                    consumables.append(card)
                    card.add_to_deck(gs)
                    placed.append(card)
            elif card_set in ("Default", "Enhanced", ""):
                # Card:init priced it with the run's inflation (card.lua:72).
                _reprice(gs, card)
                # Familiar, Grim and Incantation: create_playing_card(..., G.hand)
                # (card.lua:1336).  G.hand:emplace appends (cardarea.lua:32-37)
                # and nothing sorts the hand until the next draw.  In an
                # Arcana or Spectral pack that is the dealt hand, which returns
                # to the deck when the pack closes (G132).
                if gs.get("phase") in (GamePhase.SELECTING_HAND, GamePhase.PACK_OPENING):
                    gs.setdefault("hand", []).append(card)
                else:
                    _emplace_in_deck(gs, card)
                placed.append(card)

    # Every vanilla creation event that consumed a reservation ends with
    # G.GAME.consumeable_buffer = 0 (card.lua:1057, 2263, 2346, 2554, ...).
    gs["consumeable_buffer"] = 0
    return placed


# ---------------------------------------------------------------------------
# Shop population helpers
# ---------------------------------------------------------------------------


def _sync_played_hand_types(gs: dict[str, Any]) -> None:
    """Populate ``gs["played_hand_types"]`` from per-run play counts.

    The Planet pool softlock gate is ``G.GAME.hands[hand_type].played > 0``
    (common_events.lua:2009) — a hand type counts only once it has been
    PLAYED this run.  Leveling a secret hand (e.g. via a Ceres from a pack)
    makes it *visible* but does not unlock its planet in pools.
    """
    from jackdaw.engine.hand_levels import HandLevels

    hand_levels: HandLevels | None = gs.get("hand_levels")
    if hand_levels is None:
        return
    played: set[str] = set()
    for ht, state in hand_levels._hands.items():
        if state.played > 0:
            played.add(ht.value)
    gs["played_hand_types"] = played


def _populate_shop(gs: dict[str, Any]) -> None:
    """Generate shop cards using populate_shop and store in game_state.

    Places results in ``gs["shop_cards"]``, ``gs["shop_vouchers"]``,
    ``gs["shop_boosters"]``.
    """
    from jackdaw.engine.shop import populate_shop

    rng = gs.get("rng")
    if rng is None:
        return

    # Sync played hand types for Planet pool softlock filtering
    # (G.GAME.hands[ht].played > 0, common_events.lua:2009).
    _sync_played_hand_types(gs)

    ante = gs.get("round_resets", {}).get("ante", 1)
    result = populate_shop(rng, ante, gs)

    gs["shop_cards"] = result.get("jokers", [])
    voucher = result.get("voucher")
    gs["shop_vouchers"] = [voucher] if voucher else []
    gs["shop_boosters"] = result.get("boosters", [])

    _fire_shop_tags(gs, rerolled=False)


def _fire_shop_tags(gs: dict[str, Any], rerolled: bool) -> None:
    """Fire pending shop-context tag hooks on freshly stocked shop cards.

    Wires the hooks :func:`~jackdaw.engine.shop.populate_shop` defers (its
    "M11 tag system" note), consuming awarded tags FIFO:

    * ``store_joker_modify`` — Foil/Holo/Polychrome/Negative Tag: the next
      base-edition shop Joker gains the edition and becomes free
      (tags.lua ``Tag:apply_to_run``). Applies to rerolled jokers too.
    * ``shop_final_pass`` — Coupon Tag: initial shop cards and boosters
      become free (shop entry only).
    * ``shop_start`` — D6 Tag: rerolls start free (shop entry only).

    * ``voucher_add`` — Voucher Tag: an extra purchasable voucher rolled
      on the dedicated ``'Voucher_fromtag'`` stream (shop entry only;
      live-validated via lockstep seed LS1UBIP7).

    ``store_joker_create`` (Rare/Uncommon Tag) is wired separately via
    :func:`~jackdaw.engine.shop.apply_store_joker_create_tag`.
    """
    from jackdaw.engine.shop import shop_set_cost
    from jackdaw.engine.tags import Tag

    awarded: list[dict[str, Any]] = gs.get("awarded_tags", [])
    rng = gs.get("rng")

    for entry in awarded:
        if entry.get("shop_fired"):
            continue
        tag = Tag(entry.get("key", ""))

        result = tag.apply("store_joker_modify", gs, rng=rng)
        if result is not None and result.force_edition:
            target = None
            for c in gs.get("shop_cards", []):
                ability = getattr(c, "ability", None) or {}
                if ability.get("set") == "Joker" and not getattr(c, "edition", None):
                    target = c
                    break
            if target is None:
                continue  # no eligible joker; tag stays pending for later
            edition = {str(result.force_edition): True}
            if hasattr(target, "set_edition"):
                target.set_edition(edition)
            else:
                target.edition = edition
            # The tag's event (tag.lua:398-441): set_edition prices the
            # edition in (card.lua:461), then the joker is couponed and
            # set_cost zeroes its price (card.lua:383); the sell value keeps
            # the full price (G204).
            target.ability["couponed"] = True
            shop_set_cost(gs, target)
            entry["shop_fired"] = True
            continue

        if rerolled:
            continue  # remaining hooks fire on shop entry only

        # voucher_add — Voucher Tag adds an extra purchasable voucher
        # (tag.lua:302-318).  Vanilla fires this after boosters on shop
        # entry (game.lua:3161-3163); the roll uses the dedicated
        # 'Voucher_fromtag' stream (common_events.lua:1903, no ante
        # suffix), pool excludes used + currently-in-shop vouchers.
        result = tag.apply("voucher_add", gs, rng=rng)
        if result is not None and result.create_voucher and rng is not None:
            from jackdaw.engine.card_factory import create_voucher
            from jackdaw.engine.vouchers import get_next_voucher_key

            in_shop = [
                getattr(v, "center_key", None) or getattr(v, "key", "")
                for v in gs.get("shop_vouchers", [])
            ]
            used_v = {k: True for k in gs.get("used_vouchers", [])}
            ante = gs.get("round_resets", {}).get("ante", 1)
            v_key = get_next_voucher_key(
                rng,
                used_v,
                in_shop,
                from_tag=True,
                ante=ante,
                banned_keys=gs.get("banned_keys"),
            )
            if v_key:
                voucher = create_voucher(v_key, hands_played=gs.get("hands_played", 0))
                gs.setdefault("used_jokers", {})[v_key] = True
                voucher.set_cost(
                    inflation=gs.get("inflation", 0),
                    discount_percent=gs.get("discount_percent", 0),
                    ante=ante,
                )
                gs.setdefault("shop_vouchers", []).append(voucher)
            entry["shop_fired"] = True
            continue

        result = tag.apply("shop_final_pass", gs, rng=rng)
        if result is not None and result.coupon:
            # One Coupon Tag per shop: it applies only while
            # G.GAME.shop_free is unset and sets it (tag.lua:448-449);
            # cash_out clears it.  Otherwise it waits for the next shop.
            if gs.get("shop_free"):
                continue
            gs["shop_free"] = True
            for c in gs.get("shop_cards", []) + gs.get("shop_boosters", []):
                # couponed, then set_cost (tag.lua:450-459): the flag keeps the
                # card free through later passes in the shop.
                c.ability["couponed"] = True
                shop_set_cost(gs, c)
            entry["shop_fired"] = True
            continue

        result = tag.apply("shop_start", gs, rng=rng)
        if result is not None and result.temp_reroll_zero:
            # D6 Tag (tag.lua:383-391): rerolls START at $0 this shop
            # (then climb +1 per reroll); once per shop via shop_d6ed,
            # cleared at the next cash_out.  Live-verified: LSKWQS7C
            # entered the shop with reroll_cost 0.
            if not gs.get("shop_d6ed"):
                gs["shop_d6ed"] = True
                gs.setdefault("round_resets", {})["temp_reroll_cost"] = 0
                from jackdaw.engine.shop import calculate_reroll_cost

                calculate_reroll_cost(gs)
                entry["shop_fired"] = True
            continue
        if result is not None and result.free_rerolls:
            cr = gs.setdefault("current_round", {})
            cr["free_rerolls"] = cr.get("free_rerolls", 0) + result.free_rerolls
            entry["shop_fired"] = True


def _reroll_shop_cards(gs: dict[str, Any]) -> None:
    """Regenerate the shop joker/consumable cards (not voucher or boosters).

    Matches the repopulate step of ``reroll_shop``
    (``button_callbacks.lua:2855``).
    """
    from jackdaw.engine.shop import create_card_for_shop

    rng = gs.get("rng")
    if rng is None:
        return

    _sync_played_hand_types(gs)

    for old in gs.get("shop_cards", []):
        _release_used_key(gs, old)

    ante = gs.get("round_resets", {}).get("ante", 1)
    shop_joker_max: int = gs.get("shop", {}).get("joker_max", 2)
    # Shop cards are never soulable (UI_definitions.lua:776 passes nil);
    # this loop once missed that and burned a phantom soul roll per slot
    # (LSMORS4J: a c_black_hole where live showed c_venus).
    gs["shop_cards"] = [create_card_for_shop(gs, rng, ante) for _ in range(shop_joker_max)]
    # Pending edition tags also claim rerolled jokers (vanilla behavior).
    _fire_shop_tags(gs, rerolled=True)


def run_pending_reprice(gs: dict[str, Any]) -> None:
    """The set_cost pass an Astronomer joining or leaving queued.

    Card:add_to_deck and Card:remove_from_deck queue an event that re-runs
    set_cost on every card (card.lua:616-622, 673-679), so a Hex, a dagger,
    a debuff or a created Astronomer reprices the Planets and Celestial packs
    as a purchase or a sale does (G261). The engine has no event queue: the
    pass runs once the action that queued it is done.
    """
    if gs.pop("astronomer_reprice", False):
        _all_cards_set_cost_pass(gs)


def _all_cards_set_cost_pass(gs: dict[str, Any]) -> None:
    """Vanilla's G.I.CARD set_cost pass: clears owned coupon flags so the
    subsequent reprice restores real costs.  Triggers: Clearance Sale /
    Liquidation redeem (card.lua:1917-23), Gift Card round-end
    (handled inline), and Astronomer joining/leaving the board
    (dump card.lua:786-793 / :850; live-verified: LSVFB5K8's couponed
    foil Astronomer flipped $0 -> $10 the moment it was bought)."""
    from jackdaw.engine.shop import reprice_shop

    reprice_shop(gs)
    # Owned cards too: the coupon zeroes a price only in the shop areas
    # (card.lua:383), so this restores a couponed card's real price and
    # leaves its ``couponed`` flag, as Lua does.
    for owned in gs.get("jokers", []) + gs.get("consumables", []):
        _reprice(gs, owned)
    # An open pack's cards are in G.I.CARD too (G261).
    for card in gs.get("pack_cards", []):
        _reprice(gs, card)
    # G.I.CARD holds the playing cards too.
    for area in ("deck", "hand", "discard_pile", "played_cards_area"):
        for card in gs.get(area, []):
            _reprice(gs, card)


def _inflate(gs: dict[str, Any]) -> None:
    """Inflation challenge: every purchase raises all prices by $1.

    buy_from_shop (button_callbacks.lua:2455-2462), Card:open (card.lua:1800,
    tag packs included) and Card:redeem (card.lua:1858): under
    ``modifiers.inflation``, ``G.GAME.inflation`` goes up by 1 and every
    card re-runs set_cost (card.lua:370 adds the inflation to extra_cost).
    """
    if (gs.get("modifiers") or {}).get("inflation"):
        gs["inflation"] = gs.get("inflation", 0) + 1
        _all_cards_set_cost_pass(gs)


def _reprice(gs: dict[str, Any], card: Any) -> None:
    """``Card:set_cost`` for a card outside the shop areas (card.lua:369-385).

    Owned and newly made cards are never couponed-to-zero (the coupon only
    applies in ``G.shop_jokers``/``G.shop_booster``, card.lua:383).
    """
    from jackdaw.engine.card_utils import astronomer_active

    card.set_cost(
        inflation=gs.get("inflation", 0),
        discount_percent=gs.get("discount_percent", 0),
        ante=gs.get("round_resets", {}).get("ante", 1),
        booster_ante_scaling=gs.get("booster_ante_scaling", False),
        has_astronomer=astronomer_active(gs),
    )


def _get_card_set(card: Any) -> str:
    """Get the set name from a Card's ability dict."""
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return ability.get("set", "")
    return ""


def _fire_shop_joker_context(
    gs: dict[str, Any], *, exclude: Any = None, **context_flags: Any
) -> list[dict[str, Any]]:
    """Fire a joker context during shop phase and return mutations.

    Accepts keyword arguments matching :class:`JokerContext` flags
    (e.g. ``buying_card=True``, ``reroll_shop=True``).  ``exclude`` is a
    held card that does not get the context (the card being sold,
    button_callbacks.lua:2321-2325).
    """
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    if not jokers:
        return []

    game_snap = GameSnapshot(
        joker_count=len(jokers),
        joker_slots=gs.get("joker_slots", 5),
        money=gs.get("dollars", 0),
        probabilities_normal=gs.get("probabilities", {}).get("normal", 1.0),
        ante=gs.get("round_resets", {}).get("ante", 1),
        # Hallucination's room gate (card.lua:2336) — checked BEFORE its
        # halu roll, so full slots must suppress the pull entirely.
        consumable_count=len(gs.get("consumables", [])),
        consumable_slots=gs.get("consumable_slots", 2),
    )

    # Extract 'cards' from flags if present (for playing_card_added)
    cards_arg = context_flags.pop("cards", None)

    mutations: list[dict[str, Any]] = []
    for joker in list(jokers):  # copy to allow mutation during iteration
        if getattr(joker, "debuff", False) or joker is exclude:
            continue
        ctx = JokerContext(
            jokers=jokers,
            game=game_snap,
            rng=gs.get("rng"),
            game_state=gs,
            **context_flags,
        )
        if cards_arg is not None:
            ctx.cards = cards_arg
        result = calculate_joker(joker, ctx)
        if result and result.extra:
            mutations.append(result.extra)

    return mutations


def _apply_shop_mutations(
    gs: dict[str, Any],
    mutations: list[dict[str, Any]],
) -> None:
    """Process side-effect dicts from shop joker contexts.

    Handles Perkeo's consumable_copy creation.
    """
    for mut in mutations:
        if "create" in mut:
            create = mut["create"]
            ctype = create.get("type", "")

            if ctype == "consumable_copy":
                # Perkeo: copy a random consumable with Negative edition.
                # Vanilla copy_card + set_edition({negative}) + set_cost
                # + add_to_deck: the negative edition adds +5 to cost
                # (card.lua:372-73) and the negative CONSUMABLE bumps
                # consumable_slots by 1 (card.lua:568 routing).  The old
                # shallow copy shared the ability dict and skipped both
                # (live-verified: ESWXXNUU — live's copy buy=8/sell=4
                # with limit 3; sim kept 3/1 and limit 2).
                consumables: list = gs.get("consumables", [])
                if consumables:
                    rng = gs.get("rng")
                    if rng:
                        from jackdaw.engine.card_factory import copy_card

                        seed_val = rng.seed("perkeo")
                        original, _ = rng.element(consumables, seed_val)
                        # copy_card (card.lua:2417) makes a new Card; the
                        # Negative set_edition then reprices it (card.lua:461).
                        duplicate = copy_card(original, game_state=gs)
                        duplicate.set_edition({"negative": True})
                        _reprice(gs, duplicate)
                        consumables.append(duplicate)
                        duplicate.add_to_deck(gs)


# ---------------------------------------------------------------------------
# Pack close helper
# ---------------------------------------------------------------------------


def _close_pack(gs: dict[str, Any]) -> None:
    """Close the current booster pack and return to the previous phase.

    Matches ``end_consumeable`` in ``button_callbacks.lua:2565``:
    - Remove remaining pack cards
    - Return dealt hand cards to deck (Arcana/Spectral packs deal a hand)
    - Fire ``new_blind_choice`` tags (deferred from skip)
    - Restore phase from ``shop_return_phase``
    """
    # Clear pack state (unpicked cards dissolve -> keys released)
    for c in gs.get("pack_cards", []):
        _release_used_key(gs, c)
    gs["pack_cards"] = []
    gs["pack_choices_remaining"] = 0

    # Return dealt hand cards to deck (Arcana/Spectral packs deal from deck)
    # Lua's draw_from_hand_to_deck (state_events.lua:1121-1126) removes
    # first card from hand and inserts at position 1 (front) of deck,
    # repeated for all cards.  Net effect: hand cards end up REVERSED
    # at the FRONT of the deck.
    pack_hand: list = gs.get("pack_hand", [])
    hand: list = gs.get("hand", [])
    if pack_hand or hand:
        deck: list = gs.setdefault("deck", [])
        # draw_from_hand_to_deck (state_events.lua:1121-1126) draws every
        # held card, in the HAND's current (sorted display) order — it pops
        # hand[1] repeatedly, NOT deal order (deal-order returns diverged the
        # deck whenever no seeded shuffle intervened before the next deal;
        # found by lockstep at the first back-to-back pack open).  The hand
        # is empty when a pack opens, so it holds the dealt cards a pick left
        # plus the cards Familiar, Grim, Incantation and Cryptid made there
        # (card.lua:1213, 1336; G132).
        deck[:0] = list(reversed(hand))
        gs["hand"] = []
        gs["pack_hand"] = []

    # Queued tag packs open back-to-back before the phase is restored.  Only
    # a state saved before G106 holds any: the held pack tags now wait for
    # the new_blind_choice pass below.
    pending: list = gs.get("pending_tag_packs") or []
    if pending:
        next_key = pending.pop(0)
        saved_return = gs.get("shop_return_phase", GamePhase.BLIND_SELECT)
        _open_tag_pack(gs, next_key, force=True)
        # keep the ORIGINAL return phase (not PACK_OPENING itself)
        gs["shop_return_phase"] = saved_return
        gs["phase"] = GamePhase.PACK_OPENING
        return

    # Restore phase
    gs["phase"] = gs.get("shop_return_phase", GamePhase.SHOP)
    # end_consumeable's new_blind_choice pass (button_callbacks.lua:2617-2619):
    # the next held pack tag opens its pack, after the state is restored (G106).
    _fire_new_blind_choice_tags(gs)


# ---------------------------------------------------------------------------
# Blind:press_play — blind.lua:464
# ---------------------------------------------------------------------------


def _consumable_room(gs: dict[str, Any]) -> bool:
    """``#G.consumeables.cards + G.GAME.consumeable_buffer <
    G.consumeables.config.card_limit``."""
    return len(gs.get("consumables", [])) + gs.get("consumeable_buffer", 0) < gs.get(
        "consumable_slots", 2
    )


def _purple_seal(gs: dict[str, Any], card: Any, queue: list[dict[str, Any]]) -> None:
    """Card:calculate_seal({discard = true}) (card.lua:2242-2268): a
    non-debuffed Purple Seal with consumable room (counting
    G.GAME.consumeable_buffer) reserves a slot and queues an '8ba' Tarot,
    which the caller makes after the discard loop like vanilla's event."""
    if getattr(card, "seal", None) == "Purple" and not card.debuff and _consumable_room(gs):
        gs["consumeable_buffer"] = gs.get("consumeable_buffer", 0) + 1
        queue.append({"type": "Tarot", "count": 1, "seed": "8ba"})


def _playing_card_joker_effects(gs: dict[str, Any], cards: list[Any]) -> None:
    """``playing_card_joker_effects`` (misc_functions.lua:1580-1584) on the
    live joker list — see :func:`jackdaw.engine.jokers.playing_card_joker_effects`."""
    from jackdaw.engine.jokers import playing_card_joker_effects

    playing_card_joker_effects(gs.get("jokers", []), cards)


def _notify_cards_destroyed(gs: dict[str, Any], destroyed: list) -> None:
    """Vanilla fires ``remove_playing_cards`` to every joker wherever
    playing cards are destroyed: consumable destroys (card.lua:1370),
    discard-flow destroys (state_events.lua:426), and scoring destroys
    (state_events.lua:975 — that one lives in score_hand's Phase 11
    notify).  Caino / Glass Joker growth listens (bug #69, ES9IGE4S:
    a tarot ate two face cards — live Caino x3, sim stuck at x1)."""
    if not destroyed:
        return
    from jackdaw.engine.jokers import JokerContext, calculate_joker, effect_flags

    _playing_cards_removed(gs, destroyed)
    jokers = gs.get("jokers", [])
    flags = effect_flags(jokers)  # Caino counts is_face() cards (card.lua:2676)
    for joker in jokers:
        if getattr(joker, "debuff", False):
            continue
        calculate_joker(joker, JokerContext(cards_destroyed=destroyed, jokers=jokers, **flags))


def _playing_cards_removed(gs: dict[str, Any], cards: list) -> None:
    """``Card:remove()`` for destroyed playing cards (card.lua:4727-4733).

    Its ``remove_from_deck()`` does something only for a card that went
    through ``add_to_deck`` (a shop purchase, a Standard-pack pick, a Cryptid
    or DNA copy; card.lua:568-575): that queues ``set_blind(nil, true)``
    (card.lua:698), which re-runs the boss's debuffs over every playing card.
    The event runs after the action's other events, so the step runs it at
    the end (G205).
    """
    for card in cards:
        if getattr(card, "added_to_deck", False):
            card.added_to_deck = False
            gs["blind_redebuff_pending"] = True


def run_pending_blind_redebuff(gs: dict[str, Any]) -> None:
    """The ``set_blind(nil, true)`` a deck-added playing card's removal queued
    (see :func:`_playing_cards_removed`)."""
    if gs.pop("blind_redebuff_pending", False):
        reapply_blind_debuffs(gs)


def _fire_discard_effects(gs: dict[str, Any], discarded: list, *, hook: bool) -> None:
    """Seal effects + per-card joker ``discard`` contexts + pile move for
    a forced (Hook) discard.

    Mirrors the per-card loop of ``discard_cards_from_highlighted``
    (state_events.lua:399-430) which runs for hook discards too; the
    counter block (discards_left/discards_used/redraw) is ``not hook``
    guarded there and is NOT applied here.
    """
    from jackdaw.engine.jokers import JokerContext, calculate_joker, effect_flags

    jokers: list = gs.get("jokers", [])
    rng = gs.get("rng")
    game_snap = _build_discard_snapshot(gs, jokers)
    flags = effect_flags(jokers)  # Faceless / Castle: is_face, is_suit

    dollars_earned = 0
    destroyed: list = []
    jokers_to_remove: list = []
    purple_tarots: list = []

    for card in discarded:
        # Purple Seal → random Tarot ('8ba' append); nothing for a debuffed
        # card (card.lua:2243)
        _purple_seal(gs, card, purple_tarots)

        card_destroyed = False
        for joker in jokers:
            if getattr(joker, "debuff", False):
                continue
            ctx = JokerContext(
                discard=True,
                hook=hook,
                other_card=card,
                full_hand=discarded,
                jokers=jokers,
                rng=rng,
                game=game_snap,
                **flags,
            )
            result = calculate_joker(joker, ctx)
            if result:
                dollars_earned += result.dollars
                if result.remove:
                    if result.extra and result.extra.get("destroy"):
                        card_destroyed = True
                    elif joker not in jokers_to_remove:
                        jokers_to_remove.append(joker)
        if card_destroyed:
            # Destroyed on discard, a Glass Card shatters on the spot
            # (state_events.lua:413) — Glass Joker counts it.
            if card.ability.get("name") == "Glass Card":
                card.shattered = True
            destroyed.append(card)

    if purple_tarots:
        _resolve_create_descriptors(gs, purple_tarots)
    if dollars_earned:
        gs["dollars"] = gs.get("dollars", 0) + dollars_earned
    for joker in jokers_to_remove:
        if joker in jokers:
            jokers.remove(joker)
            joker.remove_from_deck(gs)
            _release_used_key(gs, joker)

    # remove_playing_cards joker notify (state_events.lua:426)
    _notify_cards_destroyed(gs, destroyed)

    surviving = [c for c in discarded if c not in destroyed]
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(surviving)
    gs["round_scores"] = gs.get("round_scores", {})
    gs["round_scores"]["cards_discarded"] = gs["round_scores"].get("cards_discarded", 0) + len(
        discarded
    )


def _press_play(
    gs: dict[str, Any],
    blind: Any,
    played: list,
    rng: Any,
) -> None:
    """Fire boss blind press_play effects before scoring.

    Mirrors ``Blind:press_play`` (blind.lua:464-502).
    """
    if getattr(blind, "disabled", False):
        return

    name = getattr(blind, "name", "")

    if name == "The Hook":
        # Discard 2 random cards from hand.  Vanilla routes them through
        # discard_cards_from_highlighted(nil, HOOK) (blind.lua:482 →
        # state_events.lua:379): per-card SEAL effects and joker
        # `discard` contexts DO fire (Green Joker loses mult — the
        # LSN6STIA off-by-one), only the discard counters and Burnt
        # Joker (hook-gated) are skipped.
        hand: list = gs.get("hand", [])
        position = {id(c): i for i, c in enumerate(hand)}
        hooked: list = []
        for _ in range(min(2, len(hand))):
            if hand and rng:
                seed_val = rng.seed("hook")
                target, _ = rng.element(hand, seed_val)
                hand.remove(target)
                hooked.append(target)
        # discard_cards_from_highlighted sorts the highlighted cards by
        # position (state_events.lua:392), not by pick order.
        hooked.sort(key=lambda c: position[id(c)])
        if hooked:
            _fire_discard_effects(gs, hooked, hook=True)
        blind.triggered = True

    elif name == "The Tooth":
        # Lose $1 per card played
        gs["dollars"] = gs.get("dollars", 0) - len(played)
        blind.triggered = True

    elif name == "The Fish":
        # Flip all hand cards face-down after play (blind.lua:494-496)
        blind.prepped = True

    elif name == "Crimson Heart":
        # Debuff a random joker each hand (blind.lua:488-493)
        jokers: list = gs.get("jokers", [])
        if jokers and rng:
            blind.triggered = True
            blind.prepped = True


# ---------------------------------------------------------------------------
# Double Tag check
# ---------------------------------------------------------------------------


def _check_double_tag(
    gs: dict[str, Any], awarded_tag_key: str, orbital_hand: str | None = None
) -> None:
    """Each held Double Tag converts into a copy of the next tag acquired.

    tag.lua 'tag_added' context: the Double is consumed and a duplicate of
    the new tag is added; a Double never duplicates another Double.  The
    copy keeps the original's Orbital hand (``G.orbital_hand``,
    tag.lua:324-328).

    BUG HISTORY: this used to read ``gs["tags"]`` — a key nothing writes
    (tags live in ``awarded_tags``) — so Double Tag NEVER duplicated
    anything.  Found by lockstep seed LSVZ27Z7: skipping into a Buffoon
    Tag while holding a Double opened two Mega Buffoon packs live, one in
    the sim.
    """
    if awarded_tag_key == "tag_double":
        return

    from jackdaw.engine.tags import Tag

    awarded: list = gs.setdefault("awarded_tags", [])
    doubles = [e for e in awarded if e.get("key") == "tag_double"]
    for dbl in doubles:
        dup = Tag(awarded_tag_key, orbital_hand=orbital_hand)
        dup_result = dup.apply("immediate", gs, rng=gs.get("rng"))
        awarded.append(
            {
                "key": awarded_tag_key,
                "result": dup_result,
                "blind": "double",
            }
        )
        if dup_result is not None:
            _apply_tag_result(gs, dup_result)
        awarded.remove(dbl)  # consumed
