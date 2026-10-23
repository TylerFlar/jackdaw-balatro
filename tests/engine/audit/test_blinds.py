"""Audit regressions: blinds (boss effects, disable, defeat, debuff re-checks,
blind amounts).

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

import math

import pytest

from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    PlayHand,
    SelectBlind,
    SellCard,
    UseConsumable,
)
from jackdaw.engine.blind import Blind
from jackdaw.engine.data.blind_scaling import get_blind_amount, get_blind_target
from jackdaw.engine.data.enums import Suit
from jackdaw.engine.game import forced_card_index, step
from jackdaw.engine.rng import PseudoRandom
from tests.engine.audit._boards import (
    force_boss,
    give_consumable,
    give_joker,
    ids,
    index_of,
    new_run,
    playing_card,
    start_boss,
    win_round,
)
from tests.parity import canon
from tests.parity.py_export import export_state

# ---------------------------------------------------------------------------
# G07 — Blind:disable in full; Chicot disables through it
# ---------------------------------------------------------------------------


def _target(boss: str, ante: int = 1) -> float:
    return Blind.create(boss, ante).chips


def test_g07_chicot_halves_the_wall():
    """Chicot's setting_blind queues G.GAME.blind:disable() (card.lua:2492-2500),
    which halves The Wall's target (blind.lua:377-380)."""
    gs = new_run("AUDIT_G07W")
    give_joker(gs, "j_chicot")

    start_boss(gs, "bl_wall")

    assert gs["blind"].disabled
    assert gs["blind"].chips == _target("bl_wall") / 2


def test_g07_chicot_divides_violet_vessel_by_three():
    """blind.lua:393-396."""
    gs = new_run("AUDIT_G07V")
    give_joker(gs, "j_chicot")
    gs["round_resets"]["ante"] = 8

    start_boss(gs, "bl_final_vessel")

    assert gs["blind"].chips == _target("bl_final_vessel", 8) / 3


def test_g07_chicot_gives_back_what_the_water_and_the_needle_took():
    """set_blind applies The Water / The Needle first (blind.lua:179-186);
    the deferred disable eases the discards / hands back (blind.lua:361-376)."""
    cases = (("bl_water", "discards_left", "discards"), ("bl_needle", "hands_left", "hands"))
    for boss, key, reset in cases:
        gs = new_run("AUDIT_G07N")
        give_joker(gs, "j_chicot")
        start_boss(gs, boss)
        assert gs["current_round"][key] == gs["round_resets"][reset], boss
        assert gs["blind"].disabled


def test_g07_chicot_does_not_stop_amber_acorns_shuffles():
    """Amber Acorn's three 'aajk' shuffles are queued by set_blind before
    Chicot's disable runs (blind.lua:195-203; card.lua:2492-2500); the disable
    then turns the jokers face up (blind.lua:358-360)."""
    seed = "AUDIT_G07A"
    gs = new_run(seed)
    board = [give_joker(gs, k) for k in ("j_chicot", "j_joker", "j_greedy_joker")]

    start_boss(gs, "bl_final_acorn")

    ref = PseudoRandom(seed)
    order = list(board)
    for _ in range(3):
        ref.shuffle(order, ref.seed("aajk"))
    assert ids(gs["jokers"]) == ids(order)
    assert all(j.facing == "front" for j in gs["jokers"])


def test_g07_disabling_the_wall_mid_round_can_end_the_round():
    """Blind:disable queues a check: a boss whose (halved) target the score
    already meets ends the round at once (blind.lua:397-406)."""
    gs = new_run("AUDIT_G07E")
    give_joker(gs, "j_luchador")
    start_boss(gs, "bl_wall")
    gs["chips"] = gs["blind"].chips // 2

    step(gs, SellCard(area="jokers", card_index=0))

    assert gs["phase"] == GamePhase.ROUND_EVAL


def test_g07_disabling_the_manacle_draws_two_cards():
    """Disabling The Manacle changes the hand size by +1 — which, mid-round
    with cards in hand, draws one card (cardarea.lua:100-105) — and calls
    draw_from_deck_to_hand(1) for another (blind.lua:386-390)."""
    gs = new_run("AUDIT_G07M")
    give_joker(gs, "j_luchador")
    start_boss(gs, "bl_manacle")
    assert gs["hand_size"] == 7 and len(gs["hand"]) == 7

    step(gs, SellCard(area="jokers", card_index=0))

    assert gs["hand_size"] == 8
    assert len(gs["hand"]) == 9


def test_g07_disabling_the_house_turns_the_hand_face_up():
    """blind.lua:364-373: The Wheel, The House, The Mark and The Fish flip
    the face-down hand cards back up when disabled."""
    gs = new_run("AUDIT_G07H")
    give_joker(gs, "j_luchador")
    start_boss(gs, "bl_house")
    assert all(c.facing == "back" for c in gs["hand"])

    step(gs, SellCard(area="jokers", card_index=0))

    assert all(c.facing == "front" for c in gs["hand"])


def test_g07_disabling_crimson_heart_frees_its_victim_and_passive():
    """The disable's debuff pass runs debuff_card on every joker
    (blind.lua:410-412): Crimson Heart's victim comes back with its passive."""
    gs = new_run("AUDIT_G07C")
    juggler = give_joker(gs, "j_juggler")
    give_joker(gs, "j_luchador")
    gs["jokers"].reverse()  # Juggler last on the board, Luchador first
    start_boss(gs, "bl_final_heart")
    assert [j for j in gs["jokers"] if j.debuff] == [juggler]  # this seed's pick
    assert gs["hand_size"] == 8

    step(gs, SellCard(area="jokers", card_index=0))  # Luchador

    assert not juggler.debuff
    assert gs["hand_size"] == 9


def test_g07_chicot_created_mid_round_disables_the_boss():
    """Card:add_to_deck disables an active, not yet disabled boss for Chicot
    (card.lua:596-600) — a Chicot made mid-round by The Soul or Judgement."""
    gs = start_boss(new_run("AUDIT_G07K"), "bl_wall")
    target = gs["blind"].chips

    give_joker(gs, "j_chicot")

    assert gs["blind"].disabled
    assert gs["blind"].chips == target / 2


def test_g07_disabling_cerulean_bell_releases_the_forced_card():
    """blind.lua:381-385: the forced selection is cleared on every card."""
    gs = new_run("AUDIT_G07B")
    give_joker(gs, "j_luchador")
    start_boss(gs, "bl_final_bell")
    assert forced_card_index(gs["hand"]) is not None

    step(gs, SellCard(area="jokers", card_index=0))

    assert forced_card_index(gs["hand"]) is None


# ---------------------------------------------------------------------------
# G08 — selling any joker disables Verdant Leaf
# ---------------------------------------------------------------------------


def test_g08_selling_any_joker_disables_verdant_leaf():
    """Card:sell_card queues G.GAME.blind:disable() when a Joker is sold
    under Verdant Leaf (card.lua:1616-1621); the disable's debuff pass frees
    every playing card (blind.lua:407-409)."""
    gs = new_run("AUDIT_G08")
    give_joker(gs, "j_joker")
    start_boss(gs, "bl_final_leaf")
    assert all(c.debuff for c in gs["hand"])

    step(gs, SellCard(area="jokers", card_index=0))

    assert gs["blind"].disabled
    assert not any(c.debuff for c in gs["hand"] + gs["deck"])


def test_g08_selling_a_consumable_keeps_verdant_leaf():
    """Only a sold Joker counts (``self.ability.set == 'Joker'``)."""
    gs = new_run("AUDIT_G08C")
    start_boss(gs, "bl_final_leaf")
    give_consumable(gs, "c_pluto")

    step(gs, SellCard(area="consumables", card_index=0))

    assert not gs["blind"].disabled


# ---------------------------------------------------------------------------
# G09 — The Ox: the frozen most-played hand; money zeroed before scoring
# ---------------------------------------------------------------------------


def _ox(seed: str, *jokers: str) -> dict:
    gs = new_run(seed)
    for key in jokers:
        give_joker(gs, key)
    start_boss(gs, "bl_ox")
    gs["dollars"] = 10
    return gs


def test_g09_ox_compares_with_the_frozen_most_played_hand():
    """The Ox triggers on ``G.GAME.current_round.most_played_poker_hand``
    (blind.lua:560-569), which only end_round sets, after a boss is beaten
    (state_events.lua:129-138); before that it is 'High Card' (game.lua:1964).
    A Pair at the run's first boss keeps the money even though the live play
    counts already make Pair the most played."""
    gs = _ox("AUDIT_G09")  # this seed deals a pair of 3s at indices 6 and 7
    assert gs["hand"][6].base.id == gs["hand"][7].base.id == 3

    step(gs, PlayHand(card_indices=(6, 7)))

    assert gs["last_score_result"].hand_type == "Pair"
    assert gs["dollars"] == 10


def test_g09_ox_zeroes_money_before_the_hand_scores():
    """debuff_hand runs before scoring (state_events.lua:614) and The Ox's
    ease_dollars(-G.GAME.dollars, true) is instant (blind.lua:565): money
    earned while the hand scores — Rough Gem's $1 for a scored Diamond — is
    kept."""
    gs = _ox("AUDIT_G09", "j_rough_gem")
    assert gs["hand"][0].base.suit is Suit.DIAMONDS

    step(gs, PlayHand(card_indices=(0,)))  # High Card: the frozen most-played hand

    assert gs["blind"].triggered
    assert gs["dollars"] == 1


def test_g09_ox_trigger_pays_matador():
    """The trigger is set inside debuff_hand, before the jokers score, so
    Matador's joker_main sees it (card.lua:3755-3760)."""
    gs = _ox("AUDIT_G09", "j_matador")

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["dollars"] == 8


def test_g09_most_played_hand_is_set_when_a_boss_is_beaten():
    """end_round recomputes most_played_poker_hand after a boss only
    (state_events.lua:129-138): beating the Small Blind with Pairs leaves it
    'High Card'; beating a boss sets it."""
    gs = new_run("AUDIT_G09")
    step(gs, SelectBlind())
    pair = next(
        (i, i + 1)
        for i in range(len(gs["hand"]) - 1)
        if gs["hand"][i].base.id == gs["hand"][i + 1].base.id
    )
    gs["chips"] = gs["blind"].chips - 1
    step(gs, PlayHand(card_indices=pair))
    assert gs["phase"] == GamePhase.ROUND_EVAL
    assert gs["current_round"]["most_played_poker_hand"] == "High Card"

    step(gs, CashOut())
    gs["phase"] = GamePhase.BLIND_SELECT  # straight on to the boss
    start_boss(gs, "bl_hook")
    win_round(gs)  # one High Card: Pair 1, High Card 1 — a tie

    # Ties go to the hand pairs(G.GAME.hands) visits last (end_round never
    # updates its _order, so ``_order > v.order`` is always true); High Card
    # comes after Pair in that order (G182).
    assert gs["current_round"]["most_played_poker_hand"] == "High Card"


def test_g09_most_played_tie_break_follows_the_pairs_order():
    """The order pairs() visits G.GAME.hands in is not in the source; jackdaw
    fixes it once (HANDS_PAIRS_ORDER) for every such loop, the same order
    To Do List and the Orbital Tag use."""
    from jackdaw.engine.data.hands import HANDS_PAIRS_ORDER
    from jackdaw.engine.hand_levels import HandLevels, most_played_poker_hand
    from jackdaw.engine.tags import _ORBITAL_HANDS

    assert list(HANDS_PAIRS_ORDER) == _ORBITAL_HANDS
    levels = HandLevels()
    assert most_played_poker_hand(levels) == HANDS_PAIRS_ORDER[-1].value  # all 0: the last
    for hand in ("Flush", "Pair", "High Card"):
        levels.record_play(hand)
    assert most_played_poker_hand(levels) == "High Card"  # last of the three (G182)
    levels.record_play("Pair")
    assert most_played_poker_hand(levels) == "Pair"


# ---------------------------------------------------------------------------
# G11 — Crimson Heart re-picks only after a play; its debuff toggles passives
# ---------------------------------------------------------------------------


def _crimson_seeds(gs) -> float | None:
    return gs["rng"].state.get("crimson_heart")


def test_g11_crimson_heart_repicks_after_a_play_not_a_discard():
    """drawn_to_hand re-picks only when ``self.prepped`` (blind.lua:588);
    set_blind sets it (blind.lua:94), Blind:press_play sets it on a play
    (blind.lua:488-493) and every drawn_to_hand clears it (blind.lua:602).
    A discard's redraw keeps the debuffed joker and draws no 'crimson_heart'
    seed."""
    gs = new_run("AUDIT_G11")
    for key in ("j_joker", "j_greedy_joker", "j_lusty_joker"):
        give_joker(gs, key)
    start_boss(gs, "bl_final_heart")
    victim = [j for j in gs["jokers"] if j.debuff]
    assert len(victim) == 1
    stream = _crimson_seeds(gs)

    step(gs, Discard(card_indices=(0,)))

    assert [j for j in gs["jokers"] if j.debuff] == victim
    assert _crimson_seeds(gs) == stream

    step(gs, PlayHand(card_indices=(0,)))

    assert _crimson_seeds(gs) != stream
    assert sum(j.debuff for j in gs["jokers"]) == 1


def test_g11_crimson_heart_picks_in_sort_id_order():
    """pseudorandom_element sorts a list of cards by sort_id before drawing
    (misc_functions.lua:253-264), so the pick does not follow the board's
    order."""
    seed = "AUDIT_G11O"
    gs = new_run(seed)
    jokers = [give_joker(gs, key) for key in ("j_joker", "j_greedy_joker", "j_lusty_joker")]
    gs["jokers"].reverse()  # board order no longer matches creation order
    ref = PseudoRandom(seed)
    expected, _ = ref.element(jokers, ref.seed("crimson_heart"))

    start_boss(gs, "bl_final_heart")

    assert [j for j in gs["jokers"] if j.debuff] == [expected]


def test_g11_crimson_heart_debuff_takes_the_passive_away():
    """The pick goes through Card:set_debuff, which removes the joker's
    passives (card.lua:535): a debuffed Juggler gives its hand size back
    until the defeat frees it again."""
    gs = new_run("AUDIT_G11P")
    juggler = give_joker(gs, "j_juggler")
    start_boss(gs, "bl_final_heart")
    assert juggler.debuff
    assert gs["hand_size"] == 8
    assert len(gs["hand"]) == 9  # dealt before drawn_to_hand debuffed it

    win_round(gs)

    assert not juggler.debuff
    assert gs["hand_size"] == 9


# ---------------------------------------------------------------------------
# G12 — the blind is reset after defeat (and is empty before the first one)
# ---------------------------------------------------------------------------


def _assert_empty_blind(blind) -> None:
    """set_blind(nil): no name, no boss, 0 chips/mult/dollars, not disabled
    (blind.lua:79-110)."""
    assert blind.key == "" and blind.name == ""
    assert not blind.boss and not blind.disabled and not blind.triggered
    assert blind.chips == 0 and blind.mult == 0 and blind.dollars == 0


def test_g12_the_run_starts_with_the_empty_blind():
    """start_run creates G.GAME.blind and calls set_blind(nil, nil, true)
    (game.lua:2392, 2431)."""
    _assert_empty_blind(new_run("AUDIT_G12E")["blind"])


def test_g12_a_defeated_boss_is_reset_for_the_shop():
    """evaluate_round calls Blind:defeat (state_events.lua:1148-1155), whose
    queued event runs set_blind(nil, nil, true) (blind.lua:330-337): in the
    cash-out screen and the shop G.GAME.blind is the empty blind."""
    gs = start_boss(new_run("AUDIT_G12"), "bl_manacle")
    win_round(gs)
    _assert_empty_blind(gs["blind"])

    step(gs, CashOut())

    _assert_empty_blind(gs["blind"])


def test_g12_selling_luchador_in_the_shop_disables_nothing():
    """Luchador only disables a current boss (``get_type() == 'Boss'``,
    card.lua:2355-2359); after The Manacle is defeated (its +1 hand size given
    back at defeat, blind.lua:341-343) selling Luchador in the shop must not
    hand out another +1."""
    gs = new_run("AUDIT_G12L")
    give_joker(gs, "j_luchador")
    start_boss(gs, "bl_manacle")
    hand_size = gs["hand_size"] + 1  # The Manacle's -1 comes back at defeat
    win_round(gs)
    step(gs, CashOut())
    assert gs["hand_size"] == hand_size

    step(gs, SellCard(area="jokers", card_index=0))

    assert gs["hand_size"] == hand_size


def test_g12_defeat_lifts_crimson_heart_joker_debuff():
    """set_blind(nil)'s pass runs debuff_card on every joker (blind.lua:211-213),
    and with no blind left that un-debuffs Crimson Heart's victim."""
    gs = new_run("AUDIT_G12C")
    joker = give_joker(gs, "j_joker")
    start_boss(gs, "bl_final_heart")
    assert joker.debuff

    win_round(gs)

    assert not joker.debuff


# ---------------------------------------------------------------------------
# G13 — boss debuffs are re-checked when a card or the joker set changes
# ---------------------------------------------------------------------------


def _club_in_hand(gs):
    return next(c for c in gs["hand"] if c.base.suit is Suit.CLUBS)


def test_g13_suit_change_lifts_a_suit_debuff_at_once():
    """change_suit -> set_base runs G.GAME.blind:debuff_card(self) and so does
    change_suit itself (card.lua:143, 560-561): a Club turned into a Diamond
    by The Star under The Club is no longer debuffed, so it scores at once."""
    gs = start_boss(new_run("AUDIT_G13"), "bl_club")
    club = _club_in_hand(gs)
    assert club.debuff
    give_consumable(gs, "c_star")

    step(gs, UseConsumable(card_index=0, target_indices=(index_of(gs["hand"], club),)))

    assert club.base.suit is Suit.DIAMONDS
    assert not club.debuff
    step(gs, PlayHand(card_indices=(index_of(gs["hand"], club),)))
    assert not gs["last_score_result"].debuffed
    assert gs["last_score_result"].chips > 5  # the card's own chips counted


def test_g13_enhancement_change_rechecks_the_debuff():
    """set_ability (not initial) ends with G.GAME.blind:debuff_card(self)
    (card.lua:365).  Under The Club, The Lovers' Wild Card is_suit('Clubs')
    and is debuffed; The Tower's Stone Card is no suit and is freed."""
    gs = start_boss(new_run("AUDIT_G13E"), "bl_club")
    other = next(c for c in gs["hand"] if c.base.suit is not Suit.CLUBS)
    club = _club_in_hand(gs)
    give_consumable(gs, "c_lovers")
    give_consumable(gs, "c_tower")

    step(gs, UseConsumable(card_index=0, target_indices=(index_of(gs["hand"], other),)))
    step(gs, UseConsumable(card_index=0, target_indices=(index_of(gs["hand"], club),)))

    assert other.debuff  # now Wild: counts as a Club
    assert not club.debuff  # now Stone: no suit


def test_g13_rank_change_rechecks_a_face_debuff():
    """Strength's set_base (card.lua:1134) re-checks too: under The Plant a 10
    raised to a Jack is a face card and is debuffed at once."""
    gs = start_boss(new_run("AUDIT_G13R"), "bl_plant")  # this seed deals a 10
    ten = next(c for c in gs["hand"] if c.base.id == 10)
    assert not ten.debuff
    give_consumable(gs, "c_strength")

    step(gs, UseConsumable(card_index=0, target_indices=(index_of(gs["hand"], ten),)))

    assert ten.base.id == 11
    assert ten.debuff


def test_g13_selling_pareidolia_rechecks_every_card():
    """remove_from_deck queues G.GAME.blind:set_blind(nil, true) (card.lua:698),
    a reset that re-runs debuff_card over G.playing_cards (blind.lua:207-210).
    Under The Plant with Pareidolia every card is a face card and debuffed;
    selling Pareidolia frees the non-face cards at once."""
    gs = new_run("AUDIT_G13P")
    give_joker(gs, "j_pareidolia")
    force_boss(gs, "bl_plant")
    step(gs, SelectBlind())
    assert all(c.debuff for c in gs["hand"])

    step(gs, SellCard(area="jokers", card_index=0))

    for card in gs["hand"] + gs["deck"]:
        assert card.debuff == (card.get_id() in (11, 12, 13))


def test_g13_adding_pareidolia_rechecks_every_card():
    """add_to_deck queues the same reset (card.lua:641)."""
    gs = start_boss(new_run("AUDIT_G13A"), "bl_plant")
    assert not all(c.debuff for c in gs["hand"])

    give_joker(gs, "j_pareidolia")

    assert all(c.debuff for c in gs["hand"] + gs["deck"])


def test_g13_cards_created_mid_round_are_not_debuffed_by_a_draw():
    """Debuffs come from set_blind's pass over G.playing_cards (blind.lua:207-210)
    and the re-checks above — never from a draw.  A face card made mid-round with
    no debuff_card call stays playable under The Plant through the next draw.
    (Familiar's cards are made that way, card.lua:1320-1335, but the used
    Familiar's Card:remove re-checks every playing card, G140.)"""
    gs = start_boss(new_run("AUDIT_G13F"), "bl_plant")
    made = [playing_card(s, r) for s, r in (("Hearts", "King"), ("Spades", "Queen"))]
    gs["hand"].extend(made)
    assert not any(c.debuff for c in made)
    keep = {id(c) for c in made}

    discard = tuple(i for i, c in enumerate(gs["hand"]) if id(c) not in keep)[:1]
    step(gs, Discard(card_indices=discard))

    assert not any(c.debuff for c in made)


def test_g13_familiar_cards_meet_the_boss_when_the_familiar_goes():
    """Familiar's new face cards skip debuff_card (create_playing_card,
    card.lua:1320-1335), but the used Familiar's Card:remove runs remove_from_deck,
    whose set_blind(nil, true) re-checks every playing card (card.lua:696-698;
    G140): under The Plant they end debuffed."""
    gs = start_boss(new_run("AUDIT_G13F"), "bl_plant")
    give_consumable(gs, "c_familiar")
    before = list(gs["hand"])
    step(gs, UseConsumable(card_index=0))
    made = [c for c in gs["hand"] if all(c is not b for b in before)]
    assert len(made) == 3 and all(c.debuff for c in made)


def test_g13_vampire_rechecks_the_card_it_strips():
    """Vampire strips an enhancement with set_ability(c_base) in the 'before'
    pass (card.lua:3465-3482), which re-checks the debuff (card.lua:365): a
    Stone Club under The Club becomes a plain, debuffed Club mid-play."""
    gs = start_boss(new_run("AUDIT_G13V"), "bl_club")
    club = _club_in_hand(gs)
    give_consumable(gs, "c_tower")
    step(gs, UseConsumable(card_index=0, target_indices=(index_of(gs["hand"], club),)))
    assert not club.debuff
    give_joker(gs, "j_vampire")

    step(gs, PlayHand(card_indices=(index_of(gs["hand"], club),)))

    assert club.center_key == "c_base"
    assert club.debuff


# ---------------------------------------------------------------------------
# G10 — Amber Acorn flips at set time and shuffles three times with 'aajk'
# ---------------------------------------------------------------------------


def _three_aajk_shuffles(seed: str, cards: list) -> list:
    """What three ``G.jokers:shuffle('aajk')`` calls leave, from a fresh stream."""
    ref = PseudoRandom(seed)
    order = list(cards)
    for _ in range(3):
        ref.shuffle(order, ref.seed("aajk"))
    return order


def test_g10_amber_acorn_shuffles_three_times():
    """Amber Acorn's set_blind queues three ``G.jokers:shuffle('aajk')`` events
    (blind.lua:195-203); each pseudoseed('aajk') call advances the stream, so
    the board ends in the third shuffle's order with the stream three steps
    on.  The jokers were flipped face down first (blind.lua:190-194)."""
    seed = "AUDIT_G10"
    gs = new_run(seed)
    board = [give_joker(gs, key) for key in ("j_joker", "j_greedy_joker", "j_lusty_joker")]
    force_boss(gs, "bl_final_acorn")

    step(gs, SelectBlind())

    assert ids(gs["jokers"]) == ids(_three_aajk_shuffles(seed, board))
    ref = PseudoRandom(seed)
    for _ in range(3):
        ref.seed("aajk")
    assert gs["rng"].state["aajk"] == ref.state["aajk"]
    assert all(j.facing == "back" for j in gs["jokers"])


def test_g10_jokers_made_at_setting_blind_are_shuffled_but_face_up():
    """The flip runs inside set_blind, before the jokers' setting_blind
    context (state_events.lua:333-337), so Riff-raff's new jokers are not
    flipped; the shuffles run later from queued events (blind.lua:196-203)
    and include them."""
    seed = "AUDIT_G10R"
    gs = new_run(seed)
    riff = give_joker(gs, "j_riff_raff")
    force_boss(gs, "bl_final_acorn")

    step(gs, SelectBlind())

    made = [j for j in gs["jokers"] if j is not riff]
    assert len(made) == 2
    assert riff.facing == "back"
    assert all(j.facing == "front" for j in made)
    assert ids(gs["jokers"]) == ids(_three_aajk_shuffles(seed, gs["jokers"]))


# ---------------------------------------------------------------------------
# G51 — get_blind_amount is NaN once the formula overflows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", [1, 2, 3])
@pytest.mark.parametrize("ante", [39, 40, 100])
def test_g51_overflowed_blind_amount_is_nan(scaling: int, ante: int):
    """get_blind_amount (misc_functions.lua:919-954) rounds with
    ``amount - amount % 10^floor(log10(amount) - 1)``.  Once the formula
    overflows (ante 39), amount is inf, log10 gives inf, and Lua's
    ``inf % inf`` is ``inf - floor(inf/inf)*inf`` = NaN, so it returns NaN —
    not inf.  The blind's chips (blind.lua:107) are NaN too."""
    assert math.isnan(get_blind_amount(ante, scaling))
    assert math.isnan(get_blind_target(ante, "Boss", scaling))
    assert math.isnan(Blind.create("bl_small", ante, scaling).chips)


def test_g51_last_finite_ante_is_unchanged():
    assert get_blind_amount(38, 1) == 4.5e288


def test_g51_nan_blind_can_never_be_beaten():
    """The win checks are ``G.GAME.chips - G.GAME.blind.chips >= 0``
    (state_events.lua:96, game.lua:3196): with a NaN target the difference
    is NaN and the comparison is false for any score, inf included."""
    blind = Blind.create("bl_small", 39)
    for chips in (0, 10**300, math.inf):
        assert not blind.beaten_by(chips)


def test_g51_nan_round_plays_to_game_over_without_crashing():
    """A run at ante 39 sets a NaN blind and plays on; when the hands run out
    the round is lost (end_round's game_over, state_events.lua:96-116)."""
    gs = new_run("AUDIT_G51")
    gs["round_resets"]["ante"] = 39
    step(gs, SelectBlind())
    assert math.isnan(gs["blind"].chips)
    gs["chips"] = math.inf
    canon.digest(export_state(gs))  # NaN and inf encode without raising

    while gs["phase"] == GamePhase.SELECTING_HAND:
        step(gs, PlayHand(card_indices=(0,)))

    assert gs["phase"] == GamePhase.GAME_OVER


def test_g51_disabling_a_nan_wall_keeps_nan():
    """Blind:disable halves The Wall (blind.lua:377-380): NaN/2 is NaN."""
    wall = Blind.create("bl_wall", 39)
    wall.disable()
    assert math.isnan(wall.chips)
    vessel = Blind.create("bl_final_vessel", 39)
    vessel.disable()
    assert math.isnan(vessel.chips)
