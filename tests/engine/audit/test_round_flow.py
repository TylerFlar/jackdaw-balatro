"""Audit regressions: round flow (draws, hand-size changes, draw-to-hand order,
running out of cards).

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

from jackdaw.engine.actions import (
    Discard,
    GamePhase,
    PlayHand,
    SelectBlind,
    SellCard,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.game import forced_card_index, step
from jackdaw.engine.rng import PseudoRandom
from tests.engine.audit._boards import force_boss, give_consumable, give_joker, ids, new_run

# ---------------------------------------------------------------------------
# G14 — first_hand_drawn runs before the boss's drawn_to_hand
# ---------------------------------------------------------------------------


def test_g14_certificate_card_is_dealt_before_cerulean_bell_picks():
    """update_draw_to_hand fires the first_hand_drawn context (Certificate's
    card) and only then, in a later event, switches to SELECTING_HAND and
    calls Blind:drawn_to_hand (game.lua:3226-3239): Cerulean Bell picks its
    forced card from the hand with Certificate's card in it."""
    seed = "AUDIT_G14"
    gs = new_run(seed)
    give_joker(gs, "j_certificate")
    force_boss(gs, "bl_final_bell")

    step(gs, SelectBlind())

    hand = gs["hand"]
    assert len(hand) == 9  # 8 dealt + Certificate's card
    ref = PseudoRandom(seed)
    expected, _ = ref.element(list(hand), ref.seed("cerulean_bell"))
    assert hand[forced_card_index(hand)] is expected


# ---------------------------------------------------------------------------
# G36 — running out of cards ends the round; no hand size is game over
# ---------------------------------------------------------------------------


def test_g36_an_emptied_hand_is_redrawn():
    """Game:update moves an empty SELECTING_HAND hand with cards left in the
    deck to DRAW_TO_HAND (game.lua:2544-2548): Immolate burning a 5-card hand
    is followed by a fresh draw."""
    gs = new_run("AUDIT_G36D")
    give_joker(gs, "j_stuntman")  # -2
    give_joker(gs, "j_merry_andy")  # -1
    step(gs, SelectBlind())
    assert len(gs["hand"]) == 5
    give_consumable(gs, "c_immolate")

    step(gs, UseConsumable(card_index=0))

    assert gs["phase"] == GamePhase.SELECTING_HAND
    assert len(gs["hand"]) == 5


def test_g36_round_ends_when_hand_and_deck_run_out():
    """With hand, deck and play all empty, update_selecting_hand calls
    end_round() (game.lua:3056-3069); short of the target that is a loss
    (state_events.lua:96-116) even with hands left."""
    gs = new_run("AUDIT_G36E")
    step(gs, SelectBlind())
    gs["deck"].clear()

    step(gs, Discard(card_indices=(0, 1, 2, 3, 4)))
    assert len(gs["hand"]) == 3 and gs["phase"] == GamePhase.SELECTING_HAND
    step(gs, PlayHand(card_indices=(0, 1, 2)))

    assert gs["current_round"]["hands_left"] > 0
    assert gs["phase"] == GamePhase.GAME_OVER


def test_g36_mr_bones_saves_a_round_that_ran_out_of_cards():
    """end_round's end_of_round pass lets Mr. Bones save a round that ran out
    of cards at 25% of the target (card.lua:3047-3055)."""
    gs = new_run("AUDIT_G36B")
    give_joker(gs, "j_mr_bones")
    step(gs, SelectBlind())
    gs["deck"].clear()
    gs["chips"] = gs["blind"].chips // 2

    step(gs, Discard(card_indices=(0, 1, 2, 3, 4)))
    step(gs, Discard(card_indices=(0, 1, 2)))

    assert gs["phase"] == GamePhase.ROUND_EVAL
    assert not any(j.center_key == "j_mr_bones" for j in gs["jokers"])


def test_g36_no_hand_size_is_game_over():
    """draw_from_deck_to_hand: with a hand limit of 0 or less and an empty
    hand the game is over (state_events.lua:355-360)."""
    gs = new_run("AUDIT_G36Z")
    for _ in range(4):
        give_joker(gs, "j_stuntman")  # -2 each
    assert gs["hand_size"] == 0

    step(gs, SelectBlind())

    assert gs["phase"] == GamePhase.GAME_OVER
    assert get_legal_actions(gs) == []


# ---------------------------------------------------------------------------
# G37 — growing the hand size mid-round draws at once
# ---------------------------------------------------------------------------


def test_g37_growing_the_hand_size_mid_round_draws_the_difference():
    """G.hand:change_size(delta) (cardarea.lua:94-111): when the hand grows
    while it holds cards in SELECTING_HAND (or DRAW_TO_HAND), it draws
    ``delta`` cards from the deck at once, then sorts.  Selling Stuntman
    gives back its 2 hand size (card.lua:684-686), so 2 cards are drawn."""
    gs = new_run("AUDIT_G37")
    give_joker(gs, "j_stuntman")
    step(gs, SelectBlind())
    assert gs["hand_size"] == 6 and len(gs["hand"]) == 6
    top_two = gs["deck"][-2:]

    step(gs, SellCard(area="jokers", card_index=0))

    assert gs["hand_size"] == 8
    assert len(gs["hand"]) == 8
    assert set(ids(top_two)) <= set(ids(gs["hand"]))
    hand = gs["hand"]
    assert ids(hand) == ids(sorted(hand, key=lambda c: c.get_nominal(), reverse=True))


def test_g37_hand_size_draws_use_the_boss_flip_rules():
    """The change_size draws go through draw_card -> draw_card_from, which asks
    Blind:stay_flipped (cardarea.lua:600; blind.lua:611): under The House,
    before any play or discard, they arrive face down."""
    gs = new_run("AUDIT_G37H")
    give_joker(gs, "j_stuntman")
    force_boss(gs, "bl_house")
    step(gs, SelectBlind())
    top_two = gs["deck"][-2:]

    step(gs, SellCard(area="jokers", card_index=0))

    assert all(c.facing == "back" for c in top_two)


def test_g37_shrinking_the_hand_size_draws_nothing():
    """change_size only draws for delta > 0; a smaller hand keeps its cards
    (cardarea.lua:98-100)."""
    gs = new_run("AUDIT_G37S")
    give_joker(gs, "j_juggler")
    step(gs, SelectBlind())
    assert len(gs["hand"]) == 9

    step(gs, SellCard(area="jokers", card_index=0))

    assert gs["hand_size"] == 8
    assert len(gs["hand"]) == 9
    step(gs, Discard(card_indices=(0,)))  # the next draw refills to 8 only
    assert len(gs["hand"]) == 8
