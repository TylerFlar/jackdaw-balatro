"""Audit regressions: round flow (draws, hand-size changes, draw-to-hand order,
running out of cards).

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

from jackdaw.engine.actions import Discard, SelectBlind, SellCard
from jackdaw.engine.game import step
from tests.engine.audit._boards import force_boss, give_joker, ids, new_run

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
