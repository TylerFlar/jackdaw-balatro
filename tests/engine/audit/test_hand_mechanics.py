"""Audit regressions: hand mechanics (play/discard order, sorting, deck emplace,
card bases, hand visibility, blind amounts).

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

from jackdaw.engine.actions import Discard, PlayHand, SelectBlind
from jackdaw.engine.game import step
from tests.engine.audit._boards import give_joker, ids, new_run

# ---------------------------------------------------------------------------
# G49 — played and discarded cards are processed in hand order
# ---------------------------------------------------------------------------


def test_g49_played_cards_score_in_hand_order():
    """play_cards_from_highlighted sorts G.hand.highlighted by T.x — the hand
    position — before moving the cards to G.play (state_events.lua:463), and
    evaluate_play scores them in play-area order (state_events.lua:600).  The
    order the indices were clicked in does not matter."""
    gs = new_run("AUDIT_G49")
    step(gs, SelectBlind())
    give_joker(gs, "j_splash")  # every played card scores
    hand = list(gs["hand"])
    first, second = hand[0], hand[3]

    step(gs, PlayHand(card_indices=(3, 0)))

    assert ids(gs["last_score_result"].scoring_cards) == ids([first, second])
    # draw_from_play_to_discard moves G.play.cards in order (state_events.lua:1091)
    assert ids(gs["discard_pile"]) == ids([first, second])


def test_g49_discarded_cards_follow_hand_order():
    """discard_cards_from_highlighted sorts the highlight by T.x too
    (state_events.lua:392) before the per-card discard loop."""
    gs = new_run("AUDIT_G49D")
    step(gs, SelectBlind())
    hand = list(gs["hand"])

    step(gs, Discard(card_indices=(4, 1)))

    assert ids(gs["discard_pile"]) == ids([hand[1], hand[4]])
