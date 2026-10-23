"""Audit regressions: run setup, decks and challenge starting state (G01-G03, G06)."""

from __future__ import annotations

from jackdaw.engine import CashOut, GamePhase, SelectBlind, step
from tests.engine.audit._helpers import new_run, win_blind

# ---------------------------------------------------------------------------
# G01 — Green Deck economy
# ---------------------------------------------------------------------------


def test_green_deck_pays_per_hand_and_discard_and_no_interest():
    """Green Deck: $2 per unused hand, $1 per unused discard, no interest.

    back.lua:269-277 (Back:apply_to_run) stores the three values in
    ``G.GAME.modifiers``; state_events.lua:1165-1191 (evaluate_round) reads
    them from there: ``hands_left*(modifiers.money_per_hand or 1)``,
    ``discards_left*modifiers.money_per_discard`` and no interest row when
    ``modifiers.no_interest`` is set.
    """
    gs = new_run("b_green", "AUDIT_G01")
    assert gs["modifiers"]["money_per_hand"] == 2
    assert gs["modifiers"]["money_per_discard"] == 1
    assert gs["modifiers"]["no_interest"] is True

    step(gs, SelectBlind())
    gs["dollars"] = 50  # interest would be $5 on a normal deck
    win_blind(gs)

    cr = gs["current_round"]
    earnings = gs["round_earnings"]
    assert cr["hands_left"] == 3 and cr["discards_left"] == 3
    assert earnings.unused_hands_bonus == 3 * 2
    assert earnings.unused_discards_bonus == 3 * 1
    assert earnings.interest == 0

    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    # blind reward $3 + $6 hands + $3 discards, nothing else
    assert gs["dollars"] == 50 + 3 + 6 + 3
