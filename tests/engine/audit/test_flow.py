"""Audit regressions: run flow (G48)."""

from __future__ import annotations

from jackdaw.engine import GamePhase, PlayHand, SelectBlind, step
from tests.engine.audit._helpers import new_run


def test_won_survives_a_later_endless_loss():
    """Once won, always won: a later game over in endless mode keeps G.GAME.won.

    state_events.lua:111-114 (end_round) sets ``G.GAME.won = true`` when the
    win-ante boss falls, and nothing ever clears it: the game-over branch
    (state_events.lua:115-122) only reads it.
    """
    gs = new_run(seed="AUDIT_G48")
    gs["won"] = True  # the ante-8 boss fell earlier; now in endless mode
    step(gs, SelectBlind())
    gs["current_round"]["hands_left"] = 1
    step(gs, PlayHand(card_indices=(0,)))  # one card cannot reach 300 chips
    assert gs["phase"] == GamePhase.GAME_OVER
    assert gs["won"] is True
