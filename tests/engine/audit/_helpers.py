"""Shared drivers for the Lua-audit regression tests.

The audit tests drive the public engine (``initialize_run`` + ``step``) and
only write state by hand where reaching it by play would take hundreds of
steps (a chip total just under the blind, a forced boss, a forced tag).
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine import (
    CashOut,
    GamePhase,
    PlayHand,
    SelectBlind,
    initialize_run,
    step,
)
from jackdaw.engine.challenges import get_challenge


def new_run(
    back: str = "b_red", seed: str = "AUDIT", *, stake: int = 1, challenge: str | None = None
) -> dict[str, Any]:
    """A fresh run at blind select (challenge runs use the Challenge Deck, as in vanilla)."""
    ch = get_challenge(challenge) if challenge else None
    if ch is not None:
        back = "b_challenge"
    return initialize_run(back, stake, seed, challenge=ch)


def win_blind(gs: dict[str, Any]) -> dict[str, Any]:
    """From SELECTING_HAND, win the current blind with one play.

    Sets the round's chips just under the target (golden ``last_hand``
    style) and plays the first five cards, which always score > 0.
    """
    assert gs["phase"] == GamePhase.SELECTING_HAND
    gs["chips"] = gs["blind"].chips - 1
    step(gs, PlayHand(card_indices=tuple(range(min(5, len(gs["hand"]))))))
    assert gs["phase"] == GamePhase.ROUND_EVAL, gs["phase"]
    return gs


def play_round(gs: dict[str, Any]) -> dict[str, Any]:
    """SelectBlind, win it, and cash out into the shop."""
    step(gs, SelectBlind())
    win_blind(gs)
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    return gs


def force_boss(gs: dict[str, Any], key: str) -> None:
    """Make ``key`` this ante's boss and put it on deck (parity-spec ``force_boss``)."""
    gs["round_resets"]["blind_choices"]["Boss"] = key
    gs["blind_on_deck"] = "Boss"
