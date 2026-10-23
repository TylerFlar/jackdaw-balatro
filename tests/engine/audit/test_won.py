"""G260: end_round's win check (state_events.lua:111-114).

``if G.GAME.round_resets.ante == G.GAME.win_ante and G.GAME.blind:get_type() ==
'Boss' then G.GAME.won = true end`` runs after the jokers' end_of_round pass
and before the game-over test: the boss of exactly ``win_ante`` sets ``won``
whether it was beaten or not, and a boss of another ante never does. The env
counts a run as won only when it didn't end at that boss (``run_won``).
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import GamePhase
from jackdaw.env.game_interface import run_won
from tests.parity.pyengine import PyEngine


def _at_boss(ante: int, **values: Any) -> PyEngine:
    """The Wall (set at ante 1, so its target stays small) selected, then the ante set."""
    return PyEngine(
        {"back": "b_red", "stake": 1, "seed": "AUDIT_WON", "challenge": None},
        [
            {"op": "force_boss", "key": "bl_wall"},
            {"op": "action", "action": ["SelectBlind"]},
            {"op": "set", "ante": ante, **values},
        ],
    )


def test_losing_the_win_ante_boss_marks_the_run_won() -> None:
    engine = _at_boss(8, hands_left=1)
    engine.step(["PlayHand", [0]])  # one card can't beat The Wall
    gs = engine.gs
    assert gs["phase"] == GamePhase.GAME_OVER
    assert gs["won"] is True
    assert not run_won(gs)  # for agents it is a loss


def test_beating_the_win_ante_boss_wins() -> None:
    engine = _at_boss(8, chips=10**9)
    engine.step(["PlayHand", [0]])
    gs = engine.gs
    assert gs["phase"] == GamePhase.ROUND_EVAL
    assert gs["won"] is True
    assert run_won(gs)


def test_a_boss_past_win_ante_does_not_win() -> None:
    engine = _at_boss(9, chips=10**9)
    engine.step(["PlayHand", [0]])
    assert engine.gs["phase"] == GamePhase.ROUND_EVAL
    assert engine.gs["won"] is False


def test_a_loss_in_endless_mode_keeps_the_win() -> None:
    gs = {"won": True, "phase": GamePhase.GAME_OVER, "round_resets": {"ante": 10}, "win_ante": 8}
    assert run_won(gs)
