"""Tests for jackdaw.env.rewards — Transition events and the example rewards."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import ActionType
from jackdaw.env.game_interface import GameState
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.rewards import (
    ShapedReward,
    Transition,
    blind_progress_reward,
    sparse_reward,
)


def _state(**overrides: Any) -> GameState:
    fields: dict[str, Any] = {
        "phase": GamePhase.SELECTING_HAND,
        "ante": 1,
        "round": 1,
        "dollars": 4,
        "hands_left": 4,
        "discards_left": 3,
        "hand_size": 8,
        "joker_slots": 5,
        "consumable_slots": 2,
        "blind_on_deck": "Small",
        "blind_chips": 300,
        "chips": 0,
        "won": False,
        "done": False,
    }
    fields.update(overrides)
    return GameState(**fields)


def _t(
    prev: GameState,
    curr: GameState,
    *,
    action: ActionType = ActionType.PlayHand,
    terminated: bool = False,
    truncated: bool = False,
) -> Transition:
    return Transition(
        prev=prev,
        curr=curr,
        action=FactoredAction(action_type=action),
        terminated=terminated,
        truncated=truncated,
    )


def _select_blind() -> Transition:
    """Selecting the Small blind: round increments, chips reset to 0."""
    return _t(
        _state(phase=GamePhase.BLIND_SELECT, round=0, chips=500),
        _state(round=1),
        action=ActionType.SelectBlind,
    )


def _clear(*, ante: int = 1, boss: bool = False, hands_left: int = 3) -> Transition:
    """The hand that clears a 300-chip blind from 100 chips."""
    return _t(
        _state(ante=ante, blind_on_deck="Boss" if boss else "Small", chips=100),
        _state(
            phase=GamePhase.ROUND_EVAL,
            ante=ante + 1 if boss else ante,
            blind_on_deck="Small" if boss else "Big",
            chips=400,
            hands_left=hands_left,
        ),
    )


def _game_over(*, won: bool = False) -> Transition:
    return _t(
        _state(won=won),
        _state(phase=GamePhase.GAME_OVER, chips=50, hands_left=0, won=won, done=True),
        terminated=True,
    )


# ---------------------------------------------------------------------------
# Transition events
# ---------------------------------------------------------------------------


class TestTransitionEvents:
    def test_selecting_a_blind_is_not_beating_it(self):
        # round increments at selection — the old reward's "blind beaten"
        t = _select_blind()
        assert not t.blind_beaten
        assert t.chips_scored == 0  # the reset to 0 isn't scoring

    def test_clearing_hand_beats_the_blind(self):
        t = _clear()
        assert t.blind_beaten
        assert not t.boss_beaten
        assert t.chips_scored == 300

    def test_boss_beaten_reports_the_boss_ante(self):
        t = _clear(ante=3, boss=True)
        assert t.boss_beaten
        assert t.curr.ante == 4
        assert t.blind_ante == 3

    def test_hand_that_falls_short(self):
        t = _t(_state(chips=0), _state(chips=120, hands_left=3))
        assert not t.blind_beaten
        assert t.chips_scored == 120

    def test_run_won_fires_once(self):
        assert _t(_state(), _state(phase=GamePhase.ROUND_EVAL, won=True)).run_won
        assert not _t(_state(won=True), _state(won=True)).run_won

    def test_overflowed_scores_never_read_as_nan(self):
        t = _t(_state(chips=math.inf), _state(chips=math.inf))
        assert t.chips_scored == 0

    def test_from_step(self):
        prev, curr = _state(), _state(chips=10)
        action = FactoredAction(action_type=ActionType.PlayHand, card_target=(0,))
        info = {"prev_state": prev, "state": curr, "raw_state": {"chips": 10}}
        t = Transition.from_step(action, False, True, info)
        assert (t.prev, t.curr, t.action) == (prev, curr, action)
        assert t.truncated and t.done
        assert t.raw_state == {"chips": 10}


# ---------------------------------------------------------------------------
# Example reward functions
# ---------------------------------------------------------------------------


class TestSparseReward:
    def test_zero_mid_episode(self):
        assert sparse_reward(_clear()) == 0.0

    def test_loss(self):
        assert sparse_reward(_game_over()) == -1.0

    def test_won_run_that_dies_in_endless_mode(self):
        assert sparse_reward(_game_over(won=True)) == 1.0

    def test_truncation_without_a_win(self):
        assert sparse_reward(_t(_state(), _state(), truncated=True)) == -1.0


class TestBlindProgressReward:
    def test_selecting_scores_nothing(self):
        assert blind_progress_reward(_select_blind()) == 0.0

    def test_each_blind_beaten(self):
        assert blind_progress_reward(_clear()) == 1.0

    def test_loss(self):
        assert blind_progress_reward(_game_over()) == -1.0


class TestShapedReward:
    reward = ShapedReward()

    def test_selecting_a_blind_costs_a_step(self):
        assert self.reward(_select_blind()) == pytest.approx(-0.001)

    def test_shop_actions_cost_double(self):
        t = _t(
            _state(phase=GamePhase.SHOP),
            _state(phase=GamePhase.SHOP),
            action=ActionType.Reroll,
        )
        assert self.reward(t) == pytest.approx(-0.002)

    def test_clearing_a_blind(self):
        # step + blind x 1/8 + 3 hands left + progress (300 of 300)
        expected = -0.001 + 0.15 / 8 + 0.01 * 3 + 0.02
        assert self.reward(_clear()) == pytest.approx(expected)

    def test_boss_bonus_scales_with_the_boss_ante(self):
        expected = -0.001 + (0.15 + 0.10) * 3 / 8 + 0.01 * 3 + 0.02
        assert self.reward(_clear(ante=3, boss=True)) == pytest.approx(expected)

    def test_partial_progress(self):
        t = _t(_state(chips=0), _state(chips=150, hands_left=3))
        assert self.reward(t) == pytest.approx(-0.001 + 0.02 * 0.5)

    def test_win_paid_on_the_winning_step(self):
        t = _clear(ante=8, boss=True)
        won = _t(t.prev, replace(t.curr, won=True))
        assert self.reward(won) == pytest.approx(self.reward(t) + 0.5)

    def test_loss_penalty_only_without_a_win(self):
        assert self.reward(_game_over()) == pytest.approx(-0.001 + 0.02 * 50 / 300 - 0.2)
        assert self.reward(_game_over(won=True)) == pytest.approx(-0.001 + 0.02 * 50 / 300)

    def test_overflowed_target_stays_finite(self):
        t = _t(_state(blind_chips=math.inf), _state(blind_chips=math.inf, chips=10**300))
        assert self.reward(t) == pytest.approx(-0.001)

    def test_weights_are_configurable(self):
        quiet = ShapedReward(step_cost=0.0, blind_beaten=1.0, hands_left=0.0, score_progress=0.0)
        assert quiet(_clear(ante=8)) == pytest.approx(1.0)
