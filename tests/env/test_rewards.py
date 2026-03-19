"""Tests for reward shaping module."""

from __future__ import annotations

import math
import random

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import ActionType, FactoredAction
from jackdaw.env.rewards import (
    DenseRewardWrapper,
    RewardCalculator,
    RewardConfig,
    SparseRewardWrapper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    phase: GamePhase = GamePhase.SELECTING_HAND,
    ante: int = 1,
    dollars: int = 4,
    chips: int = 0,
    blind_chips: int = 300,
    blind_on_deck: str | None = "Small",
    won: bool = False,
    hand_best: int = 0,
    round_num: int = 1,
) -> dict:
    """Build a minimal game state dict for reward testing."""
    state: dict = {
        "phase": phase,
        "dollars": dollars,
        "chips": chips,
        "blind_chips": blind_chips,
        "blind_on_deck": blind_on_deck,
        "won": won,
        "round": round_num,
        "round_resets": {"ante": ante},
        "round_scores": {"hand": hand_best},
        "current_round": {"hands_left": 4, "discards_left": 3},
    }
    return state


def _action(atype: ActionType) -> FactoredAction:
    if atype in (ActionType.PlayHand, ActionType.Discard):
        return FactoredAction(action_type=int(atype), card_target=(0, 1), entity_target=None)
    return FactoredAction(action_type=int(atype), card_target=None, entity_target=None)


# ---------------------------------------------------------------------------
# Terminal rewards
# ---------------------------------------------------------------------------


class TestTerminalRewards:
    def test_win_reward_positive(self):
        calc = RewardCalculator()
        prev = _make_state()
        nxt = _make_state(phase=GamePhase.GAME_OVER, won=True, ante=8)
        r = calc.step_reward(prev, _action(ActionType.PlayHand), nxt)
        assert r > 0.0

    def test_loss_reward_negative(self):
        calc = RewardCalculator()
        prev = _make_state(ante=4)
        nxt = _make_state(phase=GamePhase.GAME_OVER, won=False, ante=4)
        r = calc.step_reward(prev, _action(ActionType.PlayHand), nxt)
        assert r < 0.0

    def test_early_win_more_rewarding(self):
        calc1 = RewardCalculator()
        calc2 = RewardCalculator()
        # Use same ante in prev and next to isolate terminal reward
        early = _make_state(phase=GamePhase.GAME_OVER, won=True, ante=3)
        late = _make_state(phase=GamePhase.GAME_OVER, won=True, ante=8)
        r_early = calc1.step_reward(_make_state(ante=3), _action(ActionType.PlayHand), early)
        r_late = calc2.step_reward(_make_state(ante=8), _action(ActionType.PlayHand), late)
        assert r_early > r_late

    def test_late_loss_less_penalty(self):
        calc1 = RewardCalculator()
        calc2 = RewardCalculator()
        # Match ante in prev/next to isolate terminal reward
        early_loss = _make_state(phase=GamePhase.GAME_OVER, won=False, ante=2)
        late_loss = _make_state(phase=GamePhase.GAME_OVER, won=False, ante=7)
        r_early = calc1.step_reward(_make_state(ante=2), _action(ActionType.PlayHand), early_loss)
        r_late = calc2.step_reward(_make_state(ante=7), _action(ActionType.PlayHand), late_loss)
        # Early loss has smaller magnitude penalty (ante/8 scaling)
        assert r_early > r_late  # less negative = greater

    def test_win_dominates_shaping(self):
        """Win reward must be much larger than any per-step shaping component."""
        config = RewardConfig()
        assert config.win_bonus > 10 * config.score_efficiency_weight
        assert config.win_bonus > 10 * config.economy_weight
        # Win bonus should be significantly larger than a single ante advance
        assert config.win_bonus >= 10 * config.ante_advance_reward


# ---------------------------------------------------------------------------
# Ante advancement
# ---------------------------------------------------------------------------


class TestAnteAdvancement:
    def test_ante_advance_positive(self):
        calc = RewardCalculator()
        prev = _make_state(ante=1, blind_on_deck="Boss")
        nxt = _make_state(ante=2, blind_on_deck="Small")
        r = calc.step_reward(prev, _action(ActionType.NextRound), nxt)
        assert r > 0.0

    def test_blind_beaten_within_ante(self):
        calc = RewardCalculator()
        prev = _make_state(ante=1, blind_on_deck="Small")
        nxt = _make_state(ante=1, blind_on_deck="Big")
        r = calc.step_reward(prev, _action(ActionType.NextRound), nxt)
        assert r > 0.0

    def test_boss_more_than_small(self):
        calc1 = RewardCalculator()
        calc2 = RewardCalculator()
        # Small beaten
        prev_s = _make_state(ante=1, blind_on_deck="Small")
        nxt_s = _make_state(ante=1, blind_on_deck="Big")
        r_small = calc1.step_reward(prev_s, _action(ActionType.NextRound), nxt_s)
        # Boss beaten (ante advances)
        prev_b = _make_state(ante=1, blind_on_deck="Boss")
        nxt_b = _make_state(ante=2, blind_on_deck="Small")
        r_boss = calc2.step_reward(prev_b, _action(ActionType.NextRound), nxt_b)
        assert r_boss > r_small


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------


class TestEconomy:
    def test_economy_scales_with_dollars(self):
        calc = RewardCalculator()
        poor = _make_state(dollars=2)
        rich = _make_state(dollars=20)
        r_poor = calc.step_reward(
            _make_state(), _action(ActionType.CashOut), poor
        )
        calc2 = RewardCalculator()
        r_rich = calc2.step_reward(
            _make_state(), _action(ActionType.CashOut), rich
        )
        assert r_rich > r_poor

    def test_bankrupt_negative_economy(self):
        calc = RewardCalculator()
        bankrupt = _make_state(dollars=-3)
        calc.step_reward(
            _make_state(), _action(ActionType.CashOut), bankrupt
        )
        # Economy component should be negative
        assert calc._cumulative["economy"] < 0.0


# ---------------------------------------------------------------------------
# Dense rewards bounded
# ---------------------------------------------------------------------------


class TestDenseBounded:
    def test_dense_rewards_bounded(self):
        """Dense rewards should stay bounded and not explode."""
        wrapper = DenseRewardWrapper()
        wrapper.reset()
        rewards = []
        state = _make_state()
        for _ in range(100):
            action = _action(ActionType.PlayHand)
            next_s = _make_state(
                chips=random.randint(0, 100000),
                blind_chips=random.randint(100, 10000),
                dollars=random.randint(0, 50),
                hand_best=random.randint(0, 50000),
            )
            r = wrapper.reward(state, action, next_s)
            rewards.append(r)
            state = next_s

        for r in rewards:
            assert math.isfinite(r), f"Non-finite reward: {r}"
            assert not math.isnan(r), f"NaN reward: {r}"
            # Individual step shaping should be small
            assert abs(r) < 5.0, f"Reward too large: {r}"


# ---------------------------------------------------------------------------
# Sparse rewards
# ---------------------------------------------------------------------------


class TestSparseRewards:
    def test_sparse_zero_before_terminal(self):
        wrapper = SparseRewardWrapper()
        wrapper.reset()
        state = _make_state()
        for _ in range(50):
            r = wrapper.reward(state, _action(ActionType.PlayHand), state)
            assert r == 0.0

    def test_sparse_nonzero_at_terminal(self):
        wrapper = SparseRewardWrapper()
        wrapper.reset()
        prev = _make_state()
        terminal = _make_state(phase=GamePhase.GAME_OVER, won=True, ante=5)
        r = wrapper.reward(prev, _action(ActionType.PlayHand), terminal)
        assert r != 0.0

    def test_sparse_loss_at_terminal(self):
        wrapper = SparseRewardWrapper()
        wrapper.reset()
        prev = _make_state()
        terminal = _make_state(phase=GamePhase.GAME_OVER, won=False, ante=3)
        r = wrapper.reward(prev, _action(ActionType.PlayHand), terminal)
        assert r < 0.0


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------


class TestEpisodeSummary:
    def test_summary_keys(self):
        calc = RewardCalculator()
        calc.reset()
        summary = calc.episode_summary()
        expected = {
            "terminal", "ante_advance", "score_efficiency",
            "economy", "hand_improvement", "penalties",
            "total", "steps",
        }
        assert set(summary.keys()) == expected

    def test_summary_accumulates(self):
        calc = RewardCalculator()
        calc.reset()
        prev = _make_state(ante=1, blind_on_deck="Small")
        nxt = _make_state(ante=1, blind_on_deck="Big")
        calc.step_reward(prev, _action(ActionType.NextRound), nxt)
        summary = calc.episode_summary()
        assert summary["steps"] == 1.0
        assert summary["ante_advance"] > 0.0
        assert summary["total"] > 0.0


# ---------------------------------------------------------------------------
# No NaN/Inf across random steps
# ---------------------------------------------------------------------------


class TestNoNanInf:
    def test_100_random_steps_no_nan_inf(self):
        calc = RewardCalculator()
        calc.reset()
        action_types = [
            ActionType.PlayHand,
            ActionType.Discard,
            ActionType.Reroll,
            ActionType.CashOut,
            ActionType.SelectBlind,
            ActionType.NextRound,
            ActionType.SortHandRank,
            ActionType.SortHandSuit,
        ]
        state = _make_state()
        for _ in range(100):
            at = random.choice(action_types)
            action = _action(at)
            next_s = _make_state(
                ante=random.randint(1, 8),
                dollars=random.randint(-5, 100),
                chips=random.randint(0, 10**9),
                blind_chips=random.randint(1, 10**7),
                hand_best=random.randint(0, 10**9),
                blind_on_deck=random.choice(["Small", "Big", "Boss", None]),
            )
            r = calc.step_reward(state, action, next_s)
            assert math.isfinite(r), f"Non-finite reward at step: {r}"
            assert not math.isnan(r), f"NaN reward at step: {r}"
            state = next_s

        summary = calc.episode_summary()
        for k, v in summary.items():
            assert math.isfinite(v), f"Non-finite in summary[{k}]: {v}"


# ---------------------------------------------------------------------------
# Scoring efficiency
# ---------------------------------------------------------------------------


class TestScoringEfficiency:
    def test_capped_at_2x(self):
        """Score efficiency ratio should be capped at 2.0."""
        calc = RewardCalculator()
        config = calc.config
        prev = _make_state()
        # Massive overkill: 1M chips vs 300 target
        nxt = _make_state(chips=1_000_000, blind_chips=300)
        r = calc.step_reward(prev, _action(ActionType.PlayHand), nxt)
        # Max possible from efficiency: weight * 2.0
        max_eff = config.score_efficiency_weight * 2.0
        # Total includes other components, but efficiency alone should be capped
        assert math.isfinite(r), f"Non-finite reward: {r}"
        assert calc._cumulative["score_efficiency"] <= max_eff + 1e-9
