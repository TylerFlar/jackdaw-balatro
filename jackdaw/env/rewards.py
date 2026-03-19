"""Reward shaping for Balatro RL training.

Balatro is a long-horizon game with sparse terminal rewards.  This module
provides dense reward shaping to guide learning while keeping the terminal
win/loss signal dominant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from jackdaw.env.action_space import ActionType, FactoredAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Tunable knobs for reward shaping."""

    # Terminal
    win_bonus: float = 10.0
    loss_penalty: float = -1.0

    # Progress (per-ante advancement)
    ante_advance_reward: float = 1.0

    # Scoring efficiency
    score_efficiency_weight: float = 0.01
    overkill_bonus_weight: float = 0.005

    # Economy
    economy_weight: float = 0.01
    interest_threshold: float = 5.0

    # Shaping
    hand_score_delta_weight: float = 0.001
    joker_synergy_weight: float = 0.0  # future

    # Penalties
    wasted_action_penalty: float = -0.001
    death_scaling: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOG_FLOOR = 1.0  # avoid log(0)


def _safe_log(x: float) -> float:
    """Log-scale large numbers, clamped to avoid domain errors."""
    return math.log(max(x, _LOG_FLOOR))


def _get_blind_type(state: dict[str, Any]) -> str | None:
    """Return the current blind type ('Small', 'Big', 'Boss') or None."""
    return state.get("blind_on_deck")


def _get_ante(state: dict[str, Any]) -> int:
    rr = state.get("round_resets", {})
    return rr.get("ante", state.get("ante", 1))


def _get_dollars(state: dict[str, Any]) -> int:
    return state.get("dollars", 0)


def _get_chips(state: dict[str, Any]) -> int:
    return state.get("chips", 0)


def _get_blind_chips(state: dict[str, Any]) -> int:
    blind = state.get("blind")
    if blind is not None:
        return getattr(blind, "chips", 0)
    return state.get("blind_chips", 0)


def _is_done(state: dict[str, Any]) -> bool:
    from jackdaw.engine.actions import GamePhase

    phase = state.get("phase")
    if phase is None:
        return False
    # GamePhase is a StrEnum so isinstance(phase, str) is always True;
    # compare directly against the enum member (works for both str and enum).
    return phase == GamePhase.GAME_OVER


def _has_won(state: dict[str, Any]) -> bool:
    return state.get("won", False)


def _get_round_scores(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("round_scores", {})


# ---------------------------------------------------------------------------
# RewardCalculator
# ---------------------------------------------------------------------------


class RewardCalculator:
    """Calculates step-by-step rewards for a Balatro episode.

    Tracks cumulative reward breakdown by category for diagnostics.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.reset()

    def reset(self) -> None:
        """Reset tracking for a new episode."""
        self._best_hand_score: float = 0.0
        self._prev_ante: int = 0
        self._cumulative: dict[str, float] = {
            "terminal": 0.0,
            "ante_advance": 0.0,
            "score_efficiency": 0.0,
            "economy": 0.0,
            "hand_improvement": 0.0,
            "penalties": 0.0,
        }
        self._step_count: int = 0

    def step_reward(
        self,
        prev_state: dict[str, Any],
        action: FactoredAction,
        next_state: dict[str, Any],
    ) -> float:
        """Calculate reward for a single step transition.

        Deterministic given (prev_state, action, next_state).
        """
        self._step_count += 1
        reward = 0.0

        # --- Terminal ---
        if _is_done(next_state):
            term = self._terminal_reward(next_state)
            reward += term
            self._cumulative["terminal"] += term

        # --- Ante advancement ---
        ante_r = self._ante_advance_reward(prev_state, next_state)
        if ante_r != 0.0:
            reward += ante_r
            self._cumulative["ante_advance"] += ante_r

        # --- Scoring efficiency (after playing a hand) ---
        if action.action_type == ActionType.PlayHand:
            eff = self._scoring_efficiency(next_state)
            reward += eff
            self._cumulative["score_efficiency"] += eff

        # --- Economy (at cash-out) ---
        if action.action_type == ActionType.CashOut:
            econ = self._economy_reward(next_state)
            reward += econ
            self._cumulative["economy"] += econ

        # --- Hand improvement ---
        hand_imp = self._hand_improvement(next_state)
        if hand_imp != 0.0:
            reward += hand_imp
            self._cumulative["hand_improvement"] += hand_imp

        # --- Wasted action penalty ---
        pen = self._wasted_action_penalty(action, prev_state, next_state)
        if pen != 0.0:
            reward += pen
            self._cumulative["penalties"] += pen

        return reward

    def episode_summary(self) -> dict[str, float]:
        """Return breakdown of cumulative rewards by category."""
        summary = dict(self._cumulative)
        summary["total"] = sum(self._cumulative.values())
        summary["steps"] = float(self._step_count)
        return summary

    # -- Component implementations --

    def _terminal_reward(self, next_state: dict[str, Any]) -> float:
        c = self.config
        ante = _get_ante(next_state)
        if _has_won(next_state):
            # Bonus for winning earlier (ante < 8)
            return c.win_bonus * (1.0 + 0.1 * max(8 - ante, 0))
        else:
            # Less penalty for dying later (more progress)
            return c.loss_penalty * c.death_scaling * (ante / 8.0)

    def _ante_advance_reward(
        self,
        prev_state: dict[str, Any],
        next_state: dict[str, Any],
    ) -> float:
        c = self.config
        prev_ante = _get_ante(prev_state)
        next_ante = _get_ante(next_state)

        if next_ante <= prev_ante:
            # Check if a blind was beaten within the same ante
            # by comparing blind_on_deck progression: Small -> Big -> Boss -> (next ante)
            prev_bod = _get_blind_type(prev_state)
            next_bod = _get_blind_type(next_state)
            if prev_bod == next_bod or prev_bod is None:
                return 0.0
            # Blind was beaten (blind_on_deck advanced)
            blind_type = prev_bod
        else:
            # Ante advanced — boss was beaten
            blind_type = "Boss"

        ante_multiplier = 1.0 + 0.1 * prev_ante
        if blind_type in ("Small", "Big"):
            type_mult = 0.3
        else:
            type_mult = 1.0

        return c.ante_advance_reward * ante_multiplier * type_mult

    def _scoring_efficiency(self, next_state: dict[str, Any]) -> float:
        c = self.config
        blind_chips = _get_blind_chips(next_state)
        if blind_chips <= 0:
            return 0.0

        # Use the score from last hand played
        last_result = next_state.get("last_score_result")
        if last_result is not None:
            score = getattr(last_result, "total", 0)
        else:
            # Fallback: use current chips accumulation
            score = _get_chips(next_state)

        if score <= 0:
            return 0.0

        ratio = min(score / blind_chips, 2.0)
        return c.score_efficiency_weight * ratio

    def _economy_reward(self, next_state: dict[str, Any]) -> float:
        c = self.config
        dollars = _get_dollars(next_state)
        if dollars < 0:
            # Bankrupt penalty
            return c.economy_weight * (dollars / c.interest_threshold)
        return c.economy_weight * (dollars / c.interest_threshold)

    def _hand_improvement(self, next_state: dict[str, Any]) -> float:
        c = self.config
        rs = _get_round_scores(next_state)
        current_best = rs.get("hand", 0)
        if current_best <= 0:
            return 0.0

        if current_best > self._best_hand_score and self._best_hand_score > 0:
            delta = c.hand_score_delta_weight * _safe_log(current_best / self._best_hand_score)
            self._best_hand_score = float(current_best)
            return delta

        if self._best_hand_score <= 0:
            self._best_hand_score = float(current_best)
        return 0.0

    def _wasted_action_penalty(
        self,
        action: FactoredAction,
        prev_state: dict[str, Any],
        next_state: dict[str, Any],
    ) -> float:
        c = self.config
        # Penalize rerolling when you can't afford useful cards
        if action.action_type == ActionType.Reroll:
            if _get_dollars(next_state) < 0:
                return c.wasted_action_penalty
        # Penalize sort/swap actions mildly (they use a step but don't advance game)
        if action.action_type in (
            ActionType.SortHandRank,
            ActionType.SortHandSuit,
            ActionType.SwapHandLeft,
            ActionType.SwapHandRight,
        ):
            return c.wasted_action_penalty
        return 0.0


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class DenseRewardWrapper:
    """Uses all reward components — for training."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.calculator = RewardCalculator(config)

    def reset(self) -> None:
        self.calculator.reset()

    def reward(
        self,
        prev_state: dict[str, Any],
        action: FactoredAction,
        next_state: dict[str, Any],
    ) -> float:
        return self.calculator.step_reward(prev_state, action, next_state)

    def episode_summary(self) -> dict[str, float]:
        return self.calculator.episode_summary()


class SparseRewardWrapper:
    """Only terminal win/loss — for evaluation."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._cumulative_terminal: float = 0.0
        self._step_count: int = 0

    def reset(self) -> None:
        self._cumulative_terminal = 0.0
        self._step_count = 0

    def reward(
        self,
        prev_state: dict[str, Any],
        action: FactoredAction,
        next_state: dict[str, Any],
    ) -> float:
        self._step_count += 1
        if not _is_done(next_state):
            return 0.0
        c = self.config
        ante = _get_ante(next_state)
        if _has_won(next_state):
            r = c.win_bonus * (1.0 + 0.1 * max(8 - ante, 0))
        else:
            r = c.loss_penalty * c.death_scaling * (ante / 8.0)
        self._cumulative_terminal = r
        return r

    def episode_summary(self) -> dict[str, float]:
        return {
            "terminal": self._cumulative_terminal,
            "total": self._cumulative_terminal,
            "steps": float(self._step_count),
        }
