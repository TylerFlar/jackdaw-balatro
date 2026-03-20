"""Curriculum learning for staged Balatro training.

Balatro is brutally hard — even hand-crafted heuristics average ante 2.17.
This module provides a progression of difficulty stages with tuned reward
weights so the RL agent can learn incrementally.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from jackdaw.env.rewards import RewardConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    """A single stage of curriculum learning."""

    name: str
    target_ante: int
    target_rate: float
    window_size: int
    reward_config: RewardConfig
    min_episodes: int = 500
    max_episodes: int = 50_000


@dataclass
class CurriculumConfig:
    """Multi-stage curriculum configuration."""

    stages: list[CurriculumStage]
    auto_advance: bool = True


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class CurriculumManager:
    """Tracks episode results and manages stage transitions."""

    def __init__(self, config: CurriculumConfig) -> None:
        if not config.stages:
            raise ValueError("CurriculumConfig must have at least one stage")
        self.config = config
        self._stage_idx = 0
        self._stage_episodes = 0
        self._antes: deque[int] = deque(maxlen=config.stages[0].window_size)
        self._wins: deque[bool] = deque(maxlen=config.stages[0].window_size)
        self._history: list[dict[str, Any]] = []

    # -- Properties --

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def current_stage(self) -> CurriculumStage:
        return self.config.stages[self._stage_idx]

    @property
    def current_reward_config(self) -> RewardConfig:
        return self.current_stage.reward_config

    @property
    def is_final_stage(self) -> bool:
        return self._stage_idx >= len(self.config.stages) - 1

    # -- Core logic --

    def record_episode(self, ante: int, won: bool) -> bool:
        """Record an episode result. Returns True if stage advanced."""
        self._antes.append(ante)
        self._wins.append(won)
        self._stage_episodes += 1

        if self.is_final_stage:
            return False

        if self.config.auto_advance and self.should_advance():
            self.advance()
            return True

        return False

    def should_advance(self) -> bool:
        """Check if the current stage's advancement criteria are met."""
        if self.is_final_stage:
            return False

        stage = self.current_stage

        # Force advance after max_episodes
        if self._stage_episodes >= stage.max_episodes:
            return True

        # Need min_episodes before checking target rate
        if self._stage_episodes < stage.min_episodes:
            return False

        # Need a full window to evaluate
        if len(self._antes) < stage.window_size:
            return False

        # Check if target_rate of episodes in the window reached target_ante
        hits = sum(1 for a in self._antes if a >= stage.target_ante)
        rate = hits / len(self._antes)
        return rate >= stage.target_rate

    def advance(self) -> None:
        """Move to the next stage."""
        if self.is_final_stage:
            return

        self._history.append({
            "from_stage": self._stage_idx,
            "stage_name": self.current_stage.name,
            "episodes": self._stage_episodes,
        })

        self._stage_idx += 1
        self._stage_episodes = 0
        new_stage = self.current_stage
        self._antes = deque(maxlen=new_stage.window_size)
        self._wins = deque(maxlen=new_stage.window_size)

    def get_metrics(self) -> dict[str, float]:
        """Return metrics for TensorBoard logging."""
        stage = self.current_stage
        metrics: dict[str, float] = {
            "curriculum/stage": float(self._stage_idx),
            "curriculum/stage_episodes": float(self._stage_episodes),
        }
        if self._antes:
            hits = sum(1 for a in self._antes if a >= stage.target_ante)
            metrics["curriculum/target_rate"] = hits / len(self._antes)
        else:
            metrics["curriculum/target_rate"] = 0.0
        return metrics

    @property
    def transition_history(self) -> list[dict[str, Any]]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Default curriculum
# ---------------------------------------------------------------------------


def default_curriculum() -> CurriculumConfig:
    """The standard 4-stage Balatro curriculum."""
    return CurriculumConfig(
        stages=[
            CurriculumStage(
                name="survive_ante_1",
                target_ante=2,
                target_rate=0.5,
                window_size=100,
                reward_config=RewardConfig(
                    win_bonus=10.0,
                    loss_penalty=-0.5,
                    ante_advance_reward=3.0,
                    score_efficiency_weight=0.05,
                    economy_weight=0.01,
                    hand_score_delta_weight=0.001,
                    wasted_action_penalty=-0.001,
                ),
                min_episodes=500,
                max_episodes=50_000,
            ),
            CurriculumStage(
                name="ante_1_consistent",
                target_ante=2,
                target_rate=0.8,
                window_size=100,
                reward_config=RewardConfig(
                    win_bonus=10.0,
                    loss_penalty=-0.75,
                    ante_advance_reward=2.0,
                    score_efficiency_weight=0.03,
                    economy_weight=0.02,
                    hand_score_delta_weight=0.001,
                    wasted_action_penalty=-0.001,
                ),
                min_episodes=500,
                max_episodes=50_000,
            ),
            CurriculumStage(
                name="reach_ante_3",
                target_ante=3,
                target_rate=0.5,
                window_size=100,
                reward_config=RewardConfig(
                    win_bonus=10.0,
                    loss_penalty=-1.0,
                    ante_advance_reward=1.5,
                    score_efficiency_weight=0.01,
                    economy_weight=0.01,
                    hand_score_delta_weight=0.001,
                    wasted_action_penalty=-0.001,
                ),
                min_episodes=500,
                max_episodes=50_000,
            ),
            CurriculumStage(
                name="full_game",
                target_ante=8,
                target_rate=0.01,
                window_size=200,
                reward_config=RewardConfig(),
                min_episodes=1000,
                max_episodes=500_000,
            ),
        ],
    )
