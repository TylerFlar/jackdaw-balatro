"""Tests for curriculum learning."""

from __future__ import annotations

import pytest

from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.rewards import DenseRewardWrapper, RewardConfig

_SPEC = balatro_game_spec()
from jackdaw.env.training.curriculum import (
    CurriculumConfig,
    CurriculumManager,
    CurriculumStage,
    default_curriculum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_stage_config(
    min_eps: int = 5,
    max_eps: int = 100,
    window: int = 10,
) -> CurriculumConfig:
    """Simple 2-stage config for unit tests."""
    return CurriculumConfig(
        stages=[
            CurriculumStage(
                name="stage_0",
                target_ante=2,
                target_rate=0.5,
                window_size=window,
                reward_config=RewardConfig(ante_advance_reward=3.0, loss_penalty=-0.5),
                min_episodes=min_eps,
                max_episodes=max_eps,
            ),
            CurriculumStage(
                name="stage_1",
                target_ante=3,
                target_rate=0.5,
                window_size=window,
                reward_config=RewardConfig(),
                min_episodes=min_eps,
                max_episodes=max_eps,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# CurriculumManager unit tests
# ---------------------------------------------------------------------------


class TestCurriculumManager:
    def test_initial_state(self):
        cm = CurriculumManager(_two_stage_config())
        assert cm.stage_index == 0
        assert cm.current_stage.name == "stage_0"
        assert not cm.is_final_stage

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="at least one stage"):
            CurriculumManager(CurriculumConfig(stages=[]))

    def test_no_advance_before_min_episodes(self):
        cfg = _two_stage_config(min_eps=10, window=5)
        cm = CurriculumManager(cfg)
        # Record 5 perfect episodes (all ante >= 2) — under min_episodes
        for _ in range(5):
            cm.record_episode(ante=3, won=False)
        assert cm.stage_index == 0

    def test_advance_on_target_rate(self):
        cfg = _two_stage_config(min_eps=5, window=10)
        cm = CurriculumManager(cfg)
        # Record 5 episodes below target, then 5 at/above — fills window
        for _ in range(5):
            cm.record_episode(ante=1, won=False)
        for _ in range(5):
            cm.record_episode(ante=2, won=False)
        # 5/10 = 50% >= target_rate (0.5), and >= min_episodes (5)
        assert cm.stage_index == 1
        assert cm.current_stage.name == "stage_1"

    def test_force_advance_on_max_episodes(self):
        cfg = _two_stage_config(min_eps=5, max_eps=20, window=10)
        cm = CurriculumManager(cfg)
        # 20 episodes all at ante=1 — never meets target_rate, but hits max
        for _ in range(20):
            cm.record_episode(ante=1, won=False)
        assert cm.stage_index == 1

    def test_no_advance_past_final_stage(self):
        cfg = _two_stage_config(min_eps=2, max_eps=5, window=3)
        cm = CurriculumManager(cfg)
        # Force through both stages
        for _ in range(5):
            cm.record_episode(ante=5, won=True)
        assert cm.stage_index == 1
        assert cm.is_final_stage
        # More episodes don't crash or advance
        for _ in range(10):
            cm.record_episode(ante=5, won=True)
        assert cm.stage_index == 1

    def test_window_resets_on_advance(self):
        cfg = _two_stage_config(min_eps=3, window=5)
        cm = CurriculumManager(cfg)
        for _ in range(5):
            cm.record_episode(ante=3, won=False)
        assert cm.stage_index == 1
        # After advance, stage_episodes resets
        assert cm._stage_episodes == 0

    def test_auto_advance_disabled(self):
        cfg = _two_stage_config(min_eps=3, window=5)
        cfg.auto_advance = False
        cm = CurriculumManager(cfg)
        for _ in range(10):
            cm.record_episode(ante=5, won=True)
        # Should NOT auto-advance
        assert cm.stage_index == 0
        # But should_advance returns True
        assert cm.should_advance()
        # Manual advance works
        cm.advance()
        assert cm.stage_index == 1

    def test_reward_config_per_stage(self):
        cfg = _two_stage_config(min_eps=3, window=5)
        cm = CurriculumManager(cfg)
        assert cm.current_reward_config.ante_advance_reward == 3.0
        assert cm.current_reward_config.loss_penalty == -0.5
        # Advance
        for _ in range(5):
            cm.record_episode(ante=3, won=False)
        assert cm.current_reward_config.ante_advance_reward == 1.0  # default
        assert cm.current_reward_config.loss_penalty == -1.0  # default

    def test_get_metrics(self):
        cfg = _two_stage_config(min_eps=3, window=5)
        cm = CurriculumManager(cfg)
        cm.record_episode(ante=2, won=False)
        cm.record_episode(ante=1, won=False)
        m = cm.get_metrics()
        assert m["curriculum/stage"] == 0.0
        assert m["curriculum/stage_episodes"] == 2.0
        assert m["curriculum/target_rate"] == 0.5  # 1 of 2 >= target_ante=2

    def test_transition_history(self):
        cfg = _two_stage_config(min_eps=3, window=5)
        cm = CurriculumManager(cfg)
        for _ in range(5):
            cm.record_episode(ante=3, won=False)
        h = cm.transition_history
        assert len(h) == 1
        assert h[0]["from_stage"] == 0
        assert h[0]["stage_name"] == "stage_0"
        assert h[0]["episodes"] == 5

    def test_record_returns_true_on_advance(self):
        cfg = _two_stage_config(min_eps=3, window=5)
        cm = CurriculumManager(cfg)
        results = []
        for _ in range(5):
            results.append(cm.record_episode(ante=3, won=False))
        # Exactly one True (the episode that triggered advancement)
        assert sum(results) == 1


# ---------------------------------------------------------------------------
# DenseRewardWrapper.update_config
# ---------------------------------------------------------------------------


class TestRewardConfigUpdate:
    def test_update_config_changes_weights(self):
        wrapper = DenseRewardWrapper(RewardConfig(ante_advance_reward=1.0))
        assert wrapper.calculator.config.ante_advance_reward == 1.0
        wrapper.update_config(RewardConfig(ante_advance_reward=5.0))
        assert wrapper.calculator.config.ante_advance_reward == 5.0


# ---------------------------------------------------------------------------
# default_curriculum
# ---------------------------------------------------------------------------


class TestDefaultCurriculum:
    def test_has_four_stages(self):
        cfg = default_curriculum()
        assert len(cfg.stages) == 4

    def test_stages_have_increasing_difficulty(self):
        cfg = default_curriculum()
        antes = [s.target_ante for s in cfg.stages]
        assert antes == [2, 2, 3, 8]

    def test_first_stage_has_high_ante_reward(self):
        cfg = default_curriculum()
        assert cfg.stages[0].reward_config.ante_advance_reward == 3.0

    def test_last_stage_has_default_config(self):
        cfg = default_curriculum()
        last = cfg.stages[-1].reward_config
        default = RewardConfig()
        assert last.win_bonus == default.win_bonus
        assert last.loss_penalty == default.loss_penalty
        assert last.ante_advance_reward == default.ante_advance_reward

    def test_manager_creation(self):
        """default_curriculum() creates a valid CurriculumManager."""
        cm = CurriculumManager(default_curriculum())
        assert cm.stage_index == 0
        assert cm.current_stage.name == "survive_ante_1"


# ---------------------------------------------------------------------------
# Integration: PPOTrainer with curriculum
# ---------------------------------------------------------------------------


class TestCurriculumIntegration:
    def test_ppo_with_curriculum_smoke(self):
        """2 PPO updates with curriculum — no crashes."""
        from jackdaw.env.training.ppo import PPOConfig, PPOTrainer

        cfg = PPOConfig(
            num_envs=2,
            total_timesteps=512,
            num_steps=128,
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            log_interval=1,
            eval_interval=999999,
            save_interval=999999,
            curriculum=_two_stage_config(min_eps=1, max_eps=5, window=3),
            game_spec=_SPEC,
        )
        trainer = PPOTrainer(cfg)
        assert trainer.curriculum is not None
        result = trainer.train()
        assert result.total_updates == 2
        assert result.total_timesteps == 512

    def test_reward_config_propagates_to_envs(self):
        """Curriculum stage change updates reward weights in all envs."""
        from jackdaw.env.training.ppo import PPOConfig, PPOTrainer

        cfg = PPOConfig(
            num_envs=2,
            total_timesteps=256,
            num_steps=128,
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            eval_interval=999999,
            save_interval=999999,
            curriculum=_two_stage_config(min_eps=1, max_eps=3, window=2),
            game_spec=_SPEC,
        )
        trainer = PPOTrainer(cfg)

        # Check initial reward config is from stage 0
        for env in trainer.envs.envs:
            assert env._reward.calculator.config.ante_advance_reward == 3.0

        # Force advance
        trainer.curriculum.advance()
        trainer._on_curriculum_advance()

        # All envs should now have stage 1 config
        for env in trainer.envs.envs:
            assert env._reward.calculator.config.ante_advance_reward == 1.0
