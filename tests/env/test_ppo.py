"""Tests for CleanRL-style PPO training loop.

Tests the mechanics only — NOT convergence (too slow).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from jackdaw.env.action_space import FactoredAction, get_action_mask
from jackdaw.env.balatro_env import _action_mask_to_game, _compute_shop_splits
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.game_spec import GameActionMask, GameObservation
from jackdaw.env.observation import encode_observation
from jackdaw.env.training.ppo import (
    EvalMetrics,
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    StepData,
    SyncVectorEnv,
    TrainResult,
    _make_env,
    _resolve_device,
    train_ppo,
)

# ---------------------------------------------------------------------------
# Minimal config for fast tests
# ---------------------------------------------------------------------------

_SPEC = balatro_game_spec()

FAST_CONFIG = PPOConfig(
    num_envs=2,
    num_steps=4,
    update_epochs=1,
    num_minibatches=2,
    total_timesteps=16,  # 2 envs * 4 steps * 2 updates
    eval_episodes=2,
    log_interval=1,
    eval_interval=100,  # skip eval during training
    save_interval=100,  # skip save during training
    embed_dim=32,
    num_heads=2,
    num_layers=1,
    device="cpu",
    game_spec=_SPEC,
)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def test_resolve_device_cpu():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto():
    device = _resolve_device("auto")
    assert isinstance(device, torch.device)


# ---------------------------------------------------------------------------
# SyncVectorEnv
# ---------------------------------------------------------------------------


class TestSyncVectorEnv:
    def test_reset_returns_correct_count(self):
        env_fns = [_make_env(["b_red"], [1], 100, i) for i in range(3)]
        vec_env = SyncVectorEnv(env_fns)
        results = vec_env.reset()
        assert len(results) == 3
        for obs, mask, info in results:
            assert isinstance(obs, GameObservation)
            assert isinstance(mask, GameActionMask)
            assert isinstance(info, dict)

    def test_step_returns_correct_shapes(self):
        env_fns = [_make_env(["b_red"], [1], 100, i) for i in range(2)]
        vec_env = SyncVectorEnv(env_fns)
        reset_data = vec_env.reset()

        # Get legal actions for each env
        actions = []
        for obs, mask, info in reset_data:
            legal_types = np.nonzero(mask.type_mask)[0]
            at = int(legal_types[0])
            actions.append(FactoredAction(action_type=at))

        obs_list, rewards, terms, truncs, masks, infos = vec_env.step(actions)
        assert len(obs_list) == 2
        assert rewards.shape == (2,)
        assert terms.shape == (2,)
        assert truncs.shape == (2,)
        assert len(masks) == 2
        assert len(infos) == 2


# ---------------------------------------------------------------------------
# _EnvInstance
# ---------------------------------------------------------------------------


class TestEnvInstance:
    def test_reset_and_step(self):
        env = _make_env(["b_red"], [1], 100, 0)()
        obs, mask, info = env.reset()
        assert isinstance(obs, GameObservation)
        assert obs.global_context.shape[0] > 0

        legal_types = np.nonzero(mask.type_mask)[0]
        action = FactoredAction(action_type=int(legal_types[0]))
        obs2, reward, term, trunc, mask2, info2 = env.step(action)
        assert isinstance(obs2, GameObservation)
        assert isinstance(reward, float)

    def test_episode_tracking(self):
        env = _make_env(["b_red"], [1], 100, 0)()
        env.reset()
        assert env.episode_return == 0.0
        assert env.episode_length == 0


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


class TestRolloutBuffer:
    def _make_dummy_step(self, num_envs: int) -> StepData:
        """Create dummy step data for testing."""
        adapter = DirectAdapter()
        adapter.reset("b_red", 1, "TEST_BUFFER")
        gs = adapter.raw_state
        obs = encode_observation(gs)
        mask = get_action_mask(gs)
        game_obs = obs.to_game_observation()
        game_mask = _action_mask_to_game(mask)

        obs_list = [game_obs] * num_envs
        masks = [game_mask] * num_envs
        legal_types = np.nonzero(mask.type_mask)[0]
        actions = [FactoredAction(action_type=int(legal_types[0]))] * num_envs

        shop_splits = [_compute_shop_splits(gs)] * num_envs

        return StepData(
            obs=obs_list,
            masks=masks,
            shop_splits=shop_splits,
            actions=actions,
            log_probs=torch.randn(num_envs),
            values=torch.randn(num_envs),
            rewards=np.random.randn(num_envs).astype(np.float32),
            dones=np.zeros(num_envs, dtype=np.bool_),
            infos=[{}] * num_envs,
        )

    def test_add_and_shape(self):
        num_steps, num_envs = 4, 2
        buffer = RolloutBuffer(num_steps, num_envs)
        for _ in range(num_steps):
            buffer.add(self._make_dummy_step(num_envs))
        assert len(buffer.steps) == num_steps

    def test_compute_returns(self):
        num_steps, num_envs = 4, 2
        buffer = RolloutBuffer(num_steps, num_envs)
        for _ in range(num_steps):
            buffer.add(self._make_dummy_step(num_envs))

        last_value = torch.randn(num_envs)
        last_done = np.zeros(num_envs, dtype=np.bool_)
        buffer.compute_returns(last_value, last_done, gamma=0.99, gae_lambda=0.95)

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (num_steps, num_envs)
        assert buffer.returns.shape == (num_steps, num_envs)

    def test_get_batches(self):
        num_steps, num_envs = 4, 2
        num_minibatches = 2
        buffer = RolloutBuffer(num_steps, num_envs)
        for _ in range(num_steps):
            buffer.add(self._make_dummy_step(num_envs))

        last_value = torch.randn(num_envs)
        last_done = np.zeros(num_envs, dtype=np.bool_)
        buffer.compute_returns(last_value, last_done, gamma=0.99, gae_lambda=0.95)

        batches = list(buffer.get_batches(num_minibatches, torch.device("cpu"), game_spec=_SPEC))
        assert len(batches) == num_minibatches

        total_samples = sum(len(mb.actions) for mb in batches)
        assert total_samples == num_steps * num_envs

        for mb in batches:
            assert isinstance(mb.batch, dict)
            assert "global_context" in mb.batch
            assert mb.old_log_probs.shape[0] == len(mb.actions)
            assert mb.advantages.shape[0] == len(mb.actions)
            assert mb.returns.shape[0] == len(mb.actions)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


class TestPPOTrainer:
    def test_init(self):
        """PPOTrainer initializes without error."""
        trainer = PPOTrainer(FAST_CONFIG)
        assert trainer.policy is not None
        assert trainer.optimizer is not None
        assert trainer.envs.num_envs == FAST_CONFIG.num_envs
        assert trainer.global_step == 0

    def test_collect_rollouts(self):
        """collect_rollouts produces correct buffer shapes."""
        trainer = PPOTrainer(FAST_CONFIG)

        # Initial reset
        reset_data = trainer.envs.reset()
        trainer._current_obs = [d[0] for d in reset_data]
        trainer._current_masks = [d[1] for d in reset_data]
        trainer._current_infos = [d[2] for d in reset_data]

        buffer = trainer._collect_rollouts()
        assert len(buffer.steps) == FAST_CONFIG.num_steps
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (
            FAST_CONFIG.num_steps,
            FAST_CONFIG.num_envs,
        )

    def test_single_update(self):
        """One full update step runs without error."""
        trainer = PPOTrainer(FAST_CONFIG)

        # Initial reset
        reset_data = trainer.envs.reset()
        trainer._current_obs = [d[0] for d in reset_data]
        trainer._current_masks = [d[1] for d in reset_data]
        trainer._current_infos = [d[2] for d in reset_data]

        buffer = trainer._collect_rollouts()
        metrics = trainer._update(buffer)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "clip_fraction" in metrics
        assert "explained_variance" in metrics
        assert "learning_rate" in metrics

        # Sanity: losses should be finite
        for key in ["policy_loss", "value_loss", "entropy", "total_loss"]:
            assert np.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"

    def test_full_training_loop(self):
        """Full training loop completes (tiny config)."""
        result = train_ppo(FAST_CONFIG)
        assert isinstance(result, TrainResult)
        assert result.total_timesteps > 0
        assert result.total_updates > 0
        assert result.wall_time > 0

    def test_checkpoint_save_load(self):
        """Checkpoint save/load preserves policy weights."""
        trainer = PPOTrainer(FAST_CONFIG)

        # Get initial weights
        initial_params = {
            k: v.clone() for k, v in trainer.policy.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_checkpoint.pt")
            trainer.save_checkpoint(path)

            # Create a fresh trainer and load
            trainer2 = PPOTrainer(FAST_CONFIG)
            trainer2.load_checkpoint(path)

            # Verify weights match
            for key in initial_params:
                assert torch.allclose(
                    initial_params[key],
                    trainer2.policy.state_dict()[key],
                ), f"Mismatch in {key}"

    def test_checkpoint_preserves_training_state(self):
        """Checkpoint preserves global_step, update_count, episode_count."""
        trainer = PPOTrainer(FAST_CONFIG)
        trainer.global_step = 42
        trainer.update_count = 7
        trainer.episode_count = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_state.pt")
            trainer.save_checkpoint(path)

            trainer2 = PPOTrainer(FAST_CONFIG)
            trainer2.load_checkpoint(path)

            assert trainer2.global_step == 42
            assert trainer2.update_count == 7
            assert trainer2.episode_count == 3

    def test_evaluate(self):
        """Evaluate returns sensible metrics."""
        trainer = PPOTrainer(FAST_CONFIG)
        eval_metrics = trainer._evaluate()

        assert isinstance(eval_metrics, EvalMetrics)
        assert eval_metrics.n_episodes == FAST_CONFIG.eval_episodes
        assert 0.0 <= eval_metrics.win_rate <= 1.0
        assert eval_metrics.avg_ante >= 1.0
        assert eval_metrics.avg_length >= 0


# ---------------------------------------------------------------------------
# PPOConfig
# ---------------------------------------------------------------------------


class TestPPOConfig:
    def test_defaults(self):
        cfg = PPOConfig(game_spec=_SPEC)
        assert cfg.num_envs == 8
        assert cfg.total_timesteps == 1_000_000
        assert cfg.clip_coef == 0.2
        assert cfg.device == "auto"

    def test_custom(self):
        cfg = PPOConfig(num_envs=4, learning_rate=1e-3, device="cpu", game_spec=_SPEC)
        assert cfg.num_envs == 4
        assert cfg.learning_rate == 1e-3
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# Full pipeline smoke test — 10K timesteps
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_ppo_10k_smoke():
    """Full PPO training pipeline runs without crashing for 10K steps.

    Exercises the ENTIRE pipeline end-to-end: env reset, observation
    encoding, action masking, policy forward pass, action sampling,
    engine step, reward computation, GAE, PPO update, gradient clipping.
    """
    config = PPOConfig(
        num_envs=2,
        total_timesteps=10_000,
        num_steps=64,
        num_minibatches=2,
        update_epochs=2,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        learning_rate=3e-4,
        log_interval=1,
        eval_interval=999_999,   # effectively skip eval
        save_interval=999_999,   # effectively skip checkpointing
        device="cpu",
        back_keys="b_red",
        stake=1,
        game_spec=_SPEC,
    )

    result = train_ppo(config)

    # Basic sanity checks — actual steps may be slightly under total_timesteps
    # due to integer division of timesteps by (num_envs * num_steps)
    assert result.total_timesteps >= 9_000
    assert result.total_updates > 0
    assert result.total_episodes > 0
    assert result.wall_time > 0
    assert len(result.log_history) > 0

    # Verify no NaN/Inf in logged metrics
    for entry in result.log_history:
        for key, value in entry.items():
            if isinstance(value, float):
                assert not math.isnan(value), (
                    f"NaN in metric {key} at update {entry.get('update')}"
                )
                assert not math.isinf(value), (
                    f"Inf in metric {key} at update {entry.get('update')}"
                )

    # Verify loss values are reasonable (not exploding)
    last_entry = result.log_history[-1]
    assert abs(last_entry["policy_loss"]) < 100, "Policy loss exploded"
    assert abs(last_entry["value_loss"]) < 1000, "Value loss exploded"
