"""CleanRL-style PPO training loop for Balatro.

Custom loop required because the policy architecture (variable-length entities,
autoregressive action heads, pointer networks) doesn't fit standard RL library
interfaces like SB3.

Reference: https://docs.cleanrl.dev/rl-algorithms/ppo/
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.game_interface import DirectAdapter, GameAdapter
from jackdaw.env.game_spec import (
    FactoredAction,
    GameActionMask,
    GameObservation,
    GameSpec,
)
from jackdaw.env.policy.policy import (
    BalatroPolicy,
    PolicyInput,
    collate_policy_inputs,
)
from jackdaw.env.rewards import DenseRewardWrapper, RewardConfig
from jackdaw.env.training.curriculum import CurriculumConfig, CurriculumManager

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment, misc]
    _HAS_TENSORBOARD = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Environment
    num_envs: int = 8
    back_keys: list[str] | str = "b_red"  # single key or list to sample from
    stake: list[int] | int = 1  # single stake or list to sample from
    max_steps_per_episode: int = 10_000

    # PPO hyperparameters
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_steps: int = 128  # steps per rollout per env
    num_minibatches: int = 4
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Policy architecture
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 20
    save_interval: int = 100
    log_dir: str = "runs"
    run_name: str | None = None

    # Curriculum
    curriculum: CurriculumConfig | None = None

    # Game spec (required — drives policy architecture)
    game_spec: GameSpec | None = None  # None only for backward-compat; will error at policy init

    # Device
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


ALL_DECKS = [
    "b_red",
    "b_blue",
    "b_yellow",
    "b_green",
    "b_black",
    "b_magic",
    "b_nebula",
    "b_ghost",
    "b_abandoned",
    "b_checkered",
    "b_zodiac",
    "b_painted",
    "b_anaglyph",
    "b_plasma",
    "b_erratic",
]
"""All 15 standard Balatro deck back keys."""


def _normalize_back_keys(back_keys: list[str] | str) -> list[str]:
    """Normalize back_keys config to a list."""
    if isinstance(back_keys, str):
        if back_keys == "all":
            return list(ALL_DECKS)
        return [back_keys]
    return list(back_keys)


def _normalize_stakes(stake: list[int] | int) -> list[int]:
    """Normalize stake config to a list."""
    if isinstance(stake, int):
        return [stake]
    return list(stake)


# ---------------------------------------------------------------------------
# Synchronous vectorized environment
# ---------------------------------------------------------------------------


class SyncVectorEnv:
    """Synchronous vectorized environment for Balatro.

    Standard vectorized envs (gymnasium.vector) assume fixed-shape
    observations.  This wrapper handles variable-length entity lists
    by returning lists of dicts/dataclasses instead of stacked arrays.
    """

    def __init__(self, env_fns: list[Callable[[], _EnvInstance]]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self) -> list[tuple[GameObservation, GameActionMask, dict[str, Any]]]:
        """Reset all environments, return (obs, mask, info) per env."""
        results = []
        for env in self.envs:
            obs, mask, info = env.reset()
            results.append((obs, mask, info))
        return results

    def step(
        self, actions: list[FactoredAction]
    ) -> tuple[
        list[GameObservation],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[GameActionMask],
        list[dict[str, Any]],
    ]:
        """Step all environments.

        Returns (obs_list, rewards, terminateds, truncateds, masks, infos).
        Auto-resets environments that are done.
        """
        obs_list: list[GameObservation] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=np.bool_)
        truncateds = np.zeros(self.num_envs, dtype=np.bool_)
        masks: list[GameActionMask] = []
        infos: list[dict[str, Any]] = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, mask, info = env.step(action)
            rewards[i] = reward
            terminateds[i] = terminated
            truncateds[i] = truncated

            if terminated or truncated:
                # Store terminal info before auto-reset
                info["terminal_observation"] = obs
                info["terminal_mask"] = mask
                info["episode_return"] = env.episode_return
                info["episode_length"] = env.episode_length
                info["episode_won"] = env.episode_won
                info["episode_ante"] = env.episode_ante
                obs, mask, reset_info = env.reset()
                info.update({f"reset_{k}": v for k, v in reset_info.items()})

            obs_list.append(obs)
            masks.append(mask)
            infos.append(info)

        return obs_list, rewards, terminateds, truncateds, masks, infos


class _EnvInstance:
    """Single environment instance delegating to a :class:`BalatroEnvironment`.

    Adapts the game-agnostic ``GameEnvironment`` interface back to the
    Balatro-typed ``Observation``/``ActionMask`` objects that the rest of the
    training loop expects.  Episode tracking attributes are forwarded from
    the underlying environment.
    """

    def __init__(self, env: BalatroEnvironment) -> None:
        self._env = env

    # Expose reward wrapper for curriculum updates
    @property
    def _reward(self) -> DenseRewardWrapper:
        return self._env.reward_wrapper

    @property
    def episode_return(self) -> float:
        return self._env.episode_return

    @property
    def episode_length(self) -> int:
        return self._env.episode_length

    @property
    def episode_won(self) -> bool:
        return self._env.episode_won

    @property
    def episode_ante(self) -> int:
        return self._env.episode_ante

    def reset(self) -> tuple[GameObservation, GameActionMask, dict[str, Any]]:
        game_obs, game_mask, info = self._env.reset()
        return game_obs, game_mask, info  # type: ignore[return-value]

    def step(
        self, action: FactoredAction,
    ) -> tuple[GameObservation, float, bool, bool, GameActionMask, dict[str, Any]]:
        game_obs, reward, terminated, truncated, game_mask, info = self._env.step(action)
        return game_obs, reward, terminated, truncated, game_mask, info  # type: ignore[return-value]


def _make_env(
    back_keys: list[str],
    stakes: list[int],
    max_steps: int,
    env_idx: int,
    adapter_factory: Callable[[], GameAdapter] | None = None,
    reward_config: RewardConfig | None = None,
) -> Callable[[], _EnvInstance]:
    """Create a factory function for a single environment instance."""

    def _factory() -> _EnvInstance:
        factory = adapter_factory or DirectAdapter
        env = BalatroEnvironment(
            adapter_factory=factory,
            reward_config=reward_config,
            back_keys=back_keys,
            stakes=stakes,
            max_steps=max_steps,
            seed_prefix=f"TRAIN_E{env_idx}",
        )
        return _EnvInstance(env)

    return _factory


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


@dataclass
class StepData:
    """Data for a single step across all environments."""

    obs: list[GameObservation]  # per-env observations
    masks: list[GameActionMask]  # per-env action masks
    shop_splits: list[tuple[int, int, int]]
    actions: list[FactoredAction]
    log_probs: torch.Tensor  # (num_envs,) total log prob
    values: torch.Tensor  # (num_envs,)
    rewards: np.ndarray  # (num_envs,)
    dones: np.ndarray  # (num_envs,) bool
    infos: list[dict[str, Any]]


@dataclass
class MiniBatch:
    """A minibatch of experience for PPO update."""

    batch: dict[str, Any]  # collated policy inputs
    actions: list[FactoredAction]
    old_log_probs: torch.Tensor  # (mb_size,)
    advantages: torch.Tensor  # (mb_size,)
    returns: torch.Tensor  # (mb_size,)
    old_values: torch.Tensor  # (mb_size,)


class RolloutBuffer:
    """Stores rollout data for PPO. Handles variable-length observations via padding.

    Stores raw Observation/ActionMask objects and only collates into padded
    tensors when generating minibatches, so per-step storage is lightweight.
    """

    def __init__(self, num_steps: int, num_envs: int) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.steps: list[StepData] = []

        # Computed after rollout
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(self, step_data: StepData) -> None:
        """Append one timestep of data across all environments."""
        self.steps.append(step_data)

    def compute_returns(
        self,
        last_value: torch.Tensor,
        last_done: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and discounted returns."""
        device = last_value.device
        advantages = torch.zeros(self.num_steps, self.num_envs, device=device)
        last_gae = torch.zeros(self.num_envs, device=device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - torch.from_numpy(last_done).float().to(device)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - torch.from_numpy(self.steps[t + 1].dones).float().to(
                    device
                )
                next_values = self.steps[t + 1].values.to(device)

            rewards = torch.from_numpy(self.steps[t].rewards).float().to(device)
            values = self.steps[t].values.to(device)

            delta = rewards + gamma * next_values * next_non_terminal - values
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        values_tensor = torch.stack([s.values for s in self.steps]).to(device)
        self.returns = advantages + values_tensor

    def get_batches(
        self,
        num_minibatches: int,
        device: torch.device,
        *,
        game_spec: GameSpec | None = None,
    ) -> Iterator[MiniBatch]:
        """Yield shuffled minibatches of experience.

        Flattens (num_steps, num_envs) into a single dimension, shuffles,
        and splits into num_minibatches chunks. Each chunk gets collated
        into padded tensors via collate_policy_inputs.
        """
        assert self.advantages is not None, "Call compute_returns first"

        total = self.num_steps * self.num_envs
        mb_size = total // num_minibatches
        indices = np.random.permutation(total)

        # Flatten all data
        all_obs: list[GameObservation] = []  # type: ignore[type-arg]
        all_masks: list[GameActionMask] = []  # type: ignore[type-arg]
        all_shop_splits: list[tuple[int, int, int]] = []
        all_actions: list[FactoredAction] = []
        all_log_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for step in self.steps:
            for env_idx in range(self.num_envs):
                all_obs.append(step.obs[env_idx])
                all_masks.append(step.masks[env_idx])
                all_shop_splits.append(step.shop_splits[env_idx])
                all_actions.append(step.actions[env_idx])
                all_log_probs.append(step.log_probs[env_idx])
                all_values.append(step.values[env_idx])

        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_log_probs = torch.stack(all_log_probs)
        flat_values = torch.stack(all_values)

        for start in range(0, total, mb_size):
            end = min(start + mb_size, total)
            mb_indices = indices[start:end]

            mb_obs = [all_obs[i] for i in mb_indices]
            mb_masks = [all_masks[i] for i in mb_indices]
            mb_shop_splits = [all_shop_splits[i] for i in mb_indices]
            mb_actions = [all_actions[i] for i in mb_indices]
            mb_policy_inputs = [
                PolicyInput(obs=o, action_mask=m, shop_splits=ss)
                for o, m, ss in zip(mb_obs, mb_masks, mb_shop_splits)
            ]

            batch = collate_policy_inputs(mb_policy_inputs, game_spec, device=device)

            yield MiniBatch(
                batch=batch,
                actions=mb_actions,
                old_log_probs=flat_log_probs[mb_indices].to(device),
                advantages=flat_advantages[mb_indices].to(device),
                returns=flat_returns[mb_indices].to(device),
                old_values=flat_values[mb_indices].to(device),
            )


# ---------------------------------------------------------------------------
# Training results
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Result of a PPO training run."""

    total_timesteps: int = 0
    total_updates: int = 0
    total_episodes: int = 0
    wall_time: float = 0.0
    final_eval: EvalMetrics | None = None
    log_history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class EvalMetrics:
    """Evaluation metrics from periodic evaluation."""

    win_rate: float = 0.0
    avg_ante: float = 0.0
    avg_return: float = 0.0
    avg_length: float = 0.0
    n_episodes: int = 0


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """CleanRL-style PPO trainer for Balatro."""

    def __init__(
        self,
        config: PPOConfig,
        adapter_factory: Callable[[], GameAdapter] | None = None,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.back_keys = _normalize_back_keys(config.back_keys)
        self.stakes = _normalize_stakes(config.stake)

        # Curriculum
        self.curriculum: CurriculumManager | None = None
        initial_reward_config: RewardConfig | None = None
        if config.curriculum is not None:
            self.curriculum = CurriculumManager(config.curriculum)
            initial_reward_config = self.curriculum.current_reward_config
            print(
                f"[Jackdaw] Curriculum: {len(config.curriculum.stages)} stages, "
                f"starting with '{self.curriculum.current_stage.name}'"
            )

        # Create vectorized environment
        env_fns = [
            _make_env(
                self.back_keys,
                self.stakes,
                config.max_steps_per_episode,
                i,
                adapter_factory,
                reward_config=initial_reward_config,
            )
            for i in range(config.num_envs)
        ]
        self.envs = SyncVectorEnv(env_fns)

        # Create policy
        self.policy = BalatroPolicy(
            config.game_spec,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        ).to(self.device)

        # Device info
        n_params = sum(p.numel() for p in self.policy.parameters())
        device_name = str(self.device)
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            device_name = f"{self.device} ({gpu_name})"
        print(f"[Jackdaw] Using device: {device_name}")
        print(f"[Jackdaw] Policy parameters: {n_params:,}")

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate, eps=1e-5)

        # Tracking
        self.global_step = 0
        self.update_count = 0
        self.episode_count = 0

        # Episode rolling window
        self._episode_returns: deque[float] = deque(maxlen=100)
        self._episode_lengths: deque[float] = deque(maxlen=100)
        self._episode_antes: deque[float] = deque(maxlen=100)
        self._episode_wins: deque[bool] = deque(maxlen=100)

        # TensorBoard
        self.writer: SummaryWriter | None = None  # type: ignore[assignment]
        if _HAS_TENSORBOARD:
            run_name = config.run_name
            if run_name is None:
                back = (
                    config.back_keys if isinstance(config.back_keys, str) else config.back_keys[0]
                )
                run_name = f"balatro_{back}_{int(time.time())}"
            log_path = f"{config.log_dir}/{run_name}"
            self.writer = SummaryWriter(log_dir=log_path)
            self.writer.add_text("config", str(asdict(config)))
        else:
            logger.warning(
                "tensorboard not installed — training will log to stdout only. "
                "Install with: uv pip install tensorboard"
            )

        # Current observations/masks (set on first reset)
        self._current_obs: list[GameObservation] = []  # type: ignore[type-arg]
        self._current_masks: list[GameActionMask] = []  # type: ignore[type-arg]
        self._current_infos: list[dict[str, Any]] = []

    def train(self) -> TrainResult:
        """Main training loop."""
        cfg = self.config
        num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
        result = TrainResult()
        start_time = time.time()

        # Initial reset
        reset_data = self.envs.reset()
        self._current_obs = [d[0] for d in reset_data]
        self._current_masks = [d[1] for d in reset_data]
        self._current_infos = [d[2] for d in reset_data]

        for update in range(1, num_updates + 1):
            # Collect rollouts
            buffer = self._collect_rollouts()

            # Update policy
            update_metrics = self._update(buffer)
            self.update_count = update

            # Logging
            if update % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / max(elapsed, 1e-8)
                metrics = {
                    "update": float(update),
                    "global_step": float(self.global_step),
                    "episodes": float(self.episode_count),
                    "steps_per_second": sps,
                    "wall_time": elapsed,
                    **update_metrics,
                }
                result.log_history.append(metrics)
                self._log_metrics(metrics)

                if self.writer is not None:
                    gs = self.global_step
                    self.writer.add_scalar("charts/SPS", sps, gs)
                    self.writer.add_scalar("charts/episodes", self.episode_count, gs)
                    um = update_metrics
                    self.writer.add_scalar("losses/policy_loss", um["policy_loss"], gs)
                    self.writer.add_scalar("losses/value_loss", um["value_loss"], gs)
                    self.writer.add_scalar("losses/entropy", um["entropy"], gs)
                    self.writer.add_scalar("losses/total_loss", um["total_loss"], gs)
                    self.writer.add_scalar("losses/clip_fraction", um["clip_fraction"], gs)
                    self.writer.add_scalar("losses/approx_kl", um["approx_kl"], gs)
                    ev = um["explained_variance"]
                    self.writer.add_scalar("losses/explained_variance", ev, gs)
                    self.writer.add_scalar("losses/learning_rate", um["learning_rate"], gs)

                    if self.curriculum is not None:
                        for k, v in self.curriculum.get_metrics().items():
                            self.writer.add_scalar(k, v, gs)

            # Evaluation
            if update % cfg.eval_interval == 0:
                eval_metrics = self._evaluate()
                result.final_eval = eval_metrics
                self._log_eval(eval_metrics, update)
                if self.writer is not None:
                    gs = self.global_step
                    self.writer.add_scalar("eval/win_rate", eval_metrics.win_rate, gs)
                    self.writer.add_scalar("eval/avg_ante", eval_metrics.avg_ante, gs)
                    self.writer.add_scalar("eval/avg_length", eval_metrics.avg_length, gs)

            # Checkpointing
            if update % cfg.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{update}.pt")

        if self.writer is not None:
            self.writer.close()

        result.total_timesteps = self.global_step
        result.total_updates = self.update_count
        result.total_episodes = self.episode_count
        result.wall_time = time.time() - start_time
        return result

    def _collect_rollouts(self) -> RolloutBuffer:
        """Collect num_steps of experience from all environments."""
        cfg = self.config
        buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs)
        self.policy.eval()

        for step in range(cfg.num_steps):
            # Build policy inputs from current observations
            step_shop_splits = [
                info.get("shop_splits", (0, 0, 0)) for info in self._current_infos
            ]
            policy_inputs = [
                PolicyInput(obs=obs, action_mask=mask, shop_splits=ss)
                for obs, mask, ss in zip(self._current_obs, self._current_masks, step_shop_splits)
            ]
            batch = collate_policy_inputs(
                policy_inputs, self.config.game_spec, device=self.device,
            )

            # Sample actions (also returns value estimates)
            with torch.no_grad():
                actions, log_probs_dict, values = self.policy.sample_action(batch)
                values = values.cpu()

            total_log_probs = log_probs_dict["total"].cpu()

            # Step environments
            obs_list, rewards, terminateds, truncateds, masks, infos = self.envs.step(actions)
            dones = terminateds | truncateds
            self.global_step += cfg.num_envs

            # Track completed episodes
            for i, info in enumerate(infos):
                if dones[i]:
                    self.episode_count += 1
                    ep_ante = info.get("episode_ante", 1)
                    ep_won = bool(info.get("episode_won", False))
                    self._episode_returns.append(info.get("episode_return", 0.0))
                    self._episode_lengths.append(info.get("episode_length", 0.0))
                    self._episode_antes.append(float(ep_ante))
                    self._episode_wins.append(ep_won)

                    # Curriculum tracking
                    if self.curriculum is not None:
                        advanced = self.curriculum.record_episode(ep_ante, ep_won)
                        if advanced:
                            self._on_curriculum_advance()

                    if self.writer is not None and self._episode_returns:
                        gs = self.global_step
                        self.writer.add_scalar(
                            "charts/episode_return", np.mean(self._episode_returns), gs
                        )
                        self.writer.add_scalar(
                            "charts/episode_length", np.mean(self._episode_lengths), gs
                        )
                        self.writer.add_scalar("charts/avg_ante", np.mean(self._episode_antes), gs)
                        self.writer.add_scalar("charts/win_rate", np.mean(self._episode_wins), gs)
                        self.writer.add_scalar("charts/max_ante", max(self._episode_antes), gs)

            buffer.add(
                StepData(
                    obs=self._current_obs,
                    masks=self._current_masks,
                    shop_splits=step_shop_splits,
                    actions=actions,
                    log_probs=total_log_probs,
                    values=values,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                )
            )

            # Update current state (envs auto-reset)
            self._current_obs = obs_list
            self._current_masks = masks
            self._current_infos = infos

        # Compute bootstrap value for last step
        with torch.no_grad():
            bootstrap_shop_splits = [
                info.get("shop_splits", (0, 0, 0)) for info in self._current_infos
            ]
            policy_inputs = [
                PolicyInput(obs=obs, action_mask=mask, shop_splits=ss)
                for obs, mask, ss in zip(
                    self._current_obs, self._current_masks, bootstrap_shop_splits
                )
            ]
            batch = collate_policy_inputs(
                policy_inputs, self.config.game_spec, device=self.device,
            )
            out = self.policy.forward(batch)
            last_value = out["value"].squeeze(-1).cpu()

        last_done = np.zeros(cfg.num_envs, dtype=np.bool_)
        buffer.compute_returns(last_value, last_done, cfg.gamma, cfg.gae_lambda)
        return buffer

    def _update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update epochs on the collected rollout."""
        cfg = self.config
        self.policy.train()

        clip_fracs: list[float] = []
        all_pg_loss: list[float] = []
        all_vf_loss: list[float] = []
        all_ent_loss: list[float] = []
        all_total_loss: list[float] = []

        for _epoch in range(cfg.update_epochs):
            for mb in buffer.get_batches(
                cfg.num_minibatches, self.device, game_spec=self.config.game_spec,
            ):
                # Evaluate actions under current policy
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    mb.batch, mb.actions
                )

                # Normalize advantages
                advantages = mb.advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss with clipping
                log_ratio = new_log_probs - mb.old_log_probs
                ratio = log_ratio.exp()

                # Clip fraction for diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    clip_fracs.append(clip_frac)

                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (new_values - mb.returns) ** 2
                v_clipped = mb.old_values + torch.clamp(
                    new_values - mb.old_values, -cfg.clip_coef, cfg.clip_coef
                )
                v_loss_clipped = (v_clipped - mb.returns) ** 2
                vf_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy loss (maximize entropy -> negative sign)
                ent_loss = entropy.mean()

                # Total loss
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * vf_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                all_pg_loss.append(pg_loss.item())
                all_vf_loss.append(vf_loss.item())
                all_ent_loss.append(ent_loss.item())
                all_total_loss.append(loss.item())

        # Explained variance
        assert buffer.returns is not None
        y_pred = torch.stack([s.values for s in buffer.steps]).reshape(-1)
        y_true = buffer.returns.cpu().reshape(-1)
        var_y = y_true.var()
        explained_var = float("nan") if var_y == 0 else 1 - (y_true - y_pred).var() / var_y

        return {
            "policy_loss": float(np.mean(all_pg_loss)),
            "value_loss": float(np.mean(all_vf_loss)),
            "entropy": float(np.mean(all_ent_loss)),
            "total_loss": float(np.mean(all_total_loss)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "approx_kl": approx_kl,
            "explained_variance": float(explained_var),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _evaluate(self) -> EvalMetrics:
        """Evaluate the current policy over several episodes."""
        from jackdaw.env.agents import evaluate_agent

        # Wrap policy as an Agent
        agent = _PolicyAgent(self.policy, self.device)
        result = evaluate_agent(
            agent,
            n_episodes=self.config.eval_episodes,
            back_key=self.back_keys[0],
            stake=self.stakes[0],
            max_steps=self.config.max_steps_per_episode,
        )
        return EvalMetrics(
            win_rate=result.win_rate,
            avg_ante=result.avg_ante,
            avg_return=0.0,  # evaluate_agent doesn't track returns
            avg_length=result.avg_actions,
            n_episodes=result.n_episodes,
        )

    def save_checkpoint(self, path: str, extra: dict[str, Any] | None = None) -> None:
        """Save model weights, optimizer state, and config.

        Parameters
        ----------
        path:
            File path for the checkpoint.
        extra:
            Optional extra data to include (e.g. best_avg_ante).
        """
        checkpoint: dict[str, Any] = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "config": asdict(self.config),
        }
        if self.curriculum is not None:
            checkpoint["curriculum_state"] = {
                "stage_idx": self.curriculum.stage_index,
                "stage_episodes": self.curriculum._stage_episodes,
                "history": self.curriculum.transition_history,
            }
        if extra:
            checkpoint.update(extra)

        # Atomic write: save to .tmp then rename
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        Path(tmp_path).replace(path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load model weights, optimizer state, and training progress.

        Returns the full checkpoint dict for callers that need extra fields.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.update_count = checkpoint.get("update_count", 0)
        self.episode_count = checkpoint.get("episode_count", 0)

        # Restore curriculum state
        cur_state = checkpoint.get("curriculum_state")
        if cur_state is not None and self.curriculum is not None:
            self.curriculum._stage_idx = cur_state["stage_idx"]
            self.curriculum._stage_episodes = cur_state["stage_episodes"]
            self.curriculum._history = cur_state.get("history", [])
            # Update env reward configs to match restored stage
            new_config = self.curriculum.current_reward_config
            for env in self.envs.envs:
                env._reward.update_config(new_config)

        return checkpoint

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Print training metrics."""
        print(
            f"[Update {int(metrics['update']):>5d}] "
            f"step={int(metrics['global_step']):>8d}  "
            f"episodes={int(metrics['episodes']):>5d}  "
            f"SPS={metrics['steps_per_second']:.0f}  "
            f"pg_loss={metrics['policy_loss']:.4f}  "
            f"vf_loss={metrics['value_loss']:.4f}  "
            f"entropy={metrics['entropy']:.4f}  "
            f"clip={metrics['clip_fraction']:.3f}  "
            f"ev={metrics['explained_variance']:.3f}"
        )

    def _log_eval(self, eval_metrics: EvalMetrics, update: int) -> None:
        """Print evaluation metrics."""
        print(
            f"[Eval @ update {update}] "
            f"win_rate={eval_metrics.win_rate:.1%}  "
            f"avg_ante={eval_metrics.avg_ante:.1f}  "
            f"avg_length={eval_metrics.avg_length:.0f}  "
            f"({eval_metrics.n_episodes} episodes)"
        )

    def _on_curriculum_advance(self) -> None:
        """Handle curriculum stage transition."""
        assert self.curriculum is not None
        stage = self.curriculum.current_stage
        new_config = self.curriculum.current_reward_config
        print(
            f"[Jackdaw] Curriculum → stage {self.curriculum.stage_index}: "
            f"'{stage.name}' (episode {self.episode_count})"
        )
        # Update reward config in all envs
        for env in self.envs.envs:
            env._reward.update_config(new_config)

        if self.writer is not None:
            gs = self.global_step
            self.writer.add_scalar("curriculum/stage", float(self.curriculum.stage_index), gs)


# ---------------------------------------------------------------------------
# Policy agent wrapper (for evaluation harness)
# ---------------------------------------------------------------------------


class _PolicyAgent:
    """Wraps BalatroPolicy as an Agent for use with evaluate_agent.

    This is Balatro-specific glue code — it encodes raw game states into
    the game-agnostic observation format expected by the policy.
    """

    def __init__(self, policy: BalatroPolicy, device: torch.device) -> None:
        self._policy = policy
        self._device = device

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: object, info: dict) -> FactoredAction:
        # Late imports: Balatro-specific encoding used only here
        from jackdaw.env.balatro_env import _action_mask_to_game, _compute_shop_splits
        from jackdaw.env.observation import encode_observation

        gs = info["raw_state"]
        encoded_obs = encode_observation(gs)
        game_obs = encoded_obs.to_game_observation()
        game_mask = _action_mask_to_game(action_mask)  # type: ignore[arg-type]
        policy_input = PolicyInput(
            obs=game_obs,
            action_mask=game_mask,
            shop_splits=_compute_shop_splits(gs),
        )
        batch = collate_policy_inputs(
            [policy_input], self._policy.game_spec, device=self._device,
        )

        self._policy.eval()
        with torch.no_grad():
            actions, _, _ = self._policy.sample_action(batch)
        return actions[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train_ppo(
    config: PPOConfig | None = None,
    adapter_factory: Callable[[], GameAdapter] | None = None,
) -> TrainResult:
    """Train a Balatro agent with PPO.

    Parameters
    ----------
    config:
        Training configuration. Uses defaults if None.
    adapter_factory:
        Optional factory for game adapters. Defaults to DirectAdapter.

    Returns
    -------
    TrainResult
        Training statistics and final evaluation.
    """
    trainer = PPOTrainer(config or PPOConfig(), adapter_factory=adapter_factory)
    return trainer.train()
