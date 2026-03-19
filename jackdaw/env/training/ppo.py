"""CleanRL-style PPO training loop for Balatro.

Custom loop required because the policy architecture (variable-length entities,
autoregressive action heads, pointer networks) doesn't fit standard RL library
interfaces like SB3.

Reference: https://docs.cleanrl.dev/rl-algorithms/ppo/
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from jackdaw.env.action_space import (
    ActionMask,
    FactoredAction,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.game_interface import DirectAdapter, GameAdapter
from jackdaw.env.observation import Observation, encode_observation
from jackdaw.env.policy.policy import (
    BalatroPolicy,
    PolicyInput,
    collate_policy_inputs,
)
from jackdaw.env.rewards import DenseRewardWrapper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Environment
    num_envs: int = 8
    back_keys: list[str] | str = "b_red"  # single key or list to sample from
    stake: int = 1
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

    def reset(self) -> list[tuple[Observation, ActionMask, dict[str, Any]]]:
        """Reset all environments, return (obs, mask, info) per env."""
        results = []
        for env in self.envs:
            obs, mask, info = env.reset()
            results.append((obs, mask, info))
        return results

    def step(
        self, actions: list[FactoredAction]
    ) -> tuple[
        list[Observation],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[ActionMask],
        list[dict[str, Any]],
    ]:
        """Step all environments.

        Returns (obs_list, rewards, terminateds, truncateds, masks, infos).
        Auto-resets environments that are done.
        """
        obs_list: list[Observation] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=np.bool_)
        truncateds = np.zeros(self.num_envs, dtype=np.bool_)
        masks: list[ActionMask] = []
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
    """Single environment instance wrapping a GameAdapter with reward shaping."""

    def __init__(
        self,
        adapter_factory: Callable[[], GameAdapter],
        reward_wrapper: DenseRewardWrapper,
        back_keys: list[str],
        stake: int,
        max_steps: int,
        seed_prefix: str = "TRAIN",
    ) -> None:
        self._adapter_factory = adapter_factory
        self._adapter = adapter_factory()
        self._reward = reward_wrapper
        self._back_keys = back_keys
        self._stake = stake
        self._max_steps = max_steps
        self._seed_prefix = seed_prefix
        self._episode_count = 0
        self._step_count = 0

        # Episode tracking
        self.episode_return: float = 0.0
        self.episode_length: int = 0
        self.episode_won: bool = False
        self.episode_ante: int = 1

    def reset(self) -> tuple[Observation, ActionMask, dict[str, Any]]:
        seed = f"{self._seed_prefix}_{self._episode_count}"
        self._episode_count += 1
        self._step_count = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_won = False
        self.episode_ante = 1

        self._adapter = self._adapter_factory()
        if len(self._back_keys) > 1:
            back_key = random.choice(self._back_keys)
        else:
            back_key = self._back_keys[0]
        self._adapter.reset(back_key, self._stake, seed)
        self._reward.reset()

        gs = self._adapter.raw_state
        obs = encode_observation(gs)
        mask = get_action_mask(gs)
        info = {"raw_state": gs}
        return obs, mask, info

    def step(
        self, action: FactoredAction
    ) -> tuple[Observation, float, bool, bool, ActionMask, dict[str, Any]]:
        gs_prev = self._adapter.raw_state
        engine_action = factored_to_engine_action(action, gs_prev)
        self._adapter.step(engine_action)

        gs = self._adapter.raw_state
        reward = self._reward.reward(gs_prev, action, gs)
        self.episode_return += reward
        self.episode_length += 1
        self._step_count += 1

        terminated = self._adapter.done
        truncated = self._step_count >= self._max_steps

        # Track episode stats
        rr = gs.get("round_resets", {})
        self.episode_ante = rr.get("ante", 1)
        self.episode_won = self._adapter.won

        obs = encode_observation(gs)
        mask = get_action_mask(gs)
        info = {"raw_state": gs}
        return obs, reward, terminated, truncated, mask, info


def _make_env(
    back_keys: list[str],
    stake: int,
    max_steps: int,
    env_idx: int,
    adapter_factory: Callable[[], GameAdapter] | None = None,
) -> Callable[[], _EnvInstance]:
    """Create a factory function for a single environment instance."""

    def _factory() -> _EnvInstance:
        factory = adapter_factory or DirectAdapter
        return _EnvInstance(
            adapter_factory=factory,
            reward_wrapper=DenseRewardWrapper(),
            back_keys=back_keys,
            stake=stake,
            max_steps=max_steps,
            seed_prefix=f"TRAIN_E{env_idx}",
        )

    return _factory


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


@dataclass
class StepData:
    """Data for a single step across all environments."""

    obs: list[Observation]
    masks: list[ActionMask]
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
        all_obs: list[Observation] = []
        all_masks: list[ActionMask] = []
        all_actions: list[FactoredAction] = []
        all_log_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for step in self.steps:
            for env_idx in range(self.num_envs):
                all_obs.append(step.obs[env_idx])
                all_masks.append(step.masks[env_idx])
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
            mb_actions = [all_actions[i] for i in mb_indices]
            mb_policy_inputs = [PolicyInput(obs=o, action_mask=m) for o, m in zip(mb_obs, mb_masks)]

            batch = collate_policy_inputs(mb_policy_inputs, device=device)

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

        # Create vectorized environment
        env_fns = [
            _make_env(
                self.back_keys,
                config.stake,
                config.max_steps_per_episode,
                i,
                adapter_factory,
            )
            for i in range(config.num_envs)
        ]
        self.envs = SyncVectorEnv(env_fns)

        # Create policy
        self.policy = BalatroPolicy(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate, eps=1e-5)

        # Tracking
        self.global_step = 0
        self.update_count = 0
        self.episode_count = 0

        # Current observations/masks (set on first reset)
        self._current_obs: list[Observation] = []
        self._current_masks: list[ActionMask] = []
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

            # Evaluation
            if update % cfg.eval_interval == 0:
                eval_metrics = self._evaluate()
                result.final_eval = eval_metrics
                self._log_eval(eval_metrics, update)

            # Checkpointing
            if update % cfg.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{update}.pt")

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
            policy_inputs = [
                PolicyInput(obs=obs, action_mask=mask)
                for obs, mask in zip(self._current_obs, self._current_masks)
            ]
            batch = collate_policy_inputs(policy_inputs, device=self.device)

            # Sample actions
            with torch.no_grad():
                actions, log_probs_dict = self.policy.sample_action(batch)
                # Get values
                out = self.policy.forward(batch)
                values = out["value"].squeeze(-1).cpu()

            total_log_probs = log_probs_dict["total"].cpu()

            # Step environments
            obs_list, rewards, terminateds, truncateds, masks, infos = self.envs.step(actions)
            dones = terminateds | truncateds
            self.global_step += cfg.num_envs

            # Track completed episodes
            for i, info in enumerate(infos):
                if dones[i]:
                    self.episode_count += 1

            buffer.add(
                StepData(
                    obs=self._current_obs,
                    masks=self._current_masks,
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
            policy_inputs = [
                PolicyInput(obs=obs, action_mask=mask)
                for obs, mask in zip(self._current_obs, self._current_masks)
            ]
            batch = collate_policy_inputs(policy_inputs, device=self.device)
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
            for mb in buffer.get_batches(cfg.num_minibatches, self.device):
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
            stake=self.config.stake,
            max_steps=self.config.max_steps_per_episode,
        )
        return EvalMetrics(
            win_rate=result.win_rate,
            avg_ante=result.avg_ante,
            avg_return=0.0,  # evaluate_agent doesn't track returns
            avg_length=result.avg_actions,
            n_episodes=result.n_episodes,
        )

    def save_checkpoint(self, path: str) -> None:
        """Save model weights, optimizer state, and config."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "config": asdict(self.config),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights, optimizer state, and training progress."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.update_count = checkpoint.get("update_count", 0)
        self.episode_count = checkpoint.get("episode_count", 0)

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


# ---------------------------------------------------------------------------
# Policy agent wrapper (for evaluation harness)
# ---------------------------------------------------------------------------


class _PolicyAgent:
    """Wraps BalatroPolicy as an Agent for use with evaluate_agent."""

    def __init__(self, policy: BalatroPolicy, device: torch.device) -> None:
        self._policy = policy
        self._device = device

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        gs = info["raw_state"]
        encoded_obs = encode_observation(gs)
        policy_input = PolicyInput(obs=encoded_obs, action_mask=action_mask)
        batch = collate_policy_inputs([policy_input], device=self._device)

        self._policy.eval()
        with torch.no_grad():
            actions, _ = self._policy.sample_action(batch)
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
