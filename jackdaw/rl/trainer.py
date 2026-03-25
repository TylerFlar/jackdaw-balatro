"""CleanRL-style PPO trainer for the factored Balatro policy."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_spec import FactoredAction, GameActionMask
from jackdaw.rl.network import (
    ENTITY_MAX_COUNTS,
    FactoredPolicy,
    NEEDS_CARDS,
    NEEDS_ENTITY,
)
from jackdaw.rl.rollout import RolloutBuffer, Transition
from jackdaw.rl.vec_env import SubprocVecEnv

_SPEC = balatro_game_spec()
HAND_CARD_MAX = _SPEC.entity_types[0].max_count  # 8


def _pad_mask(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad a variable-length bool mask to a fixed length."""
    if len(arr) >= target_len:
        return arr[:target_len].astype(bool)
    padded = np.zeros(target_len, dtype=bool)
    padded[: len(arr)] = arr
    return padded


def _masks_to_numpy(
    mask: GameActionMask,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], int, int]:
    """Convert a GameActionMask to padded numpy arrays for storage."""
    type_mask = mask.type_mask.astype(bool)
    card_mask = _pad_mask(mask.card_mask, HAND_CARD_MAX)

    entity_masks: dict[int, np.ndarray] = {}
    for atype, emask in mask.entity_masks.items():
        etype_idx = _SPEC.entity_type_for_action(atype)
        if etype_idx >= 0:
            entity_masks[atype] = _pad_mask(emask, ENTITY_MAX_COUNTS[etype_idx])

    return type_mask, card_mask, entity_masks, mask.min_card_select, mask.max_card_select


def _batch_obs_to_device(
    obs_list: list[dict[str, np.ndarray]], device: torch.device
) -> dict[str, torch.Tensor]:
    """Stack N obs dicts into batched (B=N) tensors."""
    keys = list(obs_list[0].keys())
    return {
        k: torch.from_numpy(np.stack([o[k] for o in obs_list])).float().to(device)
        for k in keys
    }


def _batch_masks_to_device(
    masks_np: list[tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], int, int]],
    device: torch.device,
) -> dict[str, Any]:
    """Stack N mask tuples into batched tensors."""
    type_masks = torch.from_numpy(np.stack([m[0] for m in masks_np])).bool().to(device)
    card_masks = torch.from_numpy(np.stack([m[1] for m in masks_np])).bool().to(device)

    min_cs = torch.tensor([m[3] for m in masks_np], dtype=torch.long, device=device)
    max_cs = torch.tensor([m[4] for m in masks_np], dtype=torch.long, device=device)

    # Entity masks: union of all keys across envs
    all_keys: set[int] = set()
    for m in masks_np:
        all_keys.update(m[2].keys())

    entity_masks: dict[int, torch.Tensor] = {}
    for atype in all_keys:
        arrs = []
        ref_shape: tuple[int, ...] | None = None
        for m in masks_np:
            if atype in m[2]:
                arrs.append(m[2][atype])
                if ref_shape is None:
                    ref_shape = m[2][atype].shape
            else:
                arrs.append(None)
        # Fill None entries with zeros
        assert ref_shape is not None
        filled = [a if a is not None else np.zeros(ref_shape, dtype=bool) for a in arrs]
        entity_masks[atype] = torch.from_numpy(np.stack(filled)).bool().to(device)

    return {
        "type_mask": type_masks,
        "card_mask": card_masks,
        "entity_masks": entity_masks,
        "min_card_select": min_cs,
        "max_card_select": max_cs,
    }


class BalatroTrainer:
    """PPO trainer for the factored Balatro policy.

    Supports both single-env and multi-env (SubprocVecEnv) collection.

    Parameters
    ----------
    vec_env : SubprocVecEnv
        Vectorized environment with N workers.
    network : FactoredPolicy
    lr : learning rate
    gamma, gae_lambda : GAE parameters
    clip_range : PPO clipping epsilon
    ent_coef : entropy targeting coefficient
    entropy_target : target entropy for quadratic penalty
    vf_coef : value loss coefficient
    n_steps : total rollout transitions per update (across all envs)
    n_epochs : PPO epochs per rollout
    batch_size : minibatch size for PPO updates
    max_grad_norm : gradient clipping
    device : "auto", "cpu", or "cuda"
    log_dir : tensorboard log directory
    total_timesteps : total env steps (used for LR schedule length)
    checkpoint_interval : save checkpoint every N updates
    """

    def __init__(
        self,
        vec_env: SubprocVecEnv,
        network: FactoredPolicy,
        lr: float = 2e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.15,
        ent_coef: float = 0.2,
        entropy_target: float = 2.0,
        vf_coef: float = 0.5,
        n_steps: int = 4096,
        n_epochs: int = 10,
        batch_size: int = 512,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        log_dir: str = "runs/balatro_factored",
        total_timesteps: int = 5_000_000,
        checkpoint_interval: int = 50,
    ) -> None:
        self.vec_env = vec_env
        self.n_envs = vec_env.n_envs
        self.network = network
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.entropy_target = entropy_target
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.log_dir = log_dir
        self.checkpoint_interval = checkpoint_interval

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        total_updates = total_timesteps // n_steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_updates,
            eta_min=lr / 10,
        )

        self.writer = SummaryWriter(log_dir)

        # Episode tracking
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_antes: list[int] = []
        self._ep_rounds: list[int] = []
        self._ep_wins: list[bool] = []
        self._recent_wins: deque[bool] = deque(maxlen=100)

        # Per-env state
        self._env_obs: list[dict[str, np.ndarray] | None] = [None] * self.n_envs
        self._env_masks: list[GameActionMask | None] = [None] * self.n_envs
        self._env_ep_reward: list[float] = [0.0] * self.n_envs
        self._env_ep_len: list[int] = [0] * self.n_envs

    def _reset_all(self) -> None:
        """Reset all environments."""
        obs_list, mask_list = self.vec_env.reset_all()
        for i in range(self.n_envs):
            self._env_obs[i] = obs_list[i]
            self._env_masks[i] = mask_list[i]
            self._env_ep_reward[i] = 0.0
            self._env_ep_len[i] = 0

    def collect_rollout(self) -> tuple[RolloutBuffer, list[float]]:
        """Collect n_steps transitions across all environments.

        Returns (buffer, last_values) where last_values is per-env bootstrap values.
        """
        buf = RolloutBuffer(n_envs=self.n_envs)
        self.network.eval()

        if self._env_obs[0] is None:
            self._reset_all()

        steps_per_env = self.n_steps // self.n_envs
        for _ in range(steps_per_env):
            # Prepare batched numpy masks for storage
            masks_np = []
            for i in range(self.n_envs):
                masks_np.append(_masks_to_numpy(self._env_masks[i]))

            # Batch observations and masks for network
            obs_t = _batch_obs_to_device(self._env_obs, self.device)
            masks_t = _batch_masks_to_device(masks_np, self.device)

            with torch.no_grad():
                out = self.network(obs_t, masks_t)

            # Extract per-env actions
            action_types = out["action_type"].cpu().numpy()  # (N,)
            entity_targets = out["entity_target"].cpu().numpy()  # (N,)
            card_targets = out["card_target"].cpu().numpy()  # (N, max_hand)
            log_probs = out["log_prob"].cpu().numpy()  # (N,)
            values = out["value"].cpu().numpy()  # (N,)

            # Build FactoredActions for each env
            actions: list[FactoredAction] = []
            for i in range(self.n_envs):
                at = int(action_types[i])
                et_val = int(entity_targets[i])
                ct_arr = card_targets[i]

                ct: tuple[int, ...] | None = None
                et: int | None = None

                if at in NEEDS_ENTITY and et_val >= 0:
                    et = et_val
                if at in NEEDS_CARDS:
                    selected = np.nonzero(ct_arr)[0]
                    if len(selected) > 0:
                        ct = tuple(int(j) for j in selected)

                actions.append(FactoredAction(action_type=at, card_target=ct, entity_target=et))

            # Step all envs
            results = self.vec_env.step(actions)

            # Record transitions
            for i in range(self.n_envs):
                obs, reward, done, mask, terminal_info, new_obs, new_mask = results[i]

                buf.add(
                    Transition(
                        obs=self._env_obs[i],
                        action_type=int(action_types[i]),
                        entity_target=int(entity_targets[i]),
                        card_target=card_targets[i].astype(bool),
                        log_prob=float(log_probs[i]),
                        value=float(values[i]),
                        reward=reward,
                        done=done,
                        type_mask=masks_np[i][0],
                        card_mask=masks_np[i][1],
                        entity_masks=masks_np[i][2],
                        min_card_select=masks_np[i][3],
                        max_card_select=masks_np[i][4],
                    ),
                    env_idx=i,
                )

                self._env_ep_reward[i] += reward
                self._env_ep_len[i] += 1

                if done:
                    self._ep_rewards.append(self._env_ep_reward[i])
                    self._ep_lengths.append(self._env_ep_len[i])
                    if terminal_info:
                        self._ep_antes.append(terminal_info.get("balatro/ante_reached", 1))
                        self._ep_rounds.append(terminal_info.get("balatro/rounds_beaten", 0))
                        won = terminal_info.get("balatro/won", False)
                        self._ep_wins.append(won)
                        self._recent_wins.append(won)
                    # Auto-reset: worker already reset, use new obs/mask
                    self._env_obs[i] = new_obs
                    self._env_masks[i] = new_mask
                    self._env_ep_reward[i] = 0.0
                    self._env_ep_len[i] = 0
                else:
                    self._env_obs[i] = obs
                    self._env_masks[i] = mask

        # Bootstrap values for all envs
        with torch.no_grad():
            obs_t = _batch_obs_to_device(self._env_obs, self.device)
            masks_np = [_masks_to_numpy(self._env_masks[i]) for i in range(self.n_envs)]
            masks_t = _batch_masks_to_device(masks_np, self.device)
            state, _ = self.network._encode(obs_t)
            last_values = self.network.value_head(state).squeeze(-1).cpu().numpy().tolist()

        return buf, last_values

    def train_epoch(self, data: dict[str, Any]) -> dict[str, float]:
        """One epoch of PPO updates over the buffer."""
        self.network.train()
        N = data["action_type"].shape[0]
        indices = np.arange(N)
        np.random.shuffle(indices)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_grad_norm = 0.0
        total_max_ratio = 0.0
        total_lr_clipped_frac = 0.0
        total_entropy_deviation = 0.0
        n_batches = 0

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            idx = indices[start:end]
            idx_t = torch.from_numpy(idx).long().to(self.device)

            # Slice batch
            obs_b = {k: v[idx_t] for k, v in data["obs"].items()}
            at_b = data["action_type"][idx_t]
            et_b = data["entity_target"][idx_t]
            ct_b = data["card_target"][idx_t]
            old_lp_b = data["old_log_prob"][idx_t]
            adv_b = data["advantages"][idx_t]
            ret_b = data["returns"][idx_t]

            # Slice action masks
            masks_b = {
                "type_mask": data["action_masks"]["type_mask"][idx_t],
                "card_mask": data["action_masks"]["card_mask"][idx_t],
                "entity_masks": {
                    atype: m[idx_t] for atype, m in data["action_masks"]["entity_masks"].items()
                },
                "min_card_select": data["action_masks"]["min_card_select"][idx_t],
                "max_card_select": data["action_masks"]["max_card_select"][idx_t],
            }

            # Normalize advantages
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            # Re-evaluate
            new_lp, new_val, entropy = self.network.evaluate(obs_b, masks_b, at_b, et_b, ct_b)

            # NaN guard — skip batch if network produced NaN
            if torch.isnan(new_lp).any() or torch.isnan(new_val).any():
                continue

            # PPO policy loss — clamp ratio to prevent exp() overflow
            raw_log_ratio = new_lp - old_lp_b
            log_ratio = raw_log_ratio.clamp(-5.0, 5.0)
            ratio = log_ratio.exp()
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss
            old_val_b = data["old_values"][idx_t]
            value_clipped = old_val_b + (new_val - old_val_b).clamp(
                -self.clip_range, self.clip_range
            )
            value_loss_unclipped = (new_val - ret_b).pow(2)
            value_loss_clipped = (value_clipped - ret_b).pow(2)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # Entropy targeting — pull entropy toward target from both directions
            entropy_mean = entropy.mean()
            entropy_loss = self.ent_coef * (entropy_mean - self.entropy_target).pow(2)

            loss = policy_loss + self.vf_coef * value_loss + entropy_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl = (old_lp_b - new_lp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                batch_max_ratio = ratio.max().item()
                batch_lr_clipped_frac = (raw_log_ratio.abs() > 5.0).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            total_approx_kl += approx_kl
            total_clip_frac += clip_frac
            total_grad_norm += grad_norm.item()
            total_max_ratio = max(total_max_ratio, batch_max_ratio)
            total_lr_clipped_frac += batch_lr_clipped_frac
            total_entropy_deviation += abs(entropy_mean.item() - self.entropy_target)
            n_batches += 1

            # Per-minibatch KL check: bail out mid-epoch on catastrophic divergence
            if approx_kl > 0.2:
                break

        n_batches = max(n_batches, 1)
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "approx_kl": total_approx_kl / n_batches,
            "clip_fraction": total_clip_frac / n_batches,
            "grad_norm": total_grad_norm / n_batches,
            "max_ratio": total_max_ratio,
            "log_ratio_clipped_frac": total_lr_clipped_frac / n_batches,
            "entropy_deviation": total_entropy_deviation / n_batches,
        }

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint and return the global step to resume from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["global_step"]
        print(f"Resumed from checkpoint: {path} (global_step={global_step})")
        return global_step

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        resume_step: int = 0,
    ) -> None:
        """Main training loop."""
        num_updates = total_timesteps // self.n_steps
        global_step = resume_step
        start_update = resume_step // self.n_steps

        print(f"Device: {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"Training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Environments: {self.n_envs}")
        if resume_step > 0:
            print(f"Resuming from step {resume_step} (update {start_update})")
        print(f"Rollout: {self.n_steps} steps, {self.n_epochs} epochs, batch={self.batch_size}")

        for update in range(start_update + 1, num_updates + 1):
            t0 = time.time()

            # Collect rollout
            buf, last_values = self.collect_rollout()
            advantages, returns = buf.compute_gae(last_values, self.gamma, self.gae_lambda)
            data = buf.to_tensors(self.device, advantages, returns)
            global_step += self.n_steps
            n_episodes = len(self._ep_rewards)

            # PPO epochs with KL early stopping
            epoch_stats: dict[str, float] = {}
            epochs_used = 0
            for epoch_idx in range(self.n_epochs):
                epoch_stats = self.train_epoch(data)
                epochs_used = epoch_idx + 1
                if epoch_stats["approx_kl"] > 0.1:
                    break

            # Step LR scheduler
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_last_lr()[0]

            dt = time.time() - t0
            fps = self.n_steps / dt

            # Log to tensorboard
            self.writer.add_scalar("train/policy_loss", epoch_stats["policy_loss"], global_step)
            self.writer.add_scalar("train/value_loss", epoch_stats["value_loss"], global_step)
            self.writer.add_scalar("train/entropy", epoch_stats["entropy"], global_step)
            self.writer.add_scalar("train/approx_kl", epoch_stats["approx_kl"], global_step)
            self.writer.add_scalar("train/clip_fraction", epoch_stats["clip_fraction"], global_step)
            self.writer.add_scalar("train/learning_rate", current_lr, global_step)
            self.writer.add_scalar("train/epochs_used", epochs_used, global_step)
            self.writer.add_scalar("train/grad_norm", epoch_stats["grad_norm"], global_step)
            self.writer.add_scalar(
                "train/entropy_deviation", epoch_stats["entropy_deviation"], global_step
            )
            self.writer.add_scalar(
                "train/max_ratio", epoch_stats["max_ratio"], global_step
            )
            self.writer.add_scalar(
                "train/log_ratio_clipped_frac",
                epoch_stats["log_ratio_clipped_frac"],
                global_step,
            )
            self.writer.add_scalar("perf/fps", fps, global_step)
            self.writer.add_scalar("balatro/ep_per_rollout", n_episodes, global_step)

            # Log episode metrics
            if self._ep_rewards:
                self.writer.add_scalar(
                    "rollout/ep_rew_mean", np.mean(self._ep_rewards), global_step
                )
                self.writer.add_scalar(
                    "rollout/ep_len_mean", np.mean(self._ep_lengths), global_step
                )
                self.writer.add_scalar(
                    "balatro/mean_ante_reached", np.mean(self._ep_antes), global_step
                )
                self.writer.add_scalar(
                    "balatro/max_ante_reached", np.max(self._ep_antes), global_step
                )
                self.writer.add_scalar(
                    "balatro/mean_rounds_beaten", np.mean(self._ep_rounds), global_step
                )
                self.writer.add_scalar(
                    "balatro/win_rate",
                    np.mean(self._recent_wins) if self._recent_wins else 0.0,
                    global_step,
                )

            # Console output
            if update % log_interval == 0:
                ep_info = ""
                if self._ep_rewards:
                    ep_info = (
                        f" | ep_rew={np.mean(self._ep_rewards):.3f}"
                        f" ep_len={np.mean(self._ep_lengths):.0f}"
                        f" ante={np.mean(self._ep_antes):.1f}"
                    )
                print(
                    f"Update {update}/{num_updates} | "
                    f"step={global_step} | "
                    f"fps={fps:.0f} | "
                    f"lr={current_lr:.2e} "
                    f"epochs={epochs_used}/{self.n_epochs} | "
                    f"ploss={epoch_stats['policy_loss']:.4f} "
                    f"vloss={epoch_stats['value_loss']:.4f} "
                    f"ent={epoch_stats['entropy']:.3f} "
                    f"kl={epoch_stats['approx_kl']:.4f}"
                    f"{ep_info}"
                )

            # Checkpoint
            if update % self.checkpoint_interval == 0:
                ckpt = {
                    "network": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.lr_scheduler.state_dict(),
                    "global_step": global_step,
                    "update": update,
                }
                ckpt_path = Path(self.log_dir) / f"checkpoint_{global_step}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

            # Clear per-rollout episode stats
            self._ep_rewards.clear()
            self._ep_lengths.clear()
            self._ep_antes.clear()
            self._ep_rounds.clear()
            self._ep_wins.clear()

            buf.clear()

        self.vec_env.close()
        self.writer.close()
        print("Training complete.")
