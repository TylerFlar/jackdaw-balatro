"""CleanRL-style PPO trainer for the factored Balatro policy."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_spec import FactoredAction, GameActionMask
from jackdaw.rl.env_wrapper import FactoredBalatroEnv
from jackdaw.rl.network import (
    ENTITY_MAX_COUNTS,
    ENTITY_NAMES,
    FactoredPolicy,
    NEEDS_CARDS,
    NEEDS_ENTITY,
)
from jackdaw.rl.rollout import RolloutBuffer, Transition

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


def _obs_to_device(
    obs: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    """Convert single obs dict to batched (B=1) tensors."""
    return {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in obs.items()}


def _masks_to_device(
    type_mask: np.ndarray,
    card_mask: np.ndarray,
    entity_masks: dict[int, np.ndarray],
    min_card_select: int,
    max_card_select: int,
    device: torch.device,
) -> dict[str, Any]:
    """Convert single-step masks to batched (B=1) tensors."""
    em = {
        atype: torch.from_numpy(m).bool().unsqueeze(0).to(device)
        for atype, m in entity_masks.items()
    }
    return {
        "type_mask": torch.from_numpy(type_mask).bool().unsqueeze(0).to(device),
        "card_mask": torch.from_numpy(card_mask).bool().unsqueeze(0).to(device),
        "entity_masks": em,
        "min_card_select": torch.tensor([min_card_select], dtype=torch.long, device=device),
        "max_card_select": torch.tensor([max_card_select], dtype=torch.long, device=device),
    }


class BalatroTrainer:
    """PPO trainer for the factored Balatro policy.

    Parameters
    ----------
    env : FactoredBalatroEnv
    network : FactoredPolicy
    lr : learning rate
    gamma, gae_lambda : GAE parameters
    clip_range : PPO clipping epsilon
    ent_coef : entropy bonus coefficient
    vf_coef : value loss coefficient
    n_steps : rollout length per update
    n_epochs : PPO epochs per rollout
    batch_size : minibatch size for PPO updates
    max_grad_norm : gradient clipping
    device : "auto", "cpu", or "cuda"
    log_dir : tensorboard log directory
    """

    def __init__(
        self,
        env: FactoredBalatroEnv,
        network: FactoredPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.15,
        ent_coef: float = 0.05,
        vf_coef: float = 0.5,
        n_steps: int = 4096,
        n_epochs: int = 10,
        batch_size: int = 512,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        log_dir: str = "runs/balatro_factored",
    ) -> None:
        self.env = env
        self.network = network
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.writer = SummaryWriter(log_dir)

        # Episode tracking
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_antes: list[int] = []
        self._ep_rounds: list[int] = []
        self._ep_wins: list[bool] = []
        # Rolling stats
        self._recent_wins: deque[bool] = deque(maxlen=100)

        # Env state (persists across rollouts)
        self._obs: dict[str, np.ndarray] | None = None
        self._mask: GameActionMask | None = None
        self._ep_reward_accum: float = 0.0
        self._ep_len_accum: int = 0

    def _reset_env(self) -> None:
        obs, mask, info = self.env.reset()
        self._obs = obs
        self._mask = mask
        self._ep_reward_accum = 0.0
        self._ep_len_accum = 0

    def collect_rollout(self) -> RolloutBuffer:
        """Run n_steps in the environment, collecting transitions."""
        buf = RolloutBuffer()
        self.network.eval()

        if self._obs is None:
            self._reset_env()

        for _ in range(self.n_steps):
            obs = self._obs
            assert obs is not None and self._mask is not None

            type_mask, card_mask, entity_masks, min_cs, max_cs = _masks_to_numpy(self._mask)

            # Forward pass (no grad)
            obs_t = _obs_to_device(obs, self.device)
            masks_t = _masks_to_device(type_mask, card_mask, entity_masks, min_cs, max_cs, self.device)

            with torch.no_grad():
                out = self.network(obs_t, masks_t)

            action_type = out["action_type"].item()
            entity_target = out["entity_target"].item()
            card_target_arr = out["card_target"][0].cpu().numpy()  # (max_hand,) bool
            log_prob = out["log_prob"].item()
            value = out["value"].item()

            # Convert to FactoredAction
            ct: tuple[int, ...] | None = None
            et: int | None = None

            if action_type in NEEDS_ENTITY and entity_target >= 0:
                et = entity_target
            if action_type in NEEDS_CARDS:
                selected = np.nonzero(card_target_arr)[0]
                if len(selected) > 0:
                    ct = tuple(int(i) for i in selected)

            fa = FactoredAction(action_type=action_type, card_target=ct, entity_target=et)

            # Step environment
            next_obs, reward, terminated, truncated, next_mask, info = self.env.step(fa)
            done = terminated or truncated

            buf.add(Transition(
                obs=obs,
                action_type=action_type,
                entity_target=entity_target,
                card_target=card_target_arr,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
                type_mask=type_mask,
                card_mask=card_mask,
                entity_masks=entity_masks,
                min_card_select=min_cs,
                max_card_select=max_cs,
            ))

            self._ep_reward_accum += reward
            self._ep_len_accum += 1

            if done:
                self._ep_rewards.append(self._ep_reward_accum)
                self._ep_lengths.append(self._ep_len_accum)
                self._ep_antes.append(info.get("balatro/ante_reached", 1))
                self._ep_rounds.append(info.get("balatro/rounds_beaten", 0))
                won = info.get("balatro/won", False)
                self._ep_wins.append(won)
                self._recent_wins.append(won)
                self._reset_env()
            else:
                self._obs = next_obs
                self._mask = next_mask

        # Bootstrap value for the last step
        with torch.no_grad():
            obs_t = _obs_to_device(self._obs, self.device)
            # Just need the value, but we need masks too for the forward pass
            type_mask, card_mask, entity_masks, min_cs, max_cs = _masks_to_numpy(self._mask)
            masks_t = _masks_to_device(type_mask, card_mask, entity_masks, min_cs, max_cs, self.device)
            # Use _encode + value_head directly to avoid sampling
            state, _ = self.network._encode(obs_t)
            last_value = self.network.value_head(state).item()

        return buf, last_value

    def train_epoch(
        self, data: dict[str, Any]
    ) -> dict[str, float]:
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
            new_lp, new_val, entropy = self.network.evaluate(
                obs_b, masks_b, at_b, et_b, ct_b
            )

            # NaN guard — skip batch if network produced NaN
            if torch.isnan(new_lp).any() or torch.isnan(new_val).any():
                continue

            # PPO policy loss — clamp ratio to prevent exp() overflow
            log_ratio = (new_lp - old_lp_b).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss
            value_loss = (new_val - ret_b).pow(2).mean()

            # Entropy bonus
            entropy_mean = entropy.mean()

            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl = (old_lp_b - new_lp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            total_approx_kl += approx_kl
            total_clip_frac += clip_frac
            n_batches += 1

        n_batches = max(n_batches, 1)
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "approx_kl": total_approx_kl / n_batches,
            "clip_fraction": total_clip_frac / n_batches,
        }

    def train(self, total_timesteps: int, log_interval: int = 1) -> None:
        """Main training loop."""
        num_updates = total_timesteps // self.n_steps
        global_step = 0

        print(f"Device: {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"Training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Rollout: {self.n_steps} steps, {self.n_epochs} epochs, batch={self.batch_size}")

        for update in range(1, num_updates + 1):
            t0 = time.time()

            # Collect rollout
            buf, last_value = self.collect_rollout()
            advantages, returns = buf.compute_gae(last_value, self.gamma, self.gae_lambda)
            data = buf.to_tensors(self.device, advantages, returns)
            global_step += self.n_steps

            # PPO epochs
            epoch_stats: dict[str, float] = {}
            for _ in range(self.n_epochs):
                epoch_stats = self.train_epoch(data)

            dt = time.time() - t0
            fps = self.n_steps / dt

            # Log to tensorboard
            self.writer.add_scalar("train/policy_loss", epoch_stats["policy_loss"], global_step)
            self.writer.add_scalar("train/value_loss", epoch_stats["value_loss"], global_step)
            self.writer.add_scalar("train/entropy", epoch_stats["entropy"], global_step)
            self.writer.add_scalar("train/approx_kl", epoch_stats["approx_kl"], global_step)
            self.writer.add_scalar("train/clip_fraction", epoch_stats["clip_fraction"], global_step)
            self.writer.add_scalar("perf/fps", fps, global_step)

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
                    f"ploss={epoch_stats['policy_loss']:.4f} "
                    f"vloss={epoch_stats['value_loss']:.4f} "
                    f"ent={epoch_stats['entropy']:.3f} "
                    f"kl={epoch_stats['approx_kl']:.4f}"
                    f"{ep_info}"
                )

            # Clear per-rollout episode stats
            self._ep_rewards.clear()
            self._ep_lengths.clear()
            self._ep_antes.clear()
            self._ep_rounds.clear()
            self._ep_wins.clear()

            buf.clear()

        self.writer.close()
        print("Training complete.")
