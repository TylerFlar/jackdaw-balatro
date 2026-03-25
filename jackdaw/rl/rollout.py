"""Rollout buffer with GAE for the factored PPO trainer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Transition:
    """Single environment step."""

    obs: dict[str, np.ndarray]
    action_type: int
    entity_target: int  # -1 if unused
    card_target: np.ndarray  # (max_hand,) bool
    log_prob: float
    value: float
    reward: float
    done: bool
    # Action masks for this step (needed for re-evaluation)
    type_mask: np.ndarray  # (21,) bool
    card_mask: np.ndarray  # (max_hand,) bool padded
    entity_masks: dict[int, np.ndarray]  # action_type -> (max_entity,) bool padded
    min_card_select: int
    max_card_select: int


class RolloutBuffer:
    """Stores transitions and computes GAE advantages.

    Supports both single-env and multi-env collection. For multi-env,
    transitions are stored per-env so GAE is computed independently.
    """

    def __init__(self, n_envs: int = 1) -> None:
        self.n_envs = n_envs
        self._env_transitions: list[list[Transition]] = [[] for _ in range(n_envs)]

    def add(self, t: Transition, env_idx: int = 0) -> None:
        self._env_transitions[env_idx].append(t)

    def __len__(self) -> int:
        return sum(len(ts) for ts in self._env_transitions)

    def _all_transitions(self) -> list[Transition]:
        """Flatten all per-env transitions into a single list."""
        flat: list[Transition] = []
        for ts in self._env_transitions:
            flat.extend(ts)
        return flat

    def compute_gae(
        self,
        last_values: list[float] | float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_values : float or list of floats
            Bootstrap value(s). If a single float, used for all envs.

        Returns (advantages, returns) each of shape (N,).
        """
        if isinstance(last_values, (int, float)):
            last_values = [float(last_values)] * self.n_envs

        all_advantages: list[np.ndarray] = []
        all_values: list[np.ndarray] = []

        for env_idx in range(self.n_envs):
            transitions = self._env_transitions[env_idx]
            N = len(transitions)
            if N == 0:
                continue

            advantages = np.zeros(N, dtype=np.float32)
            last_gae = 0.0
            lv = last_values[env_idx]

            for t in reversed(range(N)):
                tr = transitions[t]
                next_value = lv if t == N - 1 else transitions[t + 1].value
                if tr.done:
                    delta = tr.reward - tr.value
                    last_gae = delta
                else:
                    delta = tr.reward + gamma * next_value - tr.value
                    last_gae = delta + gamma * gae_lambda * last_gae
                advantages[t] = last_gae

            values = np.array([t.value for t in transitions], dtype=np.float32)
            all_advantages.append(advantages)
            all_values.append(values)

        advantages = np.concatenate(all_advantages)
        values = np.concatenate(all_values)
        returns = advantages + values
        return advantages, returns

    def to_tensors(
        self,
        device: torch.device,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, torch.Tensor | dict]:
        """Convert buffer to batched tensors for training.

        Returns a dict of tensors, all with leading dim N.
        """
        transitions = self._all_transitions()
        N = len(transitions)

        # Stack observations
        obs_keys = list(transitions[0].obs.keys())
        obs = {
            k: torch.from_numpy(np.stack([t.obs[k] for t in transitions])).float().to(device)
            for k in obs_keys
        }

        # Actions
        action_type = torch.tensor(
            [t.action_type for t in transitions], dtype=torch.long, device=device
        )
        entity_target = torch.tensor(
            [t.entity_target for t in transitions], dtype=torch.long, device=device
        )
        card_target = torch.from_numpy(
            np.stack([t.card_target for t in transitions])
        ).bool().to(device)

        # Old log probs and values
        old_log_prob = torch.tensor(
            [t.log_prob for t in transitions], dtype=torch.float32, device=device
        )
        old_values = torch.tensor(
            [t.value for t in transitions], dtype=torch.float32, device=device
        )

        # Action masks
        type_mask = torch.from_numpy(
            np.stack([t.type_mask for t in transitions])
        ).bool().to(device)

        card_mask = torch.from_numpy(
            np.stack([t.card_mask for t in transitions])
        ).bool().to(device)

        min_card_select = torch.tensor(
            [t.min_card_select for t in transitions], dtype=torch.long, device=device
        )
        max_card_select = torch.tensor(
            [t.max_card_select for t in transitions], dtype=torch.long, device=device
        )

        # Entity masks: collect all action types that appear, pad to max
        all_emask_keys: set[int] = set()
        for t in transitions:
            all_emask_keys.update(t.entity_masks.keys())
        entity_masks_t: dict[int, torch.Tensor] = {}
        for atype in all_emask_keys:
            arrs = []
            for t in transitions:
                if atype in t.entity_masks:
                    arrs.append(t.entity_masks[atype])
                else:
                    # Get the right shape from any transition that has it
                    ref = next(
                        tr.entity_masks[atype] for tr in transitions if atype in tr.entity_masks
                    )
                    arrs.append(np.zeros_like(ref))
            entity_masks_t[atype] = torch.from_numpy(np.stack(arrs)).bool().to(device)

        action_masks_dict = {
            "type_mask": type_mask,
            "card_mask": card_mask,
            "entity_masks": entity_masks_t,
            "min_card_select": min_card_select,
            "max_card_select": max_card_select,
        }

        return {
            "obs": obs,
            "action_type": action_type,
            "entity_target": entity_target,
            "card_target": card_target,
            "old_log_prob": old_log_prob,
            "old_values": old_values,
            "advantages": torch.from_numpy(advantages).to(device),
            "returns": torch.from_numpy(returns).to(device),
            "action_masks": action_masks_dict,
        }

    def clear(self) -> None:
        for ts in self._env_transitions:
            ts.clear()
