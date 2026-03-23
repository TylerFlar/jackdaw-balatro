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
    """Stores transitions and computes GAE advantages."""

    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    def add(self, t: Transition) -> None:
        self._transitions.append(t)

    def __len__(self) -> int:
        return len(self._transitions)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        Returns (advantages, returns) each of shape (N,).
        """
        N = len(self._transitions)
        advantages = np.zeros(N, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(N)):
            tr = self._transitions[t]
            next_value = last_value if t == N - 1 else self._transitions[t + 1].value
            next_done = False if t == N - 1 else self._transitions[t + 1].done
            # If current step is terminal, next_value should be 0
            if tr.done:
                delta = tr.reward - tr.value
                last_gae = delta
            else:
                delta = tr.reward + gamma * next_value - tr.value
                next_nonterminal = 0.0 if (t < N - 1 and self._transitions[t + 1].done) else 1.0
                # Actually: the mask should be based on whether THIS step is done
                # If this step is done, we already handled it above.
                # If not done, propagate GAE.
                last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        values = np.array([t.value for t in self._transitions], dtype=np.float32)
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
        N = len(self._transitions)

        # Stack observations
        obs_keys = list(self._transitions[0].obs.keys())
        obs = {
            k: torch.from_numpy(np.stack([t.obs[k] for t in self._transitions])).float().to(device)
            for k in obs_keys
        }

        # Actions
        action_type = torch.tensor(
            [t.action_type for t in self._transitions], dtype=torch.long, device=device
        )
        entity_target = torch.tensor(
            [t.entity_target for t in self._transitions], dtype=torch.long, device=device
        )
        card_target = torch.from_numpy(
            np.stack([t.card_target for t in self._transitions])
        ).bool().to(device)

        # Old log probs and values
        old_log_prob = torch.tensor(
            [t.log_prob for t in self._transitions], dtype=torch.float32, device=device
        )
        old_values = torch.tensor(
            [t.value for t in self._transitions], dtype=torch.float32, device=device
        )

        # Action masks
        type_mask = torch.from_numpy(
            np.stack([t.type_mask for t in self._transitions])
        ).bool().to(device)

        card_mask = torch.from_numpy(
            np.stack([t.card_mask for t in self._transitions])
        ).bool().to(device)

        min_card_select = torch.tensor(
            [t.min_card_select for t in self._transitions], dtype=torch.long, device=device
        )
        max_card_select = torch.tensor(
            [t.max_card_select for t in self._transitions], dtype=torch.long, device=device
        )

        # Entity masks: collect all action types that appear, pad to max
        all_emask_keys: set[int] = set()
        for t in self._transitions:
            all_emask_keys.update(t.entity_masks.keys())
        entity_masks_t: dict[int, torch.Tensor] = {}
        for atype in all_emask_keys:
            arrs = []
            for t in self._transitions:
                if atype in t.entity_masks:
                    arrs.append(t.entity_masks[atype])
                else:
                    # Get the right shape from any transition that has it
                    ref = next(
                        tr.entity_masks[atype] for tr in self._transitions if atype in tr.entity_masks
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
        self._transitions.clear()
