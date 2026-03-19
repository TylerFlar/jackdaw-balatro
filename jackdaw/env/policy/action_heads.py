"""Factored action heads: type -> entity -> card selection + value.

The action type head produces logits over 21 action types.
The entity head uses pointer-network attention to select a target entity.
The card head produces independent Bernoulli logits for hand card selection.
The value head estimates the state value for PPO.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from jackdaw.env.action_space import NUM_ACTION_TYPES

# Action types that require an entity target (indices 8-18).
NEEDS_ENTITY: frozenset[int] = frozenset(range(8, 19))

# Action types that require card targets.
NEEDS_CARDS: frozenset[int] = frozenset({0, 1, 11})


class ActionHeads(nn.Module):
    """Factored action heads for the Balatro policy."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_action_types: int = NUM_ACTION_TYPES,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_action_types = num_action_types

        # Type head: global_repr -> type logits
        self.type_head = nn.Linear(embed_dim, num_action_types)

        # Entity pointer: scaled dot-product between query (from global) and
        # keys (from entity reprs).
        self.entity_query = nn.Linear(embed_dim, embed_dim)
        self.entity_key = nn.Linear(embed_dim, embed_dim)

        # Card selection: per-card Bernoulli scorer
        self.card_scorer = nn.Linear(embed_dim, 1)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._scale = math.sqrt(embed_dim)

    # ------------------------------------------------------------------

    def type_logits(
        self,
        global_repr: torch.Tensor,
        type_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked action type logits.

        Parameters
        ----------
        global_repr : ``(B, E)``
        type_mask : ``(B, 21)`` bool

        Returns
        -------
        logits : ``(B, 21)`` with ``-1e9`` for invalid types.
        """
        logits = self.type_head(global_repr)
        return logits.masked_fill(~type_mask, -1e9)

    def entity_logits(
        self,
        global_repr: torch.Tensor,
        entity_reprs: torch.Tensor,
        pointer_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pointer-network attention logits over entities.

        Parameters
        ----------
        global_repr : ``(B, E)``
        entity_reprs : ``(B, N, E)``
        pointer_mask : ``(B, N)`` bool

        Returns
        -------
        logits : ``(B, N)`` with ``-1e9`` for masked entities.
        """
        query = self.entity_query(global_repr)  # (B, E)
        keys = self.entity_key(entity_reprs)  # (B, N, E)
        scores = (query.unsqueeze(1) * keys).sum(-1) / self._scale  # (B, N)
        return scores.masked_fill(~pointer_mask, -1e9)

    def card_logits(
        self,
        entity_reprs: torch.Tensor,
        n_hand: int,
        card_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-card Bernoulli logits for hand cards.

        Hand cards occupy the first ``n_hand`` positions of ``entity_reprs``.

        Parameters
        ----------
        entity_reprs : ``(B, N_total, E)``
        n_hand : int — max hand size in this batch.
        card_mask : ``(B, n_hand)`` bool.

        Returns
        -------
        logits : ``(B, n_hand)`` with ``-1e9`` for masked cards.
        """
        if n_hand == 0:
            B = entity_reprs.shape[0]
            return torch.zeros(B, 0, device=entity_reprs.device)
        hand_reprs = entity_reprs[:, :n_hand, :]  # (B, n_hand, E)
        logits = self.card_scorer(hand_reprs).squeeze(-1)  # (B, n_hand)
        return logits.masked_fill(~card_mask, -1e9)

    def value(self, global_repr: torch.Tensor) -> torch.Tensor:
        """State value estimate ``(B, 1)``."""
        return self.value_head(global_repr)
