"""Factored action heads: type -> entity -> card selection + value.

The action type head produces logits over action types.
The entity head uses pointer-network attention to select a target entity.
The card head produces independent Bernoulli logits for hand card selection.
The value head estimates the state value for PPO.

Architecture is fully parametric — driven by :class:`~jackdaw.env.game_spec.GameSpec`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from jackdaw.env.game_spec import GameSpec


class ActionHeads(nn.Module):
    """Factored action heads for the policy.

    Parameters
    ----------
    game_spec:
        Game specification defining action types and their entity/card requirements.
    embed_dim:
        Input/output embedding dimension.
    """

    def __init__(
        self,
        game_spec: GameSpec,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_action_types = game_spec.num_action_types
        self.needs_entity = game_spec.needs_entity_set
        self.needs_cards = game_spec.needs_cards_set

        # Type head: global_repr -> type logits
        self.type_head = nn.Linear(embed_dim, self.num_action_types)

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
        type_mask : ``(B, num_action_types)`` bool

        Returns
        -------
        logits : ``(B, num_action_types)`` with ``-1e9`` for invalid types.
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
