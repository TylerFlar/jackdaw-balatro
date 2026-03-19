"""Transformer core for contextualizing entity representations.

Prepends a learned [CLS] token derived from the global context vector.
Self-attention lets each entity attend to all other entities and to the
global game state.  The output CLS representation is used as the global
state summary for the action heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from jackdaw.env.observation import D_GLOBAL


class TransformerCore(nn.Module):
    """Self-attention across all entities + global context."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        global_dim: int = D_GLOBAL,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.global_encoder = nn.Linear(global_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        global_context: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Transformer over ``[CLS] + entities``.

        Parameters
        ----------
        global_context : Tensor
            ``(B, D_global)`` global feature vector.
        entities : Tensor
            ``(B, N, embed_dim)`` entity embeddings.
        entity_mask : Tensor
            ``(B, N)`` bool — True for real entities, False for padding.

        Returns
        -------
        global_repr : Tensor
            ``(B, embed_dim)`` — CLS output (global state representation).
        entity_reprs : Tensor
            ``(B, N, embed_dim)`` — contextualized entity representations.
        """
        B = global_context.shape[0]
        device = global_context.device

        # CLS token from global context
        cls_token = self.global_encoder(global_context).unsqueeze(1)  # (B, 1, E)

        # Prepend CLS to entity sequence
        sequence = torch.cat([cls_token, entities], dim=1)  # (B, 1+N, E)

        # Padding mask: True = position to IGNORE (PyTorch convention)
        cls_real = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_real = torch.cat([cls_real, entity_mask], dim=1)  # (B, 1+N)
        padding_mask = ~full_real

        output = self.transformer(sequence, src_key_padding_mask=padding_mask)

        global_repr = output[:, 0, :]  # (B, E)
        entity_reprs = output[:, 1:, :]  # (B, N, E)

        return global_repr, entity_reprs
