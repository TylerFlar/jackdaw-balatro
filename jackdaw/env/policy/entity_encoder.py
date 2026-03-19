"""Per-type entity encoders projecting heterogeneous features to a common embedding space.

Each entity type (hand cards, jokers, consumables, shop items, pack cards) gets
its own MLP encoder.  Entity types with center_key identifiers (jokers,
consumables, shop items) also receive a learned center_key embedding added as a
residual.  A learned type embedding distinguishes entity types in the
concatenated sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    NUM_CENTER_KEYS,
)

# Entity type indices used for the type embedding table.
HAND_TYPE = 0
JOKER_TYPE = 1
CONSUMABLE_TYPE = 2
SHOP_TYPE = 3
PACK_TYPE = 4
NUM_ENTITY_TYPES = 5


class EntityEncoder(nn.Module):
    """Per-type entity encoders that project heterogeneous features to a common embedding dim.

    Each entity type gets a 2-layer MLP.  Types carrying a ``center_key``
    (jokers, consumables, shop items) additionally receive a learned embedding
    looked up from the normalized center_key_id at feature index 0.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_center_keys: int = NUM_CENTER_KEYS,
        center_key_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_center_keys = num_center_keys

        # Per-type MLPs
        self.card_encoder = self._make_mlp(D_PLAYING_CARD, embed_dim)
        self.joker_encoder = self._make_mlp(D_JOKER, embed_dim)
        self.consumable_encoder = self._make_mlp(D_CONSUMABLE, embed_dim)
        self.shop_encoder = self._make_mlp(D_SHOP, embed_dim)
        self.pack_encoder = self._make_mlp(D_PLAYING_CARD, embed_dim)

        # Center key embedding (shared across jokers, consumables, shop items)
        self.center_key_embedding = nn.Embedding(
            num_center_keys + 1, center_key_embed_dim, padding_idx=0
        )
        self.center_key_proj = nn.Linear(center_key_embed_dim, embed_dim)

        # Entity type embedding (5 types)
        self.type_embedding = nn.Embedding(NUM_ENTITY_TYPES, embed_dim)

    # ------------------------------------------------------------------

    @staticmethod
    def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _add_center_key_embed(self, encoded: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Add center_key embedding as a residual signal.

        Feature index 0 is the normalized ``center_key_id / NUM_CENTER_KEYS``.
        We recover the integer id by rounding.
        """
        if features.shape[1] == 0:
            return encoded
        ck_ids = (
            (features[:, :, 0] * self.num_center_keys).round().long().clamp(0, self.num_center_keys)
        )
        ck_embed = self.center_key_proj(self.center_key_embedding(ck_ids))
        return encoded + ck_embed

    def _add_type_embed(self, encoded: torch.Tensor, type_id: int) -> torch.Tensor:
        if encoded.shape[1] == 0:
            return encoded
        B, N, _ = encoded.shape
        ids = torch.full((B, N), type_id, dtype=torch.long, device=encoded.device)
        return encoded + self.type_embedding(ids)

    # ------------------------------------------------------------------

    def forward(
        self,
        hand_cards: torch.Tensor,
        jokers: torch.Tensor,
        consumables: torch.Tensor,
        shop_cards: torch.Tensor,
        pack_cards: torch.Tensor,
        hand_mask: torch.Tensor,
        joker_mask: torch.Tensor,
        cons_mask: torch.Tensor,
        shop_mask: torch.Tensor,
        pack_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all entity types and concatenate into a single sequence.

        All feature inputs have shape ``(B, N_type, D_type)`` (zero-padded).
        All mask inputs have shape ``(B, N_type)`` (bool, True = real entity).

        Returns
        -------
        entities : Tensor
            ``(B, N_total, embed_dim)`` concatenated entity embeddings.
        entity_mask : Tensor
            ``(B, N_total)`` boolean mask (True = real).
        """
        # Encode each entity type through its MLP
        h_enc = self.card_encoder(hand_cards)
        j_enc = self.joker_encoder(jokers)
        c_enc = self.consumable_encoder(consumables)
        s_enc = self.shop_encoder(shop_cards)
        p_enc = self.pack_encoder(pack_cards)

        # Add center_key residual embeddings
        j_enc = self._add_center_key_embed(j_enc, jokers)
        c_enc = self._add_center_key_embed(c_enc, consumables)
        s_enc = self._add_center_key_embed(s_enc, shop_cards)

        # Add entity type embeddings
        h_enc = self._add_type_embed(h_enc, HAND_TYPE)
        j_enc = self._add_type_embed(j_enc, JOKER_TYPE)
        c_enc = self._add_type_embed(c_enc, CONSUMABLE_TYPE)
        s_enc = self._add_type_embed(s_enc, SHOP_TYPE)
        p_enc = self._add_type_embed(p_enc, PACK_TYPE)

        # Concatenate: [hand | joker | consumable | shop | pack]
        entities = torch.cat([h_enc, j_enc, c_enc, s_enc, p_enc], dim=1)
        entity_mask = torch.cat([hand_mask, joker_mask, cons_mask, shop_mask, pack_mask], dim=1)

        return entities, entity_mask
