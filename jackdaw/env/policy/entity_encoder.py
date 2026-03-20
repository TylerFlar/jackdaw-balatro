"""Per-type entity encoders projecting heterogeneous features to a common embedding space.

Each entity type gets its own MLP encoder.  Entity types with center_key
identifiers also receive a learned center_key embedding added as a residual.
A learned type embedding distinguishes entity types in the concatenated sequence.

Architecture is fully parametric — driven by :class:`~jackdaw.env.game_spec.GameSpec`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from jackdaw.env.game_spec import GameSpec


class EntityEncoder(nn.Module):
    """Per-type entity encoders that project heterogeneous features to a common embedding dim.

    Each entity type gets a 2-layer MLP.  Types with ``has_catalog_id``
    additionally receive a learned embedding looked up from the normalized
    center_key_id at feature index 0.

    Parameters
    ----------
    game_spec:
        Game specification defining entity types and their feature dimensions.
    embed_dim:
        Output embedding dimension for all entity types.
    center_key_embed_dim:
        Intermediate dimension for catalog ID embeddings before projection.
    """

    def __init__(
        self,
        game_spec: GameSpec,
        embed_dim: int = 128,
        center_key_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._game_spec = game_spec

        feature_dims = [et.feature_dim for et in game_spec.entity_types]
        self._catalog_indices = frozenset(
            i for i, et in enumerate(game_spec.entity_types) if et.has_catalog_id
        )
        max_catalog = max(
            (et.catalog_size for et in game_spec.entity_types if et.has_catalog_id),
            default=0,
        )
        self.num_center_keys = max_catalog

        # Per-type MLPs
        self.type_encoders = nn.ModuleList(
            [self._make_mlp(d, embed_dim) for d in feature_dims]
        )

        # Catalog ID embedding (shared across types with catalog IDs)
        if max_catalog > 0:
            self.center_key_embedding = nn.Embedding(
                max_catalog + 1, center_key_embed_dim, padding_idx=0
            )
            self.center_key_proj = nn.Linear(center_key_embed_dim, embed_dim)

        # Entity type embedding
        self.type_embedding = nn.Embedding(game_spec.num_entity_types, embed_dim)

    # ------------------------------------------------------------------

    @staticmethod
    def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _add_center_key_embed(self, encoded: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Add catalog ID embedding as a residual signal.

        Feature index 0 is the normalized ``catalog_id / num_catalog_ids``.
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
        features_list: list[torch.Tensor],
        masks_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all entity types and concatenate into a single sequence.

        Parameters
        ----------
        features_list:
            Per-type feature tensors, each ``(B, N_type, D_type)`` (zero-padded).
        masks_list:
            Per-type boolean masks, each ``(B, N_type)`` (True = real entity).

        Returns
        -------
        entities : Tensor
            ``(B, N_total, embed_dim)`` concatenated entity embeddings.
        entity_mask : Tensor
            ``(B, N_total)`` boolean mask (True = real).
        """
        encoded_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []

        for i, (feats, mask, encoder) in enumerate(
            zip(features_list, masks_list, self.type_encoders)
        ):
            enc = encoder(feats)
            if i in self._catalog_indices:
                enc = self._add_center_key_embed(enc, feats)
            enc = self._add_type_embed(enc, i)
            encoded_parts.append(enc)
            mask_parts.append(mask)

        entities = torch.cat(encoded_parts, dim=1)
        entity_mask = torch.cat(mask_parts, dim=1)
        return entities, entity_mask
