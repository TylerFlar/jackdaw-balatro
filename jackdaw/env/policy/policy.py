"""Game-agnostic policy network combining encoder, transformer, and action heads.

Also provides :func:`collate_policy_inputs` for batching variable-length
observations into padded tensors suitable for the network.

All architecture dimensions are driven by :class:`~jackdaw.env.game_spec.GameSpec`.
No game-specific imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from jackdaw.env.game_spec import (
    FactoredAction,
    GameActionMask,
    GameObservation,
    GameSpec,
)
from jackdaw.env.policy.action_heads import ActionHeads
from jackdaw.env.policy.entity_encoder import EntityEncoder
from jackdaw.env.policy.transformer import TransformerCore

# ---------------------------------------------------------------------------
# PolicyInput — wraps observation + mask for collation
# ---------------------------------------------------------------------------


@dataclass
class PolicyInput:
    """Single-step input for the policy network.

    Parameters
    ----------
    obs:
        Game-agnostic observation with global context + entity dict.
    action_mask:
        Game-agnostic action mask.
    shop_splits:
        ``(n_shop_cards, n_vouchers, n_boosters)`` — needed to correctly
        place entity pointer masks for shop sub-types.  Defaults to
        treating all entities as a single range per type.
    """

    obs: GameObservation
    action_mask: GameActionMask
    shop_splits: tuple[int, int, int] = (0, 0, 0)


# ---------------------------------------------------------------------------
# Collation utilities
# ---------------------------------------------------------------------------


def _pad_2d(arrays: list[np.ndarray], max_n: int, d: int) -> torch.Tensor:
    """Pad and stack variable-length 2D arrays into ``(B, max_n, d)``."""
    B = len(arrays)
    mn = max(max_n, 0)
    buf = np.zeros((B, mn, d), dtype=np.float32)
    for i, arr in enumerate(arrays):
        n = arr.shape[0]
        if n > 0:
            buf[i, :n] = arr
    return torch.from_numpy(buf.copy())


def _make_mask(arrays: list[np.ndarray], max_n: int) -> torch.Tensor:
    """Create ``(B, max_n)`` bool mask from variable-length arrays."""
    B = len(arrays)
    mn = max(max_n, 0)
    buf = np.zeros((B, mn), dtype=np.bool_)
    for i, arr in enumerate(arrays):
        buf[i, : arr.shape[0]] = True
    return torch.from_numpy(buf.copy())


def collate_policy_inputs(
    inputs: list[PolicyInput],
    game_spec: GameSpec,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Batch a list of :class:`PolicyInput` into padded tensors.

    Parameters
    ----------
    inputs:
        List of single-step policy inputs.
    game_spec:
        Game specification driving entity types and action count.
    device:
        Target device for all tensors.  If *None*, tensors stay on CPU.

    Returns a dict that can be passed directly to
    :meth:`BalatroPolicy.forward`.
    """
    B = len(inputs)
    n_entity_types = game_spec.num_entity_types
    entity_names = [et.name for et in game_spec.entity_types]
    feature_dims = [et.feature_dim for et in game_spec.entity_types]

    # Extract per-type entity arrays: entity_arrays[type_idx][batch_idx]
    entity_arrays: list[list[np.ndarray]] = [[] for _ in range(n_entity_types)]
    for inp in inputs:
        for t, name in enumerate(entity_names):
            entity_arrays[t].append(inp.obs.entities[name])

    # Max sizes per entity type
    max_sizes = [
        max(arr.shape[0] for arr in entity_arrays[t]) for t in range(n_entity_types)
    ]
    N_total = sum(max_sizes)

    # Padded feature tensors and boolean masks (as lists)
    entity_features: list[torch.Tensor] = []
    entity_masks: list[torch.Tensor] = []
    for t in range(n_entity_types):
        entity_features.append(_pad_2d(entity_arrays[t], max_sizes[t], feature_dims[t]))
        entity_masks.append(_make_mask(entity_arrays[t], max_sizes[t]))

    # Global context
    global_ctx = torch.from_numpy(np.stack([inp.obs.global_context for inp in inputs]))

    # Type mask (B, num_action_types)
    type_mask = torch.from_numpy(np.stack([inp.action_mask.type_mask for inp in inputs]).copy())

    # Card mask (B, max_hand) — hand cards are entity type 0
    max_hand = max_sizes[0]
    mh = max(max_hand, 0)
    card_mask_np = np.zeros((B, mh), dtype=np.bool_)
    for i, inp in enumerate(inputs):
        n = len(inp.action_mask.card_mask)
        if n > 0:
            card_mask_np[i, :n] = inp.action_mask.card_mask
    card_mask = torch.from_numpy(card_mask_np.copy())

    # Card selection limits
    max_card_select = torch.tensor(
        [inp.action_mask.max_card_select for inp in inputs], dtype=torch.long
    )
    min_card_select = torch.tensor(
        [inp.action_mask.min_card_select for inp in inputs], dtype=torch.long
    )

    # Entity offsets in the concatenated sequence (cumulative sum of max sizes)
    cumulative = [0]
    for s in max_sizes[:-1]:
        cumulative.append(cumulative[-1] + s)
    entity_offsets = torch.tensor(cumulative, dtype=torch.long)

    # Shop splits per item
    shop_splits = torch.tensor([inp.shop_splits for inp in inputs], dtype=torch.long)

    # Pointer masks: (B, num_action_types, N_total)
    pointer_masks = torch.zeros(B, game_spec.num_action_types, max(N_total, 0), dtype=torch.bool)
    _build_pointer_masks(pointer_masks, inputs, game_spec, entity_offsets)

    batch: dict[str, Any] = {
        "global_context": global_ctx,
        "entity_features": entity_features,
        "entity_masks": entity_masks,
        "type_mask": type_mask,
        "card_mask": card_mask,
        "pointer_masks": pointer_masks,
        "entity_offsets": entity_offsets,
        "shop_splits": shop_splits,
        "max_card_select": max_card_select,
        "min_card_select": min_card_select,
        "max_hand": max_hand,
    }

    if device is not None:
        batch = batch_to_device(batch, device)

    return batch


def _build_pointer_masks(
    pointer_masks: torch.Tensor,
    inputs: list[PolicyInput],
    game_spec: GameSpec,
    entity_offsets: torch.Tensor,
) -> None:
    """Build pointer masks using GameSpec entity_type_index mapping."""
    offsets = entity_offsets.tolist()

    for b, inp in enumerate(inputs):
        n_sc, n_sv, n_sb = inp.shop_splits
        for at_int, emask in inp.action_mask.entity_masks.items():
            at = int(at_int)
            if at >= game_spec.num_action_types:
                continue
            mask_t = torch.from_numpy(emask)
            n = len(emask)
            at_spec = game_spec.action_types[at]
            if not at_spec.needs_entity_target or at_spec.entity_type_index < 0:
                continue
            etype_idx = at_spec.entity_type_index
            base = offsets[etype_idx]

            # Shop sub-splitting: multiple action types target the same
            # entity type at different sub-offsets.
            etype_name = game_spec.entity_types[etype_idx].name
            if etype_name == "shop_item":
                if at_spec.name == "redeem_voucher":
                    base += n_sc
                elif at_spec.name == "open_booster":
                    base += n_sc + n_sv

            pointer_masks[b, at, base : base + n] = mask_t


def batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move all tensors in a collated batch to the given device."""
    result: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            result[k] = [t.to(device) for t in v]
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Pointer index <-> entity_target conversion
# ---------------------------------------------------------------------------


def _pointer_to_entity_target(
    ptr_idx: int,
    action_type: int,
    game_spec: GameSpec,
    entity_offsets: torch.Tensor,
    shop_splits: torch.Tensor,
) -> int:
    """Convert a position in the concatenated entity sequence to a per-type index."""
    at_spec = game_spec.action_types[action_type]
    etype_idx = at_spec.entity_type_index
    offsets = entity_offsets.tolist()
    base = offsets[etype_idx]
    result = ptr_idx - base

    etype_name = game_spec.entity_types[etype_idx].name
    if etype_name == "shop_item":
        if at_spec.name == "redeem_voucher":
            result -= int(shop_splits[0].item())
        elif at_spec.name == "open_booster":
            result -= int(shop_splits[0].item()) + int(shop_splits[1].item())
    return result


def _entity_target_to_pointer(
    entity_target: int,
    action_type: int,
    game_spec: GameSpec,
    entity_offsets: torch.Tensor,
    shop_splits: torch.Tensor,
) -> int:
    """Convert a per-type entity index to a position in the concatenated sequence."""
    at_spec = game_spec.action_types[action_type]
    etype_idx = at_spec.entity_type_index
    offsets = entity_offsets.tolist()
    base = offsets[etype_idx]
    result = entity_target + base

    etype_name = game_spec.entity_types[etype_idx].name
    if etype_name == "shop_item":
        if at_spec.name == "redeem_voucher":
            result += int(shop_splits[0].item())
        elif at_spec.name == "open_booster":
            result += int(shop_splits[0].item()) + int(shop_splits[1].item())
    return result


# ---------------------------------------------------------------------------
# Card sampling with constraints
# ---------------------------------------------------------------------------


def _sample_cards(
    logits: torch.Tensor,
    mask: torch.Tensor,
    min_cards: int,
    max_cards: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample card targets from Bernoulli logits with min/max constraints.

    Returns ``(selected_mask, log_prob)``.
    """
    probs = torch.sigmoid(logits) * mask.float()
    selected = (torch.bernoulli(probs) * mask.float()).bool()
    n_selected = int(selected.sum().item())

    if n_selected < min_cards:
        scores = probs.clone()
        scores[selected | ~mask] = -1.0
        available = int((scores > -1.0).sum().item())
        needed = min(min_cards - n_selected, available)
        if needed > 0:
            _, top_idx = scores.topk(needed)
            selected[top_idx] = True
    elif n_selected > max_cards:
        scores = probs.clone()
        scores[~selected] = -1.0
        _, top_idx = scores.topk(max_cards)
        new_selected = torch.zeros_like(selected)
        new_selected[top_idx] = True
        selected = new_selected

    selected_f = selected.float()
    log_prob = F.logsigmoid(logits) * selected_f + F.logsigmoid(-logits) * (1.0 - selected_f)
    log_prob = (log_prob * mask.float()).sum()

    return selected, log_prob


# ---------------------------------------------------------------------------
# BalatroPolicy
# ---------------------------------------------------------------------------


class BalatroPolicy(nn.Module):
    """Full policy network combining encoder, transformer, and action heads.

    Designed for PPO training with variable-length entity observations.
    All architecture dimensions are derived from :class:`GameSpec`.

    Parameters
    ----------
    game_spec:
        Game specification driving all architecture dimensions.
    embed_dim:
        Embedding dimension for all sub-modules.
    num_heads:
        Number of attention heads in the transformer.
    num_layers:
        Number of transformer layers.
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        game_spec: GameSpec,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._game_spec = game_spec
        self.embed_dim = embed_dim

        self.entity_encoder = EntityEncoder(game_spec, embed_dim=embed_dim)
        self.transformer = TransformerCore(
            global_dim=game_spec.global_feature_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.action_heads = ActionHeads(game_spec, embed_dim=embed_dim)

    @property
    def game_spec(self) -> GameSpec:
        return self._game_spec

    @property
    def device(self) -> torch.device:
        """Device of the model parameters (convenience for collation)."""
        return next(self.parameters()).device

    def prepare_batch(self, inputs: list[PolicyInput]) -> dict[str, Any]:
        """Collate inputs and move to this model's device."""
        return collate_policy_inputs(inputs, self._game_spec, device=self.device)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Full forward pass through encoder, transformer, and heads.

        Parameters
        ----------
        batch:
            Output of :func:`collate_policy_inputs`.

        Returns
        -------
        dict with keys:
            ``type_logits`` ``(B, num_action_types)``,
            ``card_logits`` ``(B, N_hand)``,
            ``value`` ``(B, 1)``, ``global_repr`` ``(B, E)``,
            ``entity_reprs`` ``(B, N_total, E)``.
        """
        # Encode entities
        entities, entity_mask = self.entity_encoder(
            batch["entity_features"], batch["entity_masks"],
        )

        # Contextualize via transformer
        global_repr, entity_reprs = self.transformer(batch["global_context"], entities, entity_mask)

        # Action heads
        type_logits = self.action_heads.type_logits(global_repr, batch["type_mask"])
        max_hand: int = batch["max_hand"]
        card_logits = self.action_heads.card_logits(entity_reprs, max_hand, batch["card_mask"])
        value = self.action_heads.value(global_repr)

        return {
            "type_logits": type_logits,
            "card_logits": card_logits,
            "value": value,
            "global_repr": global_repr,
            "entity_reprs": entity_reprs,
        }

    # ------------------------------------------------------------------
    # Autoregressive action sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_action(
        self, batch: dict[str, Any]
    ) -> tuple[list[FactoredAction], dict[str, torch.Tensor], torch.Tensor]:
        """Sample actions autoregressively: type -> entity -> cards.

        Returns ``(actions, log_probs_dict, values)`` where log_probs_dict has
        keys ``"type"``, ``"entity"``, ``"card"``, ``"total"``, and values is
        ``(B,)`` state value estimates.
        """
        needs_entity = self.action_heads.needs_entity
        needs_cards = self.action_heads.needs_cards
        game_spec = self._game_spec

        self.eval()
        out = self.forward(batch)
        B = out["type_logits"].shape[0]
        device = out["type_logits"].device

        # Stage 1: batch type sampling
        type_dist = Categorical(logits=out["type_logits"])
        sampled_types = type_dist.sample()
        type_lp = type_dist.log_prob(sampled_types)
        at_list = sampled_types.tolist()

        # Stage 2: entity targets — grouped by action type
        entity_lp = torch.zeros(B, device=device)
        entity_targets: list[int | None] = [None] * B

        entity_groups: dict[int, list[int]] = {}
        for b, at in enumerate(at_list):
            if at in needs_entity:
                pmask = batch["pointer_masks"][b, at]
                if pmask.any():
                    entity_groups.setdefault(at, []).append(b)

        for at, indices in entity_groups.items():
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            g_repr = out["global_repr"][idx_t]
            e_repr = out["entity_reprs"][idx_t]
            pmask = batch["pointer_masks"][idx_t, at]
            e_lgts = self.action_heads.entity_logits(g_repr, e_repr, pmask)
            e_dist = Categorical(logits=e_lgts)
            e_idx = e_dist.sample()
            entity_lp[idx_t] = e_dist.log_prob(e_idx)

            e_idx_list = e_idx.tolist()
            for i, b in enumerate(indices):
                entity_targets[b] = _pointer_to_entity_target(
                    e_idx_list[i], at, game_spec,
                    batch["entity_offsets"],
                    batch["shop_splits"][b],
                )

        # Stage 3: card targets
        card_lp = torch.zeros(B, device=device)
        card_targets: list[tuple[int, ...] | None] = [None] * B

        for b, at in enumerate(at_list):
            if at in needs_cards and batch["card_mask"][b].any():
                min_c = int(batch["min_card_select"][b].item())
                max_c = int(batch["max_card_select"][b].item())
                selected, c_lp = _sample_cards(
                    out["card_logits"][b],
                    batch["card_mask"][b],
                    min_c,
                    max_c,
                )
                card_lp[b] = c_lp
                indices = selected.nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    card_targets[b] = tuple(int(i) for i in indices)

        actions = [
            FactoredAction(
                action_type=at,
                card_target=card_targets[b],
                entity_target=entity_targets[b],
            )
            for b, at in enumerate(at_list)
        ]

        total = type_lp + entity_lp + card_lp
        log_probs = {
            "type": type_lp,
            "entity": entity_lp,
            "card": card_lp,
            "total": total,
        }
        values = out["value"].squeeze(-1)
        return actions, log_probs, values

    # ------------------------------------------------------------------
    # Action evaluation (for PPO loss)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        batch: dict[str, Any],
        actions: list[FactoredAction],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities and entropy for given actions.

        Parameters
        ----------
        batch:
            Collated batch from :func:`collate_policy_inputs`.
        actions:
            List of ``B`` actions to evaluate.

        Returns
        -------
        log_probs : ``(B,)``
        entropy : ``(B,)``
        values : ``(B,)``
        """
        needs_entity = self.action_heads.needs_entity
        needs_cards = self.action_heads.needs_cards
        game_spec = self._game_spec

        out = self.forward(batch)
        B = out["type_logits"].shape[0]
        device = out["type_logits"].device

        # -- Vectorised type log-prob + entropy --
        action_types = torch.tensor(
            [fa.action_type for fa in actions], dtype=torch.long, device=device
        )
        type_dist = Categorical(logits=out["type_logits"])
        type_lp = type_dist.log_prob(action_types)
        type_ent = type_dist.entropy()

        # -- Entity log-probs: grouped by action type --
        entity_lp = torch.zeros(B, device=device)
        entity_ent = torch.zeros(B, device=device)

        entity_groups: dict[int, list[tuple[int, int]]] = {}
        for b, fa in enumerate(actions):
            if fa.action_type in needs_entity and fa.entity_target is not None:
                ptr = _entity_target_to_pointer(
                    fa.entity_target, fa.action_type, game_spec,
                    batch["entity_offsets"],
                    batch["shop_splits"][b],
                )
                entity_groups.setdefault(fa.action_type, []).append((b, ptr))

        for at, items in entity_groups.items():
            indices = [b for b, _ in items]
            ptrs = torch.tensor([p for _, p in items], dtype=torch.long, device=device)
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            g_repr = out["global_repr"][idx_t]
            e_repr = out["entity_reprs"][idx_t]
            pmask = batch["pointer_masks"][idx_t, at]
            e_lgts = self.action_heads.entity_logits(g_repr, e_repr, pmask)
            e_dist = Categorical(logits=e_lgts)
            entity_lp[idx_t] = e_dist.log_prob(ptrs)
            entity_ent[idx_t] = e_dist.entropy()

        # -- Card log-probs + entropy: fully vectorised --
        max_hand = out["card_logits"].shape[1]
        selected = torch.zeros(B, max_hand, device=device)
        needs_cards_flag = torch.zeros(B, device=device)

        for b, fa in enumerate(actions):
            if fa.action_type in needs_cards and fa.card_target:
                needs_cards_flag[b] = 1.0
                for idx in fa.card_target:
                    if idx < max_hand:
                        selected[b, idx] = 1.0

        c_lgts = out["card_logits"]
        c_mask_f = batch["card_mask"].float()

        lp_pos = F.logsigmoid(c_lgts)
        lp_neg = F.logsigmoid(-c_lgts)
        card_lp_all = (lp_pos * selected + lp_neg * (1.0 - selected)) * c_mask_f
        card_lp = card_lp_all.sum(dim=1) * needs_cards_flag

        p = torch.sigmoid(c_lgts)
        per_card_ent = lp_pos * p + lp_neg * (1.0 - p)
        card_ent = -(per_card_ent * c_mask_f).sum(dim=1) * needs_cards_flag

        total_lp = type_lp + entity_lp + card_lp
        total_ent = type_ent + entity_ent + card_ent
        values = out["value"].squeeze(-1)

        return total_lp, total_ent, values
