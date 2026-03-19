"""BalatroPolicy: full policy network combining encoder, transformer, and action heads.

Also provides :func:`collate_policy_inputs` for batching variable-length
observations into padded tensors suitable for the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from jackdaw.env.action_space import (
    NUM_ACTION_TYPES,
    ActionMask,
    FactoredAction,
)
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
)
from jackdaw.env.policy.action_heads import NEEDS_CARDS, NEEDS_ENTITY, ActionHeads
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
        Encoded observation from :func:`encode_observation`.
    action_mask:
        Legal action mask from :func:`get_action_mask`.
    shop_splits:
        ``(n_shop_cards, n_vouchers, n_boosters)`` — needed to correctly
        place entity pointer masks for shop sub-types.  Defaults to
        treating all shop items as buyable cards.
    """

    obs: Observation
    action_mask: ActionMask
    shop_splits: tuple[int, int, int] = (0, 0, 0)


# ---------------------------------------------------------------------------
# Collation utilities
# ---------------------------------------------------------------------------


def _pad_2d(arrays: list[np.ndarray], max_n: int, d: int) -> torch.Tensor:
    """Pad and stack variable-length 2D arrays into ``(B, max_n, d)``."""
    B = len(arrays)
    out = torch.zeros(B, max(max_n, 0), d)
    for i, arr in enumerate(arrays):
        n = arr.shape[0]
        if n > 0:
            out[i, :n] = torch.from_numpy(arr)
    return out


def _make_mask(arrays: list[np.ndarray], max_n: int) -> torch.Tensor:
    """Create ``(B, max_n)`` bool mask from variable-length arrays."""
    B = len(arrays)
    mask = torch.zeros(B, max(max_n, 0), dtype=torch.bool)
    for i, arr in enumerate(arrays):
        mask[i, : arr.shape[0]] = True
    return mask


def collate_policy_inputs(
    inputs: list[PolicyInput],
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Batch a list of :class:`PolicyInput` into padded tensors.

    Parameters
    ----------
    inputs:
        List of single-step policy inputs.
    device:
        Target device for all tensors (e.g. ``"cuda"``).  If *None*,
        tensors are created on CPU.

    Returns a dict that can be passed directly to
    :meth:`BalatroPolicy.forward`.
    """
    B = len(inputs)

    # Max sizes per entity type
    max_hand = max(inp.obs.hand_cards.shape[0] for inp in inputs)
    max_joker = max(inp.obs.jokers.shape[0] for inp in inputs)
    max_cons = max(inp.obs.consumables.shape[0] for inp in inputs)
    max_shop = max(inp.obs.shop_cards.shape[0] for inp in inputs)
    max_pack = max(inp.obs.pack_cards.shape[0] for inp in inputs)
    N_total = max_hand + max_joker + max_cons + max_shop + max_pack

    # Padded feature tensors
    hand_feats = _pad_2d([i.obs.hand_cards for i in inputs], max_hand, D_PLAYING_CARD)
    joker_feats = _pad_2d([i.obs.jokers for i in inputs], max_joker, D_JOKER)
    cons_feats = _pad_2d([i.obs.consumables for i in inputs], max_cons, D_CONSUMABLE)
    shop_feats = _pad_2d([i.obs.shop_cards for i in inputs], max_shop, D_SHOP)
    pack_feats = _pad_2d([i.obs.pack_cards for i in inputs], max_pack, D_PLAYING_CARD)

    # Boolean masks
    hand_mask = _make_mask([i.obs.hand_cards for i in inputs], max_hand)
    joker_mask = _make_mask([i.obs.jokers for i in inputs], max_joker)
    cons_mask = _make_mask([i.obs.consumables for i in inputs], max_cons)
    shop_mask = _make_mask([i.obs.shop_cards for i in inputs], max_shop)
    pack_mask = _make_mask([i.obs.pack_cards for i in inputs], max_pack)

    # Global context
    global_ctx = torch.stack([torch.from_numpy(inp.obs.global_context) for inp in inputs])

    # Type mask (B, 21)
    type_mask = torch.stack([torch.from_numpy(inp.action_mask.type_mask.copy()) for inp in inputs])

    # Card mask (B, max_hand)
    card_mask = torch.zeros(B, max(max_hand, 0), dtype=torch.bool)
    for i, inp in enumerate(inputs):
        n = len(inp.action_mask.card_mask)
        if n > 0:
            card_mask[i, :n] = torch.from_numpy(inp.action_mask.card_mask)

    # Card selection limits
    max_card_select = torch.tensor(
        [inp.action_mask.max_card_select for inp in inputs], dtype=torch.long
    )
    min_card_select = torch.tensor(
        [inp.action_mask.min_card_select for inp in inputs], dtype=torch.long
    )

    # Entity offsets in the concatenated sequence (same for all batch items)
    entity_offsets = torch.tensor(
        [
            0,
            max_hand,
            max_hand + max_joker,
            max_hand + max_joker + max_cons,
            max_hand + max_joker + max_cons + max_shop,
        ],
        dtype=torch.long,
    )

    # Shop splits per item
    shop_splits = torch.tensor([inp.shop_splits for inp in inputs], dtype=torch.long)  # (B, 3)

    # Pointer masks: (B, 21, N_total)
    pointer_masks = torch.zeros(B, NUM_ACTION_TYPES, max(N_total, 0), dtype=torch.bool)

    hand_start = 0
    joker_start = max_hand
    cons_start = max_hand + max_joker
    shop_start = max_hand + max_joker + max_cons
    pack_start = max_hand + max_joker + max_cons + max_shop

    for b, inp in enumerate(inputs):
        n_sc, n_sv, n_sb = inp.shop_splits
        for at_int, emask in inp.action_mask.entity_masks.items():
            at = int(at_int)
            mask_t = torch.from_numpy(emask)
            n = len(emask)
            if at in (17, 18):  # SwapHand -> hand
                pointer_masks[b, at, hand_start : hand_start + n] = mask_t
            elif at in (9, 15, 16):  # Joker targets
                pointer_masks[b, at, joker_start : joker_start + n] = mask_t
            elif at in (10, 11):  # Consumable targets
                pointer_masks[b, at, cons_start : cons_start + n] = mask_t
            elif at == 8:  # BuyCard -> shop_cards portion
                pointer_masks[b, at, shop_start : shop_start + n] = mask_t
            elif at == 12:  # RedeemVoucher -> shop_vouchers
                off = shop_start + n_sc
                pointer_masks[b, at, off : off + n] = mask_t
            elif at == 13:  # OpenBooster -> shop_boosters
                off = shop_start + n_sc + n_sv
                pointer_masks[b, at, off : off + n] = mask_t
            elif at == 14:  # PickPackCard
                pointer_masks[b, at, pack_start : pack_start + n] = mask_t

    batch = {
        "global_context": global_ctx,
        "hand_cards": hand_feats,
        "jokers": joker_feats,
        "consumables": cons_feats,
        "shop_cards": shop_feats,
        "pack_cards": pack_feats,
        "hand_mask": hand_mask,
        "joker_mask": joker_mask,
        "cons_mask": cons_mask,
        "shop_mask": shop_mask,
        "pack_mask": pack_mask,
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


def batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move all tensors in a collated batch to the given device.

    Non-tensor values (e.g. ``max_hand``) are passed through unchanged.
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Pointer index <-> entity_target conversion
# ---------------------------------------------------------------------------

# Maps action_type -> (entity_type_offset_index, sub_offset_key)
# entity_type_offset_index: 0=hand, 1=joker, 2=cons, 3=shop, 4=pack
_ACTION_ENTITY_MAP: dict[int, tuple[int, str]] = {
    8: (3, "cards"),
    9: (1, ""),
    10: (2, ""),
    11: (2, ""),
    12: (3, "vouchers"),
    13: (3, "boosters"),
    14: (4, ""),
    15: (1, ""),
    16: (1, ""),
    17: (0, ""),
    18: (0, ""),
}


def _pointer_to_entity_target(
    ptr_idx: int,
    action_type: int,
    entity_offsets: torch.Tensor,
    shop_splits: torch.Tensor,
) -> int:
    """Convert a position in the concatenated entity sequence to a per-type index."""
    offsets = entity_offsets.tolist()
    type_idx, sub = _ACTION_ENTITY_MAP[action_type]
    base = offsets[type_idx]
    result = ptr_idx - base
    if sub == "vouchers":
        result -= int(shop_splits[0].item())
    elif sub == "boosters":
        result -= int(shop_splits[0].item()) - int(shop_splits[1].item())
    return result


def _entity_target_to_pointer(
    entity_target: int,
    action_type: int,
    entity_offsets: torch.Tensor,
    shop_splits: torch.Tensor,
) -> int:
    """Convert a per-type entity index to a position in the concatenated sequence."""
    offsets = entity_offsets.tolist()
    type_idx, sub = _ACTION_ENTITY_MAP[action_type]
    base = offsets[type_idx]
    result = entity_target + base
    if sub == "vouchers":
        result += int(shop_splits[0].item())
    elif sub == "boosters":
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
        # Force-select highest-scored unselected valid cards
        scores = probs.clone()
        scores[selected | ~mask] = -1.0
        available = int((scores > -1.0).sum().item())
        needed = min(min_cards - n_selected, available)
        if needed > 0:
            _, top_idx = scores.topk(needed)
            selected[top_idx] = True
    elif n_selected > max_cards:
        # Keep only top-scored selected cards
        scores = probs.clone()
        scores[~selected] = -1.0
        _, top_idx = scores.topk(max_cards)
        new_selected = torch.zeros_like(selected)
        new_selected[top_idx] = True
        selected = new_selected

    # Independent Bernoulli log_prob (approximation — ignores constraint)
    selected_f = selected.float()
    log_prob = F.logsigmoid(logits) * selected_f + F.logsigmoid(-logits) * (1.0 - selected_f)
    log_prob = (log_prob * mask.float()).sum()

    return selected, log_prob


# ---------------------------------------------------------------------------
# BalatroPolicy
# ---------------------------------------------------------------------------


class BalatroPolicy(nn.Module):
    """Full policy network combining encoder, transformer, and action heads.

    Designed for PPO training with the Balatro Gymnasium environment.
    Handles variable-length observations via padded batching.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.entity_encoder = EntityEncoder(embed_dim=embed_dim)
        self.transformer = TransformerCore(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.action_heads = ActionHeads(embed_dim=embed_dim)

    @property
    def device(self) -> torch.device:
        """Device of the model parameters (convenience for collation)."""
        return next(self.parameters()).device

    def prepare_batch(self, inputs: list[PolicyInput]) -> dict[str, Any]:
        """Collate inputs and move to this model's device."""
        return collate_policy_inputs(inputs, device=self.device)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Full forward pass through encoder, transformer, and heads.

        Parameters
        ----------
        batch:
            Output of :func:`collate_policy_inputs`.

        Returns
        -------
        dict with keys:
            ``type_logits`` ``(B, 21)``, ``card_logits`` ``(B, N_hand)``,
            ``value`` ``(B, 1)``, ``global_repr`` ``(B, E)``,
            ``entity_reprs`` ``(B, N_total, E)``.
        """
        # Encode entities
        entities, entity_mask = self.entity_encoder(
            batch["hand_cards"],
            batch["jokers"],
            batch["consumables"],
            batch["shop_cards"],
            batch["pack_cards"],
            batch["hand_mask"],
            batch["joker_mask"],
            batch["cons_mask"],
            batch["shop_mask"],
            batch["pack_mask"],
        )

        # Contextualize via transformer
        global_repr, entity_reprs = self.transformer(batch["global_context"], entities, entity_mask)

        # Action heads (type + card + value — entity computed on demand)
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
    ) -> tuple[list[FactoredAction], dict[str, torch.Tensor]]:
        """Sample actions autoregressively: type -> entity -> cards.

        Returns ``(actions, log_probs_dict)`` where log_probs_dict has keys
        ``"type"``, ``"entity"``, ``"card"``, ``"total"``.
        """
        self.eval()
        out = self.forward(batch)
        B = out["type_logits"].shape[0]
        device = out["type_logits"].device

        actions: list[FactoredAction] = []
        type_lps: list[torch.Tensor] = []
        entity_lps: list[torch.Tensor] = []
        card_lps: list[torch.Tensor] = []

        for b in range(B):
            # Stage 1: action type
            type_dist = Categorical(logits=out["type_logits"][b])
            action_type = type_dist.sample()
            type_lps.append(type_dist.log_prob(action_type))
            at = action_type.item()

            # Stage 2: entity target
            entity_target: int | None = None
            e_lp = torch.zeros((), device=device)

            if at in NEEDS_ENTITY:
                pmask = batch["pointer_masks"][b, at]
                if pmask.any():
                    e_lgts = self.action_heads.entity_logits(
                        out["global_repr"][b : b + 1],
                        out["entity_reprs"][b : b + 1],
                        pmask.unsqueeze(0),
                    ).squeeze(0)
                    e_dist = Categorical(logits=e_lgts)
                    e_idx = e_dist.sample()
                    e_lp = e_dist.log_prob(e_idx)
                    entity_target = _pointer_to_entity_target(
                        e_idx.item(),
                        at,
                        batch["entity_offsets"],
                        batch["shop_splits"][b],
                    )
            entity_lps.append(e_lp)

            # Stage 3: card targets
            card_target: tuple[int, ...] | None = None
            c_lp = torch.zeros((), device=device)

            if at in NEEDS_CARDS and batch["card_mask"][b].any():
                min_c = int(batch["min_card_select"][b].item())
                max_c = int(batch["max_card_select"][b].item())
                selected, c_lp = _sample_cards(
                    out["card_logits"][b],
                    batch["card_mask"][b],
                    min_c,
                    max_c,
                )
                indices = selected.nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    card_target = tuple(int(i) for i in indices)
            card_lps.append(c_lp)

            actions.append(
                FactoredAction(
                    action_type=at,
                    card_target=card_target,
                    entity_target=entity_target,
                )
            )

        log_probs = {
            "type": torch.stack(type_lps),
            "entity": torch.stack(entity_lps),
            "card": torch.stack(card_lps),
            "total": (torch.stack(type_lps) + torch.stack(entity_lps) + torch.stack(card_lps)),
        }
        return actions, log_probs

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
            Total log-probability ``log p(type) + log p(entity|type) + log p(cards|type)``.
        entropy : ``(B,)``
            Sum of component entropies.
        values : ``(B,)``
            State value estimates.
        """
        out = self.forward(batch)
        B = out["type_logits"].shape[0]
        device = out["type_logits"].device

        # Vectorised type log-prob + entropy
        action_types = torch.tensor(
            [fa.action_type for fa in actions], dtype=torch.long, device=device
        )
        type_dist = Categorical(logits=out["type_logits"])
        type_lp = type_dist.log_prob(action_types)  # (B,)
        type_ent = type_dist.entropy()  # (B,)

        # Per-item entity + card (loop — hard to vectorise due to varying masks)
        entity_lps: list[torch.Tensor] = []
        entity_ents: list[torch.Tensor] = []
        card_lps: list[torch.Tensor] = []

        for b in range(B):
            fa = actions[b]

            # Entity
            e_lp = torch.zeros((), device=device)
            e_ent = torch.zeros((), device=device)
            if fa.action_type in NEEDS_ENTITY and fa.entity_target is not None:
                pmask = batch["pointer_masks"][b, fa.action_type]
                if pmask.any():
                    e_lgts = self.action_heads.entity_logits(
                        out["global_repr"][b : b + 1],
                        out["entity_reprs"][b : b + 1],
                        pmask.unsqueeze(0),
                    ).squeeze(0)
                    e_dist = Categorical(logits=e_lgts)
                    ptr = _entity_target_to_pointer(
                        fa.entity_target,
                        fa.action_type,
                        batch["entity_offsets"],
                        batch["shop_splits"][b],
                    )
                    e_lp = e_dist.log_prob(torch.tensor(ptr, dtype=torch.long, device=device))
                    e_ent = e_dist.entropy()
            entity_lps.append(e_lp)
            entity_ents.append(e_ent)

            # Card
            c_lp = torch.zeros((), device=device)
            if fa.action_type in NEEDS_CARDS and fa.card_target:
                c_lgts = out["card_logits"][b]
                c_mask = batch["card_mask"][b]
                selected = torch.zeros(c_lgts.shape[0], device=device)
                for idx in fa.card_target:
                    if idx < len(selected):
                        selected[idx] = 1.0
                lp = F.logsigmoid(c_lgts) * selected + F.logsigmoid(-c_lgts) * (1.0 - selected)
                c_lp = (lp * c_mask.float()).sum()
            card_lps.append(c_lp)

        entity_lp = torch.stack(entity_lps)
        entity_ent = torch.stack(entity_ents)
        card_lp = torch.stack(card_lps)

        total_lp = type_lp + entity_lp + card_lp
        total_ent = type_ent + entity_ent
        values = out["value"].squeeze(-1)

        return total_lp, total_ent, values
