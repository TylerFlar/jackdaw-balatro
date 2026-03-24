"""Factored policy + value network for Balatro.

The network decomposes the action into three sequential heads:
  1. Action type head  — categorical over 21 types (masked)
  2. Entity pointer head — pointer over the relevant entity list (masked, conditional)
  3. Card selection head — independent Bernoulli per hand card (masked, conditional)
  4. Value head — scalar state value

Log-prob of a full action = sum of log-probs from the heads that fired.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from jackdaw.env.balatro_spec import balatro_game_spec

_SPEC = balatro_game_spec()

# Entity layout: (name, max_count, feat_dim)
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]

NUM_ACTION_TYPES: int = _SPEC.num_action_types  # 21
NEEDS_ENTITY: frozenset[int] = _SPEC.needs_entity_set
NEEDS_CARDS: frozenset[int] = _SPEC.needs_cards_set

# Map action_type -> entity_type_index (for pointer head routing)
ACTION_TO_ENTITY_TYPE: dict[int, int] = {
    i: _SPEC.entity_type_for_action(i)
    for i in range(NUM_ACTION_TYPES)
    if _SPEC.entity_type_for_action(i) >= 0
}

# Entity type dims/max counts indexed by entity_type_index
ENTITY_NAMES: list[str] = [et.name for et in _SPEC.entity_types]
ENTITY_MAX_COUNTS: list[int] = [et.max_count for et in _SPEC.entity_types]
ENTITY_FEAT_DIMS: list[int] = [et.feature_dim for et in _SPEC.entity_types]

# Indices
HAND_CARD_IDX = 0  # hand_card entity type index
D_GLOBAL = _SPEC.global_feature_dim  # 235

# Encoder output dims
GLOBAL_EMBED = 128
ENTITY_EMBED = 64  # all entity types (unified for cross-attention)
STATE_EMBED = 256

POINTER_DIM = 64  # query/key dim for pointer attention

# Cross-entity attention config
NUM_ATTN_LAYERS = 2
NUM_ATTN_HEADS = 4
ATTN_FFN_DIM = 128
MAX_ENTITIES = sum(et.max_count for et in _SPEC.entity_types)  # 30

# Logits are clamped to [-_LOGIT_CLAMP, _LOGIT_CLAMP] before masking, then
# illegal positions are set to _MASK_VALUE.  _MASK_VALUE must be well below
# -_LOGIT_CLAMP so masked positions can never be sampled.
_LOGIT_CLAMP = 20.0
_MASK_VALUE = -1e4  # exp(-1e4) == 0 in float32; safe from NaN unlike -1e8


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block (no dropout — RL setting)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class FactoredPolicy(nn.Module):
    """Factored actor-critic for Balatro.

    Observation format (dict of tensors, batch dim first):
        "global":        (B, 235)
        "hand_card":     (B, 8, 15)
        "joker":         (B, 5, 15)
        "consumable":    (B, 2, 7)
        "shop_item":     (B, 10, 9)
        "pack_card":     (B, 5, 15)
        "entity_counts": (B, 5)
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Entity encoders (separate weights per type, unified output dim) ---
        self.entity_encoders = nn.ModuleDict()
        for i, (name, _, feat_dim) in enumerate(_ENTITY_INFO):
            self.entity_encoders[name] = _mlp(feat_dim, ENTITY_EMBED, ENTITY_EMBED)

        # --- Global encoder ---
        self.global_encoder = _mlp(D_GLOBAL, GLOBAL_EMBED, GLOBAL_EMBED)

        # --- Cross-entity attention ---
        self.entity_type_embed = nn.Embedding(len(_ENTITY_INFO), ENTITY_EMBED)
        self.cross_attn_layers = nn.ModuleList([
            _TransformerBlock(ENTITY_EMBED, NUM_ATTN_HEADS, ATTN_FFN_DIM)
            for _ in range(NUM_ATTN_LAYERS)
        ])

        # --- Attention pooling queries (one per entity type) ---
        self.pool_queries = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, ENTITY_EMBED) * 0.02)
            for name, _, _ in _ENTITY_INFO
        })

        # --- Combiner: concat pooled entities + global -> state embedding ---
        pool_total = GLOBAL_EMBED + ENTITY_EMBED * len(_ENTITY_INFO)
        self.combiner = _mlp(pool_total, STATE_EMBED, STATE_EMBED)

        # --- Action type head ---
        self.action_type_head = nn.Linear(STATE_EMBED, NUM_ACTION_TYPES)

        # --- Entity pointer head (one query projection per entity type) ---
        self.pointer_queries = nn.ModuleDict()
        for i, name in enumerate(ENTITY_NAMES):
            self.pointer_queries[name] = nn.Linear(STATE_EMBED, ENTITY_EMBED)

        # --- Card selection head ---
        # query from state, score via MLP on concat(query, card_embed)
        self.card_query = nn.Linear(STATE_EMBED, ENTITY_EMBED)
        self.card_score = nn.Sequential(
            nn.Linear(ENTITY_EMBED * 2, ENTITY_EMBED),
            nn.ReLU(),
            nn.Linear(ENTITY_EMBED, 1),
        )

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(STATE_EMBED, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm, nn.MultiheadAttention)):
                continue  # use default init for these
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m is self.action_type_head else 1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode observation into state embedding + per-entity embeddings.

        Returns
        -------
        state : (B, STATE_EMBED)
        entity_embeds : {entity_name: (B, max_count, ENTITY_EMBED)}
        """
        B = obs["global"].shape[0]
        device = obs["global"].device
        counts = obs["entity_counts"]  # (B, 5)

        # Global
        g = self.global_encoder(obs["global"])  # (B, GLOBAL_EMBED)

        # 1. Encode each entity type independently
        all_embeds: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        offsets: list[tuple[int, int]] = []
        offset = 0

        for i, (name, max_count, _) in enumerate(_ENTITY_INFO):
            raw = obs[name]  # (B, max_count, feat_dim)
            embed = self.entity_encoders[name](raw)  # (B, max_count, ENTITY_EMBED)
            embed = embed + self.entity_type_embed.weight[i]  # add type embedding
            all_embeds.append(embed)

            c = counts[:, i].long()
            mask = torch.arange(max_count, device=device).unsqueeze(0) < c.unsqueeze(1)
            all_masks.append(mask)
            offsets.append((offset, offset + max_count))
            offset += max_count

        # 2. Concatenate all entities into single sequence for cross-attention
        seq = torch.cat(all_embeds, dim=1)  # (B, MAX_ENTITIES, ENTITY_EMBED)
        # MHA convention: True = padded/ignored
        pad_mask = ~torch.cat(all_masks, dim=1)  # (B, MAX_ENTITIES)

        # 3. Cross-entity self-attention
        # Zero padded positions instead of using key_padding_mask to avoid
        # NaN gradients from all-masked attention rows in MHA.
        valid_mask = (~pad_mask).unsqueeze(-1)  # (B, MAX_ENTITIES, 1)
        seq = seq * valid_mask
        for layer in self.cross_attn_layers:
            seq = layer(seq)
            seq = seq * valid_mask

        # 4. Split back into per-type and attention-pool
        entity_embeds: dict[str, torch.Tensor] = {}
        pooled_parts: list[torch.Tensor] = [g]

        for i, (name, max_count, _) in enumerate(_ENTITY_INFO):
            start, end = offsets[i]
            ent_seq = seq[:, start:end, :]  # (B, max_count, ENTITY_EMBED)
            entity_embeds[name] = ent_seq

            # Attention pooling: learned query attends over entities
            query = self.pool_queries[name].expand(B, -1, -1)  # (B, 1, ENTITY_EMBED)
            scores = (query @ ent_seq.transpose(-1, -2)) / (ENTITY_EMBED ** 0.5)
            scores = scores.masked_fill(~all_masks[i].unsqueeze(1), float("-inf"))
            weights = torch.softmax(scores, dim=-1)  # (B, 1, max_count)
            weights = weights.nan_to_num(0.0)  # handle all-padded types
            pooled = (weights @ ent_seq).squeeze(1)  # (B, ENTITY_EMBED)
            pooled_parts.append(pooled)

        combined = torch.cat(pooled_parts, dim=-1)  # (B, pool_total)
        state = self.combiner(combined)  # (B, STATE_EMBED)
        return state, entity_embeds

    # ------------------------------------------------------------------
    # Forward: sample actions
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        action_masks: dict[str, Any],
    ) -> dict[str, Any]:
        """Sample a full factored action.

        Parameters
        ----------
        obs : padded observation tensors, each with batch dim
        action_masks : dict with:
            "type_mask": (B, 21) bool
            "card_mask": (B, 8) bool  (padded to max hand size)
            "entity_masks": {action_type_int: (B, max_entity_count) bool}
            "min_card_select": (B,) int
            "max_card_select": (B,) int

        Returns
        -------
        dict with: action_type, entity_target, card_target,
                   log_prob, value, entropy
        """
        state, entity_embeds = self._encode(obs)
        B = state.shape[0]
        device = state.device

        # --- Action type ---
        type_logits = self.action_type_head(state).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        type_mask = action_masks["type_mask"]  # (B, 21) bool
        type_logits = type_logits.masked_fill(~type_mask, _MASK_VALUE)
        type_dist = Categorical(logits=type_logits)
        action_type = type_dist.sample()  # (B,)
        type_lp = type_dist.log_prob(action_type)  # (B,)
        type_entropy = type_dist.entropy()  # (B,)

        # --- Entity pointer (conditional) ---
        entity_target = torch.full((B,), -1, dtype=torch.long, device=device)
        entity_lp = torch.zeros(B, device=device)

        # Group samples by which entity type they need
        for etype_idx in range(len(_ENTITY_INFO)):
            name = ENTITY_NAMES[etype_idx]
            # Which action types point to this entity type?
            relevant_actions = [a for a, e in ACTION_TO_ENTITY_TYPE.items() if e == etype_idx]
            if not relevant_actions:
                continue
            # Which batch elements chose one of these action types?
            batch_mask = torch.zeros(B, dtype=torch.bool, device=device)
            for a in relevant_actions:
                batch_mask |= (action_type == a)
            if not batch_mask.any():
                continue

            idx = batch_mask.nonzero(as_tuple=True)[0]
            query = self.pointer_queries[name](state[idx])  # (n, embed_dim)
            keys = entity_embeds[name][idx]  # (n, max_count, embed_dim)
            scores = ((query.unsqueeze(1) * keys).sum(-1) / (query.shape[-1] ** 0.5)).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)

            # Build entity mask for these batch elements
            max_count = ENTITY_MAX_COUNTS[etype_idx]
            emask = torch.zeros(len(idx), max_count, dtype=torch.bool, device=device)
            for a in relevant_actions:
                if a in action_masks["entity_masks"]:
                    a_mask = action_masks["entity_masks"][a][idx]  # (n, max_count)
                    chose_a = (action_type[idx] == a).unsqueeze(-1)
                    emask = emask | (a_mask & chose_a)

            # Ensure at least one entity is valid per sample to avoid all-masked NaN
            no_valid = ~emask.any(dim=-1)
            if no_valid.any():
                emask[no_valid, 0] = True

            scores = scores.masked_fill(~emask, _MASK_VALUE)
            edist = Categorical(logits=scores)
            etgt = edist.sample()
            entity_target[idx] = etgt
            entity_lp[idx] = edist.log_prob(etgt)

        # --- Card selection (conditional) ---
        card_target = torch.zeros(B, ENTITY_MAX_COUNTS[HAND_CARD_IDX], dtype=torch.bool, device=device)
        card_lp = torch.zeros(B, device=device)
        card_entropy = torch.zeros(B, device=device)

        needs_cards_mask = torch.zeros(B, dtype=torch.bool, device=device)
        for a in NEEDS_CARDS:
            needs_cards_mask |= (action_type == a)

        if needs_cards_mask.any():
            idx = needs_cards_mask.nonzero(as_tuple=True)[0]
            max_hand = ENTITY_MAX_COUNTS[HAND_CARD_IDX]  # 8
            q = self.card_query(state[idx])  # (n, ENTITY_EMBED)
            hand_embeds = entity_embeds["hand_card"][idx]  # (n, max_hand, ENTITY_EMBED)
            q_expanded = q.unsqueeze(1).expand(-1, max_hand, -1)  # (n, max_hand, ENTITY_EMBED)
            cat_qk = torch.cat([q_expanded, hand_embeds], dim=-1)  # (n, max_hand, ENTITY_EMBED*2)
            card_logits = self.card_score(cat_qk).squeeze(-1)  # (n, max_hand)

            cmask = action_masks["card_mask"][idx]  # (n, max_hand) bool
            # Clamp logits before masking to prevent overflow
            card_logits = card_logits.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
            card_logits = card_logits.masked_fill(~cmask, _MASK_VALUE)

            # Independent Bernoulli per card
            card_probs = torch.sigmoid(card_logits)  # (n, max_hand)
            card_sample = torch.bernoulli(card_probs).bool()  # (n, max_hand)
            card_sample = card_sample & cmask  # enforce mask

            # Enforce min/max card select constraints
            min_sel = action_masks["min_card_select"][idx]  # (n,)
            max_sel = action_masks["max_card_select"][idx]  # (n,)
            card_sample = self._enforce_card_counts(
                card_sample, card_logits, cmask, min_sel, max_sel
            )

            card_target[idx] = card_sample

            # Log prob: sum of log Bernoulli probs for each card
            safe_probs = card_probs.clamp(1e-7, 1.0 - 1e-7)
            per_card_lp = (
                card_sample.float() * safe_probs.log()
                + (~card_sample).float() * (1.0 - safe_probs).log()
            )
            per_card_lp = per_card_lp * cmask.float()  # zero out padding
            card_lp[idx] = per_card_lp.sum(dim=-1)

            # Entropy of independent Bernoulli
            per_card_ent = -(
                safe_probs * safe_probs.log() + (1 - safe_probs) * (1 - safe_probs).log()
            )
            per_card_ent = per_card_ent * cmask.float()
            card_entropy[idx] = per_card_ent.sum(dim=-1)

        # --- Value ---
        value = self.value_head(state).squeeze(-1)  # (B,)

        # --- Total log prob ---
        total_lp = type_lp + entity_lp + card_lp

        return {
            "action_type": action_type,         # (B,) int
            "entity_target": entity_target,     # (B,) int, -1 if unused
            "card_target": card_target,         # (B, max_hand) bool
            "log_prob": total_lp,               # (B,)
            "value": value,                     # (B,)
            "entropy": type_entropy + card_entropy,  # (B,)
        }

    # ------------------------------------------------------------------
    # Evaluate: compute log_prob of given actions (for PPO ratio)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        obs: dict[str, torch.Tensor],
        action_masks: dict[str, Any],
        action_type: torch.Tensor,      # (B,)
        entity_target: torch.Tensor,    # (B,)  -1 if unused
        card_target: torch.Tensor,      # (B, max_hand) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate log_prob, value, and entropy for given actions.

        Returns (log_prob, value, entropy) each of shape (B,).
        """
        state, entity_embeds = self._encode(obs)
        B = state.shape[0]
        device = state.device

        # --- Action type ---
        type_logits = self.action_type_head(state)
        type_mask = action_masks["type_mask"]
        # Ensure at least the chosen action type is unmasked (handles stale masks)
        one_hot = F.one_hot(action_type, NUM_ACTION_TYPES).bool()
        type_mask = type_mask | one_hot
        type_logits = type_logits.masked_fill(~type_mask, _MASK_VALUE)
        type_dist = Categorical(logits=type_logits)
        type_lp = type_dist.log_prob(action_type)
        type_entropy = type_dist.entropy()

        # --- Entity pointer ---
        entity_lp = torch.zeros(B, device=device)
        for etype_idx in range(len(_ENTITY_INFO)):
            name = ENTITY_NAMES[etype_idx]
            relevant_actions = [a for a, e in ACTION_TO_ENTITY_TYPE.items() if e == etype_idx]
            if not relevant_actions:
                continue
            batch_mask = torch.zeros(B, dtype=torch.bool, device=device)
            for a in relevant_actions:
                batch_mask |= (action_type == a)
            if not batch_mask.any():
                continue

            idx = batch_mask.nonzero(as_tuple=True)[0]
            query = self.pointer_queries[name](state[idx])
            keys = entity_embeds[name][idx]
            scores = (query.unsqueeze(1) * keys).sum(-1) / (query.shape[-1] ** 0.5)

            max_count = ENTITY_MAX_COUNTS[etype_idx]
            emask = torch.zeros(len(idx), max_count, dtype=torch.bool, device=device)
            for a in relevant_actions:
                if a in action_masks["entity_masks"]:
                    a_mask = action_masks["entity_masks"][a][idx]
                    chose_a = (action_type[idx] == a).unsqueeze(-1)
                    emask = emask | (a_mask & chose_a)

            # Ensure the stored entity target is always unmasked
            et_local = entity_target[idx]
            valid_et = (et_local >= 0) & (et_local < max_count)
            if valid_et.any():
                rows = torch.arange(len(idx), device=device)[valid_et]
                cols = et_local[valid_et]
                emask[rows, cols] = True

            # Fallback: if still all-masked, unmask slot 0
            no_valid = ~emask.any(dim=-1)
            if no_valid.any():
                emask[no_valid, 0] = True

            scores = scores.masked_fill(~emask, _MASK_VALUE)
            edist = Categorical(logits=scores)
            # Clamp entity_target to valid range for log_prob
            safe_et = et_local.clamp(0, max_count - 1)
            entity_lp[idx] = edist.log_prob(safe_et)

        # --- Card selection ---
        card_lp = torch.zeros(B, device=device)
        card_entropy = torch.zeros(B, device=device)

        needs_cards_mask = torch.zeros(B, dtype=torch.bool, device=device)
        for a in NEEDS_CARDS:
            needs_cards_mask |= (action_type == a)

        if needs_cards_mask.any():
            idx = needs_cards_mask.nonzero(as_tuple=True)[0]
            max_hand = ENTITY_MAX_COUNTS[HAND_CARD_IDX]
            q = self.card_query(state[idx])
            hand_embeds = entity_embeds["hand_card"][idx]
            q_expanded = q.unsqueeze(1).expand(-1, max_hand, -1)
            cat_qk = torch.cat([q_expanded, hand_embeds], dim=-1)
            card_logits = self.card_score(cat_qk).squeeze(-1)

            cmask = action_masks["card_mask"][idx]
            # Ensure selected cards are in mask (handles stale masks)
            cmask = cmask | card_target[idx]
            card_logits = card_logits.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
            card_logits = card_logits.masked_fill(~cmask, _MASK_VALUE)

            card_probs = torch.sigmoid(card_logits)
            safe_probs = card_probs.clamp(1e-7, 1.0 - 1e-7)
            ct = card_target[idx].float()
            per_card_lp = ct * safe_probs.log() + (1.0 - ct) * (1.0 - safe_probs).log()
            per_card_lp = per_card_lp * cmask.float()
            card_lp[idx] = per_card_lp.sum(dim=-1)

            per_card_ent = -(
                safe_probs * safe_probs.log() + (1 - safe_probs) * (1 - safe_probs).log()
            )
            per_card_ent = per_card_ent * cmask.float()
            card_entropy[idx] = per_card_ent.sum(dim=-1)

        # --- Value ---
        value = self.value_head(state).squeeze(-1)

        total_lp = type_lp + entity_lp + card_lp
        total_entropy = type_entropy + card_entropy

        return total_lp, value, total_entropy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_card_counts(
        selected: torch.Tensor,   # (n, max_hand) bool
        logits: torch.Tensor,     # (n, max_hand)
        mask: torch.Tensor,       # (n, max_hand) bool
        min_sel: torch.Tensor,    # (n,)
        max_sel: torch.Tensor,    # (n,)
    ) -> torch.Tensor:
        """Clamp card selection counts to [min_sel, max_sel] per sample.

        If too few selected, force-select highest-logit legal cards.
        If too many, keep highest-logit ones.
        """
        n, max_hand = selected.shape
        sel = selected.clone()
        for i in range(n):
            legal = mask[i]
            chosen = sel[i] & legal
            count = chosen.sum().item()
            mn = min_sel[i].item()
            mx = max_sel[i].item()

            if count < mn:
                scores = logits[i].clone()
                scores[~legal | chosen] = -100.0
                needed = mn - count
                k = min(needed, max(0, int(legal.sum().item() - count)))
                if k > 0:
                    _, topk = scores.topk(k)
                    chosen[topk] = True
                sel[i] = chosen
            elif count > mx:
                scores = logits[i].clone()
                scores[~chosen] = -100.0
                _, topk = scores.topk(mx)
                new_sel = torch.zeros_like(chosen)
                new_sel[topk] = True
                sel[i] = new_sel

        return sel
