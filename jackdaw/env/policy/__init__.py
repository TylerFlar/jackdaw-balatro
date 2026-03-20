"""Transformer-based policy network for card game RL.

Follows the AlphaStar entity encoder pattern: per-type entity encoders
project heterogeneous features to a common embedding, a Transformer
contextualizes all entities, and factored action heads produce
autoregressive action distributions.

Architecture is fully parametric — driven by :class:`~jackdaw.env.game_spec.GameSpec`.
"""

from jackdaw.env.policy.action_heads import ActionHeads
from jackdaw.env.policy.entity_encoder import EntityEncoder
from jackdaw.env.policy.policy import (
    BalatroPolicy,
    PolicyInput,
    batch_to_device,
    collate_policy_inputs,
)
from jackdaw.env.policy.transformer import TransformerCore

__all__ = [
    "ActionHeads",
    "BalatroPolicy",
    "EntityEncoder",
    "PolicyInput",
    "TransformerCore",
    "batch_to_device",
    "collate_policy_inputs",
]
