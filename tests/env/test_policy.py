"""Tests for the Transformer-based policy network."""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.env.action_space import NUM_ACTION_TYPES, ActionMask, ActionType
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
)
from jackdaw.env.policy.action_heads import NEEDS_CARDS, NEEDS_ENTITY
from jackdaw.env.policy.policy import (
    BalatroPolicy,
    PolicyInput,
    collate_policy_inputs,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------


def _make_obs(
    n_hand: int = 5,
    n_joker: int = 2,
    n_cons: int = 1,
    n_shop: int = 3,
    n_pack: int = 0,
) -> Observation:
    """Create a synthetic Observation with random features."""
    rng = np.random.default_rng(42)
    return Observation(
        global_context=rng.standard_normal(D_GLOBAL).astype(np.float32),
        hand_cards=rng.standard_normal((n_hand, D_PLAYING_CARD)).astype(np.float32),
        jokers=rng.standard_normal((n_joker, D_JOKER)).astype(np.float32),
        consumables=rng.standard_normal((n_cons, D_CONSUMABLE)).astype(np.float32),
        shop_cards=rng.standard_normal((n_shop, D_SHOP)).astype(np.float32),
        pack_cards=rng.standard_normal((n_pack, D_PLAYING_CARD)).astype(np.float32),
    )


def _make_mask(
    n_hand: int = 5,
    n_joker: int = 2,
    n_cons: int = 1,
    n_shop: int = 3,
    n_pack: int = 0,
    phase: str = "selecting_hand",
) -> ActionMask:
    """Create a synthetic ActionMask for the given phase."""
    type_mask = np.zeros(NUM_ACTION_TYPES, dtype=bool)
    entity_masks: dict[int, np.ndarray] = {}
    card_mask = np.ones(n_hand, dtype=bool) if n_hand > 0 else np.zeros(0, dtype=bool)
    max_card_select = 5
    min_card_select = 1

    if phase == "selecting_hand":
        if n_hand > 0:
            type_mask[ActionType.PlayHand] = True
            type_mask[ActionType.Discard] = True
        if n_joker > 1:
            type_mask[ActionType.SellJoker] = True
            entity_masks[ActionType.SellJoker] = np.ones(n_joker, dtype=bool)
        if n_cons > 0:
            type_mask[ActionType.UseConsumable] = True
            entity_masks[ActionType.UseConsumable] = np.ones(n_cons, dtype=bool)

    elif phase == "shop":
        type_mask[ActionType.NextRound] = True
        if n_shop > 0:
            type_mask[ActionType.BuyCard] = True
            entity_masks[ActionType.BuyCard] = np.ones(n_shop, dtype=bool)
        if n_joker > 0:
            type_mask[ActionType.SellJoker] = True
            entity_masks[ActionType.SellJoker] = np.ones(n_joker, dtype=bool)

    elif phase == "pack":
        type_mask[ActionType.SkipPack] = True
        if n_pack > 0:
            type_mask[ActionType.PickPackCard] = True
            entity_masks[ActionType.PickPackCard] = np.ones(n_pack, dtype=bool)

    elif phase == "blind_select":
        type_mask[ActionType.SelectBlind] = True
        type_mask[ActionType.SkipBlind] = True

    return ActionMask(type_mask, card_mask, entity_masks, max_card_select, min_card_select)


def _make_input(
    n_hand: int = 5,
    n_joker: int = 2,
    n_cons: int = 1,
    n_shop: int = 3,
    n_pack: int = 0,
    phase: str = "selecting_hand",
) -> PolicyInput:
    obs = _make_obs(n_hand, n_joker, n_cons, n_shop, n_pack)
    mask = _make_mask(n_hand, n_joker, n_cons, n_shop, n_pack, phase)
    return PolicyInput(obs=obs, action_mask=mask, shop_splits=(n_shop, 0, 0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollation:
    def test_shapes(self):
        inputs = [
            _make_input(n_hand=5, n_joker=2, n_cons=1, n_shop=3),
            _make_input(n_hand=3, n_joker=0, n_cons=2, n_shop=1),
        ]
        batch = collate_policy_inputs(inputs)

        assert batch["hand_cards"].shape == (2, 5, D_PLAYING_CARD)
        assert batch["jokers"].shape == (2, 2, D_JOKER)
        assert batch["consumables"].shape == (2, 2, D_CONSUMABLE)
        assert batch["shop_cards"].shape == (2, 3, D_SHOP)
        assert batch["hand_mask"].shape == (2, 5)
        assert batch["type_mask"].shape == (2, NUM_ACTION_TYPES)
        assert batch["card_mask"].shape == (2, 5)
        assert batch["pointer_masks"].shape[0] == 2
        assert batch["pointer_masks"].shape[1] == NUM_ACTION_TYPES

    def test_masks_correct(self):
        inp = _make_input(n_hand=3, n_joker=2)
        batch = collate_policy_inputs([inp])

        # Hand mask: first 3 True, rest False (max_hand=3 here)
        assert batch["hand_mask"][0, :3].all()
        assert batch["max_hand"] == 3

    def test_empty_entities(self):
        inp = _make_input(n_hand=0, n_joker=0, n_cons=0, n_shop=0, n_pack=0, phase="blind_select")
        batch = collate_policy_inputs([inp])
        assert batch["hand_cards"].shape[1] == 0
        assert batch["jokers"].shape[1] == 0


class TestForwardPass:
    def test_output_shapes(self):
        torch.manual_seed(0)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(4)]
        batch = collate_policy_inputs(inputs)
        out = policy.forward(batch)

        assert out["type_logits"].shape == (4, NUM_ACTION_TYPES)
        assert out["card_logits"].shape == (4, batch["max_hand"])
        assert out["value"].shape == (4, 1)
        assert out["global_repr"].shape == (4, 64)
        N_total = sum(
            batch[k].shape[1]
            for k in ["hand_cards", "jokers", "consumables", "shop_cards", "pack_cards"]
        )
        assert out["entity_reprs"].shape == (4, N_total, 64)

    def test_no_entities(self):
        """Forward pass works with zero entities (only CLS token)."""
        torch.manual_seed(0)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inp = _make_input(n_hand=0, n_joker=0, n_cons=0, n_shop=0, n_pack=0, phase="blind_select")
        batch = collate_policy_inputs([inp])
        out = policy.forward(batch)

        assert out["type_logits"].shape == (1, NUM_ACTION_TYPES)
        assert out["value"].shape == (1, 1)


class TestSampleAction:
    def test_produces_valid_types(self):
        """Sampled action types are always legal."""
        torch.manual_seed(1)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(8)]
        batch = collate_policy_inputs(inputs)
        actions, log_probs = policy.sample_action(batch)

        assert len(actions) == 8
        for i, (action, inp) in enumerate(zip(actions, inputs)):
            assert inp.action_mask.type_mask[action.action_type], (
                f"Action {i}: type {action.action_type} not legal"
            )

    def test_entity_targets_present(self):
        """Entity-targeting actions have non-None entity_target."""
        torch.manual_seed(2)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        # Use shop phase to get entity-targeting actions
        inputs = [_make_input(n_shop=3, phase="shop") for _ in range(16)]
        batch = collate_policy_inputs(inputs)
        actions, _ = policy.sample_action(batch)

        for action in actions:
            if action.action_type in NEEDS_ENTITY:
                assert action.entity_target is not None

    def test_card_targets_present(self):
        """Card-targeting actions have non-None card_target."""
        torch.manual_seed(3)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input(n_hand=5) for _ in range(16)]
        batch = collate_policy_inputs(inputs)
        actions, _ = policy.sample_action(batch)

        for action in actions:
            if action.action_type in NEEDS_CARDS:
                assert action.card_target is not None
                assert 1 <= len(action.card_target) <= 5

    def test_log_probs_finite(self):
        torch.manual_seed(4)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(4)]
        batch = collate_policy_inputs(inputs)
        _, log_probs = policy.sample_action(batch)

        assert torch.isfinite(log_probs["total"]).all()
        assert torch.isfinite(log_probs["type"]).all()


class TestEvaluateActions:
    def test_finite_outputs(self):
        torch.manual_seed(5)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(4)]
        batch = collate_policy_inputs(inputs)

        # Sample actions first
        actions, _ = policy.sample_action(batch)

        # Evaluate
        log_probs, entropy, values = policy.evaluate_actions(batch, actions)

        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert values.shape == (4,)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(values).all()

    def test_log_probs_negative(self):
        """Log-probabilities should be <= 0."""
        torch.manual_seed(6)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(4)]
        batch = collate_policy_inputs(inputs)
        actions, _ = policy.sample_action(batch)
        log_probs, _, _ = policy.evaluate_actions(batch, actions)

        assert (log_probs <= 0.0).all()


class TestMasking:
    def test_masked_types_zero_probability(self):
        """Invalid action types receive ~0 probability."""
        torch.manual_seed(7)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inp = _make_input()
        batch = collate_policy_inputs([inp])
        out = policy.forward(batch)

        probs = torch.softmax(out["type_logits"][0], dim=0)
        type_mask = torch.from_numpy(inp.action_mask.type_mask)
        assert (probs[~type_mask] < 1e-6).all()

    def test_masked_cards_zero_probability(self):
        """Masked card positions receive ~0 selection probability."""
        torch.manual_seed(8)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        # Create input with some cards masked
        inp = _make_input(n_hand=5)
        inp.action_mask.card_mask[3] = False
        inp.action_mask.card_mask[4] = False
        batch = collate_policy_inputs([inp])
        out = policy.forward(batch)

        card_probs = torch.sigmoid(out["card_logits"][0])
        assert card_probs[3] < 1e-6
        assert card_probs[4] < 1e-6


class TestVariableLength:
    def test_different_sizes(self):
        """Batch items with different entity counts work correctly."""
        torch.manual_seed(9)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [
            _make_input(n_hand=3, n_joker=1, n_cons=0, n_shop=2),
            _make_input(n_hand=8, n_joker=5, n_cons=3, n_shop=0),
            _make_input(n_hand=1, n_joker=0, n_cons=0, n_shop=0, phase="blind_select"),
        ]
        batch = collate_policy_inputs(inputs)
        out = policy.forward(batch)

        assert out["type_logits"].shape == (3, NUM_ACTION_TYPES)
        assert out["value"].shape == (3, 1)

    def test_sample_with_variable_sizes(self):
        torch.manual_seed(10)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [
            _make_input(n_hand=5, n_joker=2, n_cons=1, n_shop=3),
            _make_input(n_hand=2, n_joker=0, n_cons=0, n_shop=0, phase="blind_select"),
        ]
        batch = collate_policy_inputs(inputs)
        actions, log_probs = policy.sample_action(batch)

        assert len(actions) == 2
        assert torch.isfinite(log_probs["total"]).all()


class TestGradientFlow:
    def test_gradients_through_forward(self):
        """loss.backward() from forward outputs produces gradients for all parameters."""
        torch.manual_seed(11)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(2)]
        batch = collate_policy_inputs(inputs)
        out = policy.forward(batch)

        # Combine all differentiable outputs
        loss = out["type_logits"].sum() + out["card_logits"].sum() + out["value"].sum()

        # Also compute entity logits to test that path
        pmask = torch.ones(2, out["entity_reprs"].shape[1], dtype=torch.bool)
        e_lgts = policy.action_heads.entity_logits(out["global_repr"], out["entity_reprs"], pmask)
        loss = loss + e_lgts.sum()

        loss.backward()

        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradients_through_evaluate(self):
        """evaluate_actions produces gradients for PPO training."""
        torch.manual_seed(12)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [
            _make_input(n_hand=5, phase="selecting_hand"),
            _make_input(n_shop=3, phase="shop"),
        ]
        batch = collate_policy_inputs(inputs)

        # Sample actions (no_grad)
        actions, _ = policy.sample_action(batch)

        # Evaluate (with grad)
        policy.train()
        log_probs, entropy, values = policy.evaluate_actions(batch, actions)
        loss = (-log_probs).sum() + values.sum()
        loss.backward()

        # At minimum, type_head and value_head should have gradients
        assert policy.action_heads.type_head.weight.grad is not None
        assert policy.action_heads.value_head[0].weight.grad is not None


class TestPackPhase:
    def test_pack_card_selection(self):
        """Pack phase with entity-targeting works."""
        torch.manual_seed(13)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input(n_hand=0, n_joker=0, n_cons=0, n_shop=0, n_pack=3, phase="pack")]
        batch = collate_policy_inputs(inputs)
        actions, log_probs = policy.sample_action(batch)

        assert len(actions) == 1
        action = actions[0]
        if action.action_type == ActionType.PickPackCard:
            assert action.entity_target is not None
            assert 0 <= action.entity_target < 3
