"""Tests for the Transformer-based policy network."""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.engine.actions import GamePhase, SelectBlind
from jackdaw.env.action_space import (
    NUM_ACTION_TYPES,
    ActionMask,
    ActionType,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
    encode_observation,
)
from jackdaw.env.policy.action_heads import NEEDS_CARDS, NEEDS_ENTITY
from jackdaw.env.policy.policy import (
    BalatroPolicy,
    PolicyInput,
    collate_policy_inputs,
)
from jackdaw.env.training.ppo import _compute_shop_splits

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
        actions, log_probs, _ = policy.sample_action(batch)

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
        actions, _, _ = policy.sample_action(batch)

        for action in actions:
            if action.action_type in NEEDS_ENTITY:
                assert action.entity_target is not None

    def test_card_targets_present(self):
        """Card-targeting actions have non-None card_target."""
        torch.manual_seed(3)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input(n_hand=5) for _ in range(16)]
        batch = collate_policy_inputs(inputs)
        actions, _, _ = policy.sample_action(batch)

        for action in actions:
            if action.action_type in NEEDS_CARDS:
                assert action.card_target is not None
                assert 1 <= len(action.card_target) <= 5

    def test_log_probs_finite(self):
        torch.manual_seed(4)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(4)]
        batch = collate_policy_inputs(inputs)
        _, log_probs, _ = policy.sample_action(batch)

        assert torch.isfinite(log_probs["total"]).all()
        assert torch.isfinite(log_probs["type"]).all()


class TestEvaluateActions:
    def test_finite_outputs(self):
        torch.manual_seed(5)
        policy = BalatroPolicy(embed_dim=64, num_heads=2, num_layers=1)
        inputs = [_make_input() for _ in range(4)]
        batch = collate_policy_inputs(inputs)

        # Sample actions first
        actions, _, _ = policy.sample_action(batch)

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
        actions, _, _ = policy.sample_action(batch)
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
        actions, log_probs, _ = policy.sample_action(batch)

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
        actions, _, _ = policy.sample_action(batch)

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
        actions, log_probs, _ = policy.sample_action(batch)

        assert len(actions) == 1
        action = actions[0]
        if action.action_type == ActionType.PickPackCard:
            assert action.entity_target is not None
            assert 0 <= action.entity_target < 3


# ---------------------------------------------------------------------------
# Tests with real game observations
# ---------------------------------------------------------------------------


def _make_real_policy_input(gs: dict) -> PolicyInput:
    """Build a PolicyInput from a real game state with correct shop_splits."""
    obs = encode_observation(gs)
    mask = get_action_mask(gs)
    ss = _compute_shop_splits(gs)
    return PolicyInput(obs=obs, action_mask=mask, shop_splits=ss)


def _collect_diverse_game_states(n: int = 8) -> list[dict]:
    """Collect game states from diverse phases by playing with HeuristicAgent."""
    from jackdaw.env.agents import HeuristicAgent

    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "POLICY_TEST_DIVERSE")
    agent = HeuristicAgent()
    agent.reset()

    states: list[dict] = []
    seen_phases: set[str] = set()
    gs = adapter.raw_state

    for _ in range(5000):
        if adapter.done:
            break
        phase = gs.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break

        # Collect from diverse phases
        phase_str = str(phase)
        if phase_str not in seen_phases or len(states) < n:
            states.append(gs)
            seen_phases.add(phase_str)

        if len(states) >= n and len(seen_phases) >= 3:
            break

        legal = adapter.get_legal_actions()
        if not legal:
            break

        mask = get_action_mask(gs)
        info = {"raw_state": gs, "legal_actions": legal}
        fa = agent.act({}, mask, info)
        engine_action = factored_to_engine_action(fa, gs)
        adapter.step(engine_action)
        gs = adapter.raw_state

    return states[:n]


class TestRealObservationForward:
    """Test 1: Single real observation forward pass — no NaN."""

    def test_single_observation_no_nan(self):
        torch.manual_seed(100)
        adapter = DirectAdapter()
        adapter.reset("b_red", 1, "POLICY_SINGLE_OBS")
        gs = adapter.raw_state

        pi = _make_real_policy_input(gs)
        batch = collate_policy_inputs([pi])

        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        policy.eval()
        with torch.no_grad():
            out = policy.forward(batch)

        assert torch.isfinite(out["type_logits"]).all(), "NaN in type_logits"
        assert torch.isfinite(out["card_logits"]).all(), "NaN in card_logits"
        assert torch.isfinite(out["value"]).all(), "NaN in value"
        assert torch.isfinite(out["global_repr"]).all(), "NaN in global_repr"
        assert torch.isfinite(out["entity_reprs"]).all(), "NaN in entity_reprs"

    def test_multiple_phases_no_nan(self):
        """Forward pass through blind_select, selecting_hand, shop states."""
        torch.manual_seed(101)
        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        policy.eval()

        adapter = DirectAdapter()
        adapter.reset("b_red", 1, "POLICY_PHASES")

        # blind_select phase
        gs = adapter.raw_state
        assert gs.get("phase") == GamePhase.BLIND_SELECT
        pi = _make_real_policy_input(gs)
        batch = collate_policy_inputs([pi])
        with torch.no_grad():
            out = policy.forward(batch)
        assert torch.isfinite(out["type_logits"]).all()
        assert torch.isfinite(out["value"]).all()

        # Advance to selecting_hand
        adapter.step(SelectBlind())
        gs = adapter.raw_state
        assert gs.get("phase") == GamePhase.SELECTING_HAND
        pi = _make_real_policy_input(gs)
        batch = collate_policy_inputs([pi])
        with torch.no_grad():
            out = policy.forward(batch)
        assert torch.isfinite(out["type_logits"]).all()
        assert torch.isfinite(out["card_logits"]).all()
        assert torch.isfinite(out["value"]).all()


class TestBatchedRealObservations:
    """Test 2: Batched forward with variable-length real observations."""

    def test_batched_diverse_states(self):
        torch.manual_seed(200)
        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        policy.eval()

        states = _collect_diverse_game_states(8)
        assert len(states) >= 3, "Need at least 3 diverse game states"

        policy_inputs = [_make_real_policy_input(gs) for gs in states]
        batch = collate_policy_inputs(policy_inputs)
        B = len(states)

        with torch.no_grad():
            out = policy.forward(batch)

        # Shape checks
        assert out["type_logits"].shape == (B, NUM_ACTION_TYPES)
        assert out["value"].shape == (B, 1)
        assert out["global_repr"].shape[0] == B
        assert out["entity_reprs"].shape[0] == B

        # No NaN
        assert torch.isfinite(out["type_logits"]).all(), "NaN in batched type_logits"
        assert torch.isfinite(out["card_logits"]).all(), "NaN in batched card_logits"
        assert torch.isfinite(out["value"]).all(), "NaN in batched value"
        assert torch.isfinite(out["global_repr"]).all(), "NaN in batched global_repr"
        assert torch.isfinite(out["entity_reprs"]).all(), "NaN in batched entity_reprs"


class TestSampleActionRealObs:
    """Test 3: sample_action produces valid FactoredActions from real observations."""

    def test_sampled_actions_valid(self):
        torch.manual_seed(300)
        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)

        states = _collect_diverse_game_states(8)
        policy_inputs = [_make_real_policy_input(gs) for gs in states]
        batch = collate_policy_inputs(policy_inputs)

        actions, log_probs, values = policy.sample_action(batch)

        assert len(actions) == len(states)
        assert torch.isfinite(log_probs["total"]).all()
        assert torch.isfinite(values).all()

        for i, (action, gs, pi) in enumerate(zip(actions, states, policy_inputs)):
            # action_type is valid and legal
            assert 0 <= action.action_type < NUM_ACTION_TYPES, (
                f"Action {i}: invalid type {action.action_type}"
            )
            assert pi.action_mask.type_mask[action.action_type], (
                f"Action {i}: type {action.action_type} not legal"
            )

            # entity_target present when needed, within bounds
            if action.action_type in NEEDS_ENTITY:
                if action.entity_target is not None:
                    assert action.entity_target >= 0, (
                        f"Action {i}: negative entity_target {action.entity_target}"
                    )

            # card_target within hand size
            if action.action_type in NEEDS_CARDS and action.card_target:
                hand_size = len(gs.get("hand", []))
                for idx in action.card_target:
                    assert 0 <= idx < hand_size, (
                        f"Action {i}: card index {idx} out of range "
                        f"(hand_size={hand_size})"
                    )

            # Converts to engine action without error
            try:
                factored_to_engine_action(action, gs)
            except (ValueError, IndexError) as e:
                pytest.fail(
                    f"Action {i}: failed engine conversion: {e}\n"
                    f"  action={action}\n"
                    f"  phase={gs.get('phase')}"
                )


class TestEvaluateActionsGradientFlow:
    """Test 4: evaluate_actions gradient flow with real observations."""

    def test_no_nan_gradients(self):
        torch.manual_seed(400)
        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)

        states = _collect_diverse_game_states(6)
        policy_inputs = [_make_real_policy_input(gs) for gs in states]
        batch = collate_policy_inputs(policy_inputs)

        # Sample actions (no grad)
        actions, _, _ = policy.sample_action(batch)

        # Evaluate with grad
        policy.train()
        log_probs, entropy, values = policy.evaluate_actions(batch, actions)

        # Compute PPO-style loss
        loss = -(log_probs + 0.01 * entropy).mean() + 0.5 * values.mean() ** 2
        loss.backward()

        # Check loss is finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        # Check no NaN in any gradient that was computed.
        # Some heads (e.g. card_scorer) may not participate if no sampled
        # action required card selection — that's fine, just check what's there.
        params_with_grad = 0
        for name, param in policy.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"NaN/Inf gradient in {name}"
                )
                params_with_grad += 1

        assert params_with_grad > 0, "No parameters had gradients"

    def test_multiple_backward_passes(self):
        """Multiple training steps don't accumulate NaN."""
        torch.manual_seed(401)
        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        states = _collect_diverse_game_states(4)
        policy_inputs = [_make_real_policy_input(gs) for gs in states]
        batch = collate_policy_inputs(policy_inputs)

        actions, _, _ = policy.sample_action(batch)

        for step in range(5):
            policy.train()
            log_probs, entropy, values = policy.evaluate_actions(batch, actions)
            loss = -(log_probs + 0.01 * entropy).mean() + 0.5 * values.mean() ** 2

            optimizer.zero_grad()
            loss.backward()

            for name, param in policy.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), (
                        f"NaN gradient in {name} at step {step}"
                    )

            optimizer.step()

            # Check parameters are still finite after optimizer step
            for name, param in policy.named_parameters():
                assert torch.isfinite(param).all(), (
                    f"NaN parameter in {name} after step {step}"
                )


class TestCollationPointerMasks:
    """Test 5: Pointer masks correctly point to entity regions."""

    def _make_shop_input(
        self,
        n_hand: int = 5,
        n_joker: int = 2,
        n_cons: int = 1,
        n_shop_cards: int = 2,
        n_vouchers: int = 1,
        n_boosters: int = 1,
        n_pack: int = 0,
    ) -> PolicyInput:
        """Create an input with known shop splits and entity masks."""
        rng = np.random.default_rng(42)
        n_shop_total = n_shop_cards + n_vouchers + n_boosters

        obs = Observation(
            global_context=rng.standard_normal(D_GLOBAL).astype(np.float32),
            hand_cards=rng.standard_normal((n_hand, D_PLAYING_CARD)).astype(np.float32),
            jokers=rng.standard_normal((n_joker, D_JOKER)).astype(np.float32),
            consumables=rng.standard_normal((n_cons, D_CONSUMABLE)).astype(np.float32),
            shop_cards=rng.standard_normal((n_shop_total, D_SHOP)).astype(np.float32),
            pack_cards=rng.standard_normal((n_pack, D_PLAYING_CARD)).astype(np.float32),
        )

        type_mask = np.zeros(NUM_ACTION_TYPES, dtype=bool)
        entity_masks: dict[int, np.ndarray] = {}

        # BuyCard (8) — targets shop_cards portion
        if n_shop_cards > 0:
            type_mask[ActionType.BuyCard] = True
            entity_masks[ActionType.BuyCard] = np.ones(n_shop_cards, dtype=bool)

        # SellJoker (9) — targets joker region
        if n_joker > 0:
            type_mask[ActionType.SellJoker] = True
            entity_masks[ActionType.SellJoker] = np.ones(n_joker, dtype=bool)

        # RedeemVoucher (12) — targets voucher portion of shop
        if n_vouchers > 0:
            type_mask[ActionType.RedeemVoucher] = True
            entity_masks[ActionType.RedeemVoucher] = np.ones(n_vouchers, dtype=bool)

        # OpenBooster (13) — targets booster portion of shop
        if n_boosters > 0:
            type_mask[ActionType.OpenBooster] = True
            entity_masks[ActionType.OpenBooster] = np.ones(n_boosters, dtype=bool)

        # PickPackCard (14) — targets pack region
        if n_pack > 0:
            type_mask[ActionType.PickPackCard] = True
            entity_masks[ActionType.PickPackCard] = np.ones(n_pack, dtype=bool)

        # NextRound so there's always a legal action
        type_mask[ActionType.NextRound] = True

        card_mask = np.ones(n_hand, dtype=bool) if n_hand > 0 else np.zeros(0, dtype=bool)
        mask = ActionMask(type_mask, card_mask, entity_masks, 5, 1)

        return PolicyInput(
            obs=obs,
            action_mask=mask,
            shop_splits=(n_shop_cards, n_vouchers, n_boosters),
        )

    def test_buycard_mask_spans_shop_cards_region(self):
        """BuyCard (8) pointer mask is in the shop_cards portion only."""
        pi = self._make_shop_input(
            n_hand=3, n_joker=2, n_cons=1,
            n_shop_cards=2, n_vouchers=1, n_boosters=1,
        )
        batch = collate_policy_inputs([pi])
        pmask = batch["pointer_masks"][0, ActionType.BuyCard]

        # shop region starts at hand + joker + cons = 3 + 2 + 1 = 6
        shop_start = 3 + 2 + 1
        n_shop_cards = 2

        # BuyCard mask should be True in [shop_start, shop_start + n_shop_cards)
        for i in range(len(pmask)):
            if shop_start <= i < shop_start + n_shop_cards:
                assert pmask[i].item(), (
                    f"BuyCard mask should be True at position {i}"
                )
            else:
                assert not pmask[i].item(), (
                    f"BuyCard mask should be False at position {i}"
                )

    def test_redeemvoucher_mask_spans_voucher_region(self):
        """RedeemVoucher (12) pointer mask is in the voucher portion only."""
        pi = self._make_shop_input(
            n_hand=3, n_joker=2, n_cons=1,
            n_shop_cards=2, n_vouchers=1, n_boosters=1,
        )
        batch = collate_policy_inputs([pi])
        pmask = batch["pointer_masks"][0, ActionType.RedeemVoucher]

        shop_start = 3 + 2 + 1
        voucher_start = shop_start + 2  # after shop_cards
        n_vouchers = 1

        for i in range(len(pmask)):
            if voucher_start <= i < voucher_start + n_vouchers:
                assert pmask[i].item(), (
                    f"RedeemVoucher mask should be True at position {i}"
                )
            else:
                assert not pmask[i].item(), (
                    f"RedeemVoucher mask should be False at position {i}"
                )

    def test_openbooster_mask_spans_booster_region(self):
        """OpenBooster (13) pointer mask is in the booster portion only."""
        pi = self._make_shop_input(
            n_hand=3, n_joker=2, n_cons=1,
            n_shop_cards=2, n_vouchers=1, n_boosters=1,
        )
        batch = collate_policy_inputs([pi])
        pmask = batch["pointer_masks"][0, ActionType.OpenBooster]

        shop_start = 3 + 2 + 1
        booster_start = shop_start + 2 + 1  # after shop_cards + vouchers
        n_boosters = 1

        for i in range(len(pmask)):
            if booster_start <= i < booster_start + n_boosters:
                assert pmask[i].item(), (
                    f"OpenBooster mask should be True at position {i}"
                )
            else:
                assert not pmask[i].item(), (
                    f"OpenBooster mask should be False at position {i}"
                )

    def test_selljoker_mask_spans_joker_region(self):
        """SellJoker (9) pointer mask is in the joker region only."""
        pi = self._make_shop_input(
            n_hand=3, n_joker=2, n_cons=1,
            n_shop_cards=2, n_vouchers=0, n_boosters=0,
        )
        batch = collate_policy_inputs([pi])
        pmask = batch["pointer_masks"][0, ActionType.SellJoker]

        joker_start = 3  # after hand
        n_joker = 2

        for i in range(len(pmask)):
            if joker_start <= i < joker_start + n_joker:
                assert pmask[i].item(), (
                    f"SellJoker mask should be True at position {i}"
                )
            else:
                assert not pmask[i].item(), (
                    f"SellJoker mask should be False at position {i}"
                )

    def test_pickpackcard_mask_spans_pack_region(self):
        """PickPackCard (14) pointer mask is in the pack region only."""
        pi = self._make_shop_input(
            n_hand=3, n_joker=2, n_cons=1,
            n_shop_cards=0, n_vouchers=0, n_boosters=0,
            n_pack=3,
        )
        batch = collate_policy_inputs([pi])
        pmask = batch["pointer_masks"][0, ActionType.PickPackCard]

        # pack starts after hand + joker + cons + shop = 3 + 2 + 1 + 0 = 6
        pack_start = 3 + 2 + 1 + 0
        n_pack = 3

        for i in range(len(pmask)):
            if pack_start <= i < pack_start + n_pack:
                assert pmask[i].item(), (
                    f"PickPackCard mask should be True at position {i}"
                )
            else:
                assert not pmask[i].item(), (
                    f"PickPackCard mask should be False at position {i}"
                )

    def test_pointer_roundtrip_with_shop_splits(self):
        """Pointer index -> entity_target -> pointer index roundtrips correctly."""
        from jackdaw.env.policy.policy import (
            _entity_target_to_pointer,
            _pointer_to_entity_target,
        )

        pi = self._make_shop_input(
            n_hand=4, n_joker=3, n_cons=2,
            n_shop_cards=2, n_vouchers=1, n_boosters=1,
            n_pack=2,
        )
        batch = collate_policy_inputs([pi])
        offsets = batch["entity_offsets"]
        splits = batch["shop_splits"][0]

        # Test each action type that targets entities
        test_cases = [
            # (action_type, entity_target)
            (ActionType.BuyCard, 0),
            (ActionType.BuyCard, 1),
            (ActionType.SellJoker, 0),
            (ActionType.SellJoker, 2),
            (ActionType.RedeemVoucher, 0),
            (ActionType.OpenBooster, 0),
            (ActionType.PickPackCard, 0),
            (ActionType.PickPackCard, 1),
        ]

        for at, et in test_cases:
            ptr = _entity_target_to_pointer(et, at, offsets, splits)
            recovered_et = _pointer_to_entity_target(ptr, at, offsets, splits)
            assert recovered_et == et, (
                f"Roundtrip failed for action_type={at}, entity_target={et}: "
                f"ptr={ptr}, recovered={recovered_et}"
            )

    def test_batched_shop_splits_different_per_item(self):
        """Different batch items can have different shop splits."""
        pi1 = self._make_shop_input(
            n_hand=3, n_joker=1, n_cons=0,
            n_shop_cards=3, n_vouchers=0, n_boosters=0,
        )
        pi2 = self._make_shop_input(
            n_hand=3, n_joker=1, n_cons=0,
            n_shop_cards=1, n_vouchers=1, n_boosters=1,
        )
        batch = collate_policy_inputs([pi1, pi2])

        # pi1: all 3 shop items are cards
        assert batch["shop_splits"][0].tolist() == [3, 0, 0]
        # pi2: 1 card, 1 voucher, 1 booster
        assert batch["shop_splits"][1].tolist() == [1, 1, 1]

        # BuyCard mask for pi1: 3 items starting at shop_start
        shop_start = 3 + 1 + 0  # hand + joker + cons (max across batch)
        pmask1 = batch["pointer_masks"][0, ActionType.BuyCard]
        assert pmask1[shop_start:shop_start + 3].all()

        # BuyCard mask for pi2: only 1 item
        pmask2 = batch["pointer_masks"][1, ActionType.BuyCard]
        assert pmask2[shop_start].item()
        assert not pmask2[shop_start + 1].item()  # voucher, not buyable

        # RedeemVoucher for pi2: at shop_start + 1
        vmask2 = batch["pointer_masks"][1, ActionType.RedeemVoucher]
        assert vmask2[shop_start + 1].item()
        assert not vmask2[shop_start].item()
        assert not vmask2[shop_start + 2].item()
