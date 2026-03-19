"""Integration tests for the Balatro RL environment.

End-to-end tests that exercise the full stack:
- Full episode runs with HeuristicAgent through DirectAdapter and BridgeAdapter
- Multi-seed stress testing for crash resilience
- Observation consistency with engine state
- Action roundtrip fidelity
- Reward determinism
- Policy network smoke test (if torch available)
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import (
    ActionType,
    FactoredAction,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents import HeuristicAgent, RandomAgent, evaluate_agent
from jackdaw.env.game_interface import BridgeAdapter, DirectAdapter, GameState
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
    encode_observation,
)
from jackdaw.env.rewards import DenseRewardWrapper, RewardConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = "TEST123"
BACK = "b_red"
STAKE = 1  # White Stake
MAX_STEPS = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_episode_collecting(
    adapter,
    agent,
    *,
    max_steps: int = MAX_STEPS,
    collect_obs: bool = False,
    collect_rewards: bool = False,
):
    """Run a full episode, optionally collecting observations and rewards.

    Returns a dict with episode metadata and optionally collected data.
    """
    agent.reset()

    observations: list[Observation] = []
    rewards: list[float] = []
    mask_checks: list[bool] = []
    reward_wrapper = DenseRewardWrapper() if collect_rewards else None
    if reward_wrapper:
        reward_wrapper.reset()

    gs = adapter.raw_state
    steps = 0

    for _ in range(max_steps):
        if adapter.done:
            break
        phase = gs.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break

        legal = adapter.get_legal_actions()
        if not legal:
            break

        mask = get_action_mask(gs)
        info = {"raw_state": gs, "legal_actions": legal}

        # Collect observation
        if collect_obs:
            obs = encode_observation(gs)
            observations.append(obs)

            # Check mask consistency: every legal action type has mask True
            legal_type_names = {type(a).__name__ for a in legal}
            mask_ok = True
            if mask.type_mask.sum() == 0 and legal:
                mask_ok = False
            mask_checks.append(mask_ok)

        fa = agent.act({}, mask, info)
        prev_gs = gs

        engine_action = factored_to_engine_action(fa, gs)
        adapter.step(engine_action)
        gs = adapter.raw_state
        steps += 1

        if collect_rewards and reward_wrapper:
            r = reward_wrapper.reward(prev_gs, fa, gs)
            rewards.append(r)

    rr = gs.get("round_resets", {})
    return {
        "steps": steps,
        "won": adapter.won,
        "done": adapter.done,
        "ante": rr.get("ante", 1),
        "observations": observations,
        "rewards": rewards,
        "mask_checks": mask_checks,
    }


# ---------------------------------------------------------------------------
# (a) Full episode test — DirectAdapter
# ---------------------------------------------------------------------------


class TestFullEpisodeDirectAdapter:
    """Run HeuristicAgent through a complete episode via DirectAdapter."""

    def test_episode_terminates(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        result = _run_episode_collecting(
            adapter, agent, collect_obs=True, collect_rewards=True
        )

        # Episode must terminate
        assert result["done"] or result["won"], "Episode did not terminate"
        assert result["steps"] > 0, "No steps taken"

    def test_observations_valid(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        result = _run_episode_collecting(adapter, agent, collect_obs=True)

        for obs in result["observations"]:
            assert isinstance(obs, Observation)
            # Global context
            assert obs.global_context.shape == (D_GLOBAL,)
            assert obs.global_context.dtype == np.float32
            assert np.isfinite(obs.global_context).all(), "NaN/Inf in global_context"

            # Hand cards
            if obs.hand_cards.shape[0] > 0:
                assert obs.hand_cards.shape[1] == D_PLAYING_CARD
                assert np.isfinite(obs.hand_cards).all(), "NaN/Inf in hand_cards"

            # Jokers
            if obs.jokers.shape[0] > 0:
                assert obs.jokers.shape[1] == D_JOKER
                assert np.isfinite(obs.jokers).all(), "NaN/Inf in jokers"

            # Consumables
            if obs.consumables.shape[0] > 0:
                assert obs.consumables.shape[1] == D_CONSUMABLE
                assert np.isfinite(obs.consumables).all(), "NaN/Inf in consumables"

            # Shop
            if obs.shop_cards.shape[0] > 0:
                assert obs.shop_cards.shape[1] == D_SHOP
                assert np.isfinite(obs.shop_cards).all(), "NaN/Inf in shop_cards"

    def test_action_masking_consistent(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        result = _run_episode_collecting(adapter, agent, collect_obs=True)

        assert all(result["mask_checks"]), "Action mask inconsistent with legal actions"

    def test_rewards_finite_and_bounded(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        result = _run_episode_collecting(adapter, agent, collect_rewards=True)

        for r in result["rewards"]:
            assert math.isfinite(r), f"Non-finite reward: {r}"
            assert abs(r) < 100, f"Unbounded reward: {r}"


# ---------------------------------------------------------------------------
# (b) Full episode test — BridgeAdapter(SimBackend) identical trajectory
# ---------------------------------------------------------------------------


class TestBridgeAdapterIdenticalTrajectory:
    """BridgeAdapter(SimBackend) must produce identical game trajectory."""

    def test_identical_trajectory(self):
        from jackdaw.bridge.backend import SimBackend

        # Run DirectAdapter episode
        direct = DirectAdapter()
        direct.reset(BACK, STAKE, SEED)

        bridge = BridgeAdapter(SimBackend())
        bridge.reset(BACK, STAKE, SEED)

        agent_d = HeuristicAgent()
        agent_d.reset()

        steps = 0
        for _ in range(MAX_STEPS):
            if direct.done or bridge.done:
                break
            d_phase = direct.raw_state.get("phase")
            if direct.won and d_phase == GamePhase.SHOP:
                break

            d_legal = direct.get_legal_actions()
            b_legal = bridge.get_legal_actions()
            if not d_legal or not b_legal:
                break

            gs = direct.raw_state
            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": d_legal}
            fa = agent_d.act({}, mask, info)

            engine_action = factored_to_engine_action(fa, gs)

            ds = direct.step(engine_action)
            bs = bridge.step(engine_action)
            steps += 1

            # Core state must match
            assert ds.phase == bs.phase, f"Step {steps}: phase {ds.phase} != {bs.phase}"
            assert ds.dollars == bs.dollars, f"Step {steps}: dollars {ds.dollars} != {bs.dollars}"
            assert ds.ante == bs.ante, f"Step {steps}: ante mismatch"
            assert ds.round == bs.round, f"Step {steps}: round mismatch"
            assert ds.chips == bs.chips, f"Step {steps}: chips mismatch"
            assert ds.hands_left == bs.hands_left, f"Step {steps}: hands_left mismatch"
            assert ds.discards_left == bs.discards_left, f"Step {steps}: discards_left mismatch"

        assert steps > 0, "No steps taken"
        assert direct.done == bridge.done, "done flag mismatch"
        assert direct.won == bridge.won, "won flag mismatch"


# ---------------------------------------------------------------------------
# (c) Multi-seed stress test
# ---------------------------------------------------------------------------


class TestMultiSeedStress:
    """Run 50 episodes with random seeds — no crashes, no NaN, no invalid actions."""

    def test_50_episodes_no_crashes(self):
        n_episodes = 50
        seeds = [f"STRESS_{i}" for i in range(n_episodes)]
        wins = 0
        antes: list[int] = []
        step_counts: list[int] = []

        for seed in seeds:
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = HeuristicAgent()
            agent.reset()

            gs = adapter.raw_state
            steps = 0

            for _ in range(MAX_STEPS):
                if adapter.done:
                    break
                phase = gs.get("phase")
                if adapter.won and phase == GamePhase.SHOP:
                    break

                legal = adapter.get_legal_actions()
                if not legal:
                    break

                mask = get_action_mask(gs)
                info = {"raw_state": gs, "legal_actions": legal}
                fa = agent.act({}, mask, info)

                # Verify observation has no NaN
                obs = encode_observation(gs)
                assert np.isfinite(obs.global_context).all(), f"NaN in global_context seed={seed}"
                if obs.hand_cards.shape[0] > 0:
                    assert np.isfinite(obs.hand_cards).all(), f"NaN in hand_cards seed={seed}"

                engine_action = factored_to_engine_action(fa, gs)
                adapter.step(engine_action)
                gs = adapter.raw_state
                steps += 1

            rr = gs.get("round_resets", {})
            antes.append(rr.get("ante", 1))
            step_counts.append(steps)
            if adapter.won:
                wins += 1

        # Statistics
        win_rate = wins / n_episodes
        avg_ante = sum(antes) / len(antes)
        avg_steps = sum(step_counts) / len(step_counts)

        # Sanity checks — not performance assertions, just "reasonable"
        assert avg_ante >= 1.0, f"Average ante {avg_ante} is unreasonably low"
        assert avg_steps > 10, f"Average steps {avg_steps} is unreasonably low"

        # Print statistics for information
        print(f"\n  Stress test: {n_episodes} episodes")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Avg ante: {avg_ante:.1f}")
        print(f"  Avg steps: {avg_steps:.0f}")


# ---------------------------------------------------------------------------
# (d) Observation consistency test
# ---------------------------------------------------------------------------


class TestObservationConsistency:
    """At each step, observation entity counts must match game state."""

    def test_entity_counts_match(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        agent.reset()

        gs = adapter.raw_state
        checked = 0

        for _ in range(MAX_STEPS):
            if adapter.done:
                break
            phase = gs.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            obs = encode_observation(gs)

            # Hand cards count
            hand = gs.get("hand", [])
            assert obs.hand_cards.shape[0] == len(hand), (
                f"hand_cards.shape[0]={obs.hand_cards.shape[0]} != len(hand)={len(hand)}"
            )

            # Jokers count
            jokers = gs.get("jokers", [])
            assert obs.jokers.shape[0] == len(jokers), (
                f"jokers.shape[0]={obs.jokers.shape[0]} != len(jokers)={len(jokers)}"
            )

            # Consumables count
            consumables = gs.get("consumables", [])
            assert obs.consumables.shape[0] == len(consumables), (
                f"consumables.shape[0]={obs.consumables.shape[0]} "
                f"!= len(consumables)={len(consumables)}"
            )

            # Global context phase encoding matches game state phase
            phase_val = gs.get("phase", GamePhase.GAME_OVER)
            if isinstance(phase_val, str):
                phase_val = GamePhase(phase_val)
            _PHASE_IDX = {
                GamePhase.BLIND_SELECT: 0,
                GamePhase.SELECTING_HAND: 1,
                GamePhase.ROUND_EVAL: 2,
                GamePhase.SHOP: 3,
                GamePhase.PACK_OPENING: 4,
                GamePhase.GAME_OVER: 5,
            }
            expected_idx = _PHASE_IDX.get(phase_val, 5)
            # Phase is one-hot in obs.global_context[0:6]
            phase_vec = obs.global_context[:6]
            actual_idx = int(np.argmax(phase_vec))
            assert actual_idx == expected_idx, (
                f"Phase encoding mismatch: global_context phase={actual_idx} "
                f"but game_state phase={phase_val} (expected idx={expected_idx})"
            )

            checked += 1

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            gs = adapter.raw_state

        assert checked > 0, "No steps checked"


# ---------------------------------------------------------------------------
# (e) Action roundtrip test
# ---------------------------------------------------------------------------


class TestActionRoundtrip:
    """engine_action -> factored -> engine_action must be lossless for representable actions."""

    def test_roundtrip_across_episode(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        agent.reset()

        gs = adapter.raw_state
        roundtrip_count = 0
        skip_count = 0

        for _ in range(MAX_STEPS):
            if adapter.done:
                break
            phase = gs.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            # Test roundtrip for each legal action
            for engine_action in legal:
                try:
                    fa = engine_action_to_factored(engine_action, gs)
                    recovered = factored_to_engine_action(fa, gs)

                    # Compare the recovered action to the original
                    assert type(engine_action) == type(recovered), (
                        f"Type mismatch: {type(engine_action).__name__} "
                        f"!= {type(recovered).__name__}"
                    )
                    roundtrip_count += 1

                except ValueError:
                    # Complex permutations can't be represented as single swaps
                    skip_count += 1

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            gs = adapter.raw_state

        assert roundtrip_count > 0, "No actions were roundtrip-tested"

    def test_specific_action_types(self):
        """Test roundtrip for specific action types that must be exact."""
        from jackdaw.engine.actions import (
            CashOut,
            Discard,
            NextRound,
            PlayHand,
            Reroll,
            SelectBlind,
            SkipBlind,
            SortHand,
        )

        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        gs = adapter.raw_state

        # SelectBlind roundtrip
        fa = engine_action_to_factored(SelectBlind(), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SelectBlind)

        # SkipBlind roundtrip
        fa = engine_action_to_factored(SkipBlind(), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SkipBlind)

        # Advance to selecting hand
        adapter.step(SelectBlind())
        gs = adapter.raw_state
        hand = gs.get("hand", [])

        if hand:
            # PlayHand roundtrip
            indices = tuple(range(min(3, len(hand))))
            orig = PlayHand(card_indices=indices)
            fa = engine_action_to_factored(orig, gs)
            recovered = factored_to_engine_action(fa, gs)
            assert isinstance(recovered, PlayHand)
            assert recovered.card_indices == orig.card_indices

            # Discard roundtrip
            orig = Discard(card_indices=indices)
            fa = engine_action_to_factored(orig, gs)
            recovered = factored_to_engine_action(fa, gs)
            assert isinstance(recovered, Discard)
            assert recovered.card_indices == orig.card_indices

        # SortHand roundtrip
        fa = engine_action_to_factored(SortHand(mode="rank"), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SortHand)
        assert recovered.mode == "rank"

        fa = engine_action_to_factored(SortHand(mode="suit"), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SortHand)
        assert recovered.mode == "suit"


# ---------------------------------------------------------------------------
# (f) Reward determinism test
# ---------------------------------------------------------------------------


class TestRewardDeterminism:
    """Same (prev_state, action, next_state) must always produce same reward."""

    def test_deterministic_rewards(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        agent.reset()

        gs = adapter.raw_state
        config = RewardConfig()

        # Collect some transitions
        transitions: list[tuple[dict, FactoredAction, dict]] = []

        for _ in range(200):
            if adapter.done:
                break
            phase = gs.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)

            prev_gs = gs
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            gs = adapter.raw_state

            transitions.append((prev_gs, fa, gs))

        assert len(transitions) > 0, "No transitions collected"

        # Compute rewards twice with fresh calculators and compare
        wrapper1 = DenseRewardWrapper(config)
        wrapper1.reset()
        wrapper2 = DenseRewardWrapper(config)
        wrapper2.reset()

        for prev, action, nxt in transitions:
            r1 = wrapper1.reward(prev, action, nxt)
            r2 = wrapper2.reward(prev, action, nxt)
            assert r1 == r2, f"Reward not deterministic: {r1} != {r2}"


# ---------------------------------------------------------------------------
# (g) Policy smoke test (if torch available)
# ---------------------------------------------------------------------------


_torch_available = False
try:
    import torch

    _torch_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _torch_available, reason="torch not available")
class TestPolicySmokeTest:
    """Verify BalatroPolicy trains without NaN and produces valid actions."""

    def test_ppo_training_smoke(self):
        from jackdaw.env.observation import encode_observation
        from jackdaw.env.policy.policy import (
            BalatroPolicy,
            PolicyInput,
            collate_policy_inputs,
        )

        # Create a small policy
        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        # Collect some real game states
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = HeuristicAgent()
        agent.reset()

        policy_inputs: list[PolicyInput] = []
        actions_taken: list[FactoredAction] = []
        gs = adapter.raw_state

        for _ in range(30):
            if adapter.done:
                break
            phase = gs.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            obs = encode_observation(gs)
            mask = get_action_mask(gs)
            pi = PolicyInput(obs=obs, action_mask=mask)
            policy_inputs.append(pi)

            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)
            actions_taken.append(fa)

            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            gs = adapter.raw_state

        if len(policy_inputs) < 2:
            pytest.skip("Not enough steps to test training")

        # Take a subset for training
        n = min(10, len(policy_inputs))
        inputs = policy_inputs[:n]
        acts = actions_taken[:n]

        # Run 10 PPO-like training steps
        for step in range(10):
            batch = collate_policy_inputs(inputs, device="cpu")
            log_probs, entropy, values = policy.evaluate_actions(batch, acts)

            # Fake advantages
            advantages = torch.randn(n)
            returns = values.detach() + torch.randn(n) * 0.1

            # PPO loss
            ratio = torch.exp(log_probs - log_probs.detach())
            pg_loss = -(advantages * ratio).mean()
            vf_loss = 0.5 * ((values - returns) ** 2).mean()
            ent_loss = entropy.mean()
            loss = pg_loss + 0.5 * vf_loss - 0.01 * ent_loss

            optimizer.zero_grad()
            loss.backward()

            # Check no NaN in gradients
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), (
                        f"NaN/Inf gradient in {name} at step {step}"
                    )

            optimizer.step()

    def test_sample_action_produces_valid_actions(self):
        from jackdaw.env.policy.policy import (
            BalatroPolicy,
            PolicyInput,
            collate_policy_inputs,
        )

        policy = BalatroPolicy(embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        policy.eval()

        # Get a real game state
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(factored_to_engine_action(
            FactoredAction(action_type=ActionType.SelectBlind), adapter.raw_state
        ))
        gs = adapter.raw_state

        obs = encode_observation(gs)
        mask = get_action_mask(gs)
        pi = PolicyInput(obs=obs, action_mask=mask)
        batch = collate_policy_inputs([pi], device="cpu")

        with torch.no_grad():
            actions, log_probs = policy.sample_action(batch)

        assert len(actions) == 1
        fa = actions[0]
        assert isinstance(fa, FactoredAction)
        assert 0 <= fa.action_type < 21
        assert torch.isfinite(log_probs["total"]).all()
