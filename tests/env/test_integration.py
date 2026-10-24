"""Integration tests for the Balatro RL environment.

End-to-end tests that exercise the full stack:
- Full episode runs with RandomAgent through DirectAdapter and BridgeAdapter
- Multi-seed stress testing for crash resilience
- Observation consistency with engine state
- Action roundtrip fidelity
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from jackdaw.bridge.backend import SimBackend
from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import (
    ActionType,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents import RandomAgent
from jackdaw.env.game_interface import BridgeAdapter, DirectAdapter
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
    encode_observation,
)

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


def _finished(adapter) -> bool:
    return adapter.done or (adapter.won and adapter.game.phase == GamePhase.SHOP)


def _agent_step(adapter, agent) -> None:
    """One RandomAgent step on the adapter's run."""
    game = adapter.game
    info = {"game": game}
    fa = agent.act({}, get_action_mask(game), info)
    adapter.step(factored_to_engine_action(fa, game))


def _run_episode_collecting(
    adapter,
    agent,
    *,
    max_steps: int = MAX_STEPS,
    collect_obs: bool = False,
):
    """Run a full episode, optionally collecting observations.

    Returns a dict with episode metadata and optionally collected data.
    """
    agent.reset()

    observations: list[Observation] = []
    mask_checks: list[bool] = []

    steps = 0

    for _ in range(max_steps):
        if _finished(adapter):
            break

        legal = adapter.get_legal_actions()
        if not legal:
            break

        game = adapter.game
        mask = get_action_mask(game)

        # Collect observation
        if collect_obs:
            observations.append(encode_observation(game))

            # Check mask consistency: some action type is legal while legal actions exist
            mask_checks.append(bool(mask.type_mask.any()))

        fa = agent.act({}, mask, {"game": game, "legal_actions": legal})
        adapter.step(factored_to_engine_action(fa, game))
        steps += 1

    return {
        "steps": steps,
        "won": adapter.won,
        "done": adapter.done,
        "ante": adapter.game.ante,
        "observations": observations,
        "mask_checks": mask_checks,
    }


# ---------------------------------------------------------------------------
# (a) Full episode test — DirectAdapter
# ---------------------------------------------------------------------------


class TestFullEpisodeDirectAdapter:
    """Run RandomAgent through a complete episode via DirectAdapter."""

    def test_episode_terminates(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        result = _run_episode_collecting(adapter, agent, collect_obs=True)

        # Episode must terminate
        assert result["done"] or result["won"], "Episode did not terminate"
        assert result["steps"] > 0, "No steps taken"

    def test_observations_valid(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
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
        agent = RandomAgent()
        result = _run_episode_collecting(adapter, agent, collect_obs=True)

        assert all(result["mask_checks"]), "Action mask inconsistent with legal actions"


# ---------------------------------------------------------------------------
# (b) Full episode test — BridgeAdapter(SimBackend) identical trajectory
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBridgeAdapterIdenticalTrajectory:
    """BridgeAdapter(SimBackend) must produce identical game trajectory."""

    def test_identical_trajectory(self):
        random.seed(2024)
        direct = DirectAdapter()
        direct.reset(BACK, STAKE, SEED)

        bridge = BridgeAdapter(SimBackend())
        bridge.reset(BACK, STAKE, SEED)

        agent_d = RandomAgent()
        agent_d.reset()

        steps = 0
        for _ in range(MAX_STEPS):
            if direct.done or bridge.done or _finished(direct):
                break

            d_legal = direct.get_legal_actions()
            b_legal = bridge.get_legal_actions()
            if not d_legal or not b_legal:
                break

            # The bridge offers only what BalatroBot executes: draw from its mask. A sort
            # goes out as a rearrange, which orders the hand but sets no sort mode, so the
            # cards drawn after it would land elsewhere: no sorts.
            game = bridge.game
            mask = get_action_mask(game, bridge.legality)
            mask.type_mask[[ActionType.SortHandRank, ActionType.SortHandSuit]] = False
            fa = agent_d.act({}, mask, {"game": game, "legal_actions": b_legal})

            engine_action = factored_to_engine_action(fa, game)

            ds = direct.step(engine_action)
            bs = bridge.step(engine_action)
            steps += 1

            # Core state must match
            assert ds == bs, f"Step {steps}: {ds} != {bs}"

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
            agent = RandomAgent()
            agent.reset()

            steps = 0

            for _ in range(MAX_STEPS):
                if _finished(adapter):
                    break

                legal = adapter.get_legal_actions()
                if not legal:
                    break

                # Verify observation has no NaN
                obs = encode_observation(adapter.game)
                assert np.isfinite(obs.global_context).all(), f"NaN in global_context seed={seed}"
                if obs.hand_cards.shape[0] > 0:
                    assert np.isfinite(obs.hand_cards).all(), f"NaN in hand_cards seed={seed}"

                _agent_step(adapter, agent)
                steps += 1

            antes.append(adapter.game.ante)
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

_PHASE_IDX = {
    GamePhase.BLIND_SELECT: 0,
    GamePhase.SELECTING_HAND: 1,
    GamePhase.ROUND_EVAL: 2,
    GamePhase.SHOP: 3,
    GamePhase.PACK_OPENING: 4,
    GamePhase.GAME_OVER: 5,
}


class TestObservationConsistency:
    """At each step, observation entity counts must match game state."""

    def test_entity_counts_match(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        agent.reset()

        checked = 0

        for _ in range(MAX_STEPS):
            if _finished(adapter):
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            game = adapter.game
            obs = encode_observation(game)

            # Hand cards count
            assert obs.hand_cards.shape[0] == len(game.hand)
            # Jokers count
            assert obs.jokers.shape[0] == len(game.jokers)
            # Consumables count
            assert obs.consumables.shape[0] == len(game.consumables)
            # Shop rows: cards, vouchers, then boosters
            shop = len(game.shop_jokers) + len(game.shop_vouchers) + len(game.shop_boosters)
            assert obs.shop_cards.shape[0] == shop
            # Pack cards count
            assert obs.pack_cards.shape[0] == len(game.pack_cards)

            # Global context phase encoding matches game state phase (one-hot [0:6])
            actual_idx = int(np.argmax(obs.global_context[:6]))
            assert actual_idx == _PHASE_IDX[game.phase], (
                f"Phase encoding mismatch: global_context phase={actual_idx} "
                f"but game phase={game.phase}"
            )

            checked += 1
            _agent_step(adapter, agent)

        assert checked > 0, "No steps checked"


# ---------------------------------------------------------------------------
# (e) Action roundtrip test
# ---------------------------------------------------------------------------


class TestActionRoundtrip:
    """engine_action -> factored -> engine_action must be lossless."""

    def test_roundtrip_across_episode(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        agent.reset()

        roundtrip_count = 0

        for _ in range(MAX_STEPS):
            if _finished(adapter):
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            # Test roundtrip for each legal action
            game = adapter.game
            for engine_action in legal:
                fa = engine_action_to_factored(engine_action)
                try:
                    recovered = factored_to_engine_action(fa, game)
                except ValueError:
                    # The PlayHand / Discard markers carry no selection
                    assert not getattr(engine_action, "card_indices", True)
                    continue

                # Compare the recovered action to the original
                assert type(engine_action) is type(recovered), (
                    f"Type mismatch: {type(engine_action).__name__} != {type(recovered).__name__}"
                )
                roundtrip_count += 1

            _agent_step(adapter, agent)

        assert roundtrip_count > 0, "No actions were roundtrip-tested"

    def test_specific_action_types(self):
        """Test roundtrip for specific action types that must be exact."""
        from jackdaw.engine.actions import (
            Discard,
            PlayHand,
            SelectBlind,
            SkipBlind,
            SortHand,
        )

        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        game = adapter.game

        # SelectBlind roundtrip
        fa = engine_action_to_factored(SelectBlind())
        recovered = factored_to_engine_action(fa, game)
        assert isinstance(recovered, SelectBlind)

        # SkipBlind roundtrip
        fa = engine_action_to_factored(SkipBlind())
        recovered = factored_to_engine_action(fa, game)
        assert isinstance(recovered, SkipBlind)

        # Advance to selecting hand
        adapter.step(SelectBlind())
        hand = game.hand
        assert hand

        # PlayHand roundtrip
        indices = tuple(range(min(3, len(hand))))
        orig = PlayHand(card_indices=indices)
        fa = engine_action_to_factored(orig)
        recovered = factored_to_engine_action(fa, game)
        assert isinstance(recovered, PlayHand)
        assert recovered.card_indices == orig.card_indices

        # Discard roundtrip
        orig = Discard(card_indices=indices)
        fa = engine_action_to_factored(orig)
        recovered = factored_to_engine_action(fa, game)
        assert isinstance(recovered, Discard)
        assert recovered.card_indices == orig.card_indices

        # SortHand roundtrip
        fa = engine_action_to_factored(SortHand(mode="rank"))
        recovered = factored_to_engine_action(fa, game)
        assert isinstance(recovered, SortHand)
        assert recovered.mode == "rank"

        fa = engine_action_to_factored(SortHand(mode="suit"))
        recovered = factored_to_engine_action(fa, game)
        assert isinstance(recovered, SortHand)
        assert recovered.mode == "suit"


# ---------------------------------------------------------------------------
# (h) Full episode with RandomAgent — end-to-end pipeline validation
# ---------------------------------------------------------------------------


def _run_full_episode(adapter, agent, *, max_steps=MAX_STEPS):
    """Run a full episode returning rich diagnostics.

    Returns dict with: steps, won, done, ante, observations,
    masks_valid, action_conversions_ok.
    """
    agent.reset()

    steps = 0
    obs_list: list[Observation] = []
    masks_valid: list[bool] = []
    action_ok: list[bool] = []

    for _ in range(max_steps):
        if _finished(adapter):
            break

        legal = adapter.get_legal_actions()
        if not legal:
            break

        game = adapter.game
        # Observation
        obs_list.append(encode_observation(game))

        # Action mask
        mask = get_action_mask(game)
        masks_valid.append(bool(mask.type_mask.sum() > 0))

        fa = agent.act({}, mask, {"game": game, "legal_actions": legal})

        # Action conversion
        try:
            engine_action = factored_to_engine_action(fa, game)
            action_ok.append(True)
        except ValueError:
            action_ok.append(False)
            break

        adapter.step(engine_action)
        steps += 1

    return {
        "steps": steps,
        "won": adapter.won,
        "done": adapter.done,
        "ante": adapter.game.ante,
        "observations": obs_list,
        "masks_valid": masks_valid,
        "action_ok": action_ok,
    }


class TestFullEpisodeRandomAgent:
    """Run RandomAgent for 20 seeds — validates full pipeline end-to-end."""

    N_SEEDS = 20

    def _seeds(self):
        return [f"RANDOM_E2E_{i}" for i in range(self.N_SEEDS)]

    @pytest.mark.slow
    def test_no_crashes(self):
        """Primary goal: no crashes across 20 episodes."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert result["steps"] > 0, f"seed={seed}: zero steps"

    @pytest.mark.slow
    def test_observations_valid(self):
        """Every observation has correct shapes and no NaN."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)

            for i, obs in enumerate(result["observations"]):
                assert obs.global_context.shape == (D_GLOBAL,), (
                    f"seed={seed} step={i}: bad global_context shape"
                )
                assert np.isfinite(obs.global_context).all(), (
                    f"seed={seed} step={i}: NaN in global_context"
                )
                if obs.hand_cards.shape[0] > 0:
                    assert obs.hand_cards.shape[1] == D_PLAYING_CARD
                    assert np.isfinite(obs.hand_cards).all()
                if obs.jokers.shape[0] > 0:
                    assert obs.jokers.shape[1] == D_JOKER
                    assert np.isfinite(obs.jokers).all()

    @pytest.mark.slow
    def test_masks_have_legal_type(self):
        """Every ActionMask has at least one legal type until game over."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert all(result["masks_valid"]), (
                f"seed={seed}: mask with zero legal types before game end"
            )

    @pytest.mark.slow
    def test_action_conversions_valid(self):
        """Every FactoredAction converts to engine action without ValueError."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert all(result["action_ok"]), f"seed={seed}: action conversion failed"

    @pytest.mark.slow
    def test_episodes_terminate(self):
        """Every episode terminates within max_steps."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert result["done"] or result["won"], f"seed={seed}: episode did not terminate"


# ---------------------------------------------------------------------------
# (j) Action roundtrip fidelity — comprehensive per-step test
# ---------------------------------------------------------------------------


class TestActionRoundtripFidelity:
    """For every step of a full episode, verify all legal actions roundtrip."""

    @pytest.mark.slow
    def test_all_legal_actions_roundtrip(self):
        """For every legal action at every step, convert to factored and back."""
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, "ROUNDTRIP_FULL")
        agent = RandomAgent()
        agent.reset()

        total_tested = 0
        total_skipped = 0
        failures: list[str] = []

        for step_idx in range(MAX_STEPS):
            if _finished(adapter):
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            game = adapter.game
            for engine_action in legal:
                fa = engine_action_to_factored(engine_action)
                try:
                    recovered = factored_to_engine_action(fa, game)
                except ValueError:
                    # The PlayHand / Discard markers carry no selection
                    total_skipped += 1
                    continue

                # Type must match
                if not isinstance(recovered, type(engine_action)):
                    failures.append(
                        f"step={step_idx} type mismatch: "
                        f"{type(engine_action).__name__} != "
                        f"{type(recovered).__name__}"
                    )

                # For actions with card_indices, verify exact match
                orig_ci = getattr(engine_action, "card_indices", None)
                recov_ci = getattr(recovered, "card_indices", None)
                if orig_ci is not None and recov_ci is not None:
                    if tuple(orig_ci) != tuple(recov_ci):
                        failures.append(
                            f"step={step_idx} card_indices mismatch: {orig_ci} != {recov_ci}"
                        )

                total_tested += 1

            _agent_step(adapter, agent)

        assert total_tested > 0, "No actions roundtrip-tested"
        assert not failures, (
            f"{len(failures)} roundtrip failures "
            f"(tested={total_tested}, skipped={total_skipped}):\n" + "\n".join(failures[:20])
        )


# ---------------------------------------------------------------------------
# (k) Observation encoding stress test
# ---------------------------------------------------------------------------


class TestObservationEncodingStress:
    """Run 50 seeds through RandomAgent, encode at every step."""

    N_SEEDS = 50
    VALUE_LIMIT = 1e9  # no astronomically large numbers

    @pytest.mark.slow
    def test_no_nan_correct_shapes_reasonable_values(self):
        seeds = [f"OBS_STRESS_{i}" for i in range(self.N_SEEDS)]
        total_obs_checked = 0

        for seed in seeds:
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            agent.reset()

            for _ in range(MAX_STEPS):
                if _finished(adapter):
                    break

                legal = adapter.get_legal_actions()
                if not legal:
                    break

                game = adapter.game
                obs = encode_observation(game)

                # Shape checks
                assert obs.global_context.shape == (D_GLOBAL,), (
                    f"seed={seed}: global_context shape {obs.global_context.shape}"
                )

                assert obs.hand_cards.shape[0] == len(game.hand), f"seed={seed}: hand rows"
                if obs.hand_cards.shape[0] > 0:
                    assert obs.hand_cards.shape[1] == D_PLAYING_CARD

                assert obs.jokers.shape[0] == len(game.jokers), f"seed={seed}: joker rows"
                if obs.jokers.shape[0] > 0:
                    assert obs.jokers.shape[1] == D_JOKER

                # No NaN/Inf, no astronomically large values (catches log-scale overflow)
                for name in (
                    "global_context",
                    "hand_cards",
                    "jokers",
                    "consumables",
                    "shop_cards",
                    "pack_cards",
                ):
                    arr = getattr(obs, name)
                    assert np.isfinite(arr).all(), f"seed={seed}: NaN/Inf in {name}"
                    if arr.size:
                        assert np.abs(arr).max() < self.VALUE_LIMIT, f"seed={seed}: {name}"

                total_obs_checked += 1
                _agent_step(adapter, agent)

        assert total_obs_checked > 1000, (
            f"Only checked {total_obs_checked} observations "
            f"(expected >1000 across {self.N_SEEDS} seeds)"
        )


# ---------------------------------------------------------------------------
# (m) Multi-deck, multi-stake stress test
# ---------------------------------------------------------------------------


ALL_DECK_KEYS = [
    "b_red",
    "b_blue",
    "b_yellow",
    "b_green",
    "b_black",
    "b_magic",
    "b_nebula",
    "b_ghost",
    "b_abandoned",
    "b_checkered",
    "b_zodiac",
    "b_painted",
    "b_anaglyph",
    "b_plasma",
    "b_erratic",
]


class TestMultiDeckMultiStake:
    """Run RandomAgent on all 15 decks × 3 seeds = 45 episodes. No crashes."""

    N_SEEDS_PER_DECK = 3

    @pytest.mark.slow
    def test_all_decks_no_crashes(self):
        failures: list[str] = []

        for deck in ALL_DECK_KEYS:
            for seed_idx in range(self.N_SEEDS_PER_DECK):
                seed = f"DECK_{deck}_{seed_idx}"
                try:
                    adapter = DirectAdapter()
                    adapter.reset(deck, STAKE, seed)
                    agent = RandomAgent()
                    result = _run_full_episode(adapter, agent, max_steps=MAX_STEPS)
                    assert result["steps"] > 0, "zero steps"
                except Exception as e:
                    failures.append(f"{deck} seed={seed}: {type(e).__name__}: {e}")

        assert not failures, f"{len(failures)}/45 episodes crashed:\n" + "\n".join(failures[:10])
