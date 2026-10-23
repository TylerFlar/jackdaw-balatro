"""Performance benchmarks for the env layer.

Run with: uv run pytest tests/benchmarks/test_env_bench.py -m benchmark
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import pytest

from jackdaw.engine.actions import Discard, PlayHand
from jackdaw.env.action_space import engine_action_to_factored, get_action_mask
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.observation import encode_observation


def _resolve_action(action, gs: dict, rng=random):
    """Fill in card indices for marker PlayHand/Discard actions."""
    hand = gs.get("hand", [])
    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = rng.randint(1, n)
        indices = tuple(sorted(rng.sample(range(len(hand)), count)))
        return PlayHand(card_indices=indices)
    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = rng.randint(1, n)
        indices = tuple(sorted(rng.sample(range(len(hand)), count)))
        return Discard(card_indices=indices)
    return action


def _make_adapter_at_hand_phase() -> tuple[DirectAdapter, dict]:
    """Create an adapter and advance to SELECTING_HAND phase."""
    from jackdaw.engine.actions import GamePhase, SelectBlind

    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "BENCH_SEED_42")

    # Advance to selecting_hand
    for _ in range(10):
        gs = adapter.raw_state
        if gs.get("phase") == GamePhase.SELECTING_HAND:
            return adapter, gs
        legal = adapter.get_legal_actions()
        if not legal:
            break
        # Prefer SelectBlind to get to hand phase quickly
        select = [a for a in legal if isinstance(a, SelectBlind)]
        adapter.step(select[0] if select else legal[0])

    return adapter, adapter.raw_state


def _collect_game_states(n: int) -> list[dict]:
    """Collect diverse game states by playing random actions.

    ``raw_state`` is the live engine dict, mutated in place by every step,
    so each sample is deep-copied; otherwise every sample in an episode
    would alias that episode's final state.
    """
    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "BENCH_COLLECT_0")
    states = []
    seed_counter = 0

    while len(states) < n:
        if adapter.done:
            seed_counter += 1
            adapter.reset("b_red", 1, f"BENCH_COLLECT_{seed_counter}")

        states.append(copy.deepcopy(adapter.raw_state))

        legal = adapter.get_legal_actions()
        if not legal:
            seed_counter += 1
            adapter.reset("b_red", 1, f"BENCH_COLLECT_{seed_counter}")
            continue
        adapter.step(_resolve_action(random.choice(legal), adapter.raw_state))

    return states[:n]


@pytest.mark.benchmark
class TestEnvStepsPerSecond:
    """Assert that DirectAdapter env steps run fast enough for training."""

    def test_env_steps_per_second(self):
        """Full env step loop (engine + encode + mask) > 500 steps/sec."""
        adapter = DirectAdapter()
        adapter.reset("b_red", 1, "BENCH_SPS_0")
        n = 500
        done_count = 0

        t0 = time.perf_counter()
        for i in range(n):
            if adapter.done:
                done_count += 1
                adapter.reset("b_red", 1, f"BENCH_SPS_{done_count}")

            gs = adapter.raw_state
            encode_observation(gs)
            get_action_mask(gs)

            legal = adapter.get_legal_actions()
            if not legal:
                done_count += 1
                adapter.reset("b_red", 1, f"BENCH_SPS_{done_count}")
                continue
            adapter.step(_resolve_action(random.choice(legal), adapter.raw_state))

        elapsed = time.perf_counter() - t0
        sps = n / elapsed
        assert sps > 500, f"Only {sps:.0f} steps/sec (target: >500)"


@pytest.mark.benchmark
class TestEncodingLatency:
    """Assert encode_observation is fast enough."""

    def test_encoding_latency(self):
        """encode_observation < 500us mean across diverse game states."""
        states = _collect_game_states(500)

        times = []
        for gs in states:
            t0 = time.perf_counter()
            encode_observation(gs)
            times.append(time.perf_counter() - t0)

        mean_us = sum(times) / len(times) * 1e6
        assert mean_us < 500, f"encode_observation mean {mean_us:.1f}us (target: <500us)"


@pytest.mark.benchmark
class TestRustEncodingLatency:
    """The Rust encoder (``jackdaw._core``, encoder v1) against ``encode_observation``."""

    def test_rust_encoding_latency(self):
        """encode_game (Rust encode + numpy views) < 50us mean over the tapes' states."""
        from jackdaw.core import Game
        from jackdaw.env.rust_observation import encode_game
        from tests.parity.harness import load_specs
        from tests.parity.pyengine import PyEngine

        games = []
        py_times = []
        for spec in load_specs()[:40]:
            engine = PyEngine(spec.start, spec.setup)
            for action in spec.tape[:80]:
                t0 = time.perf_counter()
                encode_observation(engine.gs)
                py_times.append(time.perf_counter() - t0)
                games.append(Game.from_dict(engine.snapshot()))
                try:
                    engine.step(action)
                except Exception:
                    break

        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            for game in games:
                encode_game(game)
            best = min(best, time.perf_counter() - t0)
        rust_us = best / len(games) * 1e6
        py_us = sum(py_times) / len(py_times) * 1e6
        print(f"\n{len(games)} states: Python {py_us:.1f}us, Rust {rust_us:.2f}us per encode")
        assert rust_us < 50, f"Rust encode mean {rust_us:.1f}us (target: <50us)"


@pytest.mark.benchmark
class TestActionMaskLatency:
    """Assert get_action_mask is fast enough."""

    def test_action_mask_latency(self):
        """get_action_mask < 100us mean across diverse game states."""
        states = _collect_game_states(500)

        times = []
        for gs in states:
            t0 = time.perf_counter()
            get_action_mask(gs)
            times.append(time.perf_counter() - t0)

        mean_us = sum(times) / len(times) * 1e6
        assert mean_us < 100, f"get_action_mask mean {mean_us:.1f}us (target: <100us)"


def _random_gym_steps(env: BalatroGymnasiumEnv, n: int, seed: int) -> float:
    """Run ``n`` uniformly random masked steps, resetting as needed; return steps/sec."""
    policy = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    episode = 0
    t0 = time.perf_counter()
    for _ in range(n):
        action = int(policy.choice(np.flatnonzero(info["action_mask"])))
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            episode += 1
            _, info = env.reset(seed=seed + episode)
    return n / (time.perf_counter() - t0)


@pytest.mark.benchmark
class TestWrapperThroughput:
    """End-to-end throughput of the two env wrappers (resets included)."""

    def test_gymnasium_env_steps_per_second(self):
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
        sps = _random_gym_steps(env, 2000, seed=1)
        print(f"\nBalatroGymnasiumEnv: {sps:.0f} steps/sec")
        assert sps > 300, f"Only {sps:.0f} steps/sec (target: >300)"

    def test_balatro_environment_steps_per_second(self):
        env = BalatroEnvironment(adapter_factory=DirectAdapter, seed_prefix="BENCH_FACTORED")
        rng = random.Random(1)
        _, _, info = env.reset()
        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            gs = info["raw_state"]
            legal = env._adapter.get_legal_actions()
            action = engine_action_to_factored(_resolve_action(rng.choice(legal), gs, rng), gs)
            _, terminated, truncated, _, info = env.step(action)
            if terminated or truncated:
                _, _, info = env.reset()
        sps = n / (time.perf_counter() - t0)
        print(f"\nBalatroEnvironment: {sps:.0f} steps/sec")
        assert sps > 1000, f"Only {sps:.0f} steps/sec (target: >1000)"
