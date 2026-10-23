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
from jackdaw.env.action_space import ActionType, engine_action_to_factored, get_action_mask
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.game_interface import DirectAdapter, OracleAdapter
from jackdaw.env.game_spec import FactoredAction
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


def _make_adapter_at_hand_phase() -> tuple[OracleAdapter, dict]:
    """Create an adapter and advance to SELECTING_HAND phase."""
    from jackdaw.engine.actions import GamePhase, SelectBlind

    adapter = OracleAdapter()
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
    adapter = OracleAdapter()
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
    """Assert that OracleAdapter env steps run fast enough for training."""

    def test_env_steps_per_second(self):
        """Full env step loop (engine + encode + mask) > 500 steps/sec."""
        adapter = OracleAdapter()
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
        env = BalatroGymnasiumEnv(adapter_factory=OracleAdapter)
        sps = _random_gym_steps(env, 2000, seed=1)
        print(f"\nBalatroGymnasiumEnv: {sps:.0f} steps/sec")
        assert sps > 300, f"Only {sps:.0f} steps/sec (target: >300)"

    def test_balatro_environment_steps_per_second(self):
        env = BalatroEnvironment(adapter_factory=OracleAdapter, seed_prefix="BENCH_FACTORED")
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


def _require_release_build() -> None:
    from jackdaw import _core

    if _core._build_info()["profile"] != "release":
        pytest.skip("the Rust env targets are for a release build of jackdaw._core")


def _random_factored(mask, rng: random.Random) -> FactoredAction:
    """A random action the factored mask allows, picked with as little Python as possible."""
    types = np.flatnonzero(mask.type_mask)
    action_type = int(types[rng.randrange(len(types))])
    entity = None
    entities = mask.entity_masks.get(action_type)
    if entities is not None:
        legal = np.flatnonzero(entities)
        entity = int(legal[rng.randrange(len(legal))])
    cards = None
    if action_type in (ActionType.PlayHand, ActionType.Discard):
        hand = len(mask.card_mask)
        chosen = rng.sample(range(hand), rng.randint(1, min(mask.max_card_select, hand)))
        forced = mask.forced_card
        if forced is not None and forced not in chosen:
            chosen[0] = forced
        cards = tuple(sorted(chosen))
    return FactoredAction(action_type=action_type, card_target=cards, entity_target=entity)


@pytest.mark.benchmark
class TestRustEnvThroughput:
    """The env wrappers on the Rust engine against the plan's targets (release build).

    Each measures the time inside the env (steps and resets), not the policy choosing the
    actions, and keeps the best of three runs.
    """

    def test_balatro_environment(self):
        """BalatroEnvironment >= 50K steps/sec."""
        _require_release_build()
        best = 0.0
        for run in range(3):
            env = BalatroEnvironment(DirectAdapter, seed_prefix=f"BENCH_RUST_ENV_{run}")
            rng = random.Random(run)
            _, mask, _ = env.reset()
            n, inside = 20_000, 0.0
            for _ in range(n):
                action = _random_factored(mask, rng)
                t0 = time.perf_counter()
                _, terminated, truncated, mask, _ = env.step(action)
                if terminated or truncated:
                    _, mask, _ = env.reset()
                inside += time.perf_counter() - t0
            best = max(best, n / inside)
        print(f"\nBalatroEnvironment (Rust): {best:,.0f} steps/sec")
        assert best >= 50_000, f"Only {best:,.0f} steps/sec (target: >= 50K)"

    def test_gymnasium_env(self):
        """BalatroGymnasiumEnv >= 25K steps/sec."""
        _require_release_build()
        best = 0.0
        for run in range(3):
            env = BalatroGymnasiumEnv(DirectAdapter, seed_prefix=f"BENCH_RUST_GYM_{run}")
            rng = random.Random(run)
            _, info = env.reset(seed=run)
            n, inside = 20_000, 0.0
            for _ in range(n):
                action = rng.randrange(int(info["action_mask"].sum()))  # the first slots
                t0 = time.perf_counter()
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    _, info = env.reset()
                inside += time.perf_counter() - t0
            best = max(best, n / inside)
        print(f"\nBalatroGymnasiumEnv (Rust): {best:,.0f} steps/sec")
        assert best >= 25_000, f"Only {best:,.0f} steps/sec (target: >= 25K)"

    def test_vector_env(self):
        """BalatroVectorEnv on 8 threads >= 300K steps/sec (256 envs)."""
        _require_release_build()
        from jackdaw.env import BalatroVectorEnv

        n_envs, steps, best = 256, 200, 0.0
        for run in range(3):
            env = BalatroVectorEnv(n_envs, num_threads=8, seed_prefix=f"BENCH_RUST_VEC_{run}")
            policy = np.random.default_rng(run)
            _, info = env.reset(seed=run)
            inside = 0.0
            for _ in range(steps):
                legal = info["action_mask"].sum(axis=1)  # the first slots of each table
                actions = (policy.random(n_envs) * np.maximum(legal, 1)).astype(np.int64)
                t0 = time.perf_counter()
                _, _, _, _, info = env.step(actions)
                inside += time.perf_counter() - t0
            best = max(best, n_envs * steps / inside)
        print(f"\nBalatroVectorEnv (Rust, 8 threads, 256 envs): {best:,.0f} steps/sec")
        assert best >= 300_000, f"Only {best:,.0f} steps/sec (target: >= 300K)"
