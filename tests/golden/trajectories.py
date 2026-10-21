"""Deterministic trajectories for the golden behaviour-equivalence tests.

Each trajectory plays a seeded run and fingerprints everything observable
after every step.  The per-step digests are chained, so a single changed
value anywhere (game state, RNG state, observation, mask, Gym action table,
reward) changes the trajectory digest from that step on.

Two kinds:

* ``engine``: drives the engine directly with the smart agent mixed with
  seeded random legal actions, optionally starting from a run with injected
  jokers so scoring sees developed boards.  Each step also fingerprints
  ``encode_observation``, ``get_action_mask``, the legal action list and the
  Gym wrapper's flat action table for that state.
* ``gym``: drives :class:`BalatroGymnasiumEnv` end to end with a seeded
  uniform policy over the action mask.

Regenerate the fixture with ``uv run python scripts/generate_golden_trajectories.py``
only when a behaviour change is intended.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jackdaw.engine import card as card_module
from jackdaw.engine.actions import Discard, GamePhase, PlayHand, get_legal_actions
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.challenges import get_challenge
from jackdaw.engine.data.prototypes import JOKERS
from jackdaw.engine.game import step as engine_step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.action_space import get_action_mask
from jackdaw.env.balatro_env import _action_mask_to_game
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.observation import encode_observation
from tests.golden.fingerprint import fingerprint

CHECKPOINT_EVERY = 25


@dataclass(frozen=True)
class EngineSpec:
    name: str
    back: str
    stake: int
    seed: str
    challenge: str | None = None
    epsilon: float = 0.15
    jokers: int = 0
    dollars: int | None = None
    max_steps: int = 400


@dataclass(frozen=True)
class GymSpec:
    name: str
    back: str
    stake: int
    seed: int
    episodes: int = 3
    max_steps: int = 120


@dataclass
class Result:
    digest: str = ""
    steps: int = 0
    checkpoints: list[str] = field(default_factory=list)

    def push(self, step_digest: str) -> None:
        self.digest = fingerprint(self.digest, step_digest)
        self.steps += 1
        if self.steps % CHECKPOINT_EVERY == 0:
            self.checkpoints.append(self.digest[:16])

    def as_json(self) -> dict[str, Any]:
        return {"digest": self.digest, "steps": self.steps, "checkpoints": self.checkpoints}


_BACKS = ["b_red", "b_blue", "b_yellow", "b_green", "b_black", "b_magic", "b_nebula", "b_ghost",
          "b_abandoned", "b_checkered", "b_zodiac", "b_painted", "b_anaglyph", "b_plasma",
          "b_erratic"]  # fmt: skip
_CHALLENGES = ["c_omelette_1", "c_city_1", "c_rich_1", "c_knife_1", "c_xray_1", "c_mad_world_1",
               "c_luxury_1", "c_non_perishable_1", "c_medusa_1", "c_double_nothing_1",
               "c_typecast_1", "c_inflation_1", "c_bram_poker_1", "c_fragile_1", "c_monolith_1",
               "c_blast_off_1", "c_five_card_1", "c_golden_needle_1", "c_cruelty_1",
               "c_jokerless_1"]  # fmt: skip


def engine_specs() -> list[EngineSpec]:
    specs: list[EngineSpec] = []
    for i, back in enumerate(_BACKS):
        specs.append(EngineSpec(f"deck_{back}", back, 1 + i % 8, f"GOLD_D{i}"))
    for stake in range(1, 9):
        specs.append(EngineSpec(f"stake_{stake}", "b_red", stake, f"GOLD_S{stake}", epsilon=0.3))
    for i, ch in enumerate(_CHALLENGES):
        specs.append(EngineSpec(f"challenge_{ch}", "b_challenge", 1, f"GOLD_C{i}", challenge=ch))
    for i in range(32):
        specs.append(
            EngineSpec(
                f"jokers_{i}",
                _BACKS[i % len(_BACKS)],
                1 + i % 4,
                f"GOLD_J{i}",
                epsilon=0.25,
                jokers=3 + i % 4,
                dollars=60,
            )
        )
    return specs


def gym_specs() -> list[GymSpec]:
    return [GymSpec(f"gym_{back}", back, 1 + i % 8, 1000 + i) for i, back in enumerate(_BACKS)]


@contextmanager
def _fresh_sort_ids() -> Iterator[None]:
    """Start each trajectory from sort_id 0 so digests don't depend on test order."""
    saved = card_module._sort_id_counter
    card_module.reset_sort_id_counter()
    try:
        yield
    finally:
        card_module._sort_id_counter = saved


def _resolve(action: Any, gs: dict[str, Any], rng: random.Random) -> Any:
    hand = gs.get("hand", [])
    if isinstance(action, (PlayHand, Discard)) and not action.card_indices and hand:
        count = rng.randint(1, min(5, len(hand)))
        idx = tuple(sorted(rng.sample(range(len(hand)), count)))
        return type(action)(card_indices=idx)
    return action


def run_engine(spec: EngineSpec) -> Result:
    from jackdaw.cli.smart_agent import smart_agent

    res = Result()
    rng = random.Random(spec.seed)
    with _fresh_sort_ids():
        challenge = get_challenge(spec.challenge) if spec.challenge else None
        gs = initialize_run(spec.back, spec.stake, spec.seed, challenge=challenge)
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"
        if spec.jokers:
            keys = sorted(JOKERS)
            for key in rng.sample(keys, spec.jokers):
                gs["jokers"].append(create_joker(key))
        if spec.dollars is not None:
            gs["dollars"] = spec.dollars

        # A wrapper instance only for its (pure) action-table builder.
        gym = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
        gym._rng = np.random.default_rng(zlib.crc32(spec.seed.encode()))

        for _ in range(spec.max_steps):
            obs = encode_observation(gs)
            mask = get_action_mask(gs)
            table = gym._enumerate_actions(_action_mask_to_game(mask), {"raw_state": gs})
            legal = get_legal_actions(gs)
            res.push(fingerprint(gs, obs, mask, table, legal))
            if gs.get("phase") == GamePhase.GAME_OVER or gs.get("won") or not legal:
                break
            if rng.random() < spec.epsilon:
                action = _resolve(rng.choice(legal), gs, rng)
            else:
                action = smart_agent(gs, legal)
            try:
                engine_step(gs, action)
            except Exception as exc:  # behaviour includes raising
                res.push(fingerprint("raised", type(exc).__name__, str(exc), action))
                break
            res.push(fingerprint(action))
    return res


def run_gym(spec: GymSpec) -> Result:
    res = Result()
    with _fresh_sort_ids():
        env = BalatroGymnasiumEnv(
            adapter_factory=DirectAdapter,
            back_keys=[spec.back],
            stakes=[spec.stake],
            seed_prefix=f"GOLD_G_{spec.name}",
        )
        policy = np.random.default_rng(spec.seed)
        for ep in range(spec.episodes):
            obs, info = env.reset(seed=spec.seed * 100 + ep)
            res.push(fingerprint(obs, info, env._action_table, env._inner._adapter.raw_state))
            for _ in range(spec.max_steps):
                legal = np.flatnonzero(info["action_mask"])
                action = int(policy.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)
                res.push(
                    fingerprint(
                        action,
                        obs,
                        reward,
                        terminated,
                        truncated,
                        info,
                        env._action_table,
                        env._inner._adapter.raw_state,
                    )
                )
                if terminated or truncated:
                    break
    return res


def run_all() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for spec in engine_specs():
        out[spec.name] = run_engine(spec).as_json()
    for spec in gym_specs():
        out[spec.name] = run_gym(spec).as_json()
    return out
