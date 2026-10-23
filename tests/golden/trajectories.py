"""Deterministic trajectories for the golden behaviour-equivalence tests.

Each trajectory plays a seeded run and fingerprints everything observable
after every step.  The per-step digests are chained, so a single changed
value anywhere (game state, RNG state, observation, mask, Gym action table,
env info, reward) changes the trajectory digest from that step on.

Runs are driven by a recorded *action tape*: generation chooses actions
with a policy (the smart agent mixed with seeded random legal actions) and
stores them; the tests replay the tape.  A behaviour change therefore shows
up as a digest mismatch (or a now-illegal / missing taped action), and the
gate does not depend on the CLI agent's policy.

Kinds:

* ``engine``: the engine driven directly, from every deck, stake and
  challenge, from random injected joker boards, and from curated scenarios
  (forced boss blinds, copy/retrigger boards, Mr. Bones saves, Sixth Sense,
  mid-round creation with enhancement-gated pools).  Each step also
  fingerprints ``encode_observation``, ``get_action_mask``, the legal
  action list and the Gym wrapper's flat action table for that state.
* ``factored``: :class:`BalatroEnvironment` end to end over DirectAdapter
  and BridgeAdapter(SimBackend), fingerprinting the complete reset/step
  return values, the reward ``Transition`` and the built-in rewards.
* ``gym``: :class:`BalatroGymnasiumEnv` end to end (uniform and smart
  policies, sparse and shaped rewards, forced truncation), including the
  inner env's info and the wrapper RNG state.
* ``probe``: direct calls into RNG, hand evaluation, pools and action
  enumeration over edge-case inputs.

Regenerate the fixtures with ``uv run python scripts/generate_golden_trajectories.py``
only when a behaviour change is intended.
"""

from __future__ import annotations

import dataclasses
import math
import random
import struct
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jackdaw.bridge.backend import SimBackend
from jackdaw.engine import actions as A
from jackdaw.engine import card as card_module
from jackdaw.engine.actions import GamePhase, get_legal_actions
from jackdaw.engine.challenges import get_challenge
from jackdaw.engine.data.prototypes import JOKERS
from jackdaw.engine.game import step as engine_step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.action_space import (
    ActionType,
    engine_action_to_factored,
    get_action_mask,
)
from jackdaw.env.balatro_env import BalatroEnvironment, _action_mask_to_game
from jackdaw.env.game_interface import BridgeAdapter, DirectAdapter
from jackdaw.env.game_spec import GameActionMask
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.observation import encode_observation
from jackdaw.env.rewards import ShapedReward, Transition, blind_progress_reward
from tests.golden.fingerprint import fingerprint

CHECKPOINT_EVERY = 25

# ---------------------------------------------------------------------------
# Results and action tapes
# ---------------------------------------------------------------------------


@dataclass
class Result:
    digest: str = ""
    steps: int = 0
    checkpoints: list[str] = field(default_factory=list)
    tape: list[Any] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def push(self, step_digest: str) -> None:
        self.digest = fingerprint(self.digest, step_digest)
        self.steps += 1
        if self.steps % CHECKPOINT_EVERY == 0:
            self.checkpoints.append(self.digest[:16])

    def as_json(self) -> dict[str, Any]:
        return {"digest": self.digest, "steps": self.steps, "checkpoints": self.checkpoints}


_ACTION_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        A.SelectBlind, A.SkipBlind, A.PlayHand, A.Discard, A.BuyCard, A.SellCard,
        A.UseConsumable, A.RedeemVoucher, A.OpenBooster, A.Reroll, A.NextRound,
        A.PickPackCard, A.SkipPack, A.CashOut, A.SortHand, A.SwapHandLeft,
        A.SwapHandRight, A.SwapJokersLeft, A.SwapJokersRight, A.RerollBoss, A.BuyAndUse,
    )
}  # fmt: skip


def encode_action(action: Any) -> list[Any]:
    out: list[Any] = [type(action).__name__]
    for f in dataclasses.fields(action):
        v = getattr(action, f.name)
        out.append(list(v) if isinstance(v, tuple) else v)
    return out


def decode_action(entry: list[Any]) -> Any:
    cls = _ACTION_TYPES[entry[0]]
    values = [tuple(v) if isinstance(v, list) else v for v in entry[1:]]
    return cls(*values)


class TapeExhausted(Exception):
    """Replay needed an action the recorded run never took."""


class _Chooser:
    """Picks actions: from a policy while recording, or from a tape."""

    def __init__(self, tape: list[Any] | None, res: Result) -> None:
        self._tape = tape
        self._pos = 0
        self._res = res

    def __call__(self, policy: Callable[[], Any], encode: Callable[[Any], Any]) -> Any:
        if self._tape is None:
            choice = encode(policy())
            self._res.tape.append(choice)
            return choice
        if self._pos >= len(self._tape):
            raise TapeExhausted
        entry = self._tape[self._pos]
        self._pos += 1
        self._res.tape.append(entry)
        return entry


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
    if isinstance(action, (A.PlayHand, A.Discard)) and not action.card_indices and hand:
        count = rng.randint(1, min(5, len(hand)))
        idx = tuple(sorted(rng.sample(range(len(hand)), count)))
        return type(action)(card_indices=idx)
    return action


def _with_forced_card(action: Any, gs: dict[str, Any]) -> Any:
    """Include Cerulean Bell's forced card, which vanilla always selects."""
    if not isinstance(action, (A.PlayHand, A.Discard)):
        return action
    forced = [i for i, c in enumerate(gs.get("hand", [])) if c.ability.get("forced_selection")]
    if not forced or forced[0] in action.card_indices:
        return action
    kept = [i for i in action.card_indices][:4]
    return type(action)(card_indices=tuple(sorted({*kept, forced[0]})))


def _mixed_policy(gs: dict[str, Any], legal: list[Any], rng: random.Random, eps: float) -> Any:
    from jackdaw.cli.smart_agent import smart_agent

    if rng.random() < eps:
        return _with_forced_card(_resolve(rng.choice(legal), gs, rng), gs)
    return _with_forced_card(smart_agent(gs, legal), gs)


# ---------------------------------------------------------------------------
# Curated scenario setups (engine kind)
# ---------------------------------------------------------------------------


def _add(gs: dict[str, Any], key: str, **params: Any) -> None:
    """Add a card through the balatrobot ``add`` path (edition poll, add_to_deck)."""
    backend = SimBackend()
    backend._gs = gs
    backend.handle("add", {"key": key, **params})


def _hand_index(gs: dict[str, Any], card_key: str, nth: int = 0) -> int:
    hits = [i for i, c in enumerate(gs["hand"]) if c.card_key == card_key]
    return hits[nth]


def _last_hand(gs: dict[str, Any], *, chips_fraction: float) -> None:
    gs["current_round"]["hands_left"] = 1
    gs["hands_left"] = 1
    gs["chips"] = int(gs["blind"].chips * chips_fraction)


@dataclass(frozen=True)
class Scenario:
    """Board set up after the first SelectBlind, scripted opening actions, a check."""

    setup: Callable[[dict[str, Any]], None]
    script: tuple[Callable[[dict[str, Any]], Any], ...] = ()
    check: Callable[[dict[str, Any]], bool] = lambda gs: True
    boss: str | None = None


def _play(*keys: str) -> Callable[[dict[str, Any]], Any]:
    return lambda gs: A.PlayHand(card_indices=tuple(sorted(_hand_index(gs, k) for k in keys)))


def _score_positive(gs: dict[str, Any]) -> bool:
    res = gs.get("last_score_result")
    return res is not None and res.total > 0


def _setup_copy_retriggers(gs: dict[str, Any]) -> None:
    for key in ("j_blueprint", "j_hack", "j_brainstorm", "j_mime", "j_baseball"):
        _add(gs, key)
    _add(gs, "H_2", seal="Red")
    _add(gs, "S_3")
    _add(gs, "D_4")
    _add(gs, "C_5")
    _add(gs, "C_K", enhancement="m_steel")
    _add(gs, "H_K", enhancement="m_steel", seal="Red")


def _setup_face_retriggers(gs: dict[str, Any]) -> None:
    for key in ("j_sock_and_buskin", "j_hanging_chad", "j_blueprint", "j_selzer", "j_dusk"):
        _add(gs, key)
    _add(gs, "H_K")
    _add(gs, "H_Q", seal="Red")
    _add(gs, "H_J")
    gs["current_round"]["hands_left"] = 1
    gs["hands_left"] = 1


def _setup_bones(gs: dict[str, Any]) -> None:
    _add(gs, "j_mr_bones")
    _add(gs, "j_joker")
    _add(gs, "S_2")
    _last_hand(gs, chips_fraction=0.3)


def _bones_saved(gs: dict[str, Any]) -> bool:
    return gs["phase"] != GamePhase.GAME_OVER and not any(
        j.center_key == "j_mr_bones" for j in gs["jokers"]
    )


def _setup_psychic_bones(gs: dict[str, Any]) -> None:
    for key in ("j_matador", "j_mr_bones", "j_ice_cream"):
        _add(gs, key)
    _add(gs, "S_2")
    _last_hand(gs, chips_fraction=0.3)


def _debuffed_and_saved(gs: dict[str, Any]) -> bool:
    res = gs.get("last_score_result")
    return res is not None and res.debuffed and _bones_saved(gs)


def _setup_sixth_sense(gs: dict[str, Any]) -> None:
    for key in ("j_sixth_sense", "j_caino", "j_glass", "j_hiker"):
        _add(gs, key)
    _add(gs, "H_6")


def _six_destroyed(gs: dict[str, Any]) -> bool:
    return any(c.ability.get("set") == "Spectral" for c in gs["consumables"])


def _setup_space_dna(gs: dict[str, Any]) -> None:
    for key in ("j_oops", "j_space", "j_dna", "j_obelisk", "j_reserved_parking"):
        _add(gs, key)
    _add(gs, "S_9")
    _add(gs, "H_K")
    _add(gs, "S_Q")


def _setup_creation_gates(gs: dict[str, Any]) -> None:
    _add(gs, "j_riff_raff")
    _add(gs, "c_judgement")
    _add(gs, "c_wraith")
    for card, enh in (
        ("H_7", "m_glass"),
        ("S_7", "m_steel"),
        ("D_7", "m_lucky"),
        ("C_7", "m_gold"),
    ):
        _add(gs, card, enhancement=enh)
    gs["dollars"] = 80


def _discard_enhanced(gs: dict[str, Any]) -> Any:
    idx = [i for i, c in enumerate(gs["hand"]) if c.center_key in ("m_glass", "m_lucky")]
    return A.Discard(card_indices=tuple(idx))


SCENARIOS: dict[str, Scenario] = {
    "copy_retriggers": Scenario(
        _setup_copy_retriggers, (_play("H_2", "S_3", "D_4", "C_5"),), _score_positive
    ),
    "face_retriggers": Scenario(_setup_face_retriggers, (_play("H_K", "H_Q", "H_J"),)),
    "bones_save": Scenario(_setup_bones, (_play("S_2"),), _bones_saved),
    "psychic_bones": Scenario(
        _setup_psychic_bones, (_play("S_2"),), _debuffed_and_saved, boss="bl_psychic"
    ),
    "sixth_sense": Scenario(_setup_sixth_sense, (_play("H_6"),), _six_destroyed),
    "space_dna": Scenario(_setup_space_dna, (_play("S_9"),), _score_positive),
    "creation_gates": Scenario(
        _setup_creation_gates,
        (
            lambda gs: A.UseConsumable(card_index=0),
            lambda gs: A.UseConsumable(card_index=0),
            _discard_enhanced,
        ),
        lambda gs: len(gs["jokers"]) >= 2,
    ),
}


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


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
    boss: str | None = None
    scenario: str | None = None


@dataclass(frozen=True)
class FactoredSpec:
    name: str
    back: str
    stake: int
    adapter: str  # "direct" | "bridge"
    episodes: int = 2
    max_steps: int = 10_000
    epsilon: float = 0.2


@dataclass(frozen=True)
class GymSpec:
    name: str
    back: str
    stake: int
    seed: int
    episodes: int = 3
    max_steps: int = 120
    policy: str = "uniform"  # "uniform" | "smart"
    reward_shaping: bool = False
    env_max_steps: int = 10_000


_BACKS = ["b_red", "b_blue", "b_yellow", "b_green", "b_black", "b_magic", "b_nebula", "b_ghost",
          "b_abandoned", "b_checkered", "b_zodiac", "b_painted", "b_anaglyph", "b_plasma",
          "b_erratic"]  # fmt: skip
_CHALLENGES = ["c_omelette_1", "c_city_1", "c_rich_1", "c_knife_1", "c_xray_1", "c_mad_world_1",
               "c_luxury_1", "c_non_perishable_1", "c_medusa_1", "c_double_nothing_1",
               "c_typecast_1", "c_inflation_1", "c_bram_poker_1", "c_fragile_1", "c_monolith_1",
               "c_blast_off_1", "c_five_card_1", "c_golden_needle_1", "c_cruelty_1",
               "c_jokerless_1"]  # fmt: skip
_BOSSES = ["bl_final_heart", "bl_psychic", "bl_eye", "bl_mouth", "bl_plant", "bl_hook",
           "bl_flint", "bl_arm", "bl_club", "bl_window", "bl_head", "bl_goad", "bl_wall",
           "bl_water", "bl_needle", "bl_manacle", "bl_serpent", "bl_pillar", "bl_tooth",
           "bl_house", "bl_wheel", "bl_mark", "bl_fish", "bl_ox", "bl_final_bell",
           "bl_final_leaf", "bl_final_vessel", "bl_final_acorn"]  # fmt: skip


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
    for i, boss in enumerate(_BOSSES):
        specs.append(
            EngineSpec(f"boss_{boss}", "b_red", 1, f"GOLD_B{i}", jokers=4, dollars=30, boss=boss)
        )
    for i, name in enumerate(SCENARIOS):
        specs.append(
            EngineSpec(f"scenario_{name}", "b_red", 1, f"GOLD_X{i}", scenario=name, max_steps=250)
        )
    specs.append(
        EngineSpec(
            "scenario_crimson_bones", "b_red", 1, "GOLD_XC", scenario="bones_save",
            boss="bl_final_heart",
        )
    )  # fmt: skip
    return specs


def factored_specs() -> list[FactoredSpec]:
    return [
        FactoredSpec("factored_direct_red", "b_red", 1, "direct"),
        FactoredSpec("factored_direct_plasma", "b_plasma", 3, "direct"),
        FactoredSpec("factored_direct_truncated", "b_yellow", 2, "direct", max_steps=40),
        FactoredSpec("factored_bridge_red", "b_red", 1, "bridge", episodes=1),
        FactoredSpec("factored_bridge_ghost", "b_ghost", 2, "bridge", episodes=1, max_steps=120),
    ]


def gym_specs() -> list[GymSpec]:
    specs = [GymSpec(f"gym_{back}", back, 1 + i % 8, 1000 + i) for i, back in enumerate(_BACKS)]
    specs += [
        GymSpec("gym_smart_red", "b_red", 1, 2000, episodes=2, max_steps=400, policy="smart"),
        GymSpec(
            "gym_smart_shaped", "b_magic", 2, 2001, episodes=2, max_steps=400, policy="smart",
            reward_shaping=True,
        ),
        GymSpec(
            "gym_truncated", "b_blue", 1, 2002, episodes=2, max_steps=100, policy="smart",
            reward_shaping=True, env_max_steps=30,
        ),
    ]  # fmt: skip
    return specs


PROBES = ("rng", "hand_eval", "pools", "enumerate")


# ---------------------------------------------------------------------------
# Engine trajectories
# ---------------------------------------------------------------------------


def run_engine(spec: EngineSpec, tape: list[Any] | None = None) -> Result:
    res = Result()
    choose = _Chooser(tape, res)
    policy_rng = random.Random(spec.seed)
    setup_rng = random.Random(spec.seed + "/setup")
    scenario = SCENARIOS[spec.scenario] if spec.scenario else None
    with _fresh_sort_ids():
        challenge = get_challenge(spec.challenge) if spec.challenge else None
        gs = initialize_run(spec.back, spec.stake, spec.seed, challenge=challenge)
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"
        boss = spec.boss or (scenario.boss if scenario else None)
        if boss:
            gs["round_resets"]["blind_choices"]["Boss"] = boss
            gs["blind_on_deck"] = "Boss"
        if spec.jokers or scenario:
            engine_step(gs, A.SelectBlind())
            res.push(fingerprint("setup-select", gs))
        if spec.jokers:
            for key in setup_rng.sample(sorted(JOKERS), spec.jokers):
                _add(gs, key)
        if scenario:
            scenario.setup(gs)
        if spec.dollars is not None:
            gs["dollars"] = spec.dollars
        script = list(scenario.script) if scenario else []
        n_scripted = len(script)
        checked = scenario is None
        taken = 0

        # A wrapper instance only for its action-table builder.
        gym = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
        gym._rng = np.random.default_rng(zlib.crc32(spec.seed.encode()))

        for _ in range(spec.max_steps):
            if not checked and taken >= n_scripted:
                res.checks["scenario"] = bool(scenario.check(gs))  # type: ignore[union-attr]
                checked = True
            obs = encode_observation(gs)
            mask = get_action_mask(gs)
            table = gym._enumerate_actions(_action_mask_to_game(mask), {"raw_state": gs})
            legal = get_legal_actions(gs)
            res.push(fingerprint(gs, obs, mask, table, legal, gym._rng.bit_generator.state))
            if gs.get("phase") == GamePhase.GAME_OVER or gs.get("won") or not legal:
                break

            def policy(legal: list[Any] = legal) -> Any:
                if script:
                    return script.pop(0)(gs)
                return _mixed_policy(gs, legal, policy_rng, spec.epsilon)

            try:
                action = decode_action(choose(policy, encode_action))
            except TapeExhausted:
                res.push(fingerprint("tape-exhausted"))
                break
            try:
                engine_step(gs, action)
            except Exception as exc:  # behaviour includes raising
                res.push(fingerprint("raised", type(exc).__name__, str(exc), action))
                break
            taken += 1
            res.push(fingerprint(action))
        if not checked:
            res.checks["scenario"] = bool(scenario.check(gs))  # type: ignore[union-attr]
        res.push(fingerprint("final", gs))
    return res


# ---------------------------------------------------------------------------
# Factored env trajectories
# ---------------------------------------------------------------------------


def _adapter_factory(kind: str) -> Callable[[], Any]:
    if kind == "direct":
        return DirectAdapter
    return lambda: BridgeAdapter(SimBackend())


def run_factored(spec: FactoredSpec, tape: list[Any] | None = None) -> Result:
    res = Result()
    choose = _Chooser(tape, res)
    policy_rng = random.Random(spec.name)
    with _fresh_sort_ids():
        env = BalatroEnvironment(
            adapter_factory=_adapter_factory(spec.adapter),
            back_keys=[spec.back],
            stakes=[spec.stake],
            max_steps=spec.max_steps,
            seed_prefix=f"GOLD_F_{spec.name}",
        )
        shaped = ShapedReward()
        try:
            for _ep in range(spec.episodes):
                game_obs, game_mask, info = env.reset()
                shaped = ShapedReward()
                res.push(fingerprint("reset", game_obs, game_mask, info))
                while True:
                    gs = info["raw_state"]
                    legal = env._adapter.get_legal_actions()
                    res.push(fingerprint("legal", legal))
                    if not legal:
                        break
                    action = decode_action(
                        choose(
                            lambda gs=gs, legal=legal: _mixed_policy(
                                gs, legal, policy_rng, spec.epsilon
                            ),
                            encode_action,
                        )
                    )
                    factored = engine_action_to_factored(action, gs)
                    out = env.step(factored)
                    game_obs, terminated, truncated, game_mask, info = out
                    t = Transition.from_step(factored, terminated, truncated, info)
                    res.push(fingerprint(factored, out, t, shaped(t), blind_progress_reward(t)))
                    if terminated or truncated:
                        break
        except TapeExhausted:
            res.push(fingerprint("tape-exhausted"))
        except Exception as exc:  # behaviour includes raising
            res.push(fingerprint("raised", type(exc).__name__, str(exc)))
    return res


# ---------------------------------------------------------------------------
# Gym trajectories
# ---------------------------------------------------------------------------


def _gym_smart_index(env: BalatroGymnasiumEnv, policy: np.random.Generator) -> int:
    from jackdaw.cli.smart_agent import smart_agent

    gs = env._inner._adapter.raw_state
    legal = env._inner._adapter.get_legal_actions()
    table = env._action_table
    if legal:
        try:
            wanted = engine_action_to_factored(smart_agent(gs, legal), gs)
        except (ValueError, KeyError, IndexError):
            wanted = None
        if wanted is not None:
            if wanted in table:
                return table.index(wanted)
            # The exact card selection was subsampled out of the table:
            # take another action of the same type so the run keeps going.
            same = [i for i, a in enumerate(table) if a.action_type == wanted.action_type]
            if same:
                return int(policy.choice(same))
    return int(policy.choice(len(table)))


def run_gym(spec: GymSpec, tape: list[Any] | None = None) -> Result:
    res = Result()
    choose = _Chooser(tape, res)
    with _fresh_sort_ids():
        env = BalatroGymnasiumEnv(
            adapter_factory=DirectAdapter,
            back_keys=[spec.back],
            stakes=[spec.stake],
            seed_prefix=f"GOLD_G_{spec.name}",
            reward_shaping=spec.reward_shaping,
            max_steps=spec.env_max_steps,
        )
        inner_infos: list[Any] = []
        inner_step = env._inner.step
        inner_reset = env._inner.reset

        def capture_step(action: Any) -> Any:
            out = inner_step(action)
            inner_infos.append(out)
            return out

        def capture_reset(**kwargs: Any) -> Any:
            out = inner_reset(**kwargs)
            inner_infos.append(out)
            return out

        env._inner.step = capture_step  # type: ignore[method-assign]
        env._inner.reset = capture_reset  # type: ignore[method-assign]
        shaped = ShapedReward()
        policy = np.random.default_rng(spec.seed)
        try:
            for ep in range(spec.episodes):
                obs, info = env.reset(seed=spec.seed * 100 + ep)
                shaped = ShapedReward()
                res.push(
                    fingerprint(
                        obs, info, env._action_table, inner_infos.pop(),
                        env._rng.bit_generator.state,
                    )
                )  # fmt: skip
                for _ in range(spec.max_steps):
                    mask = info["action_mask"]

                    def pick(mask: np.ndarray = mask) -> int:
                        if spec.policy == "smart":
                            return _gym_smart_index(env, policy)
                        return int(policy.choice(np.flatnonzero(mask)))

                    action = choose(pick, int)
                    factored = env._action_table[action]
                    obs, reward, terminated, truncated, info = env.step(action)
                    inner = inner_infos.pop()
                    t = Transition.from_step(factored, terminated, truncated, inner[4])
                    res.push(
                        fingerprint(
                            action, obs, reward, terminated, truncated, info,
                            env._action_table, inner, shaped(t), blind_progress_reward(t),
                            env._rng.bit_generator.state,
                        )
                    )  # fmt: skip
                    if terminated or truncated:
                        break
        except TapeExhausted:
            res.push(fingerprint("tape-exhausted"))
        except Exception as exc:  # behaviour includes raising
            res.push(fingerprint("raised", type(exc).__name__, str(exc)))
    return res


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


class _SortObj:
    def __init__(self, sort_id: int, tag: str) -> None:
        self.sort_id = sort_id
        self.tag = tag


def _rng_probe(res: Result) -> None:
    from jackdaw.engine import rng as R

    pi, e = R._PI, R._E
    # Seeds whose k-th seeding iterate lands near 0.  Only word 0's u < MIN
    # adjustment is reachable from a double (via exactly -e/pi); the others
    # need an iterate within ~1e-318 of 0.  Neighbours are included too.
    edge: list[float] = []
    target = 0.0
    for _ in range(4):
        target = (target - e) / pi
        edge.append(target)
    seeds = [0.0, -0.0, 0.5, 1.0, -1.0, 5e-324, 1e-300, 1e300, math.inf, -math.inf, math.nan,
             1 - 2**-53, 2**-53]  # fmt: skip
    for s in edge:
        seeds += [math.nextafter(s, math.inf), math.nextafter(s, -math.inf), s]
    gen = random.Random(7)
    seeds += [gen.random() for _ in range(3000)]
    seeds += [struct.unpack("<d", struct.pack("<Q", gen.getrandbits(64)))[0] for _ in range(1000)]
    for s in seeds:
        st = R._luajit_seed(s)
        res.push(fingerprint(s, st, R._luajit_random(list(st)), R._luajit_random_int(st, -3, 17)))

    tables: list[Any] = [
        ["a", "b", "c", "d", "e"],
        ("x", "y", "z"),
        [1],
        [],
        {},
        [{"sort_id": 5, "k": "p"}, {"sort_id": 2, "k": "q"}, {"sort_id": 9, "k": "r"}],
        [{"sort_id": 3, "k": "t1"}, {"sort_id": 3, "k": "t2"}, {"sort_id": 1, "k": "t3"}],
        [_SortObj(4, "o1"), _SortObj(2, "o2"), _SortObj(2, "o3"), _SortObj(8, "o4")],
        {"b": 1, "a": 2, "c": 3},
        {3: "x", 1: "y", 2: "z"},
        {"k": "s", 2: "i", 1: "j"},
        {"m": {"sort_id": 2}, "n": {"sort_id": 1}},
    ]
    for t in tables:
        for s in seeds[:200]:
            try:
                v, k = R.pseudorandom_element(t, s)
                out: Any = (getattr(v, "tag", v), k)
            except Exception as exc:  # noqa: BLE001
                out = ("raised", type(exc).__name__, str(exc))
            res.push(fingerprint(s, out))
            prng = R.PseudoRandom("ELEM")
            try:
                v, k = prng.element(t, s)
                out = (getattr(v, "tag", v), k)
            except Exception as exc:  # noqa: BLE001
                out = ("raised", type(exc).__name__, str(exc))
            res.push(fingerprint(s, out))
    for s in seeds[:300]:
        for lst in ([], list(range(13)), [_SortObj(i % 5, str(i)) for i in range(9)],
                    [{"sort_id": 9 - i, "v": i} for i in range(9)]):  # fmt: skip
            work = list(lst)
            R.pseudoshuffle(work, s)
            res.push(fingerprint(s, [getattr(x, "tag", x) for x in work]))
    prng = R.PseudoRandom("PROBESEED")
    for i in range(400):
        key = f"k{i % 23}"
        res.push(
            fingerprint(
                prng.seed(key), prng.random(key + "r"), prng.random(key, 1, 52),
                prng.predict_seed(key, "PRED"), dict(prng.state),
            )
        )  # fmt: skip
    res.push(fingerprint(R.pseudohash(""), [R.pseudohash(f"s{i}") for i in range(200)]))


def _hand_eval_probe(res: Result) -> None:
    from jackdaw.engine.card_factory import create_playing_card
    from jackdaw.engine.data.enums import Rank, Suit
    from jackdaw.engine.hand_eval import evaluate_hand, evaluate_poker_hand, get_best_hand

    gen = random.Random(11)
    suits, ranks = list(Suit), list(Rank)
    enh = ["c_base", "c_base", "c_base", "m_stone", "m_wild", "m_bonus"]
    for _ in range(2500):
        n = gen.randint(0, 9)
        hand = [
            create_playing_card(gen.choice(suits), gen.choice(ranks), enhancement=gen.choice(enh))
            for _ in range(n)
        ]
        for c in hand:
            if gen.random() < 0.1:
                c.debuff = True
        pos = {id(c): i for i, c in enumerate(hand)}

        def enc(groups: Any, pos: dict[int, int] = pos) -> Any:
            return [[pos[id(c)] for c in g] for g in groups]

        flags = {
            "four_fingers": gen.random() < 0.3,
            "shortcut": gen.random() < 0.3,
            "smeared": gen.random() < 0.3,
        }
        results = evaluate_poker_hand(hand, **flags)
        name, scoring, full = get_best_hand(hand, **flags)
        res.push(
            fingerprint(
                flags,
                {k: enc(v) for k, v in results.items()},
                name,
                [pos[id(c)] for c in scoring],
                {k: enc(v) for k, v in full.items()},
            )
        )
        if hand:
            res.push(fingerprint(evaluate_hand(hand, [])))


def _pools_probe(res: Result) -> None:
    from jackdaw.engine.card_factory import (
        create_card,
        create_joker,
        create_playing_card,
        deck_enhancements_from_state,
        showman_active,
    )
    from jackdaw.engine.data.enums import Rank, Suit
    from jackdaw.engine.data.prototypes import CENTER_POOLS
    from jackdaw.engine.pools import get_current_pool
    from jackdaw.engine.rng import PseudoRandom

    enh_sets: list[set[str]] = [set(), {"m_glass"}, {"m_steel", "m_lucky"}, {"m_gold", "m_stone"}]
    flag_sets: list[dict[str, bool]] = [
        {},
        {"gros_michel_extinct": True},
        {"gros_michel_extinct": False},
    ]
    for pool_type in ("Joker", "Tarot", "Planet", "Spectral", "Voucher", "Enhanced", "Tag"):
        for i, enh in enumerate(enh_sets):
            for flags in flag_sets:
                for showman in (False, True):
                    rng = PseudoRandom(f"POOL{i}")
                    out = get_current_pool(
                        pool_type, rng, 1 + i,
                        used_jokers={"j_joker", "c_fool", "v_overstock_norm"},
                        used_vouchers={"v_overstock_norm"} if i % 2 else set(),
                        pool_flags=flags, has_showman=showman, deck_enhancements=enh,
                        played_hand_types={"Pair", "Flush"} if i % 2 else set(),
                        shop_vouchers={"v_clearance_sale"},
                    )  # fmt: skip
                    res.push(fingerprint(pool_type, i, flags, showman, out, dict(rng.state)))
    everything = set(CENTER_POOLS.get("Tarot", []))
    res.push(fingerprint(get_current_pool("Tarot", PseudoRandom("F"), 1, banned_keys=everything)))

    # create_card over game states with enhanced cards in different areas,
    # including only in the discard pile, and a (sometimes debuffed) Showman.
    areas = ("deck", "hand", "discard_pile")
    enhancements = ("m_glass", "m_steel", "m_lucky", "m_gold", "m_stone")
    for i in range(60):
        rng = PseudoRandom(f"CC{i}")
        gs: dict[str, Any] = {a: [] for a in areas}
        gs["jokers"] = []
        gs["used_jokers"] = {"j_joker"}
        gs["used_vouchers"] = {}
        gs["pool_flags"] = {"gros_michel_extinct": i % 3 == 0}
        gs[areas[i % 3]].append(
            create_playing_card(Suit.HEARTS, Rank.SEVEN, enhancement=enhancements[i % 5])
        )
        if i % 4 == 0:
            showman = create_joker("j_ring_master")
            showman.debuff = i % 8 == 0
            gs["jokers"].append(showman)
        # The gated Joker pools as get_current_pool sees them for this state
        # (enhancement gates read every owned playing card, in any area).
        for rarity in (1, 2, 3):
            pool = get_current_pool(
                "Joker", PseudoRandom(f"G{i}"), 1, rarity=rarity,
                deck_enhancements=deck_enhancements_from_state(gs),
                has_showman=showman_active(gs), used_jokers=gs["used_jokers"],
            )  # fmt: skip
            res.push(fingerprint("gated", rarity, pool))
        for card_type in ("Joker", "Tarot", "Planet", "Spectral", "Base", "Enhanced") * 3:
            try:
                card = create_card(card_type, rng, 1 + i % 8, game_state=gs, append=f"p{i}")
                out: Any = (card.center_key, card.card_key, card.edition, card.ability)
            except Exception as exc:  # noqa: BLE001
                out = ("raised", type(exc).__name__, str(exc))
            res.push(fingerprint(card_type, out, dict(rng.state)))


def _enumerate_probe(res: Result) -> None:
    from jackdaw.engine.card_factory import create_consumable, create_joker

    n_types = len(ActionType)
    consumables = [
        create_consumable(k)
        for k in ("c_magician", "c_death", "c_fool", "c_aura", "c_hanged_man", "c_cryptid")
    ]
    pack = [create_consumable("c_lovers"), create_joker("j_joker"), create_consumable("c_sigil")]
    raw = {"consumables": consumables, "pack_cards": pack}
    gym = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
    gen = random.Random(5)
    for n in range(0, 16):
        for lo, hi in ((1, 5), (1, 1), (2, 3), (0, 0), (5, 5), (3, 2)):
            for variant in range(4):
                gym._rng = np.random.default_rng(n * 1000 + lo * 10 + hi + variant)
                type_mask = np.zeros(n_types, dtype=bool)
                for t in gen.sample(range(n_types), gen.randint(1, n_types)):
                    type_mask[t] = True
                if variant == 0:
                    type_mask[ActionType.PlayHand] = type_mask[ActionType.Discard] = True
                card_mask = np.array([gen.random() < 0.85 for _ in range(n)], dtype=bool)
                entity_masks = {
                    int(t): np.array(
                        [gen.random() < 0.7 for _ in range(gen.randint(0, 8))], dtype=bool
                    )
                    for t in np.flatnonzero(type_mask)
                    if t not in (ActionType.PlayHand, ActionType.Discard) and gen.random() < 0.8
                }
                mask = GameActionMask(type_mask, card_mask, entity_masks, lo, hi)
                table = gym._enumerate_actions(mask, {"raw_state": raw})
                res.push(fingerprint(n, lo, hi, table, gym._rng.bit_generator.state))


_PROBE_FUNCS: dict[str, Callable[[Result], None]] = {
    "rng": _rng_probe,
    "hand_eval": _hand_eval_probe,
    "pools": _pools_probe,
    "enumerate": _enumerate_probe,
}


def run_probe(name: str, tape: list[Any] | None = None) -> Result:
    res = Result()
    with _fresh_sort_ids():
        _PROBE_FUNCS[name](res)
    return res


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def all_cases() -> list[tuple[str, Callable[..., Result], Any]]:
    """``(name, runner, spec)``; ``runner(spec, tape)`` returns a Result."""
    cases: list[tuple[str, Callable[..., Result], Any]] = []
    cases += [(s.name, run_engine, s) for s in engine_specs()]
    cases += [(s.name, run_factored, s) for s in factored_specs()]
    cases += [(s.name, run_gym, s) for s in gym_specs()]
    cases += [(f"probe_{p}", run_probe, p) for p in PROBES]
    return cases
