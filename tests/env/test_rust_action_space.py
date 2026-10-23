"""The Rust action space matches ``jackdaw.env.action_space`` and the Gym table bit for bit.

``jackdaw._core`` ports the action mask (``get_action_mask``), the factored
action conversions (``factored_to_engine_action``, ``engine_action_to_factored``),
the default hand selections and the Gym wrapper's action table
(``BalatroGymnasiumEnv._enumerate_actions``, with its xoshiro256** sampler).
At every state of every parity tape and along about 10,000 fuzz states, the
Python functions read the live oracle state and the Rust ones the oracle's
canonical snapshot; the observation must match too.  Random masks cover the
table's edge cases (empty hands, inverted selection bounds, forced cards).
"""

from __future__ import annotations

import json
import random
import zlib
from typing import Any

import numpy as np
import pytest

from jackdaw import _core
from jackdaw.engine.actions import get_legal_actions
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.env.action_space import (
    ActionMask,
    ActionType,
    _default_buy_targets,
    _default_pick_targets,
    _default_use_targets,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
    get_consumable_target_info,
)
from jackdaw.env.balatro_env import _action_mask_to_game
from jackdaw.env.game_interface import OracleAdapter
from jackdaw.env.game_spec import FactoredAction, GameActionMask
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.observation import encode_observation
from jackdaw.env.rust_env import action_mask_from_parts
from jackdaw.env.rust_observation import observation_from_buffer
from jackdaw.env.sampler import Xoshiro256, subsample_indices
from tests.env.test_rust_observation import assert_same as assert_obs_same
from tests.golden.trajectories import encode_action
from tests.parity import canon
from tests.parity.fuzz import _choose, random_case
from tests.parity.harness import load_specs
from tests.parity.pyengine import PyEngine

_SPECS = load_specs()
# A wrapper instance only for its action-table builder.
_GYM = BalatroGymnasiumEnv(adapter_factory=OracleAdapter)


def assert_mask_same(py: ActionMask, rs: ActionMask, where: str) -> None:
    """Same arrays (dtype, shape, values), entity keys in the same order, same bounds."""
    for name in ("type_mask", "card_mask"):
        a, b = getattr(py, name), getattr(rs, name)
        assert a.dtype == b.dtype == np.bool_, f"{where} {name}: dtype {a.dtype} vs {b.dtype}"
        assert a.shape == b.shape and np.array_equal(a, b), (
            f"{where} {name}: py {a.astype(int).tolist()} rust {b.astype(int).tolist()}"
        )
    assert list(py.entity_masks) == list(rs.entity_masks), (
        f"{where} entity types: py {list(map(int, py.entity_masks))} "
        f"rust {list(map(int, rs.entity_masks))}"
    )
    for t, a in py.entity_masks.items():
        b = rs.entity_masks[t]
        assert a.dtype == b.dtype == np.bool_, f"{where} entity {int(t)}: dtype"
        assert a.shape == b.shape and np.array_equal(a, b), (
            f"{where} entity {int(t)}: py {a.astype(int).tolist()} rust {b.astype(int).tolist()}"
        )
    for name in ("min_card_select", "max_card_select", "forced_card"):
        assert getattr(py, name) == getattr(rs, name), f"{where} {name}"


def _parts(fa: FactoredAction) -> tuple[int, int | None, list[int] | None]:
    cards = list(fa.card_target) if fa.card_target is not None else None
    return int(fa.action_type), fa.entity_target, cards


def _engine(convert: Any) -> Any:
    try:
        return convert()
    except ValueError as exc:
        return ("ValueError", str(exc))


def check_state(
    engine: PyEngine, where: str, *, table_seed: int | None = None, conversions: bool = True
) -> dict[str, Any]:
    """Compare observation, mask, table and conversions on the oracle's current state."""
    snap = engine.snapshot()
    game = _core.Game.from_snapshot(json.dumps(canon.to_json_compatible(snap)))
    with engine._ids():
        gs = engine.gs
        assert_obs_same(encode_observation(gs), observation_from_buffer(*game.observation()), where)
        py_mask = get_action_mask(gs)
        assert_mask_same(py_mask, action_mask_from_parts(game.action_mask()), where)
        if table_seed is not None:
            _GYM._rng = Xoshiro256(table_seed)
            table = _GYM._enumerate_actions(_action_mask_to_game(py_mask), {"raw_state": gs})
            rs_table, rs_state = game.action_table(list(Xoshiro256(table_seed).state))
            assert [_parts(a) for a in table] == [tuple(a) for a in rs_table], (
                f"{where}: action tables differ"
            )
            assert _GYM._rng.state == tuple(rs_state), f"{where}: sampler states differ"
        # At game over (cards may be left in hand) the oracle's consumable gate, which ignores
        # the phase, still names default selections; the engine's gates on the game state and
        # names none. No action is legal there, so only conversions of illegal actions differ.
        if not conversions or snap["phase"] == "game_over":
            return snap
        # Factored -> engine for a sample of the table, then for every entity type with no
        # card target (the engine's default selection).
        samples = [a for a in (table[::13] if table_seed is not None else [])]
        for t in (ActionType.UseConsumable, ActionType.PickPackCard, ActionType.BuyAndUse):
            samples += [FactoredAction(action_type=t, entity_target=e) for e in range(4)]
        samples += [FactoredAction(action_type=ActionType.PlayHand)]
        for fa in samples:
            py_a = _engine(lambda fa=fa: encode_action(factored_to_engine_action(fa, gs)))
            rs_a = _engine(
                lambda fa=fa: json.loads(
                    game.factored_to_engine(int(fa.action_type), fa.entity_target, fa.card_target)
                )
            )
            assert py_a == rs_a, f"{where}: {fa} -> python {py_a}, rust {rs_a}"
        # Engine -> factored for every legal action.
        for a in get_legal_actions(gs):
            t, e, c = _core.engine_to_factored(json.dumps(encode_action(a)))
            assert _parts(engine_action_to_factored(a, gs)) == (t, e, c), f"{where}: {a}"
        # Default selections and targeting info.
        defaults = (
            ("consumables", _default_use_targets),
            ("pack_cards", _default_pick_targets),
            ("shop_jokers", _default_buy_targets),
        )
        for area, py_default in defaults:
            for i in range(len(snap["areas"][area])):
                py_t = py_default(gs, i)
                assert game.default_targets(area, i) == (list(py_t) if py_t else py_t), (
                    f"{where}: default targets of {area}[{i}]"
                )
        for area, key in (("consumables", "consumables"), ("pack_cards", "pack_cards")):
            for i, card in enumerate(gs.get(key, [])):
                info = tuple(get_consumable_target_info(card))
                assert game.consumable_target_info(area, i) == info, f"{where}: info {area}[{i}]"
    return snap


# ---------------------------------------------------------------------------
# Tapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_tape_states_match(spec: Any) -> None:
    engine = PyEngine(spec.start, spec.setup)
    for pos in range(len(spec.tape) + 1):
        where = f"{spec.name}@{pos}"
        snap = check_state(engine, where, table_seed=zlib.crc32(where.encode()))
        if snap["phase"] == "game_over" or snap["run"]["won"] or pos == len(spec.tape):
            return
        try:
            engine.step(spec.tape[pos])
        except Exception:  # the oracle refuses the action: the tape ends here
            return


# ---------------------------------------------------------------------------
# Fuzz: random starts, setups (bosses, joker boards, money) and policies
# ---------------------------------------------------------------------------

_FUZZ_CHUNKS = 16
_FUZZ_CASES = 14  # per chunk: about 10,000 states in all


@pytest.mark.parametrize("chunk", range(_FUZZ_CHUNKS))
def test_fuzz_states_match(chunk: int) -> None:
    rng = random.Random(f"ENV_FUZZ_{chunk}")
    for i in range(_FUZZ_CASES):
        case = random_case(chunk * _FUZZ_CASES + i, rng, max_steps=300)
        engine = PyEngine(case.start, case.setup)
        policy = random.Random(case.policy_seed)
        for pos in range(case.max_steps):
            where = f"{case.name}@{pos}"
            # Tables and conversions on every fourth state keep the suite fast.
            full = pos % 4 == 0
            snap = check_state(
                engine,
                where,
                table_seed=zlib.crc32(where.encode()) if full else None,
                conversions=full,
            )
            if snap["phase"] == "game_over" or snap["run"]["won"]:
                break
            action = _choose(engine, policy, case.smart)
            if action is None:
                break
            try:
                engine.step(action)
            except Exception:
                break


# ---------------------------------------------------------------------------
# The table on arbitrary masks, and the sampler
# ---------------------------------------------------------------------------


def test_tables_on_random_masks() -> None:
    consumables = [
        create_consumable(k)
        for k in ("c_magician", "c_death", "c_fool", "c_aura", "c_hanged_man", "c_cryptid")
    ]
    pack = [create_consumable("c_lovers"), create_joker("j_joker"), create_consumable("c_sigil")]
    raw = {"consumables": consumables, "pack_cards": pack}
    info = {
        (int(t), e): tuple(get_consumable_target_info(card))
        for t, source in ((ActionType.UseConsumable, consumables), (ActionType.PickPackCard, pack))
        for e, card in enumerate(source)
    }
    n_types = len(ActionType)
    gen = random.Random(5)
    for case in range(600):
        n = gen.randint(0, 15)
        lo, hi = gen.choice(((1, 5), (1, 1), (2, 3), (0, 0), (5, 5), (3, 2)))
        type_mask = np.zeros(n_types, dtype=bool)
        for t in gen.sample(range(n_types), gen.randint(1, n_types)):
            type_mask[t] = True
        if case % 4 == 0:
            type_mask[ActionType.PlayHand] = type_mask[ActionType.Discard] = True
        card_mask = np.array([gen.random() < 0.85 for _ in range(n)], dtype=bool)
        entity_masks = {
            int(t): np.array([gen.random() < 0.7 for _ in range(gen.randint(0, 8))], dtype=bool)
            for t in np.flatnonzero(type_mask)
            if t not in (ActionType.PlayHand, ActionType.Discard) and gen.random() < 0.8
        }
        forced = gen.randrange(n) if n and gen.random() < 0.3 else None
        mask = GameActionMask(type_mask, card_mask, entity_masks, lo, hi, forced)
        seed = gen.getrandbits(64)
        _GYM._rng = Xoshiro256(seed)
        table = _GYM._enumerate_actions(mask, {"raw_state": raw})
        rs_table, rs_state = _core._enumerate_actions(
            type_mask.tolist(),
            card_mask.tolist(),
            {t: m.tolist() for t, m in entity_masks.items()},
            lo,
            hi,
            forced,
            info,
            list(Xoshiro256(seed).state),
        )
        assert [_parts(a) for a in table] == [tuple(a) for a in rs_table], f"case {case}"
        assert _GYM._rng.state == tuple(rs_state), f"case {case}: sampler states"


def test_sampler_matches_rust() -> None:
    gen = random.Random(3)
    for _ in range(2000):
        n = gen.randint(0, 900)
        k = gen.randint(0, n)
        seed = gen.getrandbits(64)
        rng = Xoshiro256(seed)
        got = subsample_indices(n, k, rng)
        want, state = _core._subsample_indices(n, k, list(Xoshiro256(seed).state))
        assert got == want and rng.state == tuple(state), (n, k)
        assert len(got) == k and got == sorted(set(got))


def test_xoshiro_reference_outputs() -> None:
    # xoshiro256** from state [1, 2, 3, 4], as the reference C implementation computes it.
    rng = Xoshiro256.from_state([1, 2, 3, 4])
    assert [rng.next_u64() for _ in range(4)] == [11520, 0, 1509978240, 1215971899390074240]
