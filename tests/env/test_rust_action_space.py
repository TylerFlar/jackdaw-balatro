"""The Rust action space: masks, factored-action conversions and the Gym action table.

``jackdaw._core`` ports the action mask, the factored action conversions, the default hand
selections and the Gym wrapper's action table (``BalatroGymnasiumEnv._enumerate_actions``,
with its xoshiro256** sampler). The Python mask and conversions read the oracle's state; their
outputs are kept as digests (:mod:`tests.parity.frozen`). The parity tapes' states are checked
in ``test_golden_env.py``; here, about 10,000 states of seeded fuzz runs
(``tests/fixtures/env_fuzz.json``). The Python table builder stays and must match Rust's table
at every one of those states and on random masks covering its edge cases (empty hands,
inverted selection bounds, forced cards).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import pytest

from jackdaw import _core
from jackdaw.core import Game, IllegalActionError, Legality
from jackdaw.env.action_space import ActionMask, ActionType
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.game_spec import GameActionMask
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.rust_env import action_mask
from jackdaw.env.sampler import Xoshiro256, subsample_indices
from tests.parity import frozen
from tests.parity.harness import load_specs

_SPECS = load_specs()
_FUZZ = frozen.load("env_fuzz.json")["action_space"]
# A wrapper instance only for its Python action-table builder.
_GYM = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)


def assert_mask_same(py: ActionMask, rs: ActionMask, where: str) -> None:
    """Same arrays (dtype, shape, values), entity keys in the same order, same bounds."""
    for name in ("type_mask", "card_mask"):
        a, b = getattr(py, name), getattr(rs, name)
        assert a.dtype == b.dtype == np.bool_, f"{where} {name}: dtype {a.dtype} vs {b.dtype}"
        assert a.shape == b.shape and np.array_equal(a, b), (
            f"{where} {name}: {a.astype(int).tolist()} vs {b.astype(int).tolist()}"
        )
    assert list(py.entity_masks) == list(rs.entity_masks), (
        f"{where} entity types: {list(map(int, py.entity_masks))} "
        f"vs {list(map(int, rs.entity_masks))}"
    )
    for t, a in py.entity_masks.items():
        b = rs.entity_masks[t]
        assert a.dtype == b.dtype == np.bool_, f"{where} entity {int(t)}: dtype"
        assert a.shape == b.shape and np.array_equal(a, b), (
            f"{where} entity {int(t)}: {a.astype(int).tolist()} vs {b.astype(int).tolist()}"
        )
    for name in ("min_card_select", "max_card_select", "forced_card"):
        assert getattr(py, name) == getattr(rs, name), f"{where} {name}"


# ---------------------------------------------------------------------------
# Fuzz: random starts, setups (bosses, joker boards, money) and policies
# ---------------------------------------------------------------------------

_FUZZ_CHUNKS = 16


@pytest.mark.parametrize("chunk", range(_FUZZ_CHUNKS))
def test_fuzz_states_match(chunk: int) -> None:
    """The observation and mask at every state, the tables and conversions on every fourth."""
    for run in _FUZZ[chunk::_FUZZ_CHUNKS]:
        frozen.replay_env_run(run, every=4)


# ---------------------------------------------------------------------------
# The table on arbitrary masks, and the sampler
# ---------------------------------------------------------------------------

_CONSUMABLES = ("c_magician", "c_death", "c_fool", "c_aura", "c_hanged_man", "c_cryptid")


class _Targets:
    """Stands in for a run in the Python table builder: only the targeting info of the
    consumables and pack cards."""

    def __init__(self, info: dict[str, list[tuple[int, int, bool]]]) -> None:
        self._g = self
        self._info = info

    def consumable_target_info(self, area: str, index: int) -> tuple[int, int, bool]:
        if index >= len(self._info[area]):
            raise ValueError(f"no card {index} in {area}")
        return self._info[area][index]


def _target_info() -> dict[str, list[tuple[int, int, bool]]]:
    """Real targeting info: six consumables, and a pack of The Lovers, a joker and Sigil."""
    game = Game.new("b_red", 1, "TABLES")
    game.step(["SelectBlind"])
    for key in (*_CONSUMABLES, "c_lovers", "c_sigil"):
        game.debug_add(key)
    game.debug_add("j_joker")
    info = [tuple(game._g.consumable_target_info("consumables", i)) for i in range(8)]
    joker = tuple(game._g.consumable_target_info("jokers", 0))
    return {"consumables": info[:6], "pack_cards": [info[6], joker, info[7]]}


def test_tables_on_random_masks() -> None:
    targets = _target_info()
    info = {
        (int(t), e): target
        for t, area in (
            (ActionType.UseConsumable, "consumables"),
            (ActionType.PickPackCard, "pack_cards"),
        )
        for e, target in enumerate(targets[area])
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
        table = _GYM._enumerate_actions(mask, {"game": _Targets(targets)})
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
        assert frozen.table_parts(table) == frozen.table_parts(rs_table), f"case {case}"
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


# ---------------------------------------------------------------------------
# BALATROBOT legality
# ---------------------------------------------------------------------------


def _tape_states(spec: Any, limit: int = 120) -> Any:
    """The Rust run at each state along the first ``limit`` steps of a tape."""
    game = frozen.new_game(spec.start, spec.setup)
    for pos, action in enumerate(spec.tape[:limit]):
        if game.phase == "game_over":
            return
        yield pos, game
        try:
            game.step(action)
        except IllegalActionError:
            return


@pytest.mark.parametrize("spec", _SPECS[::10], ids=[s.name for s in _SPECS[::10]])
def test_balatrobot_masks_narrow_the_vanilla_ones(spec) -> None:
    """BALATROBOT legality (jackdaw-bot's dry run of the endpoint checks) only
    takes actions away: never a sim-only type, never one vanilla refuses."""
    for pos, game in _tape_states(spec):
        vanilla = action_mask(game)
        bot = action_mask(game, Legality.BALATROBOT)
        where = f"{spec.name}@{pos}"
        assert not (bot.type_mask & ~vanilla.type_mask).any(), where
        assert not bot.type_mask[ActionType.RerollBoss], where
        assert not bot.type_mask[ActionType.BuyAndUse], where
        for t, m in bot.entity_masks.items():
            assert not (m & ~vanilla.entity_masks[t]).any(), f"{where} entity {int(t)}"


_SWAPS = (
    ActionType.SwapJokersLeft,
    ActionType.SwapJokersRight,
    ActionType.SwapHandLeft,
    ActionType.SwapHandRight,
)


def test_balatrobot_reorders_only_where_rearrange_works() -> None:
    """G206: vanilla reorders the jokers in every phase and the hand whenever it
    holds cards. BalatroBot's ``rearrange`` takes SELECTING_HAND, SHOP and
    SMODS_BOOSTER_OPENED (rearrange.lua's requires_state), so the BALATROBOT mask
    drops the swaps at blind select and round eval and keeps vanilla's elsewhere."""
    seen: Counter[str] = Counter()
    for spec in _SPECS[::10]:
        for pos, game in _tape_states(spec):
            phase = game.phase
            vanilla = action_mask(game)
            bot = action_mask(game, Legality.BALATROBOT)
            for t in _SWAPS:
                where = f"{spec.name}@{pos} {phase} {t.name}"
                if not vanilla.type_mask[t]:
                    assert not bot.type_mask[t], where
                    continue
                seen[f"{phase}:{t.name}"] += 1
                if phase in ("blind_select", "round_eval"):
                    assert not bot.type_mask[t] and t not in bot.entity_masks, where
                else:
                    assert bot.type_mask[t], where
                    assert np.array_equal(bot.entity_masks[t], vanilla.entity_masks[t]), where
    for phase, kinds in (
        ("blind_select", ("SwapJokersLeft",)),
        ("round_eval", ("SwapJokersLeft",)),
        ("pack_opening", ("SwapJokersLeft", "SwapHandLeft")),
        ("shop", ("SwapJokersLeft",)),
        ("selecting_hand", ("SwapJokersLeft", "SwapHandLeft")),
    ):
        for kind in kinds:
            assert seen[f"{phase}:{kind}"], (phase, kind, dict(seen))
