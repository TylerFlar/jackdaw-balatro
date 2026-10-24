"""The env controls: the cap on free moves in a row, refused actions, the vector env's autoreset
modes and partial resets, and the action space's version."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from jackdaw.core import IllegalActionError
from jackdaw.env import (
    ACTION_VERSION,
    DEFAULT_MAX_FREE_MOVES,
    FREE_ACTION_TYPES,
    ActionType,
    BalatroEnvironment,
    BalatroGymnasiumEnv,
    BalatroVectorEnv,
    DirectAdapter,
    FactoredAction,
)
from tests.env.generic import GenericAdapter

_ADAPTERS = pytest.mark.parametrize(
    "adapter", [DirectAdapter, GenericAdapter], ids=["direct", "generic"]
)


def _free_in_mask(mask: Any) -> bool:
    return any(bool(mask.type_mask[t]) for t in FREE_ACTION_TYPES)


def _free_in_table(env: BalatroGymnasiumEnv) -> bool:
    return any(a.action_type in FREE_ACTION_TYPES for a in env._action_table)


# ---------------------------------------------------------------------------
# The free-move cap
# ---------------------------------------------------------------------------


def test_the_default_cap() -> None:
    assert DEFAULT_MAX_FREE_MOVES == 10
    assert BalatroEnvironment(DirectAdapter).max_free_moves == DEFAULT_MAX_FREE_MOVES
    assert BalatroGymnasiumEnv(DirectAdapter).max_free_moves == DEFAULT_MAX_FREE_MOVES
    assert BalatroVectorEnv(1).max_free_moves == DEFAULT_MAX_FREE_MOVES
    assert {t.name for t in FREE_ACTION_TYPES} == {
        "SwapJokersLeft",
        "SwapJokersRight",
        "SwapHandLeft",
        "SwapHandRight",
        "SortHandRank",
        "SortHandSuit",
    }


@_ADAPTERS
def test_balatro_environment_caps_free_moves(adapter: type) -> None:
    env = BalatroEnvironment(adapter_factory=adapter, max_free_moves=3)
    env.reset(seed="CAP")
    _, _, _, mask, _ = env.step(FactoredAction(ActionType.SelectBlind))
    for _ in range(3):
        assert _free_in_mask(mask)
        _, _, _, mask, _ = env.step(FactoredAction(ActionType.SortHandSuit))
    assert not _free_in_mask(mask), "three free moves in a row: none more"
    assert mask.type_mask[ActionType.PlayHand]
    _, _, _, mask, _ = env.step(FactoredAction(ActionType.Discard, card_target=(0,)))
    assert _free_in_mask(mask), "another action starts the count again"


def test_no_cap() -> None:
    env = BalatroEnvironment(adapter_factory=DirectAdapter, max_free_moves=None)
    env.reset(seed="NOCAP")
    _, _, _, mask, _ = env.step(FactoredAction(ActionType.SelectBlind))
    for _ in range(50):
        _, _, _, mask, _ = env.step(FactoredAction(ActionType.SortHandRank))
    assert _free_in_mask(mask)


def _prefer_free(rng: np.random.Generator) -> Callable[[BalatroGymnasiumEnv], int]:
    """A policy that takes a free move whenever the table has one (so it reaches the cap)."""

    def pick(env: BalatroGymnasiumEnv) -> int:
        table = list(env._action_table)
        free = [i for i, a in enumerate(table) if a.action_type in FREE_ACTION_TYPES]
        pool = free if free and rng.random() < 0.9 else list(range(len(table)))
        return int(rng.choice(pool))

    return pick


def test_both_gym_paths_cap_alike() -> None:
    """The Rust Gym env and the generic path (BalatroEnvironment's mask, the Python table)
    apply the cap the same way."""
    rust = BalatroGymnasiumEnv(DirectAdapter, seed_prefix="CAPEQ", max_free_moves=2)
    generic = BalatroGymnasiumEnv(GenericAdapter, seed_prefix="CAPEQ", max_free_moves=2)
    pick = _prefer_free(np.random.default_rng(5))
    rust.reset(seed=4)
    generic.reset(seed=4)
    capped = 0
    for step in range(400):
        assert list(rust._action_table) == list(generic._action_table), f"step {step}"
        if not _free_in_table(rust):
            capped += 1
        action = pick(rust)
        r = rust.step(action)
        g = generic.step(action)
        assert r[1:4] == g[1:4], f"step {step}"
        if r[2] or r[3]:
            rust.reset()
            generic.reset()
    assert capped > 20, "the policy should reach the cap"


def test_the_vector_env_caps_free_moves() -> None:
    vec = BalatroVectorEnv(1, seed_prefix="VCAP", max_free_moves=1)
    _, info = vec.reset(seed=1)
    _, _, _, _, info = vec.step([0])  # SelectBlind
    table = vec.action_table(0)
    sort = next(i for i, a in enumerate(table) if a.action_type == ActionType.SortHandRank)
    _, _, _, _, info = vec.step([sort])
    assert not any(a.action_type in FREE_ACTION_TYPES for a in vec.action_table(0))


# ---------------------------------------------------------------------------
# Refused actions
# ---------------------------------------------------------------------------


class _Refusing(GenericAdapter):
    """Refuses every step, as the engine refuses an action the masks wrongly allowed."""

    def step(self, action: Any) -> Any:
        raise IllegalActionError("pretend")


def test_a_refusal_ends_the_episode() -> None:
    env = BalatroEnvironment(adapter_factory=_Refusing)
    _, _, info = env.reset()
    before = info["game"].state_hash()
    _, terminated, truncated, _, info = env.step(FactoredAction(ActionType.SelectBlind))
    assert (terminated, truncated) == (False, True)
    assert info["refused"] == "pretend"
    assert info["game"].state_hash() == before


def test_a_refusal_can_raise() -> None:
    env = BalatroEnvironment(adapter_factory=_Refusing, on_refusal="raise")
    env.reset()
    with pytest.raises(IllegalActionError):
        env.step(FactoredAction(ActionType.SelectBlind))
    gym = BalatroGymnasiumEnv(_Refusing, on_refusal="raise")
    gym.reset()
    with pytest.raises(IllegalActionError):
        gym.step(0)


def test_the_gym_env_flags_a_refusal() -> None:
    gym = BalatroGymnasiumEnv(_Refusing)
    gym.reset()
    _, _, terminated, truncated, info = gym.step(0)
    assert (terminated, truncated) == (False, True)
    assert info["balatro/refused"] == "pretend"
    assert not info["action_mask"].any()


def test_bad_settings() -> None:
    for make in (BalatroEnvironment, BalatroGymnasiumEnv):
        with pytest.raises(ValueError):
            make(DirectAdapter, on_refusal="ignore")
        with pytest.raises(ValueError):
            make(DirectAdapter, max_free_moves=-1)
    with pytest.raises(ValueError):
        BalatroVectorEnv(1, on_refusal="ignore")
    with pytest.raises(ValueError):
        BalatroVectorEnv(1, autoreset_mode=AutoresetMode.DISABLED)


# ---------------------------------------------------------------------------
# The vector env's autoreset modes and partial resets
# ---------------------------------------------------------------------------


def _same_obs(a: dict[str, np.ndarray], b: dict[str, np.ndarray], where: str) -> None:
    assert list(a) == list(b), where
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), f"{where}: {k}"


def test_same_step_autoreset_is_the_gym_env() -> None:
    gym = BalatroGymnasiumEnv(DirectAdapter, seed_prefix="SAME_0", reward_shaping=True)
    vec = BalatroVectorEnv(
        1,
        seed_prefix="SAME",
        reward_shaping=True,
        num_threads=1,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    assert vec.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP
    rng = np.random.default_rng(2)
    g_obs, g_info = gym.reset(seed=12)
    v_obs, v_info = vec.reset(seed=12)
    episodes = 0
    for pos in range(1500):
        where = f"step {pos}"
        _same_obs(g_obs, {k: v[0] for k, v in v_obs.items()}, where)
        assert g_info["action_mask"].tobytes() == v_info["action_mask"][0].tobytes(), where
        action = int(rng.choice(np.flatnonzero(g_info["action_mask"])))
        g_obs, g_r, g_term, g_trunc, g_info = gym.step(action)
        v_obs, v_r, v_term, v_trunc, v_info = vec.step([action])
        assert (g_r, g_term, g_trunc) == (float(v_r[0]), bool(v_term[0]), bool(v_trunc[0])), where
        if g_term or g_trunc:
            assert v_info["_final_obs"][0]
            _same_obs(g_obs, v_info["final_obs"][0], f"{where} final obs")
            final = v_info["final_info"]
            assert final["_balatro/ante_reached"][0]
            assert g_info["balatro/ante_reached"] == final["balatro/ante_reached"][0]
            assert g_info["balatro/rounds_beaten"] == final["balatro/rounds_beaten"][0]
            # The same step already started the next episode.
            g_obs, g_info = gym.reset()
            episodes += 1
        else:
            assert "final_obs" not in v_info
    assert episodes >= 5


def test_a_partial_reset() -> None:
    vec = BalatroVectorEnv(3, seed_prefix="PART", num_threads=2)
    obs, info = vec.reset(seed=30)
    for _ in range(3):
        actions = [int(np.flatnonzero(m)[0]) for m in info["action_mask"]]
        obs, _, _, _, info = vec.step(actions)
    before = {k: v.copy() for k, v in obs.items()}
    after, info = vec.reset(seed=[None, 9, None], options={"reset_mask": [False, True, False]})
    for k in before:
        assert after[k][0].tobytes() == before[k][0].tobytes()
        assert after[k][2].tobytes() == before[k][2].tobytes()
    fresh = BalatroGymnasiumEnv(DirectAdapter, seed_prefix="PART_1")
    f_obs, f_info = fresh.reset(seed=9)
    _same_obs(f_obs, {k: v[1] for k, v in after.items()}, "reset env")
    assert f_info["action_mask"].tobytes() == info["action_mask"][1].tobytes()
    assert vec.state(0).round == 1 and vec.state(1).round == 0
    with pytest.raises(ValueError):
        vec.reset(options={"reset_mask": [True]})


# ---------------------------------------------------------------------------
# The action space's version
# ---------------------------------------------------------------------------


def test_the_action_version() -> None:
    assert ACTION_VERSION == 2
    for env in (
        BalatroEnvironment(DirectAdapter),
        BalatroGymnasiumEnv(DirectAdapter),
        BalatroVectorEnv(1),
    ):
        assert env.action_version == ACTION_VERSION
    assert BalatroGymnasiumEnv.metadata["action_version"] == ACTION_VERSION
    assert BalatroVectorEnv(1).metadata["action_version"] == ACTION_VERSION
