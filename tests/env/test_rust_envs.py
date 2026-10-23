"""The environments on the Rust engine play exactly as on the Python engine.

``BalatroEnvironment`` and ``BalatroGymnasiumEnv`` with ``DirectAdapter``
(the Rust engine, ``jackdaw._core``) are stepped side by side with the same
environments on ``OracleAdapter`` (the Python engine), with the same seeds and
actions: observations and masks must be bitwise equal, and states, flags,
rewards, action tables and episode stats equal.  ``BalatroVectorEnv`` with one
env must step exactly as the Rust Gym env (autoreset included), and give the
same results whatever its thread count.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from jackdaw.core import Game
from jackdaw.core.game import NotPortedError
from jackdaw.engine import card as card_module
from jackdaw.engine.actions import GamePhase, PlayHand, SelectBlind
from jackdaw.env import (
    BalatroEnvironment,
    BalatroGymnasiumEnv,
    BalatroVectorEnv,
    DirectAdapter,
    GameAdapter,
    OracleAdapter,
    ShapedReward,
    Transition,
    blind_progress_reward,
    sparse_reward,
)
from jackdaw.env.action_space import ActionType, engine_action_to_factored
from jackdaw.env.game_spec import FactoredAction, GameActionMask, GameObservation
from jackdaw.env.rust_gym import MAX_ACTIONS, OBS_LEN
from tests.env.test_rust_action_space import assert_mask_same
from tests.golden.trajectories import _gym_smart_index, _mixed_policy


def _same_array(a: np.ndarray, b: np.ndarray, where: str) -> None:
    assert a.dtype == b.dtype, f"{where}: dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{where}: shape {a.shape} vs {b.shape}"
    assert a.tobytes() == b.tobytes(), f"{where}: values differ"


def _same_game_obs(a: GameObservation, b: GameObservation, where: str) -> None:
    _same_array(a.global_context, b.global_context, f"{where} global")
    assert list(a.entities) == list(b.entities), where
    for k in a.entities:
        _same_array(a.entities[k], b.entities[k], f"{where} {k}")


def _same_game_mask(a: GameActionMask, b: GameActionMask, where: str) -> None:
    _same_array(a.type_mask, b.type_mask, f"{where} type_mask")
    _same_array(a.card_mask, b.card_mask, f"{where} card_mask")
    assert list(a.entity_masks) == list(b.entity_masks), f"{where} entity types"
    for k in a.entity_masks:
        _same_array(a.entity_masks[k], b.entity_masks[k], f"{where} entity {int(k)}")
    for f in ("min_card_select", "max_card_select", "forced_card"):
        assert getattr(a, f) == getattr(b, f), f"{where} {f}"


def _same_gym_obs(a: dict[str, np.ndarray], b: dict[str, np.ndarray], where: str) -> None:
    assert list(a) == list(b), where
    for k in a:
        _same_array(a[k], b[k], f"{where} {k}")


# ---------------------------------------------------------------------------
# BalatroEnvironment
# ---------------------------------------------------------------------------


def _check_env_reset(rust: BalatroEnvironment, oracle: BalatroEnvironment, where: str) -> Any:
    card_module.reset_sort_id_counter()  # the oracle numbers cards from 0, as the Rust engine
    o_obs, o_mask, o_info = oracle.reset()
    r_obs, r_mask, r_info = rust.reset()
    _same_game_obs(o_obs, r_obs, where)
    _same_game_mask(o_mask, r_mask, where)
    assert_mask_same(o_info["action_mask"], r_info["action_mask"], where)
    assert o_info["state"] == r_info["state"], where
    assert tuple(o_info["shop_splits"]) == tuple(r_info["shop_splits"]), where
    assert isinstance(r_info["game"], Game) and "raw_state" not in r_info
    return o_info


@pytest.mark.parametrize(
    ("back", "stake", "eps"),
    [("b_red", 1, 0.1), ("b_plasma", 3, 0.3), ("b_ghost", 5, 0.2), ("b_magic", 8, 0.5)],
)
def test_balatro_environment_matches_the_oracle(back: str, stake: int, eps: float) -> None:
    kwargs = {"back_keys": [back], "stakes": [stake], "seed_prefix": f"ENVPAR_{back}_{stake}"}
    rust = BalatroEnvironment(adapter_factory=DirectAdapter, max_steps=300, **kwargs)
    oracle = BalatroEnvironment(adapter_factory=OracleAdapter, max_steps=300, **kwargs)
    policy = random.Random(f"{back}{stake}")
    steps = 0
    for episode in range(5):
        o_info = _check_env_reset(rust, oracle, f"{back} ep{episode} reset")
        for pos in range(300):
            where = f"{back} ep{episode}@{pos}"
            gs = o_info["raw_state"]
            legal = oracle._adapter.get_legal_actions()
            action = _mixed_policy(gs, legal, policy, eps)
            fa = engine_action_to_factored(action, gs)
            o_obs, o_term, o_trunc, o_mask, o_info = oracle.step(fa)
            r_obs, r_term, r_trunc, r_mask, r_info = rust.step(fa)
            assert "not_ported" not in r_info, f"{where}: {r_info['not_ported']}"
            steps += 1
            _same_game_obs(o_obs, r_obs, where)
            _same_game_mask(o_mask, r_mask, where)
            assert_mask_same(o_info["action_mask"], r_info["action_mask"], where)
            assert (o_term, o_trunc) == (r_term, r_trunc), where
            assert o_info["prev_state"] == r_info["prev_state"], where
            assert o_info["state"] == r_info["state"], where
            assert tuple(o_info["shop_splits"]) == tuple(r_info["shop_splits"]), where
            for attr in ("episode_length", "episode_won", "episode_ante"):
                assert getattr(oracle, attr) == getattr(rust, attr), f"{where} {attr}"
            if o_term or o_trunc:
                break
    assert steps > 50


def test_not_ported_ends_the_episode_truncated() -> None:
    class Unported(DirectAdapter):
        """Refuses every step as not ported (the generic Rust path calls adapter.step)."""

        def step(self, action: Any) -> Any:
            raise NotPortedError("pretend")

    env = BalatroEnvironment(adapter_factory=Unported)
    _, _, info = env.reset()
    before = info["game"].state_hash()
    _, terminated, truncated, _, info = env.step(FactoredAction(ActionType.SelectBlind))
    assert (terminated, truncated) == (False, True)
    assert info["not_ported"] == "pretend"
    assert info["game"].state_hash() == before
    assert info["state"] == info["prev_state"]


def test_direct_adapter_runs_the_rust_engine() -> None:
    adapter = DirectAdapter()
    assert isinstance(adapter, GameAdapter)
    state = adapter.reset("b_red", 1, "ADAPTER")
    assert state.phase == GamePhase.BLIND_SELECT and not adapter.done
    assert isinstance(adapter.game, Game)
    assert not hasattr(adapter, "raw_state")
    legal = adapter.get_legal_actions()
    assert SelectBlind() in legal
    after = adapter.step(SelectBlind())
    assert after.phase == GamePhase.SELECTING_HAND and after.round == 1
    frozen = adapter.game.freeze()
    adapter.step(PlayHand(card_indices=(0,)))
    assert frozen.state_hash() != adapter.game.state_hash()


# ---------------------------------------------------------------------------
# BalatroGymnasiumEnv
# ---------------------------------------------------------------------------


def _custom_reward(t: Transition) -> float:
    reward = 0.5 if t.blind_beaten else 0.0
    return reward + 0.001 * max(t.curr.dollars - t.prev.dollars, 0) - (1.0 if t.done else 0.0)


_GYM_CASES: dict[str, dict[str, Any]] = {
    "sparse": {"reward_fn": sparse_reward},
    "blind_progress": {"reward_fn": blind_progress_reward},
    "shaped": {"reward_shaping": True},
    "shaped_custom_weights": {"reward_fn": ShapedReward(win=1.0, step_cost=0.0, loss=0.5)},
    "custom": {"reward_fn": _custom_reward},
    "truncated": {"reward_shaping": True, "max_steps": 25},
    "endless": {"reward_fn": blind_progress_reward, "end_on_win": False},
}


def _gym_pair(case: str, back: str, stake: int) -> tuple[BalatroGymnasiumEnv, BalatroGymnasiumEnv]:
    kwargs = dict(_GYM_CASES[case])
    kwargs.update(back_keys=[back], stakes=[stake], seed_prefix=f"GYMPAR_{case}")
    return (
        BalatroGymnasiumEnv(adapter_factory=DirectAdapter, **kwargs),
        BalatroGymnasiumEnv(adapter_factory=OracleAdapter, **kwargs),
    )


def _check_tables(rust: BalatroGymnasiumEnv, oracle: BalatroGymnasiumEnv, where: str) -> None:
    assert len(rust._action_table) == len(oracle._action_table), f"{where}: table size"
    assert list(rust._action_table) == list(oracle._action_table), f"{where}: tables differ"


@pytest.mark.parametrize("case", sorted(_GYM_CASES))
def test_gym_env_matches_the_oracle(case: str) -> None:
    rust, oracle = _gym_pair(case, "b_red", 1 + len(case) % 4)
    policy = np.random.default_rng(len(case))
    steps = 0
    for episode in range(6):
        card_module.reset_sort_id_counter()
        seed = 9000 + episode if episode < 5 else None  # the last reset is unseeded
        o_obs, o_info = oracle.reset(seed=seed)
        r_obs, r_info = rust.reset(seed=seed)
        _same_gym_obs(o_obs, r_obs, f"{case} reset {episode}")
        _same_array(o_info["action_mask"], r_info["action_mask"], f"{case} reset mask")
        _check_tables(rust, oracle, f"{case} reset {episode}")
        for pos in range(250):
            where = f"{case} ep{episode}@{pos}"
            if pos % 5:
                action = _gym_smart_index(oracle, policy)
            else:
                action = int(policy.choice(np.flatnonzero(o_info["action_mask"])))
            o_obs, o_r, o_term, o_trunc, o_info = oracle.step(action)
            r_obs, r_r, r_term, r_trunc, r_info = rust.step(action)
            assert "balatro/not_ported" not in r_info, where
            steps += 1
            _same_gym_obs(o_obs, r_obs, where)
            assert (o_r, o_term, o_trunc) == (r_r, r_term, r_trunc), where
            assert list(o_info) == list(r_info), where
            _same_array(o_info["action_mask"], r_info["action_mask"], f"{where} mask")
            for key in ("balatro/ante_reached", "balatro/rounds_beaten", "balatro/won"):
                assert o_info.get(key) == r_info.get(key), f"{where} {key}"
            _check_tables(rust, oracle, where)
            if o_term or o_trunc:
                break
    assert steps > 40


def test_gym_env_rejects_bad_actions() -> None:
    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
    _, info = env.reset(seed=1)
    with pytest.raises(IndexError):
        env.step(int(info["action_mask"].sum()))
    assert isinstance(env.game, Game)


# ---------------------------------------------------------------------------
# BalatroVectorEnv
# ---------------------------------------------------------------------------


def _vec_policy(rng: np.random.Generator) -> Callable[[np.ndarray], int]:
    def pick(mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal)) if legal.size else 0

    return pick


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward_fn": sparse_reward},
        {"reward_shaping": True, "max_steps": 40},
        {"reward_fn": blind_progress_reward, "back_keys": ["b_yellow"], "stakes": [4]},
    ],
    ids=["sparse", "shaped_truncated", "blind_progress"],
)
def test_vector_env_of_one_is_the_gym_env(kwargs: dict[str, Any]) -> None:
    gym = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, seed_prefix="VEQ_0", **kwargs)
    vec = BalatroVectorEnv(1, seed_prefix="VEQ", num_threads=1, **kwargs)
    pick = _vec_policy(np.random.default_rng(11))
    g_obs, g_info = gym.reset(seed=77)
    v_obs, v_info = vec.reset(seed=77)
    episodes = 0
    for pos in range(1500):
        where = f"step {pos}"
        _same_gym_obs(g_obs, {k: v[0] for k, v in v_obs.items()}, where)
        _same_array(g_info["action_mask"], v_info["action_mask"][0], f"{where} mask")
        action = pick(g_info["action_mask"])
        g_obs, g_r, g_term, g_trunc, g_info = gym.step(action)
        v_obs, v_r, v_term, v_trunc, v_info = vec.step([action])
        assert (g_r, g_term, g_trunc) == (float(v_r[0]), bool(v_term[0]), bool(v_trunc[0])), where
        if g_term or g_trunc:
            assert v_info["_balatro/ante_reached"][0]
            assert g_info["balatro/ante_reached"] == v_info["balatro/ante_reached"][0]
            assert g_info["balatro/rounds_beaten"] == v_info["balatro/rounds_beaten"][0]
            assert g_info["balatro/won"] == v_info["balatro/won"][0]
            _same_gym_obs(g_obs, {k: v[0] for k, v in v_obs.items()}, f"{where} final")
            # Next-step autoreset: the vector env resets on its next step.
            g_obs, g_info = gym.reset()
            v_obs, v_r, v_term, v_trunc, v_info = vec.step([0])
            assert (float(v_r[0]), bool(v_term[0]), bool(v_trunc[0])) == (0.0, False, False)
            episodes += 1
    assert episodes >= 5


def _vector_run(threads: int) -> list[Any]:
    vec = BalatroVectorEnv(12, seed_prefix="THREADS", reward_shaping=True, num_threads=threads)
    pick = _vec_policy(np.random.default_rng(3))
    obs, info = vec.reset(seed=500)
    record: list[Any] = [obs["global"].tobytes(), info["action_mask"].tobytes()]
    for _ in range(400):
        actions = [pick(m) for m in info["action_mask"]]
        obs, rewards, term, trunc, info = vec.step(actions)
        record.append(b"".join(v.tobytes() for v in obs.values()))
        record.append((rewards.tobytes(), term.tobytes(), trunc.tobytes()))
        record.append(info["action_mask"].tobytes())
        record.append({k: np.asarray(v).tobytes() for k, v in info.items() if k != "action_mask"})
    return record


def test_vector_env_results_do_not_depend_on_threads() -> None:
    one = _vector_run(1)
    assert one == _vector_run(4)
    assert one == _vector_run(8)


def test_vector_env_api() -> None:
    vec = BalatroVectorEnv(3, num_threads=2, copy=False)
    obs, info = vec.reset(seed=[1, None, 3])
    assert vec.observation_space.contains(obs)
    assert info["action_mask"].shape == (3, MAX_ACTIONS)
    assert vec.action_masks().dtype == np.bool_
    assert vec.num_threads == 2
    assert isinstance(vec.game(1), Game)
    assert vec.state(2).phase == GamePhase.BLIND_SELECT
    assert vec.action_table(0)[0].action_type == ActionType.SelectBlind
    flat = np.zeros((3, OBS_LEN), dtype=np.float32)
    assert flat.size == sum(v.size for v in obs.values())
    with pytest.raises(IndexError):  # past the table: nothing is stepped
        vec.step([0, 0, 10_000])
    assert vec.state(0).phase == GamePhase.BLIND_SELECT
    with pytest.raises(TypeError):
        BalatroVectorEnv(1, reward_fn=lambda t: 0.0)
