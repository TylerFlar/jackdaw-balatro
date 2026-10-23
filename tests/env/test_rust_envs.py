"""The environments on the Rust engine, against the oracle's frozen runs.

``BalatroEnvironment`` and ``BalatroGymnasiumEnv`` were stepped side by side on the Rust
engine (``DirectAdapter``) and on the Python engine (``OracleAdapter``), with the same seeds
and actions, until the Python engine was retired. The runs are kept in
``tests/fixtures/env_runs.json``: each episode's actions and a digest chain of what every reset
and step returned (observations, masks, states, flags, rewards, action tables, episode stats;
:mod:`tests.parity.frozen`). Both the all-Rust path and the generic path
(:class:`tests.env.generic.GenericAdapter`, the path ``LiveAdapter`` takes) must reproduce
them. ``BalatroVectorEnv`` with one env must step exactly as the Rust Gym env (autoreset
included), and give the same results whatever its thread count.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from jackdaw.core import Game
from jackdaw.core.game import NotPortedError
from jackdaw.engine.actions import GamePhase, PlayHand, SelectBlind
from jackdaw.env import (
    BalatroEnvironment,
    BalatroGymnasiumEnv,
    BalatroVectorEnv,
    DirectAdapter,
    GameAdapter,
    ShapedReward,
    Transition,
    blind_progress_reward,
    sparse_reward,
)
from jackdaw.env.action_space import ActionType
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.rust_gym import MAX_ACTIONS, OBS_LEN
from tests.env.generic import GenericAdapter
from tests.parity import frozen

_RUNS = frozen.load("env_runs.json")
_ADAPTERS = pytest.mark.parametrize(
    "adapter", [DirectAdapter, GenericAdapter], ids=["direct", "generic"]
)


def _same_array(a: np.ndarray, b: np.ndarray, where: str) -> None:
    assert a.dtype == b.dtype, f"{where}: dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{where}: shape {a.shape} vs {b.shape}"
    assert a.tobytes() == b.tobytes(), f"{where}: values differ"


def _same_gym_obs(a: dict[str, np.ndarray], b: dict[str, np.ndarray], where: str) -> None:
    assert list(a) == list(b), where
    for k in a:
        _same_array(a[k], b[k], f"{where} {k}")


# ---------------------------------------------------------------------------
# BalatroEnvironment
# ---------------------------------------------------------------------------


@_ADAPTERS
@pytest.mark.parametrize("case", sorted(_RUNS["environment"]))
def test_balatro_environment_matches_the_oracle(case: str, adapter: type) -> None:
    back, stake, _ = case.split("-")
    kwargs = {"back_keys": [back], "stakes": [int(stake)], "seed_prefix": f"ENVPAR_{back}_{stake}"}
    env = BalatroEnvironment(adapter_factory=adapter, max_steps=300, **kwargs)
    for n, episode in enumerate(_RUNS["environment"][case]):
        chain = frozen.Chain()
        _, _, info = env.reset()
        assert isinstance(info["game"], Game) and "raw_state" not in info
        chain.push("reset", frozen.env_reset_record(info))
        for pos, (t, entity, cards) in enumerate(episode["tape"]):
            fa = FactoredAction(
                action_type=t,
                card_target=None if cards is None else tuple(cards),
                entity_target=entity,
            )
            _, terminated, truncated, _, info = env.step(fa)
            assert "not_ported" not in info, f"{case} ep{n}@{pos}: {info['not_ported']}"
            chain.push("step", frozen.env_step_record(env, terminated, truncated, info))
        frozen.check_chain(episode["chain"], chain, f"{case} episode {n}")


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
    frozen_run = adapter.game.freeze()
    adapter.step(PlayHand(card_indices=(0,)))
    assert frozen_run.state_hash() != adapter.game.state_hash()


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


@_ADAPTERS
@pytest.mark.parametrize("case", sorted(_GYM_CASES))
def test_gym_env_matches_the_oracle(case: str, adapter: type) -> None:
    kwargs = dict(_GYM_CASES[case])
    kwargs.update(back_keys=["b_red"], stakes=[1 + len(case) % 4], seed_prefix=f"GYMPAR_{case}")
    env = BalatroGymnasiumEnv(adapter_factory=adapter, **kwargs)
    episodes = _RUNS["gym"][case]
    assert len(episodes) == 6
    for n, episode in enumerate(episodes):
        chain = frozen.Chain()
        obs, info = env.reset(seed=episode["seed"])  # the last reset is unseeded
        chain.push("reset", frozen.gym_record(env, obs, info))
        for pos, action in enumerate(episode["tape"]):
            obs, reward, terminated, truncated, info = env.step(action)
            assert "balatro/not_ported" not in info, f"{case} ep{n}@{pos}"
            chain.push("step", frozen.gym_record(env, obs, info, reward, terminated, truncated))
        frozen.check_chain(episode["chain"], chain, f"{case} episode {n}")


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


def test_vector_env_checks_its_buffers() -> None:
    """The worker threads write into the buffers: each must be writable, C-contiguous, the
    right size, and apart from the others."""
    vec = BalatroVectorEnv(2, num_threads=2)
    core = vec._core
    resets = [("b_red", 1, f"BUF{i}", None, i) for i in range(2)]
    obs = np.zeros((2, OBS_LEN), dtype=np.float32)
    masks = np.zeros((2, MAX_ACTIONS), dtype=np.uint8)
    core.reset(resets, obs, masks)
    assert obs.any() and masks.any()
    shared = np.zeros(2 * OBS_LEN * 4 + 2 * MAX_ACTIONS, dtype=np.uint8)
    overlapping = shared[: 2 * OBS_LEN * 4].view(np.float32), shared[: 2 * MAX_ACTIONS]
    read_only = masks.copy()
    read_only.flags.writeable = False
    for bad_obs, bad_masks in (
        overlapping,
        (obs, read_only),
        (obs[:, ::2].copy(), masks),  # too small
        (np.zeros((OBS_LEN, 2), dtype=np.float32).T, masks),  # not C-contiguous
    ):
        with pytest.raises((ValueError, BufferError)):
            core.reset(resets, bad_obs, bad_masks)
