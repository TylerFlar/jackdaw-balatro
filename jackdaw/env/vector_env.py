"""BalatroVectorEnv: many Balatro Gym environments stepped in parallel in Rust.

``num_envs`` copies of :class:`~jackdaw.env.gymnasium_wrapper.BalatroGymnasiumEnv`
(on the Rust engine) run in ``jackdaw._core.VectorEnv``: one ``step`` steps
them all on worker threads with the GIL released, writing observations,
action masks and per-env results into numpy arrays this object owns.

Sub-env ``i`` behaves exactly as ``BalatroGymnasiumEnv(DirectAdapter,
seed_prefix=f"{seed_prefix}_{i}", ...)``: the same runs, tables, rewards and
episode stats.  ``reset(seed=s)`` starts sub-env ``i`` from seed ``s + i``
(gymnasium's convention).  Episodes reset on the step after they end
(``AutoresetMode.NEXT_STEP``): that step ignores the env's action and returns
the new episode's first observation with reward 0.

Rewards are the built-in ones, computed in Rust: :func:`~jackdaw.env.rewards.sparse_reward`
(default), :func:`~jackdaw.env.rewards.blind_progress_reward` or a
:class:`~jackdaw.env.rewards.ShapedReward`.  Results don't depend on the
number of threads.

Example::

    from jackdaw.env import BalatroVectorEnv

    envs = BalatroVectorEnv(num_envs=64, num_threads=8)
    obs, info = envs.reset(seed=0)
    masks = envs.action_masks()        # (64, 500) bool
    obs, rewards, terminated, truncated, info = envs.step(actions)
"""

from __future__ import annotations

import secrets
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from jackdaw import _core
from jackdaw.core import Game
from jackdaw.env.game_interface import GameState, game_state_from_parts
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.rewards import RewardFn, ShapedReward, sparse_reward
from jackdaw.env.rust_gym import (
    ENTITY_INFO,
    GLOBAL_DIM,
    MAX_ACTIONS,
    OBS_LEN,
    EpisodeStarts,
    batched_gym_obs,
    check_reward,
    factored_from_parts,
    sampler_seed,
)

# Columns of the per-env results jackdaw._core.VectorEnv writes.
_REWARD, _TERMINATED, _TRUNCATED, _ENDED, _ANTE, _ROUNDS, _WON, _NOT_PORTED, _REFUSED = range(9)


def single_observation_space() -> spaces.Dict:
    """The observation space of one env (``BalatroGymnasiumEnv.observation_space``)."""
    obs: dict[str, spaces.Space] = {
        "global": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_DIM,), dtype=np.float32)
    }
    for name, max_count, dim in ENTITY_INFO:
        obs[name] = spaces.Box(-np.inf, np.inf, shape=(max_count, dim), dtype=np.float32)
    obs["entity_counts"] = spaces.Box(
        low=0,
        high=np.array([m for _, m, _ in ENTITY_INFO], dtype=np.float32),
        shape=(len(ENTITY_INFO),),
        dtype=np.float32,
    )
    return spaces.Dict(obs)


class BalatroVectorEnv(VectorEnv):
    """``num_envs`` Balatro environments stepped in parallel in Rust.

    Parameters
    ----------
    num_envs:
        Number of sub-environments.
    back_keys, stakes, max_steps, seed_prefix, end_on_win:
        As for :class:`~jackdaw.env.gymnasium_wrapper.BalatroGymnasiumEnv`; sub-env ``i``
        uses the seed prefix ``f"{seed_prefix}_{i}"``.
    reward_shaping, reward_fn:
        As for the Gym env, limited to the built-in rewards.
    num_threads:
        Worker threads (0: one per core).
    copy:
        Return copies of the observation arrays (default), or views of the buffers the
        next step overwrites.

    A sub-env step the engine refuses (a table action that turns out illegal, or one it
    can't take yet) ends that episode as truncated; ``info["balatro/refused"]`` and
    ``info["balatro/not_ported"]`` flag them.
    """

    metadata: dict[str, Any] = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        num_envs: int,
        back_keys: list[str] | None = None,
        stakes: list[int] | None = None,
        max_steps: int = 10_000,
        seed_prefix: str = "TRAIN",
        reward_shaping: bool = False,
        reward_fn: RewardFn | None = None,
        end_on_win: bool = True,
        num_threads: int = 0,
        copy: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if reward_fn is not None and reward_shaping:
            raise ValueError("Pass either reward_fn or reward_shaping=True, not both")
        if reward_fn is None:
            reward_fn = ShapedReward() if reward_shaping else sparse_reward
        kind, weights = check_reward(reward_fn)
        self.reward_fn = reward_fn
        self.num_envs = num_envs
        self.copy = copy

        self.single_observation_space = single_observation_space()
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Discrete(MAX_ACTIONS)
        self.action_space = spaces.MultiDiscrete(np.full(num_envs, MAX_ACTIONS))

        self._starts = [
            EpisodeStarts(back_keys, stakes, f"{seed_prefix}_{i}") for i in range(num_envs)
        ]
        first = self._starts[0]
        inits = [
            (first.back_keys[0], first.stakes[0], "INIT", None, secrets.randbits(64))
            for _ in range(num_envs)
        ]
        self._core = _core.VectorEnv(inits, max_steps, end_on_win, kind, weights, num_threads)
        self._obs = np.zeros((num_envs, OBS_LEN), dtype=np.float32)
        self._masks = np.zeros((num_envs, MAX_ACTIONS), dtype=np.bool_)
        self._stats = np.zeros((num_envs, 9), dtype=np.float64)
        self._obs_views = batched_gym_obs(self._obs)
        self._autoreset = np.zeros(num_envs, dtype=np.bool_)
        self._reset_once = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start a new episode in every sub-env; ``seed`` an int (sub-env ``i`` gets
        ``seed + i``), a list (one per sub-env) or None."""
        if seed is None:
            seeds: list[int | None] = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        resets = [self._reset_tuple(i, s) for i, s in enumerate(seeds)]
        self._core.reset(resets, self._obs, self._masks.view(np.uint8))
        self._autoreset[:] = False
        self._reset_once = True
        return self._observations(), {"action_mask": self._masks.copy()}

    def step(
        self, actions: Any
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step every sub-env with its action (an index into its action table); a sub-env
        whose episode ended on the last step starts a new one instead."""
        if not self._reset_once:
            raise RuntimeError("BalatroVectorEnv: call reset() before step()")
        actions = np.ascontiguousarray(actions, dtype=np.int64).reshape(self.num_envs)
        resets = [
            self._reset_tuple(i, None) if self._autoreset[i] else None for i in range(self.num_envs)
        ]
        self._core.step(actions, resets, self._obs, self._masks.view(np.uint8), self._stats)
        stats = self._stats
        rewards = stats[:, _REWARD].copy()
        terminated = stats[:, _TERMINATED] != 0
        truncated = stats[:, _TRUNCATED] != 0
        ended = stats[:, _ENDED] != 0
        self._autoreset = ended.copy()
        info: dict[str, Any] = {"action_mask": self._masks.copy()}
        if ended.any():
            info["balatro/ante_reached"] = stats[:, _ANTE].astype(np.int64)
            info["_balatro/ante_reached"] = ended
            info["balatro/rounds_beaten"] = stats[:, _ROUNDS].astype(np.int64)
            info["_balatro/rounds_beaten"] = ended
            info["balatro/won"] = stats[:, _WON] != 0
            info["_balatro/won"] = ended
            for key, col in (("balatro/not_ported", _NOT_PORTED), ("balatro/refused", _REFUSED)):
                flags = stats[:, col] != 0
                if flags.any():
                    info[key] = flags
                    info["_" + key] = flags
        return self._observations(), rewards, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """``(num_envs, 500)`` bool: each sub-env's legal actions (for MaskablePPO)."""
        return self._masks.copy()

    def close_extras(self, **kwargs: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def game(self, i: int) -> Game:
        """A copy of sub-env ``i``'s run."""
        return Game(self._core.game(i))

    def state(self, i: int) -> GameState:
        """Sub-env ``i``'s current :class:`~jackdaw.env.game_interface.GameState`."""
        return game_state_from_parts(self._core.state(i))

    def action_table(self, i: int) -> list[FactoredAction]:
        """Sub-env ``i``'s action table (what each action index means)."""
        return [factored_from_parts(p) for p in self._core.action_table(i)]

    @property
    def num_threads(self) -> int:
        return self._core.threads

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_tuple(self, i: int, seed: int | None) -> tuple[Any, ...]:
        back, stake, run_seed = self._starts[i].next(None if seed is None else str(seed))
        rng_seed = None if seed is None else sampler_seed(seed)
        return (back, stake, run_seed, None, rng_seed)

    def _observations(self) -> dict[str, np.ndarray]:
        if self.copy:
            return {k: v.copy() for k, v in self._obs_views.items()}
        return self._obs_views
