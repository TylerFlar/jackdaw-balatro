"""BalatroVectorEnv: many Balatro Gym environments stepped in parallel in Rust.

``num_envs`` copies of :class:`~jackdaw.env.gymnasium_wrapper.BalatroGymnasiumEnv`
(on the Rust engine) run in ``jackdaw._core.VectorEnv``: one ``step`` steps
them all on worker threads with the GIL released, writing observations,
action masks and per-env results into numpy arrays this object owns.

Sub-env ``i`` behaves exactly as ``BalatroGymnasiumEnv(DirectAdapter,
seed_prefix=f"{seed_prefix}_{i}", ...)``: the same runs, tables, rewards and
episode stats.  ``reset(seed=s)`` starts sub-env ``i`` from seed ``s + i``
(gymnasium's convention); ``reset(options={"reset_mask": mask})`` resets only
the sub-envs ``mask`` selects.

Two autoreset modes, as in gymnasium:

- ``AutoresetMode.NEXT_STEP`` (default): an episode resets on the step after it
  ends.  That step ignores the env's action and returns the new episode's first
  observation with reward 0.
- ``AutoresetMode.SAME_STEP``: the step that ends an episode resets it and
  returns the new episode's first observation and mask; the last observation is
  in ``info["final_obs"]`` and the step's episode stats in ``info["final_info"]``.
  This is what Stable-Baselines3's ``VecEnv`` expects.

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
from jackdaw.core import Game, IllegalActionError
from jackdaw.env.action_space import ACTION_VERSION, DEFAULT_MAX_FREE_MOVES
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
    autoreset_mode:
        ``AutoresetMode.NEXT_STEP`` (default) or ``AutoresetMode.SAME_STEP``.
    max_free_moves:
        As for the Gym env: after this many free moves in a row (swaps, sorts) a sub-env's
        table holds none until another action.  ``None``: no cap.
    on_refusal:
        ``"truncate"`` (default): a table action the engine refuses ends that episode as
        truncated, flagged in ``info["balatro/refused"]``.  ``"raise"``: the step raises
        ``IllegalActionError`` after it (the refused episodes were cut).  A sub-env the
        engine can't step yet is cut too, flagged in ``info["balatro/not_ported"]``.

    :attr:`action_version` is the action space's
    :data:`~jackdaw.env.action_space.ACTION_VERSION`.
    """

    metadata: dict[str, Any] = {
        "autoreset_mode": AutoresetMode.NEXT_STEP,
        "action_version": ACTION_VERSION,
    }
    action_version: int = ACTION_VERSION

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
        autoreset_mode: AutoresetMode = AutoresetMode.NEXT_STEP,
        max_free_moves: int | None = DEFAULT_MAX_FREE_MOVES,
        on_refusal: str = "truncate",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        autoreset_mode = AutoresetMode(autoreset_mode)
        if autoreset_mode not in (AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP):
            raise ValueError(f"autoreset_mode {autoreset_mode} is not NEXT_STEP or SAME_STEP")
        if on_refusal not in ("truncate", "raise"):
            raise ValueError(f"on_refusal must be 'truncate' or 'raise', not {on_refusal!r}")
        if max_free_moves is not None and max_free_moves < 0:
            raise ValueError("max_free_moves must be None or at least 0")
        self.autoreset_mode = autoreset_mode
        self.metadata = {**type(self).metadata, "autoreset_mode": autoreset_mode}
        self.max_free_moves = max_free_moves
        self.on_refusal = on_refusal
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
        self._core = _core.VectorEnv(
            inits, max_steps, end_on_win, kind, weights, num_threads, max_free_moves
        )
        # Each step writes fresh observation and mask arrays (copy=True) or reuses these.
        self._obs = np.zeros((num_envs, OBS_LEN), dtype=np.float32)
        self._masks = np.zeros((num_envs, MAX_ACTIONS), dtype=np.bool_)
        self._stats = np.zeros((num_envs, 9), dtype=np.float64)
        self._obs_views = batched_gym_obs(self._obs)
        # Sub-envs whose episode ended on the last step: the next step resets them.
        self._pending: list[int] = []
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
        ``seed + i``), a list (one per sub-env) or None.  With ``options={"reset_mask":
        mask}`` only the sub-envs ``mask`` (``num_envs`` bools) selects start one; the others
        keep their episodes and observations."""
        if seed is None:
            seeds: list[int | None] = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        reset_mask = (options or {}).get("reset_mask")
        if reset_mask is not None:
            reset_mask = np.asarray(reset_mask, dtype=np.bool_).reshape(-1)
            if reset_mask.shape != (self.num_envs,):
                raise ValueError(f"reset_mask must hold {self.num_envs} bools")
            if not self._reset_once:
                raise RuntimeError("BalatroVectorEnv: a full reset() comes first")
            chosen = np.flatnonzero(reset_mask).tolist()
            obs, masks = self._outputs(keep=True)
            resets = [(i, self._reset_tuple(i, seeds[i])) for i in chosen]
            self._core.reset_some(resets, obs, masks.view(np.uint8))
            self._pending = [i for i in self._pending if not reset_mask[i]]
            return self._observations(obs), {"action_mask": masks}
        resets = [self._reset_tuple(i, s) for i, s in enumerate(seeds)]
        obs, masks = self._outputs()
        self._core.reset(resets, obs, masks.view(np.uint8))
        self._pending = []
        self._reset_once = True
        return self._observations(obs), {"action_mask": masks}

    def step(
        self, actions: Any
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step every sub-env with its action (an index into its action table).  With
        ``NEXT_STEP``, a sub-env whose episode ended on the last step starts a new one
        instead (its action is ignored: pass anything); with ``SAME_STEP``, a sub-env whose
        episode ends now starts the next one before the step returns."""
        if not self._reset_once:
            raise RuntimeError("BalatroVectorEnv: call reset() before step()")
        actions = np.ascontiguousarray(actions, dtype=np.int64).reshape(self.num_envs)
        resets = [(i, self._reset_tuple(i, None)) for i in self._pending]
        obs, masks = self._outputs()
        self._core.step(actions, resets, obs, masks.view(np.uint8), self._stats)
        stats = self._stats
        rewards = stats[:, _REWARD].copy()
        terminated = stats[:, _TERMINATED] != 0
        truncated = stats[:, _TRUNCATED] != 0
        ended = stats[:, _ENDED] != 0
        info: dict[str, Any] = {"action_mask": masks}
        if ended.any():
            episode = self._episode_info(stats, ended)
            if self.autoreset_mode is AutoresetMode.SAME_STEP:
                ended_envs = np.flatnonzero(ended).tolist()
                final = batched_gym_obs(obs[ended_envs].copy())
                final_obs = np.full(self.num_envs, None, dtype=object)
                for row, i in enumerate(ended_envs):
                    final_obs[i] = {key: value[row] for key, value in final.items()}
                info["final_obs"] = final_obs
                info["_final_obs"] = ended
                info["final_info"] = episode
                info["_final_info"] = ended
                resets = [(i, self._reset_tuple(i, None)) for i in ended_envs]
                self._core.reset_some(resets, obs, masks.view(np.uint8))
                self._pending = []
            else:
                info.update(episode)
                self._pending = np.flatnonzero(ended).tolist()
        else:
            self._pending = []
        if self.on_refusal == "raise" and stats[:, _REFUSED].any():
            refused = np.flatnonzero(stats[:, _REFUSED]).tolist()
            raise IllegalActionError(
                f"BalatroVectorEnv: the engine refused the action of sub-envs {refused}"
            )
        return self._observations(obs), rewards, terminated, truncated, info

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

    def _outputs(self, keep: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """The arrays the next step writes: fresh ones (copy=True, so what was handed out stays
        as it was), holding the last values with ``keep``, else the env's own."""
        if self.copy:
            if keep:
                self._obs = self._obs.copy()
                self._masks = self._masks.copy()
            else:
                self._obs = np.empty((self.num_envs, OBS_LEN), dtype=np.float32)
                self._masks = np.empty((self.num_envs, MAX_ACTIONS), dtype=np.bool_)
        return self._obs, self._masks

    @staticmethod
    def _episode_info(stats: np.ndarray, ended: np.ndarray) -> dict[str, Any]:
        """The ended episodes' stats, in gymnasium's vector info form (a value array and a
        ``_``-prefixed mask per key)."""
        info: dict[str, Any] = {
            "balatro/ante_reached": stats[:, _ANTE].astype(np.int64),
            "_balatro/ante_reached": ended,
            "balatro/rounds_beaten": stats[:, _ROUNDS].astype(np.int64),
            "_balatro/rounds_beaten": ended,
            "balatro/won": stats[:, _WON] != 0,
            "_balatro/won": ended,
        }
        for key, col in (("balatro/not_ported", _NOT_PORTED), ("balatro/refused", _REFUSED)):
            flags = stats[:, col] != 0
            if flags.any():
                info[key] = flags
                info["_" + key] = flags
        return info

    def _observations(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        if self.copy:
            return batched_gym_obs(obs)
        return self._obs_views
