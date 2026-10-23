"""Glue between the Rust Gym environments (``jackdaw._core.GymEnv`` /
``VectorEnv``) and the Python wrappers: the flat observation layout, the
action table, built-in rewards and how episodes start.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any, overload

import numpy as np

from jackdaw import _core
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.rewards import RewardFn, ShapedReward, blind_progress_reward, sparse_reward

_SPEC = balatro_game_spec()
#: ``(name, max_count, feature_dim)`` of each entity type, in observation order.
ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]
GLOBAL_DIM: int = _SPEC.global_feature_dim
MAX_ACTIONS: int = _core.MAX_ACTIONS
OBS_LEN: int = _core.GYM_OBS_LEN

if list(_core._gym_max_counts()) != [m for _, m, _ in ENTITY_INFO] or OBS_LEN != (
    GLOBAL_DIM + sum(m * d for _, m, d in ENTITY_INFO) + len(ENTITY_INFO)
):
    raise ImportError("jackdaw._core's Gym observation layout differs from balatro_game_spec()")

_F32 = np.dtype(np.float32)
_MASK64 = (1 << 64) - 1

# (key, shape, byte offset) of each array in the flattened observation.
_LAYOUT: list[tuple[str, tuple[int, ...], int]] = []
_offset = GLOBAL_DIM
_LAYOUT.append(("global", (GLOBAL_DIM,), 0))
for _name, _count, _dim in ENTITY_INFO:
    _LAYOUT.append((_name, (_count, _dim), _offset * 4))
    _offset += _count * _dim
_LAYOUT.append(("entity_counts", (len(ENTITY_INFO),), _offset * 4))


def gym_obs(data: bytearray | memoryview) -> dict[str, np.ndarray]:
    """The Gym observation dict viewing one flattened observation (``OBS_LEN`` float32)."""
    return {key: np.ndarray(shape, _F32, data, off) for key, shape, off in _LAYOUT}


def batched_gym_obs(flat: np.ndarray) -> dict[str, np.ndarray]:
    """The batched observation dict viewing ``flat`` (``num_envs x OBS_LEN`` float32)."""
    n = flat.shape[0]
    out: dict[str, np.ndarray] = {}
    for key, shape, off in _LAYOUT:
        start = off // 4
        size = int(np.prod(shape))
        out[key] = flat[:, start : start + size].reshape((n, *shape))
    return out


def factored_from_parts(parts: tuple[int, int | None, list[int] | None]) -> FactoredAction:
    """A table entry as the Python table holds it (``action_type`` a plain int)."""
    t, entity, cards = parts
    return FactoredAction(
        action_type=t,
        card_target=tuple(cards) if cards is not None else None,
        entity_target=entity,
    )


class RustActionTable(Sequence[FactoredAction]):
    """The live action table of a Rust Gym env, read lazily (it changes every step)."""

    __slots__ = ("_core",)

    def __init__(self, core: Any) -> None:
        self._core = core

    def __len__(self) -> int:
        return self._core.n_actions

    @overload
    def __getitem__(self, i: int) -> FactoredAction: ...

    @overload
    def __getitem__(self, i: slice) -> list[FactoredAction]: ...

    def __getitem__(self, i: int | slice) -> FactoredAction | list[FactoredAction]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"action table index {i} out of range ({n} actions)")
        return factored_from_parts(self._core.action(i))

    def __iter__(self) -> Iterator[FactoredAction]:
        return (factored_from_parts(p) for p in self._core.action_table())

    def __repr__(self) -> str:
        return f"RustActionTable({len(self)} actions)"


def rust_reward(fn: Any) -> tuple[str, tuple[float, ...] | None] | None:
    """The built-in reward ``fn`` is, as ``(kind, weights)`` for ``jackdaw._core``, or None
    for a custom one (computed in Python)."""
    if fn is sparse_reward:
        return "sparse", None
    if fn is blind_progress_reward:
        return "blind_progress", None
    if type(fn) is ShapedReward:
        return "shaped", (
            float(fn.step_cost),
            float(fn.shop_step_cost),
            float(fn.blind_beaten),
            float(fn.boss_beaten),
            float(fn.hands_left),
            float(fn.score_progress),
            float(fn.win),
            float(fn.loss),
        )
    return None


class EpisodeStarts:
    """How :class:`~jackdaw.env.balatro_env.BalatroEnvironment` starts each episode: the seed
    (given, else ``f"{seed_prefix}_{n}"`` for the n-th episode), then a back and a stake drawn
    with :func:`random.choice` when there are several."""

    __slots__ = ("back_keys", "stakes", "seed_prefix", "episode_count")

    def __init__(
        self, back_keys: list[str] | None, stakes: list[int] | None, seed_prefix: str
    ) -> None:
        self.back_keys = back_keys or ["b_red"]
        self.stakes = stakes or [1]
        self.seed_prefix = seed_prefix
        self.episode_count = 0

    def next(self, seed: str | None = None) -> tuple[str, int, str]:
        """``(back, stake, run_seed)`` for the next episode."""
        run_seed = seed if seed is not None else f"{self.seed_prefix}_{self.episode_count}"
        self.episode_count += 1
        back = random.choice(self.back_keys) if len(self.back_keys) > 1 else self.back_keys[0]
        stake = random.choice(self.stakes) if len(self.stakes) > 1 else self.stakes[0]
        return back, stake, run_seed


def sampler_seed(seed: int) -> int:
    """A Gym seed as the action-table sampler's (``Xoshiro256(seed)`` takes it modulo 2**64)."""
    return int(seed) & _MASK64


def check_reward(fn: RewardFn) -> tuple[str, tuple[float, ...] | None]:
    """The built-in reward for a vector env (which computes rewards in Rust only)."""
    kind = rust_reward(fn)
    if kind is None:
        raise TypeError(
            "BalatroVectorEnv computes rewards in Rust: use sparse_reward, "
            "blind_progress_reward or a ShapedReward"
        )
    return kind
