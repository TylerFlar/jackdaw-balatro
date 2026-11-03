"""What the Gym envs hand back, pinned exactly: values, dtypes, shapes, layout and which arrays a
later step may overwrite.

Each scenario plays a seeded random policy and folds everything a reset or step returns
(every observation array, reward, flag, mask and info entry, with its dtype, shape,
contiguity and writability) into a digest frozen from the envs' outputs before their
Python-side bookkeeping was reworked for speed. Every array an env hands out as its own must
stay as it was through later steps.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from jackdaw.env import BalatroGymnasiumEnv, BalatroVectorEnv, DirectAdapter


def _feed(h: Any, value: Any) -> None:
    """Fold ``value`` into ``h``: arrays with their dtype, shape and layout flags; dicts in
    key order; everything else by its type and repr."""
    if isinstance(value, np.ndarray):
        h.update(
            f"array {value.dtype.str} {value.shape} "
            f"{value.flags.c_contiguous} {value.flags.writeable}|".encode()
        )
        if value.dtype == object:
            for item in value.ravel():
                _feed(h, item)
        else:
            h.update(value.tobytes())
    elif isinstance(value, dict):
        h.update(f"dict {len(value)}|".encode())
        for k, v in value.items():
            h.update(f"{k!r}=".encode())
            _feed(h, v)
    elif isinstance(value, tuple | list):
        h.update(f"{type(value).__name__} {len(value)}|".encode())
        for v in value:
            _feed(h, v)
    else:
        h.update(f"{type(value).__name__} {value!r}|".encode())


def _digest(value: Any) -> str:
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()[:16]


def _owned(result: tuple[Any, ...], copy: bool) -> tuple[Any, ...]:
    """The parts of a reset/step result the caller owns: all of it with ``copy``, else all but
    the observation and the action mask (views of buffers the next step overwrites)."""
    if copy:
        return result
    obs, *rest, info = result
    return (*rest, {k: v for k, v in info.items() if k != "action_mask"})


def _pick(rng: np.random.Generator, mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal)) if legal.size else 0


class _Recorder:
    """The digests of a run's results, and what the caller owns of each with its digest then."""

    def __init__(self, copy: bool) -> None:
        self.copy = copy
        self.digests: list[str] = []
        self.owned: list[tuple[str, Any]] = []

    def record(self, result: tuple[Any, ...]) -> None:
        self.digests.append(_digest(result))
        owned = _owned(result, self.copy)
        self.owned.append((_digest(owned), owned))

    def check_owned(self) -> None:
        """Every array handed out as the caller's is as it was: later steps left it alone."""
        for pos, (digest, owned) in enumerate(self.owned):
            assert _digest(owned) == digest, f"result {pos} changed after it was returned"

    @property
    def digest(self) -> str:
        return hashlib.sha256("".join(self.digests).encode()).hexdigest()[:16]


def _gym_run(copy: bool, steps: int = 600, **kwargs: Any) -> _Recorder:
    env = BalatroGymnasiumEnv(DirectAdapter, seed_prefix="OUT", max_steps=60, **kwargs)
    rng = np.random.default_rng(5)
    run = _Recorder(copy)

    result: tuple[Any, ...] = env.reset(seed=41)
    run.record(result)
    for _ in range(steps):
        result = env.step(_pick(rng, result[-1]["action_mask"]))
        run.record(result)
        if result[2] or result[3]:
            result = env.reset()
            run.record(result)
    return run


def _vector_run(copy: bool, mode: AutoresetMode, steps: int = 150) -> _Recorder:
    vec = BalatroVectorEnv(
        5,
        seed_prefix="OUT",
        max_steps=40,
        reward_shaping=True,
        num_threads=2,
        copy=copy,
        autoreset_mode=mode,
    )
    rng = np.random.default_rng(9)
    run = _Recorder(copy)

    result: tuple[Any, ...] = vec.reset(seed=300)
    run.record(result)
    for pos in range(steps):
        # An env whose episode just ended (NEXT_STEP) has no legal action: it resets.
        actions = [_pick(rng, m) for m in result[-1]["action_mask"]]
        result = vec.step(actions)
        run.record(result)
        if pos == steps // 2:
            mask = [True, False, True, False, False]
            result = vec.reset(seed=[7, None, None, None, None], options={"reset_mask": mask})
            run.record(result)
    return run


#: The runs' digests, frozen before the Python-side bookkeeping was reworked.
_GYM = "fa73e209deeecac9"
#: (``copy=False`` hands out the same values: views of buffers the next step overwrites.)
_VECTOR = {
    AutoresetMode.NEXT_STEP: "77002dd2640540f9",
    AutoresetMode.SAME_STEP: "c257fdfba478b4ed",
}


def test_gym_env_outputs_are_unchanged() -> None:
    run = _gym_run(copy=True)
    assert run.digest == _GYM
    run.check_owned()


@pytest.mark.parametrize("copy", [True, False], ids=["copy", "views"])
@pytest.mark.parametrize(
    "mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP], ids=["next_step", "same_step"]
)
def test_vector_env_outputs_are_unchanged(copy: bool, mode: AutoresetMode) -> None:
    run = _vector_run(copy, mode)
    assert run.digest == _VECTOR[mode]
    run.check_owned()
