"""The Python half of the PGO training workload (scripts/pgo.sh runs it).

    python -I scripts/pgo_workload.py [--steps N]

Plays random legal actions through the installed ``jackdaw`` package, the way a training
loop drives it: ``BalatroGymnasiumEnv`` (sparse and shaped rewards) and ``BalatroVectorEnv``
(several threads, both autoreset modes). Run against an instrumented wheel, it records how
the extension's PyO3 layer and the vector env's threads are used; the Rust ``rollout``
example covers the engine itself. Nothing here checks results: it only has to run the code.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from gymnasium.vector import AutoresetMode

from jackdaw.env import BalatroGymnasiumEnv, BalatroVectorEnv, DirectAdapter


def gym_env(steps: int, reward_shaping: bool, seed: int) -> None:
    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_shaping=reward_shaping)
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    mask = info["action_mask"]
    for _ in range(steps):
        action = int(rng.choice(np.flatnonzero(mask)))
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            _, info = env.reset()
        mask = info["action_mask"]


def vector_env(steps: int, num_envs: int, threads: int, mode: AutoresetMode, seed: int) -> None:
    envs = BalatroVectorEnv(num_envs, num_threads=threads, autoreset_mode=mode)
    rng = np.random.default_rng(seed)
    _, info = envs.reset(seed=seed)
    for _ in range(steps):
        # One random legal action per sub-env (a sub-env that NEXT_STEP resets ignores it).
        masks = info["action_mask"]
        scores = rng.random(masks.shape) * masks
        _, _, _, _, info = envs.step(scores.argmax(axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100_000, help="steps of the first Gym env run")
    args = parser.parse_args()
    start = time.perf_counter()
    gym_env(args.steps, reward_shaping=False, seed=1)
    gym_env(args.steps // 2, reward_shaping=True, seed=2)
    vector_env(args.steps // 16, 16, 4, AutoresetMode.NEXT_STEP, seed=3)
    vector_env(args.steps // 32, 16, 2, AutoresetMode.SAME_STEP, seed=4)
    print(f"pgo_workload: done in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
