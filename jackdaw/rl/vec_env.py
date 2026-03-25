"""Subprocess-based vectorized environment for parallel rollout collection."""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection
from collections.abc import Callable
from typing import Any

import numpy as np

from jackdaw.env.game_spec import FactoredAction, GameActionMask
from jackdaw.rl.env_wrapper import FactoredBalatroEnv


def _worker(
    pipe: multiprocessing.connection.Connection,
    env_fn: Callable[[], FactoredBalatroEnv],
) -> None:
    """Worker process: run an environment and respond to commands."""
    env = env_fn()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                obs, mask, info = env.reset()
                pipe.send(("reset", obs, mask, info))
            elif cmd == "step":
                obs, reward, terminated, truncated, mask, info = env.step(data)
                done = terminated or truncated
                if done:
                    # Auto-reset: send terminal info + fresh obs
                    terminal_info = {
                        k: v for k, v in info.items() if k.startswith("balatro/")
                    }
                    new_obs, new_mask, _ = env.reset()
                    pipe.send(("step", obs, reward, done, mask, terminal_info, new_obs, new_mask))
                else:
                    pipe.send(("step", obs, reward, done, mask, None, None, None))
            elif cmd == "close":
                break
    finally:
        pipe.close()


class SubprocVecEnv:
    """Vectorized environment using subprocess workers.

    Parameters
    ----------
    env_fns : list of callables
        Each callable creates a FactoredBalatroEnv instance.
    """

    def __init__(self, env_fns: list[Callable[[], FactoredBalatroEnv]]) -> None:
        self.n_envs = len(env_fns)
        ctx = mp.get_context("spawn")
        self._parent_pipes: list[multiprocessing.connection.Connection] = []
        self._procs: list[mp.Process] = []

        for i, fn in enumerate(env_fns):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child_conn, fn), daemon=True)
            proc.start()
            child_conn.close()
            self._parent_pipes.append(parent_conn)
            self._procs.append(proc)

    def reset_all(
        self,
    ) -> tuple[list[dict[str, np.ndarray]], list[GameActionMask]]:
        """Reset all environments. Returns (obs_list, mask_list)."""
        for pipe in self._parent_pipes:
            pipe.send(("reset", None))
        obs_list = []
        mask_list = []
        for pipe in self._parent_pipes:
            _, obs, mask, info = pipe.recv()
            obs_list.append(obs)
            mask_list.append(mask)
        return obs_list, mask_list

    def step(
        self, actions: list[FactoredAction]
    ) -> list[tuple[dict[str, np.ndarray], float, bool, GameActionMask, dict | None, dict[str, np.ndarray] | None, GameActionMask | None]]:
        """Step all environments. Returns list of (obs, reward, done, mask, terminal_info, reset_obs, reset_mask)."""
        for pipe, action in zip(self._parent_pipes, actions):
            pipe.send(("step", action))
        results = []
        for pipe in self._parent_pipes:
            msg = pipe.recv()
            # msg = ("step", obs, reward, done, mask, terminal_info, new_obs, new_mask)
            results.append(msg[1:])  # strip the "step" tag
        return results

    def close(self) -> None:
        """Shut down all workers."""
        for pipe in self._parent_pipes:
            try:
                pipe.send(("close", None))
            except BrokenPipeError:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
