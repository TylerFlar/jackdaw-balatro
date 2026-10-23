"""Pieces shared by the MaskablePPO demo scripts: the environments, the training callbacks and
the evaluation loop.

Training runs ``BalatroGymnasiumEnv`` on the Rust engine in Stable-Baselines3's
``DummyVecEnv``.  ``BalatroVectorEnv`` steps faster, but it resets an ended episode on the
*following* step (gymnasium's ``AutoresetMode.NEXT_STEP``), while an SB3 ``VecEnv`` must return
the next episode's first observation, and its action mask, from the very step that ended the
last one.  The vector env can only reset a sub-env by stepping all the others, so wrapping it
would put one transition per episode whose action is ignored into PPO's buffer (see README.md).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor

from jackdaw.core import IllegalActionError as RustIllegalActionError
from jackdaw.engine.game import IllegalActionError as PyIllegalActionError
from jackdaw.env import (
    BalatroGymnasiumEnv,
    DirectAdapter,
    GameAdapter,
    ShapedReward,
    blind_progress_reward,
    sparse_reward,
)
from jackdaw.env.rewards import RewardFn


def make_reward(spec: str) -> RewardFn:
    """A built-in reward (the Rust engine computes all three): ``"sparse"``,
    ``"blind_progress"``, ``"shaped"``, or ``"shaped:loss=0,score_progress=0.1"`` for a
    ``ShapedReward`` with some weights changed."""
    name, _, weights = spec.partition(":")
    if name == "sparse" and not weights:
        return sparse_reward
    if name == "blind_progress" and not weights:
        return blind_progress_reward
    if name == "shaped":
        overrides = {}
        for item in filter(None, weights.split(",")):
            key, _, value = item.partition("=")
            overrides[key.strip()] = float(value)
        return ShapedReward(**overrides)
    raise ValueError(f"unknown reward {spec!r}")


#: Episode cap. A won run takes a few hundred steps; the cap only cuts a policy that stalls.
MAX_STEPS = 2_000

#: Evaluation runs use the seeds EVAL_SEED_BASE + k. Training runs start from seeds
#: ``seed + i`` and then use string seeds ``TRAIN<seed>_<i>_<n>``, so the two never meet.
EVAL_SEED_BASE = 10_000_000


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


class EndOnRefusal(gymnasium.Wrapper):
    """End the episode (truncated) when the engine refuses an action from the table.

    ``BalatroVectorEnv`` does the same (``info["balatro/refused"]``); the single env raises
    ``IllegalActionError`` instead, which would stop a long training run.  It never happened in
    this demo's runs: a refusal would be an env bug, so the wrapper prints it.
    """

    def __init__(self, env: BalatroGymnasiumEnv) -> None:
        super().__init__(env)
        self._last_obs: dict[str, np.ndarray] | None = None
        self.refusals = 0

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except (RustIllegalActionError, PyIllegalActionError) as exc:
            self.refusals += 1
            print(f"[EndOnRefusal] the engine refused table action {action}: {exc}", flush=True)
            env: BalatroGymnasiumEnv = self.env.unwrapped  # type: ignore[assignment]
            assert self._last_obs is not None
            info = {"action_mask": np.zeros_like(env.action_masks()), "balatro/refused": True}
            return self._last_obs, 0.0, False, True, info
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()  # type: ignore[attr-defined]


def make_env(
    rank: int = 0,
    seed: int = 0,
    reward: str = "shaped",
    max_steps: int = MAX_STEPS,
    adapter_factory: Callable[[], GameAdapter] = DirectAdapter,
) -> EndOnRefusal:
    """One Gym env on the Rust engine (``DirectAdapter``) or the Python one (``OracleAdapter``).

    ``reset()`` without a seed starts the run ``f"TRAIN{seed}_{rank}_{n}"`` for the env's n-th
    episode, so sub-envs never replay each other's runs.
    """
    env = BalatroGymnasiumEnv(
        adapter_factory=adapter_factory,
        reward_fn=make_reward(reward),
        max_steps=max_steps,
        seed_prefix=f"TRAIN{seed}_{rank}",
    )
    return EndOnRefusal(env)


def make_training_env(
    n_envs: int, seed: int, reward: str = "shaped", max_steps: int = MAX_STEPS
) -> VecEnv:
    """``n_envs`` envs in a ``DummyVecEnv`` with episode statistics (``VecMonitor``).

    ``MaskablePPO(seed=...)`` seeds it: sub-env ``i``'s first run is seed ``seed + i``.
    ``DummyVecEnv`` resets an env within the step that ends its episode and keeps the final
    observation in ``infos[i]["terminal_observation"]``; ``env_method("action_masks")`` then
    reads each env's mask for the observation the policy sees next.
    """
    factories = [partial(make_env, i, seed, reward, max_steps) for i in range(n_envs)]
    return VecMonitor(DummyVecEnv(factories))


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------


class BalatroMetricsCallback(BaseCallback):
    """Logs the env's episode-end metrics to TensorBoard, averaged over the episodes that
    ended during each rollout: ``balatro/ante_reached``, ``balatro/blinds_beaten``,
    ``balatro/win_rate`` and ``balatro/episodes``."""

    def __init__(self) -> None:
        super().__init__()
        self._antes: list[float] = []
        self._blinds: list[float] = []
        self._wins: list[float] = []
        self.refusals = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if not done:
                continue
            if info.get("balatro/refused"):
                self.refusals += 1
                continue
            self._antes.append(float(info["balatro/ante_reached"]))
            self._blinds.append(float(info["balatro/rounds_beaten"]))
            self._wins.append(float(info["balatro/won"]))
        return True

    def _on_rollout_end(self) -> None:
        if self._antes:
            self.logger.record("balatro/ante_reached", float(np.mean(self._antes)))
            self.logger.record("balatro/blinds_beaten", float(np.mean(self._blinds)))
            self.logger.record("balatro/win_rate", float(np.mean(self._wins)))
        self.logger.record("balatro/episodes", len(self._antes))
        self.logger.record("balatro/refusals", self.refusals)
        self._antes, self._blinds, self._wins = [], [], []


class FixedSeedEvalCallback(BaseCallback):
    """Every ``eval_freq`` timesteps, play the policy on the same ``seeds`` and log the means:
    greedy (the most likely legal action) under ``eval/``, sampled under ``eval_sampled/``.
    Keeps the best model (most blinds beaten by the sampled policy) as ``best_model.zip``."""

    def __init__(
        self,
        seeds: Sequence[int],
        eval_freq: int,
        save_dir: Path,
        reward: str = "shaped",
        max_steps: int = MAX_STEPS,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.seeds = list(seeds)
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.reward = reward
        self.max_steps = max_steps
        self.best = -np.inf
        self._next = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next = self.num_timesteps + self.eval_freq
            self.evaluate()
        return True

    def evaluate(self) -> dict[str, float]:
        summaries = {}
        for prefix, deterministic in (("eval", True), ("eval_sampled", False)):
            start = time.perf_counter()
            results = run_episodes(
                model_chooser(self.model, deterministic=deterministic),
                self.seeds,
                reward=self.reward,
                max_steps=self.max_steps,
            )
            summary = summaries[prefix] = summarize(results)
            for key, value in summary.items():
                self.logger.record(f"{prefix}/{key}", value)
            self.logger.record(f"{prefix}/seconds", time.perf_counter() - start)
            if self.verbose:
                print(f"[{prefix} @ {self.num_timesteps:,}] {format_summary(summary)}", flush=True)
        if summaries["eval_sampled"]["blinds_beaten"] > self.best:
            self.best = summaries["eval_sampled"]["blinds_beaten"]
            self.model.save(self.save_dir / "best_model")
        return summaries["eval_sampled"]


class StopFileCallback(BaseCallback):
    """Ends training early, cleanly (the script still saves the final model), once ``path``
    exists: create a file named ``STOP`` in the run directory."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        if self.n_calls % 256 == 0 and self.path.exists():
            print(f"[stop] found {self.path}: ending training", flush=True)
            return False
        return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

#: ``choose(envs, obs, masks) -> actions``: one action per env, given the batched
#: observations (a dict of stacked arrays) and the ``(n, 500)`` action masks.
Chooser = Callable[[list[EndOnRefusal], dict[str, np.ndarray], np.ndarray], Sequence[int]]


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    ante_reached: int
    blinds_beaten: int
    won: bool
    reward: float
    length: int
    truncated: bool
    refused: bool = False


def run_episodes(
    choose: Chooser,
    seeds: Sequence[int],
    *,
    reward: str = "shaped",
    max_steps: int = MAX_STEPS,
    adapter_factory: Callable[[], GameAdapter] = DirectAdapter,
    batch: int = 64,
) -> list[EpisodeResult]:
    """Play one episode from each seed (``env.reset(seed=s)``), ``batch`` envs at a time, and
    return the results in seed order."""
    results: list[EpisodeResult] = []
    for start in range(0, len(seeds), batch):
        chunk = list(seeds[start : start + batch])
        envs = [
            make_env(reward=reward, max_steps=max_steps, adapter_factory=adapter_factory)
            for _ in chunk
        ]
        obs = [env.reset(seed=s)[0] for env, s in zip(envs, chunk, strict=True)]
        totals = [0.0] * len(chunk)
        lengths = [0] * len(chunk)
        done: dict[int, EpisodeResult] = {}
        active = list(range(len(chunk)))
        while active:
            batch_obs = {key: np.stack([obs[i][key] for i in active]) for key in obs[active[0]]}
            masks = np.stack([envs[i].action_masks() for i in active])
            actions = choose([envs[i] for i in active], batch_obs, masks)
            still: list[int] = []
            for i, action in zip(active, actions, strict=True):
                o, r, terminated, truncated, info = envs[i].step(int(action))
                totals[i] += float(r)
                lengths[i] += 1
                if terminated or truncated:
                    refused = bool(info.get("balatro/refused", False))
                    done[i] = EpisodeResult(
                        seed=chunk[i],
                        ante_reached=0 if refused else int(info["balatro/ante_reached"]),
                        blinds_beaten=0 if refused else int(info["balatro/rounds_beaten"]),
                        won=False if refused else bool(info["balatro/won"]),
                        reward=totals[i],
                        length=lengths[i],
                        truncated=bool(truncated),
                        refused=refused,
                    )
                else:
                    obs[i] = o
                    still.append(i)
            active = still
        results.extend(done[i] for i in range(len(chunk)))
    return results


def summarize(results: Sequence[EpisodeResult]) -> dict[str, float]:
    """Means over the episodes (``win_rate`` and ``truncated`` as fractions)."""
    return {
        "episodes": float(len(results)),
        "ante_reached": float(np.mean([r.ante_reached for r in results])),
        "blinds_beaten": float(np.mean([r.blinds_beaten for r in results])),
        "win_rate": float(np.mean([r.won for r in results])),
        "reward": float(np.mean([r.reward for r in results])),
        "length": float(np.mean([r.length for r in results])),
        "truncated": float(np.mean([r.truncated for r in results])),
        "refused": float(sum(r.refused for r in results)),
    }


def format_summary(s: dict[str, float]) -> str:
    return (
        f"ante {s['ante_reached']:.2f}, blinds {s['blinds_beaten']:.2f}, "
        f"wins {100 * s['win_rate']:.1f}%, reward {s['reward']:+.3f}, length {s['length']:.1f}"
    )


def model_chooser(model: Any, deterministic: bool = True) -> Chooser:
    """The policy of a ``MaskablePPO`` model, masked: greedy with ``deterministic``."""

    def choose(
        envs: list[EndOnRefusal], obs: dict[str, np.ndarray], masks: np.ndarray
    ) -> Sequence[int]:
        actions, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        return [int(a) for a in np.asarray(actions).reshape(-1)]

    return choose


def random_chooser(seed: int) -> Chooser:
    """Uniform over each env's legal actions."""
    rng = np.random.default_rng(seed)

    def choose(
        envs: list[EndOnRefusal], obs: dict[str, np.ndarray], masks: np.ndarray
    ) -> Sequence[int]:
        return [int(rng.choice(np.flatnonzero(m))) for m in masks]

    return choose
