"""Pieces shared by the MaskablePPO demo scripts: the environments, the training callbacks and
the evaluation loop.

Training runs ``BalatroVectorEnv`` with same-step autoreset (``AutoresetMode.SAME_STEP``) behind
:class:`BalatroSB3VecEnv`, a Stable-Baselines3 ``VecEnv``: the step that ends an episode returns
the next one's first observation and mask, and the last observation as SB3's
``terminal_observation``.  Evaluation plays ``BalatroGymnasiumEnv``s, one per seed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.vector import AutoresetMode
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from jackdaw.env import (
    BalatroGymnasiumEnv,
    BalatroVectorEnv,
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


def make_env(
    rank: int = 0,
    seed: int = 0,
    reward: str = "shaped",
    max_steps: int = MAX_STEPS,
    adapter_factory: Callable[[], GameAdapter] = DirectAdapter,
) -> BalatroGymnasiumEnv:
    """One Gym env, on the Rust engine in process by default (``DirectAdapter``).

    ``reset()`` without a seed starts the run ``f"TRAIN{seed}_{rank}_{n}"`` for the env's n-th
    episode, so sub-envs never replay each other's runs.  A table action the engine refuses (an
    env bug) ends the episode, flagged in ``info["balatro/refused"]``.
    """
    return BalatroGymnasiumEnv(
        adapter_factory=adapter_factory,
        reward_fn=make_reward(reward),
        max_steps=max_steps,
        seed_prefix=f"TRAIN{seed}_{rank}",
    )


class BalatroSB3VecEnv(VecEnv):
    """``BalatroVectorEnv`` (same-step autoreset) as a Stable-Baselines3 ``VecEnv``.

    Sub-env ``i`` plays the runs ``BalatroGymnasiumEnv(seed_prefix=f"TRAIN{seed}_{i}")``
    would: SB3's ``seed(s)`` makes its first run seed ``s + i``.  An ended episode's last
    observation is ``infos[i]["terminal_observation"]`` (SB3 bootstraps truncated episodes
    from it), next to the episode's stats; ``env_method("action_masks")`` gives each env's mask
    for the observation the policy sees next, as ``MaskablePPO`` reads it.
    """

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        reward: str = "shaped",
        max_steps: int = MAX_STEPS,
        num_threads: int = 1,
    ) -> None:
        self.venv = BalatroVectorEnv(
            num_envs,
            reward_fn=make_reward(reward),
            max_steps=max_steps,
            seed_prefix=f"TRAIN{seed}",
            num_threads=num_threads,
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        super().__init__(
            num_envs, self.venv.single_observation_space, self.venv.single_action_space
        )
        self._actions: np.ndarray | None = None

    def reset(self) -> dict[str, np.ndarray]:
        seeds = None if self._seeds[0] is None else list(self._seeds)
        obs, _ = self.venv.reset(seed=seeds)
        self._reset_seeds()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        obs, rewards, terminated, truncated, info = self.venv.step(self._actions)
        dones = terminated | truncated
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if dones.any():
            final = info["final_info"]
            for i in np.flatnonzero(dones):
                infos[i] = {
                    "terminal_observation": info["final_obs"][i],
                    "TimeLimit.truncated": bool(truncated[i] and not terminated[i]),
                    "balatro/ante_reached": int(final["balatro/ante_reached"][i]),
                    "balatro/rounds_beaten": int(final["balatro/rounds_beaten"][i]),
                    "balatro/won": bool(final["balatro/won"][i]),
                }
                if "balatro/refused" in final and final["balatro/refused"][i]:
                    infos[i]["balatro/refused"] = True
        return obs, rewards.astype(np.float32), dones, infos

    def action_masks(self) -> np.ndarray:
        return self.venv.action_masks()

    def env_method(
        self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        if method_name != "action_masks":
            raise NotImplementedError(f"BalatroSB3VecEnv.env_method({method_name!r})")
        masks = self.venv.action_masks()
        return [masks[i] for i in self._get_indices(indices)]

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        if attr_name == "action_masks":
            return [self.action_masks for _ in self._get_indices(indices)]
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        raise AttributeError(f"BalatroSB3VecEnv has no sub-env attribute {attr_name!r}")

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        raise AttributeError(f"BalatroSB3VecEnv can't set {attr_name!r}")

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def close(self) -> None:
        self.venv.close()


def make_training_env(
    n_envs: int,
    seed: int,
    reward: str = "shaped",
    max_steps: int = MAX_STEPS,
    num_threads: int = 1,
) -> VecEnv:
    """``n_envs`` envs stepped together in Rust, with episode statistics (``VecMonitor``).
    ``MaskablePPO(seed=...)`` seeds it: sub-env ``i``'s first run is seed ``seed + i``."""
    return VecMonitor(BalatroSB3VecEnv(n_envs, seed, reward, max_steps, num_threads))


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
Chooser = Callable[[list[BalatroGymnasiumEnv], dict[str, np.ndarray], np.ndarray], Sequence[int]]


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
        envs: list[BalatroGymnasiumEnv], obs: dict[str, np.ndarray], masks: np.ndarray
    ) -> Sequence[int]:
        actions, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        return [int(a) for a in np.asarray(actions).reshape(-1)]

    return choose


def random_chooser(seed: int) -> Chooser:
    """Uniform over each env's legal actions."""
    rng = np.random.default_rng(seed)

    def choose(
        envs: list[BalatroGymnasiumEnv], obs: dict[str, np.ndarray], masks: np.ndarray
    ) -> Sequence[int]:
        return [int(rng.choice(np.flatnonzero(m))) for m in masks]

    return choose
