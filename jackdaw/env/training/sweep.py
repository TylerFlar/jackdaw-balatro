"""Hyperparameter sweep for PPO training.

Random search over a configurable search space with JSON result logging.
No external dependencies beyond torch (already required for training).

Usage:
    uv run python -m jackdaw.env.training.sweep
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.training.ppo import PPOConfig, TrainResult, train_ppo

# ---------------------------------------------------------------------------
# Search space defaults
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: dict[str, Any] = {
    "learning_rate": (1e-5, 1e-3, "log"),
    "ent_coef": (0.001, 0.1, "log"),
    "gae_lambda": [0.9, 0.95, 0.98],
    "num_steps": [64, 128, 256, 512],
    "num_minibatches": [2, 4, 8],
    "update_epochs": [2, 4, 8],
    "gamma": [0.95, 0.99, 0.995],
    "embed_dim": [64, 128, 256],
    "num_layers": [2, 3, 4],
    "num_envs": [4, 8, 16],
    "clip_coef": [0.1, 0.2, 0.3],
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Hyperparameter sweep configuration."""

    search_space: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SEARCH_SPACE))
    n_trials: int = 20
    timesteps_per_trial: int = 200_000
    eval_episodes: int = 50
    metric: str = "avg_ante"
    seed: int = 42
    max_parallel: int = 1
    output_dir: str = "sweep_results"

    # Fixed PPO params (not searched)
    back_keys: list[str] | str = "b_red"
    stake: list[int] | int = 1
    device: str = "auto"


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class RandomSampler:
    """Generates random hyperparameter configurations from a search space."""

    def __init__(self, search_space: dict[str, Any], seed: int = 42) -> None:
        self.search_space = search_space
        self.rng = random.Random(seed)

    def sample(self) -> dict[str, Any]:
        """Sample one configuration from the search space."""
        config: dict[str, Any] = {}
        for name, spec in self.search_space.items():
            if isinstance(spec, list):
                config[name] = self.rng.choice(spec)
            elif isinstance(spec, tuple):
                lo, hi, *flags = spec
                if "log" in flags:
                    log_lo, log_hi = math.log(lo), math.log(hi)
                    config[name] = math.exp(self.rng.uniform(log_lo, log_hi))
                else:
                    config[name] = self.rng.uniform(lo, hi)
            else:
                config[name] = spec
        return config


def _validate_and_fix(params: dict[str, Any]) -> dict[str, Any]:
    """Validate sampled params and fix constraint violations.

    Ensures num_steps * num_envs is divisible by num_minibatches.
    Ensures num_heads divides embed_dim.
    """
    params = dict(params)

    num_steps = params.get("num_steps", 128)
    num_envs = params.get("num_envs", 8)
    num_minibatches = params.get("num_minibatches", 4)

    total = num_steps * num_envs
    if total % num_minibatches != 0:
        # Pick the largest valid num_minibatches <= sampled value
        for mb in sorted([2, 4, 8], reverse=True):
            if mb <= num_minibatches and total % mb == 0:
                params["num_minibatches"] = mb
                break
        else:
            params["num_minibatches"] = 1

    # Ensure num_heads divides embed_dim
    embed_dim = params.get("embed_dim", 128)
    # Use 4 heads by default; if embed_dim < 4, use 1 or 2
    if embed_dim >= 8:
        params["num_heads"] = 4
    elif embed_dim >= 4:
        params["num_heads"] = 2
    else:
        params["num_heads"] = 1

    return params


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of a single sweep trial."""

    trial_id: int
    params: dict[str, Any]
    avg_ante: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    avg_length: float = 0.0
    total_episodes: int = 0
    wall_time: float = 0.0
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "avg_ante": self.avg_ante,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "avg_length": self.avg_length,
            "total_episodes": self.total_episodes,
            "wall_time": self.wall_time,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def _make_ppo_config(
    params: dict[str, Any],
    sweep_cfg: SweepConfig,
    trial_id: int,
) -> PPOConfig:
    """Build a PPOConfig from sampled params + sweep fixed params."""
    timestamp = int(time.time())
    return PPOConfig(
        # Sampled hyperparams
        learning_rate=params.get("learning_rate", 2.5e-4),
        num_steps=params.get("num_steps", 128),
        num_minibatches=params.get("num_minibatches", 4),
        update_epochs=params.get("update_epochs", 4),
        gamma=params.get("gamma", 0.99),
        gae_lambda=params.get("gae_lambda", 0.95),
        clip_coef=params.get("clip_coef", 0.2),
        ent_coef=params.get("ent_coef", 0.01),
        embed_dim=params.get("embed_dim", 128),
        num_layers=params.get("num_layers", 3),
        num_heads=params.get("num_heads", 4),
        num_envs=params.get("num_envs", 8),
        # Fixed params
        total_timesteps=sweep_cfg.timesteps_per_trial,
        eval_episodes=sweep_cfg.eval_episodes,
        back_keys=sweep_cfg.back_keys,
        stake=sweep_cfg.stake,
        device=sweep_cfg.device,
        log_dir=sweep_cfg.output_dir,
        run_name=f"sweep_{trial_id}_{timestamp}",
        # Less frequent logging/eval during sweep
        log_interval=max(1, sweep_cfg.timesteps_per_trial // 10_000),
        eval_interval=max(1, sweep_cfg.timesteps_per_trial // 50_000),
        save_interval=999_999,  # don't save checkpoints during sweep
        game_spec=balatro_game_spec(),
    )


def run_sweep(sweep_cfg: SweepConfig | None = None) -> list[TrialResult]:
    """Run a hyperparameter sweep.

    Generates n_trials random configurations, trains each, and
    collects evaluation metrics. Results are saved to JSON.

    Parameters
    ----------
    sweep_cfg:
        Sweep configuration. Uses defaults if None.

    Returns
    -------
    list[TrialResult]
        Results sorted by metric (best first).
    """
    if sweep_cfg is None:
        sweep_cfg = SweepConfig()

    output_dir = Path(sweep_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = RandomSampler(sweep_cfg.search_space, seed=sweep_cfg.seed)
    results: list[TrialResult] = []

    print("=" * 70)
    print(f"  HYPERPARAMETER SWEEP — {sweep_cfg.n_trials} trials")
    print(f"  {sweep_cfg.timesteps_per_trial:,} timesteps/trial, metric={sweep_cfg.metric}")
    print("=" * 70)

    for trial_id in range(sweep_cfg.n_trials):
        raw_params = sampler.sample()
        params = _validate_and_fix(raw_params)
        trial = TrialResult(trial_id=trial_id, params=params)

        print(f"\n--- Trial {trial_id}/{sweep_cfg.n_trials - 1} ---")
        print(f"  params: {_format_params(params)}")

        ppo_cfg = _make_ppo_config(params, sweep_cfg, trial_id)

        try:
            train_result = train_ppo(ppo_cfg)
            _populate_trial(trial, train_result)
            trial.status = "completed"
        except Exception as e:
            trial.status = f"failed: {e}"
            print(f"  FAILED: {e}")

        results.append(trial)
        print(f"  result: avg_ante={trial.avg_ante:.2f}, win_rate={trial.win_rate:.1%}")

        # Save incremental results
        _save_results(results, output_dir / "sweep_results.json")

    # Rank and report
    ranked = _rank_results(results, sweep_cfg.metric)
    print("\n" + "=" * 70)
    print("  TOP 5 CONFIGS")
    print("=" * 70)
    for i, trial in enumerate(ranked[:5]):
        print(
            f"  #{i + 1} (trial {trial.trial_id}): "
            f"avg_ante={trial.avg_ante:.2f}, win_rate={trial.win_rate:.1%}"
        )
        print(f"       {_format_params(trial.params)}")

    _save_results(ranked, output_dir / "sweep_results.json")
    print(f"\nResults saved to {output_dir / 'sweep_results.json'}")

    return ranked


def _populate_trial(trial: TrialResult, train_result: TrainResult) -> None:
    """Fill trial metrics from training result."""
    trial.total_episodes = train_result.total_episodes
    trial.wall_time = train_result.wall_time
    if train_result.final_eval is not None:
        trial.avg_ante = train_result.final_eval.avg_ante
        trial.win_rate = train_result.final_eval.win_rate
        trial.avg_return = train_result.final_eval.avg_return
        trial.avg_length = train_result.final_eval.avg_length


def _rank_results(results: list[TrialResult], metric: str) -> list[TrialResult]:
    """Sort results by metric, best first."""
    completed = [r for r in results if r.status == "completed"]
    return sorted(completed, key=lambda r: getattr(r, metric, 0.0), reverse=True)


def _format_params(params: dict[str, Any]) -> str:
    """Format params dict for compact display."""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _save_results(results: list[TrialResult], path: Path) -> None:
    """Save results to JSON."""
    data = [r.to_dict() for r in results]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_sweep(results_path: str) -> dict[str, Any]:
    """Analyze sweep results from JSON.

    Returns dict with best config and per-param correlation with metric.
    """
    path = Path(results_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    completed = [d for d in data if d["status"] == "completed"]
    if not completed:
        print("No completed trials found.")
        return {"best": None, "param_correlations": {}}

    # Sort by avg_ante (primary metric)
    ranked = sorted(completed, key=lambda d: d["avg_ante"], reverse=True)

    print("=" * 70)
    print("  SWEEP ANALYSIS")
    print("=" * 70)

    print("\n  Top 5 configs:")
    for i, trial in enumerate(ranked[:5]):
        print(
            f"  #{i + 1} (trial {trial['trial_id']}): "
            f"avg_ante={trial['avg_ante']:.2f}, win_rate={trial['win_rate']:.1%}"
        )
        print(f"       {_format_params(trial['params'])}")

    # Param correlation with metric
    correlations = _compute_correlations(completed, "avg_ante")
    print("\n  Parameter correlations with avg_ante:")
    for param, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {param:<20s}  r={corr:+.3f}")

    # Save analysis
    analysis = {
        "best": ranked[0] if ranked else None,
        "param_correlations": correlations,
        "n_completed": len(completed),
        "n_total": len(data),
    }

    output_path = path.parent / "sweep_analysis.json"
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nAnalysis saved to {output_path}")

    return analysis


def _compute_correlations(
    trials: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    """Compute Pearson correlation between each param and the metric."""
    if len(trials) < 3:
        return {}

    metric_vals = [t[metric] for t in trials]
    metric_mean = sum(metric_vals) / len(metric_vals)
    metric_var = sum((v - metric_mean) ** 2 for v in metric_vals)

    if metric_var == 0:
        return {}

    # Collect all numeric param names
    all_params: set[str] = set()
    for t in trials:
        for k, v in t["params"].items():
            if isinstance(v, (int, float)):
                all_params.add(k)

    correlations: dict[str, float] = {}
    for param in sorted(all_params):
        vals = []
        m_vals = []
        for t, m in zip(trials, metric_vals):
            v = t["params"].get(param)
            if isinstance(v, (int, float)):
                vals.append(float(v))
                m_vals.append(m)

        if len(vals) < 3:
            continue

        p_mean = sum(vals) / len(vals)
        p_var = sum((v - p_mean) ** 2 for v in vals)
        if p_var == 0:
            correlations[param] = 0.0
            continue

        m_mean = sum(m_vals) / len(m_vals)
        cov = sum((v - p_mean) * (m - m_mean) for v, m in zip(vals, m_vals))
        m_var_local = sum((m - m_mean) ** 2 for m in m_vals)

        denom = math.sqrt(p_var * m_var_local)
        correlations[param] = cov / denom if denom > 0 else 0.0

    return correlations


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run_sweep()
