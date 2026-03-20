"""Tests for hyperparameter sweep infrastructure."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from jackdaw.env.training.sweep import (
    DEFAULT_SEARCH_SPACE,
    RandomSampler,
    SweepConfig,
    TrialResult,
    _compute_correlations,
    _validate_and_fix,
    analyze_sweep,
    run_sweep,
)

# ---------------------------------------------------------------------------
# RandomSampler
# ---------------------------------------------------------------------------


class TestRandomSampler:
    def test_sample_returns_all_keys(self):
        sampler = RandomSampler(DEFAULT_SEARCH_SPACE, seed=0)
        config = sampler.sample()
        for key in DEFAULT_SEARCH_SPACE:
            assert key in config, f"Missing key: {key}"

    def test_categorical_values_in_range(self):
        sampler = RandomSampler(DEFAULT_SEARCH_SPACE, seed=42)
        for _ in range(50):
            config = sampler.sample()
            assert config["num_steps"] in [64, 128, 256, 512]
            assert config["num_minibatches"] in [2, 4, 8]
            assert config["update_epochs"] in [2, 4, 8]
            assert config["gamma"] in [0.95, 0.99, 0.995]
            assert config["embed_dim"] in [64, 128, 256]
            assert config["num_layers"] in [2, 3, 4]
            assert config["num_envs"] in [4, 8, 16]
            assert config["clip_coef"] in [0.1, 0.2, 0.3]
            assert config["gae_lambda"] in [0.9, 0.95, 0.98]

    def test_continuous_log_values_in_range(self):
        sampler = RandomSampler(DEFAULT_SEARCH_SPACE, seed=42)
        for _ in range(100):
            config = sampler.sample()
            assert 1e-5 <= config["learning_rate"] <= 1e-3
            assert 0.001 <= config["ent_coef"] <= 0.1

    def test_deterministic_with_seed(self):
        s1 = RandomSampler(DEFAULT_SEARCH_SPACE, seed=123)
        s2 = RandomSampler(DEFAULT_SEARCH_SPACE, seed=123)
        for _ in range(10):
            assert s1.sample() == s2.sample()

    def test_different_seeds_produce_different_configs(self):
        s1 = RandomSampler(DEFAULT_SEARCH_SPACE, seed=0)
        s2 = RandomSampler(DEFAULT_SEARCH_SPACE, seed=1)
        configs1 = [s1.sample() for _ in range(5)]
        configs2 = [s2.sample() for _ in range(5)]
        assert configs1 != configs2

    def test_continuous_uniform(self):
        """Non-log continuous params sample uniformly."""
        space = {"x": (0.0, 1.0)}
        sampler = RandomSampler(space, seed=42)
        vals = [sampler.sample()["x"] for _ in range(100)]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_scalar_passthrough(self):
        """Scalar values pass through unchanged."""
        space = {"fixed": 42}
        sampler = RandomSampler(space, seed=0)
        assert sampler.sample()["fixed"] == 42


# ---------------------------------------------------------------------------
# _validate_and_fix
# ---------------------------------------------------------------------------


class TestValidateAndFix:
    def test_valid_config_unchanged(self):
        params = {"num_steps": 128, "num_envs": 8, "num_minibatches": 4, "embed_dim": 128}
        fixed = _validate_and_fix(params)
        assert fixed["num_minibatches"] == 4

    def test_fixes_indivisible_minibatches(self):
        # 64 * 4 = 256, not divisible by 8 when num_steps=64, num_envs=4
        # Wait: 64 * 4 = 256, 256 / 8 = 32. Actually that works.
        # Let's use a case that truly fails: num_steps=64, num_envs=6 => 384
        # 384 % 8 = 0, so that works too. Try num_steps=100, num_envs=3 => 300
        params = {"num_steps": 100, "num_envs": 3, "num_minibatches": 8, "embed_dim": 128}
        fixed = _validate_and_fix(params)
        total = 100 * 3
        assert total % fixed["num_minibatches"] == 0

    def test_all_default_space_configs_valid(self):
        """Every sampled config from DEFAULT_SEARCH_SPACE is valid after fix."""
        sampler = RandomSampler(DEFAULT_SEARCH_SPACE, seed=42)
        for _ in range(100):
            raw = sampler.sample()
            fixed = _validate_and_fix(raw)
            total = fixed["num_steps"] * fixed["num_envs"]
            assert total % fixed["num_minibatches"] == 0, (
                f"Invalid: {fixed['num_steps']}*{fixed['num_envs']} "
                f"% {fixed['num_minibatches']} != 0"
            )
            assert fixed["embed_dim"] % fixed["num_heads"] == 0

    def test_num_heads_set_correctly(self):
        assert _validate_and_fix({"embed_dim": 256})["num_heads"] == 4
        assert _validate_and_fix({"embed_dim": 64})["num_heads"] == 4
        assert _validate_and_fix({"embed_dim": 4})["num_heads"] == 2


# ---------------------------------------------------------------------------
# TrialResult
# ---------------------------------------------------------------------------


class TestTrialResult:
    def test_to_dict(self):
        trial = TrialResult(trial_id=0, params={"lr": 0.001}, avg_ante=2.5)
        d = trial.to_dict()
        assert d["trial_id"] == 0
        assert d["avg_ante"] == 2.5
        assert d["params"] == {"lr": 0.001}


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_perfect_positive(self):
        trials = [
            {"params": {"x": 1.0}, "avg_ante": 1.0},
            {"params": {"x": 2.0}, "avg_ante": 2.0},
            {"params": {"x": 3.0}, "avg_ante": 3.0},
        ]
        corr = _compute_correlations(trials, "avg_ante")
        assert abs(corr["x"] - 1.0) < 1e-6

    def test_perfect_negative(self):
        trials = [
            {"params": {"x": 1.0}, "avg_ante": 3.0},
            {"params": {"x": 2.0}, "avg_ante": 2.0},
            {"params": {"x": 3.0}, "avg_ante": 1.0},
        ]
        corr = _compute_correlations(trials, "avg_ante")
        assert abs(corr["x"] - (-1.0)) < 1e-6

    def test_no_correlation(self):
        # Constant metric => correlation undefined, returns empty
        trials = [
            {"params": {"x": 1.0}, "avg_ante": 2.0},
            {"params": {"x": 2.0}, "avg_ante": 2.0},
            {"params": {"x": 3.0}, "avg_ante": 2.0},
        ]
        corr = _compute_correlations(trials, "avg_ante")
        assert corr == {}

    def test_weak_correlation(self):
        # Non-constant metric with mixed param values
        trials = [
            {"params": {"x": 1.0}, "avg_ante": 1.0},
            {"params": {"x": 2.0}, "avg_ante": 3.0},
            {"params": {"x": 3.0}, "avg_ante": 2.0},
        ]
        corr = _compute_correlations(trials, "avg_ante")
        assert "x" in corr
        assert abs(corr["x"]) < 1.0  # not perfect

    def test_too_few_trials(self):
        trials = [{"params": {"x": 1.0}, "avg_ante": 1.0}]
        corr = _compute_correlations(trials, "avg_ante")
        assert corr == {}


# ---------------------------------------------------------------------------
# analyze_sweep
# ---------------------------------------------------------------------------


class TestAnalyzeSweep:
    def test_analyze_from_json(self):
        data = [
            TrialResult(
                trial_id=0, params={"lr": 0.01, "layers": 2}, avg_ante=2.0, status="completed"
            ).to_dict(),
            TrialResult(
                trial_id=1, params={"lr": 0.001, "layers": 3}, avg_ante=3.0, status="completed"
            ).to_dict(),
            TrialResult(
                trial_id=2, params={"lr": 0.0001, "layers": 4}, avg_ante=1.5, status="completed"
            ).to_dict(),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            analysis = analyze_sweep(str(path))
            assert analysis["best"]["trial_id"] == 1
            assert analysis["n_completed"] == 3
            assert "lr" in analysis["param_correlations"]
            # Analysis file written
            assert (Path(tmpdir) / "sweep_analysis.json").exists()


# ---------------------------------------------------------------------------
# Integration: run_sweep smoke test
# ---------------------------------------------------------------------------


class TestRunSweep:
    def test_smoke_2_trials(self):
        """Run sweep with 2 trials, 1024 timesteps each."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_cfg = SweepConfig(
                search_space={
                    "learning_rate": (1e-4, 1e-3, "log"),
                    "num_steps": [64, 128],
                    "num_envs": [2],
                    "num_minibatches": [2],
                    "update_epochs": [2],
                    "embed_dim": [32],
                    "num_layers": [1],
                },
                n_trials=2,
                timesteps_per_trial=1024,
                eval_episodes=5,
                seed=42,
                output_dir=tmpdir,
                device="cpu",
            )
            results = run_sweep(sweep_cfg)
            assert len(results) >= 1  # at least 1 completed
            assert all(r.status == "completed" for r in results)

            # JSON file written
            json_path = Path(tmpdir) / "sweep_results.json"
            assert json_path.exists()
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert len(data) == 2
