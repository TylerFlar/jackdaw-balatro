"""Tests for the validation CLI subcommands."""

from __future__ import annotations

import pytest

from jackdaw.cli.validate import run_benchmark, run_crash
from jackdaw.engine.runner import greedy_play_agent, simulate_run


class TestCrash:
    def test_crash_basic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """run_crash with a small count completes without exception."""
        exit_code = run_crash(count=5, agent_name="random")
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_crash_reports_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = run_crash(count=3, agent_name="random")
        captured = capsys.readouterr()
        assert "Runs:" in captured.out
        assert "Crashes:" in captured.out
        assert "determinism" in captured.out


class TestBenchmark:
    def test_benchmark_basic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """run_benchmark with a small count completes without exception."""
        run_benchmark(count=5)
        captured = capsys.readouterr()
        assert "Runs/sec:" in captured.out
        assert "Actions/sec:" in captured.out


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        """Two runs with the same seed and deterministic agent produce identical results."""
        gs1 = simulate_run("b_red", 1, "DET_TEST", greedy_play_agent, max_actions=500)
        gs2 = simulate_run("b_red", 1, "DET_TEST", greedy_play_agent, max_actions=500)
        assert gs1.get("dollars") == gs2.get("dollars")
        assert gs1.get("round") == gs2.get("round")
        assert gs1.get("won") == gs2.get("won")
        assert gs1.get("actions_taken") == gs2.get("actions_taken")


@pytest.mark.live
class TestSeed:
    """Seed validation tests — require a running balatrobot instance."""

    def test_seed_placeholder(self) -> None:
        """Placeholder — run_seed requires balatrobot, tested manually."""
        pytest.skip("Requires running balatrobot instance")
