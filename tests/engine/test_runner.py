"""Tests for jackdaw.engine.runner — simulate_run, random_agent, greedy_play_agent.

Coverage
--------
* 100 seeds with random_agent: all complete without crash.
* 100 seeds with greedy_play_agent: all complete without crash.
* Stats tracked: actions_taken, round, phase at end.
* Scripted agent deterministic: same seed → same outcome.
* Performance: average run time well under 100ms.
* Various decks and stakes: Abandoned, Zodiac, Black+stake5.
"""

from __future__ import annotations

import time

import pytest

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.runner import greedy_play_agent, random_agent, simulate_run


# ---------------------------------------------------------------------------
# Crash tests — 100 seeds each
# ---------------------------------------------------------------------------


class TestRandomAgentNoCrash:
    @pytest.mark.parametrize("seed_idx", range(100))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"RAND_{seed_idx}", random_agent, max_actions=500)
        assert gs.get("phase") in (
            GamePhase.GAME_OVER,
            GamePhase.SHOP,
            GamePhase.ROUND_EVAL,
            GamePhase.BLIND_SELECT,
        )
        assert gs["actions_taken"] <= 500


class TestGreedyAgentNoCrash:
    @pytest.mark.parametrize("seed_idx", range(100))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"GREEDY_{seed_idx}", greedy_play_agent, max_actions=500)
        assert gs.get("phase") in (
            GamePhase.GAME_OVER,
            GamePhase.SHOP,
            GamePhase.ROUND_EVAL,
            GamePhase.BLIND_SELECT,
        )
        assert gs["actions_taken"] <= 500


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


class TestStatsTracking:
    def test_actions_taken_tracked(self):
        gs = simulate_run("b_red", 1, "STATS1", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0

    def test_round_counter(self):
        gs = simulate_run("b_red", 1, "STATS2", greedy_play_agent, max_actions=500)
        # Even if lost, round should be ≥ 0
        assert gs.get("round", 0) >= 0

    def test_phase_is_terminal(self):
        gs = simulate_run("b_red", 1, "STATS3", greedy_play_agent, max_actions=500)
        assert gs.get("phase") in (GamePhase.GAME_OVER, GamePhase.SHOP)

    def test_hands_played_tracked(self):
        gs = simulate_run("b_red", 1, "STATS4", greedy_play_agent, max_actions=500)
        assert gs.get("hands_played", 0) >= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_greedy_same_seed_same_outcome(self):
        gs1 = simulate_run("b_red", 1, "DETERM_A", greedy_play_agent, max_actions=500)
        gs2 = simulate_run("b_red", 1, "DETERM_A", greedy_play_agent, max_actions=500)
        assert gs1["actions_taken"] == gs2["actions_taken"]
        assert gs1.get("won") == gs2.get("won")
        assert gs1.get("chips") == gs2.get("chips")
        assert gs1.get("dollars") == gs2.get("dollars")

    def test_different_seeds_may_differ(self):
        results = set()
        for i in range(10):
            gs = simulate_run("b_red", 1, f"DIFF_{i}", greedy_play_agent, max_actions=500)
            results.add(gs.get("chips", 0))
        # With 10 different seeds, chips should vary
        assert len(results) > 1


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_average_run_time_under_100ms(self):
        n = 50
        t0 = time.time()
        for i in range(n):
            simulate_run("b_red", 1, f"PERF_{i}", greedy_play_agent, max_actions=500)
        elapsed = time.time() - t0
        avg_ms = 1000 * elapsed / n
        assert avg_ms < 100, f"Average {avg_ms:.1f}ms exceeds 100ms target"

    def test_random_agent_performance(self):
        n = 50
        t0 = time.time()
        for i in range(n):
            simulate_run("b_red", 1, f"RPERF_{i}", random_agent, max_actions=500)
        elapsed = time.time() - t0
        avg_ms = 1000 * elapsed / n
        assert avg_ms < 100, f"Average {avg_ms:.1f}ms exceeds 100ms target"


# ---------------------------------------------------------------------------
# Different decks and stakes
# ---------------------------------------------------------------------------


class TestVariousDecks:
    @pytest.mark.parametrize("back_key", [
        "b_red", "b_blue", "b_black", "b_abandoned", "b_zodiac",
        "b_magic", "b_plasma", "b_green", "b_erratic", "b_painted",
    ])
    def test_deck_completes(self, back_key):
        gs = simulate_run(back_key, 1, "DECK_TEST", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0

    @pytest.mark.parametrize("stake", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_stake_completes(self, stake):
        gs = simulate_run("b_red", stake, "STAKE_TEST", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0


# ---------------------------------------------------------------------------
# Max actions limit
# ---------------------------------------------------------------------------


class TestMaxActions:
    def test_respects_max_actions(self):
        gs = simulate_run("b_red", 1, "MAX_ACT", random_agent, max_actions=3)
        assert gs["actions_taken"] <= 3


# ---------------------------------------------------------------------------
# Edge: game won scenario
# ---------------------------------------------------------------------------


class TestGameWon:
    def test_won_run_terminates(self):
        """A run that wins should terminate gracefully."""
        # Force win by setting win_ante=1 via post-init
        from jackdaw.engine.run_init import initialize_run
        from jackdaw.engine.game import step
        from jackdaw.engine.actions import SelectBlind, PlayHand, CashOut, NextRound, SkipBlind

        gs = initialize_run("b_red", 1, "WIN_TEST")
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"
        gs["win_ante"] = 1

        # Skip to Boss
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        # Select Boss
        step(gs, SelectBlind())
        # Beat it
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs.get("won") is True
        # Cash out
        step(gs, CashOut())
        # The run is won — should be in SHOP
        assert gs["phase"] == GamePhase.SHOP
