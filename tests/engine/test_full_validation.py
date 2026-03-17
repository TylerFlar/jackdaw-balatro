"""Full validation tests — crash resistance, determinism, performance.

Coverage
--------
* 100 random seeds with random_agent: all complete without crash.
* 100 random seeds with greedy_agent: all complete without crash.
* 5 scripted runs with deterministic outcomes (same seed → same result).
* Performance: >500 runs/sec.
* Various deck types: all 15 decks complete without crash.
* All 8 stake levels complete without crash.
* Max actions limit works (no infinite loops).
"""

from __future__ import annotations

import time

import pytest

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
    SkipBlind,
)
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import greedy_play_agent, random_agent, simulate_run


# ---------------------------------------------------------------------------
# Crash resistance — random agent
# ---------------------------------------------------------------------------


class TestRandomAgentCrashResistance:
    @pytest.mark.parametrize("seed_idx", range(100))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"FVRAND_{seed_idx}", random_agent, max_actions=500)
        assert gs["actions_taken"] <= 500
        assert gs.get("phase") in (
            GamePhase.GAME_OVER, GamePhase.SHOP,
            GamePhase.ROUND_EVAL, GamePhase.BLIND_SELECT,
        )


# ---------------------------------------------------------------------------
# Crash resistance — greedy agent
# ---------------------------------------------------------------------------


class TestGreedyAgentCrashResistance:
    @pytest.mark.parametrize("seed_idx", range(100))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"FVGRDY_{seed_idx}", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] <= 500


# ---------------------------------------------------------------------------
# Scripted determinism
# ---------------------------------------------------------------------------


def _scripted_run(seed: str) -> dict:
    """Play a scripted Small→Big→Boss sequence and return final state."""
    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    gs["jokers"] = []
    gs["consumables"] = []

    for blind_name in ("Small", "Big", "Boss"):
        if blind_name != "Small":
            step(gs, NextRound())
        step(gs, SelectBlind())
        gs["blind"].chips = 1  # easy win
        # Play hands until we win or run out
        for _ in range(10):
            if gs["phase"] != GamePhase.SELECTING_HAND:
                break
            hand = gs.get("hand", [])
            n = min(5, len(hand))
            if n == 0:
                break
            step(gs, PlayHand(card_indices=tuple(range(n))))
        if gs["phase"] == GamePhase.ROUND_EVAL:
            step(gs, CashOut())

    return gs


class TestScriptedDeterminism:
    @pytest.mark.parametrize("seed", [
        "SCRIPT_A", "SCRIPT_B", "SCRIPT_C", "SCRIPT_D", "SCRIPT_E",
    ])
    def test_same_seed_same_outcome(self, seed):
        gs1 = _scripted_run(seed)
        gs2 = _scripted_run(seed)
        assert gs1["dollars"] == gs2["dollars"]
        assert gs1["chips"] == gs2["chips"]
        assert gs1.get("won") == gs2.get("won")
        assert gs1["round_resets"]["ante"] == gs2["round_resets"]["ante"]

    def test_different_seeds_differ(self):
        results = set()
        for seed in ["DIFF_1", "DIFF_2", "DIFF_3", "DIFF_4", "DIFF_5"]:
            gs = _scripted_run(seed)
            results.add(gs["dollars"])
        assert len(results) > 1, "5 different seeds should produce at least 2 different dollar amounts"

    def test_scripted_beats_three_blinds(self):
        gs = _scripted_run("SCRIPT_FULL")
        # After beating Small, Big, Boss, ante should advance to 2
        assert gs["round_resets"]["ante"] == 2
        assert gs.get("phase") == GamePhase.SHOP

    def test_scripted_accumulates_money(self):
        gs = _scripted_run("SCRIPT_MONEY")
        # After 3 blinds worth of earnings, should have more than starting $4
        assert gs["dollars"] > 4

    def test_scripted_shop_populated(self):
        gs = _scripted_run("SCRIPT_SHOP")
        # After CashOut, shop should be populated
        assert len(gs.get("shop_cards", [])) > 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_random_agent_over_500_runs_per_sec(self):
        n = 200
        t0 = time.time()
        for i in range(n):
            simulate_run("b_red", 1, f"PERF_R_{i}", random_agent, max_actions=500)
        elapsed = time.time() - t0
        runs_per_sec = n / elapsed
        assert runs_per_sec > 500, f"Only {runs_per_sec:.0f} runs/sec (target: >500)"

    def test_greedy_agent_over_500_runs_per_sec(self):
        n = 200
        t0 = time.time()
        for i in range(n):
            simulate_run("b_red", 1, f"PERF_G_{i}", greedy_play_agent, max_actions=500)
        elapsed = time.time() - t0
        runs_per_sec = n / elapsed
        assert runs_per_sec > 500, f"Only {runs_per_sec:.0f} runs/sec (target: >500)"


# ---------------------------------------------------------------------------
# All deck types
# ---------------------------------------------------------------------------


ALL_DECKS = [
    "b_red", "b_blue", "b_yellow", "b_green", "b_black",
    "b_magic", "b_nebula", "b_ghost", "b_abandoned", "b_checkered",
    "b_zodiac", "b_painted", "b_anaglyph", "b_plasma", "b_erratic",
]


class TestAllDecks:
    @pytest.mark.parametrize("back_key", ALL_DECKS)
    def test_deck_completes(self, back_key):
        gs = simulate_run(back_key, 1, "DECK_FV", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0


# ---------------------------------------------------------------------------
# All stake levels
# ---------------------------------------------------------------------------


class TestAllStakes:
    @pytest.mark.parametrize("stake", range(1, 9))
    def test_stake_completes(self, stake):
        gs = simulate_run("b_red", stake, "STAKE_FV", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0


# ---------------------------------------------------------------------------
# Max actions limit
# ---------------------------------------------------------------------------


class TestMaxActionsLimit:
    def test_respects_limit(self):
        gs = simulate_run("b_red", 1, "LIMIT", random_agent, max_actions=5)
        assert gs["actions_taken"] <= 5

    def test_very_high_limit_still_terminates(self):
        gs = simulate_run("b_red", 1, "HIGH_LIMIT", greedy_play_agent, max_actions=10000)
        assert gs["actions_taken"] < 10000  # should hit GAME_OVER first


# ---------------------------------------------------------------------------
# Skip blind flow
# ---------------------------------------------------------------------------


class TestSkipBlindFlow:
    def test_skip_small_skip_big_select_boss(self):
        gs = initialize_run("b_red", 1, "SKIP_FLOW")
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"
        gs["jokers"] = []
        gs["consumables"] = []

        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Big"
        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Boss"
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND
        gs["blind"].chips = 1
        gs["blind"].disabled = True  # disable boss effect to avoid total debuff
        # Play until we beat the blind
        for _ in range(10):
            if gs["phase"] != GamePhase.SELECTING_HAND:
                break
            hand = gs.get("hand", [])
            n = min(5, len(hand))
            if n == 0:
                break
            step(gs, PlayHand(card_indices=tuple(range(n))))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        assert len(gs.get("shop_cards", [])) > 0
