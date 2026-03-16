"""Performance benchmarks for the hand evaluator.

The evaluator runs inside the optimizer's play enumeration — for a hand of
8 cards, C(8,1)+C(8,2)+C(8,3)+C(8,4)+C(8,5) = 218 subsets need evaluation.
During RL training this happens millions of times.

Run with: uv run pytest tests/engine/test_hand_eval_perf.py -v -s
"""

from __future__ import annotations

import itertools
import time

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.hand_eval import evaluate_hand, evaluate_poker_hand


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    sl = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rl = {
        "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
        "8": "8", "9": "9", "10": "T", "Jack": "J", "Queen": "Q",
        "King": "K", "Ace": "A",
    }
    c = Card()
    c.set_base(f"{sl[suit]}_{rl[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _make_8_card_hand() -> list[Card]:
    """Representative 8-card hand with mixed suits and ranks."""
    reset_sort_id_counter()
    return [
        _card("Hearts", "2"),
        _card("Hearts", "5"),
        _card("Hearts", "8"),
        _card("Spades", "Jack"),
        _card("Diamonds", "King"),
        _card("Clubs", "Ace"),
        _card("Hearts", "Queen"),
        _card("Spades", "7"),
    ]


def _all_subsets(hand: list[Card], max_size: int = 5) -> list[list[Card]]:
    """Generate all subsets of size 1..max_size (matching playable hands)."""
    subsets = []
    for r in range(1, max_size + 1):
        for combo in itertools.combinations(hand, r):
            subsets.append(list(combo))
    return subsets


class TestSingleHandPerformance:
    """Micro-benchmarks for single hand evaluation."""

    def test_evaluate_poker_hand_single(self):
        """evaluate_poker_hand on a single 5-card hand."""
        reset_sort_id_counter()
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "King"), _card("Diamonds", "King"),
            _card("Hearts", "Ace"),
        ]

        # Warm up
        for _ in range(100):
            evaluate_poker_hand(hand)

        n = 10_000
        start = time.perf_counter()
        for _ in range(n):
            evaluate_poker_hand(hand)
        elapsed = time.perf_counter() - start

        us_per_call = elapsed / n * 1_000_000
        print(f"\n  evaluate_poker_hand: {us_per_call:.1f} µs/call ({n:,} calls)")
        assert us_per_call < 100, f"Too slow: {us_per_call:.1f} µs (target < 100 µs)"

    def test_evaluate_hand_single(self):
        """evaluate_hand (full pipeline) on a single 5-card hand."""
        reset_sort_id_counter()
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "King"), _card("Diamonds", "King"),
            _card("Hearts", "Ace"),
        ]

        for _ in range(100):
            evaluate_hand(hand)

        n = 10_000
        start = time.perf_counter()
        for _ in range(n):
            evaluate_hand(hand)
        elapsed = time.perf_counter() - start

        us_per_call = elapsed / n * 1_000_000
        print(f"\n  evaluate_hand: {us_per_call:.1f} µs/call ({n:,} calls)")
        assert us_per_call < 200, f"Too slow: {us_per_call:.1f} µs (target < 200 µs)"


class TestSubsetEnumeration:
    """Benchmark evaluating all playable subsets from an 8-card hand."""

    def test_218_subsets(self):
        """All C(8,1)+...+C(8,5) = 218 subsets of an 8-card hand."""
        hand = _make_8_card_hand()
        subsets = _all_subsets(hand)
        assert len(subsets) == 8 + 28 + 56 + 70 + 56  # 218

        # Warm up
        for sub in subsets:
            evaluate_poker_hand(sub)

        n = 100
        start = time.perf_counter()
        for _ in range(n):
            for sub in subsets:
                evaluate_poker_hand(sub)
        elapsed = time.perf_counter() - start

        ms_per_enum = elapsed / n * 1000
        total_evals = n * len(subsets)
        us_per_eval = elapsed / total_evals * 1_000_000
        print(
            f"\n  218 subsets × {n} iterations: "
            f"{ms_per_enum:.2f} ms/enum, {us_per_eval:.1f} µs/eval"
        )
        assert ms_per_enum < 50, f"Too slow: {ms_per_enum:.1f} ms (target < 50 ms)"

    def test_218_subsets_with_joker_flags(self):
        """Same 218 subsets with Four Fingers + Shortcut active."""
        hand = _make_8_card_hand()
        subsets = _all_subsets(hand)

        n = 100
        start = time.perf_counter()
        for _ in range(n):
            for sub in subsets:
                evaluate_poker_hand(sub, four_fingers=True, shortcut=True)
        elapsed = time.perf_counter() - start

        ms_per_enum = elapsed / n * 1000
        print(f"\n  218 subsets (FF+SC) × {n}: {ms_per_enum:.2f} ms/enum")
        assert ms_per_enum < 50, f"Too slow: {ms_per_enum:.1f} ms"

    def test_218_subsets_full_pipeline(self):
        """218 subsets through evaluate_hand (full pipeline with jokers)."""
        hand = _make_8_card_hand()
        subsets = _all_subsets(hand)
        jokers = [create_joker("j_four_fingers"), create_joker("j_shortcut")]

        n = 50
        start = time.perf_counter()
        for _ in range(n):
            for sub in subsets:
                evaluate_hand(sub, jokers=jokers)
        elapsed = time.perf_counter() - start

        ms_per_enum = elapsed / n * 1000
        print(f"\n  218 subsets (full pipeline + jokers) × {n}: {ms_per_enum:.2f} ms/enum")
        assert ms_per_enum < 100, f"Too slow: {ms_per_enum:.1f} ms"


class TestMonteCarloScale:
    """Simulating Monte Carlo across many hands."""

    def test_1000_enumerations(self):
        """218 subsets × 1000 iterations (e.g. 1000 different game states)."""
        hand = _make_8_card_hand()
        subsets = _all_subsets(hand)

        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            for sub in subsets:
                evaluate_poker_hand(sub)
        elapsed = time.perf_counter() - start

        total = n * len(subsets)
        rate = total / elapsed
        print(
            f"\n  Monte Carlo: {total:,} evaluations in {elapsed:.2f}s "
            f"= {rate:,.0f} eval/sec"
        )
        assert elapsed < 30, f"Too slow: {elapsed:.1f}s (target < 30s)"


class TestBreakdown:
    """Profile individual components to identify bottlenecks."""

    def test_component_breakdown(self):
        """Time each detection component separately."""
        from jackdaw.engine.hand_eval import get_flush, get_highest, get_straight, get_x_same

        reset_sort_id_counter()
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "King"), _card("Diamonds", "King"),
            _card("Hearts", "Ace"),
        ]

        n = 10_000
        components = {
            "get_x_same(5)": lambda: get_x_same(5, hand),
            "get_x_same(4)": lambda: get_x_same(4, hand),
            "get_x_same(3)": lambda: get_x_same(3, hand),
            "get_x_same(2)": lambda: get_x_same(2, hand),
            "get_flush": lambda: get_flush(hand),
            "get_straight": lambda: get_straight(hand),
            "get_highest": lambda: get_highest(hand),
        }

        print("\n  Component breakdown:")
        total_us = 0.0
        for name, func in components.items():
            # warm up
            for _ in range(100):
                func()
            start = time.perf_counter()
            for _ in range(n):
                func()
            elapsed = time.perf_counter() - start
            us = elapsed / n * 1_000_000
            total_us += us
            print(f"    {name:20s}: {us:.2f} µs")

        print(f"    {'TOTAL':20s}: {total_us:.2f} µs")
        # Just documenting, not asserting thresholds on individual components
