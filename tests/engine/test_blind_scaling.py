"""Tests for blind chip requirement scaling.

Verifies hardcoded antes 1-8 at all scaling levels, exponential
formula for antes 9-12 against LuaJIT ground truth, blind type
multipliers, and Plasma Deck ante_scaling.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.data.blind_scaling import (
    BLIND_MULT,
    get_blind_amount,
    get_blind_target,
)

# ============================================================================
# Hardcoded tables (antes 1-8)
# ============================================================================

# Ground truth from misc_functions.lua:922-944
SCALING_1 = [300, 800, 2_000, 5_000, 11_000, 20_000, 35_000, 50_000]
SCALING_2 = [300, 900, 2_600, 8_000, 20_000, 36_000, 60_000, 100_000]
SCALING_3 = [300, 1_000, 3_200, 9_000, 25_000, 60_000, 110_000, 200_000]


class TestScaling1:
    """White/Red stake (scaling=1)."""

    @pytest.mark.parametrize("ante,expected", enumerate(SCALING_1, 1))
    def test_antes_1_to_8(self, ante: int, expected: int):
        assert get_blind_amount(ante, scaling=1) == expected


class TestScaling2:
    """Green-Blue stake (scaling=2)."""

    @pytest.mark.parametrize("ante,expected", enumerate(SCALING_2, 1))
    def test_antes_1_to_8(self, ante: int, expected: int):
        assert get_blind_amount(ante, scaling=2) == expected


class TestScaling3:
    """Purple-Gold stake (scaling=3)."""

    @pytest.mark.parametrize("ante,expected", enumerate(SCALING_3, 1))
    def test_antes_1_to_8(self, ante: int, expected: int):
        assert get_blind_amount(ante, scaling=3) == expected


# ============================================================================
# Exponential formula (antes 9+) — ground truth from LuaJIT
# ============================================================================

EXPONENTIAL_TRUTH = {
    1: {9: 110_000, 10: 560_000, 11: 7_200_000, 12: 300_000_000},
    2: {9: 230_000, 10: 1_100_000, 11: 14_000_000, 12: 600_000_000},
    3: {9: 460_000, 10: 2_200_000, 11: 29_000_000, 12: 1_200_000_000},
}


class TestExponentialFormula:
    """Antes 9-12 use exponential growth with sig-fig rounding."""

    @pytest.mark.parametrize("scaling", [1, 2, 3])
    @pytest.mark.parametrize("ante", [9, 10, 11, 12])
    def test_matches_luajit(self, scaling: int, ante: int):
        expected = EXPONENTIAL_TRUTH[scaling][ante]
        actual = get_blind_amount(ante, scaling)
        assert actual == expected, f"scaling={scaling} ante={ante}: {actual} != {expected}"


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_ante_0(self):
        assert get_blind_amount(0) == 100

    def test_ante_negative(self):
        assert get_blind_amount(-1) == 100

    def test_ante_1_all_scalings_same(self):
        """All scaling levels agree at ante 1."""
        assert get_blind_amount(1, 1) == 300
        assert get_blind_amount(1, 2) == 300
        assert get_blind_amount(1, 3) == 300

    def test_default_scaling_is_1(self):
        assert get_blind_amount(5) == get_blind_amount(5, scaling=1)

    def test_unknown_scaling_defaults_to_1(self):
        """Unknown scaling levels fall back to level 1."""
        assert get_blind_amount(5, scaling=99) == get_blind_amount(5, scaling=1)

    def test_ante_9_scaling_increases(self):
        """Higher scaling = harder at the same ante."""
        a1 = get_blind_amount(9, 1)
        a2 = get_blind_amount(9, 2)
        a3 = get_blind_amount(9, 3)
        assert a1 < a2 < a3


# ============================================================================
# Blind type multipliers
# ============================================================================


class TestBlindMult:
    def test_small_mult(self):
        assert BLIND_MULT["Small"] == 1.0

    def test_big_mult(self):
        assert BLIND_MULT["Big"] == 1.5

    def test_boss_mult(self):
        assert BLIND_MULT["Boss"] == 2.0


# ============================================================================
# get_blind_target (full chip target)
# ============================================================================


class TestGetBlindTarget:
    def test_small_blind_ante_1(self):
        # 300 × 1.0 × 1.0 = 300
        assert get_blind_target(1, "Small") == 300

    def test_big_blind_ante_1(self):
        # 300 × 1.5 × 1.0 = 450
        assert get_blind_target(1, "Big") == 450

    def test_boss_blind_ante_1(self):
        # 300 × 2.0 × 1.0 = 600
        assert get_blind_target(1, "Boss") == 600

    def test_boss_blind_ante_8_scaling_1(self):
        # 50000 × 2.0 × 1.0 = 100000
        assert get_blind_target(8, "Boss", scaling=1) == 100_000

    def test_boss_blind_ante_8_scaling_3(self):
        # 200000 × 2.0 × 1.0 = 400000
        assert get_blind_target(8, "Boss", scaling=3) == 400_000

    def test_plasma_deck_doubles(self):
        """Plasma Deck: ante_scaling=2.0 doubles all targets."""
        normal = get_blind_target(1, "Boss", scaling=1, ante_scaling=1.0)
        plasma = get_blind_target(1, "Boss", scaling=1, ante_scaling=2.0)
        assert plasma == normal * 2

    def test_plasma_deck_ante_5(self):
        # 11000 × 2.0 × 2.0 = 44000
        assert get_blind_target(5, "Boss", scaling=1, ante_scaling=2.0) == 44_000

    def test_plasma_small_ante_3_scaling_2(self):
        # 2600 × 1.0 × 2.0 = 5200
        assert get_blind_target(3, "Small", scaling=2, ante_scaling=2.0) == 5_200

    def test_big_blind_ante_12_scaling_3(self):
        # 1200000000 × 1.5 = 1800000000
        assert get_blind_target(12, "Big", scaling=3) == 1_800_000_000
