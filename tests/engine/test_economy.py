"""Tests for jackdaw.engine.economy — end-of-round money calculation."""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.economy import (
    RoundEarnings,
    calculate_discard_cost,
    calculate_round_earnings,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_blind() -> Blind:
    """Small Blind at ante 1: $3 reward."""
    return Blind.create("bl_small", ante=1)


def _big_blind() -> Blind:
    """Big Blind at ante 1: $4 reward."""
    return Blind.create("bl_big", ante=1)


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    c.sell_cost = 1
    return c


def _rental_joker() -> Card:
    """A rental joker (costs $3/round)."""
    j = _joker("j_spare_trousers")
    j.ability["rental"] = True
    return j


# ---------------------------------------------------------------------------
# calculate_discard_cost
# ---------------------------------------------------------------------------

class TestCalculateDiscardCost:
    def test_no_modifier_returns_zero(self):
        assert calculate_discard_cost({}) == 0

    def test_no_modifiers_key_returns_zero(self):
        assert calculate_discard_cost({"modifiers": {}}) == 0

    def test_golden_needle_returns_one(self):
        gs = {"modifiers": {"discard_cost": 1}}
        assert calculate_discard_cost(gs) == 1

    def test_custom_cost(self):
        gs = {"modifiers": {"discard_cost": 3}}
        assert calculate_discard_cost(gs) == 3


# ---------------------------------------------------------------------------
# RoundEarnings dataclass
# ---------------------------------------------------------------------------

class TestRoundEarnings:
    def test_default_all_zero(self):
        e = RoundEarnings()
        assert e.blind_reward == 0
        assert e.unused_hands_bonus == 0
        assert e.unused_discards_bonus == 0
        assert e.interest == 0
        assert e.joker_dollars == 0
        assert e.rental_cost == 0
        assert e.total == 0

    def test_total_field_is_explicit(self):
        e = RoundEarnings(blind_reward=3, unused_hands_bonus=2, interest=2, total=7)
        assert e.total == 7


# ---------------------------------------------------------------------------
# Basic round — blind + hands + interest
# ---------------------------------------------------------------------------

class TestBasicRound:
    """Beat Small Blind with 2 unused hands, $12 in bank.

    Expected: blind $3 + hands $2 (2×$1) + interest $2 (12//5=2 brackets) = $7.
    """

    def test_basic_earnings(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=2,
            discards_left=0,
            money=12,
            jokers=[],
            game_state={},
        )
        assert result.blind_reward == 3
        assert result.unused_hands_bonus == 2
        assert result.interest == 2
        assert result.joker_dollars == 0
        assert result.rental_cost == 0
        assert result.total == 7

    def test_big_blind_reward(self):
        result = calculate_round_earnings(
            blind=_big_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[],
            game_state={},
        )
        assert result.blind_reward == 4
        assert result.total == 4

    def test_zero_hands_left_no_hand_bonus(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[],
            game_state={},
        )
        assert result.unused_hands_bonus == 0

    def test_no_extra_hand_money_modifier(self):
        gs = {"modifiers": {"no_extra_hand_money": True}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=3,
            discards_left=0,
            money=0,
            jokers=[],
            game_state=gs,
        )
        assert result.unused_hands_bonus == 0

    def test_custom_money_per_hand(self):
        gs = {"modifiers": {"money_per_hand": 3}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=2,
            discards_left=0,
            money=0,
            jokers=[],
            game_state=gs,
        )
        assert result.unused_hands_bonus == 6


# ---------------------------------------------------------------------------
# Green Deck — money per discard modifier
# ---------------------------------------------------------------------------

class TestGreenDeck:
    """Green Deck: money_per_discard=1. 3 unused hands × $1 + 2 unused discards × $1 = $5 bonus."""

    def test_green_deck_discards(self):
        gs = {"modifiers": {"money_per_discard": 1}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=3,
            discards_left=2,
            money=0,
            jokers=[],
            game_state=gs,
        )
        assert result.unused_hands_bonus == 3
        assert result.unused_discards_bonus == 2

    def test_green_deck_higher_rate(self):
        gs = {"modifiers": {"money_per_discard": 2}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=3,
            discards_left=2,
            money=0,
            jokers=[],
            game_state=gs,
        )
        assert result.unused_discards_bonus == 4

    def test_no_money_per_discard_modifier_zero(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=5,
            money=0,
            jokers=[],
            game_state={},
        )
        assert result.unused_discards_bonus == 0

    def test_zero_discards_left_no_bonus(self):
        gs = {"modifiers": {"money_per_discard": 1}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[],
            game_state=gs,
        )
        assert result.unused_discards_bonus == 0


# ---------------------------------------------------------------------------
# Interest calculation
# ---------------------------------------------------------------------------

class TestInterest:
    """Interest = interest_amount × min(effective_money // 5, interest_cap // 5)."""

    def test_below_threshold_no_interest(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=4,
            jokers=[],
            game_state={},
        )
        assert result.interest == 0

    def test_exactly_five_dollars_one_bracket(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=5,
            jokers=[],
            game_state={},
        )
        assert result.interest == 1

    def test_twelve_dollars_two_brackets(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=12,
            jokers=[],
            game_state={},
        )
        assert result.interest == 2  # 12 // 5 = 2

    def test_no_interest_modifier(self):
        gs = {"modifiers": {"no_interest": True}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=100,
            jokers=[],
            game_state=gs,
        )
        assert result.interest == 0

    def test_interest_capped_at_default_25(self):
        """Default cap=25 → at most 5 brackets → $5 interest."""
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=200,
            jokers=[],
            game_state={},
        )
        # min(200//5=40, 25//5=5) = 5
        assert result.interest == 5

    def test_interest_cap_custom_yields_25(self):
        """interest_cap=125: min(200//5=40, 125//5=25)=25 → $25."""
        gs = {"interest_cap": 125}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=200,
            jokers=[],
            game_state=gs,
        )
        assert result.interest == 25

    def test_seed_money_voucher_cap_50(self):
        """Seed Money sets interest_cap=50 → min(200//5=40, 50//5=10)=10."""
        gs = {"interest_cap": 50}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=200,
            jokers=[],
            game_state=gs,
        )
        assert result.interest == 10

    def test_money_tree_voucher_cap_100(self):
        """Money Tree sets interest_cap=100 → min(200//5=40, 100//5=20)=20."""
        gs = {"interest_cap": 100}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=200,
            jokers=[],
            game_state=gs,
        )
        assert result.interest == 20


# ---------------------------------------------------------------------------
# To the Moon — increased interest_amount
# ---------------------------------------------------------------------------

class TestToTheMoon:
    """To the Moon joker increases interest_amount by 1 per copy."""

    def test_double_interest_amount(self):
        """interest_amount=2: $10 → min(10//5=2, 25//5=5)=2 × 2 = $4."""
        gs = {"interest_amount": 2}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=10,
            jokers=[],
            game_state=gs,
        )
        assert result.interest == 4

    def test_triple_interest_amount(self):
        """interest_amount=3: $15 → min(15//5=3, 25//5=5)=3 × 3 = $9."""
        gs = {"interest_amount": 3}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=15,
            jokers=[],
            game_state=gs,
        )
        assert result.interest == 9

    def test_interest_still_capped(self):
        """interest_amount=2, cap=25: min(200//5=40, 5)=5 × 2 = $10."""
        gs = {"interest_amount": 2}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=200,
            jokers=[],
            game_state=gs,
        )
        assert result.interest == 10


# ---------------------------------------------------------------------------
# Rental joker costs
# ---------------------------------------------------------------------------

class TestRentalJokers:
    """Rental jokers are deducted BEFORE interest is computed."""

    def test_one_rental_costs_three(self):
        j = _rental_joker()
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state={},
        )
        assert result.rental_cost == 3
        # total = blind(3) - rental(3) = 0
        assert result.total == 0

    def test_two_rentals_costs_six(self):
        j1 = _rental_joker()
        j2 = _rental_joker()
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j1, j2],
            game_state={},
        )
        assert result.rental_cost == 6

    def test_debuffed_rental_not_charged(self):
        j = _rental_joker()
        j.debuff = True
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state={},
        )
        assert result.rental_cost == 0

    def test_non_rental_joker_no_charge(self):
        j = _joker("j_golden", extra=4)
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state={},
        )
        assert result.rental_cost == 0

    def test_rental_reduces_effective_money_for_interest(self):
        """$10 - $3 rental = $7 effective; 7//5=1 bracket → $1 interest."""
        j = _rental_joker()
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=10,
            jokers=[j],
            game_state={},
        )
        assert result.rental_cost == 3
        assert result.interest == 1  # effective=7, 7//5=1

    def test_rental_below_interest_threshold(self):
        """$8 - $6 (2 rentals) = $2 effective; below $5 threshold → $0 interest."""
        j1 = _rental_joker()
        j2 = _rental_joker()
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=8,
            jokers=[j1, j2],
            game_state={},
        )
        assert result.rental_cost == 6
        assert result.interest == 0

    def test_custom_rental_rate(self):
        j = _rental_joker()
        gs = {"rental_rate": 5}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state=gs,
        )
        assert result.rental_cost == 5


# ---------------------------------------------------------------------------
# Joker dollar bonuses
# ---------------------------------------------------------------------------

class TestJokerDollars:
    """Joker end-of-round dollar bonuses via on_end_of_round."""

    def test_golden_joker_four_dollars(self):
        j = _joker("j_golden", extra=4)
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state={},
        )
        assert result.joker_dollars == 4

    def test_cloud_9_three_nines(self):
        j = _joker("j_cloud_9", extra=1, nine_tally=3)
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state={},
        )
        assert result.joker_dollars == 3

    def test_no_jokers_zero_dollars(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[],
            game_state={},
        )
        assert result.joker_dollars == 0

    def test_joker_dollars_not_used_for_interest(self):
        """Joker dollars are awarded after interest; they don't affect interest."""
        j = _joker("j_golden", extra=4)
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state={},
        )
        # money=0 → interest=0 regardless of joker payout
        assert result.interest == 0
        assert result.joker_dollars == 4


# ---------------------------------------------------------------------------
# Combined scenario
# ---------------------------------------------------------------------------

class TestCombinedScenario:
    """Full scenario: blind + hands + interest + Golden Joker + Cloud 9."""

    def test_golden_joker_and_cloud_9(self):
        """$20 bank, 1 hand left, Golden Joker + Cloud 9 (3 nines).

        blind=3, hands=1, interest=min(20//5=4, 5)×1=4, jokers=4+3=7
        total = 3 + 1 + 4 + 7 = 15
        """
        j_golden = _joker("j_golden", extra=4)
        j_cloud = _joker("j_cloud_9", extra=1, nine_tally=3)
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=1,
            discards_left=0,
            money=20,
            jokers=[j_golden, j_cloud],
            game_state={},
        )
        assert result.blind_reward == 3
        assert result.unused_hands_bonus == 1
        assert result.interest == 4
        assert result.joker_dollars == 7
        assert result.rental_cost == 0
        assert result.total == 15

    def test_rental_joker_reduces_interest_in_combined(self):
        """$15 bank, 1 rental joker, Golden Joker.

        rental=3, effective=12, interest=12//5=2, jokers=4, blind=3
        total = 3 + 2 + 4 - 3 = 6
        """
        j_rental = _rental_joker()
        j_golden = _joker("j_golden", extra=4)
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=15,
            jokers=[j_rental, j_golden],
            game_state={},
        )
        assert result.rental_cost == 3
        assert result.interest == 2  # effective=12, 12//5=2
        assert result.joker_dollars == 4
        assert result.blind_reward == 3
        assert result.total == 6

    def test_green_deck_full(self):
        """Green Deck: 3 unused hands × $1 + 2 unused discards × $1 = $5 bonus."""
        gs = {"modifiers": {"money_per_discard": 1}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=3,
            discards_left=2,
            money=0,
            jokers=[],
            game_state=gs,
        )
        assert result.unused_hands_bonus == 3
        assert result.unused_discards_bonus == 2
        assert result.blind_reward == 3
        assert result.total == 8

    def test_no_blind_reward_stake(self):
        """No-blind-reward stake: dollars=0 on the blind."""
        blind = Blind.create("bl_small", ante=1, no_blind_reward=True)
        result = calculate_round_earnings(
            blind=blind,
            hands_left=2,
            discards_left=0,
            money=10,
            jokers=[],
            game_state={},
        )
        assert result.blind_reward == 0
        # hands + interest only
        assert result.total == result.unused_hands_bonus + result.interest
