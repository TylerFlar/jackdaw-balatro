"""Tests for end-of-round economy jokers and lifecycle hooks.

Validates calc_dollar_bonus, end-of-round mutations, and
the on_end_of_round orchestration function.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.jokers import (
    GameSnapshot,
    JokerContext,
    calc_dollar_bonus,
    calculate_joker,
    on_end_of_round,
)
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    c.sell_cost = 1
    return c


_SL = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RL = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "Jack": "J",
    "Queen": "Q",
    "King": "K",
    "Ace": "A",
}


def _card(suit: str, rank: str) -> Card:
    c = Card()
    c.set_base(f"{_SL[suit]}_{_RL[rank]}", suit, rank)
    c.set_ability("c_base")
    return c


# ============================================================================
# Golden Joker: +$4 per round
# ============================================================================


class TestGoldenJoker:
    def test_always_four_dollars(self):
        j = _joker("j_golden", extra=4)
        game = GameSnapshot()
        assert calc_dollar_bonus(j, game) == 4

    def test_debuffed_no_dollars(self):
        j = _joker("j_golden", extra=4)
        j.debuff = True
        assert calc_dollar_bonus(j, GameSnapshot()) == 0

    def test_via_on_end_of_round(self):
        j = _joker("j_golden", extra=4)
        result = on_end_of_round([j], GameSnapshot())
        assert result["dollars_earned"] == 4


# ============================================================================
# Cloud 9: +$1 per 9-rank card in full deck
# ============================================================================


class TestCloud9:
    def test_three_nines(self):
        j = _joker("j_cloud_9", extra=1, nine_tally=3)
        assert calc_dollar_bonus(j, GameSnapshot()) == 3

    def test_zero_nines(self):
        j = _joker("j_cloud_9", extra=1, nine_tally=0)
        assert calc_dollar_bonus(j, GameSnapshot()) == 0

    def test_no_tally_field(self):
        j = _joker("j_cloud_9", extra=1)
        assert calc_dollar_bonus(j, GameSnapshot()) == 0


# ============================================================================
# Rocket: +$1 base, +$2 after each boss beaten
# ============================================================================


class TestRocket:
    def test_base_dollars(self):
        j = _joker("j_rocket", extra={"dollars": 1, "increase": 2})
        assert calc_dollar_bonus(j, GameSnapshot()) == 1

    def test_after_boss_increases(self):
        j = _joker("j_rocket", extra={"dollars": 1, "increase": 2})
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(end_of_round=True, blind=boss)
        calculate_joker(j, ctx)
        # dollars should now be 3 (1 + 2)
        assert j.ability["extra"]["dollars"] == 3
        assert calc_dollar_bonus(j, GameSnapshot()) == 3

    def test_after_two_bosses(self):
        j = _joker("j_rocket", extra={"dollars": 1, "increase": 2})
        boss = Blind.create("bl_hook", ante=1)
        for _ in range(2):
            calculate_joker(j, JokerContext(end_of_round=True, blind=boss))
        # 1 + 2 + 2 = 5
        assert calc_dollar_bonus(j, GameSnapshot()) == 5

    def test_non_boss_no_increase(self):
        j = _joker("j_rocket", extra={"dollars": 1, "increase": 2})
        small = Blind.create("bl_small", ante=1)
        calculate_joker(j, JokerContext(end_of_round=True, blind=small))
        assert calc_dollar_bonus(j, GameSnapshot()) == 1


# ============================================================================
# Satellite: +$1 per unique Planet type used
# ============================================================================


class TestSatellite:
    def test_three_planet_types(self):
        j = _joker("j_satellite", extra=1, planet_types_used=3)
        assert calc_dollar_bonus(j, GameSnapshot()) == 3

    def test_no_planets(self):
        j = _joker("j_satellite", extra=1, planet_types_used=0)
        assert calc_dollar_bonus(j, GameSnapshot()) == 0


# ============================================================================
# Delayed Gratification: +$2 per discard remaining if none used
# ============================================================================


class TestDelayedGratification:
    def test_zero_used_three_remaining(self):
        j = _joker("j_delayed_grat", extra=2)
        game = GameSnapshot(discards_used=0, discards_left=3)
        assert calc_dollar_bonus(j, game) == 6

    def test_one_used_no_effect(self):
        j = _joker("j_delayed_grat", extra=2)
        game = GameSnapshot(discards_used=1, discards_left=2)
        assert calc_dollar_bonus(j, game) == 0

    def test_zero_discards_left_no_effect(self):
        j = _joker("j_delayed_grat", extra=2)
        game = GameSnapshot(discards_used=0, discards_left=0)
        assert calc_dollar_bonus(j, game) == 0

    def test_via_on_end_of_round(self):
        j = _joker("j_delayed_grat", extra=2)
        game = GameSnapshot(discards_used=0, discards_left=4)
        result = on_end_of_round([j], game)
        assert result["dollars_earned"] == 8


# ============================================================================
# Egg: +$3 sell value per round
# ============================================================================


class TestEgg:
    def test_sell_value_increases(self):
        j = _joker("j_egg", extra=3)
        j.sell_cost = 2
        ctx = JokerContext(end_of_round=True)
        calculate_joker(j, ctx)
        assert j.ability["extra_value"] == 3
        assert j.sell_cost == 5  # 2 + 3

    def test_accumulates(self):
        j = _joker("j_egg", extra=3)
        j.sell_cost = 2
        for _ in range(3):
            calculate_joker(j, JokerContext(end_of_round=True))
        assert j.ability["extra_value"] == 9
        assert j.sell_cost == 11  # 2 + 3*3


# ============================================================================
# Gift Card: +$1 sell value to ALL jokers
# ============================================================================


class TestGiftCard:
    def test_all_jokers_increase(self):
        gift = _joker("j_gift", extra=1)
        j1 = _joker("j_joker", mult=4)
        j2 = _joker("j_stuntman", extra={"chip_mod": 250})
        jokers = [gift, j1, j2]
        for j in jokers:
            j.sell_cost = 2

        ctx = JokerContext(end_of_round=True, jokers=jokers)
        calculate_joker(gift, ctx)

        for j in jokers:
            assert j.ability.get("extra_value", 0) == 1
            assert j.sell_cost == 3

    def test_accumulates_across_rounds(self):
        gift = _joker("j_gift", extra=1)
        j1 = _joker("j_joker", mult=4)
        jokers = [gift, j1]
        for j in jokers:
            j.sell_cost = 1

        for _ in range(3):
            calculate_joker(gift, JokerContext(end_of_round=True, jokers=jokers))

        assert j1.ability["extra_value"] == 3
        assert j1.sell_cost == 4


# ============================================================================
# Invisible Joker: counts rounds, duplicates on sell
# ============================================================================


class TestInvisible:
    def test_counts_rounds(self):
        j = _joker("j_invisible", extra=2, invis_rounds=0)
        calculate_joker(j, JokerContext(end_of_round=True))
        assert j.ability["invis_rounds"] == 1

    def test_sell_before_threshold_no_effect(self):
        j = _joker("j_invisible", extra=2, invis_rounds=1)
        result = calculate_joker(j, JokerContext(selling_self=True))
        assert result is None

    def test_sell_at_threshold_duplicates(self):
        j = _joker("j_invisible", extra=2, invis_rounds=2)
        result = calculate_joker(j, JokerContext(selling_self=True))
        assert result is not None
        assert result.extra["duplicate_random_joker"] is True


# ============================================================================
# Diet Cola: sell → create Double Tag
# ============================================================================


class TestDietCola:
    def test_sell_creates_tag(self):
        j = _joker("j_diet_cola")
        result = calculate_joker(j, JokerContext(selling_self=True))
        assert result is not None
        assert result.extra["create"]["type"] == "Tag"
        assert result.extra["create"]["key"] == "tag_double"


# ============================================================================
# Space Joker: 1/4 chance to level up hand
# ============================================================================


class TestSpaceJoker:
    def test_high_probability_levels_up(self):
        j = _joker("j_space", extra=4)
        ctx = JokerContext(
            before=True,
            rng=PseudoRandom("SP"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.level_up is True

    def test_no_rng_no_effect(self):
        j = _joker("j_space", extra=4)
        ctx = JokerContext(before=True)
        assert calculate_joker(j, ctx) is None

    def test_other_context_no_effect(self):
        j = _joker("j_space", extra=4)
        ctx = JokerContext(
            joker_main=True,
            rng=PseudoRandom("SP"),
            probabilities_normal=1000.0,
        )
        assert calculate_joker(j, ctx) is None


# ============================================================================
# on_end_of_round: orchestration
# ============================================================================


class TestOnEndOfRound:
    def test_multiple_dollar_jokers(self):
        golden = _joker("j_golden", extra=4)
        cloud = _joker("j_cloud_9", extra=1, nine_tally=5)
        game = GameSnapshot()
        result = on_end_of_round([golden, cloud], game)
        assert result["dollars_earned"] == 9  # 4 + 5

    def test_self_destruct_collected(self):
        gros = _joker("j_gros_michel", extra={"mult": 15, "odds": 6})
        game = GameSnapshot()
        result = on_end_of_round(
            [gros],
            game,
            rng=PseudoRandom("GM"),
        )
        # Either removed or saved — verify structure
        if gros in result["jokers_removed"]:
            assert len(result["jokers_removed"]) == 1
        else:
            assert len(result["jokers_removed"]) == 0

    def test_egg_mutation_in_eor(self):
        egg = _joker("j_egg", extra=3)
        egg.sell_cost = 2
        on_end_of_round([egg], GameSnapshot())
        assert egg.ability["extra_value"] == 3
