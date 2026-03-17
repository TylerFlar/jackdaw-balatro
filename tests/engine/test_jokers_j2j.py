"""Tests for joker-on-joker effects (Phase 9c) and sell-value jokers.

Validates Baseball Card's Uncommon rarity check through both direct
handler calls and the full scoring pipeline.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import JokerContext, calculate_joker
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


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


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    return c


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# Baseball Card: x1.5 mult if other_joker is Uncommon (rarity 2)
# ============================================================================


class TestBaseballCard:
    """j_baseball fires in other_joker context for Uncommon jokers."""

    def test_uncommon_joker_triggers(self):
        """j_steel_joker is rarity 2 (Uncommon) → x1.5."""
        baseball = _joker("j_baseball", extra=1.5)
        uncommon = _joker("j_steel_joker", extra=0.2)
        ctx = JokerContext(other_joker=uncommon)
        result = calculate_joker(baseball, ctx)
        assert result is not None
        assert result.Xmult_mod == 1.5

    def test_common_joker_no_effect(self):
        """j_joker is rarity 1 (Common) → no effect."""
        baseball = _joker("j_baseball", extra=1.5)
        common = _joker("j_joker", mult=4)
        ctx = JokerContext(other_joker=common)
        result = calculate_joker(baseball, ctx)
        assert result is None

    def test_rare_joker_no_effect(self):
        """j_blueprint is rarity 3 (Rare) → no effect."""
        baseball = _joker("j_baseball", extra=1.5)
        rare = _joker("j_blueprint")
        ctx = JokerContext(other_joker=rare)
        assert calculate_joker(baseball, ctx) is None

    def test_does_not_trigger_on_self(self):
        """Baseball Card should not trigger on itself (even if Rare)."""
        baseball = _joker("j_baseball", extra=1.5)
        ctx = JokerContext(other_joker=baseball)
        assert calculate_joker(baseball, ctx) is None

    def test_pipeline_three_uncommon_jokers(self):
        """Three Uncommon jokers + Baseball Card → x1.5 fires 3 times.

        Pair of Aces: 32 chips, 2 mult.
        Phase 9 processing order (left to right):
        - j_blackboard (Uncommon, rarity 2): x3 mult → 6
          - 9c: Baseball reacts to Blackboard → x1.5 → 9
        - j_bull (Uncommon): +0 chips (money=0)
          - 9c: Baseball reacts to Bull → x1.5 → 13.5
        - j_card_sharp (Uncommon): no effect (played_this_round=0)
          - 9c: Baseball reacts to Card Sharp → x1.5 → 20.25
        - j_baseball (Rare): no self-trigger
          - 9c: no joker reacts (none check other_joker)

        Score: 32 × 20.25 = 648."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]  # all black for Blackboard

        blackboard = _joker("j_blackboard", extra=3)  # rarity 2
        bull = _joker("j_bull", extra=2)  # rarity 2
        card_sharp = _joker("j_card_sharp", extra={"Xmult": 3})  # rarity 2
        baseball = _joker("j_baseball", extra=1.5)  # rarity 3

        result = score_hand(
            played,
            held,
            [blackboard, bull, card_sharp, baseball],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Base: 32 chips, 2 mult
        # Blackboard x3 → 6, Baseball x1.5 → 9
        # Bull +0 (no money), Baseball x1.5 → 13.5
        # Card Sharp no effect, Baseball x1.5 → 20.25
        # Baseball: no self-react
        assert result.mult == pytest.approx(20.25)
        assert result.total == 648

    def test_pipeline_no_uncommon_no_effect(self):
        """All Common jokers → Baseball never fires in 9c."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        joker = _joker("j_joker", mult=4)  # rarity 1
        baseball = _joker("j_baseball", extra=1.5)  # rarity 3

        result = score_hand(
            played,
            [],
            [joker, baseball],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Base: 32, 2. j_joker +4 → 6. Baseball: no react to Common j_joker.
        assert result.mult == 6.0
        assert result.total == 192

    def test_pipeline_blueprint_not_uncommon(self):
        """Blueprint (Rare) copying j_joker: Baseball doesn't react to Blueprint."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bp = _joker("j_blueprint")  # rarity 3
        joker = _joker("j_joker", mult=4)  # rarity 1
        baseball = _joker("j_baseball", extra=1.5)  # rarity 3

        result = score_hand(
            played,
            [],
            [bp, joker, baseball],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # bp copies joker → +4 mult. joker → +4 mult. Total add: +8.
        # Baseball: bp is Rare (no), joker is Common (no), self (skip).
        # mult = 2 + 4 + 4 = 10
        assert result.mult == 10.0
        assert result.total == 320


# ============================================================================
# Swashbuckler: +mult = sum of all other jokers' sell values
# ============================================================================


class TestSwashbuckler:
    """j_swashbuckler: mult from other jokers' sell costs."""

    def test_basic_sell_sum(self):
        swash = _joker("j_swashbuckler")
        j1 = _joker("j_joker", mult=4)
        j1.sell_cost = 2
        j2 = _joker("j_stuntman", extra={"chip_mod": 250})
        j2.sell_cost = 4
        jokers = [swash, j1, j2]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        result = calculate_joker(swash, ctx)
        assert result is not None
        assert result.mult_mod == 6  # 2 + 4

    def test_excludes_self(self):
        swash = _joker("j_swashbuckler")
        swash.sell_cost = 3
        jokers = [swash]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(swash, ctx) is None

    def test_excludes_debuffed(self):
        swash = _joker("j_swashbuckler")
        j1 = _joker("j_joker", mult=4)
        j1.sell_cost = 5
        j1.debuff = True
        jokers = [swash, j1]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(swash, ctx) is None

    def test_pipeline_with_other_jokers(self):
        """Swashbuckler in pipeline adds sell value sum as mult."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        swash = _joker("j_swashbuckler")
        j1 = _joker("j_joker", mult=4)
        j1.sell_cost = 2
        j2 = _joker("j_stuntman", extra={"chip_mod": 250, "h_size": 2})
        j2.sell_cost = 4

        result = score_hand(
            played,
            [],
            [j1, swash, j2],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Base: 32 chips, 2 mult
        # j_joker: +4 mult → 6
        # swash: +6 mult (sell 2+4) → 12
        # j_stuntman: +250 chips → 282
        assert result.mult == 12.0
        assert result.chips == 282.0
        assert result.total == 3384
