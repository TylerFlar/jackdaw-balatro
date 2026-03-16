"""Tests for xMult scaling jokers that accumulate multiplicative mult.

Validates multi-hand/multi-round sequences, reset conditions,
side effects, and self-destruction.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import JokerContext, calculate_joker


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_SL = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RL = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "10": "T", "Jack": "J", "Queen": "Q",
    "King": "K", "Ace": "A",
}


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    c = Card()
    c.set_base(f"{_SL[suit]}_{_RL[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    return c


# ============================================================================
# Campfire: +0.25 xMult per card sold, reset on boss
# ============================================================================

class TestCampfire:
    def _make(self):
        return _joker("j_campfire", x_mult=1, extra=0.25)

    def test_sell_increments(self):
        j = self._make()
        ctx = JokerContext(selling_card=True)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.25)

    def test_three_sells(self):
        j = self._make()
        for _ in range(3):
            calculate_joker(j, JokerContext(selling_card=True))
        assert j.ability["x_mult"] == pytest.approx(1.75)

    def test_boss_resets(self):
        j = self._make()
        for _ in range(4):
            calculate_joker(j, JokerContext(selling_card=True))
        assert j.ability["x_mult"] == pytest.approx(2.0)

        boss = Blind.create("bl_hook", ante=1)
        calculate_joker(j, JokerContext(end_of_round=True, blind=boss))
        assert j.ability["x_mult"] == 1

    def test_non_boss_no_reset(self):
        j = self._make()
        for _ in range(2):
            calculate_joker(j, JokerContext(selling_card=True))

        small = Blind.create("bl_small", ante=1)
        calculate_joker(j, JokerContext(end_of_round=True, blind=small))
        assert j.ability["x_mult"] == pytest.approx(1.5)

    def test_joker_main_returns(self):
        j = self._make()
        calculate_joker(j, JokerContext(selling_card=True))
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == pytest.approx(1.25)

    def test_no_sells_no_effect(self):
        j = self._make()
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


# ============================================================================
# Hologram: +0.25 xMult per card added to deck
# ============================================================================

class TestHologram:
    def _make(self):
        return _joker("j_hologram", x_mult=1, extra=0.25)

    def test_add_one_card(self):
        j = self._make()
        ctx = JokerContext(playing_card_added=True, cards=[_card("Hearts", "5")])
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.25)

    def test_add_three_cards(self):
        j = self._make()
        cards = [_card("Hearts", "5"), _card("Spades", "3"), _card("Clubs", "King")]
        ctx = JokerContext(playing_card_added=True, cards=cards)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.75)


# ============================================================================
# Constellation: +0.1 xMult per Planet card used
# ============================================================================

class TestConstellation:
    def _make(self):
        return _joker("j_constellation", x_mult=1, extra=0.1)

    def _planet(self) -> Card:
        c = Card()
        c.ability = {"set": "Planet", "name": "Jupiter"}
        return c

    def _tarot(self) -> Card:
        c = Card()
        c.ability = {"set": "Tarot", "name": "The Fool"}
        return c

    def test_planet_used(self):
        j = self._make()
        ctx = JokerContext(using_consumeable=True, consumeable=self._planet())
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.1)

    def test_tarot_no_effect(self):
        j = self._make()
        ctx = JokerContext(using_consumeable=True, consumeable=self._tarot())
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1

    def test_three_planets(self):
        j = self._make()
        for _ in range(3):
            ctx = JokerContext(using_consumeable=True, consumeable=self._planet())
            calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.3)


# ============================================================================
# Glass Joker: +0.75 xMult per Glass Card destroyed
# ============================================================================

class TestGlassJoker:
    def _make(self):
        return _joker("j_glass", x_mult=1, extra=0.75)

    def test_one_glass_destroyed(self):
        j = self._make()
        glass = _card("Hearts", "5", enhancement="m_glass")
        ctx = JokerContext(cards_destroyed=[glass])
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.75)

    def test_two_glass_destroyed(self):
        j = self._make()
        glass1 = _card("Hearts", "5", enhancement="m_glass")
        glass2 = _card("Spades", "3", enhancement="m_glass")
        ctx = JokerContext(cards_destroyed=[glass1, glass2])
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(2.5)

    def test_non_glass_destroyed_no_effect(self):
        j = self._make()
        normal = _card("Hearts", "5")
        ctx = JokerContext(cards_destroyed=[normal])
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1


# ============================================================================
# Caino: +1.0 xMult per face card destroyed
# ============================================================================

class TestCaino:
    def _make(self):
        return _joker("j_caino", caino_xmult=1, extra=1)

    def test_face_destroyed(self):
        j = self._make()
        king = _card("Hearts", "King")
        ctx = JokerContext(cards_destroyed=[king])
        calculate_joker(j, ctx)
        assert j.ability["caino_xmult"] == 2

    def test_two_faces_destroyed(self):
        j = self._make()
        king = _card("Hearts", "King")
        queen = _card("Spades", "Queen")
        ctx = JokerContext(cards_destroyed=[king, queen])
        calculate_joker(j, ctx)
        assert j.ability["caino_xmult"] == 3

    def test_non_face_no_effect(self):
        j = self._make()
        five = _card("Hearts", "5")
        ctx = JokerContext(cards_destroyed=[five])
        calculate_joker(j, ctx)
        assert j.ability["caino_xmult"] == 1

    def test_joker_main_returns(self):
        j = self._make()
        king = _card("Hearts", "King")
        calculate_joker(j, JokerContext(cards_destroyed=[king]))
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 2

    def test_caino_and_glass_both_trigger_on_glass_face(self):
        """A Glass King is both a face card AND a Glass Card."""
        caino = self._make()
        glass_j = _joker("j_glass", x_mult=1, extra=0.75)
        glass_king = _card("Hearts", "King", enhancement="m_glass")
        ctx = JokerContext(cards_destroyed=[glass_king])

        calculate_joker(caino, ctx)
        calculate_joker(glass_j, ctx)

        assert caino.ability["caino_xmult"] == 2  # face card
        assert glass_j.ability["x_mult"] == pytest.approx(1.75)  # glass card


# ============================================================================
# Vampire: +0.1 xMult per enhancement stripped
# ============================================================================

class TestVampire:
    def _make(self):
        return _joker("j_vampire", x_mult=1, extra=0.1)

    def test_strips_one_enhancement(self):
        j = self._make()
        bonus = _card("Hearts", "5", enhancement="m_bonus")
        ctx = JokerContext(
            individual_hand_end=True, scoring_hand=[bonus],
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.1)
        assert bonus.ability.get("name") == "Default Base"

    def test_strips_two_enhancements(self):
        j = self._make()
        bonus = _card("Hearts", "5", enhancement="m_bonus")
        mult_c = _card("Spades", "3", enhancement="m_mult")
        ctx = JokerContext(
            individual_hand_end=True, scoring_hand=[bonus, mult_c],
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.2)

    def test_base_card_no_effect(self):
        j = self._make()
        base = _card("Hearts", "5")
        ctx = JokerContext(
            individual_hand_end=True, scoring_hand=[base],
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1

    def test_accumulates_across_hands(self):
        """3 hands, each with 1 enhanced card → x_mult = 1.3."""
        j = self._make()
        for _ in range(3):
            enhanced = _card("Hearts", "5", enhancement="m_bonus")
            ctx = JokerContext(
                individual_hand_end=True, scoring_hand=[enhanced],
            )
            calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.3)

    def test_debuffed_card_not_stripped(self):
        j = self._make()
        bonus = _card("Hearts", "5", enhancement="m_bonus")
        bonus.debuff = True
        ctx = JokerContext(
            individual_hand_end=True, scoring_hand=[bonus],
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1


# ============================================================================
# Obelisk: +0.2 xMult per non-most-played hand, reset on most-played
# ============================================================================

class TestObelisk:
    def _make(self):
        return _joker("j_obelisk", x_mult=1, extra=0.2)

    def test_non_most_played_increments(self):
        j = self._make()
        levels = HandLevels()
        levels.record_play("Pair")
        levels.record_play("Pair")
        levels.record_play("Pair")
        levels.record_play("Flush")  # Flush=1, Pair=3 → Flush is NOT most
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_name="Flush",
            hand_levels=levels,
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.2)

    def test_most_played_resets(self):
        j = self._make()
        j.ability["x_mult"] = 2.0  # accumulated
        levels = HandLevels()
        levels.record_play("Pair")
        levels.record_play("Pair")
        # Pair is most played (2), playing Pair → reset
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_name="Pair",
            hand_levels=levels,
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1

    def test_sequence_accumulate_then_reset(self):
        j = self._make()
        levels = HandLevels()
        # Play Pair 3 times (most played)
        for _ in range(3):
            levels.record_play("Pair")

        # Play Flush twice → not most played → +0.2 each
        for _ in range(2):
            levels.record_play("Flush")
            ctx = JokerContext(
                individual_hand_end=True,
                scoring_name="Flush",
                hand_levels=levels,
            )
            calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.4)

        # Play Pair (still most played) → resets
        levels.record_play("Pair")
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_name="Pair",
            hand_levels=levels,
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1


# ============================================================================
# Madness: +0.5 xMult per non-boss blind, signals joker destruction
# ============================================================================

class TestMadness:
    def _make(self):
        return _joker("j_madness", x_mult=1, extra=0.5)

    def test_non_boss_increments(self):
        j = self._make()
        small = Blind.create("bl_small", ante=1)
        ctx = JokerContext(setting_blind=True, blind=small)
        result = calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.5)
        assert result is not None
        assert result.extra == {"destroy_random_joker": True}

    def test_boss_no_effect(self):
        j = self._make()
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(setting_blind=True, blind=boss)
        result = calculate_joker(j, ctx)
        assert result is None
        assert j.ability["x_mult"] == 1


# ============================================================================
# Hit the Road: +0.5 xMult per Jack discarded, reset at end of round
# ============================================================================

class TestHitTheRoad:
    def _make(self):
        return _joker("j_hit_the_road", x_mult=1, extra=0.5)

    def test_jack_discarded(self):
        j = self._make()
        jack = _card("Hearts", "Jack")
        ctx = JokerContext(discard=True, other_card=jack)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.5)

    def test_non_jack_no_effect(self):
        j = self._make()
        king = _card("Hearts", "King")
        ctx = JokerContext(discard=True, other_card=king)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1

    def test_three_jacks_then_reset(self):
        j = self._make()
        for _ in range(3):
            jack = _card("Hearts", "Jack")
            calculate_joker(j, JokerContext(discard=True, other_card=jack))
        assert j.ability["x_mult"] == pytest.approx(2.5)

        # End of round resets
        calculate_joker(j, JokerContext(end_of_round=True))
        assert j.ability["x_mult"] == 1


# ============================================================================
# Throwback: xMult = 1 + 0.25 * total_blinds_skipped (formula-based)
# ============================================================================

class TestThrowback:
    def test_zero_skips(self):
        j = _joker("j_throwback", extra=0.25)
        ctx = JokerContext(joker_main=True, skips=0)
        assert calculate_joker(j, ctx) is None

    def test_four_skips(self):
        j = _joker("j_throwback", extra=0.25)
        ctx = JokerContext(joker_main=True, skips=4)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.Xmult_mod == pytest.approx(2.0)

    def test_ten_skips(self):
        j = _joker("j_throwback", extra=0.25)
        ctx = JokerContext(joker_main=True, skips=10)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.Xmult_mod == pytest.approx(3.5)


# ============================================================================
# Yorick: xMult +1 every 23 discards
# ============================================================================

class TestYorick:
    def _make(self):
        return _joker(
            "j_yorick", x_mult=1, yorick_discards=23,
            extra={"xmult": 1, "discards": 23},
        )

    def test_22_discards_no_trigger(self):
        j = self._make()
        for _ in range(22):
            calculate_joker(j, JokerContext(discard=True))
        assert j.ability["x_mult"] == 1
        assert j.ability["yorick_discards"] == 1

    def test_23rd_discard_triggers(self):
        j = self._make()
        for _ in range(23):
            calculate_joker(j, JokerContext(discard=True))
        assert j.ability["x_mult"] == 2
        assert j.ability["yorick_discards"] == 23  # reset

    def test_46_discards_triggers_twice(self):
        j = self._make()
        for _ in range(46):
            calculate_joker(j, JokerContext(discard=True))
        assert j.ability["x_mult"] == 3

    def test_joker_main_returns(self):
        j = self._make()
        for _ in range(23):
            calculate_joker(j, JokerContext(discard=True))
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 2


# ============================================================================
# Ceremonial Dagger: +2× right neighbor's sell_cost as mult
# ============================================================================

class TestCeremonialDagger:
    def _make(self):
        return _joker("j_ceremonial", mult=0)

    def test_destroys_right_neighbor(self):
        dagger = self._make()
        target = _joker("j_joker", mult=4)
        target.sell_cost = 3
        jokers = [dagger, target]
        ctx = JokerContext(
            setting_blind=True, jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        result = calculate_joker(dagger, ctx)
        assert dagger.ability["mult"] == 6  # 3 * 2
        assert result is not None
        assert result.extra["destroy_joker"] is target

    def test_no_right_neighbor(self):
        dagger = self._make()
        jokers = [dagger]
        ctx = JokerContext(
            setting_blind=True, jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        assert calculate_joker(dagger, ctx) is None

    def test_eternal_not_destroyed(self):
        dagger = self._make()
        target = _joker("j_joker", mult=4)
        target.sell_cost = 3
        target.eternal = True
        jokers = [dagger, target]
        ctx = JokerContext(
            setting_blind=True, jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        assert calculate_joker(dagger, ctx) is None

    def test_accumulates_across_rounds(self):
        dagger = self._make()
        t1 = _joker("j_joker", mult=4)
        t1.sell_cost = 2
        jokers = [dagger, t1]
        ctx = JokerContext(
            setting_blind=True, jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        calculate_joker(dagger, ctx)
        assert dagger.ability["mult"] == 4  # 2 * 2

        # Next round, new neighbor
        t2 = _joker("j_stuntman")
        t2.sell_cost = 5
        jokers2 = [dagger, t2]
        ctx2 = JokerContext(
            setting_blind=True, jokers=jokers2,
            blind=Blind.create("bl_small", ante=1),
        )
        calculate_joker(dagger, ctx2)
        assert dagger.ability["mult"] == 14  # 4 + 5*2

    def test_joker_main_returns_mult(self):
        dagger = self._make()
        dagger.ability["mult"] = 10
        result = calculate_joker(dagger, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 10  # additive, NOT xMult
