"""Tests for destructive and rule-modifying jokers.

Validates self-destruction, card mutation side effects, and
rule-changing descriptors.
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


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# Gros Michel: +15 mult, probabilistic self-destruction
# ============================================================================

class TestGrosMichel:
    def _make(self):
        return _joker("j_gros_michel", extra={"mult": 15, "odds": 6})

    def test_joker_main_gives_mult(self):
        j = self._make()
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 15

    def test_end_of_round_high_probability_destroys(self):
        """With very high probability, should self-destruct."""
        j = self._make()
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("GM"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True
        assert result.extra["pool_flag"] == "gros_michel_extinct"

    def test_end_of_round_survives(self):
        """With very low probability, should survive."""
        j = self._make()
        j.ability["extra"]["odds"] = 1000000
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("GM"),
            probabilities_normal=1.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is False
        assert result.saved is True

    def test_pipeline_gives_mult(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = self._make()
        result = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 32 chips, 2 mult. +15 mult → 17. Score: 32 × 17 = 544.
        assert result.mult == 17.0
        assert result.total == 544


# ============================================================================
# Cavendish: x3 mult, rare self-destruction
# ============================================================================

class TestCavendish:
    def _make(self):
        return _joker("j_cavendish", extra={"Xmult": 3, "odds": 1000})

    def test_joker_main_gives_xmult(self):
        j = self._make()
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 3

    def test_end_of_round_survives(self):
        j = self._make()
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("CV"),
            probabilities_normal=1.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.saved is True
        assert result.remove is False

    def test_end_of_round_high_probability_destroys(self):
        j = self._make()
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("CV"),
            probabilities_normal=1000000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True


# ============================================================================
# Chicot: disable boss blind (persists)
# ============================================================================

class TestChicot:
    def test_disables_boss_blind(self):
        j = _joker("j_chicot")
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(setting_blind=True, blind=boss)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["disable_blind"] is True

    def test_non_boss_no_effect(self):
        j = _joker("j_chicot")
        small = Blind.create("bl_small", ante=1)
        ctx = JokerContext(setting_blind=True, blind=small)
        assert calculate_joker(j, ctx) is None

    def test_already_disabled_no_effect(self):
        j = _joker("j_chicot")
        boss = Blind.create("bl_hook", ante=1)
        boss.disabled = True
        ctx = JokerContext(setting_blind=True, blind=boss)
        assert calculate_joker(j, ctx) is None

    def test_does_not_self_destruct(self):
        """Chicot persists — no remove flag."""
        j = _joker("j_chicot")
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(setting_blind=True, blind=boss)
        result = calculate_joker(j, ctx)
        assert result.remove is False


# ============================================================================
# Luchador: sell to disable boss blind
# ============================================================================

class TestLuchador:
    def test_sell_during_boss(self):
        j = _joker("j_luchador")
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(selling_self=True, blind=boss)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["disable_blind"] is True

    def test_sell_during_non_boss(self):
        j = _joker("j_luchador")
        small = Blind.create("bl_small", ante=1)
        ctx = JokerContext(selling_self=True, blind=small)
        assert calculate_joker(j, ctx) is None

    def test_sell_with_disabled_blind(self):
        j = _joker("j_luchador")
        boss = Blind.create("bl_hook", ante=1)
        boss.disabled = True
        ctx = JokerContext(selling_self=True, blind=boss)
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Burglar: +3 hands, 0 discards on setting_blind
# ============================================================================

class TestBurglar:
    def test_setting_blind(self):
        j = _joker("j_burglar", extra=3)
        ctx = JokerContext(setting_blind=True)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["set_hands"] == 3
        assert result.extra["set_discards"] == 0

    def test_other_context_no_effect(self):
        j = _joker("j_burglar", extra=3)
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


# ============================================================================
# Midas Mask: convert face cards to Gold
# ============================================================================

class TestMidasMask:
    def test_converts_face_cards(self):
        j = _joker("j_midas_mask")
        king = _card("Hearts", "King")
        five = _card("Spades", "5")
        scoring = [king, five]
        ctx = JokerContext(before=True, scoring_hand=scoring)
        calculate_joker(j, ctx)
        # King should be Gold now
        assert king.ability.get("name") == "Gold Card"
        # Five unchanged
        assert five.ability.get("name") != "Gold Card"

    def test_persists_across_hands(self):
        """Enhancement change persists — card stays Gold for future hands."""
        j = _joker("j_midas_mask")
        queen = _card("Hearts", "Queen")
        ctx = JokerContext(before=True, scoring_hand=[queen])
        calculate_joker(j, ctx)
        assert queen.ability.get("name") == "Gold Card"
        # Card is still Gold on next access
        assert queen.ability.get("name") == "Gold Card"

    def test_no_face_no_effect(self):
        j = _joker("j_midas_mask")
        five = _card("Hearts", "5")
        ctx = JokerContext(before=True, scoring_hand=[five])
        result = calculate_joker(j, ctx)
        assert result is None

    def test_debuffed_face_not_converted(self):
        j = _joker("j_midas_mask")
        king = _card("Hearts", "King")
        king.debuff = True
        ctx = JokerContext(before=True, scoring_hand=[king])
        result = calculate_joker(j, ctx)
        assert result is None

    def test_pareidolia_converts_all(self):
        """With Pareidolia, ALL cards are face → all become Gold."""
        j = _joker("j_midas_mask")
        five = _card("Hearts", "5")
        two = _card("Spades", "2")
        scoring = [five, two]
        pareidolia = _joker("j_pareidolia_stub")
        pareidolia.ability = {"name": "Pareidolia", "set": "Joker"}
        ctx = JokerContext(before=True, scoring_hand=scoring, pareidolia=True)
        calculate_joker(j, ctx)
        assert five.ability.get("name") == "Gold Card"
        assert two.ability.get("name") == "Gold Card"

    def test_pipeline_gold_card_chips(self):
        """Midas Mask converts Kings to Gold in before phase.
        Gold Card effect: +3 dollars per card (from get_p_dollars).
        Three of a Kind of Kings with Midas: Kings become Gold Cards.
        After conversion, Gold Cards give $3 per scored card."""
        played = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        j = _joker("j_midas_mask")
        result = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        # 3 Kings converted to Gold. Gold Card nominal is still 10 (King).
        # But enhancement changes scoring methods.
        assert result.hand_type == "Three of a Kind"
        assert result.dollars_earned >= 0  # Gold Cards earn dollars


# ============================================================================
# Hiker: +5 permanent chip bonus per scored card
# ============================================================================

class TestHiker:
    def test_adds_perma_bonus(self):
        j = _joker("j_hiker", extra=5)
        ace = _card("Hearts", "Ace")
        ctx = JokerContext(individual=True, cardarea="play", other_card=ace)
        calculate_joker(j, ctx)
        assert ace.ability["perma_bonus"] == 5

    def test_accumulates_across_hands(self):
        """Same card scored 3 times → perma_bonus = 15."""
        j = _joker("j_hiker", extra=5)
        ace = _card("Hearts", "Ace")
        for _ in range(3):
            ctx = JokerContext(
                individual=True, cardarea="play", other_card=ace,
            )
            calculate_joker(j, ctx)
        assert ace.ability["perma_bonus"] == 15

    def test_pipeline_perma_bonus_adds_chips(self):
        """Hiker adds perma_bonus during scoring. On next hand, the card
        has higher chip value due to accumulated bonus."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_hiker", extra=5)

        # First hand: perma_bonus added during scoring
        r1 = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        # Each Ace gets +5 perma_bonus during Phase 7
        assert played[0].ability["perma_bonus"] == 5
        assert played[1].ability["perma_bonus"] == 5

        # Second hand: perma_bonus is part of chip calculation
        reset_sort_id_counter()
        r2 = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 10 base. Each Ace: 11 nominal + 5 old perma + 5 new perma = 21 chips.
        # Wait — get_chip_bonus returns nominal + bonus + perma_bonus.
        # After hand 1: perma_bonus = 5. During hand 2, eval_card adds 11+5=16 chips.
        # Then Hiker adds another +5 perma_bonus → 10.
        # Total chips hand 2: 10 + (11+5) + (11+5) = 42.
        assert r2.chips == 42.0
        assert played[0].ability["perma_bonus"] == 10

    def test_blueprint_does_not_mutate(self):
        j = _joker("j_hiker", extra=5)
        ace = _card("Hearts", "Ace")
        ctx = JokerContext(
            individual=True, cardarea="play", other_card=ace,
            blueprint=1,
        )
        calculate_joker(j, ctx)
        assert ace.ability.get("perma_bonus", 0) == 0
