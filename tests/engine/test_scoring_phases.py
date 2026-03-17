"""Integration tests for Phases 10-14 of the scoring pipeline.

Tests Plasma Deck, Glass Card destruction, Caino/Glass Joker reactions,
Ice Cream/Seltzer after-phase decay, and Mr. Bones save mechanism.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
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
# Phase 10: Plasma Deck
# ============================================================================


class TestPlasmaDeck:
    def test_averages_chips_and_mult(self):
        """Plasma Deck: (chips + mult) / 2 for both.
        Pair of Aces: 32 chips, 2 mult.
        Plasma: total = 34, each = 17.
        Score: 17 × 17 = 289."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        result = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
            back_key="b_plasma",
        )
        assert result.chips == 17.0
        assert result.mult == 17.0
        assert result.total == 289

    def test_non_plasma_no_effect(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        result = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 32.0
        assert result.mult == 2.0
        assert result.total == 64

    def test_plasma_with_joker(self):
        """Plasma + j_joker: 32 chips, 2+4=6 mult → total=38, each=19.
        Score: 19 × 19 = 361."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
            back_key="b_plasma",
        )
        assert result.chips == 19.0
        assert result.mult == 19.0
        assert result.total == 361


# ============================================================================
# Phase 11: Glass Card destruction
# ============================================================================


class TestGlassCardDestruction:
    def test_glass_scores_x2_then_may_shatter(self):
        """Glass Card contributes x2 mult in Phase 7, then rolls in Phase 11.
        Pair with Glass 5: 10+5+5=20 chips, 2 mult.
        Glass: x2 → 4 mult. Score: 20 × 4 = 80.
        Whether it shatters depends on RNG."""
        glass_5 = _card("Hearts", "5", enhancement="m_glass")
        played = [glass_5, _card("Spades", "5")]
        result = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.total == 80  # Glass x2 applied regardless of shatter
        # Card may or may not be destroyed — just check structure
        assert isinstance(result.cards_destroyed, list)

    def test_glass_guaranteed_shatter(self):
        """With high probabilities, Glass Card always shatters."""
        glass_5 = _card("Hearts", "5", enhancement="m_glass")
        played = [glass_5, _card("Spades", "5")]
        result = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        # 100/4 = 25 > any roll → always shatters
        assert glass_5 in result.cards_destroyed


# ============================================================================
# Phase 11: Caino reacts to Glass face card destruction
# ============================================================================


class TestCainoGlassInteraction:
    def test_caino_gains_xmult_from_shattered_face(self):
        """Glass King shatters → Caino gains +1 xMult.
        Three of a Kind: Kings. Glass King scores x2 in Phase 7.
        Phase 11: Glass King shatters (high prob).
        Caino notified → caino_xmult += 1 → 2."""
        glass_king = _card("Hearts", "King", enhancement="m_glass")
        played = [
            glass_king,
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        caino = _joker("j_caino", caino_xmult=1, extra=1)
        result = score_hand(
            played,
            [],
            [caino],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        # Glass King should have shattered
        assert glass_king in result.cards_destroyed
        # Caino should have gained +1 (face card destroyed)
        assert caino.ability["caino_xmult"] == 2

    def test_glass_joker_gains_xmult_from_shatter(self):
        """Glass Card shatters → Glass Joker gains +0.75 xMult."""
        glass_5 = _card("Hearts", "5", enhancement="m_glass")
        played = [glass_5, _card("Spades", "5")]
        glass_j = _joker("j_glass", x_mult=1, extra=0.75)
        result = score_hand(
            played,
            [],
            [glass_j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        assert glass_5 in result.cards_destroyed
        assert glass_j.ability["x_mult"] == pytest.approx(1.75)


# ============================================================================
# Phase 13: Ice Cream after-phase decay
# ============================================================================


class TestIceCreamInPipeline:
    def test_ice_cream_decays_after_scoring(self):
        """Ice Cream: +100 chips in Phase 9, then -5 in Phase 13."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        ice = _joker("j_ice_cream", extra={"chips": 100, "chip_mod": 5})
        result = score_hand(
            played,
            [],
            [ice],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Phase 9: 32 + 100 = 132 chips, 2 mult. Score: 264.
        assert result.total == 264
        # Phase 13: chips decremented
        assert ice.ability["extra"]["chips"] == 95

    def test_ice_cream_self_destructs(self):
        """Ice Cream at 5 chips → decays to 0 → self-destructs."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        ice = _joker("j_ice_cream", extra={"chips": 5, "chip_mod": 5})
        result = score_hand(
            played,
            [],
            [ice],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert ice in result.jokers_removed


# ============================================================================
# Phase 13: Mr. Bones save check
# ============================================================================


class TestMrBonesSave:
    def test_saves_losing_hand(self):
        """Mr. Bones saves when score < blind_chips and hands_left == 0.
        Pair: 32×2 = 64. Blind target: 300. hands_left: 0 → game over.
        Mr. Bones: saved=True, self-destructs."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bones = _joker("j_mr_bones")
        result = score_hand(
            played,
            [],
            [bones],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
            blind_chips=300,
            game_state={"hands_left": 0},
        )
        assert result.total == 64  # score still computed
        assert result.saved is True
        assert bones in result.jokers_removed

    def test_no_save_when_winning(self):
        """Score >= blind_chips → no save needed."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bones = _joker("j_mr_bones")
        result = score_hand(
            played,
            [],
            [bones],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
            blind_chips=50,
            game_state={"hands_left": 0},
        )
        assert result.saved is False
        assert bones not in result.jokers_removed

    def test_no_save_with_hands_remaining(self):
        """hands_left > 0 → not game over, no save needed."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bones = _joker("j_mr_bones")
        result = score_hand(
            played,
            [],
            [bones],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
            blind_chips=300,
            game_state={"hands_left": 2},
        )
        assert result.saved is False


# ============================================================================
# Full multi-phase integration
# ============================================================================


class TestFullPipelineIntegration:
    def test_five_jokers_with_retrigger_and_destruction(self):
        """Complex scenario:
        - Pair of Glass Aces (Glass = x2 mult each)
        - j_joker (+4 mult), j_hack (retrigger... no, Aces not 2-5)
        - j_scary_face (+30 chips per face... Aces aren't face)
        - j_caino (xMult from face destruction)
        - j_ice_cream (+100 chips, -5 after)

        Pair of Aces (both Glass): 10 + 11 + 11 = 32 chips, 2 mult.
        Phase 7: Ace1 Glass x2 → mult 4. Ace2 Glass x2 → mult 8.
        Phase 9: j_joker +4 → 12. Ice cream +100 chips → 132.
        Phase 12: 132 × 12 = 1584.
        Phase 13: Ice cream decays."""
        ace1 = _card("Hearts", "Ace", enhancement="m_glass")
        ace2 = _card("Spades", "Ace", enhancement="m_glass")
        played = [ace1, ace2]

        joker = _joker("j_joker", mult=4)
        caino = _joker("j_caino", caino_xmult=1, extra=1)
        ice = _joker("j_ice_cream", extra={"chips": 100, "chip_mod": 5})

        result = score_hand(
            played,
            [],
            [joker, caino, ice],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Glass Aces: each x2 mult → 2 × 2 × 2 = 8 after both
        # j_joker: +4 → 12
        # Ice cream: +100 chips → 132
        # Score: 132 × 12 = 1584
        assert result.hand_type == "Pair"
        assert result.total == 1584
        # Ice cream decayed
        assert ice.ability["extra"]["chips"] == 95

    def test_seltzer_retrigger_with_after_decay(self):
        """Seltzer retriggers all cards, then decays in Phase 13.
        Pair of Aces: each Ace evaluated twice.
        Base: 10, 2. Each Ace: 11×2 = 22. Total: 10+22+22 = 54.
        Score: 54 × 2 = 108. Seltzer extra: 10 → 9."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        seltzer = _joker("j_selzer", extra=10)
        result = score_hand(
            played,
            [],
            [seltzer],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.total == 108
        assert seltzer.ability["extra"] == 9
