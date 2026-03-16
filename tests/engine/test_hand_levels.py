"""Tests for HandLevels runtime tracking.

Verifies level-up formula, play recording, round resets, most-played
detection, Black Hole (level all), and visibility of secret hands.
"""

from __future__ import annotations

from jackdaw.engine.data.hands import HAND_BASE, HandType
from jackdaw.engine.hand_levels import HandLevels, HandState

# ============================================================================
# Initialization
# ============================================================================

class TestInit:
    def test_all_12_types_present(self):
        levels = HandLevels()
        for ht in HandType:
            assert ht in levels

    def test_all_start_at_level_1(self):
        levels = HandLevels()
        for ht in HandType:
            assert levels[ht].level == 1

    def test_base_values_match(self):
        levels = HandLevels()
        for ht, base in HAND_BASE.items():
            assert levels[ht].chips == base.s_chips
            assert levels[ht].mult == base.s_mult

    def test_played_starts_at_zero(self):
        levels = HandLevels()
        for ht in HandType:
            assert levels[ht].played == 0
            assert levels[ht].played_this_round == 0

    def test_visibility_matches_base(self):
        levels = HandLevels()
        assert levels[HandType.FLUSH_FIVE].visible is False
        assert levels[HandType.PAIR].visible is True


# ============================================================================
# get (chips, mult)
# ============================================================================

class TestGet:
    def test_pair_level_1(self):
        levels = HandLevels()
        assert levels.get(HandType.PAIR) == (10, 2)

    def test_high_card_level_1(self):
        levels = HandLevels()
        assert levels.get(HandType.HIGH_CARD) == (5, 1)

    def test_flush_five_level_1(self):
        levels = HandLevels()
        assert levels.get(HandType.FLUSH_FIVE) == (160, 16)

    def test_get_by_string(self):
        levels = HandLevels()
        assert levels.get("Pair") == (10, 2)

    def test_get_state(self):
        levels = HandLevels()
        state = levels.get_state(HandType.PAIR)
        assert isinstance(state, HandState)
        assert state.level == 1


# ============================================================================
# Level up
# ============================================================================

class TestLevelUp:
    def test_pair_level_2(self):
        """Pair at level 2: chips=10+15=25, mult=2+1=3."""
        levels = HandLevels()
        levels.level_up(HandType.PAIR)
        assert levels.get(HandType.PAIR) == (25, 3)
        assert levels[HandType.PAIR].level == 2

    def test_pair_level_5(self):
        """Pair at level 5: chips=10+15*4=70, mult=2+1*4=6."""
        levels = HandLevels()
        levels.level_up(HandType.PAIR, amount=4)  # 1→5
        assert levels.get(HandType.PAIR) == (70, 6)

    def test_high_card_level_10(self):
        """High Card at level 10: chips=5+10*9=95, mult=1+1*9=10."""
        levels = HandLevels()
        levels.level_up(HandType.HIGH_CARD, amount=9)
        assert levels.get(HandType.HIGH_CARD) == (95, 10)

    def test_flush_five_level_3(self):
        """Flush Five at level 3: chips=160+50*2=260, mult=16+3*2=22."""
        levels = HandLevels()
        levels.level_up(HandType.FLUSH_FIVE, amount=2)
        assert levels.get(HandType.FLUSH_FIVE) == (260, 22)

    def test_straight_flush_level_10(self):
        """Straight Flush at level 10: chips=100+40*9=460, mult=8+4*9=44."""
        levels = HandLevels()
        levels.level_up(HandType.STRAIGHT_FLUSH, amount=9)
        assert levels.get(HandType.STRAIGHT_FLUSH) == (460, 44)

    def test_level_up_by_string(self):
        levels = HandLevels()
        levels.level_up("Full House")
        assert levels["Full House"].level == 2

    def test_sequential_level_ups(self):
        """Multiple individual level-ups accumulate."""
        levels = HandLevels()
        for _ in range(5):
            levels.level_up(HandType.PAIR)
        assert levels[HandType.PAIR].level == 6
        # chips=10+15*5=85, mult=2+1*5=7
        assert levels.get(HandType.PAIR) == (85, 7)

    def test_level_up_makes_secret_visible(self):
        """Leveling a secret hand makes it visible."""
        levels = HandLevels()
        assert levels[HandType.FLUSH_FIVE].visible is False
        levels.level_up(HandType.FLUSH_FIVE)
        assert levels[HandType.FLUSH_FIVE].visible is True

    def test_level_down(self):
        """The Arm boss blind: level down by -1."""
        levels = HandLevels()
        levels.level_up(HandType.PAIR, amount=3)  # level 4
        levels.level_up(HandType.PAIR, amount=-1)  # level 3
        assert levels[HandType.PAIR].level == 3
        # chips=10+15*2=40, mult=2+1*2=4
        assert levels.get(HandType.PAIR) == (40, 4)

    def test_level_down_minimum_zero(self):
        """Level can't go below 0."""
        levels = HandLevels()
        levels.level_up(HandType.PAIR, amount=-5)
        assert levels[HandType.PAIR].level == 0

    def test_mult_minimum_1(self):
        """Mult is clamped to 1 even at level 0."""
        levels = HandLevels()
        levels.level_up(HandType.HIGH_CARD, amount=-5)
        chips, mult = levels.get(HandType.HIGH_CARD)
        assert mult >= 1


# ============================================================================
# Black Hole (level up all)
# ============================================================================

class TestBlackHole:
    def test_all_types_advance(self):
        levels = HandLevels()
        levels.level_up_all(amount=1)
        for ht in HandType:
            assert levels[ht].level == 2

    def test_all_chips_mult_increase(self):
        levels = HandLevels()
        before = {ht: levels.get(ht) for ht in HandType}
        levels.level_up_all(amount=1)
        for ht in HandType:
            c_before, m_before = before[ht]
            c_after, m_after = levels.get(ht)
            assert c_after > c_before
            assert m_after >= m_before  # some have l_mult=1 so mult increases too

    def test_amount_3(self):
        """Black Hole could theoretically be used with amount > 1."""
        levels = HandLevels()
        levels.level_up_all(amount=3)
        for ht in HandType:
            assert levels[ht].level == 4

    def test_secret_hands_become_visible(self):
        levels = HandLevels()
        levels.level_up_all()
        assert levels[HandType.FLUSH_FIVE].visible is True
        assert levels[HandType.FLUSH_HOUSE].visible is True
        assert levels[HandType.FIVE_OF_A_KIND].visible is True


# ============================================================================
# Play recording
# ============================================================================

class TestPlayRecording:
    def test_record_increments(self):
        levels = HandLevels()
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.PAIR)
        assert levels[HandType.PAIR].played == 2
        assert levels[HandType.PAIR].played_this_round == 2

    def test_record_by_string(self):
        levels = HandLevels()
        levels.record_play("Flush")
        assert levels[HandType.FLUSH].played == 1

    def test_multiple_types(self):
        levels = HandLevels()
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.FLUSH)
        levels.record_play(HandType.PAIR)
        assert levels[HandType.PAIR].played == 2
        assert levels[HandType.FLUSH].played == 1

    def test_reset_round_counts(self):
        levels = HandLevels()
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.FLUSH)
        levels.reset_round_counts()
        assert levels[HandType.PAIR].played_this_round == 0
        assert levels[HandType.FLUSH].played_this_round == 0
        # Total played is NOT reset
        assert levels[HandType.PAIR].played == 2
        assert levels[HandType.FLUSH].played == 1


# ============================================================================
# Most played
# ============================================================================

class TestMostPlayed:
    def test_default_is_high_card(self):
        """No plays → most_played returns High Card."""
        levels = HandLevels()
        assert levels.most_played() is HandType.HIGH_CARD

    def test_after_plays(self):
        levels = HandLevels()
        levels.record_play(HandType.FLUSH)
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.PAIR)
        assert levels.most_played() is HandType.PAIR

    def test_single_play(self):
        levels = HandLevels()
        levels.record_play(HandType.STRAIGHT)
        assert levels.most_played() is HandType.STRAIGHT


# ============================================================================
# Dict-like access
# ============================================================================

class TestDictAccess:
    def test_getitem_by_enum(self):
        levels = HandLevels()
        assert levels[HandType.PAIR].level == 1

    def test_getitem_by_string(self):
        levels = HandLevels()
        assert levels["Pair"].level == 1

    def test_contains_by_enum(self):
        levels = HandLevels()
        assert HandType.PAIR in levels

    def test_contains_by_string(self):
        levels = HandLevels()
        assert "Pair" in levels

    def test_contains_invalid(self):
        levels = HandLevels()
        assert "NotAHand" not in levels

    def test_repr_all_level_1(self):
        levels = HandLevels()
        assert "all at level 1" in repr(levels)

    def test_repr_with_levels(self):
        levels = HandLevels()
        levels.level_up(HandType.PAIR, amount=2)
        r = repr(levels)
        assert "Pair" in r
        assert "3" in r  # level 3
