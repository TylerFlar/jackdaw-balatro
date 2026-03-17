"""Tests for eval_card scoring wrapper.

Verifies the context-based dispatch and return dict assembly matching
common_events.lua:580.
"""

from __future__ import annotations

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.scoring import eval_card


def _reset():
    reset_sort_id_counter()


def _playing_card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    sl = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rl = {
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
    c = Card()
    c.set_base(f"{sl[suit]}_{rl[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


# ============================================================================
# Played cards (cardarea="play")
# ============================================================================


class TestPlayedCards:
    """eval_card with cardarea='play' — scored playing cards."""

    def test_plain_ace_of_spades(self):
        _reset()
        c = _playing_card("Spades", "Ace")
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["chips"] == 11
        assert "mult" not in ret  # mult=0 → not included
        assert "x_mult" not in ret
        assert "p_dollars" not in ret

    def test_plain_five(self):
        _reset()
        c = _playing_card("Hearts", "5")
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["chips"] == 5
        assert "mult" not in ret

    def test_bonus_card(self):
        _reset()
        c = _playing_card("Hearts", "5", enhancement="m_bonus")
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["chips"] == 35  # 5 + 30

    def test_stone_card(self):
        _reset()
        c = _playing_card("Hearts", "Ace", enhancement="m_stone")
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["chips"] == 50  # ignores nominal

    def test_mult_card(self):
        _reset()
        c = _playing_card("Hearts", "5", enhancement="m_mult")
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["chips"] == 5
        assert ret["mult"] == 4

    def test_glass_card(self):
        _reset()
        c = _playing_card("Hearts", "5", enhancement="m_glass")
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["chips"] == 5
        assert ret["x_mult"] == 2.0

    def test_gold_seal_dollars(self):
        _reset()
        c = _playing_card("Hearts", "5")
        c.set_seal("Gold")
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["p_dollars"] == 3

    def test_foil_edition(self):
        _reset()
        c = _playing_card("Hearts", "5")
        c.set_edition({"foil": True})
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["edition"]["chip_mod"] == 50

    def test_holo_edition(self):
        _reset()
        c = _playing_card("Hearts", "5")
        c.set_edition({"holo": True})
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["edition"]["mult_mod"] == 10

    def test_polychrome_edition(self):
        _reset()
        c = _playing_card("Hearts", "5")
        c.set_edition({"polychrome": True})
        ret = eval_card(c, {"cardarea": "play"})
        assert "x_mult_mod" in ret["edition"]

    def test_debuffed_returns_empty(self):
        _reset()
        c = _playing_card("Hearts", "Ace")
        c.debuff = True
        ret = eval_card(c, {"cardarea": "play"})
        assert ret == {}

    def test_combined_bonus_foil(self):
        _reset()
        c = _playing_card("Hearts", "5", enhancement="m_bonus")
        c.set_edition({"foil": True})
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["chips"] == 35  # 5 + 30
        assert ret["edition"]["chip_mod"] == 50

    def test_combined_mult_holo(self):
        _reset()
        c = _playing_card("Hearts", "5", enhancement="m_mult")
        c.set_edition({"holo": True})
        ret = eval_card(c, {"cardarea": "play"})
        assert ret["mult"] == 4
        assert ret["edition"]["mult_mod"] == 10

    def test_zero_chips_not_included(self):
        """A joker card played (not a playing card) — no chips."""
        _reset()
        c = Card()
        c.set_ability("j_joker")
        ret = eval_card(c, {"cardarea": "play"})
        # Joker has no base → get_chip_bonus returns 0
        assert "chips" not in ret


# ============================================================================
# Held-in-hand cards (cardarea="hand")
# ============================================================================


class TestHeldCards:
    """eval_card with cardarea='hand' — cards held but not played."""

    def test_plain_card_no_effect(self):
        _reset()
        c = _playing_card("Hearts", "5")
        ret = eval_card(c, {"cardarea": "hand"})
        assert ret == {}

    def test_steel_card(self):
        _reset()
        c = _playing_card("Hearts", "5", enhancement="m_steel")
        ret = eval_card(c, {"cardarea": "hand"})
        # h_x_mult maps to ret["x_mult"] (not ret["h_x_mult"])
        assert ret["x_mult"] == 1.5
        assert "h_mult" not in ret

    def test_debuffed(self):
        _reset()
        c = _playing_card("Hearts", "5", enhancement="m_steel")
        c.debuff = True
        ret = eval_card(c, {"cardarea": "hand"})
        assert ret == {}


# ============================================================================
# Repetition-only mode
# ============================================================================


class TestRepetitionOnly:
    """eval_card with repetition_only=True — only checks seals."""

    def test_red_seal(self):
        _reset()
        c = _playing_card("Hearts", "5")
        c.set_seal("Red")
        ret = eval_card(c, {"repetition_only": True})
        assert "seals" in ret
        assert ret["seals"]["repetitions"] == 1

    def test_no_seal(self):
        _reset()
        c = _playing_card("Hearts", "5")
        ret = eval_card(c, {"repetition_only": True})
        assert ret == {}

    def test_gold_seal_no_repetition(self):
        _reset()
        c = _playing_card("Hearts", "5")
        c.set_seal("Gold")
        ret = eval_card(c, {"repetition_only": True})
        assert ret == {}

    def test_debuffed_red_seal(self):
        _reset()
        c = _playing_card("Hearts", "5")
        c.set_seal("Red")
        c.debuff = True
        ret = eval_card(c, {"repetition_only": True})
        assert ret == {}

    def test_repetition_only_ignores_play_context(self):
        """repetition_only skips all play scoring even if cardarea is set."""
        _reset()
        c = _playing_card("Hearts", "Ace")
        c.set_seal("Red")
        ret = eval_card(c, {"repetition_only": True, "cardarea": "play"})
        # Should only have seals, not chips
        assert "seals" in ret
        assert "chips" not in ret


# ============================================================================
# Edition-only mode (for joker area)
# ============================================================================


class TestEditionOnly:
    """eval_card with cardarea='jokers' and edition=True."""

    def test_foil_joker(self):
        _reset()
        c = Card()
        c.set_ability("j_joker")
        c.set_edition({"foil": True})
        ret = eval_card(c, {"cardarea": "jokers", "edition": True})
        assert ret["jokers"]["chip_mod"] == 50

    def test_holo_joker(self):
        _reset()
        c = Card()
        c.set_ability("j_joker")
        c.set_edition({"holo": True})
        ret = eval_card(c, {"cardarea": "jokers", "edition": True})
        assert ret["jokers"]["mult_mod"] == 10

    def test_polychrome_joker(self):
        _reset()
        c = Card()
        c.set_ability("j_joker")
        c.set_edition({"polychrome": True})
        ret = eval_card(c, {"cardarea": "jokers", "edition": True})
        assert ret["jokers"]["x_mult_mod"] == 1.5

    def test_no_edition_joker(self):
        _reset()
        c = Card()
        c.set_ability("j_joker")
        ret = eval_card(c, {"cardarea": "jokers", "edition": True})
        assert ret == {}

    def test_debuffed_joker_edition(self):
        _reset()
        c = Card()
        c.set_ability("j_joker")
        c.set_edition({"foil": True})
        c.debuff = True
        ret = eval_card(c, {"cardarea": "jokers", "edition": True})
        assert ret == {}


# ============================================================================
# Empty / default context
# ============================================================================


class TestDefaultContext:
    def test_no_context(self):
        _reset()
        c = _playing_card("Hearts", "Ace")
        ret = eval_card(c)
        assert ret == {}

    def test_empty_context(self):
        _reset()
        c = _playing_card("Hearts", "Ace")
        ret = eval_card(c, {})
        assert ret == {}
