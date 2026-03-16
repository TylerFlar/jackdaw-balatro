"""Cross-validation: Python scoring pipeline vs Lua source.

Loads pre-generated Lua scoring output and verifies the Python
score_hand_base produces identical chips/mult/total values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand_base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "scoring_oracle.json"


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


# ============================================================================
# Reproduce each Lua scenario in Python
# ============================================================================

def _run_pair_aces_basic():
    reset_sort_id_counter()
    played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    return score_hand_base(
        played, [], HandLevels(), Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_three_kings_foil():
    reset_sort_id_counter()
    k1 = _card("Hearts", "King")
    k1.set_edition({"foil": True})
    played = [k1, _card("Spades", "King"), _card("Clubs", "King"),
              _card("Diamonds", "5"), _card("Hearts", "2")]
    return score_hand_base(
        played, [], HandLevels(), Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_flush_glass():
    reset_sort_id_counter()
    played = [
        _card("Hearts", "2"), _card("Hearts", "5"), _card("Hearts", "8"),
        _card("Hearts", "Jack"), _card("Hearts", "Ace", enhancement="m_glass"),
    ]
    return score_hand_base(
        played, [], HandLevels(), Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_full_house_steel_held():
    reset_sort_id_counter()
    played = [
        _card("Hearts", "King"), _card("Spades", "King"),
        _card("Clubs", "King"), _card("Diamonds", "5"),
        _card("Hearts", "5"),
    ]
    held = [_card("Clubs", "3", enhancement="m_steel")]
    return score_hand_base(
        played, held, HandLevels(), Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_pair_aces_red_seal():
    reset_sort_id_counter()
    c = _card("Hearts", "Ace")
    c.set_seal("Red")
    played = [c, _card("Spades", "Ace")]
    return score_hand_base(
        played, [], HandLevels(), Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_pair_aces_flint():
    reset_sort_id_counter()
    played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    return score_hand_base(
        played, [], HandLevels(), Blind.create("bl_flint", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_pair_eye_debuffed():
    reset_sort_id_counter()
    levels = HandLevels()
    blind = Blind.create("bl_eye", ante=1)

    # First pair: allowed (registers "Pair" in The Eye)
    p1 = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    score_hand_base(p1, [], levels, blind, PseudoRandom("ORACLE"))

    # Second pair: blocked
    reset_sort_id_counter()
    p2 = [_card("Hearts", "King"), _card("Spades", "King")]
    return score_hand_base(p2, [], levels, blind, PseudoRandom("ORACLE"))


# Map test names to Python runners
RUNNERS = {
    "pair_aces_basic": _run_pair_aces_basic,
    "three_kings_foil": _run_three_kings_foil,
    "flush_glass": _run_flush_glass,
    "full_house_steel_held": _run_full_house_steel_held,
    "pair_aces_red_seal": _run_pair_aces_red_seal,
    "pair_aces_flint": _run_pair_aces_flint,
    "pair_eye_debuffed": _run_pair_eye_debuffed,
}


# ============================================================================
# Fixture-based cross-validation
# ============================================================================

class TestScoringOracle:
    """Validate Python scoring against Lua ground truth."""

    @pytest.fixture(scope="class")
    def fixture(self) -> list[dict]:
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
        with open(FIXTURE_PATH) as f:
            return json.load(f)["tests"]

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_total_matches(self, fixture, test_name):
        """floor(chips * mult) must match Lua exactly."""
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.total == lua_test["total"], (
            f"{test_name}: Python total={py_result.total}, Lua total={lua_test['total']}"
        )

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_chips_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.chips == pytest.approx(lua_test["chips"], abs=0.01), (
            f"{test_name}: Python chips={py_result.chips}, Lua chips={lua_test['chips']}"
        )

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_mult_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.mult == pytest.approx(lua_test["mult"], abs=0.01), (
            f"{test_name}: Python mult={py_result.mult}, Lua mult={lua_test['mult']}"
        )

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_hand_type_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.hand_type == lua_test["hand_type"]

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_debuffed_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.debuffed == lua_test["debuffed"]
