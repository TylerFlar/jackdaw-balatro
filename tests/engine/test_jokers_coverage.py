"""Coverage audit: every joker in the prototype data is registered.

Also tests the remaining active jokers (Castle, Runner, Ramen, etc.)
and verifies passive jokers are no-ops.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.prototypes import _load_json
from jackdaw.engine.jokers import (
    _REGISTRY,
    JokerContext,
    calculate_joker,
)


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


def _poker_hands_with(*types: str) -> dict[str, list]:
    all_types = [
        "Flush Five",
        "Flush House",
        "Five of a Kind",
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Flush",
        "Straight",
        "Three of a Kind",
        "Two Pair",
        "Pair",
        "High Card",
    ]
    return {t: [["p"]] if t in types else [] for t in all_types}


# ============================================================================
# 150/150 coverage audit
# ============================================================================


class TestFullCoverage:
    """Every joker in centers.json must be registered."""

    def test_all_150_registered(self):
        centers = _load_json("centers.json")
        joker_keys = [k for k, v in centers.items() if v.get("set") == "Joker"]
        assert len(joker_keys) == 150
        missing = [k for k in joker_keys if k not in _REGISTRY]
        assert missing == [], f"Unregistered jokers: {missing}"

    def test_no_extra_registrations(self):
        """No handler registered for a non-existent joker key."""
        centers = _load_json("centers.json")
        joker_keys = set(k for k, v in centers.items() if v.get("set") == "Joker")
        for key in _REGISTRY:
            if key.startswith("j_test"):
                continue  # test fixtures
            assert key in joker_keys, f"Handler {key} has no prototype"


# ============================================================================
# Passive jokers: no-op in calculate_joker
# ============================================================================


class TestPassiveJokers:
    """Passive/meta jokers return None in all contexts."""

    @pytest.mark.parametrize(
        "key",
        [
            "j_four_fingers",
            "j_shortcut",
            "j_pareidolia",
            "j_smeared",
            "j_splash",
            "j_ring_master",
            "j_juggler",
            "j_drunkard",
            "j_troubadour",
            "j_merry_andy",
            "j_oops",
            "j_credit_card",
            "j_chaos",
            "j_astronomer",
        ],
    )
    def test_noop_in_all_contexts(self, key):
        j = _joker(key)
        for ctx in [
            JokerContext(joker_main=True),
            JokerContext(before=True),
            JokerContext(after=True),
            JokerContext(individual=True, cardarea="play"),
        ]:
            assert calculate_joker(j, ctx) is None


# ============================================================================
# Castle: +3 chips per matching suit discarded
# ============================================================================


class TestCastle:
    def test_matching_suit_accumulates(self):
        j = _joker("j_castle", extra={"chip_mod": 3, "chips": 0}, castle_card_suit="Hearts")
        heart = _card("Hearts", "5")
        ctx = JokerContext(discard=True, other_card=heart)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 3

    def test_non_matching_no_effect(self):
        j = _joker("j_castle", extra={"chip_mod": 3, "chips": 0}, castle_card_suit="Hearts")
        spade = _card("Spades", "5")
        ctx = JokerContext(discard=True, other_card=spade)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 0

    def test_joker_main_returns(self):
        j = _joker("j_castle", extra={"chip_mod": 3, "chips": 9})
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.chip_mod == 9


# ============================================================================
# Runner: +15 chips per Straight played
# ============================================================================


class TestRunner:
    def test_straight_accumulates(self):
        j = _joker("j_runner", extra={"chip_mod": 15, "chips": 0})
        ph = _poker_hands_with("Straight")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 15

    def test_non_straight_no_effect(self):
        j = _joker("j_runner", extra={"chip_mod": 15, "chips": 0})
        ph = _poker_hands_with("Pair")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 0

    def test_joker_main_returns(self):
        j = _joker("j_runner", extra={"chip_mod": 15, "chips": 30})
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.chip_mod == 30


# ============================================================================
# Ramen: x2 mult, loses 0.01 per discard, self-destructs
# ============================================================================


class TestRamen:
    def test_discard_decrements(self):
        j = _joker("j_ramen", x_mult=2.0, extra=0.01)
        calculate_joker(j, JokerContext(discard=True))
        assert j.ability["x_mult"] == pytest.approx(1.99)

    def test_self_destructs_at_1(self):
        j = _joker("j_ramen", x_mult=1.005, extra=0.01)
        result = calculate_joker(j, JokerContext(discard=True))
        assert result.remove is True

    def test_joker_main_gives_xmult(self):
        j = _joker("j_ramen", x_mult=2.0, extra=0.01)
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 2.0


# ============================================================================
# Mr. Bones: prevents game over
# ============================================================================


class TestMrBones:
    def test_game_over_saves(self):
        j = _joker("j_mr_bones")
        ctx = JokerContext()
        ctx.game_over = True  # dynamic attribute
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.saved is True
        assert result.remove is True

    def test_normal_context_no_effect(self):
        j = _joker("j_mr_bones")
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


# ============================================================================
# Burnt Joker: level up discard hand on first discard
# ============================================================================


class TestBurnt:
    def test_first_discard_levels_up(self):
        j = _joker("j_burnt", extra=4)
        last = _card("Hearts", "King")
        ctx = JokerContext(
            discard=True,
            other_card=last,
            full_hand=[last],
            discards_used=0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.level_up is True

    def test_second_discard_no_effect(self):
        j = _joker("j_burnt", extra=4)
        last = _card("Hearts", "King")
        ctx = JokerContext(
            discard=True,
            other_card=last,
            full_hand=[last],
            discards_used=1,
        )
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Turtle Bean: hand size decay, self-destructs
# ============================================================================


class TestTurtleBean:
    def test_end_of_round_decays(self):
        j = _joker("j_turtle_bean", extra={"h_size": 5, "h_mod": 1})
        calculate_joker(j, JokerContext(end_of_round=True))
        assert j.ability["extra"]["h_size"] == 4

    def test_self_destructs_at_zero(self):
        j = _joker("j_turtle_bean", extra={"h_size": 1, "h_mod": 1})
        result = calculate_joker(j, JokerContext(end_of_round=True))
        assert result.remove is True


# ============================================================================
# Perkeo: copy consumable when leaving shop
# ============================================================================


class TestPerkeo:
    def test_ending_shop_creates(self):
        j = _joker("j_perkeo")
        ctx = JokerContext(ending_shop=True)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "consumable_copy"

    def test_other_context_no_effect(self):
        j = _joker("j_perkeo")
        assert calculate_joker(j, JokerContext(joker_main=True)) is None
