"""Parity cases for the tags the recorded tapes never award (tag.lua:115-468).

Each case puts a tag on a blind with the oracle (:class:`tests.parity.pyengine.PyEngine`),
skips the blind and plays on to where the tag fires, checking every step with
:func:`tests.parity.rust_step.check_step` (the oracle's snapshot goes into Rust, both
engines take the action, and the snapshots afterwards must match).  The assertions on
the oracle's state show that the tag fired.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.card import _resolve_center
from tests.parity.pyengine import PyEngine
from tests.parity.test_audit_flow import check

_WIN = 10**12


def _start(seed: str) -> dict[str, Any]:
    return {"back": "b_red", "stake": 1, "seed": seed, "challenge": None}


def skip_small_for(seed: str, tag: str) -> PyEngine:
    """The run's first blind select, with ``tag`` on the Small Blind."""
    eng = PyEngine(_start(seed))
    eng.gs["round_resets"]["blind_tags"]["Small"] = tag
    return eng


def skip_big_for(seed: str, tag: str) -> PyEngine:
    """The Big Blind's select after a played Small Blind, with ``tag`` on it."""
    setup = [
        {"op": "action", "action": ["SelectBlind"]},
        {"op": "set", "chips": _WIN},
        {"op": "action", "action": ["PlayHand", [0]]},
        {"op": "action", "action": ["CashOut"]},
        {"op": "action", "action": ["NextRound"]},
    ]
    eng = PyEngine(_start(seed), setup)
    eng.gs["round_resets"]["blind_tags"]["Big"] = tag
    return eng


def to_the_shop(eng: PyEngine) -> dict[str, Any]:
    """Skip the Small Blind, beat the Big Blind and cash out, checking each step."""
    check(eng, ["SkipBlind"], ["SelectBlind"])
    eng.gs["chips"] = _WIN
    return check(eng, ["PlayHand", [0]], ["CashOut"])


def tags_of(snap: dict[str, Any]) -> list[str]:
    return [t["key"] for t in snap["tags"]]


class TestImmediateTags:
    def test_garbage_tag_pays_for_unused_discards(self):
        eng = skip_big_for("AU_TAG_GARBAGE", "tag_garbage")
        before = eng.snapshot()["game"]
        snap = check(eng, ["SkipBlind"])
        assert before["unused_discards"] > 0
        assert snap["game"]["dollars"] == before["dollars"] + before["unused_discards"]
        assert "tag_garbage" not in tags_of(snap)

    def test_handy_tag_pays_for_hands_played(self):
        eng = skip_big_for("AU_TAG_HANDY", "tag_handy")
        before = eng.snapshot()["game"]
        snap = check(eng, ["SkipBlind"])
        assert before["hands_played"] > 0
        assert snap["game"]["dollars"] == before["dollars"] + before["hands_played"]

    def test_orbital_tag_levels_the_blinds_hand(self):
        """G40: the hand was rolled when the blind-select screen was built."""
        eng = skip_small_for("AU_TAG_ORBITAL", "tag_orbital")
        hand = eng.snapshot()["game"]["orbital_choices"]["1"]["Small"]
        snap = check(eng, ["SkipBlind"])
        assert snap["hands"][hand]["level"] == 4


class TestNewBlindChoiceTags:
    def test_standard_tag_opens_a_mega_standard_pack(self):
        snap = check(skip_small_for("AU_TAG_STANDARD", "tag_standard"), ["SkipBlind"])
        assert snap["phase"] == "pack_opening"
        assert snap["pack"]["kind"] == "Standard"
        assert len(snap["areas"]["pack_cards"]) == 5

    def test_boss_tag_rerolls_the_boss(self):
        """G41: free, marks boss_rerolled."""
        eng = skip_small_for("AU_TAG_BOSS", "tag_boss")
        before = eng.snapshot()
        snap = check(eng, ["SkipBlind"])
        assert snap["round_resets"]["boss_rerolled"]
        assert snap["game"]["dollars"] == before["game"]["dollars"]
        assert sum(snap["sets"]["bosses_used"].values()) == (
            sum(before["sets"]["bosses_used"].values()) + 1
        )


class TestRoundStartBonusTags:
    def test_juggle_tag_raises_the_hand_size_for_one_round(self):
        eng = skip_small_for("AU_TAG_JUGGLE", "tag_juggle")
        base = eng.snapshot()["game"]["hand_size"]
        snap = check(eng, ["SkipBlind"], ["SelectBlind"])
        assert snap["game"]["hand_size"] == base + 3
        assert len(snap["areas"]["hand"]) == base + 3
        assert snap["round_resets"]["temp_handsize"] == 3
        eng.gs["chips"] = _WIN
        snap = check(eng, ["PlayHand", [0]])
        assert snap["game"]["hand_size"] == base
        assert snap["round_resets"]["temp_handsize"] is None


class TestShopTags:
    def test_d6_tag_makes_rerolls_start_at_zero(self):
        snap = to_the_shop(skip_small_for("AU_TAG_D6", "tag_d_six"))
        assert snap["phase"] == "shop"
        assert snap["current_round"]["reroll_cost"] == 0
        assert snap["round_resets"]["temp_reroll_cost"] == 0

    def test_coupon_tag_makes_jokers_and_packs_free(self):
        """G42: once per shop."""
        snap = to_the_shop(skip_small_for("AU_TAG_COUPON", "tag_coupon"))
        areas = snap["areas"]
        assert all(c["costs"]["cost"] == 0 for c in areas["shop_jokers"] + areas["shop_booster"])
        assert snap["sim"]["shop_free"]

    def test_voucher_tag_adds_a_second_voucher(self):
        snap = to_the_shop(skip_small_for("AU_TAG_VOUCHER", "tag_voucher"))
        assert len(snap["areas"]["shop_vouchers"]) == 2

    def test_rare_tag_makes_a_free_rare_joker(self):
        snap = to_the_shop(skip_small_for("AU_TAG_RARE", "tag_rare"))
        first = snap["areas"]["shop_jokers"][0]
        assert _resolve_center(first["center"])["rarity"] == 3
        assert first["costs"]["cost"] == 0

    def test_uncommon_tag_makes_a_free_uncommon_joker(self):
        snap = to_the_shop(skip_small_for("AU_TAG_UNCOMMON", "tag_uncommon"))
        first = snap["areas"]["shop_jokers"][0]
        assert _resolve_center(first["center"])["rarity"] == 2
        assert first["costs"]["cost"] == 0

    def test_edition_tags_give_a_shop_joker_their_edition(self):
        for tag, edition in (("tag_foil", "foil"), ("tag_holo", "holo")):
            snap = to_the_shop(skip_small_for(f"AU_TAG_{edition}", tag))
            jokers = [c for c in snap["areas"]["shop_jokers"] if c["center"].startswith("j_")]
            assert any((c["edition"] or {}).get(edition) for c in jokers), tag


class TestEvalTags:
    def test_investment_tag_pays_after_the_boss(self):
        eng = skip_small_for("AU_TAG_INVEST", "tag_investment")
        snap = to_the_shop(eng)
        assert "tag_investment" in tags_of(snap)
        check(eng, ["NextRound"])
        eng.gs["round_resets"]["blind_choices"]["Boss"] = "bl_wall"
        check(eng, ["SelectBlind"])
        eng.gs["chips"] = _WIN
        snap = check(eng, ["PlayHand", [0]])
        assert "tag_investment" not in tags_of(snap)
