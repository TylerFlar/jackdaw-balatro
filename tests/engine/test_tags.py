"""Tests for jackdaw.engine.tags — Tag.apply and TagResult.

Coverage
--------
* All 24 tags fire in their correct context and return the right TagResult.
* Tags return None for wrong contexts.
* Double Tag fires only when added tag is not tag_double itself.
* Orbital Tag with a known seed produces a deterministic hand type.
* Investment Tag fires only when last_blind_is_boss=True.
* Economy Tag caps at current dollars.
* Economy Tag returns 0 when dollars is 0.
* Edge: Tag.__repr__  works.
* Tag.triggered attribute starts False.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.data.hands import HandType
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.tags import Tag, TagResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tag(key: str) -> Tag:
    return Tag(key)


def _gs(**kwargs) -> dict:
    """Minimal game_state dict for tag tests."""
    defaults = {
        "dollars": 10,
        "hands_played": 5,
        "unused_discards": 3,
        "skips": 2,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Tag construction
# ---------------------------------------------------------------------------


class TestTagConstruction:
    def test_key_and_name(self):
        t = Tag("tag_economy")
        assert t.key == "tag_economy"
        assert t.name == "Economy Tag"

    def test_config_populated(self):
        t = Tag("tag_economy")
        assert t.config["max"] == 40

    def test_triggered_starts_false(self):
        assert Tag("tag_economy").triggered is False

    def test_id_defaults_none(self):
        assert Tag("tag_economy").id is None

    def test_id_set(self):
        t = Tag("tag_economy", tag_id=7)
        assert t.id == 7

    def test_repr(self):
        t = Tag("tag_boss", tag_id=3)
        assert "tag_boss" in repr(t)
        assert "3" in repr(t)

    def test_config_is_independent_copy(self):
        t1 = Tag("tag_economy")
        t2 = Tag("tag_economy")
        t1.config["extra"] = True
        assert "extra" not in t2.config


# ---------------------------------------------------------------------------
# Immediate tags
# ---------------------------------------------------------------------------


class TestTagEconomy:
    def test_dollars_equals_current_balance(self):
        gs = _gs(dollars=10)
        result = _tag("tag_economy").apply("immediate", gs)
        assert result is not None
        assert result.dollars == 10

    def test_capped_at_max(self):
        gs = _gs(dollars=100)
        result = _tag("tag_economy").apply("immediate", gs)
        assert result.dollars == 40  # config max

    def test_zero_when_broke(self):
        gs = _gs(dollars=0)
        result = _tag("tag_economy").apply("immediate", gs)
        assert result.dollars == 0

    def test_negative_dollars_gives_zero(self):
        gs = _gs(dollars=-5)
        result = _tag("tag_economy").apply("immediate", gs)
        assert result.dollars == 0

    def test_wrong_context_returns_none(self):
        assert _tag("tag_economy").apply("eval", _gs()) is None


class TestTagGarbage:
    def test_dollars_per_unused_discard(self):
        gs = _gs(unused_discards=4)
        result = _tag("tag_garbage").apply("immediate", gs)
        assert result.dollars == 4  # 4 * 1

    def test_zero_discards(self):
        gs = _gs(unused_discards=0)
        result = _tag("tag_garbage").apply("immediate", gs)
        assert result.dollars == 0

    def test_wrong_context_returns_none(self):
        assert _tag("tag_garbage").apply("eval", _gs()) is None


class TestTagHandy:
    def test_dollars_per_hand_played(self):
        gs = _gs(hands_played=7)
        result = _tag("tag_handy").apply("immediate", gs)
        assert result.dollars == 7

    def test_zero_hands(self):
        gs = _gs(hands_played=0)
        result = _tag("tag_handy").apply("immediate", gs)
        assert result.dollars == 0


class TestTagSkip:
    def test_dollars_equals_skips_times_5(self):
        gs = _gs(skips=3)
        result = _tag("tag_skip").apply("immediate", gs)
        assert result.dollars == 15  # 3 * 5

    def test_zero_skips(self):
        gs = _gs(skips=0)
        result = _tag("tag_skip").apply("immediate", gs)
        assert result.dollars == 0


class TestTagTopUp:
    def test_creates_two_jokers(self):
        result = _tag("tag_top_up").apply("immediate", _gs())
        assert result is not None
        assert result.create_jokers == 2

    def test_wrong_context_returns_none(self):
        assert _tag("tag_top_up").apply("new_blind_choice", _gs()) is None


class TestTagOrbital:
    def test_returns_level_up(self):
        rng = PseudoRandom("TEST_ORBITAL")
        result = _tag("tag_orbital").apply("immediate", _gs(), rng=rng)
        assert result is not None
        assert result.level_up is not None
        hand_type, levels = result.level_up
        assert isinstance(hand_type, HandType)
        assert levels == 3  # config["levels"]

    def test_deterministic_with_same_seed(self):
        rng1 = PseudoRandom("ORBITAL_SEED")
        rng2 = PseudoRandom("ORBITAL_SEED")
        r1 = _tag("tag_orbital").apply("immediate", _gs(), rng=rng1)
        r2 = _tag("tag_orbital").apply("immediate", _gs(), rng=rng2)
        assert r1.level_up == r2.level_up

    def test_different_seeds_may_differ(self):
        """Two different seeds should not always produce the same hand."""
        results = set()
        for i in range(20):
            rng = PseudoRandom(f"ORBITALSEED{i}")
            r = _tag("tag_orbital").apply("immediate", _gs(), rng=rng)
            results.add(r.level_up[0])
        # With 20 different seeds we expect more than 1 hand type
        assert len(results) > 1

    def test_raises_without_rng(self):
        with pytest.raises(ValueError, match="rng"):
            _tag("tag_orbital").apply("immediate", _gs(), rng=None)

    def test_wrong_context_returns_none(self):
        assert _tag("tag_orbital").apply("new_blind_choice", _gs()) is None


# ---------------------------------------------------------------------------
# new_blind_choice tags
# ---------------------------------------------------------------------------


class TestTagBoss:
    def test_reroll_boss(self):
        result = _tag("tag_boss").apply("new_blind_choice", _gs())
        assert result is not None
        assert result.reroll_boss is True

    def test_wrong_context_returns_none(self):
        assert _tag("tag_boss").apply("immediate", _gs()) is None


class TestTagBuffoon:
    def test_creates_buffoon_pack(self):
        result = _tag("tag_buffoon").apply("new_blind_choice", _gs())
        assert result.create_pack == "p_buffoon_mega_1"


class TestTagCharm:
    def test_creates_arcana_pack(self):
        result = _tag("tag_charm").apply("new_blind_choice", _gs())
        assert result.create_pack == "p_arcana_mega_1"


class TestTagMeteor:
    def test_creates_celestial_pack(self):
        result = _tag("tag_meteor").apply("new_blind_choice", _gs())
        assert result.create_pack == "p_celestial_mega_1"


class TestTagEthereal:
    def test_creates_spectral_pack(self):
        result = _tag("tag_ethereal").apply("new_blind_choice", _gs())
        assert result.create_pack == "p_spectral_normal_1"


class TestTagStandard:
    def test_creates_standard_pack(self):
        result = _tag("tag_standard").apply("new_blind_choice", _gs())
        assert result.create_pack == "p_standard_mega_1"


class TestNewBlindChoiceWrongContext:
    @pytest.mark.parametrize(
        "key",
        ["tag_boss", "tag_buffoon", "tag_charm", "tag_meteor", "tag_ethereal", "tag_standard"],
    )
    def test_wrong_context_returns_none(self, key):
        assert _tag(key).apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# eval tag — Investment
# ---------------------------------------------------------------------------


class TestTagInvestment:
    def test_fires_on_boss_blind(self):
        result = _tag("tag_investment").apply("eval", _gs(), last_blind_is_boss=True)
        assert result is not None
        assert result.dollars == 25

    def test_silent_on_non_boss_blind(self):
        result = _tag("tag_investment").apply("eval", _gs(), last_blind_is_boss=False)
        assert result is None

    def test_silent_without_kwarg(self):
        result = _tag("tag_investment").apply("eval", _gs())
        assert result is None

    def test_wrong_context_returns_none(self):
        assert _tag("tag_investment").apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# tag_add — Double Tag
# ---------------------------------------------------------------------------


class TestTagDouble:
    def test_fires_when_different_tag_added(self):
        result = _tag("tag_double").apply("tag_add", _gs(), added_tag_key="tag_economy")
        assert result is not None
        assert result.double is True

    def test_silent_when_double_tag_added(self):
        """Double Tag must not chain with itself."""
        result = _tag("tag_double").apply("tag_add", _gs(), added_tag_key="tag_double")
        assert result is None

    def test_silent_without_added_tag_kwarg(self):
        result = _tag("tag_double").apply("tag_add", _gs())
        assert result is None

    def test_wrong_context_returns_none(self):
        assert _tag("tag_double").apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# round_start_bonus — Juggle Tag
# ---------------------------------------------------------------------------


class TestTagJuggle:
    def test_hand_size_delta_three(self):
        result = _tag("tag_juggle").apply("round_start_bonus", _gs())
        assert result is not None
        assert result.hand_size_delta == 3

    def test_wrong_context_returns_none(self):
        assert _tag("tag_juggle").apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# store_joker_create — Rare and Uncommon Tags
# ---------------------------------------------------------------------------


class TestTagRare:
    def test_force_rarity_3(self):
        result = _tag("tag_rare").apply("store_joker_create", _gs())
        assert result is not None
        assert result.force_rarity == 3

    def test_wrong_context_returns_none(self):
        assert _tag("tag_rare").apply("immediate", _gs()) is None


class TestTagUncommon:
    def test_force_rarity_2(self):
        result = _tag("tag_uncommon").apply("store_joker_create", _gs())
        assert result is not None
        assert result.force_rarity == 2


# ---------------------------------------------------------------------------
# shop_start — D6 Tag
# ---------------------------------------------------------------------------


class TestTagDSix:
    def test_free_rerolls(self):
        result = _tag("tag_d_six").apply("shop_start", _gs())
        assert result is not None
        assert result.free_rerolls == 1

    def test_wrong_context_returns_none(self):
        assert _tag("tag_d_six").apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# store_joker_modify — Edition Tags
# ---------------------------------------------------------------------------


class TestEditionTags:
    @pytest.mark.parametrize(
        "key, edition",
        [
            ("tag_foil", "foil"),
            ("tag_holo", "holo"),
            ("tag_polychrome", "polychrome"),
            ("tag_negative", "negative"),
        ],
    )
    def test_force_edition(self, key, edition):
        result = _tag(key).apply("store_joker_modify", _gs())
        assert result is not None
        assert result.force_edition == edition

    @pytest.mark.parametrize(
        "key",
        ["tag_foil", "tag_holo", "tag_polychrome", "tag_negative"],
    )
    def test_wrong_context_returns_none(self, key):
        assert _tag(key).apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# shop_final_pass — Coupon Tag
# ---------------------------------------------------------------------------


class TestTagCoupon:
    def test_coupon_true(self):
        result = _tag("tag_coupon").apply("shop_final_pass", _gs())
        assert result is not None
        assert result.coupon is True

    def test_wrong_context_returns_none(self):
        assert _tag("tag_coupon").apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# voucher_add — Voucher Tag
# ---------------------------------------------------------------------------


class TestTagVoucher:
    def test_create_voucher_true(self):
        result = _tag("tag_voucher").apply("voucher_add", _gs())
        assert result is not None
        assert result.create_voucher is True

    def test_wrong_context_returns_none(self):
        assert _tag("tag_voucher").apply("immediate", _gs()) is None


# ---------------------------------------------------------------------------
# Cross-context isolation
# ---------------------------------------------------------------------------


class TestContextIsolation:
    """Each tag type returns None for every context it doesn't own."""

    @pytest.mark.parametrize(
        "ctx",
        [
            "eval",
            "tag_add",
            "round_start_bonus",
            "store_joker_create",
            "shop_start",
            "store_joker_modify",
            "shop_final_pass",
            "voucher_add",
            "unknown_context",
        ],
    )
    def test_economy_tag_ignores_other_contexts(self, ctx):
        assert _tag("tag_economy").apply(ctx, _gs()) is None

    @pytest.mark.parametrize(
        "ctx",
        [
            "immediate",
            "new_blind_choice",
            "tag_add",
            "round_start_bonus",
            "store_joker_create",
            "shop_start",
            "store_joker_modify",
            "shop_final_pass",
            "voucher_add",
        ],
    )
    def test_investment_tag_ignores_non_eval(self, ctx):
        assert _tag("tag_investment").apply(ctx, _gs(), last_blind_is_boss=True) is None

    def test_unknown_context_always_none(self):
        for key in [
            "tag_economy",
            "tag_boss",
            "tag_investment",
            "tag_double",
            "tag_juggle",
            "tag_rare",
            "tag_d_six",
            "tag_foil",
            "tag_coupon",
            "tag_voucher",
        ]:
            assert _tag(key).apply("__unknown__", _gs()) is None, key


# ---------------------------------------------------------------------------
# TagResult defaults
# ---------------------------------------------------------------------------


class TestTagResultDefaults:
    def test_all_defaults(self):
        r = TagResult()
        assert r.dollars == 0
        assert r.create_pack is None
        assert r.create_voucher is False
        assert r.force_rarity is None
        assert r.force_edition is None
        assert r.free_rerolls == 0
        assert r.double is False
        assert r.level_up is None
        assert r.reroll_boss is False
        assert r.hand_size_delta == 0
        assert r.create_jokers == 0
        assert r.coupon is False
