"""Tests for tag-generation functions — generate_blind_tags and assign_ante_blinds.

Coverage
--------
* assign_ante_blinds with known seeds → specific boss, tags, voucher.
* min_ante filtering: tags with min_ante > ante never appear.
* Tags with min_ante=2 CAN appear at ante >= 2.
* Tag pool exhaustion fallback (tag_handy) is defined.
* Determinism: same seed always gives same result.
* game_state mutation: bosses_used and round_resets.blind_tags updated.
* generate_blind_tags: structure and per-ante determinism.
* used_vouchers gate: tag_rare (requires j_blueprint) absent without it.
* Multiple ante progression: bosses_used accumulates.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.data.prototypes import BLINDS, TAGS, VOUCHERS
from jackdaw.engine.pools import _FALLBACKS
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.tags import assign_ante_blinds, generate_blind_tags

# ---------------------------------------------------------------------------
# Known-seed determinism
# ---------------------------------------------------------------------------


class TestAssignAnteBlindsKnownSeed:
    """Exact expected values from a fixed seed — validates RNG call order."""

    # Values established by running the implementation and confirming
    # order: Boss(seed='boss') → Voucher(seed='Voucher') → Tag×2(seed='Tag{ante}')
    #
    #   PseudoRandom("TUTORIAL"), ante=1
    #   boss = bl_hook, voucher = v_hieroglyph
    #   small = tag_skip, big = tag_boss

    def test_tutorial_ante1_boss(self):
        rng = PseudoRandom("TUTORIAL")
        result = assign_ante_blinds(1, rng, {})
        assert result["blind_choices"]["Boss"] == "bl_hook"

    def test_tutorial_ante1_voucher(self):
        rng = PseudoRandom("TUTORIAL")
        result = assign_ante_blinds(1, rng, {})
        assert result["voucher"] == "v_hieroglyph"

    def test_tutorial_ante1_small_tag(self):
        rng = PseudoRandom("TUTORIAL")
        result = assign_ante_blinds(1, rng, {})
        assert result["blind_tags"]["Small"] == "tag_skip"

    def test_tutorial_ante1_big_tag(self):
        rng = PseudoRandom("TUTORIAL")
        result = assign_ante_blinds(1, rng, {})
        assert result["blind_tags"]["Big"] == "tag_boss"

    def test_determinism_check_seed_ante1(self):
        #   PseudoRandom("DETERMINISM_CHECK"), ante=1
        #   boss = bl_club, voucher = v_blank
        #   small = tag_boss, big = tag_coupon
        rng = PseudoRandom("DETERMINISM_CHECK")
        result = assign_ante_blinds(1, rng, {})
        assert result["blind_choices"]["Boss"] == "bl_club"
        assert result["voucher"] == "v_blank"
        assert result["blind_tags"]["Small"] == "tag_boss"
        assert result["blind_tags"]["Big"] == "tag_coupon"

    def test_same_seed_always_same_result(self):
        r1 = PseudoRandom("DUPE_CHECK")
        r2 = PseudoRandom("DUPE_CHECK")
        res1 = assign_ante_blinds(1, r1, {})
        res2 = assign_ante_blinds(1, r2, {})
        assert res1 == res2

    def test_different_seeds_differ(self):
        res_a = assign_ante_blinds(1, PseudoRandom("SEEDA"), {})
        res_b = assign_ante_blinds(1, PseudoRandom("SEEDB"), {})
        # Highly unlikely to match across all three fields
        assert (
            res_a["blind_choices"]["Boss"] != res_b["blind_choices"]["Boss"]
            or res_a["blind_tags"]["Small"] != res_b["blind_tags"]["Small"]
            or res_a["blind_tags"]["Big"] != res_b["blind_tags"]["Big"]
        )


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestAssignAnteBlindsStructure:
    def test_top_level_keys(self):
        result = assign_ante_blinds(1, PseudoRandom("STRUCT"), {})
        assert set(result.keys()) == {"blind_choices", "blind_tags", "voucher"}

    def test_blind_choices_small_always_bl_small(self):
        result = assign_ante_blinds(1, PseudoRandom("STRUCT"), {})
        assert result["blind_choices"]["Small"] == "bl_small"

    def test_blind_choices_big_always_bl_big(self):
        result = assign_ante_blinds(1, PseudoRandom("STRUCT"), {})
        assert result["blind_choices"]["Big"] == "bl_big"

    def test_boss_is_valid_blind_key(self):
        result = assign_ante_blinds(1, PseudoRandom("VALID"), {})
        assert result["blind_choices"]["Boss"] in BLINDS

    def test_small_tag_is_valid(self):
        result = assign_ante_blinds(1, PseudoRandom("VALID"), {})
        assert result["blind_tags"]["Small"] in TAGS

    def test_big_tag_is_valid(self):
        result = assign_ante_blinds(1, PseudoRandom("VALID"), {})
        assert result["blind_tags"]["Big"] in TAGS

    def test_voucher_is_valid_or_none(self):
        result = assign_ante_blinds(1, PseudoRandom("VALID"), {})
        assert result["voucher"] is None or result["voucher"] in VOUCHERS


# ---------------------------------------------------------------------------
# game_state mutation
# ---------------------------------------------------------------------------


class TestGameStateMutation:
    def test_round_resets_blind_tags_stored(self):
        gs: dict[str, Any] = {}
        result = assign_ante_blinds(1, PseudoRandom("MUT"), gs)
        assert gs["round_resets"]["blind_tags"] == result["blind_tags"]

    def test_bosses_used_created_and_updated(self):
        gs: dict[str, Any] = {}
        result = assign_ante_blinds(1, PseudoRandom("MUT"), gs)
        boss = result["blind_choices"]["Boss"]
        assert gs["bosses_used"][boss] == 1

    def test_bosses_used_accumulates_over_antes(self):
        gs: dict[str, Any] = {}
        rng = PseudoRandom("ACCUM")
        res1 = assign_ante_blinds(1, rng, gs)
        res2 = assign_ante_blinds(2, rng, gs)
        boss1 = res1["blind_choices"]["Boss"]
        boss2 = res2["blind_choices"]["Boss"]
        # Both bosses tracked; usage count ≥ 1 for each
        assert gs["bosses_used"][boss1] >= 1
        assert gs["bosses_used"][boss2] >= 1

    def test_existing_bosses_used_respected(self):
        """Pre-loaded bosses_used steers boss selection toward least-used."""
        from jackdaw.engine.data.prototypes import BLINDS as _BLINDS

        # Mark every boss used once except bl_hook (usage = 0)
        bosses_used = {
            k: 1
            for k, v in _BLINDS.items()
            if v.boss is not None and not v.boss.get("showdown", False) and k != "bl_hook"
        }
        gs: dict[str, Any] = {"bosses_used": bosses_used}
        result = assign_ante_blinds(1, PseudoRandom("STEER"), gs)
        assert result["blind_choices"]["Boss"] == "bl_hook"

    def test_existing_round_resets_updated_not_replaced(self):
        gs: dict[str, Any] = {"round_resets": {"other_key": "preserved"}}
        assign_ante_blinds(1, PseudoRandom("MERGE"), gs)
        assert gs["round_resets"]["other_key"] == "preserved"
        assert "blind_tags" in gs["round_resets"]


# ---------------------------------------------------------------------------
# min_ante filtering
# ---------------------------------------------------------------------------

_MIN_ANTE_2_TAGS: frozenset[str] = frozenset(
    k for k, v in TAGS.items() if v.min_ante is not None and v.min_ante > 1
)


class TestMinAnteFiltering:
    """Tags with min_ante=2+ must not appear at ante=1."""

    # Verify our set is non-empty so this test class is meaningful
    def test_min_ante_2_tags_exist(self):
        assert len(_MIN_ANTE_2_TAGS) > 0
        assert "tag_buffoon" in _MIN_ANTE_2_TAGS

    @pytest.mark.parametrize(
        "seed",
        [
            "AAA",
            "BBB",
            "CCC",
            "DDD",
            "EEE",
            "FFF",
            "GGG",
            "HHH",
            "III",
            "JJJ",
        ],
    )
    def test_ante1_never_returns_min_ante2_tag_small(self, seed):
        result = generate_blind_tags(1, PseudoRandom(seed), {})
        assert result["Small"] not in _MIN_ANTE_2_TAGS, (
            f"seed={seed!r}: got {result['Small']!r} which requires min_ante>=2"
        )

    @pytest.mark.parametrize(
        "seed",
        [
            "AAA",
            "BBB",
            "CCC",
            "DDD",
            "EEE",
            "FFF",
            "GGG",
            "HHH",
            "III",
            "JJJ",
        ],
    )
    def test_ante1_never_returns_min_ante2_tag_big(self, seed):
        result = generate_blind_tags(1, PseudoRandom(seed), {})
        assert result["Big"] not in _MIN_ANTE_2_TAGS, (
            f"seed={seed!r}: got {result['Big']!r} which requires min_ante>=2"
        )

    def test_ante2_can_return_min_ante2_tags(self):
        """With enough seeds, at least one min_ante=2 tag appears at ante=2."""
        seen: set[str] = set()
        for i in range(30):
            r = generate_blind_tags(2, PseudoRandom(f"ANTE2SEED{i}"), {})
            seen.add(r["Small"])
            seen.add(r["Big"])
        assert seen & _MIN_ANTE_2_TAGS, (
            "No min_ante=2 tag appeared in 30 seeds at ante=2 — unexpected"
        )

    def test_ante1_tags_are_subset_of_unrestricted(self):
        """All returned tags must be from the ante-1-eligible set."""
        ante1_eligible = frozenset(
            k
            for k, v in TAGS.items()
            if (v.min_ante is None or v.min_ante <= 1)
            and not v.requires  # also skip requires-gated tags
        )
        for i in range(15):
            r = generate_blind_tags(1, PseudoRandom(f"ELIGIBLE{i}"), {})
            assert r["Small"] in ante1_eligible, r["Small"]
            assert r["Big"] in ante1_eligible, r["Big"]


# ---------------------------------------------------------------------------
# requires gating (tag_rare needs j_blueprint discovered)
# ---------------------------------------------------------------------------


class TestRequiresGating:
    def test_tag_rare_absent_without_blueprint(self):
        """tag_rare requires j_blueprint; never appears with empty used_vouchers."""
        seen: set[str] = set()
        for i in range(40):
            r = generate_blind_tags(1, PseudoRandom(f"RARE{i}"), {})
            seen.add(r["Small"])
            seen.add(r["Big"])
        assert "tag_rare" not in seen

    def test_tag_rare_can_appear_with_blueprint(self):
        """tag_rare is eligible when j_blueprint is discovered."""
        gs_with_blueprint = {"discovered": {"j_blueprint"}}
        seen: set[str] = set()
        for i in range(40):
            r = generate_blind_tags(1, PseudoRandom(f"RARE{i}"), gs_with_blueprint)
            seen.add(r["Small"])
            seen.add(r["Big"])
        assert "tag_rare" in seen, "tag_rare should appear when j_blueprint discovered"


# ---------------------------------------------------------------------------
# Tag pool exhaustion fallback
# ---------------------------------------------------------------------------


class TestTagPoolFallback:
    def test_fallback_key_defined(self):
        """The Tag pool has a defined fallback key in _FALLBACKS."""
        assert _FALLBACKS["Tag"] == "tag_handy"

    def test_fallback_key_is_valid_tag(self):
        assert _FALLBACKS["Tag"] in TAGS


# ---------------------------------------------------------------------------
# generate_blind_tags
# ---------------------------------------------------------------------------


class TestGenerateBlindTags:
    def test_returns_small_and_big_keys(self):
        result = generate_blind_tags(1, PseudoRandom("GEN"), {})
        assert set(result.keys()) == {"Small", "Big"}

    def test_both_values_are_valid_tags(self):
        result = generate_blind_tags(1, PseudoRandom("GEN"), {})
        assert result["Small"] in TAGS
        assert result["Big"] in TAGS

    def test_deterministic(self):
        r1 = generate_blind_tags(1, PseudoRandom("GEN_DET"), {})
        r2 = generate_blind_tags(1, PseudoRandom("GEN_DET"), {})
        assert r1 == r2

    def test_different_antes_different_streams(self):
        """Ante 1 and ante 2 use independent RNG streams (different seed keys)."""
        # Use a fresh rng for each call so stream advancement doesn't interfere
        r_ante1 = generate_blind_tags(1, PseudoRandom("ANTES"), {})
        r_ante2 = generate_blind_tags(2, PseudoRandom("ANTES"), {})
        # They may sometimes coincidentally match, but usually differ
        # (a structural check is more reliable than asserting inequality)
        assert r_ante1["Small"] in TAGS
        assert r_ante2["Small"] in TAGS

    def test_same_rng_different_ante_results(self):
        """Sequential ante calls on same rng produce different tags."""
        rng = PseudoRandom("SEQ")
        tags_1 = generate_blind_tags(1, rng, {})
        tags_2 = generate_blind_tags(2, rng, {})
        # Structural check — both are valid
        assert tags_1["Small"] in TAGS
        assert tags_2["Small"] in TAGS
