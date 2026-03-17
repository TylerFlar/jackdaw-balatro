"""Tests for the voucher effect system (jackdaw/engine/vouchers.py)."""

from __future__ import annotations

import pytest

from jackdaw.engine.data.prototypes import CENTER_POOLS, VOUCHERS
from jackdaw.engine.vouchers import (
    apply_voucher,
    check_voucher_prerequisites,
    get_available_voucher_pool,
    get_next_voucher_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gs(**kw) -> dict:
    """Minimal game_state with sensible defaults."""
    base = {
        "shop": {"joker_max": 2},
        "discount_percent": 0,
        "tarot_rate": 4,
        "planet_rate": 4,
        "playing_card_rate": 0,
        "edition_rate": 1,
        "interest_cap": 25,
        "consumable_slots": 2,
        "joker_slots": 5,
        "hand_size": 8,
        "round_resets": {
            "hands": 4,
            "discards": 3,
            "reroll_cost": 5,
            "ante": 1,
        },
        "current_round": {"reroll_cost": 5},
    }
    base.update(kw)
    return base


class _ControlledRng:
    """RNG stub that always picks the first element."""

    def seed(self, key: str) -> float:
        return 0.0

    def element(self, table: list, seed_val: float) -> tuple:
        return (table[0], 0)


# ============================================================================
# Prerequisites
# ============================================================================

class TestCheckVoucherPrerequisites:
    def test_voucher_with_no_requires_always_passes(self):
        # v_overstock_norm has no requires
        assert check_voucher_prerequisites("v_overstock_norm", {}) is True

    def test_requires_missing_returns_false(self):
        # v_overstock_plus requires v_overstock_norm
        assert check_voucher_prerequisites("v_overstock_plus", {}) is False

    def test_requires_satisfied_returns_true(self):
        assert check_voucher_prerequisites(
            "v_overstock_plus", {"v_overstock_norm": True}
        ) is True

    def test_liquidation_requires_clearance_sale(self):
        assert check_voucher_prerequisites("v_liquidation", {}) is False
        assert check_voucher_prerequisites(
            "v_liquidation", {"v_clearance_sale": True}
        ) is True

    def test_nacho_tong_requires_grabber(self):
        assert check_voucher_prerequisites("v_nacho_tong", {}) is False
        assert check_voucher_prerequisites(
            "v_nacho_tong", {"v_grabber": True}
        ) is True

    def test_glow_up_requires_hone(self):
        assert check_voucher_prerequisites("v_glow_up", {}) is False
        assert check_voucher_prerequisites("v_glow_up", {"v_hone": True}) is True

    def test_money_tree_requires_seed_money(self):
        assert check_voucher_prerequisites("v_money_tree", {}) is False
        assert check_voucher_prerequisites(
            "v_money_tree", {"v_seed_money": True}
        ) is True

    def test_unknown_key_returns_false(self):
        assert check_voucher_prerequisites("v_nonexistent", {}) is False

    def test_antimatter_requires_blank(self):
        assert check_voucher_prerequisites("v_antimatter", {}) is False
        assert check_voucher_prerequisites("v_antimatter", {"v_blank": True}) is True

    def test_retcon_requires_directors_cut(self):
        assert check_voucher_prerequisites("v_retcon", {}) is False
        assert check_voucher_prerequisites(
            "v_retcon", {"v_directors_cut": True}
        ) is True


# ============================================================================
# Pool building
# ============================================================================

class TestGetAvailableVoucherPool:
    def test_no_used_vouchers_returns_all_unlocked(self):
        pool = get_available_voucher_pool({})
        # Vouchers with requires whose prerequisite is not used must be excluded
        for key in pool:
            proto = VOUCHERS[key]
            for req in proto.requires:
                assert req not in pool or key == req, (
                    f"{key} in pool but its prereq {req} is not used"
                )

    def test_excludes_used_vouchers(self):
        pool = get_available_voucher_pool({"v_grabber": True})
        assert "v_grabber" not in pool

    def test_excludes_in_shop_vouchers(self):
        pool = get_available_voucher_pool({}, in_shop=["v_grabber"])
        assert "v_grabber" not in pool

    def test_excludes_vouchers_missing_prereqs(self):
        # v_overstock_plus requires v_overstock_norm — not used
        pool = get_available_voucher_pool({})
        assert "v_overstock_plus" not in pool

    def test_includes_upgraded_when_base_used(self):
        pool = get_available_voucher_pool({"v_overstock_norm": True})
        assert "v_overstock_plus" in pool

    def test_base_vouchers_are_always_available(self):
        pool = get_available_voucher_pool({})
        base = ["v_overstock_norm", "v_grabber", "v_wasteful", "v_hone",
                "v_clearance_sale", "v_crystal_ball", "v_seed_money",
                "v_tarot_merchant", "v_planet_merchant", "v_magic_trick",
                "v_paint_brush", "v_hieroglyph", "v_reroll_surplus",
                "v_telescope", "v_blank"]
        for key in base:
            assert key in pool, f"{key} not in pool"

    def test_pool_order_matches_centers_pool_order(self):
        pool = get_available_voucher_pool({})
        expected_order = [k for k in CENTER_POOLS["Voucher"] if k in pool]
        assert pool == expected_order


# ============================================================================
# get_next_voucher_key
# ============================================================================

class TestGetNextVoucherKey:
    def test_returns_valid_voucher_key(self):
        rng = _ControlledRng()
        key = get_next_voucher_key(rng, {})
        assert key in VOUCHERS

    def test_returns_none_when_pool_empty(self):
        all_used = {k: True for k in VOUCHERS}
        rng = _ControlledRng()
        key = get_next_voucher_key(rng, all_used)
        assert key is None

    def test_filters_out_used_vouchers(self):
        # Use all vouchers except v_wasteful, v_recyclomancy, etc.
        all_used = {k: True for k in VOUCHERS if k != "v_wasteful"}
        # Also pre-satisfy v_wasteful's prereqs (it has none)
        rng = _ControlledRng()
        key = get_next_voucher_key(rng, all_used)
        # Only v_wasteful should be eligible (or nothing if pool is empty)
        if key is not None:
            assert not all_used.get(key), f"Got already-used key: {key}"

    def test_filters_in_shop_vouchers(self):
        rng = _ControlledRng()
        pool = get_available_voucher_pool({})
        first_key = pool[0]
        key = get_next_voucher_key(rng, {}, in_shop=[first_key])
        # _ControlledRng picks first — first key is excluded, so should get second
        assert key != first_key

    def test_from_tag_uses_different_seed(self):
        """from_tag=True uses 'Voucher_fromtag' key (no assert on value, just runs)."""
        rng = _ControlledRng()
        key = get_next_voucher_key(rng, {}, from_tag=True)
        assert key in VOUCHERS

    def test_deterministic_with_real_rng(self):
        """Same PseudoRandom seed produces same voucher twice."""
        from jackdaw.engine.rng import PseudoRandom
        rng1 = PseudoRandom("test_voucher_det")
        rng2 = PseudoRandom("test_voucher_det")
        used: dict[str, bool] = {}
        k1 = get_next_voucher_key(rng1, used)
        k2 = get_next_voucher_key(rng2, used)
        assert k1 == k2


# ============================================================================
# apply_voucher — shop modifiers
# ============================================================================

class TestApplyVoucherShopModifiers:
    def test_overstock_norm_increases_shop_joker_max(self):
        gs = _gs()
        apply_voucher("v_overstock_norm", gs)
        assert gs["shop"]["joker_max"] == 3

    def test_overstock_plus_increases_shop_joker_max(self):
        gs = _gs()
        apply_voucher("v_overstock_plus", gs)
        assert gs["shop"]["joker_max"] == 3

    def test_overstock_stacks_when_applied_twice(self):
        """Applying norm then plus gives +2 total."""
        gs = _gs()
        apply_voucher("v_overstock_norm", gs)
        apply_voucher("v_overstock_plus", gs)
        assert gs["shop"]["joker_max"] == 4

    def test_clearance_sale_sets_discount_25(self):
        gs = _gs()
        apply_voucher("v_clearance_sale", gs)
        assert gs["discount_percent"] == 25

    def test_liquidation_sets_discount_50(self):
        gs = _gs()
        apply_voucher("v_liquidation", gs)
        assert gs["discount_percent"] == 50

    def test_liquidation_overwrites_clearance_sale(self):
        gs = _gs()
        apply_voucher("v_clearance_sale", gs)
        apply_voucher("v_liquidation", gs)
        assert gs["discount_percent"] == 50

    def test_tarot_merchant_sets_tarot_rate(self):
        """Tarot Merchant: tarot_rate = 4 * 2.4 = 9.6"""
        gs = _gs()
        apply_voucher("v_tarot_merchant", gs)
        assert gs["tarot_rate"] == pytest.approx(9.6)

    def test_tarot_tycoon_sets_tarot_rate(self):
        """Tarot Tycoon: tarot_rate = 4 * 8 = 32"""
        gs = _gs()
        apply_voucher("v_tarot_tycoon", gs)
        assert gs["tarot_rate"] == 32

    def test_planet_merchant_sets_planet_rate(self):
        """Planet Merchant: planet_rate = 4 * 2.4 = 9.6"""
        gs = _gs()
        apply_voucher("v_planet_merchant", gs)
        assert gs["planet_rate"] == pytest.approx(9.6)

    def test_planet_tycoon_sets_planet_rate(self):
        """Planet Tycoon: planet_rate = 4 * 8 = 32"""
        gs = _gs()
        apply_voucher("v_planet_tycoon", gs)
        assert gs["planet_rate"] == 32

    def test_hone_sets_edition_rate_2(self):
        gs = _gs()
        apply_voucher("v_hone", gs)
        assert gs["edition_rate"] == 2

    def test_glow_up_sets_edition_rate_4(self):
        gs = _gs()
        apply_voucher("v_glow_up", gs)
        assert gs["edition_rate"] == 4

    def test_hone_then_glow_up_sets_to_4(self):
        """Glow Up overwrites Hone — both SET (not multiply)."""
        gs = _gs()
        apply_voucher("v_hone", gs)
        apply_voucher("v_glow_up", gs)
        assert gs["edition_rate"] == 4

    def test_magic_trick_sets_playing_card_rate(self):
        gs = _gs()
        apply_voucher("v_magic_trick", gs)
        assert gs["playing_card_rate"] == 4

    def test_illusion_sets_playing_card_rate(self):
        gs = _gs()
        apply_voucher("v_illusion", gs)
        assert gs["playing_card_rate"] == 4


# ============================================================================
# apply_voucher — economy
# ============================================================================

class TestApplyVoucherEconomy:
    def test_seed_money_sets_interest_cap_50(self):
        gs = _gs()
        apply_voucher("v_seed_money", gs)
        assert gs["interest_cap"] == 50

    def test_money_tree_sets_interest_cap_100(self):
        gs = _gs()
        apply_voucher("v_money_tree", gs)
        assert gs["interest_cap"] == 100

    def test_reroll_surplus_reduces_reroll_cost_by_2(self):
        gs = _gs()
        apply_voucher("v_reroll_surplus", gs)
        assert gs["round_resets"]["reroll_cost"] == 3

    def test_reroll_glut_stacks_with_surplus(self):
        gs = _gs()
        apply_voucher("v_reroll_surplus", gs)
        apply_voucher("v_reroll_glut", gs)
        assert gs["round_resets"]["reroll_cost"] == 1

    def test_reroll_surplus_clamps_current_round_at_zero(self):
        gs = _gs()
        gs["current_round"]["reroll_cost"] = 1
        apply_voucher("v_reroll_surplus", gs)
        assert gs["current_round"]["reroll_cost"] == 0


# ============================================================================
# apply_voucher — hands & discards
# ============================================================================

class TestApplyVoucherHandsDiscards:
    def test_grabber_increases_round_hands_by_1(self):
        gs = _gs()
        apply_voucher("v_grabber", gs)
        assert gs["round_resets"]["hands"] == 5

    def test_nacho_tong_increases_round_hands_by_1(self):
        gs = _gs()
        apply_voucher("v_nacho_tong", gs)
        assert gs["round_resets"]["hands"] == 5

    def test_grabber_and_nacho_tong_stack(self):
        gs = _gs()
        apply_voucher("v_grabber", gs)
        apply_voucher("v_nacho_tong", gs)
        assert gs["round_resets"]["hands"] == 6

    def test_wasteful_increases_round_discards_by_1(self):
        gs = _gs()
        apply_voucher("v_wasteful", gs)
        assert gs["round_resets"]["discards"] == 4

    def test_recyclomancy_increases_round_discards_by_1(self):
        gs = _gs()
        apply_voucher("v_recyclomancy", gs)
        assert gs["round_resets"]["discards"] == 4

    def test_paint_brush_increases_hand_size(self):
        gs = _gs()
        apply_voucher("v_paint_brush", gs)
        assert gs["hand_size"] == 9

    def test_palette_increases_hand_size(self):
        gs = _gs()
        apply_voucher("v_palette", gs)
        assert gs["hand_size"] == 9

    def test_paint_brush_and_palette_stack(self):
        gs = _gs()
        apply_voucher("v_paint_brush", gs)
        apply_voucher("v_palette", gs)
        assert gs["hand_size"] == 10


# ============================================================================
# apply_voucher — slots
# ============================================================================

class TestApplyVoucherSlots:
    def test_crystal_ball_increases_consumable_slots(self):
        gs = _gs()
        apply_voucher("v_crystal_ball", gs)
        assert gs["consumable_slots"] == 3

    def test_crystal_ball_stacks(self):
        gs = _gs()
        apply_voucher("v_crystal_ball", gs)
        apply_voucher("v_crystal_ball", gs)  # hypothetical double-use
        assert gs["consumable_slots"] == 4

    def test_antimatter_increases_joker_slots(self):
        gs = _gs()
        apply_voucher("v_antimatter", gs)
        assert gs["joker_slots"] == 6


# ============================================================================
# apply_voucher — ante modifiers
# ============================================================================

class TestApplyVoucherAnteModifiers:
    def test_hieroglyph_reduces_ante(self):
        gs = _gs()
        gs["round_resets"]["ante"] = 2
        apply_voucher("v_hieroglyph", gs)
        assert gs["round_resets"]["ante"] == 1

    def test_hieroglyph_reduces_hands(self):
        gs = _gs()
        apply_voucher("v_hieroglyph", gs)
        assert gs["round_resets"]["hands"] == 3

    def test_hieroglyph_sets_blind_ante(self):
        gs = _gs()
        gs["round_resets"]["ante"] = 3
        apply_voucher("v_hieroglyph", gs)
        assert "blind_ante" in gs["round_resets"]
        assert gs["round_resets"]["blind_ante"] < 3

    def test_petroglyph_reduces_ante(self):
        gs = _gs()
        gs["round_resets"]["ante"] = 2
        apply_voucher("v_petroglyph", gs)
        assert gs["round_resets"]["ante"] == 1

    def test_petroglyph_reduces_discards(self):
        gs = _gs()
        apply_voucher("v_petroglyph", gs)
        assert gs["round_resets"]["discards"] == 2

    def test_petroglyph_does_not_reduce_hands(self):
        gs = _gs()
        apply_voucher("v_petroglyph", gs)
        assert gs["round_resets"]["hands"] == 4  # unchanged

    def test_hieroglyph_does_not_reduce_discards(self):
        gs = _gs()
        apply_voucher("v_hieroglyph", gs)
        assert gs["round_resets"]["discards"] == 3  # unchanged


# ============================================================================
# apply_voucher — boss blind rerolls
# ============================================================================

class TestApplyVoucherBossBlindRerolls:
    def test_directors_cut_enables_one_boss_blind_reroll(self):
        gs = _gs()
        apply_voucher("v_directors_cut", gs)
        assert gs["boss_blind_rerolls"] == 1

    def test_directors_cut_sets_reroll_cost(self):
        gs = _gs()
        apply_voucher("v_directors_cut", gs)
        assert gs["boss_blind_reroll_cost"] == 10  # config.extra = 10

    def test_retcon_sets_unlimited_rerolls(self):
        gs = _gs()
        apply_voucher("v_retcon", gs)
        assert gs["boss_blind_rerolls"] == -1  # -1 = unlimited

    def test_retcon_sets_reroll_cost_to_zero(self):
        gs = _gs()
        apply_voucher("v_retcon", gs)
        assert gs["boss_blind_reroll_cost"] == 0


# ============================================================================
# apply_voucher — passive / no-op vouchers
# ============================================================================

class TestApplyVoucherPassive:
    def test_blank_returns_empty_mutations(self):
        gs = _gs()
        mutations = apply_voucher("v_blank", gs)
        assert mutations == {}

    def test_telescope_returns_empty_mutations(self):
        gs = _gs()
        mutations = apply_voucher("v_telescope", gs)
        assert mutations == {}

    def test_observatory_returns_empty_mutations(self):
        gs = _gs()
        mutations = apply_voucher("v_observatory", gs)
        assert mutations == {}

    def test_omen_globe_sets_flag(self):
        gs = _gs()
        apply_voucher("v_omen_globe", gs)
        assert gs.get("omen_globe") is True

    def test_unknown_key_returns_empty_mutations(self):
        gs = _gs()
        mutations = apply_voucher("v_nonexistent_xyz", gs)
        assert mutations == {}


# ============================================================================
# apply_voucher — mutations return value
# ============================================================================

class TestApplyVoucherMutations:
    def test_grabber_mutations_contain_hands(self):
        gs = _gs()
        m = apply_voucher("v_grabber", gs)
        assert "round_resets.hands" in m
        assert m["round_resets.hands"] == gs["round_resets"]["hands"]

    def test_overstock_norm_mutations_contain_joker_max(self):
        gs = _gs()
        m = apply_voucher("v_overstock_norm", gs)
        assert "shop.joker_max" in m
        assert m["shop.joker_max"] == 3

    def test_clearance_sale_mutations_contain_discount(self):
        gs = _gs()
        m = apply_voucher("v_clearance_sale", gs)
        assert "discount_percent" in m
        assert m["discount_percent"] == 25

    def test_mutations_match_game_state_after_apply(self):
        """All values in the mutations dict should match game_state after apply."""
        gs = _gs()
        m = apply_voucher("v_wasteful", gs)
        assert m["round_resets.discards"] == gs["round_resets"]["discards"]


# ============================================================================
# All 32 vouchers are handled (no UnhandledVoucher regression)
# ============================================================================

class TestAllVouchersHandled:
    def test_apply_all_vouchers_does_not_raise(self):
        """Calling apply_voucher on every known key should not raise."""
        gs = _gs()
        for key in VOUCHERS:
            try:
                apply_voucher(key, gs)
            except Exception as exc:
                pytest.fail(f"apply_voucher('{key}') raised {exc!r}")

    def test_all_voucher_keys_recognized(self):
        """No voucher in the data should be silently ignored (non-passive ones)."""
        non_passive = {
            "v_overstock_norm", "v_overstock_plus", "v_clearance_sale",
            "v_liquidation", "v_tarot_merchant", "v_tarot_tycoon",
            "v_planet_merchant", "v_planet_tycoon", "v_hone", "v_glow_up",
            "v_magic_trick", "v_illusion", "v_crystal_ball",
            "v_seed_money", "v_money_tree", "v_reroll_surplus",
            "v_reroll_glut", "v_grabber", "v_nacho_tong",
            "v_wasteful", "v_recyclomancy", "v_paint_brush", "v_palette",
            "v_antimatter", "v_hieroglyph", "v_petroglyph",
            "v_directors_cut", "v_retcon",
        }
        for key in non_passive:
            gs = _gs()
            m = apply_voucher(key, gs)
            assert m, f"apply_voucher('{key}') returned empty mutations (expected effect)"
