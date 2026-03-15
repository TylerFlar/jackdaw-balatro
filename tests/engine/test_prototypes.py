"""Tests for prototype data loading and typed access.

Verifies counts, spot-checks specific prototypes against known Lua values,
and validates derived pools.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.data.prototypes import (
    BACKS,
    BLINDS,
    BOOSTERS,
    CENTER_POOLS,
    EDITIONS,
    ENHANCEMENTS,
    JOKER_RARITY_POOLS,
    JOKERS,
    PLANETS,
    PLAYING_CARDS,
    SEALS,
    SPECTRALS,
    STAKES,
    TAGS,
    TAROTS,
    VOUCHERS,
    BlindProto,
    JokerProto,
    PlayingCardProto,
    TagProto,
)

# ============================================================================
# Count verification
# ============================================================================

class TestCounts:
    """Verify expected prototype counts per category."""

    def test_jokers(self):
        assert len(JOKERS) == 150

    def test_tarots(self):
        assert len(TAROTS) == 22

    def test_planets(self):
        assert len(PLANETS) == 12

    def test_spectrals(self):
        assert len(SPECTRALS) == 18

    def test_vouchers(self):
        assert len(VOUCHERS) == 32

    def test_backs(self):
        assert len(BACKS) == 16  # 15 playable + Challenge Deck

    def test_boosters(self):
        assert len(BOOSTERS) == 32

    def test_enhancements(self):
        assert len(ENHANCEMENTS) == 8

    def test_editions(self):
        assert len(EDITIONS) == 5

    def test_playing_cards(self):
        assert len(PLAYING_CARDS) == 52

    def test_blinds(self):
        assert len(BLINDS) == 30

    def test_tags(self):
        assert len(TAGS) == 24

    def test_stakes(self):
        assert len(STAKES) == 8

    def test_seals(self):
        assert len(SEALS) == 4


# ============================================================================
# Joker rarity pool verification
# ============================================================================

class TestRarityPools:
    """Verify joker rarity distribution."""

    def test_common_count(self):
        assert len(JOKER_RARITY_POOLS[1]) == 61

    def test_uncommon_count(self):
        assert len(JOKER_RARITY_POOLS[2]) == 64

    def test_rare_count(self):
        assert len(JOKER_RARITY_POOLS[3]) == 20

    def test_legendary_count(self):
        assert len(JOKER_RARITY_POOLS[4]) == 5

    def test_pools_sum_to_total(self):
        total = sum(len(pool) for pool in JOKER_RARITY_POOLS.values())
        assert total == 150

    def test_all_keys_are_valid_jokers(self):
        for rarity, keys in JOKER_RARITY_POOLS.items():
            for key in keys:
                assert key in JOKERS, f"{key} not in JOKERS"
                assert JOKERS[key].rarity == rarity

    def test_pools_sorted_by_order(self):
        for keys in JOKER_RARITY_POOLS.values():
            orders = [JOKERS[k].order for k in keys]
            assert orders == sorted(orders)

    def test_legendary_jokers(self):
        expected = {"j_chicot", "j_perkeo", "j_triboulet", "j_yorick", "j_caino"}
        actual = set(JOKER_RARITY_POOLS[4])
        assert actual == expected


# ============================================================================
# Spot-check specific prototypes
# ============================================================================

class TestJokerSpotChecks:
    """Verify specific joker prototypes match Lua source values."""

    def test_joker_basic(self):
        j = JOKERS["j_joker"]
        assert isinstance(j, JokerProto)
        assert j.name == "Joker"
        assert j.rarity == 1
        assert j.cost == 2
        assert j.order == 1
        assert j.effect == "Mult"
        assert j.config == {"mult": 4}
        assert j.blueprint_compat is True
        assert j.discovered is True  # only joker that starts discovered

    def test_greedy_joker_nested_config(self):
        j = JOKERS["j_greedy_joker"]
        assert j.name == "Greedy Joker"
        assert j.rarity == 1
        assert j.cost == 5
        assert j.config["extra"]["s_mult"] == 3
        assert j.config["extra"]["suit"] == "Diamonds"

    def test_four_fingers_no_blueprint(self):
        j = JOKERS["j_four_fingers"]
        assert j.blueprint_compat is False
        assert j.rarity == 2

    def test_gros_michel_pool_flag(self):
        j = JOKERS["j_gros_michel"]
        assert j.no_pool_flag == "gros_michel_extinct"
        assert j.eternal_compat is False

    def test_cavendish_pool_flag(self):
        j = JOKERS["j_cavendish"]
        assert j.yes_pool_flag == "gros_michel_extinct"

    def test_golden_ticket_enhancement_gate(self):
        j = JOKERS["j_ticket"]
        assert j.enhancement_gate == "m_gold"
        assert j.unlocked is False

    def test_steel_joker_float_config(self):
        j = JOKERS["j_steel_joker"]
        assert j.config["extra"] == pytest.approx(0.2)
        assert j.enhancement_gate == "m_steel"

    def test_loyalty_card_complex_config(self):
        j = JOKERS["j_loyalty_card"]
        assert j.config["extra"]["Xmult"] == 4
        assert j.config["extra"]["every"] == 5

    def test_mr_bones_unlock_condition(self):
        j = JOKERS["j_mr_bones"]
        assert j.unlock_condition is not None
        assert j.unlock_condition["type"] == "c_losses"
        assert j.perishable_compat is True
        assert j.eternal_compat is False


class TestConsumableSpotChecks:
    """Verify tarot, planet, and spectral prototypes."""

    def test_magician_tarot(self):
        t = TAROTS["c_magician"]
        assert t.name == "The Magician"
        assert t.cost == 3
        assert t.config["mod_conv"] == "m_lucky"
        assert t.config["max_highlighted"] == 2

    def test_fool_tarot(self):
        t = TAROTS["c_fool"]
        assert t.name == "The Fool"

    def test_mercury_planet(self):
        p = PLANETS["c_mercury"]
        assert p.name == "Mercury"
        assert p.config["hand_type"] == "Pair"

    def test_pluto_planet(self):
        p = PLANETS["c_pluto"]
        assert p.name == "Pluto"
        assert p.config["hand_type"] == "High Card"

    def test_aura_spectral(self):
        s = SPECTRALS["c_aura"]
        assert s.name == "Aura"

    def test_soul_hidden(self):
        s = SPECTRALS["c_soul"]
        assert s.name == "The Soul"
        assert s.hidden is True

    def test_black_hole_spectral(self):
        s = SPECTRALS["c_black_hole"]
        assert s.name == "Black Hole"
        assert s.hidden is True


class TestVoucherSpotChecks:
    """Verify voucher prototypes and prerequisite chains."""

    def test_overstock(self):
        v = VOUCHERS["v_overstock_norm"]
        assert v.name == "Overstock"
        assert v.cost == 10
        assert v.requires == []

    def test_overstock_plus_requires(self):
        v = VOUCHERS["v_overstock_plus"]
        assert v.name == "Overstock Plus"
        assert "v_overstock_norm" in v.requires

    def test_clearance_sale(self):
        v = VOUCHERS["v_clearance_sale"]
        assert v.config["extra"] == 25  # 25% discount

    def test_all_voucher_costs_are_10(self):
        for key, v in VOUCHERS.items():
            assert v.cost == 10, f"{key} cost is {v.cost}, expected 10"


class TestBackSpotChecks:
    """Verify deck/back prototypes."""

    def test_red_deck(self):
        b = BACKS["b_red"]
        assert b.name == "Red Deck"
        assert b.config == {"discards": 1}

    def test_abandoned_deck(self):
        b = BACKS["b_abandoned"]
        assert b.config.get("remove_faces") is True

    def test_plasma_deck(self):
        b = BACKS["b_plasma"]
        assert b.config.get("ante_scaling") == 2

    def test_erratic_deck(self):
        b = BACKS["b_erratic"]
        assert b.config.get("randomize_rank_suit") is True

    def test_checkered_deck_empty_config(self):
        b = BACKS["b_checkered"]
        # Checkered deck effect is hardcoded by name, not in config
        assert b.config == {}


# ============================================================================
# Playing card verification
# ============================================================================

class TestPlayingCards:
    """Verify playing card computed fields."""

    def test_ace_of_spades(self):
        c = PLAYING_CARDS["S_A"]
        assert isinstance(c, PlayingCardProto)
        assert c.suit == "Spades"
        assert c.rank == "Ace"
        assert c.id == 14
        assert c.nominal == 11
        assert c.suit_nominal == 0.04
        assert c.face_nominal == 0  # Ace is not a face card

    def test_king_of_hearts(self):
        c = PLAYING_CARDS["H_K"]
        assert c.suit == "Hearts"
        assert c.rank == "King"
        assert c.id == 13
        assert c.nominal == 10
        assert c.face_nominal == 1

    def test_two_of_diamonds(self):
        c = PLAYING_CARDS["D_2"]
        assert c.suit == "Diamonds"
        assert c.rank == "2"
        assert c.id == 2
        assert c.nominal == 2
        assert c.suit_nominal == 0.01
        assert c.face_nominal == 0

    def test_ten_of_clubs(self):
        c = PLAYING_CARDS["C_T"]
        assert c.rank == "10"
        assert c.id == 10
        assert c.nominal == 10
        assert c.face_nominal == 0

    def test_all_four_suits_present(self):
        suits = {c.suit for c in PLAYING_CARDS.values()}
        assert suits == {"Spades", "Hearts", "Clubs", "Diamonds"}

    def test_13_cards_per_suit(self):
        for suit in ["Spades", "Hearts", "Clubs", "Diamonds"]:
            count = sum(1 for c in PLAYING_CARDS.values() if c.suit == suit)
            assert count == 13, f"{suit} has {count} cards"

    def test_ids_span_2_to_14(self):
        ids = {c.id for c in PLAYING_CARDS.values()}
        assert ids == set(range(2, 15))

    def test_face_cards_are_jqk(self):
        faces = [c for c in PLAYING_CARDS.values() if c.face_nominal == 1]
        assert len(faces) == 12  # 3 face cards × 4 suits
        assert all(c.rank in ("Jack", "Queen", "King") for c in faces)


# ============================================================================
# Blind verification
# ============================================================================

class TestBlinds:
    """Verify blind prototypes."""

    def test_small_blind(self):
        b = BLINDS["bl_small"]
        assert isinstance(b, BlindProto)
        assert b.name == "Small Blind"
        assert b.mult == 1
        assert b.dollars == 3
        assert b.boss is None

    def test_big_blind(self):
        b = BLINDS["bl_big"]
        assert b.mult == 1.5
        assert b.dollars == 4
        assert b.boss is None

    def test_the_hook_boss(self):
        b = BLINDS["bl_hook"]
        assert b.name == "The Hook"
        assert b.mult == 2
        assert b.boss is not None
        assert b.boss["min"] == 1
        assert b.boss["max"] == 10

    def test_the_wall_high_mult(self):
        b = BLINDS["bl_wall"]
        assert b.mult == 4

    def test_showdown_blinds(self):
        showdowns = [
            b for b in BLINDS.values()
            if b.boss and b.boss.get("showdown")
        ]
        # Cerulean Bell, Verdant Leaf, Violet Vessel, Amber Acorn, Crimson Heart
        assert len(showdowns) == 5

    def test_suit_debuff_blinds(self):
        suit_blinds = [
            b for b in BLINDS.values()
            if isinstance(b.debuff, dict) and "suit" in b.debuff
        ]
        # Club, Goad(Spades), Head(Hearts), Window(Diamonds)
        assert len(suit_blinds) == 4


# ============================================================================
# Tag verification
# ============================================================================

class TestTags:
    """Verify tag prototypes."""

    def test_uncommon_tag(self):
        t = TAGS["tag_uncommon"]
        assert isinstance(t, TagProto)
        assert t.name == "Uncommon Tag"
        assert t.config["type"] == "store_joker_create"

    def test_negative_tag_requires(self):
        t = TAGS["tag_negative"]
        assert t.requires == "e_negative"
        assert t.min_ante == 2

    def test_d6_tag(self):
        t = TAGS["tag_d_six"]
        assert t.name == "D6 Tag"
        assert t.config["type"] == "shop_start"


# ============================================================================
# Center pools
# ============================================================================

class TestCenterPools:
    """Verify derived center pool lookups."""

    def test_joker_pool_size(self):
        assert len(CENTER_POOLS["Joker"]) == 150

    def test_tarot_planet_combined(self):
        assert len(CENTER_POOLS["Tarot_Planet"]) == 34  # 22 + 12

    def test_all_pool_entries_exist(self):
        for pool_name, keys in CENTER_POOLS.items():
            for key in keys:
                found = (
                    key in JOKERS or key in TAROTS or key in PLANETS
                    or key in SPECTRALS or key in VOUCHERS or key in BACKS
                    or key in BOOSTERS or key in ENHANCEMENTS or key in EDITIONS
                )
                assert found, f"{key} in pool {pool_name!r} not found in any dict"
