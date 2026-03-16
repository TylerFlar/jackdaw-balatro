"""Tests for the Card class.

Verifies card creation from prototypes (both string-key and dict APIs),
base field population, ability initialization with post-init fields,
scoring methods, deep-copy isolation, and sort_id auto-increment.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, CardBase, reset_sort_id_counter
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.prototypes import JOKERS


@pytest.fixture(autouse=True)
def _reset_sort_ids():
    """Reset sort_id counter before each test for determinism."""
    reset_sort_id_counter()


# ============================================================================
# CardBase
# ============================================================================

class TestCardBase:
    """CardBase: playing card identity with computed numeric values."""

    def test_ace_of_spades(self):
        b = CardBase.from_card_key("S_A", "Spades", "Ace")
        assert b.suit is Suit.SPADES
        assert b.rank is Rank.ACE
        assert b.id == 14
        assert b.nominal == 11
        assert b.suit_nominal == 0.04
        assert b.suit_nominal_original == 0.004
        assert b.face_nominal == 0.4
        assert b.original_value is Rank.ACE
        assert b.times_played == 0

    def test_king_of_hearts(self):
        b = CardBase.from_card_key("H_K", "Hearts", "King")
        assert b.id == 13
        assert b.nominal == 10
        assert b.face_nominal == 0.3

    def test_two_of_diamonds(self):
        b = CardBase.from_card_key("D_2", "Diamonds", "2")
        assert b.id == 2
        assert b.nominal == 2
        assert b.face_nominal == 0.0
        assert b.suit_nominal == 0.01

    def test_jack_face_nominal(self):
        b = CardBase.from_card_key("C_J", "Clubs", "Jack")
        assert b.face_nominal == 0.1
        assert b.nominal == 10

    def test_ten_not_face(self):
        b = CardBase.from_card_key("S_T", "Spades", "10")
        assert b.face_nominal == 0.0


# ============================================================================
# Card creation and sort_id
# ============================================================================

class TestCardCreation:
    def test_sort_id_auto_increments(self):
        c1, c2, c3 = Card(), Card(), Card()
        assert (c1.sort_id, c2.sort_id, c3.sort_id) == (1, 2, 3)

    def test_sort_id_reset(self):
        Card()
        Card()
        reset_sort_id_counter()
        assert Card().sort_id == 1

    def test_defaults(self):
        c = Card()
        assert c.base is None
        assert c.center_key == "c_base"
        assert c.ability == {}
        assert c.edition is None
        assert c.seal is None
        assert c.debuff is False
        assert c.eternal is False
        assert c.perish_tally == 5


# ============================================================================
# set_ability — string key API (primary interface)
# ============================================================================

class TestSetAbilityByKey:
    """set_ability(center_key: str) looks up P_CENTERS and populates ability."""

    def test_joker_flat_mult(self):
        """j_joker: config.mult=4, no Xmult."""
        c = Card()
        c.set_ability("j_joker")
        assert c.ability["name"] == "Joker"
        assert c.ability["set"] == "Joker"
        assert c.ability["effect"] == "Mult"
        assert c.ability["mult"] == 4
        assert c.ability["x_mult"] == 1  # no Xmult in config → default 1
        assert c.center_key == "j_joker"
        assert c.base_cost == 2

    def test_greedy_joker_nested_extra(self):
        """j_greedy_joker: config.extra.s_mult=3, config.extra.suit='Diamonds'."""
        c = Card()
        c.set_ability("j_greedy_joker")
        assert c.ability["extra"]["s_mult"] == 3
        assert c.ability["extra"]["suit"] == "Diamonds"
        assert c.ability["effect"] == "Suit Mult"

    def test_duo_xmult(self):
        """j_duo: config.Xmult=2 (top-level) → ability.x_mult=2."""
        c = Card()
        c.set_ability("j_duo")
        assert c.ability["x_mult"] == 2
        assert c.ability["name"] == "The Duo"
        assert c.ability["type"] == "Pair"

    def test_ice_cream_extra_chips(self):
        """j_ice_cream: config.extra.chips=100."""
        c = Card()
        c.set_ability("j_ice_cream")
        assert c.ability["extra"]["chips"] == 100
        assert c.ability["extra"]["chip_mod"] == 5

    def test_hands_played_at_create(self):
        """All cards get hands_played_at_create from game state."""
        c = Card()
        c.set_ability("j_joker", hands_played=42)
        assert c.ability["hands_played_at_create"] == 42

    def test_hands_played_at_create_defaults_zero(self):
        c = Card()
        c.set_ability("j_joker")
        assert c.ability["hands_played_at_create"] == 0

    def test_bonus_card_enhancement(self):
        """m_bonus: config.bonus=30."""
        c = Card()
        c.set_base("D_5", "Diamonds", "5")
        c.set_ability("m_bonus")
        assert c.ability["bonus"] == 30
        assert c.get_chip_bonus() == 5 + 30

    def test_invalid_key_raises(self):
        c = Card()
        with pytest.raises(KeyError, match="Unknown center key"):
            c.set_ability("j_nonexistent_joker")


# ============================================================================
# set_ability — special post-init fields (card.lua:308-337)
# ============================================================================

class TestPostInitFields:
    """Joker-specific fields set after the main ability assignment."""

    def test_invisible_joker(self):
        c = Card()
        c.set_ability("j_invisible")
        assert c.ability["invis_rounds"] == 0

    def test_caino(self):
        c = Card()
        c.set_ability("j_caino")
        assert c.ability["caino_xmult"] == 1

    def test_yorick(self):
        c = Card()
        c.set_ability("j_yorick")
        assert c.ability["yorick_discards"] == c.ability["extra"]["discards"]
        assert c.ability["yorick_discards"] > 0  # should be 23

    def test_loyalty_card(self):
        c = Card()
        c.set_ability("j_loyalty_card")
        assert c.ability["loyalty_remaining"] == c.ability["extra"]["every"]
        assert c.ability["loyalty_remaining"] == 5
        assert c.ability["burnt_hand"] == 0

    def test_non_special_joker_no_extra_fields(self):
        """Normal jokers don't get special post-init fields."""
        c = Card()
        c.set_ability("j_joker")
        assert "invis_rounds" not in c.ability
        assert "caino_xmult" not in c.ability
        assert "yorick_discards" not in c.ability
        assert "loyalty_remaining" not in c.ability


# ============================================================================
# Deep copy isolation
# ============================================================================

class TestDeepCopyIsolation:
    """Mutating one card's extra must not affect another card or the prototype."""

    def test_two_cards_from_same_proto(self):
        c1 = Card()
        c1.set_ability("j_greedy_joker")
        c2 = Card()
        c2.set_ability("j_greedy_joker")

        c1.ability["extra"]["s_mult"] = 999
        assert c2.ability["extra"]["s_mult"] == 3  # unaffected

    def test_prototype_unaffected(self):
        c = Card()
        c.set_ability("j_greedy_joker")
        c.ability["extra"]["s_mult"] = 999
        assert JOKERS["j_greedy_joker"].config["extra"]["s_mult"] == 3

    def test_ice_cream_decay_isolated(self):
        """Simulate Ice Cream chip decay — must not affect other instances."""
        c1 = Card()
        c1.set_ability("j_ice_cream")
        c2 = Card()
        c2.set_ability("j_ice_cream")

        # Simulate decay on c1
        c1.ability["extra"]["chips"] -= 5
        assert c1.ability["extra"]["chips"] == 95
        assert c2.ability["extra"]["chips"] == 100  # unaffected

    def test_scalar_extra_not_shared(self):
        """Jokers with scalar extra (not dict) are also independent."""
        c1 = Card()
        c1.set_ability("j_fibonacci")  # extra = 8 (scalar)
        c2 = Card()
        c2.set_ability("j_fibonacci")
        # Scalars are copied by value, so this is inherently safe
        assert c1.ability["extra"] == 8
        assert c2.ability["extra"] == 8


# ============================================================================
# set_ability — dict API (backward compat)
# ============================================================================

class TestSetAbilityByDict:
    """set_ability(dict) still works for custom/test centers."""

    def test_raw_dict(self):
        c = Card()
        c.set_ability({
            "key": "test_joker", "name": "Test", "set": "Joker",
            "effect": "Mult", "config": {"mult": 7}, "cost": 3,
        })
        assert c.ability["name"] == "Test"
        assert c.ability["mult"] == 7
        assert c.center_key == "test_joker"
        assert c.base_cost == 3

    def test_consumeable_config_ref(self):
        c = Card()
        c.set_ability({
            "key": "c_magician", "name": "The Magician", "set": "Tarot",
            "config": {"mod_conv": "m_lucky", "max_highlighted": 2},
            "consumeable": True,
        })
        assert c.ability["consumeable"]["mod_conv"] == "m_lucky"


# ============================================================================
# perma_bonus preservation
# ============================================================================

class TestPermaBonus:
    def test_preserved_across_set_ability(self):
        c = Card()
        c.set_base("S_5", "Spades", "5")
        c.set_ability("c_base")
        c.ability["perma_bonus"] = 15  # Hiker effect
        c.set_ability("c_base")  # re-apply
        assert c.ability["perma_bonus"] == 15
        assert c.get_chip_bonus() == 5 + 15

    def test_preserved_across_key_change(self):
        c = Card()
        c.set_base("H_3", "Hearts", "3")
        c.set_ability("c_base")
        c.ability["perma_bonus"] = 10
        c.set_ability("m_bonus")  # change to Bonus Card
        assert c.ability["perma_bonus"] == 10
        assert c.get_chip_bonus() == 3 + 30 + 10  # nominal + bonus + perma


# ============================================================================
# Scoring methods
# ============================================================================

class TestIsFace:
    """Card.is_face() matching card.lua:964."""

    def test_jack_is_face(self):
        c = Card()
        c.set_base("H_J", "Hearts", "Jack")
        assert c.is_face() is True

    def test_queen_is_face(self):
        c = Card()
        c.set_base("S_Q", "Spades", "Queen")
        assert c.is_face() is True

    def test_king_is_face(self):
        c = Card()
        c.set_base("D_K", "Diamonds", "King")
        assert c.is_face() is True

    def test_ace_is_not_face(self):
        c = Card()
        c.set_base("D_A", "Diamonds", "Ace")
        assert c.is_face() is False

    def test_ten_is_not_face(self):
        c = Card()
        c.set_base("S_T", "Spades", "10")
        assert c.is_face() is False

    def test_five_is_not_face(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        assert c.is_face() is False

    def test_debuffed_is_not_face(self):
        c = Card()
        c.set_base("H_K", "Hearts", "King")
        c.debuff = True
        assert c.is_face() is False

    def test_debuffed_with_from_boss(self):
        """from_boss=True bypasses debuff check (The Plant boss blind)."""
        c = Card()
        c.set_base("H_K", "Hearts", "King")
        c.debuff = True
        assert c.is_face(from_boss=True) is True

    def test_pareidolia_makes_all_face(self):
        """Pareidolia joker: ALL cards are face cards."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        assert c.is_face(pareidolia=True) is True

    def test_pareidolia_ace(self):
        c = Card()
        c.set_base("D_A", "Diamonds", "Ace")
        assert c.is_face(pareidolia=True) is True

    def test_pareidolia_debuffed(self):
        """Debuffed card with Pareidolia: debuff takes precedence."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.debuff = True
        assert c.is_face(pareidolia=True) is False

    def test_pareidolia_debuffed_from_boss(self):
        """from_boss bypasses debuff even with Pareidolia."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.debuff = True
        assert c.is_face(pareidolia=True, from_boss=True) is True

    def test_no_base(self):
        c = Card()
        assert c.is_face() is False


class TestIsSuit:
    """Card.is_suit() matching card.lua:4064."""

    def test_basic_match(self):
        c = Card()
        c.set_base("S_A", "Spades", "Ace")
        assert c.is_suit("Spades") is True
        assert c.is_suit("Hearts") is False

    def test_all_four_suits(self):
        for suit_str in ["Hearts", "Diamonds", "Clubs", "Spades"]:
            c = Card()
            c.set_base(f"{suit_str[0]}_5", suit_str, "5")
            assert c.is_suit(suit_str) is True

    def test_wild_card_matches_all(self):
        """Wild Card matches every suit."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_wild")
        assert c.is_suit("Spades") is True
        assert c.is_suit("Hearts") is True
        assert c.is_suit("Diamonds") is True
        assert c.is_suit("Clubs") is True

    def test_wild_card_debuffed(self):
        """Debuffed Wild Card: debuff check returns False before Wild check."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_wild")
        c.debuff = True
        assert c.is_suit("Spades") is False

    def test_wild_card_flush_calc_debuffed(self):
        """In flush_calc mode: Wild Card still matches if debuffed? No.

        Source card.lua:4069: Wild Card matches in flush_calc only if NOT debuffed.
        """
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_wild")
        c.debuff = True
        assert c.is_suit("Spades", flush_calc=True) is False

    def test_stone_card_never_matches(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_stone")
        assert c.is_suit("Hearts") is False

    def test_smeared_red_suits(self):
        """Smeared: Heart matches Diamond (both red)."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("c_base")
        assert c.is_suit("Diamonds", smeared=True) is True
        assert c.is_suit("Hearts", smeared=True) is True

    def test_smeared_black_suits(self):
        """Smeared: Club matches Spade (both black)."""
        c = Card()
        c.set_base("C_5", "Clubs", "5")
        c.set_ability("c_base")
        assert c.is_suit("Spades", smeared=True) is True
        assert c.is_suit("Clubs", smeared=True) is True

    def test_smeared_red_not_black(self):
        """Smeared: Heart does NOT match Spade (red vs black)."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("c_base")
        assert c.is_suit("Spades", smeared=True) is False

    def test_smeared_black_not_red(self):
        c = Card()
        c.set_base("S_5", "Spades", "5")
        c.set_ability("c_base")
        assert c.is_suit("Diamonds", smeared=True) is False

    def test_bypass_debuff(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("c_base")
        c.debuff = True
        assert c.is_suit("Hearts") is False
        assert c.is_suit("Hearts", bypass_debuff=True) is True

    def test_suit_enum_input(self):
        """Accepts Suit enum as well as string."""
        from jackdaw.engine.data.enums import Suit
        c = Card()
        c.set_base("S_A", "Spades", "Ace")
        c.set_ability("c_base")
        assert c.is_suit(Suit.SPADES) is True
        assert c.is_suit(Suit.HEARTS) is False

    def test_no_base(self):
        c = Card()
        assert c.is_suit("Hearts") is False


class TestScoringMethods:
    def test_get_id_stone_card(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_stone")
        assert c.get_id() == -1

    def test_get_chip_x_mult_glass(self):
        c = Card()
        c.set_ability("m_glass")
        assert c.get_chip_x_mult() == 2

    def test_stone_card_ignores_base_nominal(self):
        c = Card()
        c.set_base("H_A", "Hearts", "Ace")
        c.set_ability("m_stone")
        assert c.get_chip_bonus() == 50

    def test_repr(self):
        c = Card()
        c.set_ability("j_joker")
        assert "Joker" in repr(c)


# ============================================================================
# Stickers
# ============================================================================

class TestStickers:
    def test_set_eternal(self):
        c = Card()
        c.set_eternal(True)
        assert c.eternal is True

    def test_set_perishable_resets_tally(self):
        c = Card()
        c.perish_tally = 0
        c.set_perishable(True)
        assert c.perish_tally == 5

    def test_set_edition(self):
        c = Card()
        c.set_edition({"foil": True})
        assert c.edition == {"foil": True}

    def test_set_seal(self):
        c = Card()
        c.set_seal("Red")
        assert c.seal == "Red"


# ============================================================================
# set_cost
# ============================================================================

class TestSetCost:
    """Card.set_cost() matching card.lua:369."""

    def _make_joker(self, key: str = "j_joker") -> Card:
        """Helper: create a joker with set_ability."""
        c = Card()
        c.set_ability(key)
        return c

    # -- Base formula --

    def test_base_cost_no_modifiers(self):
        """j_joker: base_cost=2, no inflation/discount/edition."""
        c = self._make_joker("j_joker")
        c.set_cost()
        # floor((2 + 0 + 0.5) * 100/100) = floor(2.5) = 2
        assert c.cost == 2

    def test_base_cost_5(self):
        """j_greedy_joker: base_cost=5."""
        c = self._make_joker("j_greedy_joker")
        c.set_cost()
        # floor((5 + 0 + 0.5) * 100/100) = floor(5.5) = 5
        assert c.cost == 5

    def test_sell_cost_is_half(self):
        """sell_cost = max(1, floor(cost/2))."""
        c = self._make_joker("j_greedy_joker")
        c.set_cost()
        # cost=5, sell = max(1, floor(5/2)) = max(1, 2) = 2
        assert c.sell_cost == 2

    def test_sell_cost_minimum_is_1(self):
        """Even cost=1 gives sell_cost=1."""
        c = self._make_joker("j_joker")
        c.set_cost()
        assert c.cost == 2
        # floor(2/2) = 1
        assert c.sell_cost == 1

    # -- Discount --

    def test_25_percent_discount(self):
        """Clearance Sale: 25% discount on cost=5."""
        c = self._make_joker("j_greedy_joker")
        c.set_cost(discount_percent=25)
        # floor((5 + 0 + 0.5) * 75/100) = floor(4.125) = 4
        assert c.cost == 4

    def test_50_percent_discount(self):
        """Liquidation: 50% discount on cost=5."""
        c = self._make_joker("j_greedy_joker")
        c.set_cost(discount_percent=50)
        # floor((5 + 0 + 0.5) * 50/100) = floor(2.75) = 2
        assert c.cost == 2

    def test_discount_minimum_1(self):
        """Discount can't reduce cost below 1."""
        c = self._make_joker("j_joker")  # base_cost=2
        c.set_cost(discount_percent=50)
        # floor((2 + 0 + 0.5) * 50/100) = floor(1.25) = 1
        assert c.cost == 1

    # -- Edition surcharges --

    def test_foil_surcharge(self):
        """Foil edition adds +2 to cost."""
        c = self._make_joker("j_greedy_joker")
        c.set_edition({"foil": True})
        c.set_cost()
        # floor((5 + 2 + 0.5) * 100/100) = floor(7.5) = 7
        assert c.cost == 7

    def test_holo_surcharge(self):
        """Holographic edition adds +3."""
        c = self._make_joker("j_greedy_joker")
        c.set_edition({"holo": True})
        c.set_cost()
        # floor((5 + 3 + 0.5) * 100/100) = floor(8.5) = 8
        assert c.cost == 8

    def test_polychrome_surcharge(self):
        """Polychrome edition adds +5."""
        c = self._make_joker("j_greedy_joker")
        c.set_edition({"polychrome": True})
        c.set_cost()
        # floor((5 + 5 + 0.5) * 100/100) = floor(10.5) = 10
        assert c.cost == 10

    def test_negative_surcharge(self):
        """Negative edition adds +5."""
        c = self._make_joker("j_greedy_joker")
        c.set_edition({"negative": True})
        c.set_cost()
        assert c.cost == 10

    def test_edition_with_discount(self):
        """Foil + 25% discount on cost=5."""
        c = self._make_joker("j_greedy_joker")
        c.set_edition({"foil": True})
        c.set_cost(discount_percent=25)
        # floor((5 + 2 + 0.5) * 75/100) = floor(5.625) = 5
        assert c.cost == 5

    # -- Inflation --

    def test_inflation(self):
        """Inflation adds to extra_cost."""
        c = self._make_joker("j_greedy_joker")
        c.set_cost(inflation=3)
        # floor((5 + 3 + 0.5) * 100/100) = floor(8.5) = 8
        assert c.cost == 8

    def test_inflation_with_discount(self):
        c = self._make_joker("j_greedy_joker")
        c.set_cost(inflation=3, discount_percent=25)
        # floor((5 + 3 + 0.5) * 75/100) = floor(6.375) = 6
        assert c.cost == 6

    # -- Rental override --

    def test_rental_override(self):
        """Rental cards always cost 1."""
        c = self._make_joker("j_greedy_joker")
        c.set_rental(True)
        c.set_cost()
        assert c.cost == 1
        assert c.sell_cost == 1  # floor(1/2) = 0, clamped to 1

    def test_rental_via_ability_flag(self):
        """ability.rental also triggers the override."""
        c = self._make_joker("j_greedy_joker")
        c.ability["rental"] = True
        c.set_cost()
        assert c.cost == 1

    # -- Astronomer --

    def test_astronomer_planet(self):
        """Astronomer makes planet cards cost 0."""
        c = Card()
        c.set_ability({
            "key": "c_pluto", "name": "Pluto", "set": "Planet",
            "config": {"hand_type": "High Card"}, "cost": 3,
        })
        c.set_cost(has_astronomer=True)
        assert c.cost == 0

    def test_astronomer_celestial_booster(self):
        """Astronomer makes celestial boosters cost 0."""
        c = Card()
        c.set_ability({
            "key": "p_celestial_normal_1", "name": "Celestial Pack",
            "set": "Booster", "config": {}, "cost": 4,
        })
        c.set_cost(has_astronomer=True)
        assert c.cost == 0

    def test_astronomer_non_planet(self):
        """Astronomer doesn't affect joker costs."""
        c = self._make_joker("j_joker")
        c.set_cost(has_astronomer=True)
        assert c.cost == 2  # unchanged

    # -- Couponed --

    def test_couponed(self):
        """Couponed by tag: cost = 0, but sell_cost calculated before."""
        c = self._make_joker("j_greedy_joker")
        c.set_cost(is_couponed=True)
        assert c.cost == 0
        # sell_cost is calculated before couponed override
        assert c.sell_cost == 2  # floor(5/2) = 2

    # -- Booster ante scaling --

    def test_booster_ante_scaling(self):
        """Booster cost += ante - 1 when modifier active."""
        c = Card()
        c.set_ability({
            "key": "p_arcana_normal_1", "name": "Arcana Pack",
            "set": "Booster", "config": {}, "cost": 4,
        })
        c.set_cost(ante=5, booster_ante_scaling=True)
        # base formula: floor((4 + 0 + 0.5) * 100/100) = 4
        # then + (5 - 1) = 4 + 4 = 8
        assert c.cost == 8

    def test_booster_ante_scaling_non_booster(self):
        """Ante scaling doesn't apply to non-boosters."""
        c = self._make_joker("j_joker")
        c.set_cost(ante=5, booster_ante_scaling=True)
        assert c.cost == 2  # unchanged

    # -- extra_value (Egg/Gift Card) --

    def test_extra_value_adds_to_sell(self):
        """Egg joker's extra_value increases sell_cost."""
        c = self._make_joker("j_greedy_joker")
        c.ability["extra_value"] = 9  # 3 rounds * 3
        c.set_cost()
        # cost=5, base sell = floor(5/2) = 2, + 9 = 11
        assert c.sell_cost == 11

    # -- Compound scenario --

    def test_foil_inflation_discount(self):
        """Foil + inflation=2 + 25% discount on base_cost=8."""
        c = self._make_joker("j_stencil")  # cost=8, uncommon
        c.set_edition({"foil": True})
        c.set_cost(inflation=2, discount_percent=25)
        # extra_cost = 2 + 2 = 4
        # floor((8 + 4 + 0.5) * 75/100) = floor(9.375) = 9
        assert c.cost == 9
        assert c.sell_cost == 4  # floor(9/2) = 4
