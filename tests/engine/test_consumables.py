"""Tests for the consumable dispatch system.

Validates the registry, can_use validation, and dispatch mechanism
without implementing all 52 consumables.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.consumables import (
    _ALL_HAND_TYPES,
    _CONSUMABLE_REGISTRY,
    ConsumableContext,
    ConsumableResult,
    can_use_consumable,
    register_consumable,
    registered_consumables,
    use_consumable,
)
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import JokerContext, calculate_joker


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


def _consumable(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Tarot", **ability_kw}
    return c


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    return c


# ============================================================================
# Test consumable handlers (registered for testing, cleaned up after)
# ============================================================================

_TEST_KEY = "c_test_tarot"
_TEST_KEY_2 = "c_test_spectral"


@pytest.fixture(autouse=True)
def _register_test_consumables():
    """Register test consumables before each test, clean up after."""

    @register_consumable(_TEST_KEY)
    def _test_handler(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        return ConsumableResult(
            enhance=[(h, "m_gold") for h in (ctx.highlighted or [])],
        )

    @register_consumable(_TEST_KEY_2)
    def _test_spectral(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        return ConsumableResult(dollars=20, destroy=ctx.highlighted)

    yield

    _CONSUMABLE_REGISTRY.pop(_TEST_KEY, None)
    _CONSUMABLE_REGISTRY.pop(_TEST_KEY_2, None)


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    def test_register_adds(self):
        assert _TEST_KEY in _CONSUMABLE_REGISTRY

    def test_registered_consumables_sorted(self):
        keys = registered_consumables()
        assert _TEST_KEY in keys
        assert keys == sorted(keys)

    def test_overwrite(self):
        @register_consumable(_TEST_KEY)
        def _new(card: Card, ctx: ConsumableContext) -> ConsumableResult:
            return ConsumableResult(dollars=99)

        c = _consumable(_TEST_KEY)
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.dollars == 99


# ============================================================================
# Dispatch
# ============================================================================


class TestDispatch:
    def test_dispatch_returns_result(self):
        c = _consumable(_TEST_KEY)
        highlighted = [_card("Hearts", "5")]
        ctx = ConsumableContext(card=c, highlighted=highlighted)
        result = use_consumable(c, ctx)
        assert result is not None
        assert result.enhance is not None
        assert len(result.enhance) == 1
        assert result.enhance[0][1] == "m_gold"

    def test_dispatch_with_multiple_highlighted(self):
        c = _consumable(_TEST_KEY)
        highlighted = [_card("Hearts", "5"), _card("Spades", "King")]
        ctx = ConsumableContext(card=c, highlighted=highlighted)
        result = use_consumable(c, ctx)
        assert len(result.enhance) == 2

    def test_unregistered_returns_none(self):
        c = _consumable("c_nonexistent")
        assert use_consumable(c, ConsumableContext(card=c)) is None

    def test_debuffed_returns_none(self):
        c = _consumable(_TEST_KEY)
        c.debuff = True
        assert use_consumable(c, ConsumableContext(card=c)) is None

    def test_spectral_handler(self):
        c = _consumable(_TEST_KEY_2)
        highlighted = [_card("Hearts", "5")]
        ctx = ConsumableContext(card=c, highlighted=highlighted)
        result = use_consumable(c, ctx)
        assert result.dollars == 20
        assert result.destroy == highlighted


# ============================================================================
# ConsumableResult defaults
# ============================================================================


class TestConsumableResult:
    def test_defaults_all_none(self):
        r = ConsumableResult()
        assert r.enhance is None
        assert r.change_suit is None
        assert r.destroy is None
        assert r.create is None
        assert r.dollars == 0
        assert r.level_up is None
        assert r.hand_size_mod == 0
        assert r.money_set is None

    def test_custom_values(self):
        r = ConsumableResult(
            dollars=10,
            hand_size_mod=-1,
            money_set=0,
        )
        assert r.dollars == 10
        assert r.hand_size_mod == -1
        assert r.money_set == 0


# ============================================================================
# can_use_consumable — global blockers
# ============================================================================


class TestCanUseGlobalBlockers:
    def test_cards_in_play_blocks(self):
        c = _consumable("c_hermit")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c, cards_in_play=1) is False

    def test_no_cards_in_play_allows(self):
        c = _consumable("c_hermit")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c, cards_in_play=0) is True


# ============================================================================
# can_use_consumable — always usable
# ============================================================================


class TestCanUseAlwaysUsable:
    def test_hermit_always(self):
        c = _consumable("c_hermit")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c) is True

    def test_temperance_always(self):
        c = _consumable("c_temperance")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c) is True

    def test_black_hole_always(self):
        c = _consumable("c_black_hole")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c) is True


# ============================================================================
# can_use_consumable — planets
# ============================================================================


class TestCanUsePlanets:
    def test_mercury_always(self):
        c = _consumable("c_mercury")
        c.ability["consumeable"] = {"hand_type": "Pair"}
        assert can_use_consumable(c) is True

    def test_jupiter_always(self):
        c = _consumable("c_jupiter")
        c.ability["consumeable"] = {"hand_type": "Flush"}
        assert can_use_consumable(c) is True


# ============================================================================
# can_use_consumable — highlighted card requirements
# ============================================================================


class TestCanUseHighlighted:
    def test_chariot_needs_1(self):
        """Chariot: max_highlighted=1, needs exactly 1."""
        c = _consumable("c_chariot")
        c.ability["consumeable"] = {"max_highlighted": 1, "mod_conv": "m_steel"}
        assert can_use_consumable(c, highlighted=[]) is False
        assert can_use_consumable(c, highlighted=[_card("Hearts", "5")]) is True
        assert (
            can_use_consumable(c, highlighted=[_card("Hearts", "5"), _card("Spades", "3")]) is False
        )

    def test_empress_needs_1_or_2(self):
        """Empress: max_highlighted=2, min defaults to 1."""
        c = _consumable("c_empress")
        c.ability["consumeable"] = {"max_highlighted": 2, "mod_conv": "m_mult"}
        assert can_use_consumable(c, highlighted=[]) is False
        assert can_use_consumable(c, highlighted=[_card("Hearts", "5")]) is True
        assert (
            can_use_consumable(c, highlighted=[_card("Hearts", "5"), _card("Spades", "3")]) is True
        )

    def test_death_needs_exactly_2(self):
        """Death: min_highlighted=2, max_highlighted=2."""
        c = _consumable("c_death")
        c.ability["consumeable"] = {
            "max_highlighted": 2,
            "min_highlighted": 2,
            "mod_conv": "card",
        }
        assert can_use_consumable(c, highlighted=[_card("Hearts", "5")]) is False
        assert (
            can_use_consumable(c, highlighted=[_card("Hearts", "5"), _card("Spades", "3")]) is True
        )

    def test_star_needs_1_to_3(self):
        """Star: max_highlighted=3, suit conversion."""
        c = _consumable("c_star")
        c.ability["consumeable"] = {
            "max_highlighted": 3,
            "suit_conv": "Diamonds",
            "mod_num": 3,
        }
        assert can_use_consumable(c, highlighted=[]) is False
        assert can_use_consumable(c, highlighted=[_card("Hearts", "5")]) is True
        cards3 = [_card("Hearts", "5"), _card("Spades", "3"), _card("Clubs", "King")]
        assert can_use_consumable(c, highlighted=cards3) is True


# ============================================================================
# can_use_consumable — slot requirements
# ============================================================================


class TestCanUseSlots:
    def test_emperor_needs_consumable_slot(self):
        c = _consumable("c_emperor")
        c.ability["consumeable"] = {"tarots": 2}
        # 1 consumable, limit 2 → 1 free slot
        assert (
            can_use_consumable(
                c,
                consumables=[c],
                consumable_limit=2,
            )
            is True
        )
        # Using self frees slot: [c, other] with limit 2 → effective=1 < 2
        other = _consumable("c_foo")
        assert (
            can_use_consumable(
                c,
                consumables=[c, other],
                consumable_limit=2,
            )
            is True
        )

    def test_emperor_no_slot(self):
        c = _consumable("c_emperor")
        c.ability["consumeable"] = {"tarots": 2}
        other1 = _consumable("c_foo")
        other2 = _consumable("c_bar")
        # c not in consumables → no self-free, 2/2 full
        assert (
            can_use_consumable(
                c,
                consumables=[other1, other2],
                consumable_limit=2,
            )
            is False
        )

    def test_judgement_needs_joker_slot(self):
        c = _consumable("c_judgement")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c, jokers=[], joker_limit=5) is True
        jokers_5 = [_joker(f"j_{i}") for i in range(5)]
        assert (
            can_use_consumable(
                c,
                jokers=jokers_5,
                joker_limit=5,
            )
            is False
        )


# ============================================================================
# can_use_consumable — eligible joker
# ============================================================================


class TestCanUseEligibleJoker:
    def test_wheel_needs_editionless_joker(self):
        c = _consumable("c_wheel_of_fortune")
        c.ability["consumeable"] = {}
        j = _joker("j_joker", mult=4)
        assert can_use_consumable(c, jokers=[j]) is True

    def test_wheel_all_jokers_have_editions(self):
        c = _consumable("c_wheel_of_fortune")
        c.ability["consumeable"] = {}
        j = _joker("j_joker", mult=4)
        j.edition = {"foil": True}
        assert can_use_consumable(c, jokers=[j]) is False


# ============================================================================
# can_use_consumable — hand card requirements
# ============================================================================


class TestCanUseHandCards:
    def test_familiar_needs_more_than_1_card(self):
        c = _consumable("c_familiar")
        c.ability["consumeable"] = {"extra": 3, "remove_card": True}
        assert (
            can_use_consumable(
                c,
                hand_cards=[_card("Hearts", "5")],
            )
            is False
        )
        assert (
            can_use_consumable(
                c,
                hand_cards=[_card("Hearts", "5"), _card("Spades", "3")],
            )
            is True
        )

    def test_immolate_needs_more_than_1(self):
        c = _consumable("c_immolate")
        c.ability["consumeable"] = {
            "extra": {"destroy": 5, "dollars": 20},
            "remove_card": True,
        }
        assert can_use_consumable(c, hand_cards=[]) is False
        assert (
            can_use_consumable(
                c,
                hand_cards=[_card("Hearts", "5"), _card("Spades", "3")],
            )
            is True
        )


# ============================================================================
# can_use_consumable — Aura
# ============================================================================


class TestCanUseAura:
    def test_aura_1_card_no_edition(self):
        c = _consumable("c_aura")
        c.ability["consumeable"] = {}
        card = _card("Hearts", "5")
        assert can_use_consumable(c, highlighted=[card]) is True

    def test_aura_1_card_with_edition(self):
        c = _consumable("c_aura")
        c.ability["consumeable"] = {}
        card = _card("Hearts", "5")
        card.edition = {"foil": True}
        assert can_use_consumable(c, highlighted=[card]) is False

    def test_aura_0_cards(self):
        c = _consumable("c_aura")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c, highlighted=[]) is False


# ============================================================================
# can_use_consumable — Fool needs last_tarot_planet
# ============================================================================


class TestCanUseFool:
    def test_fool_with_last_tarot(self):
        c = _consumable("c_fool")
        c.ability["consumeable"] = {}
        assert (
            can_use_consumable(
                c,
                consumables=[c],
                consumable_limit=2,
                game_state={"last_tarot_planet": "c_star"},
            )
            is True
        )

    def test_fool_without_last_tarot(self):
        c = _consumable("c_fool")
        c.ability["consumeable"] = {}
        assert (
            can_use_consumable(
                c,
                consumables=[c],
                consumable_limit=2,
            )
            is False
        )


# ============================================================================
# Enhancement tarots
# ============================================================================


class TestEnhancementTarots:
    """8 enhancement tarots: change card enhancement via Card.enhance()."""

    def test_chariot_enhances_to_steel(self):
        c = _consumable("c_chariot")
        ace = _card("Spades", "Ace")
        ctx = ConsumableContext(card=c, highlighted=[ace])
        result = use_consumable(c, ctx)
        assert result is not None
        assert len(result.enhance) == 1
        assert result.enhance[0] == (ace, "m_steel")

    def test_magician_enhances_two_cards(self):
        c = _consumable("c_magician")
        c1 = _card("Hearts", "5")
        c2 = _card("Spades", "King")
        ctx = ConsumableContext(card=c, highlighted=[c1, c2])
        result = use_consumable(c, ctx)
        assert len(result.enhance) == 2
        assert result.enhance[0][1] == "m_lucky"
        assert result.enhance[1][1] == "m_lucky"

    def test_devil_enhances_to_gold(self):
        c = _consumable("c_devil")
        card = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.enhance[0][1] == "m_gold"

    def test_tower_enhances_to_stone(self):
        c = _consumable("c_tower")
        card = _card("Diamonds", "3")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.enhance[0][1] == "m_stone"

    def test_justice_enhances_to_glass(self):
        c = _consumable("c_justice")
        card = _card("Clubs", "Queen")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.enhance[0][1] == "m_glass"

    def test_empress_enhances_to_mult(self):
        c = _consumable("c_empress")
        card = _card("Hearts", "7")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.enhance[0][1] == "m_mult"

    def test_heirophant_enhances_to_bonus(self):
        c = _consumable("c_heirophant")
        card = _card("Spades", "9")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.enhance[0][1] == "m_bonus"

    def test_lovers_enhances_to_wild(self):
        c = _consumable("c_lovers")
        card = _card("Diamonds", "Jack")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.enhance[0][1] == "m_wild"


# ============================================================================
# Card.enhance() — preserves base, edition, seal, perma_bonus
# ============================================================================


class TestCardEnhance:
    """Card.enhance() changes enhancement but preserves identity."""

    def test_enhance_changes_effect(self):
        ace = _card("Spades", "Ace")
        ace.enhance("m_glass")
        assert ace.ability["name"] == "Glass Card"
        assert ace.ability["effect"] == "Glass Card"

    def test_enhance_preserves_base(self):
        ace = _card("Spades", "Ace")
        ace.enhance("m_glass")
        assert ace.base.suit.value == "Spades"
        assert ace.base.rank.value == "Ace"
        assert ace.base.id == 14
        assert ace.base.nominal == 11

    def test_enhance_preserves_edition(self):
        ace = _card("Spades", "Ace")
        ace.set_edition({"foil": True})
        ace.enhance("m_glass")
        assert ace.edition is not None
        assert ace.edition["foil"] is True
        assert ace.edition["chips"] == 50

    def test_enhance_preserves_seal(self):
        ace = _card("Spades", "Ace")
        ace.set_seal("Red")
        ace.enhance("m_steel")
        assert ace.seal == "Red"

    def test_enhance_preserves_perma_bonus(self):
        ace = _card("Spades", "Ace")
        ace.ability["perma_bonus"] = 15  # from Hiker
        ace.enhance("m_gold")
        assert ace.ability["perma_bonus"] == 15

    def test_enhance_overwrites_previous_enhancement(self):
        """Enhancing a Bonus Card to Glass → becomes Glass."""
        card = _card("Hearts", "5", enhancement="m_bonus")
        assert card.ability["effect"] == "Bonus Card"
        card.enhance("m_glass")
        assert card.ability["effect"] == "Glass Card"

    def test_enhance_preserves_bonus(self):
        """Accumulated bonus from Bonus Card enhancement is preserved."""
        card = _card("Hearts", "5", enhancement="m_bonus")
        old_bonus = card.ability.get("bonus", 0)
        card.enhance("m_glass")
        assert card.ability.get("bonus", 0) == old_bonus

    def test_full_roundtrip(self):
        """Enhance with full state: edition + seal + perma_bonus."""
        ace = _card("Spades", "Ace")
        ace.set_edition({"polychrome": True})
        ace.set_seal("Gold")
        ace.ability["perma_bonus"] = 10
        ace.enhance("m_steel")
        # Verify all preserved
        assert ace.base.rank.value == "Ace"
        assert ace.base.suit.value == "Spades"
        assert ace.edition["polychrome"] is True
        assert ace.seal == "Gold"
        assert ace.ability["perma_bonus"] == 10
        assert ace.ability["name"] == "Steel Card"
        assert ace.ability["h_x_mult"] == 1.5  # Steel's effect


# ============================================================================
# Suit-change tarots
# ============================================================================


class TestSuitChangeTarots:
    def test_star_changes_to_diamonds(self):
        c = _consumable("c_star")
        heart = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[heart]))
        assert result is not None
        assert len(result.change_suit) == 1
        assert result.change_suit[0] == (heart, "Diamonds")

    def test_moon_changes_to_clubs(self):
        c = _consumable("c_moon")
        card = _card("Spades", "5")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.change_suit[0][1] == "Clubs"

    def test_sun_changes_to_hearts(self):
        c = _consumable("c_sun")
        card = _card("Clubs", "King")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.change_suit[0][1] == "Hearts"

    def test_world_changes_to_spades(self):
        c = _consumable("c_world")
        card = _card("Diamonds", "10")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[card]))
        assert result.change_suit[0][1] == "Spades"

    def test_star_multiple_cards(self):
        c = _consumable("c_star")
        cards = [_card("Hearts", "5"), _card("Clubs", "King"), _card("Spades", "3")]
        result = use_consumable(c, ConsumableContext(card=c, highlighted=cards))
        assert len(result.change_suit) == 3
        for _, suit in result.change_suit:
            assert suit == "Diamonds"


class TestCardChangeSuit:
    """Card.change_suit() preserves rank, changes suit."""

    def test_heart_to_diamond(self):
        card = _card("Hearts", "Ace")
        card.change_suit("Diamonds")
        assert card.base.suit.value == "Diamonds"
        assert card.base.rank.value == "Ace"
        assert card.base.id == 14
        assert card.base.nominal == 11

    def test_preserves_enhancement(self):
        card = _card("Hearts", "5", enhancement="m_glass")
        card.change_suit("Spades")
        assert card.base.suit.value == "Spades"
        # Enhancement unchanged (set_ability not called)
        assert card.ability["effect"] == "Glass Card"

    def test_suit_nominal_updates(self):
        card = _card("Hearts", "Ace")
        old_nominal = card.base.suit_nominal
        card.change_suit("Diamonds")
        assert card.base.suit_nominal != old_nominal
        assert card.base.suit_nominal == 0.01  # Diamonds


# ============================================================================
# Strength (rank increment)
# ============================================================================


class TestStrength:
    def test_king_to_ace(self):
        c = _consumable("c_strength")
        king = _card("Hearts", "King")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[king]))
        assert result is not None
        assert len(result.change_rank) == 1
        assert result.change_rank[0] == (king, "Ace")

    def test_ace_wraps_to_2(self):
        c = _consumable("c_strength")
        ace = _card("Spades", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[ace]))
        assert result.change_rank[0] == (ace, "2")

    def test_five_to_six(self):
        c = _consumable("c_strength")
        five = _card("Hearts", "5")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[five]))
        assert result.change_rank[0] == (five, "6")

    def test_nine_to_ten(self):
        c = _consumable("c_strength")
        nine = _card("Hearts", "9")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[nine]))
        assert result.change_rank[0] == (nine, "10")

    def test_ten_to_jack(self):
        c = _consumable("c_strength")
        ten = _card("Hearts", "10")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[ten]))
        assert result.change_rank[0] == (ten, "Jack")

    def test_two_cards(self):
        c = _consumable("c_strength")
        cards = [_card("Hearts", "3"), _card("Spades", "King")]
        result = use_consumable(c, ConsumableContext(card=c, highlighted=cards))
        assert len(result.change_rank) == 2
        assert result.change_rank[0][1] == "4"
        assert result.change_rank[1][1] == "Ace"


class TestCardChangeRank:
    """Card.change_rank() preserves suit, changes rank."""

    def test_king_to_ace(self):
        card = _card("Hearts", "King")
        card.change_rank("Ace")
        assert card.base.rank.value == "Ace"
        assert card.base.id == 14
        assert card.base.nominal == 11
        assert card.base.suit.value == "Hearts"

    def test_ace_to_2(self):
        card = _card("Spades", "Ace")
        card.change_rank("2")
        assert card.base.rank.value == "2"
        assert card.base.id == 2
        assert card.base.nominal == 2

    def test_preserves_suit(self):
        card = _card("Diamonds", "5")
        card.change_rank("Jack")
        assert card.base.suit.value == "Diamonds"
        assert card.base.rank.value == "Jack"

    def test_face_nominal_updates(self):
        card = _card("Hearts", "5")
        assert card.base.face_nominal == 0.0
        card.change_rank("King")
        assert card.base.face_nominal == 0.3


# ============================================================================
# Death (copy rightmost onto others)
# ============================================================================


class TestDeath:
    def test_copy_right_to_left(self):
        c = _consumable("c_death")
        left = _card("Hearts", "5")
        right = _card("Spades", "Ace")
        # right has higher sort_id (created second)
        result = use_consumable(
            c,
            ConsumableContext(card=c, highlighted=[left, right]),
        )
        assert result is not None
        assert result.copy_card is not None
        source, target = result.copy_card
        assert source is right
        assert target is left

    def test_copy_preserves_source_identity(self):
        """Death copies right card's full state onto left."""
        c = _consumable("c_death")
        left = _card("Hearts", "5")
        right = _card("Spades", "Ace", enhancement="m_glass")
        right.set_edition({"foil": True})
        right.set_seal("Gold")
        result = use_consumable(
            c,
            ConsumableContext(card=c, highlighted=[left, right]),
        )
        source, target = result.copy_card
        # Source is the Glass Foil Gold Seal Ace of Spades
        assert source.base.rank.value == "Ace"
        assert source.base.suit.value == "Spades"
        assert source.ability["effect"] == "Glass Card"
        assert source.edition["foil"] is True
        assert source.seal == "Gold"
        # Target is the card to be overwritten
        assert target is left


# ============================================================================
# Hanged Man (destroy)
# ============================================================================


class TestHangedMan:
    def test_destroys_highlighted(self):
        c = _consumable("c_hanged_man")
        cards = [_card("Hearts", "5"), _card("Spades", "King")]
        result = use_consumable(
            c,
            ConsumableContext(card=c, highlighted=cards),
        )
        assert result is not None
        assert result.destroy is not None
        assert len(result.destroy) == 2
        assert result.destroy[0] is cards[0]
        assert result.destroy[1] is cards[1]

    def test_destroys_single(self):
        c = _consumable("c_hanged_man")
        card = _card("Hearts", "Ace")
        result = use_consumable(
            c,
            ConsumableContext(card=c, highlighted=[card]),
        )
        assert len(result.destroy) == 1


# ============================================================================
# Economy tarots
# ============================================================================


class TestHermit:
    def test_gain_less_than_cap(self):
        """Hermit with $15 → gain $15 (under $20 cap)."""
        c = _consumable("c_hermit")
        c.set_ability("c_hermit")
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                game_state={"dollars": 15},
            ),
        )
        assert result is not None
        assert result.dollars == 15

    def test_gain_capped_at_20(self):
        """Hermit with $30 → gain $20 (capped)."""
        c = _consumable("c_hermit")
        c.set_ability("c_hermit")
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                game_state={"dollars": 30},
            ),
        )
        assert result.dollars == 20

    def test_gain_zero_when_broke(self):
        """Hermit with $0 → gain $0."""
        c = _consumable("c_hermit")
        c.set_ability("c_hermit")
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                game_state={"dollars": 0},
            ),
        )
        assert result.dollars == 0

    def test_no_game_state_defaults_to_zero(self):
        c = _consumable("c_hermit")
        c.set_ability("c_hermit")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.dollars == 0


class TestTemperance:
    def test_gain_sum_of_sell_costs(self):
        """Temperance with 3 jokers sell_cost 2,3,4 → gain $9."""
        c = _consumable("c_temperance")
        c.set_ability("c_temperance")
        j1 = _joker("j_joker")
        j1.sell_cost = 2
        j2 = _joker("j_joker")
        j2.sell_cost = 3
        j3 = _joker("j_joker")
        j3.sell_cost = 4
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                jokers=[j1, j2, j3],
            ),
        )
        assert result.dollars == 9

    def test_gain_capped_at_50(self):
        """Temperance: sum of joker sell costs capped at $50."""
        c = _consumable("c_temperance")
        c.set_ability("c_temperance")
        jokers = []
        for _ in range(6):
            j = _joker("j_joker")
            j.sell_cost = 10
            jokers.append(j)
        result = use_consumable(c, ConsumableContext(card=c, jokers=jokers))
        assert result.dollars == 50  # 60 capped to 50

    def test_no_jokers_gains_zero(self):
        c = _consumable("c_temperance")
        c.set_ability("c_temperance")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[]))
        assert result.dollars == 0


# ============================================================================
# Generation tarots
# ============================================================================


class TestFool:
    def test_creates_last_tarot_planet(self):
        """The Fool creates a copy of the last used Tarot/Planet."""
        c = _consumable("c_fool")
        c.set_ability("c_fool")
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                game_state={"last_tarot_planet": "c_star"},
            ),
        )
        assert result is not None
        assert result.create is not None
        assert len(result.create) == 1
        assert result.create[0]["forced_key"] == "c_star"
        assert result.create[0]["type"] == "Tarot_Planet"

    def test_no_last_tarot_returns_empty(self):
        c = _consumable("c_fool")
        c.set_ability("c_fool")
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                game_state={},
            ),
        )
        assert result is not None
        assert result.create is None

    def test_fool_cannot_replicate_itself(self):
        """can_use blocks Fool when last_tarot_planet is 'c_fool'."""
        c = _consumable("c_fool")
        c.ability["consumeable"] = {}
        assert (
            can_use_consumable(
                c,
                consumables=[c],
                consumable_limit=2,
                game_state={"last_tarot_planet": "c_fool"},
            )
            is False
        )


class TestHighPriestess:
    def test_creates_two_planets(self):
        c = _consumable("c_high_priestess")
        c.set_ability("c_high_priestess")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.create is not None
        assert result.create[0]["type"] == "Planet"
        assert result.create[0]["count"] == 2
        assert result.create[0]["seed"] == "pri"


class TestEmperor:
    def test_creates_two_tarots(self):
        c = _consumable("c_emperor")
        c.set_ability("c_emperor")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.create is not None
        assert result.create[0]["type"] == "Tarot"
        assert result.create[0]["count"] == 2
        assert result.create[0]["seed"] == "emp"


class TestJudgement:
    def test_creates_one_joker(self):
        c = _consumable("c_judgement")
        c.set_ability("c_judgement")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.create is not None
        assert result.create[0]["type"] == "Joker"
        assert result.create[0]["count"] == 1
        assert result.create[0]["seed"] == "jud"

    def test_can_use_with_joker_slot(self):
        c = _consumable("c_judgement")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c, jokers=[], joker_limit=5) is True

    def test_cannot_use_without_joker_slot(self):
        c = _consumable("c_judgement")
        c.ability["consumeable"] = {}
        full = [_joker(f"j_{i}") for i in range(5)]
        assert can_use_consumable(c, jokers=full, joker_limit=5) is False


# ============================================================================
# Wheel of Fortune
# ============================================================================


class _ControlledRng:
    """Minimal RNG stub with scripted return values for random() and seed()."""

    def __init__(self, random_values: list[float] | None = None):
        self._vals = iter(random_values or [])
        self._seed_counter = 0.5  # fixed seed value for element()

    def random(self, key: str) -> float:
        return next(self._vals)

    def seed(self, key: str) -> float:
        return self._seed_counter

    def element(self, table: list, seed_val: float) -> tuple:
        # Always pick first element deterministically (mirrors sorted order idx=0)
        return (table[0], 0)

    def shuffle(self, _lst: list, _seed_val: float) -> None:
        # Identity shuffle — preserves insertion order for deterministic testing
        pass


class TestWheelOfFortune:
    def _joker_with_sell(self, key: str, sell: int = 3) -> Card:
        j = _joker(key)
        j.sell_cost = sell
        return j

    def test_success_returns_add_edition(self):
        """Roll < 0.25 → success; add_edition descriptor returned."""
        c = _consumable("c_wheel_of_fortune")
        c.set_ability("c_wheel_of_fortune")
        j = self._joker_with_sell("j_joker")
        # roll=0.1 (< 1/4), edition_poll=0.6 (> 0.5 → holo)
        rng = _ControlledRng([0.1, 0.6])
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["target"] is j
        assert result.add_edition["edition"] == {"holo": True}

    def test_failure_returns_no_edition(self):
        """Roll >= 0.25 → failure; no add_edition."""
        c = _consumable("c_wheel_of_fortune")
        c.set_ability("c_wheel_of_fortune")
        j = self._joker_with_sell("j_joker")
        rng = _ControlledRng([0.9])  # 0.9 >= 0.25 → fail
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        assert result is not None
        assert result.add_edition is None

    def test_no_rng_returns_empty(self):
        c = _consumable("c_wheel_of_fortune")
        c.set_ability("c_wheel_of_fortune")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.add_edition is None

    def test_foil_edition(self):
        """edition_poll ≤ 0.5 → foil."""
        c = _consumable("c_wheel_of_fortune")
        c.set_ability("c_wheel_of_fortune")
        j = self._joker_with_sell("j_joker")
        rng = _ControlledRng([0.1, 0.3])  # roll passes, poll=0.3 → foil
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        assert result.add_edition["edition"] == {"foil": True}

    def test_polychrome_edition(self):
        """edition_poll > 0.85 → polychrome."""
        c = _consumable("c_wheel_of_fortune")
        c.set_ability("c_wheel_of_fortune")
        j = self._joker_with_sell("j_joker")
        rng = _ControlledRng([0.1, 0.9])  # roll passes, poll=0.9 → polychrome
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        assert result.add_edition["edition"] == {"polychrome": True}

    def test_no_negative_edition(self):
        """Wheel of Fortune never gives negative edition (no_neg=True)."""
        c = _consumable("c_wheel_of_fortune")
        c.set_ability("c_wheel_of_fortune")
        j = self._joker_with_sell("j_joker")
        # edition_poll=0.99 — would be negative if no_neg=False, but should → polychrome
        rng = _ControlledRng([0.1, 0.99])
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        assert result.add_edition["edition"] == {"polychrome": True}
        assert "negative" not in result.add_edition["edition"]

    def test_can_use_needs_editionless_joker(self):
        c = _consumable("c_wheel_of_fortune")
        c.ability["consumeable"] = {}
        j_no_edition = _joker("j_joker")
        assert can_use_consumable(c, jokers=[j_no_edition]) is True

    def test_cannot_use_all_jokers_have_editions(self):
        c = _consumable("c_wheel_of_fortune")
        c.ability["consumeable"] = {}
        j = _joker("j_joker")
        j.edition = {"foil": True}
        assert can_use_consumable(c, jokers=[j]) is False


# ============================================================================
# Card.add_to_deck / Card.remove_from_deck
# ============================================================================


def _game_state(**kw) -> dict:
    """Minimal game_state with sensible defaults."""
    defaults = {
        "hand_size": 8,
        "discards": 3,
        "joker_slots": 5,
        "probabilities_normal": 1,
        "bankrupt_at": 0,
        "free_rerolls": 0,
        "hands_per_round": 4,
        "interest_amount": 1,
    }
    defaults.update(kw)
    return defaults


def _joker_card(key: str, **ability_kw) -> Card:
    """Build a joker Card via set_ability (loads from centers.json)."""
    c = Card()
    c.set_ability(key)
    return c


class TestAddToDeck:
    def test_stuntman_reduces_hand_size(self):
        j = _joker_card("j_stuntman")
        gs = _game_state()
        j.add_to_deck(gs)
        assert gs["hand_size"] == 6  # h_size = -2

    def test_stuntman_remove_restores(self):
        j = _joker_card("j_stuntman")
        gs = _game_state()
        j.add_to_deck(gs)
        j.remove_from_deck(gs)
        assert gs["hand_size"] == 8

    def test_juggler_increases_hand_size(self):
        j = _joker_card("j_juggler")
        gs = _game_state()
        j.add_to_deck(gs)
        assert gs["hand_size"] == 9  # h_size = +1

    def test_negative_edition_increases_joker_slots(self):
        j = Card()
        j.center_key = "j_joker"
        j.ability = {"name": "Joker", "set": "Joker", "h_size": 0, "d_size": 0}
        j.set_edition({"negative": True})
        gs = _game_state()
        j.add_to_deck(gs)
        assert gs["joker_slots"] == 6

    def test_negative_edition_remove_decreases_joker_slots(self):
        j = Card()
        j.center_key = "j_joker"
        j.ability = {"name": "Joker", "set": "Joker", "h_size": 0, "d_size": 0}
        j.set_edition({"negative": True})
        gs = _game_state()
        j.add_to_deck(gs)
        j.remove_from_deck(gs)
        assert gs["joker_slots"] == 5

    def test_oops_doubles_probabilities(self):
        j = _joker_card("j_oops")
        gs = _game_state(probabilities_normal=1)
        j.add_to_deck(gs)
        assert gs["probabilities_normal"] == 2

    def test_oops_remove_halves_probabilities(self):
        j = _joker_card("j_oops")
        gs = _game_state(probabilities_normal=1)
        j.add_to_deck(gs)
        j.remove_from_deck(gs)
        assert gs["probabilities_normal"] == 1

    def test_oops_remove_clamps_at_1(self):
        """Removing Oops when prob=1 stays at 1 (max(1, ...))."""
        j = _joker_card("j_oops")
        gs = _game_state(probabilities_normal=1)
        # Remove without prior add — should not go below 1
        j.remove_from_deck(gs)
        assert gs["probabilities_normal"] == 1

    def test_credit_card_sets_bankrupt_at(self):
        j = _joker_card("j_credit_card")
        gs = _game_state(bankrupt_at=0)
        j.add_to_deck(gs)
        assert gs["bankrupt_at"] == -20

    def test_credit_card_remove_resets_bankrupt_at(self):
        j = _joker_card("j_credit_card")
        gs = _game_state(bankrupt_at=0)
        j.add_to_deck(gs)
        j.remove_from_deck(gs)
        assert gs["bankrupt_at"] == 0

    def test_chaos_increments_free_rerolls(self):
        j = _joker_card("j_chaos")
        gs = _game_state(free_rerolls=0)
        j.add_to_deck(gs)
        assert gs["free_rerolls"] == 1

    def test_chaos_remove_decrements_free_rerolls(self):
        j = _joker_card("j_chaos")
        gs = _game_state(free_rerolls=0)
        j.add_to_deck(gs)
        j.remove_from_deck(gs)
        assert gs["free_rerolls"] == 0

    def test_chaos_remove_clamps_at_0(self):
        j = _joker_card("j_chaos")
        gs = _game_state(free_rerolls=0)
        j.remove_from_deck(gs)
        assert gs["free_rerolls"] == 0

    def test_troubadour_h_size_roundtrip(self):
        """Troubadour: extra.h_size +2 and extra.h_plays -1 on add; reversed on remove."""
        j = _joker_card("j_troubadour")
        gs = _game_state()
        j.add_to_deck(gs)
        assert gs["hand_size"] == 10  # +2 (extra.h_size=2)
        assert gs["hands_per_round"] == 3  # +(-1) = 3 (extra.h_plays=-1)
        j.remove_from_deck(gs)
        assert gs["hand_size"] == 8  # back to 8
        assert gs["hands_per_round"] == 4  # back to 4

    def test_merry_andy_d_size(self):
        """Merry Andy: d_size +3, h_size -1."""
        j = _joker_card("j_merry_andy")
        gs = _game_state()
        j.add_to_deck(gs)
        assert gs["discards"] == 6  # +3
        assert gs["hand_size"] == 7  # -1

    def test_drunkard_d_size(self):
        """Drunkard: d_size +1."""
        j = _joker_card("j_drunkard")
        gs = _game_state()
        j.add_to_deck(gs)
        assert gs["discards"] == 4

    def test_plain_joker_no_effect(self):
        """A standard joker with no h_size/d_size has no game_state effect."""
        j = _joker_card("j_joker")
        gs = _game_state()
        before = dict(gs)
        j.add_to_deck(gs)
        assert gs == before

    def test_no_edition_no_slot_change(self):
        """Joker without negative edition does not touch joker_slots."""
        j = _joker_card("j_joker")
        j.set_edition({"foil": True})
        gs = _game_state()
        j.add_to_deck(gs)
        assert gs["joker_slots"] == 5


# ============================================================================
# Planet cards
# ============================================================================


def _planet(key: str) -> Card:
    c = Card()
    c.set_ability(key)
    return c


class TestPlanetLevelUp:
    """Each planet returns level_up=[(hand_type, 1)]."""

    def test_mercury_levels_pair(self):
        c = _planet("c_mercury")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.level_up == [("Pair", 1)]

    def test_mercury_chips_mult_change(self):
        """Mercury: Pair level 1→2 → chips 10→25, mult 2→3."""
        c = _planet("c_mercury")
        result = use_consumable(c, ConsumableContext(card=c))
        hl = HandLevels()
        for hand_type, amount in result.level_up:
            hl.level_up(hand_type, amount)
        chips, mult = hl.get("Pair")
        assert chips == 25  # 10 + 15*1
        assert mult == 3  # 2 + 1*1

    def test_pluto_levels_high_card(self):
        c = _planet("c_pluto")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "High Card"

    def test_uranus_levels_two_pair(self):
        c = _planet("c_uranus")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Two Pair"

    def test_venus_levels_three_of_a_kind(self):
        c = _planet("c_venus")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Three of a Kind"

    def test_saturn_levels_straight(self):
        c = _planet("c_saturn")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Straight"

    def test_jupiter_levels_flush(self):
        c = _planet("c_jupiter")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Flush"

    def test_earth_levels_full_house(self):
        c = _planet("c_earth")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Full House"

    def test_mars_levels_four_of_a_kind(self):
        c = _planet("c_mars")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Four of a Kind"

    def test_neptune_levels_straight_flush(self):
        c = _planet("c_neptune")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Straight Flush"

    def test_planet_x_levels_five_of_a_kind(self):
        c = _planet("c_planet_x")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Five of a Kind"

    def test_ceres_levels_flush_house(self):
        c = _planet("c_ceres")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Flush House"

    def test_eris_levels_flush_five(self):
        c = _planet("c_eris")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.level_up[0][0] == "Flush Five"

    def test_each_planet_levels_exactly_one_hand(self):
        """Every single-planet card returns exactly 1 level_up entry."""
        for key in _ALL_HAND_TYPES:
            pass  # _ALL_HAND_TYPES is hand-type strings, not keys
        planet_keys = [
            "c_pluto",
            "c_mercury",
            "c_uranus",
            "c_venus",
            "c_saturn",
            "c_jupiter",
            "c_earth",
            "c_mars",
            "c_neptune",
            "c_planet_x",
            "c_ceres",
            "c_eris",
        ]
        for key in planet_keys:
            c = _planet(key)
            result = use_consumable(c, ConsumableContext(card=c))
            assert result is not None, f"{key} returned None"
            assert len(result.level_up) == 1, f"{key} leveled wrong count"
            assert result.level_up[0][1] == 1, f"{key} level delta != 1"


class TestBlackHole:
    def test_levels_all_12_hands(self):
        c = _planet("c_black_hole")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.level_up is not None
        assert len(result.level_up) == 12

    def test_all_hand_types_present(self):
        c = _planet("c_black_hole")
        result = use_consumable(c, ConsumableContext(card=c))
        leveled = {ht for ht, _ in result.level_up}
        assert leveled == set(_ALL_HAND_TYPES)

    def test_all_deltas_are_1(self):
        c = _planet("c_black_hole")
        result = use_consumable(c, ConsumableContext(card=c))
        assert all(delta == 1 for _, delta in result.level_up)

    def test_apply_to_hand_levels(self):
        """Applying Black Hole raises every hand by 1 level."""
        c = _planet("c_black_hole")
        result = use_consumable(c, ConsumableContext(card=c))
        hl = HandLevels()
        for hand_type, amount in result.level_up:
            hl.level_up(hand_type, amount)
        for ht in _ALL_HAND_TYPES:
            state = hl.get_state(ht)
            assert state.level == 2, f"{ht} not at level 2 after Black Hole"


class TestPlanetUsageTracking:
    def test_mercury_increments_planet_count(self):
        c = _planet("c_mercury")
        gs = {
            "consumable_usage_total": {
                "tarot": 0,
                "planet": 0,
                "spectral": 0,
                "tarot_planet": 0,
                "all": 0,
            }
        }
        use_consumable(c, ConsumableContext(card=c, game_state=gs))
        assert gs["consumable_usage_total"]["planet"] == 1

    def test_mercury_increments_all_count(self):
        c = _planet("c_mercury")
        gs = {
            "consumable_usage_total": {
                "tarot": 0,
                "planet": 0,
                "spectral": 0,
                "tarot_planet": 0,
                "all": 0,
            }
        }
        use_consumable(c, ConsumableContext(card=c, game_state=gs))
        assert gs["consumable_usage_total"]["all"] == 1

    def test_mercury_increments_tarot_planet(self):
        c = _planet("c_mercury")
        gs = {
            "consumable_usage_total": {
                "tarot": 0,
                "planet": 0,
                "spectral": 0,
                "tarot_planet": 0,
                "all": 0,
            }
        }
        use_consumable(c, ConsumableContext(card=c, game_state=gs))
        assert gs["consumable_usage_total"]["tarot_planet"] == 1

    def test_sets_last_tarot_planet(self):
        c = _planet("c_jupiter")
        gs: dict = {}
        use_consumable(c, ConsumableContext(card=c, game_state=gs))
        assert gs["last_tarot_planet"] == "c_jupiter"

    def test_usage_accumulates_across_uses(self):
        gs = {
            "consumable_usage_total": {
                "tarot": 0,
                "planet": 0,
                "spectral": 0,
                "tarot_planet": 0,
                "all": 0,
            }
        }
        for key in ["c_mercury", "c_jupiter", "c_mars"]:
            c = _planet(key)
            use_consumable(c, ConsumableContext(card=c, game_state=gs))
        assert gs["consumable_usage_total"]["planet"] == 3
        assert gs["consumable_usage_total"]["all"] == 3

    def test_initialises_totals_if_absent(self):
        """Usage dict is created on first use if game_state lacks it."""
        c = _planet("c_mercury")
        gs: dict = {}
        use_consumable(c, ConsumableContext(card=c, game_state=gs))
        assert "consumable_usage_total" in gs
        assert gs["consumable_usage_total"]["planet"] == 1

    def test_black_hole_also_tracks(self):
        c = _planet("c_black_hole")
        gs: dict = {}
        use_consumable(c, ConsumableContext(card=c, game_state=gs))
        assert gs["consumable_usage_total"]["planet"] == 1
        assert gs["last_tarot_planet"] == "c_black_hole"

    def test_no_game_state_does_not_raise(self):
        c = _planet("c_mercury")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None


class TestPlanetNotifyJokers:
    def test_notify_flag_set(self):
        """Planet result sets notify_jokers_consumeable=True."""
        c = _planet("c_mercury")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.notify_jokers_consumeable is True

    def test_black_hole_notify_flag_set(self):
        c = _planet("c_black_hole")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.notify_jokers_consumeable is True

    def test_constellation_gains_xmult_via_joker_context(self):
        """Simulating state-machine: call calculate_joker with using_consumeable
        for a Planet → Constellation's x_mult increases by 0.1."""
        mercury = _planet("c_mercury")

        constellation = Card()
        constellation.set_ability("j_constellation")
        start_xmult = constellation.ability.get("x_mult", 1)

        ctx = JokerContext(
            using_consumeable=True,
            consumeable=mercury,
        )
        calculate_joker(constellation, ctx)

        assert constellation.ability["x_mult"] == pytest.approx(start_xmult + 0.1)

    def test_constellation_ignores_tarot(self):
        """Constellation does NOT fire when consumeable is a Tarot."""
        justice = Card()
        justice.set_ability("c_justice")

        constellation = Card()
        constellation.set_ability("j_constellation")
        start_xmult = constellation.ability.get("x_mult", 1)

        ctx = JokerContext(
            using_consumeable=True,
            consumeable=justice,
        )
        calculate_joker(constellation, ctx)

        assert constellation.ability.get("x_mult", 1) == start_xmult


# ============================================================================
# Seal spectrals
# ============================================================================


class TestSealSpectrals:
    def test_deja_vu_adds_red_seal(self):
        c = _consumable("c_deja_vu")
        c.set_ability("c_deja_vu")
        target = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target]))
        assert result is not None
        assert result.add_seal == [(target, "Red")]

    def test_talisman_adds_gold_seal(self):
        c = _consumable("c_talisman")
        c.set_ability("c_talisman")
        target = _card("Spades", "King")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target]))
        assert result.add_seal == [(target, "Gold")]

    def test_trance_adds_blue_seal(self):
        c = _consumable("c_trance")
        c.set_ability("c_trance")
        target = _card("Clubs", "5")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target]))
        assert result.add_seal == [(target, "Blue")]

    def test_medium_adds_purple_seal(self):
        c = _consumable("c_medium")
        c.set_ability("c_medium")
        target = _card("Diamonds", "3")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target]))
        assert result.add_seal == [(target, "Purple")]

    def test_overwrite_existing_seal(self):
        """Adding a Blue Seal to a card already bearing Gold Seal.

        The handler returns the new seal; the caller applies card.seal = seal.
        """
        c = _consumable("c_trance")
        c.set_ability("c_trance")
        target = _card("Hearts", "7")
        target.set_seal("Gold")
        assert target.seal == "Gold"

        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target]))
        assert result.add_seal[0] == (target, "Blue")

        # Simulate state-machine applying the descriptor
        target.set_seal(result.add_seal[0][1])
        assert target.seal == "Blue"

    def test_can_use_needs_exactly_1_highlighted(self):
        for key in ["c_talisman", "c_deja_vu", "c_trance", "c_medium"]:
            c = _consumable(key)
            c.set_ability(key)
            assert can_use_consumable(c, highlighted=[]) is False, f"{key} with 0"
            assert can_use_consumable(c, highlighted=[_card("Hearts", "5")]) is True, (
                f"{key} with 1"
            )
            assert (
                can_use_consumable(c, highlighted=[_card("Hearts", "5"), _card("Spades", "3")])
                is False
            ), f"{key} with 2"


# ============================================================================
# Cryptid
# ============================================================================


class TestCryptid:
    def _source(self) -> Card:
        card = _card("Spades", "Ace", enhancement="m_glass")
        card.set_edition({"foil": True})
        card.set_seal("Red")
        return card

    def test_returns_two_add_to_deck_descriptors(self):
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        source = self._source()
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[source]))
        assert result is not None
        assert result.add_to_deck is not None
        assert len(result.add_to_deck) == 2

    def test_descriptors_reference_source_card(self):
        """Each descriptor's copy_of points to the highlighted source."""
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        source = self._source()
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[source]))
        for desc in result.add_to_deck:
            assert desc["copy_of"] is source

    def test_copy_of_has_matching_suit_and_rank(self):
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        source = self._source()
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[source]))
        ref = result.add_to_deck[0]["copy_of"]
        assert ref.base.suit.value == "Spades"
        assert ref.base.rank.value == "Ace"

    def test_copy_of_has_matching_enhancement(self):
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        source = self._source()
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[source]))
        ref = result.add_to_deck[0]["copy_of"]
        assert ref.ability["effect"] == "Glass Card"

    def test_copy_of_has_matching_edition(self):
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        source = self._source()
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[source]))
        ref = result.add_to_deck[0]["copy_of"]
        assert ref.edition is not None
        assert ref.edition.get("foil") is True

    def test_copy_of_has_matching_seal(self):
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        source = self._source()
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[source]))
        ref = result.add_to_deck[0]["copy_of"]
        assert ref.seal == "Red"

    def test_no_highlighted_returns_empty(self):
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[]))
        assert result is not None
        assert result.add_to_deck is None

    def test_can_use_needs_exactly_1_highlighted(self):
        c = _consumable("c_cryptid")
        c.set_ability("c_cryptid")
        assert can_use_consumable(c, highlighted=[]) is False
        assert can_use_consumable(c, highlighted=[_card("Hearts", "5")]) is True


# ============================================================================
# Destroy/create spectrals: Familiar, Grim, Incantation, Immolate
# ============================================================================

_FACE_RANKS = {"Jack", "Queen", "King"}
_NUMBER_RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10"}
_ALL_SUITS = {"Spades", "Hearts", "Diamonds", "Clubs"}


def _hand(n: int) -> list[Card]:
    """Build a hand of n distinct cards."""
    suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
    cards = []
    for i in range(n):
        s = suits[i % 4]
        r = ranks[i % 13]
        cards.append(_card(s, r))
    return cards


def _spectral(key: str) -> Card:
    c = Card()
    c.set_ability(key)
    return c


class TestFamiliar:
    def test_destroys_one_card(self):
        rng = _ControlledRng()
        c = _spectral("c_familiar")
        hand = _hand(5)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert result is not None
        assert result.destroy is not None
        assert len(result.destroy) == 1
        assert result.destroy[0] in hand

    def test_creates_three_face_cards(self):
        rng = _ControlledRng()
        c = _spectral("c_familiar")
        hand = _hand(5)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert result.create is not None
        assert len(result.create) == 3

    def test_created_cards_have_face_ranks(self):
        rng = _ControlledRng()
        c = _spectral("c_familiar")
        hand = _hand(5)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        for spec in result.create:
            assert spec["rank"] in _FACE_RANKS, f"Expected face rank, got {spec['rank']}"

    def test_created_cards_have_valid_suits(self):
        rng = _ControlledRng()
        c = _spectral("c_familiar")
        hand = _hand(5)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        for spec in result.create:
            assert spec["suit"] in _ALL_SUITS

    def test_created_cards_have_non_stone_enhancement(self):
        rng = _ControlledRng()
        c = _spectral("c_familiar")
        hand = _hand(5)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        for spec in result.create:
            assert spec["enhancement"] != "m_stone"
            assert spec["enhancement"].startswith("m_")

    def test_can_use_needs_more_than_1_card(self):
        c = _spectral("c_familiar")
        one_card = [_card("Hearts", "5")]
        two_cards = [_card("Hearts", "5"), _card("Spades", "3")]
        assert can_use_consumable(c, hand_cards=one_card) is False
        assert can_use_consumable(c, hand_cards=two_cards) is True

    def test_no_rng_returns_empty(self):
        c = _spectral("c_familiar")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(3)))
        assert result is not None
        assert result.destroy is None


class TestGrim:
    def test_destroys_one_creates_two_aces(self):
        rng = _ControlledRng()
        c = _spectral("c_grim")
        hand = _hand(4)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert len(result.destroy) == 1
        assert len(result.create) == 2

    def test_created_cards_are_aces(self):
        rng = _ControlledRng()
        c = _spectral("c_grim")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(3), rng=rng))
        for spec in result.create:
            assert spec["rank"] == "Ace"

    def test_created_cards_have_non_stone_enhancement(self):
        rng = _ControlledRng()
        c = _spectral("c_grim")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(3), rng=rng))
        for spec in result.create:
            assert spec["enhancement"] != "m_stone"


class TestIncantation:
    def test_destroys_one_creates_four(self):
        rng = _ControlledRng()
        c = _spectral("c_incantation")
        hand = _hand(6)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert len(result.destroy) == 1
        assert len(result.create) == 4

    def test_created_cards_are_number_cards(self):
        rng = _ControlledRng()
        c = _spectral("c_incantation")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(4), rng=rng))
        for spec in result.create:
            assert spec["rank"] in _NUMBER_RANKS, f"Expected number, got {spec['rank']}"

    def test_created_cards_have_non_stone_enhancement(self):
        rng = _ControlledRng()
        c = _spectral("c_incantation")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(4), rng=rng))
        for spec in result.create:
            assert spec["enhancement"] != "m_stone"


class TestImmolate:
    def test_destroys_five_cards(self):
        """Immolate with 8 cards in hand destroys 5."""
        rng = _ControlledRng()
        c = _spectral("c_immolate")
        hand = _hand(8)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert result is not None
        assert result.destroy is not None
        assert len(result.destroy) == 5

    def test_gains_twenty_dollars(self):
        rng = _ControlledRng()
        c = _spectral("c_immolate")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(8), rng=rng))
        assert result.dollars == 20

    def test_destroyed_cards_are_from_hand(self):
        rng = _ControlledRng()
        c = _spectral("c_immolate")
        hand = _hand(8)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        for destroyed_card in result.destroy:
            assert destroyed_card in hand

    def test_no_creation(self):
        rng = _ControlledRng()
        c = _spectral("c_immolate")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(8), rng=rng))
        assert result.create is None

    def test_can_use_needs_more_than_1_card(self):
        c = _spectral("c_immolate")
        assert can_use_consumable(c, hand_cards=[_card("Hearts", "5")]) is False
        assert (
            can_use_consumable(c, hand_cards=[_card("Hearts", "5"), _card("Spades", "3")]) is True
        )


# ============================================================================
# Sigil and Ouija
# ============================================================================


class TestSigil:
    def test_changes_all_cards_to_same_suit(self):
        """All hand cards get change_suit entry for the same suit."""
        rng = _ControlledRng()
        c = _spectral("c_sigil")
        hand = _hand(4)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert result is not None
        assert result.change_suit is not None
        assert len(result.change_suit) == len(hand)
        suits = {suit for _, suit in result.change_suit}
        assert len(suits) == 1, "All cards should be changed to the same suit"

    def test_suit_is_valid(self):
        rng = _ControlledRng()
        c = _spectral("c_sigil")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(3), rng=rng))
        for _, suit in result.change_suit:
            assert suit in _ALL_SUITS

    def test_all_hand_cards_included(self):
        rng = _ControlledRng()
        c = _spectral("c_sigil")
        hand = _hand(5)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        changed_cards = [card for card, _ in result.change_suit]
        for card in hand:
            assert card in changed_cards

    def test_no_destruction_or_creation(self):
        rng = _ControlledRng()
        c = _spectral("c_sigil")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(4), rng=rng))
        assert result.destroy is None
        assert result.create is None

    def test_deterministic_with_fixed_seed(self):
        """Same PseudoRandom seed produces the same suit twice."""
        from jackdaw.engine.rng import PseudoRandom

        c = _spectral("c_sigil")
        hand = _hand(3)
        rng1 = PseudoRandom("test_sigil_det")
        r1 = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng1))
        rng2 = PseudoRandom("test_sigil_det")
        r2 = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng2))
        assert r1.change_suit[0][1] == r2.change_suit[0][1]

    def test_can_use_needs_more_than_1_card(self):
        c = _spectral("c_sigil")
        assert can_use_consumable(c, hand_cards=[_card("Hearts", "5")]) is False
        assert (
            can_use_consumable(c, hand_cards=[_card("Hearts", "5"), _card("Spades", "3")]) is True
        )


class TestOuija:
    def test_changes_all_cards_to_same_rank(self):
        """All hand cards get change_rank entry for the same rank."""
        rng = _ControlledRng()
        c = _spectral("c_ouija")
        hand = _hand(4)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert result is not None
        assert result.change_rank is not None
        assert len(result.change_rank) == len(hand)
        ranks = {rank for _, rank in result.change_rank}
        assert len(ranks) == 1, "All cards should be changed to the same rank"

    def test_rank_is_valid(self):
        rng = _ControlledRng()
        c = _spectral("c_ouija")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(3), rng=rng))
        valid = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"}
        for _, rank in result.change_rank:
            assert rank in valid

    def test_hand_size_mod_minus_one(self):
        rng = _ControlledRng()
        c = _spectral("c_ouija")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(4), rng=rng))
        assert result.hand_size_mod == -1

    def test_all_hand_cards_included(self):
        rng = _ControlledRng()
        c = _spectral("c_ouija")
        hand = _hand(5)
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        changed_cards = [card for card, _ in result.change_rank]
        for card in hand:
            assert card in changed_cards

    def test_no_destruction_or_creation(self):
        rng = _ControlledRng()
        c = _spectral("c_ouija")
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=_hand(4), rng=rng))
        assert result.destroy is None
        assert result.create is None

    def test_can_use_needs_more_than_1_card(self):
        c = _spectral("c_ouija")
        assert can_use_consumable(c, hand_cards=[_card("Hearts", "5")]) is False
        assert (
            can_use_consumable(c, hand_cards=[_card("Hearts", "5"), _card("Spades", "3")]) is True
        )


# ============================================================================
# Aura
# ============================================================================


class TestAura:
    def test_adds_edition_to_highlighted_card(self):
        # 0.5 → foil (> 0.0, not > 0.5)
        rng = _ControlledRng([0.5])
        c = _spectral("c_aura")
        target = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target], rng=rng))
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["target"] is target

    def test_edition_is_foil_when_poll_is_half(self):
        """poll=0.5 → foil (> 0.0, not > 0.5 for holo)."""
        rng = _ControlledRng([0.5])
        c = _spectral("c_aura")
        target = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target], rng=rng))
        assert result.add_edition["edition"] == {"foil": True}

    def test_edition_is_holo_when_poll_is_high(self):
        """poll=0.75 → holo (> 0.5, not > 0.85)."""
        rng = _ControlledRng([0.75])
        c = _spectral("c_aura")
        target = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target], rng=rng))
        assert result.add_edition["edition"] == {"holo": True}

    def test_edition_is_poly_when_poll_near_one(self):
        """poll=0.95 → polychrome (> 0.85); no negative even at this level."""
        rng = _ControlledRng([0.95])
        c = _spectral("c_aura")
        target = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target], rng=rng))
        assert result.add_edition["edition"] == {"polychrome": True}

    def test_no_negative_edition_even_at_max(self):
        """poll=0.99 → polychrome, never negative (no_neg=True)."""
        rng = _ControlledRng([0.99])
        c = _spectral("c_aura")
        target = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target], rng=rng))
        edition = result.add_edition["edition"]
        assert not edition.get("negative")

    def test_no_result_without_rng(self):
        c = _spectral("c_aura")
        target = _card("Hearts", "5")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target]))
        assert result.add_edition is None

    def test_can_use_needs_highlighted_no_edition(self):
        c = _spectral("c_aura")
        plain = _card("Hearts", "5")
        assert can_use_consumable(c, highlighted=[plain]) is True

    def test_can_use_rejects_card_with_edition(self):
        c = _spectral("c_aura")
        with_edition = _card("Hearts", "5")
        with_edition.edition = {"foil": True}
        assert can_use_consumable(c, highlighted=[with_edition]) is False

    def test_can_use_rejects_no_highlighted(self):
        c = _spectral("c_aura")
        assert can_use_consumable(c, highlighted=[]) is False


# ============================================================================
# Ectoplasm
# ============================================================================


class TestEctoplasm:
    def test_adds_negative_to_editionless_joker(self):
        rng = _ControlledRng()
        c = _spectral("c_ectoplasm")
        joker = _joker("j_test")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[joker], rng=rng))
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["edition"] == {"negative": True}
        assert result.add_edition["target"] is joker

    def test_hand_size_mod_minus_one(self):
        rng = _ControlledRng()
        c = _spectral("c_ectoplasm")
        joker = _joker("j_test")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[joker], rng=rng))
        assert result.hand_size_mod == -1

    def test_skips_jokers_with_existing_edition(self):
        """Only editionless jokers are eligible targets."""
        rng = _ControlledRng()
        c = _spectral("c_ectoplasm")
        joker_foil = _joker("j_foil")
        joker_foil.edition = {"foil": True}
        plain = _joker("j_plain")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[joker_foil, plain], rng=rng))
        assert result.add_edition["target"] is plain

    def test_no_result_without_eligible_joker(self):
        rng = _ControlledRng()
        c = _spectral("c_ectoplasm")
        foil_joker = _joker("j_foil")
        foil_joker.edition = {"foil": True}
        result = use_consumable(c, ConsumableContext(card=c, jokers=[foil_joker], rng=rng))
        assert result.add_edition is None

    def test_can_use_needs_eligible_joker(self):
        c = _spectral("c_ectoplasm")
        plain = _joker("j_plain")
        assert can_use_consumable(c, jokers=[plain]) is True
        with_edition = _joker("j_foil")
        with_edition.edition = {"foil": True}
        assert can_use_consumable(c, jokers=[with_edition]) is False


# ============================================================================
# Hex
# ============================================================================


class TestHex:
    def test_adds_polychrome_to_chosen_joker(self):
        rng = _ControlledRng()
        c = _spectral("c_hex")
        joker = _joker("j_test")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[joker], rng=rng))
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["edition"] == {"polychrome": True}
        assert result.add_edition["target"] is joker

    def test_destroys_other_non_eternal_jokers(self):
        """_ControlledRng.element picks first → j1 chosen; j2, j3 destroyed."""
        rng = _ControlledRng()
        c = _spectral("c_hex")
        j1 = _joker("j_a")
        j2 = _joker("j_b")
        j3 = _joker("j_c")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[j1, j2, j3], rng=rng))
        assert result.destroy_jokers is not None
        assert j2 in result.destroy_jokers
        assert j3 in result.destroy_jokers
        assert j1 not in result.destroy_jokers

    def test_eternal_jokers_not_destroyed(self):
        rng = _ControlledRng()
        c = _spectral("c_hex")
        j1 = _joker("j_chosen")
        eternal = _joker("j_eternal")
        eternal.eternal = True
        result = use_consumable(c, ConsumableContext(card=c, jokers=[j1, eternal], rng=rng))
        destroyed = result.destroy_jokers or []
        assert eternal not in destroyed

    def test_no_destroy_when_only_one_joker(self):
        rng = _ControlledRng()
        c = _spectral("c_hex")
        joker = _joker("j_only")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[joker], rng=rng))
        assert result.destroy_jokers is None

    def test_can_use_needs_eligible_joker(self):
        c = _spectral("c_hex")
        plain = _joker("j_plain")
        assert can_use_consumable(c, jokers=[plain]) is True
        with_edition = _joker("j_foil")
        with_edition.edition = {"foil": True}
        assert can_use_consumable(c, jokers=[with_edition]) is False


# ============================================================================
# Wraith
# ============================================================================


class TestWraith:
    def test_creates_rare_joker(self):
        c = _spectral("c_wraith")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.create is not None
        assert len(result.create) == 1
        spec = result.create[0]
        assert spec["type"] == "Joker"
        assert spec["rarity"] == 3  # Rare

    def test_sets_money_to_zero(self):
        c = _spectral("c_wraith")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.money_set == 0

    def test_no_other_side_effects(self):
        c = _spectral("c_wraith")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.destroy is None
        assert result.destroy_jokers is None
        assert result.dollars == 0

    def test_can_use_needs_joker_slot(self):
        c = _spectral("c_wraith")
        assert can_use_consumable(c, jokers=[], joker_limit=5) is True
        full = [_joker(f"j_{i}") for i in range(5)]
        assert can_use_consumable(c, jokers=full, joker_limit=5) is False


# ============================================================================
# Ankh
# ============================================================================


class TestAnkh:
    def test_duplicates_chosen_joker(self):
        """_ControlledRng picks first element → j1 is chosen and copied."""
        rng = _ControlledRng()
        c = _spectral("c_ankh")
        j1 = _joker("j_a")
        j2 = _joker("j_b")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[j1, j2], rng=rng))
        assert result is not None
        assert result.create is not None
        assert result.create[0]["copy_of"] is j1

    def test_destroys_others(self):
        rng = _ControlledRng()
        c = _spectral("c_ankh")
        j1 = _joker("j_chosen")
        j2 = _joker("j_other1")
        j3 = _joker("j_other2")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[j1, j2, j3], rng=rng))
        assert result.destroy_jokers is not None
        assert j2 in result.destroy_jokers
        assert j3 in result.destroy_jokers
        assert j1 not in result.destroy_jokers

    def test_eternal_others_not_destroyed(self):
        rng = _ControlledRng()
        c = _spectral("c_ankh")
        j1 = _joker("j_chosen")
        eternal = _joker("j_eternal")
        eternal.eternal = True
        result = use_consumable(c, ConsumableContext(card=c, jokers=[j1, eternal], rng=rng))
        destroyed = result.destroy_jokers or []
        assert eternal not in destroyed

    def test_no_destroy_when_only_one_joker(self):
        rng = _ControlledRng()
        c = _spectral("c_ankh")
        j1 = _joker("j_only")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[j1], rng=rng))
        assert result.destroy_jokers is None
        assert result.create is not None
        assert result.create[0]["copy_of"] is j1

    def test_can_use_needs_joker(self):
        c = _spectral("c_ankh")
        assert can_use_consumable(c, jokers=[], joker_limit=5) is False
        assert can_use_consumable(c, jokers=[_joker("j_a")], joker_limit=5) is True

    def test_can_use_needs_room_to_duplicate(self):
        c = _spectral("c_ankh")
        # joker_limit=1 → no room to duplicate
        assert can_use_consumable(c, jokers=[_joker("j_a")], joker_limit=1) is False
        assert can_use_consumable(c, jokers=[_joker("j_a")], joker_limit=2) is True


# ============================================================================
# Soul
# ============================================================================


class TestSoul:
    def test_creates_legendary_joker(self):
        c = _spectral("c_soul")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.create is not None
        assert len(result.create) == 1
        spec = result.create[0]
        assert spec["type"] == "Joker"
        assert spec["rarity"] == 4  # Legendary

    def test_no_money_set(self):
        c = _spectral("c_soul")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.money_set is None

    def test_no_other_side_effects(self):
        c = _spectral("c_soul")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result.destroy is None
        assert result.destroy_jokers is None
        assert result.dollars == 0

    def test_can_use_needs_joker_slot(self):
        c = _spectral("c_soul")
        assert can_use_consumable(c, jokers=[], joker_limit=5) is True
        full = [_joker(f"j_{i}") for i in range(5)]
        assert can_use_consumable(c, jokers=full, joker_limit=5) is False
