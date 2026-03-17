"""Tests for card factory functions.

Verifies card creation from prototypes, control dicts, and modifier
application for all card types.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import reset_sort_id_counter
from jackdaw.engine.card_factory import (
    RANK_LETTER,
    SUIT_LETTER,
    card_from_control,
    create_card,
    create_consumable,
    create_joker,
    create_playing_card,
    create_voucher,
)
from jackdaw.engine.data.enums import Rank, Suit


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ============================================================================
# create_playing_card
# ============================================================================


class TestCreatePlayingCard:
    def test_ace_of_spades(self):
        c = create_playing_card(Suit.SPADES, Rank.ACE)
        assert c.base is not None
        assert c.base.suit is Suit.SPADES
        assert c.base.rank is Rank.ACE
        assert c.base.nominal == 11
        assert c.base.id == 14
        assert c.card_key == "S_A"
        assert c.center_key == "c_base"
        assert c.ability["name"] == "Default Base"

    def test_two_of_hearts(self):
        c = create_playing_card(Suit.HEARTS, Rank.TWO)
        assert c.base.nominal == 2
        assert c.card_key == "H_2"

    def test_glass_enhancement(self):
        c = create_playing_card(Suit.DIAMONDS, Rank.KING, enhancement="m_glass")
        assert c.center_key == "m_glass"
        assert c.ability["effect"] == "Glass Card"
        assert c.ability["x_mult"] == 2  # Glass Card has Xmult=2

    def test_gold_enhancement(self):
        c = create_playing_card(Suit.CLUBS, Rank.FIVE, enhancement="m_gold")
        assert c.center_key == "m_gold"
        assert c.ability["effect"] == "Gold Card"

    def test_with_edition(self):
        c = create_playing_card(Suit.HEARTS, Rank.ACE, edition={"foil": True})
        assert c.edition["foil"] is True

    def test_with_seal(self):
        c = create_playing_card(Suit.SPADES, Rank.QUEEN, seal="Red")
        assert c.seal == "Red"

    def test_with_all_modifiers(self):
        c = create_playing_card(
            Suit.DIAMONDS,
            Rank.JACK,
            enhancement="m_steel",
            edition={"polychrome": True},
            seal="Gold",
        )
        assert c.center_key == "m_steel"
        assert c.edition["polychrome"] is True
        assert c.seal == "Gold"
        assert c.base.rank is Rank.JACK

    def test_playing_card_index(self):
        c = create_playing_card(Suit.HEARTS, Rank.THREE, playing_card_index=7)
        assert c.playing_card == 7

    def test_sort_id_auto_assigned(self):
        c1 = create_playing_card(Suit.HEARTS, Rank.TWO)
        c2 = create_playing_card(Suit.HEARTS, Rank.THREE)
        assert c2.sort_id == c1.sort_id + 1

    def test_all_52_cards(self):
        """Create all 52 standard cards and verify uniqueness."""
        cards = []
        for suit in Suit:
            for rank in Rank:
                cards.append(create_playing_card(suit, rank))
        assert len(cards) == 52
        keys = {c.card_key for c in cards}
        assert len(keys) == 52


# ============================================================================
# create_joker
# ============================================================================


class TestCreateJoker:
    def test_basic_joker(self):
        c = create_joker("j_joker")
        assert c.ability["name"] == "Joker"
        assert c.ability["mult"] == 4
        assert c.ability["set"] == "Joker"
        assert c.center_key == "j_joker"
        assert c.base is None
        assert c.base_cost == 2

    def test_greedy_joker(self):
        c = create_joker("j_greedy_joker")
        assert c.ability["extra"]["s_mult"] == 3
        assert c.ability["extra"]["suit"] == "Diamonds"

    def test_with_foil_edition(self):
        c = create_joker("j_joker", edition={"foil": True})
        assert c.edition["foil"] is True

    def test_eternal(self):
        c = create_joker("j_joker", eternal=True)
        assert c.eternal is True

    def test_perishable(self):
        c = create_joker("j_joker", perishable=True)
        assert c.perishable is True
        assert c.perish_tally == 5

    def test_rental(self):
        c = create_joker("j_joker", rental=True)
        assert c.rental is True

    def test_all_stickers(self):
        c = create_joker(
            "j_joker",
            edition={"negative": True},
            eternal=True,
            rental=True,
        )
        assert c.eternal is True
        assert c.rental is True
        assert c.edition["negative"] is True

    def test_ice_cream_extra(self):
        c = create_joker("j_ice_cream")
        assert c.ability["extra"]["chips"] == 100

    def test_loyalty_card_post_init(self):
        c = create_joker("j_loyalty_card")
        assert c.ability["loyalty_remaining"] == 5

    def test_hands_played_at_create(self):
        c = create_joker("j_joker", hands_played=42)
        assert c.ability["hands_played_at_create"] == 42


# ============================================================================
# create_consumable
# ============================================================================


class TestCreateConsumable:
    def test_tarot(self):
        c = create_consumable("c_magician")
        assert c.ability["name"] == "The Magician"
        assert c.ability["set"] == "Tarot"
        assert c.center_key == "c_magician"

    def test_planet(self):
        c = create_consumable("c_pluto")
        assert c.ability["name"] == "Pluto"
        assert c.ability["set"] == "Planet"

    def test_spectral(self):
        c = create_consumable("c_aura")
        assert c.ability["name"] == "Aura"
        assert c.ability["set"] == "Spectral"


# ============================================================================
# create_voucher
# ============================================================================


class TestCreateVoucher:
    def test_overstock(self):
        c = create_voucher("v_overstock_norm")
        assert c.ability["name"] == "Overstock"
        assert c.ability["set"] == "Voucher"
        assert c.center_key == "v_overstock_norm"
        assert c.base_cost == 10


# ============================================================================
# card_from_control
# ============================================================================


class TestCardFromControl:
    def test_basic_control(self):
        """Simple card: suit + rank only."""
        c = card_from_control({"s": "S", "r": "A"})
        assert c.base.suit is Suit.SPADES
        assert c.base.rank is Rank.ACE
        assert c.center_key == "c_base"
        assert c.edition is None
        assert c.seal is None

    def test_full_control(self):
        """All fields: enhancement, edition, seal."""
        c = card_from_control(
            {
                "s": "H",
                "r": "K",
                "e": "m_gold",
                "d": "holo",
                "g": "Red",
            }
        )
        assert c.base.suit is Suit.HEARTS
        assert c.base.rank is Rank.KING
        assert c.center_key == "m_gold"
        assert c.ability["effect"] == "Gold Card"
        assert c.edition["holo"] is True
        assert c.seal == "Red"

    def test_glass_enhancement(self):
        c = card_from_control({"s": "D", "r": "5", "e": "m_glass"})
        assert c.center_key == "m_glass"
        assert c.ability["x_mult"] == 2

    def test_ten_rank(self):
        c = card_from_control({"s": "C", "r": "T"})
        assert c.base.rank is Rank.TEN
        assert c.card_key == "C_T"

    def test_playing_card_index(self):
        c = card_from_control({"s": "H", "r": "2"}, playing_card_index=1)
        assert c.playing_card == 1

    def test_no_enhancement_defaults_to_base(self):
        c = card_from_control({"s": "S", "r": "J"})
        assert c.center_key == "c_base"

    def test_explicit_none_enhancement(self):
        c = card_from_control({"s": "S", "r": "J", "e": None})
        assert c.center_key == "c_base"

    def test_all_suit_letters(self):
        for letter, expected_suit in SUIT_LETTER.items():
            c = card_from_control({"s": letter, "r": "A"})
            assert c.base.suit is expected_suit

    def test_all_rank_letters(self):
        for letter, expected_rank in RANK_LETTER.items():
            c = card_from_control({"s": "H", "r": letter})
            assert c.base.rank is expected_rank


# ============================================================================
# Deep copy isolation across factory
# ============================================================================


class TestFactoryIsolation:
    def test_two_jokers_from_same_key(self):
        c1 = create_joker("j_ice_cream")
        c2 = create_joker("j_ice_cream")
        c1.ability["extra"]["chips"] = 0
        assert c2.ability["extra"]["chips"] == 100

    def test_two_playing_cards_same_enhancement(self):
        c1 = create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_bonus")
        c2 = create_playing_card(Suit.SPADES, Rank.ACE, enhancement="m_bonus")
        c1.ability["bonus"] = 999
        assert c2.ability["bonus"] == 30  # original value


# ============================================================================
# create_card — common_events.lua:2082
# ============================================================================


def _gs(**kwargs) -> dict:
    """Build a game_state dict from keyword args."""
    return dict(kwargs)


class TestCreateCardKeyDetermination:
    def test_forced_key_bypasses_pool(self):
        from jackdaw.engine.rng import PseudoRandom

        rng = PseudoRandom("FORCED_CF_TEST")
        card = create_card("Joker", rng, 1, forced_key="j_joker")
        assert card.center_key == "j_joker"

    def test_forced_key_bypasses_soul_check(self):
        """forced_key must not consume the soul_Joker stream."""
        from jackdaw.engine.rng import PseudoRandom

        rng_forced = PseudoRandom("FORCED_SOUL_TEST")
        rng_manual = PseudoRandom("FORCED_SOUL_TEST")
        create_card("Joker", rng_forced, 1, forced_key="j_joker")
        v_forced = rng_forced.random("soul_Joker1")
        v_manual = rng_manual.random("soul_Joker1")
        assert v_forced == v_manual  # both are first draws on that stream

    def test_known_seed_returns_specific_joker(self):
        """PseudoRandom('CF_JOKER_TEST') + forced_rarity=1 at ante=1 picks j_crafty."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("CF_JOKER_TEST"), 1, forced_rarity=1)
        assert card.center_key == "j_crafty"

    def test_known_seed_tarot(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Tarot", PseudoRandom("TAROT_CF_TEST"), 1)
        assert card.center_key == "c_empress"

    def test_deterministic_same_seed(self):
        from jackdaw.engine.rng import PseudoRandom

        k1 = create_card("Joker", PseudoRandom("DET_CF"), ante=2, forced_rarity=2).center_key
        k2 = create_card("Joker", PseudoRandom("DET_CF"), ante=2, forced_rarity=2).center_key
        assert k1 == k2

    def test_soulable_false_skips_soul_stream(self):
        """soulable=False must not advance the soul_Joker stream."""
        from jackdaw.engine.rng import PseudoRandom

        rng_a = PseudoRandom("SOULABLE_FALSE_CF")
        rng_b = PseudoRandom("SOULABLE_FALSE_CF")
        create_card("Joker", rng_a, 1, forced_rarity=1, soulable=False)
        rng_b.random("soul_Joker1")  # consume what soulable=True would draw
        # rng_a never consumed soul_Joker1 → it's still at its initial state
        v_a = rng_a.random("soul_Joker1")  # first draw on rng_a
        v_b = rng_b.random("soul_Joker1")  # second draw on rng_b
        assert v_a != v_b  # different stream positions

    def test_ability_set_matches_card_type(self):
        from jackdaw.engine.rng import PseudoRandom

        for ct, expected in (
            ("Joker", "Joker"),
            ("Tarot", "Tarot"),
            ("Planet", "Planet"),
            ("Spectral", "Spectral"),
        ):
            card = create_card(ct, PseudoRandom("SET_TEST"), 1, forced_rarity=1)
            assert card.ability["set"] == expected, (
                f"card_type={ct!r}: expected set={expected!r}, got {card.ability['set']!r}"
            )

    def test_returns_card_instance(self):
        from jackdaw.engine.card import Card
        from jackdaw.engine.rng import PseudoRandom

        assert isinstance(create_card("Tarot", PseudoRandom("INST_TEST"), 1), Card)


class TestCreateCardEternalPerishable:
    """ep_roll thresholds: >0.7 eternal, >0.4 perishable (shared roll, mutually exclusive).

    Seed 'CF_JOKER_TEST': ep_roll=0.9914.
    Seed 'P0': ep_roll=0.6909 (in 0.4–0.7).
    """

    def test_eternal_fires_at_stake4(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Joker",
            PseudoRandom("CF_JOKER_TEST"),
            1,
            forced_rarity=1,
            game_state=_gs(enable_eternals_in_shop=True),
        )
        assert card.eternal is True

    def test_no_eternal_when_not_enabled(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("CF_JOKER_TEST"), 1, forced_rarity=1)
        assert card.eternal is False

    def test_high_roll_falls_to_perishable_when_eternal_disabled(self):
        """ep=0.9914 >0.7; eternal=False → elif ep>0.4 and perishable enabled → True."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Joker",
            PseudoRandom("CF_JOKER_TEST"),
            1,
            forced_rarity=1,
            game_state=_gs(enable_perishables_in_shop=True),
        )
        assert card.eternal is False
        assert card.perishable is True

    def test_mid_roll_triggers_perishable(self):
        """ep=0.6909 ≤0.7 → no eternal; ep>0.4 and enable_perishables → True."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Joker",
            PseudoRandom("P0"),
            1,
            forced_rarity=1,
            game_state=_gs(enable_perishables_in_shop=True),
        )
        assert card.perishable is True
        assert card.eternal is False
        assert card.perish_tally == 5

    def test_mid_roll_no_eternal_even_if_enabled(self):
        """ep=0.6909 ≤0.7 → eternal never fires even with enable_eternals."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Joker",
            PseudoRandom("P0"),
            1,
            forced_rarity=1,
            game_state=_gs(enable_eternals_in_shop=True),
        )
        assert card.eternal is False

    def test_eternal_and_perishable_mutually_exclusive(self):
        from jackdaw.engine.rng import PseudoRandom

        for seed in ("CF_JOKER_TEST", "P0", "S15", "FOIL_TEST"):
            card = create_card(
                "Joker",
                PseudoRandom(seed),
                1,
                forced_rarity=1,
                game_state=_gs(
                    enable_eternals_in_shop=True,
                    enable_perishables_in_shop=True,
                ),
            )
            assert not (card.eternal and card.perishable), (
                f"Seed {seed!r}: eternal={card.eternal} and perishable={card.perishable} "
                "both True — should be mutually exclusive"
            )


class TestCreateCardRental:
    """Seed 'S15': ep=0.8357, r=0.8849 (>0.7)."""

    def test_rental_fires_at_stake8(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Joker",
            PseudoRandom("S15"),
            1,
            forced_rarity=1,
            game_state=_gs(enable_rentals_in_shop=True),
        )
        assert card.rental is True

    def test_no_rental_when_not_enabled(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("S15"), 1, forced_rarity=1)
        assert card.rental is False

    def test_eternal_and_rental_simultaneously_at_stake8(self):
        """Seed 'S15': ep=0.8357 and r=0.8849 both >0.7 — both stickers fire."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Joker",
            PseudoRandom("S15"),
            1,
            forced_rarity=1,
            game_state=_gs(
                enable_eternals_in_shop=True,
                enable_rentals_in_shop=True,
            ),
        )
        assert card.eternal is True
        assert card.rental is True

    def test_rental_without_eternal(self):
        """Rental is independent: fires even when eternal is disabled."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Joker",
            PseudoRandom("S15"),
            1,
            forced_rarity=1,
            game_state=_gs(
                enable_eternals_in_shop=False,
                enable_rentals_in_shop=True,
            ),
        )
        assert card.eternal is False
        assert card.rental is True


class TestCreateCardEdition:
    """Seed 'CF_JOKER_TEST': edition=negative. Seed 'FOIL_TEST': edition=holo."""

    def test_negative_edition_applied(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("CF_JOKER_TEST"), 1, forced_rarity=1)
        assert card.edition is not None
        assert card.edition.get("negative") is True

    def test_holo_edition_with_mult(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("FOIL_TEST"), 1, forced_rarity=1)
        assert card.edition is not None
        assert card.edition.get("holo") is True
        assert card.edition.get("mult") == 10

    def test_no_edition_for_tarot(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Tarot", PseudoRandom("CF_JOKER_TEST"), 1)
        assert card.edition is None

    def test_no_edition_for_spectral(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Spectral", PseudoRandom("CF_JOKER_TEST"), 1)
        assert card.edition is None

    def test_edition_key_uses_append_and_ante(self):
        """edition stream key = 'edi' + append + str(ante)."""
        from jackdaw.engine.card_utils import poll_edition
        from jackdaw.engine.pools import check_soul_chance, pick_card_from_pool
        from jackdaw.engine.rng import PseudoRandom

        seed, ante, append = "EDI_KEY_CHECK", 3, "sho"
        card = create_card(
            "Joker",
            PseudoRandom(seed),
            ante,
            forced_rarity=1,
            append=append,
        )
        rng_m = PseudoRandom(seed)
        check_soul_chance("Joker", rng_m, ante)
        pick_card_from_pool("Joker", rng_m, ante, rarity=1, append=append)
        rng_m.random("etperpoll" + str(ante))
        rng_m.random("ssjr" + str(ante))
        expected = poll_edition("edi" + append + str(ante), rng_m)
        assert card.edition == expected


class TestCreateCardAreaSeedKeys:
    def test_shop_uses_etperpoll(self):
        """shop: eternal roll stream key = 'etperpoll' + str(ante)."""
        from jackdaw.engine.pools import check_soul_chance, pick_card_from_pool
        from jackdaw.engine.rng import PseudoRandom

        seed, ante = "AREA_SHOP_TEST", 2
        card = create_card(
            "Joker",
            PseudoRandom(seed),
            ante,
            area="shop",
            forced_rarity=1,
            game_state=_gs(enable_eternals_in_shop=True),
        )
        rng_m = PseudoRandom(seed)
        check_soul_chance("Joker", rng_m, ante)
        pick_card_from_pool("Joker", rng_m, ante, rarity=1)
        ep_roll = rng_m.random("etperpoll" + str(ante))
        assert card.eternal == (ep_roll > 0.7)

    def test_pack_uses_packetper(self):
        """pack: eternal roll stream key = 'packetper' + str(ante)."""
        from jackdaw.engine.pools import check_soul_chance, pick_card_from_pool
        from jackdaw.engine.rng import PseudoRandom

        seed, ante, append = "PACK_TEST", 2, "pac"
        card = create_card(
            "Joker",
            PseudoRandom(seed),
            ante,
            area="pack",
            forced_rarity=2,
            append=append,
            game_state=_gs(enable_eternals_in_shop=True),
        )
        rng_m = PseudoRandom(seed)
        check_soul_chance("Joker", rng_m, ante)
        pick_card_from_pool("Joker", rng_m, ante, rarity=2, append=append)
        ep_roll = rng_m.random("packetper" + str(ante))
        assert card.eternal == (ep_roll > 0.7)

    def test_pack_uses_packssjr_for_rental(self):
        """pack: rental roll stream key = 'packssjr' + str(ante)."""
        from jackdaw.engine.pools import check_soul_chance, pick_card_from_pool
        from jackdaw.engine.rng import PseudoRandom

        seed, ante, append = "PACK_TEST", 2, "pac"
        card = create_card(
            "Joker",
            PseudoRandom(seed),
            ante,
            area="pack",
            forced_rarity=2,
            append=append,
            game_state=_gs(enable_rentals_in_shop=True),
        )
        rng_m = PseudoRandom(seed)
        check_soul_chance("Joker", rng_m, ante)
        pick_card_from_pool("Joker", rng_m, ante, rarity=2, append=append)
        rng_m.random("packetper" + str(ante))
        r_roll = rng_m.random("packssjr" + str(ante))
        assert card.rental == (r_roll > 0.7)


class TestCreateCardCost:
    def test_cost_set(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("COST_TEST"), 1, forced_rarity=1)
        assert card.cost >= 1

    def test_discount_reduces_cost(self):
        from jackdaw.engine.rng import PseudoRandom

        full = create_card("Joker", PseudoRandom("DISCOUNT_TEST"), 1, forced_rarity=1)
        disc = create_card(
            "Joker",
            PseudoRandom("DISCOUNT_TEST"),
            1,
            forced_rarity=1,
            game_state=_gs(discount_percent=50),
        )
        assert disc.cost < full.cost

    def test_negative_edition_adds_surcharge(self):
        """Negative edition adds +5 to extra_cost (card.lua:401)."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("CF_JOKER_TEST"), 1, forced_rarity=1)
        assert card.edition is not None and card.edition.get("negative")
        assert card.extra_cost >= 5

    def test_cost_at_least_1(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Tarot",
            PseudoRandom("MIN_COST"),
            1,
            game_state=_gs(discount_percent=99),
        )
        assert card.cost >= 1


class TestCreateCardModifierGating:
    """Modifier rolls only apply to Joker cards in shop/pack context."""

    def test_tarot_no_eternal(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Tarot",
            PseudoRandom("GATE_TEST"),
            1,
            game_state=_gs(enable_eternals_in_shop=True),
        )
        assert card.eternal is False

    def test_planet_no_rental(self):
        from jackdaw.engine.rng import PseudoRandom

        card = create_card(
            "Planet",
            PseudoRandom("GATE_TEST"),
            1,
            game_state=_gs(enable_rentals_in_shop=True),
        )
        assert card.rental is False

    def test_soul_forced_key_no_modifiers(self):
        """c_soul is Spectral — no eternal/rental/edition rolls."""
        from jackdaw.engine.rng import PseudoRandom

        card = create_card("Joker", PseudoRandom("SOUL_MOD"), 1, forced_key="c_soul")
        assert card.ability["set"] == "Spectral"
        assert card.eternal is False
        assert card.rental is False
        assert card.edition is None

    def test_tarot_does_not_consume_modifier_streams(self):
        """Tarot creation leaves etperpoll/ssjr/edi streams untouched."""
        from jackdaw.engine.rng import PseudoRandom

        rng_tarot = PseudoRandom("STREAM_GATE")
        rng_ref = PseudoRandom("STREAM_GATE")
        create_card("Tarot", rng_tarot, 2)

        assert rng_tarot.random("etperpoll2") == rng_ref.random("etperpoll2")
        assert rng_tarot.random("ssjr2") == rng_ref.random("ssjr2")
        assert rng_tarot.random("edi2") == rng_ref.random("edi2")


class TestCreateCardGameStateFiltering:
    def test_banned_key_not_returned(self):
        """banned_keys in game_state forwarded to pool selection."""
        from jackdaw.engine.rng import PseudoRandom

        normal = create_card("Joker", PseudoRandom("BAN_CF"), 1, forced_rarity=1).center_key
        result = create_card(
            "Joker",
            PseudoRandom("BAN_CF"),
            1,
            forced_rarity=1,
            game_state=_gs(banned_keys={normal}),
        ).center_key
        assert result != normal

    def test_used_jokers_not_returned(self):
        """used_jokers in game_state prevents duplicate selection."""
        from jackdaw.engine.rng import PseudoRandom

        normal = create_card("Joker", PseudoRandom("USED_CF"), 1, forced_rarity=1).center_key
        result = create_card(
            "Joker",
            PseudoRandom("USED_CF"),
            1,
            forced_rarity=1,
            game_state=_gs(used_jokers={normal}),
        ).center_key
        assert result != normal
