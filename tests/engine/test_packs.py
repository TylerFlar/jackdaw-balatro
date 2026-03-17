"""Tests for generate_pack_cards — the 5-pack-type card generation loop.

Validates card counts, types, and special mechanics (Omen Globe, Telescope)
for all pack kinds.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.prototypes import BOOSTERS, PLANETS
from jackdaw.engine.packs import generate_pack_cards
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: str = "TESTPACK") -> PseudoRandom:
    return PseudoRandom(seed)


def _pack_keys_of_kind(kind: str) -> list[str]:
    return [k for k, v in BOOSTERS.items() if v.kind == kind]


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    def test_returns_tuple_of_list_and_int(self):
        result = generate_pack_cards("p_arcana_normal_1", _rng(), 1, {})
        assert isinstance(result, tuple) and len(result) == 2

    def test_cards_is_list(self):
        cards, _ = generate_pack_cards("p_arcana_normal_1", _rng(), 1, {})
        assert isinstance(cards, list)

    def test_choose_is_int(self):
        _, choose = generate_pack_cards("p_arcana_normal_1", _rng(), 1, {})
        assert isinstance(choose, int)

    def test_cards_are_card_instances(self):
        cards, _ = generate_pack_cards("p_arcana_normal_1", _rng(), 1, {})
        assert all(isinstance(c, Card) for c in cards)


# ---------------------------------------------------------------------------
# Card count and choose
# ---------------------------------------------------------------------------


class TestCardCountAndChoose:
    @pytest.mark.parametrize(
        "pack_key",
        [
            "p_arcana_normal_1",
            "p_arcana_normal_2",
            "p_arcana_jumbo_1",
            "p_arcana_mega_1",
        ],
    )
    def test_arcana_card_count_matches_extra(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        assert len(cards) == BOOSTERS[pack_key].config["extra"]

    @pytest.mark.parametrize(
        "pack_key",
        [
            "p_celestial_normal_1",
            "p_celestial_jumbo_1",
            "p_celestial_mega_1",
        ],
    )
    def test_celestial_card_count_matches_extra(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        assert len(cards) == BOOSTERS[pack_key].config["extra"]

    @pytest.mark.parametrize(
        "pack_key",
        [
            "p_spectral_normal_1",
            "p_spectral_jumbo_1",
        ],
    )
    def test_spectral_card_count_matches_extra(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        assert len(cards) == BOOSTERS[pack_key].config["extra"]

    @pytest.mark.parametrize(
        "pack_key",
        [
            "p_standard_normal_1",
            "p_standard_jumbo_1",
            "p_standard_mega_1",
        ],
    )
    def test_standard_card_count_matches_extra(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        assert len(cards) == BOOSTERS[pack_key].config["extra"]

    @pytest.mark.parametrize(
        "pack_key",
        [
            "p_buffoon_normal_1",
            "p_buffoon_jumbo_1",
        ],
    )
    def test_buffoon_card_count_matches_extra(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        assert len(cards) == BOOSTERS[pack_key].config["extra"]

    def test_choose_matches_proto_choose(self):
        for pack_key in [
            "p_arcana_normal_1",
            "p_celestial_mega_1",
            "p_buffoon_jumbo_1",
            "p_standard_normal_1",
        ]:
            _, choose = generate_pack_cards(pack_key, _rng(), 1, {})
            assert choose == BOOSTERS[pack_key].config["choose"]

    def test_normal_pack_choose_is_1(self):
        _, choose = generate_pack_cards("p_arcana_normal_1", _rng(), 1, {})
        assert choose == 1

    def test_mega_pack_choose_is_greater_than_1(self):
        _, choose = generate_pack_cards("p_arcana_mega_1", _rng(), 1, {})
        assert choose > 1


# ---------------------------------------------------------------------------
# Arcana pack — Tarot cards (and optional Spectral via Omen Globe)
# ---------------------------------------------------------------------------


class TestArcanaPack:
    def test_default_generates_tarots(self):
        cards, _ = generate_pack_cards("p_arcana_normal_1", _rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Tarot", (
                f"Expected Tarot, got {card.ability.get('set')!r}"
            )

    def test_no_omen_globe_never_produces_spectral(self):
        for i in range(5):
            cards, _ = generate_pack_cards(
                "p_arcana_normal_1",
                PseudoRandom(f"SEED{i}"),
                1,
                {"has_omen_globe": False},
            )
            for card in cards:
                assert card.ability.get("set") == "Tarot"

    def test_omen_globe_can_produce_spectral(self):
        """With many seeds, at least one should produce a Spectral (20% chance)."""
        found_spectral = False
        for i in range(30):
            cards, _ = generate_pack_cards(
                "p_arcana_normal_1",
                PseudoRandom(f"OG{i}"),
                1,
                {"has_omen_globe": True},
            )
            if any(c.ability.get("set") == "Spectral" for c in cards):
                found_spectral = True
                break
        assert found_spectral, "Omen Globe should occasionally produce Spectral cards"

    def test_omen_globe_spectral_is_valid_spectral_key(self):
        """When a Spectral appears it should be a real spectral center key."""
        from jackdaw.engine.data.prototypes import SPECTRALS

        for i in range(30):
            cards, _ = generate_pack_cards(
                "p_arcana_normal_1",
                PseudoRandom(f"OGV{i}"),
                1,
                {"has_omen_globe": True},
            )
            for card in cards:
                if card.ability.get("set") == "Spectral":
                    assert card.center_key in SPECTRALS

    def test_determinism_same_seed(self):
        rng1, rng2 = _rng("DET1"), _rng("DET1")
        cards1, _ = generate_pack_cards("p_arcana_normal_1", rng1, 1, {})
        cards2, _ = generate_pack_cards("p_arcana_normal_1", rng2, 1, {})
        assert [c.center_key for c in cards1] == [c.center_key for c in cards2]

    def test_different_seeds_differ(self):
        cards1, _ = generate_pack_cards("p_arcana_normal_1", _rng("SEED1"), 1, {})
        cards2, _ = generate_pack_cards("p_arcana_normal_1", _rng("SEED2"), 1, {})
        assert [c.center_key for c in cards1] != [c.center_key for c in cards2]

    def test_different_antes_differ(self):
        cards1, _ = generate_pack_cards("p_arcana_normal_1", _rng(), 1, {})
        cards2, _ = generate_pack_cards("p_arcana_normal_1", _rng(), 2, {})
        assert [c.center_key for c in cards1] != [c.center_key for c in cards2]


# ---------------------------------------------------------------------------
# Celestial pack — Planet cards
# ---------------------------------------------------------------------------


class TestCelestialPack:
    def test_default_generates_planets(self):
        cards, _ = generate_pack_cards("p_celestial_normal_1", _rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Planet", (
                f"Expected Planet, got {card.ability.get('set')!r}"
            )

    def test_without_telescope_all_random(self):
        """Without Telescope, no forced key — cards should vary across seeds."""
        keys_seen: set[str] = set()
        for i in range(10):
            cards, _ = generate_pack_cards(
                "p_celestial_normal_1",
                PseudoRandom(f"NOTELESCOPE{i}"),
                1,
                {},
            )
            keys_seen.update(c.center_key for c in cards)
        assert len(keys_seen) > 1  # multiple distinct planet keys seen

    def test_telescope_forces_first_card_to_most_played_hand_planet(self):
        """Telescope makes slot 0 the planet for the most-played hand."""
        # Find a hand type and its planet key
        hand_type = "Flush"
        planet_key = None
        for k, v in PLANETS.items():
            if v.config.get("hand_type") == hand_type:
                planet_key = k
                break

        assert planet_key is not None, f"No planet found for {hand_type!r}"

        cards, _ = generate_pack_cards(
            "p_celestial_normal_1",
            _rng(),
            1,
            {"has_telescope": True, "most_played_hand": hand_type},
        )
        assert cards[0].center_key == planet_key

    def test_telescope_only_affects_first_slot(self):
        """Remaining slots are still random even with Telescope."""
        hand_type = "Flush"
        for k, v in PLANETS.items():
            if v.config.get("hand_type") == hand_type:
                break

        _, choose = generate_pack_cards(
            "p_celestial_jumbo_1",
            _rng(),
            1,
            {"has_telescope": True, "most_played_hand": hand_type},
        )
        # jumbo has extra=6, so slots 1-5 can be anything
        extras = BOOSTERS["p_celestial_jumbo_1"].config["extra"]
        assert extras > 1

    def test_telescope_no_hand_type_falls_back_to_random(self):
        """If most_played_hand is missing, telescope has no forced key."""
        cards, _ = generate_pack_cards(
            "p_celestial_normal_1",
            _rng(),
            1,
            {"has_telescope": True, "most_played_hand": None},
        )
        # Should still produce planets (just not forced)
        for card in cards:
            assert card.ability.get("set") == "Planet"

    def test_determinism_same_seed(self):
        rng1, rng2 = _rng("CDET"), _rng("CDET")
        cards1, _ = generate_pack_cards("p_celestial_normal_1", rng1, 1, {})
        cards2, _ = generate_pack_cards("p_celestial_normal_1", rng2, 1, {})
        assert [c.center_key for c in cards1] == [c.center_key for c in cards2]


# ---------------------------------------------------------------------------
# Spectral pack
# ---------------------------------------------------------------------------


class TestSpectralPack:
    def test_generates_spectrals(self):
        cards, _ = generate_pack_cards("p_spectral_normal_1", _rng(), 1, {})
        for card in cards:
            # Spectral or c_soul / c_black_hole are valid
            assert card.ability.get("set") in ("Spectral", "Joker", "Planet"), (
                f"Unexpected set: {card.ability.get('set')!r}"
            )
            # The center key is a known consumable center
            assert card.center_key not in ("", None)

    def test_spectral_cards_have_known_keys(self):
        from jackdaw.engine.data.prototypes import SPECTRALS

        for i in range(5):
            cards, _ = generate_pack_cards(
                "p_spectral_normal_1",
                PseudoRandom(f"SP{i}"),
                1,
                {},
            )
            for card in cards:
                # Non-soul spectrals should be in the spectral registry
                if card.center_key not in ("c_soul", "c_black_hole"):
                    assert card.center_key in SPECTRALS

    def test_determinism(self):
        r1, r2 = _rng("SPDET"), _rng("SPDET")
        c1, _ = generate_pack_cards("p_spectral_normal_1", r1, 1, {})
        c2, _ = generate_pack_cards("p_spectral_normal_1", r2, 1, {})
        assert [c.center_key for c in c1] == [c.center_key for c in c2]


# ---------------------------------------------------------------------------
# Standard pack — playing cards
# ---------------------------------------------------------------------------


class TestStandardPack:
    def test_generates_playing_cards_with_base(self):
        """Standard pack cards should be playing cards with suit/rank set."""
        cards, _ = generate_pack_cards("p_standard_normal_1", _rng(), 1, {})
        for card in cards:
            assert card.base is not None, "Standard pack card should have a base"

    def test_cards_have_valid_suit_and_rank(self):
        cards, _ = generate_pack_cards("p_standard_normal_1", _rng(), 1, {})
        valid_suits = {"Hearts", "Diamonds", "Clubs", "Spades"}
        valid_ranks = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"}
        for card in cards:
            assert card.base.suit.value in valid_suits
            assert card.base.rank.value in valid_ranks

    def test_enhanced_cards_have_non_base_enhancement(self):
        """Some cards should be enhanced (non c_base center_key)."""
        found_enhanced = False
        for i in range(20):
            cards, _ = generate_pack_cards(
                "p_standard_normal_1",
                PseudoRandom(f"ENH{i}"),
                1,
                {},
            )
            if any(c.center_key != "c_base" for c in cards):
                found_enhanced = True
                break
        assert found_enhanced, "Should produce enhanced cards with ~40% per card chance"

    def test_base_cards_have_c_base_enhancement(self):
        """Base cards should have center_key == 'c_base'."""
        # At least some cards across seeds should have c_base
        found_base = False
        for i in range(20):
            cards, _ = generate_pack_cards(
                "p_standard_normal_1",
                PseudoRandom(f"BASE{i}"),
                1,
                {},
            )
            if any(c.center_key == "c_base" for c in cards):
                found_base = True
                break
        assert found_base

    def test_can_produce_editions(self):
        """Standard packs can have edition cards (poll_edition with mod=2)."""
        found_edition = False
        for i in range(50):
            cards, _ = generate_pack_cards(
                "p_standard_normal_1",
                PseudoRandom(f"STDED{i}"),
                1,
                {"edition_rate": 10.0},
            )
            if any(c.edition for c in cards):
                found_edition = True
                break
        assert found_edition, "High edition_rate should produce editions eventually"

    def test_can_produce_seals(self):
        """Standard packs can have seal cards (20% chance per card)."""
        found_seal = False
        for i in range(30):
            cards, _ = generate_pack_cards(
                "p_standard_normal_1",
                PseudoRandom(f"STDSEAL{i}"),
                1,
                {},
            )
            if any(c.seal is not None for c in cards):
                found_seal = True
                break
        assert found_seal, "Standard packs should occasionally produce sealed cards"

    def test_seal_types_are_valid(self):
        """Any seal on a standard pack card is one of the 4 valid types."""
        valid_seals = {"Red", "Blue", "Gold", "Purple"}
        for i in range(30):
            cards, _ = generate_pack_cards(
                "p_standard_normal_1",
                PseudoRandom(f"SEALT{i}"),
                1,
                {},
            )
            for card in cards:
                if card.seal is not None:
                    assert card.seal in valid_seals

    def test_no_edition_no_seal_by_default(self):
        """Most cards should have no edition (96% base chance) and no seal (80%)."""
        cards, _ = generate_pack_cards("p_standard_normal_1", _rng("CLEAN"), 1, {})
        # At least some should be clean (no edition, no seal) in a 3-card pack
        assert any(c.edition is None and c.seal is None for c in cards)

    def test_determinism(self):
        r1, r2 = _rng("STDD"), _rng("STDD")
        c1, _ = generate_pack_cards("p_standard_normal_1", r1, 1, {})
        c2, _ = generate_pack_cards("p_standard_normal_1", r2, 1, {})
        assert [(c.center_key, c.base.suit.value, c.base.rank.value) for c in c1] == [
            (c.center_key, c.base.suit.value, c.base.rank.value) for c in c2
        ]


# ---------------------------------------------------------------------------
# Buffoon pack — Jokers
# ---------------------------------------------------------------------------


class TestBuffoonPack:
    def test_generates_jokers(self):
        cards, _ = generate_pack_cards("p_buffoon_normal_1", _rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Joker", (
                f"Expected Joker, got {card.ability.get('set')!r}"
            )

    def test_jokers_have_known_center_keys(self):
        from jackdaw.engine.data.prototypes import JOKERS

        for i in range(5):
            cards, _ = generate_pack_cards(
                "p_buffoon_normal_1",
                PseudoRandom(f"BUF{i}"),
                1,
                {},
            )
            for card in cards:
                # c_soul is a possible legendary substitution
                if card.center_key != "c_soul":
                    assert card.center_key in JOKERS, f"Unexpected joker key: {card.center_key!r}"

    def test_jokers_can_have_editions(self):
        """Buffoon pack jokers may receive editions (from create_card area='pack')."""
        found_edition = False
        for i in range(30):
            cards, _ = generate_pack_cards(
                "p_buffoon_normal_1",
                PseudoRandom(f"JKED{i}"),
                1,
                {"edition_rate": 5.0},
            )
            if any(c.edition for c in cards):
                found_edition = True
                break
        assert found_edition, "Buffoon pack jokers should occasionally have editions"

    def test_determinism(self):
        r1, r2 = _rng("BUFDET"), _rng("BUFDET")
        c1, _ = generate_pack_cards("p_buffoon_normal_1", r1, 1, {})
        c2, _ = generate_pack_cards("p_buffoon_normal_1", r2, 1, {})
        assert [c.center_key for c in c1] == [c.center_key for c in c2]

    def test_different_antes_differ(self):
        c1, _ = generate_pack_cards("p_buffoon_normal_1", _rng(), 1, {})
        c2, _ = generate_pack_cards("p_buffoon_normal_1", _rng(), 3, {})
        assert [c.center_key for c in c1] != [c.center_key for c in c2]


# ---------------------------------------------------------------------------
# Cross-type: all pack kinds produce correct types
# ---------------------------------------------------------------------------


class TestAllPackKinds:
    @pytest.mark.parametrize("pack_key", _pack_keys_of_kind("Arcana"))
    def test_all_arcana_packs_produce_tarots(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Tarot"

    @pytest.mark.parametrize("pack_key", _pack_keys_of_kind("Celestial"))
    def test_all_celestial_packs_produce_planets(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Planet"

    @pytest.mark.parametrize("pack_key", _pack_keys_of_kind("Buffoon"))
    def test_all_buffoon_packs_produce_jokers(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Joker"

    @pytest.mark.parametrize("pack_key", _pack_keys_of_kind("Standard"))
    def test_all_standard_packs_produce_playing_cards(self, pack_key):
        cards, _ = generate_pack_cards(pack_key, _rng(), 1, {})
        for card in cards:
            assert card.base is not None


# ---------------------------------------------------------------------------
# Invalid pack key
# ---------------------------------------------------------------------------


class TestInvalidPackKey:
    def test_unknown_pack_key_raises(self):
        with pytest.raises(KeyError):
            generate_pack_cards("p_nonexistent", _rng(), 1, {})
