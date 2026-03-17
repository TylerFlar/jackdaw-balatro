"""Tests for resolve_create_descriptor and resolve_destroy_descriptor.

Validates the bridge between joker/consumable side-effect descriptors
and actual card creation.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import (
    create_joker,
    resolve_create_descriptor,
    resolve_destroy_descriptor,
)
from jackdaw.engine.data.prototypes import JOKERS, PLANETS, TAROTS
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _rng(seed: str = "RESOLVER") -> PseudoRandom:
    return PseudoRandom(seed)


# ---------------------------------------------------------------------------
# resolve_create_descriptor — Tarot
# ---------------------------------------------------------------------------


class TestResolveCreateTarot:
    def test_bare_tarot_returns_tarot_card(self):
        card = resolve_create_descriptor({"type": "Tarot"}, _rng(), 1, {})
        assert card is not None
        assert card.ability.get("set") == "Tarot"

    def test_tarot_with_key_append_returns_tarot(self):
        card = resolve_create_descriptor({"type": "Tarot", "key": "car"}, _rng(), 1, {})
        assert card is not None
        assert card.ability.get("set") == "Tarot"

    def test_tarot_with_seed_append_returns_tarot(self):
        card = resolve_create_descriptor({"type": "Tarot", "seed": "emp"}, _rng(), 1, {})
        assert card is not None
        assert card.ability.get("set") == "Tarot"

    def test_tarot_center_key_is_known(self):
        card = resolve_create_descriptor({"type": "Tarot"}, _rng(), 1, {})
        assert card is not None
        assert card.center_key in TAROTS

    def test_tarot_forced_key_creates_specific_card(self):
        card = resolve_create_descriptor(
            {"type": "Tarot_Planet", "forced_key": "c_magician", "seed": "fool"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.center_key == "c_magician"

    def test_tarot_determinism(self):
        desc = {"type": "Tarot", "key": "car"}
        r1, r2 = _rng("DT1"), _rng("DT1")
        c1 = resolve_create_descriptor(desc, r1, 1, {})
        c2 = resolve_create_descriptor(desc, r2, 1, {})
        assert c1 is not None and c2 is not None
        assert c1.center_key == c2.center_key

    def test_tarot_different_seeds_may_differ(self):
        desc = {"type": "Tarot"}
        keys = {
            resolve_create_descriptor(desc, PseudoRandom(f"TS{i}"), 1, {}).center_key
            for i in range(10)
        }
        assert len(keys) > 1


# ---------------------------------------------------------------------------
# resolve_create_descriptor — Planet
# ---------------------------------------------------------------------------


class TestResolveCreatePlanet:
    def test_planet_descriptor_returns_planet(self):
        card = resolve_create_descriptor(
            {"type": "Planet", "seed": "pri"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.ability.get("set") == "Planet"

    def test_planet_center_key_is_known(self):
        card = resolve_create_descriptor({"type": "Planet"}, _rng(), 1, {})
        assert card is not None
        assert card.center_key in PLANETS


# ---------------------------------------------------------------------------
# resolve_create_descriptor — Spectral
# ---------------------------------------------------------------------------


class TestResolveCreateSpectral:
    def test_spectral_descriptor_returns_spectral(self):
        card = resolve_create_descriptor(
            {"type": "Spectral", "key": "sea"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.ability.get("set") == "Spectral"

    def test_spectral_center_key_is_known(self):
        card = resolve_create_descriptor({"type": "Spectral"}, _rng(), 1, {})
        assert card is not None
        assert card.center_key not in ("", None)


# ---------------------------------------------------------------------------
# resolve_create_descriptor — Joker with rarity
# ---------------------------------------------------------------------------


class TestResolveCreateJoker:
    def test_joker_descriptor_returns_joker(self):
        card = resolve_create_descriptor({"type": "Joker"}, _rng(), 1, {})
        assert card is not None
        assert card.ability.get("set") == "Joker"

    def test_joker_rarity_3_returns_rare(self):
        """Rarity 3 → Rare joker (rarity field on the prototype)."""
        found_rare = False
        for i in range(20):
            card = resolve_create_descriptor(
                {"type": "Joker", "rarity": 3, "seed": "wra"},
                PseudoRandom(f"JR3_{i}"),
                1,
                {},
            )
            assert card is not None
            assert card.ability.get("set") == "Joker"
            assert card.center_key not in ("c_soul",)  # soul is rarity 4
            if JOKERS.get(card.center_key, None) and JOKERS[card.center_key].rarity == 3:
                found_rare = True
        assert found_rare, "rarity=3 should produce Rare jokers"

    def test_joker_rarity_4_returns_legendary(self):
        """Rarity 4 → Legendary joker pool."""
        for i in range(5):
            card = resolve_create_descriptor(
                {"type": "Joker", "rarity": 4, "seed": "soul"},
                PseudoRandom(f"JR4_{i}"),
                1,
                {},
            )
            assert card is not None
            assert card.ability.get("set") in ("Joker",)
            # Legendary jokers have rarity 4 (or are c_soul itself)
            if card.center_key in JOKERS:
                assert JOKERS[card.center_key].rarity == 4

    def test_joker_rarity_common_string(self):
        """'Common' string maps to rarity 1."""
        card = resolve_create_descriptor(
            {"type": "Joker", "rarity": "Common", "key": "rif"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.ability.get("set") == "Joker"
        if card.center_key in JOKERS:
            assert JOKERS[card.center_key].rarity == 1

    def test_joker_no_stickers_in_blank_area(self):
        """Cards created via resolve_create_descriptor should have no stickers."""
        for i in range(20):
            card = resolve_create_descriptor(
                {"type": "Joker", "seed": "jud"},
                PseudoRandom(f"STICK{i}"),
                1,
                {
                    "enable_eternals_in_shop": True,
                    "enable_perishables_in_shop": True,
                    "enable_rentals_in_shop": True,
                },
            )
            assert card is not None
            assert not card.eternal
            assert not card.perishable
            assert not card.rental

    def test_joker_copy_of_returns_deep_copy(self):
        """copy_of descriptor produces an independent deep copy."""
        original = create_joker("j_joker")
        original.ability["mult"] = 99

        card = resolve_create_descriptor(
            {"type": "Joker", "copy_of": original},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card is not original
        assert card.center_key == original.center_key
        assert card.ability["mult"] == 99

        # Mutating the copy must not affect the original
        card.ability["mult"] = 0
        assert original.ability["mult"] == 99

    def test_joker_center_key_is_known(self):
        card = resolve_create_descriptor({"type": "Joker"}, _rng(), 1, {})
        assert card is not None
        # c_soul is a valid legendary substitute
        if card.center_key != "c_soul":
            assert card.center_key in JOKERS


# ---------------------------------------------------------------------------
# resolve_create_descriptor — PlayingCard
# ---------------------------------------------------------------------------


class TestResolveCreatePlayingCard:
    def test_playing_card_explicit_type(self):
        card = resolve_create_descriptor(
            {"type": "PlayingCard", "rank": "King", "suit": "Spades"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.base is not None
        assert card.base.rank.value == "King"
        assert card.base.suit.value == "Spades"

    def test_playing_card_inferred_from_rank_key(self):
        """Descriptor with 'rank' but no 'type' → PlayingCard."""
        card = resolve_create_descriptor(
            {"rank": "Ace", "suit": "Hearts", "enhancement": "m_lucky"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.base is not None
        assert card.base.rank.value == "Ace"
        assert card.base.suit.value == "Hearts"
        assert card.center_key == "m_lucky"

    def test_playing_card_default_enhancement_is_c_base(self):
        card = resolve_create_descriptor(
            {"rank": "2", "suit": "Clubs"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.center_key == "c_base"

    def test_playing_card_all_ranks(self):
        for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]:
            card = resolve_create_descriptor(
                {"rank": rank, "suit": "Diamonds"},
                _rng(),
                1,
                {},
            )
            assert card is not None
            assert card.base.rank.value == rank

    def test_playing_card_missing_suit_returns_none(self):
        card = resolve_create_descriptor(
            {"type": "PlayingCard", "rank": "King"},
            _rng(),
            1,
            {},
        )
        assert card is None

    def test_playing_card_missing_rank_returns_none(self):
        card = resolve_create_descriptor(
            {"type": "PlayingCard", "suit": "Spades"},
            _rng(),
            1,
            {},
        )
        assert card is None


# ---------------------------------------------------------------------------
# resolve_create_descriptor — unknown type
# ---------------------------------------------------------------------------


class TestResolveCreateUnknown:
    def test_unknown_type_returns_none(self):
        card = resolve_create_descriptor(
            {"type": "Unknown"},
            _rng(),
            1,
            {},
        )
        assert card is None

    def test_empty_descriptor_returns_none(self):
        card = resolve_create_descriptor({}, _rng(), 1, {})
        assert card is None


# ---------------------------------------------------------------------------
# resolve_destroy_descriptor — destroy_random_joker
# ---------------------------------------------------------------------------


class TestResolveDestroyRandomJoker:
    def _make_joker(self, key: str = "j_joker") -> Card:
        return create_joker(key)

    def test_returns_a_card(self):
        jokers = [self._make_joker(), self._make_joker(), self._make_joker()]
        result = resolve_destroy_descriptor(
            {"destroy_random_joker": True},
            jokers,
            _rng(),
        )
        assert result is not None
        assert isinstance(result, Card)

    def test_returned_card_is_from_joker_list(self):
        jokers = [self._make_joker(), self._make_joker(), self._make_joker()]
        result = resolve_destroy_descriptor(
            {"destroy_random_joker": True},
            jokers,
            _rng(),
        )
        assert result in jokers

    def test_eternal_joker_never_selected(self):
        """With 3 jokers (1 eternal), only the 2 non-eternal are eligible."""
        j_eternal = create_joker("j_joker", eternal=True)
        j1 = self._make_joker("j_greedy_joker")
        j2 = self._make_joker("j_lusty_joker")
        jokers = [j_eternal, j1, j2]

        selected = set()
        for i in range(20):
            result = resolve_destroy_descriptor(
                {"destroy_random_joker": True},
                jokers,
                PseudoRandom(f"MADNESS{i}"),
            )
            assert result is not None
            selected.add(id(result))

        assert id(j_eternal) not in selected, "Eternal joker must never be selected"

    def test_two_non_eternal_both_selectable(self):
        """Both non-eternal jokers should be reachable across seeds."""
        j_eternal = create_joker("j_joker", eternal=True)
        j1 = create_joker("j_greedy_joker")
        j2 = create_joker("j_lusty_joker")
        jokers = [j_eternal, j1, j2]

        ids_selected = set()
        for i in range(30):
            result = resolve_destroy_descriptor(
                {"destroy_random_joker": True},
                jokers,
                PseudoRandom(f"BOTH{i}"),
            )
            if result is not None:
                ids_selected.add(id(result))

        assert id(j1) in ids_selected, "j1 should be selectable"
        assert id(j2) in ids_selected, "j2 should be selectable"

    def test_all_eternal_returns_none(self):
        jokers = [
            create_joker("j_joker", eternal=True),
            create_joker("j_greedy_joker", eternal=True),
        ]
        result = resolve_destroy_descriptor(
            {"destroy_random_joker": True},
            jokers,
            _rng(),
        )
        assert result is None

    def test_empty_joker_list_returns_none(self):
        result = resolve_destroy_descriptor(
            {"destroy_random_joker": True},
            [],
            _rng(),
        )
        assert result is None

    def test_determinism_same_seed(self):
        jokers = [
            create_joker("j_joker"),
            create_joker("j_greedy_joker"),
            create_joker("j_lusty_joker"),
        ]
        r1, r2 = _rng("MDET"), _rng("MDET")
        res1 = resolve_destroy_descriptor({"destroy_random_joker": True}, jokers, r1)
        res2 = resolve_destroy_descriptor({"destroy_random_joker": True}, jokers, r2)
        assert res1 is res2  # same object (same index selected)


# ---------------------------------------------------------------------------
# resolve_destroy_descriptor — disable_blind
# ---------------------------------------------------------------------------


class TestResolveDestroyDisableBlind:
    def test_disable_blind_returns_none(self):
        result = resolve_destroy_descriptor(
            {"disable_blind": True},
            [],
            _rng(),
        )
        assert result is None

    def test_disable_blind_with_jokers_returns_none(self):
        jokers = [create_joker("j_joker")]
        result = resolve_destroy_descriptor(
            {"disable_blind": True},
            jokers,
            _rng(),
        )
        assert result is None


# ---------------------------------------------------------------------------
# resolve_destroy_descriptor — unknown descriptor
# ---------------------------------------------------------------------------


class TestResolveDestroyUnknown:
    def test_empty_descriptor_returns_none(self):
        result = resolve_destroy_descriptor({}, [], _rng())
        assert result is None

    def test_unrecognised_key_returns_none(self):
        result = resolve_destroy_descriptor(
            {"some_future_effect": True},
            [create_joker("j_joker")],
            _rng(),
        )
        assert result is None
