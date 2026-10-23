"""Tests for the consumable targeting system.

Covers:
- Every tarot card's targeting spec matches actual behavior
- Planet cards all return needs_card_targets=False
- Aura correctly filters out cards with editions
- Death requires exactly 2 targets
- Strength allows 1-2 targets (config: max_highlighted=2)
- Enhancement tarots accept 1-2 or exactly 1 targets
- Suit-change tarots accept 1-3 targets
- Spectrals match their documented targeting
- Seal spectrals require exactly 1 target
- No-target consumables reject card selections
- Validation: bounds, duplicates, filter checks
- The specs agree with the Rust engine's targeting info
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jackdaw.core import Game
from jackdaw.engine.actions import SelectBlind
from jackdaw.engine.data import load
from jackdaw.env.action_space import get_consumable_target_info
from jackdaw.env.consumable_targets import (
    get_consumable_target_spec,
    get_valid_target_cards,
    validate_card_targets,
)

# ---------------------------------------------------------------------------
# Hands: anything with an ``edition`` (a CardView's is a name or None)
# ---------------------------------------------------------------------------


def _make_hand(n: int = 5) -> list[SimpleNamespace]:
    """Create n hand cards without editions."""
    return [SimpleNamespace(edition=None) for _ in range(n)]


def _make_hand_with_editions(editions: list[str | None]) -> list[SimpleNamespace]:
    """Create hand cards with the given editions (None = no edition)."""
    return [SimpleNamespace(edition=ed) for ed in editions]


# =========================================================================
# Tests: Enhancement tarots (1-2 targets)
# =========================================================================


class TestEnhancementTarots:
    """Enhancement tarots use max_highlighted from config."""

    @pytest.mark.parametrize("key", ["c_magician", "c_empress", "c_heirophant"])
    def test_two_target_enhancement(self, key: str):
        """Enhancement tarots with max_highlighted=2 accept 1-2 cards."""
        spec = get_consumable_target_spec(key)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 2
        assert spec.exact_targets is None  # 1 or 2 is fine
        assert spec.target_filter == "any"

    @pytest.mark.parametrize("key", ["c_lovers", "c_chariot", "c_justice", "c_devil", "c_tower"])
    def test_single_target_enhancement(self, key: str):
        """Enhancement tarots with max_highlighted=1 accept exactly 1 card."""
        spec = get_consumable_target_spec(key)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1
        assert spec.target_filter == "any"


# =========================================================================
# Tests: Suit-change tarots (1-3 targets)
# =========================================================================


class TestSuitChangeTarots:
    @pytest.mark.parametrize("key", ["c_star", "c_moon", "c_sun", "c_world"])
    def test_suit_change_up_to_three(self, key: str):
        """Suit-change tarots accept 1-3 cards."""
        spec = get_consumable_target_spec(key)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 3
        assert spec.exact_targets is None


# =========================================================================
# Tests: Transformation tarots
# =========================================================================


class TestTransformationTarots:
    def test_strength_one_to_two(self):
        """Strength: config has max_highlighted=2, no min → 1-2 cards."""
        spec = get_consumable_target_spec("c_strength")
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 2

    def test_death_exactly_two(self):
        """Death: min_highlighted=2, max_highlighted=2 → exactly 2."""
        spec = get_consumable_target_spec("c_death")
        assert spec.needs_card_targets
        assert spec.min_targets == 2
        assert spec.max_targets == 2
        assert spec.exact_targets == 2

    def test_hanged_man_one_to_two(self):
        """Hanged Man: max_highlighted=2, destroys 1-2 cards."""
        spec = get_consumable_target_spec("c_hanged_man")
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 2


# =========================================================================
# Tests: Planet cards (no targets)
# =========================================================================


class TestPlanetCards:
    @pytest.mark.parametrize(
        "key",
        [
            "c_mercury",
            "c_venus",
            "c_earth",
            "c_mars",
            "c_jupiter",
            "c_saturn",
            "c_uranus",
            "c_neptune",
            "c_pluto",
            "c_planet_x",
            "c_ceres",
            "c_eris",
        ],
    )
    def test_planet_no_targets(self, key: str):
        """All planet cards need no card targets."""
        spec = get_consumable_target_spec(key)
        assert not spec.needs_card_targets
        assert spec.min_targets == 0
        assert spec.max_targets == 0

    def test_black_hole_no_targets(self):
        """Black Hole (levels all hands) needs no targets."""
        assert not get_consumable_target_spec("c_black_hole").needs_card_targets


# =========================================================================
# Tests: No-target tarots/spectrals
# =========================================================================


class TestNoTargetConsumables:
    @pytest.mark.parametrize(
        "key",
        [
            "c_hermit",
            "c_temperance",
            "c_wheel_of_fortune",
            "c_fool",
            "c_emperor",
            "c_high_priestess",
            "c_judgement",
            "c_soul",
            "c_wraith",
            "c_ectoplasm",
            "c_hex",
            "c_ankh",
        ],
    )
    def test_no_card_targets(self, key: str):
        """These consumables don't target hand cards."""
        spec = get_consumable_target_spec(key)
        assert not spec.needs_card_targets
        assert spec.min_targets == 0

    @pytest.mark.parametrize("key", ["c_familiar", "c_grim", "c_incantation", "c_immolate"])
    def test_destroy_spectrals_no_selection(self, key: str):
        """Destroy/create spectrals need hand cards but no player selection."""
        assert not get_consumable_target_spec(key).needs_card_targets


# =========================================================================
# Tests: Aura (edition filter)
# =========================================================================


class TestAura:
    def test_aura_spec(self):
        """Aura targets exactly 1 card with no edition."""
        spec = get_consumable_target_spec("c_aura")
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1
        assert spec.target_filter == "no_edition"

    def test_aura_filters_editions(self):
        """Aura only targets cards without editions."""
        hand = _make_hand_with_editions([None, "foil", None, "holo", None])
        assert get_valid_target_cards("c_aura", hand) == [0, 2, 4]

    def test_aura_all_have_editions(self):
        """No valid targets when all hand cards have editions."""
        hand = _make_hand_with_editions(["foil", "holo", "polychrome"])
        assert get_valid_target_cards("c_aura", hand) == []

    def test_aura_no_editions_all_valid(self):
        """All cards valid when none have editions."""
        assert get_valid_target_cards("c_aura", _make_hand(4)) == [0, 1, 2, 3]

    def test_aura_on_a_real_hand(self):
        """A run's hand (CardViews): the editioned card is no target."""
        game = Game.new("b_red", 1, "AURA_HAND")
        game.step(SelectBlind())
        game.debug_add("H_A", edition="foil")
        game.debug_add("c_aura")
        hand = game.hand
        (aura,) = game.consumables
        foil = len(hand) - 1  # the card just added
        valid = get_valid_target_cards(aura, hand)
        assert foil not in valid and len(valid) == len(hand) - 1
        assert not validate_card_targets(aura, (foil,), hand)
        assert validate_card_targets(aura, (0,), hand)


# =========================================================================
# Tests: Sigil and Ouija (all hand cards, no selection)
# =========================================================================


class TestSigilOuija:
    @pytest.mark.parametrize("key", ["c_sigil", "c_ouija"])
    def test_targets_all_no_selection(self, key: str):
        """Sigil/Ouija affect all hand cards — no player selection needed."""
        spec = get_consumable_target_spec(key)
        assert not spec.needs_card_targets
        assert spec.target_filter == "all_hand"

    @pytest.mark.parametrize("key", ["c_sigil", "c_ouija"])
    def test_valid_targets_empty(self, key: str):
        """get_valid_target_cards returns [] since no selection is needed."""
        assert get_valid_target_cards(key, _make_hand(5)) == []


# =========================================================================
# Tests: Seal spectrals (1 target)
# =========================================================================


class TestSealSpectrals:
    @pytest.mark.parametrize("key", ["c_talisman", "c_deja_vu", "c_trance", "c_medium"])
    def test_seal_spectral_one_target(self, key: str):
        """Seal spectrals target exactly 1 card (max_highlighted=1 in config)."""
        spec = get_consumable_target_spec(key)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1


# =========================================================================
# Tests: Cryptid (1 target)
# =========================================================================


class TestCryptid:
    def test_cryptid_one_target(self):
        """Cryptid: max_highlighted=1 in config → exactly 1 card."""
        spec = get_consumable_target_spec("c_cryptid")
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1


# =========================================================================
# Tests: get_valid_target_cards
# =========================================================================


class TestGetValidTargetCards:
    def test_enhancement_tarot_all_valid(self):
        """Enhancement tarots accept any hand card."""
        assert get_valid_target_cards("c_magician", _make_hand(5)) == [0, 1, 2, 3, 4]

    def test_planet_no_valid_targets(self):
        """Planets return no valid targets."""
        assert get_valid_target_cards("c_mercury", _make_hand(5)) == []

    def test_empty_hand(self):
        """Empty hand returns no valid targets."""
        assert get_valid_target_cards("c_magician", []) == []


# =========================================================================
# Tests: validate_card_targets
# =========================================================================


class TestValidateCardTargets:
    def test_valid_single_target(self):
        """Single target within range for max_highlighted=1."""
        assert validate_card_targets("c_chariot", (2,), _make_hand(5))

    def test_valid_two_targets(self):
        """Two targets for max_highlighted=2."""
        assert validate_card_targets("c_magician", (1, 3), _make_hand(5))

    def test_valid_death_two_targets(self):
        """Death requires exactly 2."""
        assert validate_card_targets("c_death", (0, 4), _make_hand(5))

    def test_death_one_target_invalid(self):
        """Death with 1 target is invalid (min=2)."""
        assert not validate_card_targets("c_death", (0,), _make_hand(5))

    def test_too_many_targets(self):
        """3 targets for max_highlighted=2 is invalid."""
        assert not validate_card_targets("c_magician", (0, 1, 2), _make_hand(5))

    def test_zero_targets_when_required(self):
        """Empty selection for a targeting consumable is invalid."""
        assert not validate_card_targets("c_chariot", (), _make_hand(5))

    def test_no_target_consumable_empty_ok(self):
        """No-target consumable accepts empty selection."""
        assert validate_card_targets("c_mercury", (), _make_hand(5))

    def test_no_target_consumable_rejects_targets(self):
        """No-target consumable rejects non-empty selection."""
        assert not validate_card_targets("c_mercury", (0,), _make_hand(5))

    def test_index_out_of_bounds(self):
        """Target index beyond hand size is invalid."""
        assert not validate_card_targets("c_magician", (5,), _make_hand(3))

    def test_negative_index(self):
        """Negative index is invalid."""
        assert not validate_card_targets("c_magician", (-1,), _make_hand(5))

    def test_duplicate_indices(self):
        """Duplicate indices are invalid."""
        assert not validate_card_targets("c_magician", (2, 2), _make_hand(5))

    def test_aura_valid_no_edition(self):
        """Aura accepts a card without edition."""
        hand = _make_hand_with_editions([None, "foil", None])
        assert validate_card_targets("c_aura", (0,), hand)
        assert validate_card_targets("c_aura", (2,), hand)

    def test_aura_rejects_edition_card(self):
        """Aura rejects targeting a card with edition."""
        hand = _make_hand_with_editions([None, "foil", None])
        assert not validate_card_targets("c_aura", (1,), hand)

    def test_aura_rejects_two_targets(self):
        """Aura only accepts exactly 1 target."""
        assert not validate_card_targets("c_aura", (0, 1), _make_hand(5))

    def test_suit_change_three_targets(self):
        """Suit-change tarots accept up to 3."""
        assert validate_card_targets("c_star", (0, 2, 4), _make_hand(5))

    def test_suit_change_four_invalid(self):
        """4 targets for suit-change tarot is invalid."""
        assert not validate_card_targets("c_star", (0, 1, 2, 3), _make_hand(5))


# =========================================================================
# Tests: Edge case — pack opening phase
# =========================================================================


class TestPackOpeningEdgeCase:
    def test_consumable_in_pack_no_hand(self):
        """Consumable from a pack with no hand cards: empty valid targets."""
        assert get_valid_target_cards("c_magician", []) == []
        # Validation also fails since there are no valid targets
        assert not validate_card_targets("c_magician", (0,), [])

    def test_planet_in_pack(self):
        """Planet from a pack always has no targets."""
        assert not get_consumable_target_spec("c_mercury").needs_card_targets
        assert validate_card_targets("c_mercury", (), [])


# =========================================================================
# Tests: The specs against the Rust engine
# =========================================================================

# Their targeting isn't in their config (card.lua's can_use_consumeable): the engine's
# targeting info follows max_highlighted only.
_SPECIAL = {"c_aura", "c_sigil", "c_ouija"}


def test_specs_agree_with_the_engines_targeting_info():
    """Every Tarot, Planet and Spectral: the spec's bounds are the engine's."""
    keys = sorted(
        k for k, v in load("centers").items() if v.get("set") in ("Tarot", "Planet", "Spectral")
    )
    game = Game.new("b_red", 1, "TARGET_SPECS")
    game.step(SelectBlind())
    for key in keys:
        if key in _SPECIAL:
            continue
        run = game.clone()
        run.debug_add(key)
        (card,) = run.consumables
        spec = get_consumable_target_spec(card)
        info = get_consumable_target_info(run, "consumables", 0)
        assert (spec.min_targets, spec.max_targets, spec.needs_card_targets) == info, key
