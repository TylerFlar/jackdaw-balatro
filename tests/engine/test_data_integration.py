"""Data layer integration tests.

Validates that prototypes, enums, card factory, deck builder, and blind
scaling all work together correctly.  Catches cross-reference errors,
missing fields, and inconsistencies between the data tables.
"""

from __future__ import annotations

import time

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.data.blind_scaling import get_blind_target
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.hands import HAND_BASE, HAND_ORDER, HandType
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
    SPECTRALS,
    TAGS,
    TAROTS,
    VOUCHERS,
)
from jackdaw.engine.deck_builder import build_deck
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ============================================================================
# 1. Cross-reference validation
# ============================================================================

class TestCrossReferences:
    """Verify no dangling references between data tables."""

    def test_joker_suit_references_valid(self):
        """Every joker config.extra.suit must be a valid Suit string."""
        valid_suits = {s.value for s in Suit}
        for key, j in JOKERS.items():
            extra = j.config.get("extra")
            if isinstance(extra, dict) and "suit" in extra:
                assert extra["suit"] in valid_suits, (
                    f"{key}: config.extra.suit={extra['suit']!r} not a valid Suit"
                )

    def test_voucher_prerequisites_valid(self):
        """Every voucher's requires list references existing voucher keys."""
        for key, v in VOUCHERS.items():
            for req in v.requires:
                assert req in VOUCHERS, (
                    f"{key}: requires {req!r} but that voucher doesn't exist"
                )

    def test_joker_enhancement_gates_valid(self):
        """Every enhancement_gate references a valid Enhancement key."""
        for key, j in JOKERS.items():
            if j.enhancement_gate:
                assert j.enhancement_gate in ENHANCEMENTS, (
                    f"{key}: enhancement_gate={j.enhancement_gate!r} not found"
                )

    def test_tag_requires_valid(self):
        """Tag requires field references existing center keys."""
        from jackdaw.engine.data.prototypes import _load_json
        all_centers = _load_json("centers.json")
        for key, t in TAGS.items():
            if t.requires:
                assert t.requires in all_centers, (
                    f"Tag {key}: requires {t.requires!r} not in P_CENTERS"
                )

    def test_blind_boss_fields_consistent(self):
        """Boss blinds have min/max ante range, non-bosses don't."""
        for key, b in BLINDS.items():
            if b.boss:
                assert "min" in b.boss, f"{key}: boss missing 'min'"
                assert "max" in b.boss, f"{key}: boss missing 'max'"
            else:
                assert key in ("bl_small", "bl_big"), (
                    f"{key}: not a boss but also not Small/Big"
                )

    def test_hand_types_match_planet_configs(self):
        """Every planet's hand_type matches a HandType value."""
        valid_hands = {ht.value for ht in HandType}
        for key, p in PLANETS.items():
            ht = p.config.get("hand_type")
            if ht:
                assert ht in valid_hands, (
                    f"{key}: hand_type={ht!r} not in HandType"
                )

    def test_all_center_pool_keys_exist(self):
        """Every key in CENTER_POOLS references a real prototype."""
        all_keys = (
            set(JOKERS) | set(TAROTS) | set(PLANETS) | set(SPECTRALS)
            | set(VOUCHERS) | set(BACKS) | set(BOOSTERS) | set(ENHANCEMENTS)
            | set(EDITIONS)
        )
        for pool_name, keys in CENTER_POOLS.items():
            for key in keys:
                assert key in all_keys, (
                    f"CENTER_POOLS[{pool_name!r}] has {key!r} not in any proto dict"
                )


# ============================================================================
# 2. Standard 52-card deck
# ============================================================================

class TestStandard52CardDeck:
    """Build a standard deck and verify all fields."""

    @pytest.fixture
    def deck(self) -> list[Card]:
        rng = PseudoRandom("INTEGRATION")
        return build_deck("b_red", rng)

    def test_52_cards(self, deck: list[Card]):
        assert len(deck) == 52

    def test_4_suits_13_ranks(self, deck: list[Card]):
        for suit in Suit:
            suit_cards = [c for c in deck if c.base.suit is suit]
            assert len(suit_cards) == 13

    def test_all_ranks(self, deck: list[Card]):
        for rank in Rank:
            rank_cards = [c for c in deck if c.base.rank is rank]
            assert len(rank_cards) == 4  # one per suit

    def test_ace_nominal_11(self, deck: list[Card]):
        aces = [c for c in deck if c.base.rank is Rank.ACE]
        for a in aces:
            assert a.base.nominal == 11

    def test_king_nominal_10(self, deck: list[Card]):
        kings = [c for c in deck if c.base.rank is Rank.KING]
        for k in kings:
            assert k.base.nominal == 10

    def test_number_cards_nominal(self, deck: list[Card]):
        for c in deck:
            if c.base.rank.value.isdigit():
                assert c.base.nominal == int(c.base.rank.value)

    def test_ability_from_c_base(self, deck: list[Card]):
        for c in deck:
            assert c.center_key == "c_base"
            assert c.ability["name"] == "Default Base"

    def test_sort_ids_unique(self, deck: list[Card]):
        ids = [c.sort_id for c in deck]
        assert len(ids) == len(set(ids))

    def test_sort_ids_sequential(self, deck: list[Card]):
        ids = [c.sort_id for c in deck]
        assert ids == list(range(ids[0], ids[0] + len(ids)))


# ============================================================================
# 3. Abandoned Deck
# ============================================================================

class TestAbandonedDeckIntegration:
    def test_40_cards_no_faces(self):
        rng = PseudoRandom("TESTSEED")
        deck = build_deck("b_abandoned", rng)
        assert len(deck) == 40
        for c in deck:
            assert c.base.rank not in (Rank.JACK, Rank.QUEEN, Rank.KING)


# ============================================================================
# 4. Erratic Deck
# ============================================================================

class TestErraticDeckIntegration:
    def test_52_cards_with_duplicates(self):
        rng = PseudoRandom("TESTSEED")
        deck = build_deck("b_erratic", rng)
        assert len(deck) == 52
        keys = [c.card_key for c in deck]
        assert len(set(keys)) < 52  # duplicates from randomization

    def test_deterministic(self):
        d1 = build_deck("b_erratic", PseudoRandom("TESTSEED"))
        reset_sort_id_counter()
        d2 = build_deck("b_erratic", PseudoRandom("TESTSEED"))
        assert [c.card_key for c in d1] == [c.card_key for c in d2]


# ============================================================================
# 5. All 150 jokers
# ============================================================================

class TestAllJokers:
    """Create every joker and verify key properties."""

    @pytest.fixture(scope="class")
    def all_joker_cards(self) -> dict[str, Card]:
        reset_sort_id_counter()
        return {key: create_joker(key) for key in JOKERS}

    def test_150_jokers_created(self, all_joker_cards: dict[str, Card]):
        assert len(all_joker_cards) == 150

    def test_names_match_prototypes(self, all_joker_cards: dict[str, Card]):
        for key, card in all_joker_cards.items():
            assert card.ability["name"] == JOKERS[key].name

    def test_x_mult_default_is_1(self, all_joker_cards: dict[str, Card]):
        """Jokers without top-level config.Xmult should have x_mult=1."""
        for key, card in all_joker_cards.items():
            if "Xmult" not in JOKERS[key].config:
                assert card.ability["x_mult"] == 1, (
                    f"{key}: x_mult={card.ability['x_mult']} but no Xmult in config"
                )

    def test_x_mult_matches_config(self, all_joker_cards: dict[str, Card]):
        """Jokers with top-level config.Xmult should have that value."""
        for key, card in all_joker_cards.items():
            if "Xmult" in JOKERS[key].config:
                expected = JOKERS[key].config["Xmult"]
                assert card.ability["x_mult"] == expected, (
                    f"{key}: x_mult={card.ability['x_mult']} != config.Xmult={expected}"
                )

    def test_deep_copy_isolation(self, all_joker_cards: dict[str, Card]):
        """Mutating one joker's extra must not affect another."""
        if "j_greedy_joker" in all_joker_cards:
            c1 = all_joker_cards["j_greedy_joker"]
            c2 = create_joker("j_greedy_joker")
            c1.ability["extra"]["s_mult"] = 999
            assert c2.ability["extra"]["s_mult"] == 3

    def test_invisible_joker_post_init(self, all_joker_cards: dict[str, Card]):
        c = all_joker_cards["j_invisible"]
        assert c.ability.get("invis_rounds") == 0

    def test_caino_post_init(self, all_joker_cards: dict[str, Card]):
        c = all_joker_cards["j_caino"]
        assert c.ability.get("caino_xmult") == 1

    def test_yorick_post_init(self, all_joker_cards: dict[str, Card]):
        c = all_joker_cards["j_yorick"]
        assert "yorick_discards" in c.ability
        assert c.ability["yorick_discards"] > 0

    def test_loyalty_card_post_init(self, all_joker_cards: dict[str, Card]):
        c = all_joker_cards["j_loyalty_card"]
        assert c.ability.get("loyalty_remaining") == 5
        assert c.ability.get("burnt_hand") == 0

    def test_all_have_hands_played_at_create(self, all_joker_cards: dict[str, Card]):
        for key, card in all_joker_cards.items():
            assert "hands_played_at_create" in card.ability, (
                f"{key}: missing hands_played_at_create"
            )


# ============================================================================
# 6. Rarity pools
# ============================================================================

class TestRarityPoolIntegrity:
    def test_total_is_150(self):
        total = sum(len(pool) for pool in JOKER_RARITY_POOLS.values())
        assert total == 150

    def test_no_duplicates_across_pools(self):
        all_keys: list[str] = []
        for pool in JOKER_RARITY_POOLS.values():
            all_keys.extend(pool)
        assert len(all_keys) == len(set(all_keys))

    def test_every_joker_in_a_pool(self):
        pooled = set()
        for pool in JOKER_RARITY_POOLS.values():
            pooled.update(pool)
        assert pooled == set(JOKERS.keys())


# ============================================================================
# 7. Blind scaling
# ============================================================================

class TestBlindScalingIntegration:
    def test_ante_8_boss_scaling_3(self):
        target = get_blind_target(8, "Boss", scaling=3, ante_scaling=1.0)
        assert target == 400_000  # 200000 × 2

    def test_ante_1_small_blind(self):
        target = get_blind_target(1, "Small", scaling=1)
        assert target == 300

    def test_plasma_ante_8_boss(self):
        target = get_blind_target(8, "Boss", scaling=1, ante_scaling=2.0)
        assert target == 200_000  # 50000 × 2 × 2


# ============================================================================
# 8. Hand data completeness
# ============================================================================

class TestHandDataIntegration:
    def test_all_12_hands_in_base(self):
        assert len(HAND_BASE) == 12

    def test_hand_order_has_all(self):
        assert set(HAND_ORDER) == set(HandType)

    def test_every_planet_has_a_hand(self):
        """Each planet levels up a hand that exists in HAND_BASE."""
        for key, p in PLANETS.items():
            ht_str = p.config.get("hand_type")
            if ht_str:
                ht = HandType(ht_str)
                assert ht in HAND_BASE, f"{key}: hand_type {ht_str!r} not in HAND_BASE"


# ============================================================================
# 9. Performance
# ============================================================================

class TestPerformance:
    def test_deck_build_speed(self):
        """Building a 52-card deck should be fast."""
        n = 100
        start = time.perf_counter()
        for _ in range(n):
            reset_sort_id_counter()
            build_deck("b_red", PseudoRandom("BENCH"))
        elapsed = time.perf_counter() - start
        rate = n / elapsed
        print(f"\n  Deck build: {n} decks in {elapsed:.3f}s = {rate:,.0f} decks/sec")
        assert elapsed < 5.0

    def test_joker_creation_speed(self):
        """Creating 150 jokers should be fast."""
        n = 50
        start = time.perf_counter()
        for _ in range(n):
            reset_sort_id_counter()
            for key in JOKERS:
                create_joker(key)
        elapsed = time.perf_counter() - start
        total = n * 150
        rate = total / elapsed
        print(f"\n  Joker creation: {total:,} jokers in {elapsed:.3f}s = {rate:,.0f}/sec")
        assert elapsed < 5.0
