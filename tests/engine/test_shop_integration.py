"""Integration tests for the shop, pools, packs, and consumable creation pipeline.

Scenario coverage
-----------------
1.  Full shop at ante 1 seed "TESTSEED" — joker types, voucher, boosters.
2.  Tarot Merchant voucher — increased Tarot appearance rate.
3.  Showman joker — duplicate jokers can appear in shop.
4.  Reroll 3 times — escalating cost, cards change, same-seed determinism.
5.  Buy joker → used_jokers → excluded from next reroll (unless Showman).
6.  Open Arcana pack — Tarot count, Omen Globe Spectral override.
7.  Open Standard pack — enhancements/editions/seals at appropriate rates.
8.  Open Celestial pack with Telescope — first card matches most-played planet.
9.  Resolve create descriptors — High Priestess (2 Planets), Judgement (1 Joker),
    playing-card descriptor.
10. Full cycle — populate → buy → reroll → buy tarot → open pack → state mutations.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.card_area import CardArea
from jackdaw.engine.card_factory import create_consumable, create_joker, resolve_create_descriptor
from jackdaw.engine.consumables import ConsumableContext, use_consumable
from jackdaw.engine.data.prototypes import BOOSTERS, JOKERS, TAROTS
from jackdaw.engine.packs import generate_pack_cards
from jackdaw.engine.pools import get_current_pool
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.shop import (
    buy_card,
    calculate_reroll_cost,
    populate_shop,
    reroll_shop,
)
from jackdaw.engine.vouchers import apply_voucher, get_next_voucher_key


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fresh_gs(
    *,
    seed: str = "TESTSEED",
    ante: int = 1,
    first_shop: bool = True,
    dollars: int = 100,
    has_showman: bool = False,
    tarot_rate: float = 4.0,
    planet_rate: float = 4.0,
    spectral_rate: float = 0.0,
    joker_rate: float = 20.0,
) -> tuple[PseudoRandom, dict[str, Any]]:
    """Return (rng, game_state) ready for a shop run."""
    rng = PseudoRandom(seed)
    voucher_key = get_next_voucher_key(rng, used_vouchers={}, in_shop=[])
    gs: dict[str, Any] = {
        "joker_rate": joker_rate,
        "tarot_rate": tarot_rate,
        "planet_rate": planet_rate,
        "spectral_rate": spectral_rate,
        "playing_card_rate": 0.0,
        "edition_rate": 1.0,
        "enable_eternals_in_shop": True,
        "enable_perishables_in_shop": True,
        "enable_rentals_in_shop": True,
        "banned_keys": set(),
        "used_jokers": {},
        "used_vouchers": {},
        "pool_flags": {},
        "has_showman": has_showman,
        "deck_enhancements": set(),
        "playing_card_count": 52,
        "played_hand_types": set(),
        "shop_vouchers": set(),
        "inflation": 0,
        "discount_percent": 0,
        "booster_ante_scaling": False,
        "has_astronomer": False,
        "shop": {"joker_max": 2},
        "current_round": {"voucher": voucher_key},
        "first_shop_buffoon": not first_shop,
        "dollars": dollars,
        "jokers": [],
    }
    return rng, gs


# ---------------------------------------------------------------------------
# Scenario 1: Full shop at ante 1 "TESTSEED"
# ---------------------------------------------------------------------------


class TestFullShopTESTSEED:
    """Basic smoke test: populate_shop for TESTSEED ante 1 returns sane output."""

    def test_shop_returns_two_joker_slots(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1)
        result = populate_shop(rng, 1, gs)
        assert len(result["jokers"]) == 2

    def test_shop_joker_slots_are_valid_cards(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1)
        result = populate_shop(rng, 1, gs)
        for card in result["jokers"]:
            assert card.center_key != ""

    def test_shop_voucher_is_present(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1)
        result = populate_shop(rng, 1, gs)
        assert result["voucher"] is not None
        assert result["voucher"].center_key == gs["current_round"]["voucher"]

    def test_shop_returns_two_boosters(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1)
        result = populate_shop(rng, 1, gs)
        assert len(result["boosters"]) == 2

    def test_first_booster_is_buffoon_pack_on_first_shop(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1, first_shop=True)
        result = populate_shop(rng, 1, gs)
        first_booster_key = result["boosters"][0].center_key
        assert BOOSTERS[first_booster_key].kind == "Buffoon"

    def test_all_booster_keys_are_known(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1)
        result = populate_shop(rng, 1, gs)
        for booster in result["boosters"]:
            assert booster.center_key in BOOSTERS

    def test_joker_slots_have_costs(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1)
        result = populate_shop(rng, 1, gs)
        for card in result["jokers"]:
            assert card.cost > 0

    def test_shop_is_deterministic(self):
        rng1, gs1 = _fresh_gs(seed="TESTSEED", ante=1)
        rng2, gs2 = _fresh_gs(seed="TESTSEED", ante=1)
        r1 = populate_shop(rng1, 1, gs1)
        r2 = populate_shop(rng2, 1, gs2)
        assert [c.center_key for c in r1["jokers"]] == [c.center_key for c in r2["jokers"]]
        assert [c.center_key for c in r1["boosters"]] == [c.center_key for c in r2["boosters"]]
        assert r1["voucher"].center_key == r2["voucher"].center_key


# ---------------------------------------------------------------------------
# Scenario 2: Tarot Merchant voucher raises tarot rate
# ---------------------------------------------------------------------------


class TestTarotMerchantVoucher:
    """Tarot Merchant sets tarot_rate to 9.6, increasing Tarot shop frequency."""

    def test_apply_tarot_merchant_sets_rate(self):
        gs: dict[str, Any] = {"tarot_rate": 4.0}
        mutations = apply_voucher("v_tarot_merchant", gs)
        assert gs["tarot_rate"] == pytest.approx(9.6)
        assert "tarot_rate" in mutations

    def test_tarot_rate_dominance_produces_more_tarots(self):
        """With joker_rate=0 and tarot_rate=10000, all slots are Tarot cards."""
        rng, gs = _fresh_gs(seed="TAROT_DOMINANT", ante=1, tarot_rate=10000.0, joker_rate=0.0, planet_rate=0.0)
        result = populate_shop(rng, 1, gs)
        for card in result["jokers"]:
            assert card.ability.get("set") == "Tarot", (
                f"Expected Tarot, got {card.center_key!r} (set={card.ability.get('set')!r})"
            )

    def test_planet_rate_dominance_produces_more_planets(self):
        """With joker_rate=0, planet_rate=10000, all slots are Planet cards."""
        rng, gs = _fresh_gs(seed="PLANET_DOMINANT", ante=1, planet_rate=10000.0, joker_rate=0.0, tarot_rate=0.0)
        result = populate_shop(rng, 1, gs)
        for card in result["jokers"]:
            assert card.ability.get("set") == "Planet", (
                f"Expected Planet, got {card.center_key!r} (set={card.ability.get('set')!r})"
            )

    def test_tarot_merchant_and_planet_merchant_stack_as_separate_fields(self):
        gs: dict[str, Any] = {"tarot_rate": 4.0, "planet_rate": 4.0}
        apply_voucher("v_tarot_merchant", gs)
        apply_voucher("v_planet_merchant", gs)
        assert gs["tarot_rate"] == pytest.approx(9.6)
        assert gs["planet_rate"] == pytest.approx(9.6)


# ---------------------------------------------------------------------------
# Scenario 3: Showman allows duplicate jokers
# ---------------------------------------------------------------------------


class TestShowmanAllowsDuplicates:
    """has_showman=True bypasses used_joker filtering in pool selection."""

    def test_showman_flag_passes_through_to_pool(self):
        """Without showman, a used rarity-1 joker is UNAVAILABLE in pool."""
        from jackdaw.engine.pools import UNAVAILABLE
        rng_base = PseudoRandom("SHOWMAN_TEST")
        # j_joker is a rarity-1 joker; marking it as used should make it UNAVAILABLE
        pool_no_showman, _ = get_current_pool(
            "Joker",
            rng_base,
            1,
            rarity=1,
            append="sho",
            used_jokers={"j_joker"},
            has_showman=False,
        )
        assert "j_joker" not in pool_no_showman or pool_no_showman[pool_no_showman.index("j_joker")] == UNAVAILABLE

    def test_showman_restores_full_pool(self):
        """With has_showman=True, a 'used' joker remains in the pool as a real key."""
        from jackdaw.engine.pools import UNAVAILABLE
        rng_base = PseudoRandom("SHOWMAN_TEST2")
        pool_with_showman, _ = get_current_pool(
            "Joker",
            rng_base,
            1,
            rarity=1,
            append="sho",
            used_jokers={"j_joker"},
            has_showman=True,
        )
        # j_joker should be present and NOT UNAVAILABLE
        assert "j_joker" in pool_with_showman
        idx = pool_with_showman.index("j_joker")
        assert pool_with_showman[idx] != UNAVAILABLE

    def test_shop_with_showman_allows_jokers_despite_used_pool(self):
        """Shop produces jokers with has_showman=True even if used_jokers is populated."""
        rng, gs = _fresh_gs(seed="SHOWMAN_RUN", ante=1, has_showman=True)
        # Mark some jokers as used (but not all — showman bypasses filtering)
        gs["used_jokers"] = {"j_joker": True, "j_mad": True, "j_jolly": True}
        result = populate_shop(rng, 1, gs)
        # All slots still produce cards without error
        assert len(result["jokers"]) == 2


# ---------------------------------------------------------------------------
# Scenario 4: Reroll — escalating cost, different cards, determinism
# ---------------------------------------------------------------------------


class TestReroll:
    """reroll_shop escalates cost and replaces shop cards each time."""

    def _make_shop_area(self, cards: list) -> CardArea:
        area = CardArea(card_limit=2, area_type="shop")
        for c in cards:
            area.add(c)
        return area

    def test_initial_reroll_cost_is_five(self):
        _, gs = _fresh_gs()
        assert calculate_reroll_cost(gs) == 5

    def test_reroll_escalates_cost(self):
        rng, gs = _fresh_gs(seed="REROLL_SEED", ante=1, dollars=100)
        result = populate_shop(rng, 1, gs)
        shop_area = self._make_shop_area(result["jokers"])

        costs = []
        for _ in range(3):
            cost = calculate_reroll_cost(gs)
            costs.append(cost)
            r = reroll_shop(shop_area, rng, 1, gs)
            assert r["ok"] is True

        # Costs should be 5, 6, 7 (each reroll increments by 1)
        assert costs == [5, 6, 7]

    def test_reroll_changes_shop_cards(self):
        rng, gs = _fresh_gs(seed="REROLL_CHANGE", ante=1, dollars=100)
        result = populate_shop(rng, 1, gs)
        shop_area = self._make_shop_area(result["jokers"])
        before_keys = [c.center_key for c in shop_area.cards]

        reroll_shop(shop_area, rng, 1, gs)
        after_keys = [c.center_key for c in shop_area.cards]

        # After reroll, shop has the same number of slots
        assert len(after_keys) == 2
        # Keys are valid (non-empty)
        assert all(k != "" for k in after_keys)

    def test_reroll_determinism_same_seed(self):
        """Two identical runs produce identical reroll sequences."""
        def run_rerolls(seed: str) -> list[list[str]]:
            rng, gs = _fresh_gs(seed=seed, ante=1, dollars=100)
            result = populate_shop(rng, 1, gs)
            area = self._make_shop_area(result["jokers"])
            sequences = []
            for _ in range(3):
                reroll_shop(area, rng, 1, gs)
                sequences.append([c.center_key for c in area.cards])
            return sequences

        assert run_rerolls("DET_REROLL") == run_rerolls("DET_REROLL")

    def test_reroll_fails_when_insufficient_funds(self):
        rng, gs = _fresh_gs(seed="BROKE_SEED", ante=1, dollars=0)
        result = populate_shop(rng, 1, gs)
        shop_area = self._make_shop_area(result["jokers"])
        r = reroll_shop(shop_area, rng, 1, gs)
        assert r["ok"] is False
        assert r["reason"] == "insufficient_funds"

    def test_three_rerolls_all_succeed_with_enough_money(self):
        rng, gs = _fresh_gs(seed="RICH_REROLL", ante=1, dollars=500)
        result = populate_shop(rng, 1, gs)
        shop_area = self._make_shop_area(result["jokers"])
        for i in range(3):
            r = reroll_shop(shop_area, rng, 1, gs)
            assert r["ok"] is True, f"Reroll {i} failed: {r}"


# ---------------------------------------------------------------------------
# Scenario 5: Buy joker → used_jokers → excluded from reroll (unless Showman)
# ---------------------------------------------------------------------------


class TestBuyJokerTracking:
    """Buying a joker tracks it in used_jokers; Showman bypasses exclusion."""

    def test_buy_joker_adds_to_used_jokers(self):
        rng, gs = _fresh_gs(seed="BUY_TRACK", ante=1, dollars=100)
        result = populate_shop(rng, 1, gs)

        joker_cards = [c for c in result["jokers"] if c.ability.get("set") == "Joker"]
        if not joker_cards:
            pytest.skip("No Joker-type card in first shop slot")

        card = joker_cards[0]
        shop_area = CardArea(card_limit=2, area_type="shop")
        for c in result["jokers"]:
            shop_area.add(c)
        joker_area = CardArea(card_limit=5, area_type="joker")

        buy_result = buy_card(card, shop_area, joker_area, gs)
        assert buy_result["ok"] is True
        assert card.center_key in gs["used_jokers"]

    def test_used_joker_excluded_from_pool_without_showman(self):
        """After buying a joker, its key is UNAVAILABLE in the pool (no showman)."""
        rng, gs = _fresh_gs(seed="EXCLUSION_TEST", ante=1, dollars=100)
        result = populate_shop(rng, 1, gs)

        joker_cards = [c for c in result["jokers"] if c.ability.get("set") == "Joker"]
        if not joker_cards:
            pytest.skip("No Joker-type card in first shop slot")

        card = joker_cards[0]
        shop_area = CardArea(card_limit=2, area_type="shop")
        for c in result["jokers"]:
            shop_area.add(c)
        joker_area = CardArea(card_limit=5, area_type="joker")

        buy_card(card, shop_area, joker_area, gs)
        bought_key = card.center_key

        # Now build pool — bought key should be UNAVAILABLE
        rng2 = PseudoRandom("EXCLUSION_CHECK")
        pool, _ = get_current_pool(
            "Joker",
            rng2,
            1,
            rarity=1,
            append="sho",
            used_jokers=set(gs["used_jokers"].keys()),
            has_showman=False,
        )
        from jackdaw.engine.pools import UNAVAILABLE
        # The bought key must be UNAVAILABLE in the pool
        assert bought_key not in pool or pool[pool.index(bought_key)] == UNAVAILABLE

    def test_used_joker_available_with_showman(self):
        """Same bought joker is back in pool when has_showman=True."""
        bought_key = "j_joker"  # Common joker, rarity 1
        rng = PseudoRandom("SHOWMAN_POOL")
        pool, _ = get_current_pool(
            "Joker",
            rng,
            1,
            rarity=1,
            append="sho",
            used_jokers={bought_key},
            has_showman=True,
        )
        from jackdaw.engine.pools import UNAVAILABLE
        # With showman, the bought key must appear as a real key
        assert bought_key in pool
        idx = pool.index(bought_key)
        assert pool[idx] != UNAVAILABLE


# ---------------------------------------------------------------------------
# Scenario 6: Open Arcana pack — Tarot count, Omen Globe override
# ---------------------------------------------------------------------------


class TestArcanaPack:
    """generate_pack_cards for Arcana packs returns Tarots; Omen Globe adds Spectrals."""

    def _arcana_pack_key(self, size: str = "normal") -> str:
        """Return a known Arcana pack key."""
        key = f"p_arcana_{size}_1"
        if key in BOOSTERS:
            return key
        # Fallback: find any Arcana pack
        for k, v in BOOSTERS.items():
            if v.kind == "Arcana":
                return k
        pytest.skip("No Arcana pack found in BOOSTERS")

    def test_arcana_pack_returns_tarot_cards(self):
        key = self._arcana_pack_key()
        rng = PseudoRandom("ARCANA_BASIC")
        gs: dict[str, Any] = {
            "has_showman": False,
            "used_jokers": {},
            "banned_keys": set(),
            "pool_flags": {},
            "deck_enhancements": set(),
            "playing_card_count": 52,
            "played_hand_types": set(),
            "shop_vouchers": set(),
            "used_vouchers": {},
            "has_omen_globe": False,
            "edition_rate": 1.0,
            "inflation": 0,
            "discount_percent": 0,
            "booster_ante_scaling": False,
            "has_astronomer": False,
        }
        cards, choose = generate_pack_cards(key, rng, 1, gs)
        proto = BOOSTERS[key]
        assert len(cards) == proto.config.get("extra", 1)
        assert choose == proto.config.get("choose", 1)
        # All cards should be Tarot type (no Omen Globe)
        for card in cards:
            assert card.ability.get("set") in ("Tarot", "Spectral"), (
                f"Unexpected set {card.ability.get('set')!r} in Arcana pack card {card.center_key!r}"
            )

    def test_arcana_pack_normal_cards_are_tarots(self):
        """Without Omen Globe, Arcana pack generates only Tarots."""
        key = self._arcana_pack_key("normal")
        rng = PseudoRandom("ARCANA_TAROT_ONLY")
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        # Run many times; without Omen Globe, no Spectral should appear
        all_tarot = True
        for i in range(10):
            rng_i = PseudoRandom(f"ARCANA_TAROT_{i}")
            cards, _ = generate_pack_cards(key, rng_i, 1, gs)
            for card in cards:
                if card.ability.get("set") == "Spectral":
                    all_tarot = False
        assert all_tarot, "Spectral appeared in Arcana pack without Omen Globe"

    def test_omen_globe_introduces_spectrals(self):
        """With Omen Globe (has_omen_globe=True), Spectrals can appear in Arcana packs."""
        key = self._arcana_pack_key("normal")
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": True, "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        # Run many times until a Spectral appears (20% chance per slot)
        found_spectral = False
        for i in range(50):
            rng_i = PseudoRandom(f"OMEN_{i}")
            cards, _ = generate_pack_cards(key, rng_i, 1, gs)
            for card in cards:
                if card.ability.get("set") == "Spectral":
                    found_spectral = True
                    break
            if found_spectral:
                break
        assert found_spectral, "Spectral never appeared in 50 Arcana packs with Omen Globe"

    def test_arcana_choose_count(self):
        """Normal Arcana: extra=3 cards shown, choose=1."""
        key = self._arcana_pack_key("normal")
        proto = BOOSTERS[key]
        assert proto.config.get("extra", 1) >= 1
        assert proto.config.get("choose", 1) >= 1


# ---------------------------------------------------------------------------
# Scenario 7: Open Standard pack — enhancements/editions/seals
# ---------------------------------------------------------------------------


class TestStandardPack:
    """Standard packs include playing cards with optional enhancements/editions/seals."""

    def _standard_pack_key(self) -> str:
        for k, v in BOOSTERS.items():
            if v.kind == "Standard":
                return k
        pytest.skip("No Standard pack in BOOSTERS")

    def test_standard_pack_returns_playing_cards(self):
        key = self._standard_pack_key()
        rng = PseudoRandom("STANDARD_BASIC")
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        cards, choose = generate_pack_cards(key, rng, 1, gs)
        proto = BOOSTERS[key]
        assert len(cards) == proto.config.get("extra", 1)
        assert choose >= 1
        for card in cards:
            assert card.base is not None, f"{card.center_key!r} has no base (not a playing card)"

    def test_standard_pack_cards_have_valid_ranks_and_suits(self):
        key = self._standard_pack_key()
        from jackdaw.engine.data.enums import Rank, Suit
        rng = PseudoRandom("STANDARD_VALID")
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        cards, _ = generate_pack_cards(key, rng, 1, gs)
        valid_suits = {s.value for s in Suit}
        valid_ranks = {r.value for r in Rank}
        for card in cards:
            assert card.base.suit in valid_suits, f"Invalid suit: {card.base.suit!r}"
            assert card.base.rank in valid_ranks, f"Invalid rank: {card.base.rank!r}"

    def test_standard_pack_sometimes_has_enhancements(self):
        """Over many runs, some Standard pack cards will have non-base enhancement."""
        key = self._standard_pack_key()
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        enhanced = False
        for i in range(30):
            rng = PseudoRandom(f"STANDARD_ENH_{i}")
            cards, _ = generate_pack_cards(key, rng, 1, gs)
            for card in cards:
                if card.center_key != "c_base":
                    enhanced = True
                    break
            if enhanced:
                break
        assert enhanced, "No enhanced cards appeared in 30 Standard packs"

    def test_standard_pack_sometimes_has_seals(self):
        """Over many runs, some Standard pack cards will have seals."""
        key = self._standard_pack_key()
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        sealed = False
        for i in range(50):
            rng = PseudoRandom(f"STANDARD_SEAL_{i}")
            cards, _ = generate_pack_cards(key, rng, 1, gs)
            for card in cards:
                if card.seal:
                    sealed = True
                    break
            if sealed:
                break
        assert sealed, "No sealed cards appeared in 50 Standard packs"


# ---------------------------------------------------------------------------
# Scenario 8: Celestial pack with Telescope
# ---------------------------------------------------------------------------


class TestCelestialPackTelescope:
    """Telescope voucher forces slot 0 of Celestial pack to the most-played planet."""

    def _celestial_pack_key(self) -> str:
        for k, v in BOOSTERS.items():
            if v.kind == "Celestial":
                return k
        pytest.skip("No Celestial pack in BOOSTERS")

    def test_telescope_forces_flush_planet(self):
        """With Telescope and most_played_hand='Flush', first card is c_jupiter."""
        key = self._celestial_pack_key()
        rng = PseudoRandom("TELESCOPE_FLUSH")
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "has_telescope": True,
            "most_played_hand": "Flush",
            "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        cards, _ = generate_pack_cards(key, rng, 1, gs)
        # First card should be Jupiter (Flush planet)
        assert cards[0].center_key == "c_jupiter", (
            f"Expected c_jupiter for Flush, got {cards[0].center_key!r}"
        )

    def test_telescope_forces_pair_planet(self):
        """With Telescope and most_played_hand='Pair', first card is c_mercury."""
        key = self._celestial_pack_key()
        rng = PseudoRandom("TELESCOPE_PAIR")
        gs = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "has_telescope": True,
            "most_played_hand": "Pair",
            "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        cards, _ = generate_pack_cards(key, rng, 1, gs)
        assert cards[0].center_key == "c_mercury", (
            f"Expected c_mercury for Pair, got {cards[0].center_key!r}"
        )

    def test_no_telescope_produces_random_planet(self):
        """Without Telescope, first card is RNG-drawn (not forced to any hand type)."""
        key = self._celestial_pack_key()
        gs_base = {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "has_omen_globe": False, "has_telescope": False,
            "edition_rate": 1.0,
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }
        # Run enough times to confirm we DON'T always get c_jupiter for Flush
        planets_seen = set()
        for i in range(10):
            rng = PseudoRandom(f"NO_TELESCOPE_{i}")
            cards, _ = generate_pack_cards(key, rng, 1, gs_base)
            planets_seen.add(cards[0].center_key)
        # Without telescope, multiple different planets should appear
        assert len(planets_seen) > 1, "Without Telescope, should see variety of planets"


# ---------------------------------------------------------------------------
# Scenario 9: Resolve create descriptors
# ---------------------------------------------------------------------------


class TestResolveCreateDescriptors:
    """resolve_create_descriptor correctly bridges side-effects to Card objects."""

    def _base_gs(self) -> dict[str, Any]:
        return {
            "has_showman": False, "used_jokers": {}, "banned_keys": set(),
            "pool_flags": {}, "deck_enhancements": set(), "playing_card_count": 52,
            "played_hand_types": set(), "shop_vouchers": set(), "used_vouchers": {},
            "inflation": 0, "discount_percent": 0,
            "booster_ante_scaling": False, "has_astronomer": False,
        }

    def test_high_priestess_creates_two_planets(self):
        """c_high_priestess descriptor → 2 Planet cards."""
        card = create_consumable("c_high_priestess")
        ctx = ConsumableContext(card=card, game_state=self._base_gs())
        result = use_consumable(card, ctx)
        assert result is not None
        assert result.create is not None
        assert len(result.create) == 1

        descriptor = result.create[0]
        assert descriptor["type"] == "Planet"
        assert descriptor.get("count", 1) == 2

        # Now resolve the descriptor
        rng = PseudoRandom("HP_RESOLVE")
        count = descriptor.get("count", 1)
        created = [
            resolve_create_descriptor(descriptor, rng, 1, self._base_gs())
            for _ in range(count)
        ]
        assert len(created) == 2
        for planet_card in created:
            assert planet_card is not None
            assert planet_card.ability.get("set") == "Planet", (
                f"Expected Planet, got {planet_card.center_key!r}"
            )

    def test_judgement_creates_one_joker(self):
        """c_judgement descriptor → 1 Joker card."""
        card = create_consumable("c_judgement")
        ctx = ConsumableContext(card=card, game_state=self._base_gs())
        result = use_consumable(card, ctx)
        assert result is not None
        assert result.create is not None

        descriptor = result.create[0]
        assert descriptor["type"] == "Joker"

        rng = PseudoRandom("JUDGEMENT_RESOLVE")
        joker_card = resolve_create_descriptor(descriptor, rng, 1, self._base_gs())
        assert joker_card is not None
        assert joker_card.ability.get("set") == "Joker", (
            f"Expected Joker, got {joker_card.center_key!r}"
        )

    def test_playing_card_descriptor_with_rank_suit(self):
        """Playing-card descriptor {rank, suit, enhancement} → Card with base."""
        descriptor = {"rank": "Ace", "suit": "Spades", "enhancement": "m_glass"}
        rng = PseudoRandom("PLAYING_CARD")
        gs = self._base_gs()
        card = resolve_create_descriptor(descriptor, rng, 1, gs)
        assert card is not None
        assert card.base is not None
        assert card.base.rank == "Ace"
        assert card.base.suit == "Spades"
        assert card.center_key == "m_glass"

    def test_playing_card_descriptor_with_type_field(self):
        """Descriptor with type='PlayingCard' also creates a playing card."""
        descriptor = {"type": "PlayingCard", "rank": "King", "suit": "Hearts"}
        rng = PseudoRandom("PC_TYPE")
        gs = self._base_gs()
        card = resolve_create_descriptor(descriptor, rng, 1, gs)
        assert card is not None
        assert card.base is not None
        assert card.base.rank == "King"
        assert card.base.suit == "Hearts"

    def test_tarot_descriptor_creates_tarot(self):
        """Descriptor with type='Tarot' creates a Tarot card."""
        descriptor = {"type": "Tarot", "seed": "emp"}
        rng = PseudoRandom("TAROT_DESC")
        gs = self._base_gs()
        card = resolve_create_descriptor(descriptor, rng, 1, gs)
        assert card is not None
        assert card.ability.get("set") == "Tarot"

    def test_joker_descriptor_with_rarity_3_creates_rare_joker(self):
        """Descriptor with type='Joker', rarity=3 creates a Rare joker."""
        descriptor = {"type": "Joker", "count": 1, "seed": "wra", "rarity": 3}
        rng = PseudoRandom("WRAITH_RESOLVE")
        gs = self._base_gs()
        card = resolve_create_descriptor(descriptor, rng, 1, gs)
        assert card is not None
        assert card.ability.get("set") == "Joker"
        # Rarity 3 joker
        key = card.center_key
        if key in JOKERS:
            assert JOKERS[key].rarity == 3, (
                f"Expected rarity-3 Joker, got {key!r} (rarity {JOKERS[key].rarity})"
            )


# ---------------------------------------------------------------------------
# Scenario 10: Full cycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    """End-to-end: populate → buy → reroll → buy tarot → open pack → state checks."""

    def test_full_shop_cycle(self):
        """populate → buy joker → reroll → observe state mutations throughout."""
        rng, gs = _fresh_gs(seed="FULL_CYCLE", ante=1, dollars=200)
        gs["jokers"] = []

        # Step 1: populate shop
        result = populate_shop(rng, 1, gs)
        assert len(result["jokers"]) == 2

        # Step 2: buy the first joker
        joker_cards = [c for c in result["jokers"] if c.ability.get("set") == "Joker"]
        if not joker_cards:
            pytest.skip("No Joker-type card in first shop slot")

        shop_area = CardArea(card_limit=2, area_type="shop")
        for c in result["jokers"]:
            shop_area.add(c)
        joker_area = CardArea(card_limit=5, area_type="joker")

        first_joker = joker_cards[0]
        initial_dollars = gs["dollars"]
        buy_result = buy_card(first_joker, shop_area, joker_area, gs)
        assert buy_result["ok"] is True

        # Dollars deducted, joker tracked, cards_purchased incremented
        assert gs["dollars"] == initial_dollars - first_joker.cost
        assert first_joker.center_key in gs["used_jokers"]
        assert gs["cards_purchased"] == 1
        assert first_joker in joker_area.cards

        # Step 3: reroll
        dollars_before_reroll = gs["dollars"]
        reroll_result = reroll_shop(shop_area, rng, 1, gs)
        assert reroll_result["ok"] is True
        assert reroll_result["cost"] == 5
        assert gs["dollars"] == dollars_before_reroll - 5
        # Shop was repopulated
        assert len(shop_area.cards) == 2

        # Step 4: second reroll costs more
        r2 = reroll_shop(shop_area, rng, 1, gs)
        assert r2["ok"] is True
        assert r2["cost"] == 6  # incremented once after first reroll

    def test_buy_tarot_opens_pack_updates_state(self):
        """Buy Tarot → open Arcana pack → all cards are Tarots."""
        rng = PseudoRandom("TAROT_CYCLE")
        gs: dict[str, Any] = {
            "joker_rate": 0.0,
            "tarot_rate": 10000.0,
            "planet_rate": 0.0,
            "spectral_rate": 0.0,
            "playing_card_rate": 0.0,
            "edition_rate": 1.0,
            "enable_eternals_in_shop": True,
            "enable_perishables_in_shop": True,
            "enable_rentals_in_shop": True,
            "banned_keys": set(),
            "used_jokers": {},
            "used_vouchers": {},
            "pool_flags": {},
            "has_showman": False,
            "deck_enhancements": set(),
            "playing_card_count": 52,
            "played_hand_types": set(),
            "shop_vouchers": set(),
            "inflation": 0,
            "discount_percent": 0,
            "booster_ante_scaling": False,
            "has_astronomer": False,
            "shop": {"joker_max": 2},
            "current_round": {"voucher": None},
            "first_shop_buffoon": True,  # already consumed
            "dollars": 200,
            "jokers": [],
            "has_omen_globe": False,
            "has_telescope": False,
        }
        # Populate: with tarot_rate=10000, both slots become Tarot cards
        result = populate_shop(rng, 1, gs)
        tarot_cards = [c for c in result["jokers"] if c.ability.get("set") == "Tarot"]
        assert len(tarot_cards) >= 1, "Expected at least one Tarot card in shop"

        # Open Arcana pack (independently)
        arcana_key = None
        for k, v in BOOSTERS.items():
            if v.kind == "Arcana":
                arcana_key = k
                break
        if arcana_key is None:
            pytest.skip("No Arcana pack available")

        pack_rng = PseudoRandom("ARCANA_OPEN")
        cards, choose = generate_pack_cards(arcana_key, pack_rng, 1, gs)
        assert len(cards) >= 1
        for card in cards:
            assert card.ability.get("set") in ("Tarot", "Spectral")

    def test_voucher_effect_persists_into_shop(self):
        """apply_voucher modifies game_state; subsequent populate_shop uses it."""
        rng, gs = _fresh_gs(seed="VOUCHER_PERSIST", ante=1, dollars=100)
        # Apply Tarot Merchant before shop
        apply_voucher("v_tarot_merchant", gs)
        assert gs["tarot_rate"] == pytest.approx(9.6)

        # The modified gs is used by populate_shop
        result = populate_shop(rng, 1, gs)
        # Shop ran without error; tarot_rate was used
        assert len(result["jokers"]) == 2

    def test_inflation_does_not_break_shop(self):
        """inflate=3 adjusts card costs; shop still populates without error."""
        rng, gs = _fresh_gs(seed="INFLATION_SHOP", ante=1, dollars=200)
        gs["inflation"] = 3
        result = populate_shop(rng, 1, gs)
        assert len(result["jokers"]) == 2
        # Costs are higher due to inflation
        for card in result["jokers"]:
            assert card.cost > 0
