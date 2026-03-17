"""Integration tests for the complete run initialization chain.

Validates that initialize_run → start_round produces a fully coherent
game_state with correct deck, parameters, RNG-driven selections, and
per-round reset behaviour across multiple deck/stake combinations.

All tests use seed ``'TESTSEED'`` for reproducibility.
"""

from __future__ import annotations

from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.data.prototypes import BLINDS, TAGS, VOUCHERS
from jackdaw.engine.run_init import initialize_run, start_round


# ---------------------------------------------------------------------------
# 1. Red Deck, stake 1 — baseline standard run
# ---------------------------------------------------------------------------


class TestRedDeckStake1:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_red", 1, self.SEED)

    def test_deck_size_52(self):
        assert len(self._gs()["deck"]) == 52

    def test_discards_4(self):
        assert self._gs()["starting_params"]["discards"] == 4  # 3 + 1

    def test_hands_4(self):
        assert self._gs()["starting_params"]["hands"] == 4

    def test_dollars_4(self):
        assert self._gs()["dollars"] == 4

    def test_deck_is_shuffled(self):
        """Verify deck order matches RNG-determined shuffle — not sorted."""
        gs = self._gs()
        first5 = [(c.base.rank.value, c.base.suit.value) for c in gs["deck"][:5]]
        assert first5 == [
            ("5", "Diamonds"),
            ("9", "Diamonds"),
            ("2", "Diamonds"),
            ("Jack", "Hearts"),
            ("4", "Hearts"),
        ]

    def test_boss_blind_selected(self):
        gs = self._gs()
        boss = gs["round_resets"]["blind_choices"]["Boss"]
        assert boss == "bl_head"
        assert boss in BLINDS

    def test_tags_generated(self):
        gs = self._gs()
        tags = gs["round_resets"]["blind_tags"]
        assert tags == {"Small": "tag_economy", "Big": "tag_investment"}
        assert tags["Small"] in TAGS
        assert tags["Big"] in TAGS

    def test_voucher_selected(self):
        gs = self._gs()
        voucher = gs["current_round"]["voucher"]
        assert voucher == "v_crystal_ball"
        assert voucher in VOUCHERS

    def test_no_stake_modifiers(self):
        gs = self._gs()
        assert gs.get("modifiers", {}) == {}

    def test_targeting_cards_populated(self):
        gs = self._gs()
        cr = gs["current_round"]
        assert "suit" in cr["idol_card"] and "rank" in cr["idol_card"]
        assert "rank" in cr["mail_card"]
        assert "suit" in cr["ancient_card"]
        assert "suit" in cr["castle_card"]


# ---------------------------------------------------------------------------
# 2. Abandoned Deck, stake 5 — no face cards + stake modifiers
# ---------------------------------------------------------------------------


class TestAbandonedDeckStake5:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_abandoned", 5, self.SEED)

    def test_deck_size_40(self):
        """Abandoned Deck removes J/Q/K → 52 - 12 = 40 cards."""
        assert len(self._gs()["deck"]) == 40

    def test_no_face_cards_in_deck(self):
        gs = self._gs()
        face_ranks = {"Jack", "Queen", "King"}
        for card in gs["deck"]:
            assert card.base.rank.value not in face_ranks

    def test_discards_2(self):
        """Base 3 - 1 (stake 5 Blue) = 2."""
        assert self._gs()["starting_params"]["discards"] == 2

    def test_enable_eternals(self):
        assert self._gs()["modifiers"]["enable_eternals_in_shop"] is True

    def test_scaling_2(self):
        """Stake 5 inherits scaling=2 from stake 3 (Green)."""
        assert self._gs()["modifiers"]["scaling"] == 2

    def test_no_blind_reward_small(self):
        """Stake 2+ removes Small Blind reward."""
        assert self._gs()["modifiers"]["no_blind_reward"]["Small"] is True


# ---------------------------------------------------------------------------
# 3. Zodiac Deck — 3 starting vouchers
# ---------------------------------------------------------------------------


class TestZodiacDeck:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_zodiac", 1, self.SEED)

    def test_three_vouchers_applied(self):
        uv = self._gs()["used_vouchers"]
        assert uv.get("v_tarot_merchant") is True
        assert uv.get("v_planet_merchant") is True
        assert uv.get("v_overstock_norm") is True

    def test_tarot_rate(self):
        assert self._gs()["tarot_rate"] == 9.6

    def test_planet_rate(self):
        assert self._gs()["planet_rate"] == 9.6

    def test_shop_joker_max(self):
        assert self._gs()["shop"]["joker_max"] == 3


# ---------------------------------------------------------------------------
# 4. Magic Deck — Crystal Ball voucher + 2 Fools
# ---------------------------------------------------------------------------


class TestMagicDeck:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_magic", 1, self.SEED)

    def test_crystal_ball_applied(self):
        assert self._gs()["used_vouchers"].get("v_crystal_ball") is True

    def test_consumable_slots_3(self):
        assert self._gs()["consumable_slots"] == 3

    def test_two_fools(self):
        assert self._gs()["starting_consumables"] == ["c_fool", "c_fool"]


# ---------------------------------------------------------------------------
# 5. Plasma Deck, stake 6 — ante scaling + Purple scaling
# ---------------------------------------------------------------------------


class TestPlasmaDeckStake6:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_plasma", 6, self.SEED)

    def test_ante_scaling_2(self):
        assert self._gs()["starting_params"]["ante_scaling"] == 2

    def test_scaling_3(self):
        """Purple Stake (6) overrides Green (3) scaling."""
        assert self._gs()["modifiers"]["scaling"] == 3

    def test_all_lower_stake_modifiers_present(self):
        gs = self._gs()
        mods = gs["modifiers"]
        assert mods["no_blind_reward"]["Small"] is True  # stake 2
        assert mods["enable_eternals_in_shop"] is True     # stake 4
        assert gs["starting_params"]["discards"] == 2       # stake 5 (3-1)


# ---------------------------------------------------------------------------
# 6. Full round cycle: initialize_run → start_round
# ---------------------------------------------------------------------------


class TestFullRoundCycle:
    SEED = "TESTSEED"

    def _gs(self):
        gs = initialize_run("b_red", 1, self.SEED)
        start_round(gs)
        return gs

    def test_hands_left_matches_round_resets(self):
        gs = self._gs()
        assert gs["current_round"]["hands_left"] == gs["round_resets"]["hands"]

    def test_discards_left_matches_round_resets(self):
        gs = self._gs()
        assert gs["current_round"]["discards_left"] == gs["round_resets"]["discards"]

    def test_targeting_cards_valid_after_start_round(self):
        gs = self._gs()
        cr = gs["current_round"]
        valid_suits = {s.value for s in Suit}
        valid_ranks = {r.value for r in Rank}
        assert cr["idol_card"]["suit"] in valid_suits
        assert cr["idol_card"]["rank"] in valid_ranks
        assert cr["mail_card"]["rank"] in valid_ranks
        assert cr["ancient_card"]["suit"] in valid_suits
        assert cr["castle_card"]["suit"] in valid_suits

    def test_hand_played_this_round_all_zero(self):
        gs = self._gs()
        hl = gs["hand_levels"]
        for ht in HandType:
            assert hl[ht].played_this_round == 0, f"{ht} not zeroed"

    def test_hands_played_zero(self):
        gs = self._gs()
        assert gs["current_round"]["hands_played"] == 0

    def test_discards_used_zero(self):
        gs = self._gs()
        assert gs["current_round"]["discards_used"] == 0

    def test_reroll_cost_at_base(self):
        gs = self._gs()
        assert gs["current_round"]["reroll_cost"] == gs["base_reroll_cost"]


# ---------------------------------------------------------------------------
# 7. RNG consumption chain — determinism across full init
# ---------------------------------------------------------------------------


class TestRNGConsumptionChain:
    """Same seed always produces identical game state."""

    SEED = "TESTSEED"

    def test_full_determinism(self):
        gs1 = initialize_run("b_red", 1, self.SEED)
        gs2 = initialize_run("b_red", 1, self.SEED)

        # Boss, tags, voucher
        assert gs1["round_resets"]["blind_choices"] == gs2["round_resets"]["blind_choices"]
        assert gs1["round_resets"]["blind_tags"] == gs2["round_resets"]["blind_tags"]
        assert gs1["current_round"]["voucher"] == gs2["current_round"]["voucher"]

        # Deck order (first 10 cards)
        for i in range(10):
            assert (
                gs1["deck"][i].base.rank == gs2["deck"][i].base.rank
                and gs1["deck"][i].base.suit == gs2["deck"][i].base.suit
            ), f"deck[{i}] differs"

        # Targeting cards
        assert gs1["current_round"]["idol_card"] == gs2["current_round"]["idol_card"]
        assert gs1["current_round"]["mail_card"] == gs2["current_round"]["mail_card"]
        assert gs1["current_round"]["ancient_card"] == gs2["current_round"]["ancient_card"]
        assert gs1["current_round"]["castle_card"] == gs2["current_round"]["castle_card"]

    def test_different_seed_differs(self):
        gs_a = initialize_run("b_red", 1, "SEED_A")
        gs_b = initialize_run("b_red", 1, "SEED_B")
        # At least one of boss/tags/voucher should differ
        assert (
            gs_a["round_resets"]["blind_choices"]["Boss"]
            != gs_b["round_resets"]["blind_choices"]["Boss"]
            or gs_a["round_resets"]["blind_tags"] != gs_b["round_resets"]["blind_tags"]
            or gs_a["current_round"]["voucher"] != gs_b["current_round"]["voucher"]
        )

    def test_rng_order_boss_then_voucher_then_tags(self):
        """Verify the call order: boss → voucher → small tag → big tag.

        If we change the order, the deterministic values would shift.
        This test locks in the known-good values from TESTSEED.
        """
        gs = initialize_run("b_red", 1, self.SEED)
        assert gs["round_resets"]["blind_choices"]["Boss"] == "bl_head"
        assert gs["current_round"]["voucher"] == "v_crystal_ball"
        assert gs["round_resets"]["blind_tags"]["Small"] == "tag_economy"
        assert gs["round_resets"]["blind_tags"]["Big"] == "tag_investment"


# ---------------------------------------------------------------------------
# Cross-deck: Erratic Deck
# ---------------------------------------------------------------------------


class TestErraticDeck:
    def test_52_cards_but_randomized(self):
        gs = initialize_run("b_erratic", 1, "TESTSEED")
        assert len(gs["deck"]) == 52

    def test_deck_has_randomized_ranks(self):
        """Erratic Deck randomizes ranks — we should see duplicate ranks."""
        gs = initialize_run("b_erratic", 1, "TESTSEED")
        from collections import Counter

        ranks = Counter(c.base.rank.value for c in gs["deck"])
        # Standard deck has exactly 4 of each rank; erratic should have variance
        assert max(ranks.values()) > 4, "Expected some duplicated ranks from erratic"


# ---------------------------------------------------------------------------
# Cross-deck: Checkered Deck
# ---------------------------------------------------------------------------


class TestCheckeredDeck:
    def test_only_two_suits(self):
        """Checkered Deck: Clubs→Spades, Diamonds→Hearts → only Spades+Hearts."""
        gs = initialize_run("b_checkered", 1, "TESTSEED")
        suits = {c.base.suit.value for c in gs["deck"]}
        assert suits == {"Spades", "Hearts"}
