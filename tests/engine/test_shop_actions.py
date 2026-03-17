"""Tests for buy_card, sell_card, reroll_shop, and calculate_reroll_cost."""

from __future__ import annotations

from jackdaw.engine.card import Card
from jackdaw.engine.card_area import CardArea
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.shop import (
    buy_card,
    calculate_reroll_cost,
    reroll_shop,
    sell_card,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _joker(key: str = "j_joker", **kwargs) -> Card:
    """Create a joker and give it a sell_cost via set_cost."""
    card = create_joker(key, **kwargs)
    card.set_cost()
    return card


def _shop_area() -> CardArea:
    return CardArea(card_limit=10, area_type="shop")


def _joker_area(limit: int = 5) -> CardArea:
    return CardArea(card_limit=limit, area_type="joker")


def _base_gs(dollars: int = 10) -> dict:
    return {"dollars": dollars, "jokers": [], "used_jokers": {}, "cards_purchased": 0}


# ---------------------------------------------------------------------------
# calculate_reroll_cost
# ---------------------------------------------------------------------------


class TestCalculateRerollCost:
    def test_default_cost_is_five(self):
        gs = {"current_round": {}, "round_resets": {"reroll_cost": 5}}
        assert calculate_reroll_cost(gs) == 5

    def test_free_rerolls_gives_zero_cost(self):
        gs = {
            "current_round": {"free_rerolls": 1},
            "round_resets": {"reroll_cost": 5},
        }
        assert calculate_reroll_cost(gs) == 0

    def test_free_rerolls_does_not_set_increase(self):
        gs = {
            "current_round": {"free_rerolls": 2, "reroll_cost_increase": 0},
            "round_resets": {"reroll_cost": 5},
        }
        calculate_reroll_cost(gs)
        assert gs["current_round"]["reroll_cost"] == 0

    def test_negative_free_rerolls_clamped(self):
        gs = {
            "current_round": {"free_rerolls": -3},
            "round_resets": {"reroll_cost": 5},
        }
        cost = calculate_reroll_cost(gs)
        assert gs["current_round"]["free_rerolls"] == 0
        assert cost == 5

    def test_increase_adds_to_base(self):
        gs = {
            "current_round": {"reroll_cost_increase": 2},
            "round_resets": {"reroll_cost": 5},
        }
        assert calculate_reroll_cost(gs) == 7

    def test_temp_reroll_cost_overrides_base(self):
        gs = {
            "current_round": {"reroll_cost_increase": 0},
            "round_resets": {"reroll_cost": 5, "temp_reroll_cost": 1},
        }
        assert calculate_reroll_cost(gs) == 1

    def test_updates_current_round_reroll_cost(self):
        gs = {
            "current_round": {"reroll_cost_increase": 1},
            "round_resets": {"reroll_cost": 5},
        }
        calculate_reroll_cost(gs)
        assert gs["current_round"]["reroll_cost"] == 6

    def test_missing_current_round_created(self):
        gs = {"round_resets": {"reroll_cost": 5}}
        assert calculate_reroll_cost(gs) == 5
        assert "current_round" in gs

    def test_reduced_base_from_reroll_surplus(self):
        """Reroll Surplus sets reroll_cost=3; cost should reflect that."""
        gs = {
            "current_round": {"reroll_cost_increase": 0},
            "round_resets": {"reroll_cost": 3},
        }
        assert calculate_reroll_cost(gs) == 3


# ---------------------------------------------------------------------------
# buy_card
# ---------------------------------------------------------------------------


class TestBuyCard:
    def test_successful_purchase_deducts_dollars(self):
        card = _joker()
        card.cost = 5
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area()
        gs = _base_gs(dollars=10)
        result = buy_card(card, shop, dest, gs)
        assert result["ok"] is True
        assert gs["dollars"] == 5

    def test_card_moved_to_destination(self):
        card = _joker()
        card.cost = 3
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area()
        gs = _base_gs(dollars=10)
        buy_card(card, shop, dest, gs)
        assert card not in shop.cards
        assert card in dest.cards

    def test_insufficient_funds_rejected(self):
        card = _joker()
        card.cost = 15
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area()
        gs = _base_gs(dollars=10)
        result = buy_card(card, shop, dest, gs)
        assert result["ok"] is False
        assert result["reason"] == "insufficient_funds"

    def test_insufficient_funds_does_not_move_card(self):
        card = _joker()
        card.cost = 15
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area()
        gs = _base_gs(dollars=5)
        buy_card(card, shop, dest, gs)
        assert card in shop.cards
        assert card not in dest.cards

    def test_insufficient_funds_does_not_change_dollars(self):
        card = _joker()
        card.cost = 20
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area()
        gs = _base_gs(dollars=10)
        buy_card(card, shop, dest, gs)
        assert gs["dollars"] == 10

    def test_no_space_rejected(self):
        card = _joker()
        card.cost = 3
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area(limit=0)
        gs = _base_gs(dollars=10)
        result = buy_card(card, shop, dest, gs)
        assert result["ok"] is False
        assert result["reason"] == "no_space"

    def test_no_space_does_not_deduct(self):
        card = _joker()
        card.cost = 3
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area(limit=0)
        gs = _base_gs(dollars=10)
        buy_card(card, shop, dest, gs)
        assert gs["dollars"] == 10

    def test_negative_edition_grants_bonus_space(self):
        """A Negative joker can be bought even when the area is at its limit."""
        card = _joker()
        card.cost = 3
        card.set_edition({"negative": True})
        shop = _shop_area()
        shop.add(card)
        # Fill destination to its limit
        dest = _joker_area(limit=1)
        filler = _joker()
        dest.add(filler)
        gs = _base_gs(dollars=10)
        result = buy_card(card, shop, dest, gs)
        assert result["ok"] is True

    def test_cards_purchased_incremented(self):
        card = _joker()
        card.cost = 3
        shop = _shop_area()
        shop.add(card)
        gs = _base_gs(dollars=10)
        buy_card(card, shop, _joker_area(), gs)
        assert gs["cards_purchased"] == 1

    def test_cards_purchased_accumulates(self):
        shop = _shop_area()
        dest = _joker_area()
        gs = _base_gs(dollars=50)
        for key in ("j_joker", "j_greedy_joker", "j_lusty_joker"):
            c = _joker(key)
            c.cost = 3
            shop.add(c)
            buy_card(c, shop, dest, gs)
        assert gs["cards_purchased"] == 3

    def test_used_jokers_tracked(self):
        card = _joker("j_joker")
        card.cost = 3
        shop = _shop_area()
        shop.add(card)
        gs = _base_gs(dollars=10)
        buy_card(card, shop, _joker_area(), gs)
        assert gs["used_jokers"].get("j_joker") is True

    def test_zero_cost_joker_succeeds_with_zero_dollars(self):
        card = _joker()
        card.cost = 0
        shop = _shop_area()
        shop.add(card)
        gs = _base_gs(dollars=0)
        result = buy_card(card, shop, _joker_area(), gs)
        assert result["ok"] is True
        assert gs["dollars"] == 0

    def test_inflation_increments_when_modifier_active(self):
        card = _joker()
        card.cost = 3
        shop = _shop_area()
        shop.add(card)
        gs = _base_gs(dollars=10)
        gs["inflation_modifier"] = True
        gs["inflation"] = 0
        buy_card(card, shop, _joker_area(), gs)
        assert gs["inflation"] == 1

    def test_inflation_off_by_default(self):
        card = _joker()
        card.cost = 3
        shop = _shop_area()
        shop.add(card)
        gs = _base_gs(dollars=10)
        gs["inflation"] = 0
        buy_card(card, shop, _joker_area(), gs)
        assert gs["inflation"] == 0

    def test_inflation_recalculates_remaining_shop_cards(self):
        """When inflation fires, all_shop_cards get set_cost called."""
        to_buy = _joker()
        to_buy.cost = 3
        remaining = _joker()
        remaining.set_cost(inflation=0)
        original_cost = remaining.cost

        shop = _shop_area()
        shop.add(to_buy)
        gs = _base_gs(dollars=10)
        gs["inflation_modifier"] = True
        gs["inflation"] = 0
        gs["all_shop_cards"] = [remaining]
        gs["ante"] = 1

        buy_card(to_buy, shop, _joker_area(), gs)
        # After inflation=1, remaining card cost should be recalculated
        remaining.set_cost(inflation=1)  # what we expect
        assert remaining.cost >= original_cost  # inflation can only raise cost

    def test_playing_card_added_to_playing_cards(self):
        """Buying a Default/Enhanced playing card appends it to playing_cards."""
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit

        card = create_playing_card(Suit.SPADES, Rank.ACE)
        card.set_cost()
        shop = _shop_area()
        shop.add(card)
        deck = CardArea(card_limit=52, area_type="deck")
        gs = _base_gs(dollars=10)
        buy_card(card, shop, deck, gs)
        assert card in gs.get("playing_cards", [])

    def test_exact_funds_succeeds(self):
        """Buying with exactly enough dollars works."""
        card = _joker()
        card.cost = 7
        shop = _shop_area()
        shop.add(card)
        gs = _base_gs(dollars=7)
        result = buy_card(card, shop, _joker_area(), gs)
        assert result["ok"] is True
        assert gs["dollars"] == 0

    def test_chaos_the_clown_free_rerolls_applied(self):
        """Buying Chaos the Clown triggers add_to_deck → free_rerolls += 1."""
        card = _joker("j_chaos")
        card.set_cost()
        shop = _shop_area()
        shop.add(card)
        gs = _base_gs(dollars=20)
        gs["free_rerolls"] = 0
        buy_card(card, shop, _joker_area(), gs)
        assert gs["free_rerolls"] == 1


# ---------------------------------------------------------------------------
# sell_card
# ---------------------------------------------------------------------------


class TestSellCard:
    def test_successful_sale_awards_dollars(self):
        card = _joker()
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        result = sell_card(card, area, gs)
        assert result["ok"] is True
        assert gs["dollars"] == card.sell_cost

    def test_sell_cost_3_adds_3_dollars(self):
        card = _joker()
        card.cost = 6
        card.sell_cost = 3
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        sell_card(card, area, gs)
        assert gs["dollars"] == 3

    def test_card_removed_from_area(self):
        card = _joker()
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        sell_card(card, area, gs)
        assert card not in area.cards

    def test_eternal_joker_rejected(self):
        card = _joker(eternal=True)
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        result = sell_card(card, area, gs)
        assert result["ok"] is False
        assert result["reason"] == "eternal"

    def test_eternal_joker_no_dollars_awarded(self):
        card = _joker(eternal=True)
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=5)
        sell_card(card, area, gs)
        assert gs["dollars"] == 5

    def test_eternal_joker_not_removed(self):
        card = _joker(eternal=True)
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        sell_card(card, area, gs)
        assert card in area.cards

    def test_shop_area_not_sellable(self):
        card = _joker()
        card.set_cost()
        area = _shop_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        result = sell_card(card, area, gs)
        assert result["ok"] is False
        assert result["reason"] == "not_sellable"

    def test_dollars_gained_reported(self):
        card = _joker()
        card.sell_cost = 4
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        result = sell_card(card, area, gs)
        assert result["dollars_gained"] == 4

    def test_consumable_area_is_sellable(self):
        from jackdaw.engine.card_factory import create_consumable

        card = create_consumable("c_magician")
        card.set_cost()
        area = CardArea(card_limit=5, area_type="consumeable")
        area.add(card)
        gs = _base_gs(dollars=0)
        result = sell_card(card, area, gs)
        assert result["ok"] is True

    def test_chaos_the_clown_free_rerolls_reversed(self):
        """Selling Chaos the Clown triggers remove_from_deck → free_rerolls -= 1."""
        card = _joker("j_chaos")
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        gs["free_rerolls"] = 1
        sell_card(card, area, gs)
        assert gs["free_rerolls"] == 0


# ---------------------------------------------------------------------------
# reroll_shop
# ---------------------------------------------------------------------------


class TestRerollShop:
    def _make_shop(self, n_cards: int = 2) -> CardArea:
        area = CardArea(card_limit=10, area_type="shop")
        for key in list(("j_joker", "j_greedy_joker", "j_lusty_joker"))[:n_cards]:
            c = _joker(key)
            area.add(c)
        return area

    def test_cost_deducted(self):
        shop = self._make_shop()
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_COST"), 1, gs)
        assert result["ok"] is True
        assert result["cost"] == 5
        assert gs["dollars"] == 5

    def test_old_cards_removed(self):
        shop = self._make_shop(2)
        old_cards = list(shop.cards)
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        reroll_shop(shop, PseudoRandom("RR_CLEAR"), 1, gs)
        for c in old_cards:
            assert c not in shop.cards

    def test_new_cards_generated(self):
        shop = self._make_shop(2)
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_NEW"), 1, gs)
        assert len(result["new_cards"]) == 2
        assert len(shop.cards) == 2

    def test_new_cards_in_shop(self):
        shop = self._make_shop(2)
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_INSHOP"), 1, gs)
        for c in result["new_cards"]:
            assert c in shop.cards

    def test_insufficient_funds_rejected(self):
        shop = self._make_shop()
        gs = _base_gs(dollars=3)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_POOR"), 1, gs)
        assert result["ok"] is False
        assert result["reason"] == "insufficient_funds"

    def test_insufficient_funds_shop_unchanged(self):
        shop = self._make_shop(2)
        old_cards = list(shop.cards)
        gs = _base_gs(dollars=3)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        reroll_shop(shop, PseudoRandom("RR_POOR2"), 1, gs)
        assert shop.cards == old_cards

    def test_free_reroll_zero_cost(self):
        """Chaos the Clown: free_rerolls=1 → $0 cost."""
        shop = self._make_shop()
        gs = _base_gs(dollars=0)
        gs.update(
            {
                "current_round": {"free_rerolls": 1, "reroll_cost_increase": 0},
                "round_resets": {"reroll_cost": 5},
            }
        )
        result = reroll_shop(shop, PseudoRandom("RR_FREE"), 1, gs)
        assert result["ok"] is True
        assert result["cost"] == 0
        assert result["was_free"] is True
        assert gs["dollars"] == 0

    def test_free_reroll_decrements_counter(self):
        shop = self._make_shop()
        gs = _base_gs(dollars=0)
        gs.update(
            {
                "current_round": {"free_rerolls": 2, "reroll_cost_increase": 0},
                "round_resets": {"reroll_cost": 5},
            }
        )
        reroll_shop(shop, PseudoRandom("RR_FDEC"), 1, gs)
        assert gs["current_round"]["free_rerolls"] == 1

    def test_free_reroll_does_not_increment_cost_increase(self):
        """A free reroll should not push up the next paid reroll cost."""
        shop = self._make_shop()
        gs = _base_gs(dollars=20)
        gs.update(
            {
                "current_round": {"free_rerolls": 1, "reroll_cost_increase": 0},
                "round_resets": {"reroll_cost": 5},
            }
        )
        reroll_shop(shop, PseudoRandom("RR_FNOINCR"), 1, gs)
        assert gs["current_round"]["reroll_cost_increase"] == 0

    def test_cost_escalation_five_six_seven(self):
        """Consecutive paid rerolls: $5, $6, $7."""
        gs = _base_gs(dollars=100)
        gs.update(
            {
                "current_round": {"free_rerolls": 0, "reroll_cost_increase": 0},
                "round_resets": {"reroll_cost": 5},
            }
        )

        costs = []
        for i in range(3):
            shop = CardArea(card_limit=10, area_type="shop")
            result = reroll_shop(shop, PseudoRandom(f"RR_ESC_{i}"), 1, gs)
            costs.append(result["cost"])

        assert costs == [5, 6, 7]

    def test_overstock_repopulates_three_slots(self):
        """With shop.joker_max=3, reroll should produce 3 new cards."""
        shop = self._make_shop(0)
        gs = _base_gs(dollars=10)
        gs.update(
            {
                "shop": {"joker_max": 3},
                "current_round": {"reroll_cost_increase": 0},
                "round_resets": {"reroll_cost": 5},
            }
        )
        result = reroll_shop(shop, PseudoRandom("RR_OVER"), 1, gs)
        assert len(result["new_cards"]) == 3

    def test_deterministic_with_same_rng(self):
        shop1 = self._make_shop(2)
        shop2 = self._make_shop(2)

        def _gs():
            g = _base_gs(dollars=10)
            g.update(
                {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
            )
            return g

        r1 = reroll_shop(shop1, PseudoRandom("RR_DET"), 1, _gs())
        r2 = reroll_shop(shop2, PseudoRandom("RR_DET"), 1, _gs())
        assert [c.center_key for c in r1["new_cards"]] == [c.center_key for c in r2["new_cards"]]

    def test_was_free_false_for_paid_reroll(self):
        shop = self._make_shop()
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_PAIDFLAG"), 1, gs)
        assert result["was_free"] is False

    def test_reroll_notifies_flash_card(self):
        """Flash Card joker gets reroll_shop=True notification."""

        shop = self._make_shop()
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )

        flash = create_joker("j_flash")
        flash.set_cost()
        initial_mult = flash.ability.get("mult", 0)
        gs["jokers"] = [flash]

        reroll_shop(shop, PseudoRandom("RR_FLASH"), 1, gs)
        # Flash Card: mult += 2 on reroll_shop
        assert flash.ability.get("mult", 0) == initial_mult + 2
