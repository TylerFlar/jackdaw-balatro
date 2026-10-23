"""Cash-out and shop fidelity (G54, G120, G121)."""

from __future__ import annotations

from jackdaw.engine.actions import BuyCard, CashOut, NextRound
from jackdaw.engine.card_factory import create_consumable
from jackdaw.engine.game import step

from ._helpers import new_run as new_challenge_run
from ._joker_helpers import new_run, select_blind, to_shop, win_round


class TestCashOutRefillsHandsAndDiscards:
    """cash_out's event sets the next round's hands and discards before the
    shop opens (button_callbacks.lua:2929-2930), not at the next blind."""

    def test_the_shop_shows_full_hands_and_discards(self):
        gs = select_blind(new_run("AUDIT_CASHOUT_HANDS"))
        win_round(gs)
        assert gs["current_round"]["hands_left"] < gs["round_resets"]["hands"]
        step(gs, CashOut())
        assert gs["current_round"]["hands_left"] == gs["round_resets"]["hands"]
        assert gs["current_round"]["discards_left"] == gs["round_resets"]["discards"]


class TestShopCards:
    """Every shop card goes through Card:set_ability, which stamps
    ``hands_played_at_create`` (card.lua:337) and marks its key in
    ``used_jokers`` while the card exists (card.lua:349-354, 4741-4748)."""

    def test_vouchers_and_packs_record_hands_played(self):
        gs = to_shop(new_run("AUDIT_SHOP_HPAC"))
        assert gs["hands_played"] > 0
        for card in gs["shop_vouchers"] + gs["shop_boosters"]:
            assert card.ability["hands_played_at_create"] == gs["hands_played"]

    def test_the_shop_voucher_is_marked_until_the_shop_closes(self):
        gs = to_shop(new_run("AUDIT_SHOP_VMARK"))
        voucher = gs["shop_vouchers"][0].center_key
        assert gs["used_jokers"].get(voucher)
        step(gs, NextRound())
        assert not gs["used_jokers"].get(voucher)


class TestJokersPurchased:
    """Only a joker counts in ``current_round.jokers_purchased``
    (button_callbacks.lua:2445-2448)."""

    def test_buying_a_consumable_counts_nothing(self):
        gs = to_shop(new_run("AUDIT_JOKERS_PURCHASED"))
        gs["dollars"] = 100
        gs["shop_cards"][0] = create_consumable("c_fool")
        gs["shop_cards"][0].set_cost()
        step(gs, BuyCard(shop_index=0))
        assert gs["current_round"]["jokers_purchased"] == 0


class TestPlayingCardPrices:
    """Card:init prices every card (card.lua:72): a playing card costs $1 and
    sells for $1. The Inflation challenge's purchase pass reprices every card
    instance, playing cards included (button_callbacks.lua:2455-2462)."""

    def test_the_starting_deck_is_priced(self):
        gs = new_run("AUDIT_PC_PRICE")
        for card in gs["deck"]:
            assert (card.cost, card.sell_cost) == (1, 1)

    def test_inflation_reprices_the_deck(self):
        gs = new_challenge_run(challenge="c_inflation_1")
        to_shop(gs)
        gs["dollars"] = 100
        step(gs, BuyCard(shop_index=0))
        assert gs["inflation"] == 1
        for card in gs["deck"]:
            assert (card.extra_cost, card.cost, card.sell_cost) == (1, 2, 1)
