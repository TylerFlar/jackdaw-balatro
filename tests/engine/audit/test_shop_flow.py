"""Cash-out and shop fidelity (G54, G120, G121)."""

from __future__ import annotations

from jackdaw.engine.actions import BuyCard, CashOut, NextRound, OpenBooster, RedeemVoucher
from jackdaw.engine.card_factory import create_consumable
from jackdaw.engine.game import step

from ._helpers import new_run as new_challenge_run
from ._joker_helpers import give_joker, new_run, select_blind, to_shop, win_round


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


class TestPacks:
    """Card:open sets ``G.GAME.pack_size`` to the pack's size (card.lua:1691-1705),
    and a Standard pack's cards are made by Card:init, which stamps
    ``hands_played_at_create`` (card.lua:337)."""

    def test_opening_sets_the_pack_size(self):
        gs = to_shop(new_run("AUDIT_PACK_SIZE"))
        gs["dollars"] = 100
        pack = gs["shop_boosters"][0]
        step(gs, OpenBooster(card_index=0))
        assert gs["pack_size"] == pack.ability["extra"]

    def test_standard_pack_cards_record_hands_played(self):
        from jackdaw.engine.packs import generate_pack_cards

        gs = to_shop(new_run("AUDIT_PACK_STD"))
        cards, _ = generate_pack_cards("p_standard_normal_1", gs["rng"], 1, gs)
        assert all(c.ability["hands_played_at_create"] == gs["hands_played"] for c in cards)


class TestVouchers:
    """Card:redeem's card dissolves, taking its used_jokers mark (card.lua:4741-4748);
    the hand and discard vouchers change the current round's counts too
    (ease_hands_played / ease_discard, card.lua:1931-1962)."""

    def _redeem(self, gs, key):
        from jackdaw.engine.card_factory import create_voucher

        voucher = create_voucher(key, hands_played=gs["hands_played"])
        voucher.set_cost()
        gs["shop_vouchers"] = [voucher]
        gs["used_jokers"][key] = True
        gs["dollars"] = 100
        step(gs, RedeemVoucher(card_index=0))

    def test_redeeming_clears_the_mark(self):
        gs = to_shop(new_run("AUDIT_REDEEM_MARK"))
        self._redeem(gs, "v_hone")
        assert not gs["used_jokers"].get("v_hone")

    def test_wasteful_adds_a_discard_now(self):
        gs = to_shop(new_run("AUDIT_WASTEFUL"))
        before = gs["current_round"]["discards_left"]
        self._redeem(gs, "v_wasteful")
        assert gs["current_round"]["discards_left"] == before + 1

    def test_hieroglyph_takes_a_hand_now(self):
        gs = to_shop(new_run("AUDIT_HIEROGLYPH"))
        before = gs["current_round"]["hands_left"]
        self._redeem(gs, "v_hieroglyph")
        assert gs["current_round"]["hands_left"] == before - 1


class TestCouponedFlag:
    """A tag's coupon stays on the card (``ability.couponed``); set_cost zeroes the
    price only in the shop's joker and booster areas (card.lua:383). Clearance
    Sale's pass covers every card instance, playing cards included (card.lua:1917-1923)."""

    def test_clearance_sale_keeps_the_flag_and_reprices_the_deck(self):
        gs = to_shop(new_run("AUDIT_COUPON_FLAG"))
        owned = give_joker(gs, "j_joker")
        owned.ability["couponed"] = True
        owned.cost = 0  # bought for free under a coupon
        foil = gs["deck"][0]
        foil.set_edition({"foil": True})
        TestVouchers()._redeem(gs, "v_clearance_sale")
        assert owned.ability["couponed"] is True
        # floor((base 2 + 0.5) * 75%), and a foil playing card floor((1 + 2 + 0.5) * 75%).
        assert owned.cost == 1
        assert foil.cost == 2
