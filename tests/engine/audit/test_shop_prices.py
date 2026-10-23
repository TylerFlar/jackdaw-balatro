"""Audit regressions: shop prices change only where Lua calls set_cost (G204).

Lua never reprices per frame: a card's price is recomputed only by an explicit
``Card:set_cost()`` (card.lua:369-385), which Card:init, set_edition, set_seal
and set_rental call, and the tags and all-cards passes that change prices. The
oracle used to reprice the shop after every step, which hid three missing
calls; these tests run the paths without a step around them.
"""

from __future__ import annotations

import math

from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import _all_cards_set_cost_pass, _award_tag, _fire_shop_tags
from jackdaw.engine.shop import create_card_for_shop
from tests.engine.audit._helpers import new_run, play_round

_SURCHARGE = {"foil": 2, "holo": 3, "polychrome": 5, "negative": 5}


def _full_price(gs: dict, base_cost: float, extra_cost: float) -> int:
    """card.lua:375: the price before a coupon zeroes it."""
    return max(1, math.floor((base_cost + extra_cost + 0.5) * (100 - gs["discount_percent"]) / 100))


def _shop(seed: str) -> dict:
    gs = play_round(new_run(seed=seed))
    gs["discount_percent"] = 25  # a Clearance Sale, so every price term shows
    return gs


def test_an_edition_tag_prices_its_edition_into_the_free_joker():
    """tag.lua:398-441: the edition tag's event runs set_edition (which ends
    in set_cost, card.lua:461), marks the joker couponed and calls set_cost
    again. The joker is free, but its edition surcharge and its sell value
    come from the full price."""
    gs = _shop("AUDIT_G204_TAG")
    joker = create_joker("j_joker")
    joker.set_cost(discount_percent=gs["discount_percent"])
    gs["shop_cards"] = [joker]
    _award_tag(gs, "tag_polychrome", source="test")

    _fire_shop_tags(gs, rerolled=True)

    assert joker.edition and joker.edition.get("polychrome")
    assert joker.ability.get("couponed")
    assert joker.extra_cost == gs["inflation"] + _SURCHARGE["polychrome"]
    assert joker.cost == 0
    full = _full_price(gs, joker.base_cost, joker.extra_cost)
    assert joker.sell_cost == max(1, full // 2)


def test_an_illusion_edition_is_priced_in():
    """UI_definitions.lua:786-794: Illusion's edition goes on through
    set_edition, whose set_cost prices it (card.lua:461)."""
    gs = _shop("AUDIT_G204_ILLUSION")
    gs.setdefault("used_vouchers", {})["v_illusion"] = True
    gs.update(joker_rate=0, tarot_rate=0, planet_rate=0, playing_card_rate=4)
    ante = gs["round_resets"]["ante"]
    card = None
    for _ in range(100):
        card = create_card_for_shop(gs, gs["rng"], ante)
        if card.edition:
            break
    assert card is not None and card.edition, "no Illusion edition in 100 shop cards"
    kind = card.edition["type"]
    assert card.extra_cost == gs["inflation"] + _SURCHARGE[kind]
    assert card.cost == _full_price(gs, card.base_cost, card.extra_cost)
    assert card.sell_cost == max(1, card.cost // 2)


def test_a_coupon_tag_card_stays_free_through_a_later_pass():
    """tag.lua:447-465: the Coupon Tag marks the shop's cards couponed and
    runs set_cost on each, so they cost 0 and their sell values already
    follow the run's current discount; a later set_cost pass (an inflation
    purchase, Clearance Sale) keeps them free and leaves the sell values
    (card.lua:382-383)."""
    gs = _shop("AUDIT_G204_COUPON")
    gs["shop_free"] = False
    _award_tag(gs, "tag_coupon", source="test")

    _fire_shop_tags(gs, rerolled=False)
    offered = gs["shop_cards"] + gs["shop_boosters"]
    assert offered and all(c.cost == 0 for c in offered)
    sells = [c.sell_cost for c in offered]

    _all_cards_set_cost_pass(gs)
    assert all(c.cost == 0 for c in offered)
    assert [c.sell_cost for c in offered] == sells
