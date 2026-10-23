"""G261: Astronomer's set_cost pass (card.lua:616-622, 673-679).

Card:add_to_deck and Card:remove_from_deck queue an event that re-runs
set_cost on every card in G.I.CARD, whatever brings Astronomer in or out:
a purchase, a sale, a debuff, a Hex. The pass covers an open pack's cards.
"""

from __future__ import annotations

from jackdaw.engine.card_factory import create_consumable
from jackdaw.engine.game import run_pending_reprice

from ._joker_helpers import give_joker, new_run, to_shop


def _priced_planet(gs):
    card = create_consumable("c_pluto")
    card.set_cost(inflation=gs.get("inflation", 0), discount_percent=0, has_astronomer=True)
    return card


def test_debuffing_astronomer_reprices_the_planets() -> None:
    gs = to_shop(new_run("AUDIT_ASTRO_1"))
    astronomer = give_joker(gs, "j_astronomer")
    run_pending_reprice(gs)
    planet = _priced_planet(gs)
    gs["consumables"].append(planet)
    assert planet.cost == 0
    astronomer.set_debuff(True, gs)  # remove_from_deck(true) queues the pass
    run_pending_reprice(gs)
    assert planet.cost == 3


def test_the_pass_reprices_an_open_packs_cards() -> None:
    gs = to_shop(new_run("AUDIT_ASTRO_2"))
    astronomer = give_joker(gs, "j_astronomer")
    run_pending_reprice(gs)
    planet = _priced_planet(gs)
    gs["pack_cards"] = [planet]
    gs["jokers"].remove(astronomer)
    astronomer.remove_from_deck(gs)
    run_pending_reprice(gs)
    assert planet.cost == 3
