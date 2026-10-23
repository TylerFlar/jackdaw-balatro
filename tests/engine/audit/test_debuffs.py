"""Audit regressions: debuffs toggle passives, perishables, round-end order.

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

from jackdaw.engine.actions import CashOut, GamePhase, SelectBlind, SellCard
from jackdaw.engine.game import step
from tests.engine.audit._boards import give_joker, new_run, win_round

# ---------------------------------------------------------------------------
# G15 — set_debuff toggles passives; perishables; end-of-round order
# ---------------------------------------------------------------------------


def _about_to_perish(gs: dict, key: str, **kwargs):
    joker = give_joker(gs, key, perishable=True, **kwargs)
    joker.perish_tally = 1
    return joker


def test_g15_expiring_perishable_loses_its_passive():
    """calculate_perishable debuffs the joker when its tally runs out
    (card.lua:2278-2283), and Card:set_debuff on a joker calls
    remove_from_deck(true) (card.lua:526-531): a perished Juggler gives its
    +1 hand size back, a perished Drunkard its +1 discard."""
    gs = new_run("AUDIT_G15")
    juggler = _about_to_perish(gs, "j_juggler")
    drunkard = _about_to_perish(gs, "j_drunkard")
    hand_size = gs["hand_size"]
    discards = gs["round_resets"]["discards"]
    step(gs, SelectBlind())

    win_round(gs)

    assert juggler.debuff and drunkard.debuff
    assert gs["hand_size"] == hand_size - 1
    assert gs["round_resets"]["discards"] == discards - 1


def test_g15_perished_joker_stays_debuffed():
    """Once perish_tally is 0, set_debuff keeps the card debuffed whatever it
    is asked (card.lua:527-533): a blind's reset or Crimson Heart cannot
    revive it, nor give back its passive."""
    gs = new_run("AUDIT_G15S")
    juggler = _about_to_perish(gs, "j_juggler")
    step(gs, SelectBlind())
    win_round(gs)
    hand_size = gs["hand_size"]

    juggler.set_debuff(False, gs)

    assert juggler.debuff
    assert gs["hand_size"] == hand_size


def test_g15_dollar_bonus_comes_after_the_perishable_tick():
    """end_round ticks perishables in its joker loop (state_events.lua:99-110);
    calculate_dollar_bonus runs later in evaluate_round
    (state_events.lua:1175-1182) and returns nothing for a debuffed joker
    (card.lua:1655-1656).  A Golden Joker that perishes this round pays $0."""
    gs = new_run("AUDIT_G15G")
    _about_to_perish(gs, "j_golden")
    step(gs, SelectBlind())

    win_round(gs)

    assert gs["round_earnings"].joker_dollars == 0


def test_g15_rent_is_charged_for_a_joker_that_destroys_itself():
    """The end_round loop calls calculate_rental right after each joker's
    end_of_round context (state_events.lua:101-108); a joker destroying
    itself there is only removed by a later event, so its rent is still
    charged (card.lua:2271-2276)."""
    gs = new_run("AUDIT_G15R")
    popcorn = give_joker(gs, "j_popcorn", rental=True)
    popcorn.ability["mult"] = popcorn.ability["extra"]  # eats its last mult this round
    step(gs, SelectBlind())
    dollars = gs["dollars"]

    win_round(gs)

    assert all(j is not popcorn for j in gs["jokers"])
    assert gs["dollars"] == dollars - gs["rental_rate"]


def test_g15_debuffed_negative_joker_frees_its_slot_when_sold():
    """A Negative joker debuffed by set_debuff keeps its slot
    (remove_from_deck(true) only queues the removal, card.lua:687-690); when
    the card goes, Card:remove gives the queued slot back
    (card.lua:4733-4739)."""
    gs = new_run("AUDIT_G15N")
    slots = gs["joker_slots"]
    juggler = _about_to_perish(gs, "j_juggler", edition={"negative": True})
    assert gs["joker_slots"] == slots + 1
    step(gs, SelectBlind())
    win_round(gs)
    step(gs, CashOut())
    assert juggler.debuff
    assert gs["joker_slots"] == slots + 1
    assert gs["phase"] == GamePhase.SHOP

    step(gs, SellCard(area="jokers", card_index=0))

    assert gs["joker_slots"] == slots
