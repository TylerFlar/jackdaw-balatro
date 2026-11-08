"""Debuffed jokers in Card:update's per-frame values, the editionless-joker pool and
``find_joker``, on the Rust engine.

These fixes had no gap id. Their regression tests were the oracle's own
(``tests/engine/audit/``) and went with it; the rest of them are covered by
``test_consumables_rust.py`` (The Wheel of Fortune and Ectoplasm on debuffed jokers) and
``test_jokers_batch_b.py`` (Swashbuckler). A debuffed joker is written as ``set_debuff(true)``
leaves it: debuffed and out of the deck (card.lua:526-538).
"""

from __future__ import annotations

from typing import Any

from jackdaw.core import Game, GameBuilder
from jackdaw.engine.actions import (
    CashOut,
    NextRound,
    PlayHand,
    SelectBlind,
    SellCard,
    UseConsumable,
)


def _debuff_joker(game: Game, index: int, **stickers: Any) -> None:
    snap = game.to_dict()
    card = snap["areas"]["jokers"][index]
    card["debuff"] = True
    card["added_to_deck"] = False
    card["stickers"].update(stickers)
    game.replace_state(snap)


def _plain_jokers(game: Game) -> None:
    # The debug add polls an edition (the live game doesn't); these seeds poll none.
    assert all(j.edition is None for j in game.jokers), [j.edition for j in game.jokers]


def test_hex_targets_a_debuffed_joker() -> None:
    """Hex picks from every editionless joker, debuffed or not (eligible_editionless_jokers,
    card.lua:4217-4224): the only joker is debuffed, and Hex makes it Polychrome."""
    game = Game.new("b_red", 1, "AUDIT_HEX_DEBUFF")
    game.step(SelectBlind())
    game.debug_add("j_joker")
    _plain_jokers(game)
    _debuff_joker(game, 0)
    game.debug_add("c_hex")
    assert game.jokers[0].debuff
    assert any(isinstance(a, UseConsumable) and a.card_index == 0 for a in game.legal_actions())
    game.step(UseConsumable(card_index=0))
    assert [(j.key, j.edition) for j in game.jokers] == [("j_joker", "polychrome")]


def test_stencil_counts_a_debuffed_stencil() -> None:
    """Joker Stencil's x_mult is the empty joker slots plus every Joker Stencil in G.jokers,
    debuffed or not (card.lua:4203-4208): 5 slots, 3 jokers and 2 Stencils give the working
    Stencil X4 on a High Card of 5 + 2 chips."""
    game = Game.new("b_red", 1, "AUDIT_STENCIL_DEBUFF")
    game.step(SelectBlind())
    for key in ("j_stencil", "j_stencil", "j_greedy_joker"):
        game.debug_add(key)
    _plain_jokers(game)
    two = game.debug_add("S_2")
    _debuff_joker(game, 1)
    assert [j.debuff for j in game.jokers] == [False, True, False]
    game.step(PlayHand(card_indices=([c.sort_id for c in game.hand].index(two),)))
    score = game.last_score
    assert score is not None and (score.chips, score.mult) == (5 + 2, 4)


def test_a_debuffed_chaos_gives_no_free_reroll() -> None:
    """new_round sets free_rerolls = #find_joker('Chaos the Clown') (state_events.lua:311-313),
    and find_joker skips debuffed jokers (misc_functions.lua:903-917): a perished Chaos, which
    stays debuffed (card.lua:527-533), gives the round no free reroll."""
    for perished, rerolls in ((False, 1), (True, 0)):
        game = Game.new("b_red", 1, "AUDIT_CHAOS")
        game.debug_add("j_chaos")
        _plain_jokers(game)
        if perished:
            _debuff_joker(game, 0, perishable=True, perish_tally=0)
        game.step(SelectBlind())
        assert game.jokers[0].debuff == perished
        assert game.to_dict()["current_round"]["free_rerolls"] == rerolls, perished


def _slots(game: Game) -> float:
    return game.to_dict()["game"]["joker_slots"]


def _only_joker(game: Game) -> dict[str, Any]:
    (joker,) = game.to_dict()["areas"]["jokers"]
    return joker


def test_ectoplasm_on_a_debuffed_joker_costs_a_slot_once_sold() -> None:
    """The game's own slot bug, kept on purpose: Ectoplasm's Negative on a debuffed joker adds
    no slot, and selling the joker later takes one away.

    A debuffed joker is out of the deck (set_debuff -> remove_from_deck(true), card.lua:535),
    and set_edition adds the Negative slot only to a card added to the deck (card.lua:405-415).
    Crimson Heart's defeat lifts the debuff (blind.lua:336 -> 211-212 -> 651); add_to_deck(true)
    adds no slot for a Negative coming back from a debuff (card.lua:630-633). Selling it runs
    remove_from_deck(), which takes the slot (card.lua:687-697 from Card:remove, 4727-4733):
    one slot fewer than before Ectoplasm. https://balatrowiki.org/w/Ectoplasm (Strategy)."""
    game = (
        GameBuilder("b_red", 1, "AUDIT_ECTO_SLOT")
        .boss("bl_final_heart")
        .select_blind()
        .joker("j_joker")
        .consumable("c_ectoplasm")
        .build()
    )
    _plain_jokers(game)
    slots = _slots(game)
    # Crimson Heart debuffs the only joker after the hand (blind.lua:588-600).
    game.step(PlayHand(card_indices=(0,)))
    joker = _only_joker(game)
    assert joker["debuff"] and not joker["added_to_deck"]

    game.step(UseConsumable(card_index=0))
    joker = _only_joker(game)
    assert game.jokers[0].edition == "negative" and joker["debuff"]
    assert not joker["ability"].get("queue_negative_removal")
    assert _slots(game) == slots

    game.debug_set(chips=10**9)
    game.step(PlayHand(card_indices=(0,)))
    assert game.phase == "round_eval"
    joker = _only_joker(game)
    assert not joker["debuff"] and joker["added_to_deck"]
    assert not joker["ability"].get("queue_negative_removal")
    assert _slots(game) == slots

    game.step(CashOut())
    game.step(SellCard(area="jokers", card_index=0))
    assert not game.jokers
    assert _slots(game) == slots - 1


def test_ectoplasm_on_a_perished_joker_adds_no_slot() -> None:
    """A perished joker stays debuffed (card.lua:527-533), so Ectoplasm's Negative adds no slot
    (card.lua:405-415). Sold while still debuffed, it takes none back: remove_from_deck returns
    for a card out of the deck (card.lua:646), and the removal was never queued (card.lua:687-689
    queues it only for a joker already Negative when debuffed; Card:remove, 4733-4738)."""
    game = (
        GameBuilder("b_red", 1, "AUDIT_ECTO_PERISHED")
        .select_blind()
        .joker("j_joker", perishable=1)
        .chips(10**9)
        .build()
    )
    _plain_jokers(game)
    slots = _slots(game)
    game.step(PlayHand(card_indices=(0,)))
    game.step(CashOut())
    joker = _only_joker(game)
    assert joker["debuff"] and not joker["added_to_deck"]

    game.debug_add("c_ectoplasm")
    game.step(UseConsumable(card_index=0))
    joker = _only_joker(game)
    assert game.jokers[0].edition == "negative" and joker["debuff"]
    assert _slots(game) == slots

    game.step(SellCard(area="jokers", card_index=0))
    assert not game.jokers
    assert _slots(game) == slots


def test_ectoplasm_then_perishing_costs_a_slot_once_sold() -> None:
    """The same loss through a Perishable joker: Ectoplasm lands while Crimson Heart debuffs it
    (no slot, card.lua:405-415), the boss's defeat lifts the debuff (card.lua:630-633 adds
    none), and when the joker perishes a round later, remove_from_deck(true) queues the removal
    of the Negative slot (card.lua:527-533, 687-689). Selling it takes that slot (Card:remove,
    card.lua:4733-4738): one fewer than before Ectoplasm."""
    game = (
        GameBuilder("b_red", 1, "AUDIT_ECTO_PERISH_LATER")
        .boss("bl_final_heart")
        .select_blind()
        .joker("j_joker", perishable=2)
        .consumable("c_ectoplasm")
        .build()
    )
    _plain_jokers(game)
    slots = _slots(game)
    game.step(PlayHand(card_indices=(0,)))
    game.step(UseConsumable(card_index=0))
    assert game.jokers[0].edition == "negative" and _only_joker(game)["debuff"]
    assert _slots(game) == slots

    game.debug_set(chips=10**9)
    game.step(PlayHand(card_indices=(0,)))
    joker = _only_joker(game)
    assert not joker["debuff"] and joker["added_to_deck"]
    assert joker["stickers"]["perish_tally"] == 1
    assert _slots(game) == slots

    game.step(CashOut())
    game.step(NextRound())
    game.step(SelectBlind())
    game.debug_set(chips=10**9)
    game.step(PlayHand(card_indices=(0,)))
    joker = _only_joker(game)
    assert joker["debuff"] and not joker["added_to_deck"]
    assert joker["stickers"]["perish_tally"] == 0
    assert joker["ability"].get("queue_negative_removal")
    assert _slots(game) == slots

    game.step(CashOut())
    game.step(SellCard(area="jokers", card_index=0))
    assert not game.jokers
    assert _slots(game) == slots - 1
