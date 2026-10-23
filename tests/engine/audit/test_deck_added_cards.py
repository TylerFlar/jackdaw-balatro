"""Audit regressions: playing cards that went through add_to_deck (G205).

``Card:add_to_deck`` sets ``added_to_deck`` on a playing card too before it
returns (card.lua:568-575). Only a shop purchase, a Standard-pack pick and a
Cryptid or DNA copy call it for a playing card. ``Card:remove`` then runs
``remove_from_deck`` (card.lua:4733), which for such a card queues
``G.GAME.blind:set_blind(nil, true)`` (card.lua:698): the boss's debuffs are
re-run over every playing card (blind.lua:207-210).
"""

from __future__ import annotations

from jackdaw.engine import GamePhase, PlayHand, SelectBlind, step
from jackdaw.engine.actions import BuyCard, PickPackCard, UseConsumable
from jackdaw.engine.card_factory import (
    RANK_LETTER,
    SUIT_LETTER,
    create_consumable,
    create_playing_card,
)
from tests.engine.audit._helpers import force_boss, new_run, play_round


def _in_pillar_blind(seed: str) -> dict:
    gs = new_run(seed=seed)
    force_boss(gs, "bl_pillar")
    step(gs, SelectBlind())
    gs["blind"].chips = 10**9  # the round goes on
    return gs


def test_a_shop_playing_card_is_added_to_the_deck():
    """button_callbacks.lua:2418: buy_from_shop runs c1:add_to_deck()."""
    gs = play_round(new_run(seed="AUDIT_G205_BUY"))
    card = create_playing_card(SUIT_LETTER["H"], RANK_LETTER["A"])
    card.set_cost()
    gs["shop_cards"] = [card]
    gs["dollars"] = 50
    step(gs, BuyCard(shop_index=0))
    assert card in gs["deck"] and card.added_to_deck


def test_a_standard_pack_pick_is_added_to_the_deck():
    """button_callbacks.lua:2222-2230: the picked card joins G.deck, then
    card:add_to_deck()."""
    gs = play_round(new_run(seed="AUDIT_G205_PICK"))
    card = create_playing_card(SUIT_LETTER["S"], RANK_LETTER["K"])
    gs.update(
        phase=GamePhase.PACK_OPENING,
        pack_type="Standard",
        pack_cards=[card],
        pack_choices_remaining=1,
        shop_return_phase=GamePhase.SHOP,
    )
    step(gs, PickPackCard(card_index=0))
    assert card in gs["deck"] and card.added_to_deck


def test_cryptid_copies_are_added_to_the_deck():
    """card.lua:1206-1213: each copy runs _card:add_to_deck(); the source
    card, part of the starting deck, never did."""
    gs = _in_pillar_blind("AUDIT_G205_CRYPTID")
    gs["consumables"] = [create_consumable("c_cryptid")]
    source = gs["hand"][0]
    before = list(gs["hand"])
    step(gs, UseConsumable(card_index=0, target_indices=(0,)))
    copies = [c for c in gs["hand"] if c not in before]
    assert len(copies) == 2 and all(c.added_to_deck for c in copies)
    assert not source.added_to_deck


def test_destroying_a_deck_added_card_reruns_the_boss_debuffs():
    """The Pillar debuffs the cards played this ante, but only when its
    debuffs run: at set_blind, and at a reset. A Glass Card picked from a
    Standard pack that shatters queues that reset, so the card played
    earlier in this blind (now in the discard pile) is debuffed."""
    gs = _in_pillar_blind("AUDIT_G205_PILLAR")
    gs["probabilities"]["normal"] = 4  # every Glass Card breaks
    glass = gs["hand"][0]
    glass.set_ability("m_glass")
    glass.added_to_deck = True  # as the Standard-pack pick left it

    step(gs, PlayHand(card_indices=(1,)))
    played = gs["discard_pile"][-1]
    assert played.ability.get("played_this_ante") and not played.debuff

    step(gs, PlayHand(card_indices=(gs["hand"].index(glass),)))
    assert all(glass not in gs[a] for a in ("hand", "deck", "discard_pile"))
    assert played.debuff


def test_a_starting_deck_card_queues_no_reset():
    """A card that never went through add_to_deck has nothing for
    remove_from_deck to undo (card.lua:646): no reset, so the card played
    earlier stays as it was."""
    gs = _in_pillar_blind("AUDIT_G205_PLAIN")
    gs["probabilities"]["normal"] = 4
    glass = gs["hand"][0]
    glass.set_ability("m_glass")

    step(gs, PlayHand(card_indices=(1,)))
    played = gs["discard_pile"][-1]
    step(gs, PlayHand(card_indices=(gs["hand"].index(glass),)))
    assert all(glass not in gs[a] for a in ("hand", "deck", "discard_pile"))
    assert not played.debuff
