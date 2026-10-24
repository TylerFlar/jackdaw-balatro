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

from jackdaw.core import Game
from jackdaw.engine.actions import PlayHand, SelectBlind, UseConsumable


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
