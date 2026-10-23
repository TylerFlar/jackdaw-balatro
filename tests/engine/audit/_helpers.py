"""Shared setup helpers for the parity-audit regression tests.

Runs are driven through the public engine (``initialize_run`` + ``step``).
Cards are injected the way BalatroBot's ``add`` does it
(:class:`jackdaw.bridge.backend.SimBackend`); :func:`give_joker` is the same
path minus the edition poll, so a test controls the joker's edition.
"""

from __future__ import annotations

import copy
from typing import Any

from jackdaw.bridge.backend import SimBackend
from jackdaw.engine.actions import CashOut, Discard, GamePhase, PlayHand, SelectBlind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.card_utils import astronomer_active
from jackdaw.engine.game import step
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run


def new_run(seed: str = "AUDIT", back: str = "b_red", stake: int = 1) -> dict[str, Any]:
    """A fresh run at the blind-select screen (Small Blind on deck)."""
    gs = initialize_run(back, stake, seed)
    assert gs["phase"] == GamePhase.BLIND_SELECT
    return gs


def select_blind(gs: dict[str, Any]) -> dict[str, Any]:
    step(gs, SelectBlind())
    assert gs["phase"] == GamePhase.SELECTING_HAND
    return gs


def bot_add(gs: dict[str, Any], key: str, **params: Any) -> None:
    """BalatroBot ``add`` (SimBackend): jokers poll an edition; playing cards
    join the hand in SELECTING_HAND."""
    backend = SimBackend()
    backend._gs = gs
    backend.handle("add", {"key": key, **params})


def give_joker(
    gs: dict[str, Any],
    key: str,
    *,
    edition: dict[str, bool] | None = None,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
) -> Card:
    """``add`` for a joker without the edition poll: create, price, append,
    ``add_to_deck``."""
    card = create_joker(
        key,
        edition=edition,
        eternal=eternal,
        perishable=perishable,
        rental=rental,
        hands_played=gs.get("hands_played", 0),
    )
    card.set_cost(
        inflation=gs.get("inflation", 0),
        discount_percent=gs.get("discount_percent", 0),
        ante=gs["round_resets"]["ante"],
        has_astronomer=astronomer_active(gs),
    )
    gs["jokers"].append(card)
    card.add_to_deck(gs)
    return card


def give_card(gs: dict[str, Any], key: str, **params: Any) -> Card:
    """Add a playing card (``"H_K"``) to the hand (SELECTING_HAND) via ``add``."""
    area = "hand" if gs["phase"] == GamePhase.SELECTING_HAND else "deck"
    bot_add(gs, key, **params)
    return gs[area][-1]


def give_consumable(gs: dict[str, Any], key: str) -> Card:
    bot_add(gs, key)
    return gs["consumables"][-1]


def indices(gs: dict[str, Any], *cards: Card) -> tuple[int, ...]:
    """Hand indices of *cards* (by identity), in hand order."""
    hand = gs["hand"]
    return tuple(sorted(next(i for i, c in enumerate(hand) if c is card) for card in cards))


def play(gs: dict[str, Any], *cards: Card) -> dict[str, Any]:
    return step(gs, PlayHand(card_indices=indices(gs, *cards)))


def discard(gs: dict[str, Any], *cards: Card) -> dict[str, Any]:
    return step(gs, Discard(card_indices=indices(gs, *cards)))


def win_round(gs: dict[str, Any], *cards: Card) -> dict[str, Any]:
    """Beat the current blind with one play (target lowered to 1 chip)."""
    gs["blind"].chips = 1
    if not cards:
        cards = tuple(gs["hand"][:1])
    play(gs, *cards)
    assert gs["phase"] == GamePhase.ROUND_EVAL, gs["phase"]
    return gs


def to_shop(gs: dict[str, Any]) -> dict[str, Any]:
    """From blind select: play the Small Blind, cash out, arrive in the shop."""
    select_blind(gs)
    win_round(gs)
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    return gs


def rng_twin(gs: dict[str, Any]) -> PseudoRandom:
    """An independent copy of the run RNG, to replay draws as an oracle."""
    twin = PseudoRandom(gs["rng"].seed_str)
    twin._state = copy.deepcopy(gs["rng"].state)
    return twin


def stream(gs: dict[str, Any], key: str) -> float | None:
    """Current state of RNG stream *key* (None until first drawn)."""
    return gs["rng"].state.get(key)
