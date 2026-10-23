"""Board-building helpers for the blind / round-flow / hand-mechanics audit tests.

They drive the public engine (``initialize_run`` + ``step``) and only reach
into the state where vanilla gets there by means the engine API has no action
for: forcing the boss on deck, or putting a specific joker on the board the
way any joker joins it (``G.jokers:emplace`` + ``Card:add_to_deck``).
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import SelectBlind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_consumable, create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run


def new_run(seed: str, back: str = "b_red", stake: int = 1) -> dict[str, Any]:
    """A fresh run at the Small Blind select screen."""
    return initialize_run(back, stake, seed)


def force_boss(gs: dict[str, Any], boss_key: str) -> None:
    """Put *boss_key* on deck as the next blind (skipping Small and Big)."""
    gs["round_resets"]["blind_choices"]["Boss"] = boss_key
    gs["blind_on_deck"] = "Boss"


def start_boss(gs: dict[str, Any], boss_key: str) -> dict[str, Any]:
    """Select *boss_key* as the next blind and play into SELECTING_HAND."""
    force_boss(gs, boss_key)
    return step(gs, SelectBlind())


def give_joker(gs: dict[str, Any], key: str, **kwargs: Any) -> Card:
    """Add a joker the way every joker joins the board: emplace, then add_to_deck."""
    card = create_joker(key, **kwargs)
    gs.setdefault("jokers", []).append(card)
    card.add_to_deck(gs)
    return card


def give_consumable(gs: dict[str, Any], key: str) -> Card:
    card = create_consumable(key)
    gs.setdefault("consumables", []).append(card)
    card.add_to_deck(gs)
    return card


def playing_card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    return create_playing_card(Suit(suit), Rank(rank), enhancement=enhancement)


def index_of(cards: list[Card], card: Card) -> int:
    """Position of *card* in *cards* by identity (Cards compare by value)."""
    for i, c in enumerate(cards):
        if c is card:
            return i
    raise ValueError(f"{card!r} is not in the list")


def ids(cards: list[Card]) -> list[int]:
    return [c.sort_id for c in cards]
