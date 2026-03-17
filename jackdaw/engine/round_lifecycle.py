"""Per-round card maintenance — perishable, rental, and self-destruct.

Fires at end of round after joker ``calculate_joker({end_of_round=true})``
but before ``evaluate_round`` (the cash-out calculation in :mod:`economy`).

Source: ``state_events.lua:87-110`` — the ``end_round`` function calls
``calculate_joker``, then ``calculate_rental``, then ``calculate_perishable``
for each joker in order.

This module handles the card-level state mutations that
:func:`~jackdaw.engine.jokers.on_end_of_round` does not cover:

* **Perishable countdown**: ``perish_tally`` decrements each round;
  when it hits 0 the card is permanently debuffed.
* **Rental charges**: each rental card costs ``rental_rate`` ($3) per round,
  deducted directly from ``game_state["dollars"]``.

Glass Card behaviour
--------------------
Glass Cards only shatter during scoring (Phase 11 of the scoring pipeline).
They do **not** shatter at round end — confirmed by reviewing ``card.lua``
which only calls ``card:shatter()`` inside ``G.FUNCS.play_cards_from_highlighted``
and ``calculate_joker`` destruction contexts, never in ``end_round``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.card import Card


@dataclass
class RoundEndResult:
    """Descriptor returned by :func:`process_round_end_cards`.

    All lists contain references to the actual Card objects that were
    affected — callers can inspect or remove them as needed.
    """

    perished: list[Card] = field(default_factory=list)
    """Jokers whose ``perish_tally`` hit 0 this round and became debuffed."""

    rental_cost: int = 0
    """Total dollars deducted for rental cards this round."""

    rental_cards: list[Card] = field(default_factory=list)
    """Cards that incurred a rental charge."""


def process_round_end_cards(
    jokers: list[Card],
    game_state: dict[str, Any],
) -> RoundEndResult:
    """Process card maintenance at end of round.

    Called after :func:`~jackdaw.engine.jokers.on_end_of_round` and before
    :func:`~jackdaw.engine.economy.calculate_round_earnings`.

    Mirrors the per-joker loop in ``state_events.lua:99-109``::

        for i = 1, #G.jokers.cards do
            G.jokers.cards[i]:calculate_joker({end_of_round = true, ...})
            G.jokers.cards[i]:calculate_rental()
            G.jokers.cards[i]:calculate_perishable()
        end

    The ``calculate_joker`` call is handled by
    :func:`~jackdaw.engine.jokers.on_end_of_round`;
    this function handles the remaining two steps.

    Parameters
    ----------
    jokers:
        Active joker cards.  Mutated in place (``perish_tally``,
        ``debuff`` fields).
    game_state:
        Mutable run-state dict.  ``game_state["dollars"]`` is decremented
        by rental charges.  Reads ``game_state.get("rental_rate", 3)``.

    Returns
    -------
    RoundEndResult
        Summary of what happened.
    """
    rental_rate: int = game_state.get("rental_rate", 3)
    result = RoundEndResult()

    for joker in jokers:
        # ---------------------------------------------------------------
        # calculate_rental — card.lua:2271-2276
        # ---------------------------------------------------------------
        if _is_rental(joker):
            game_state["dollars"] = game_state.get("dollars", 0) - rental_rate
            result.rental_cost += rental_rate
            result.rental_cards.append(joker)

        # ---------------------------------------------------------------
        # calculate_perishable — card.lua:2278-2289
        # ---------------------------------------------------------------
        if _is_perishable(joker):
            tally = _get_perish_tally(joker)
            if tally > 0:
                if tally == 1:
                    # Hits zero — permanently debuff
                    _set_perish_tally(joker, 0)
                    joker.debuff = True
                    result.perished.append(joker)
                else:
                    _set_perish_tally(joker, tally - 1)

    return result


# ---------------------------------------------------------------------------
# Card field accessors — handle both attribute and dict-based ability
# ---------------------------------------------------------------------------
# The Card dataclass stores perishable/rental as top-level fields AND
# historically some code stores them in card.ability dict (matching Lua's
# self.ability.perishable).  We check both for robustness.


def _is_rental(card: Card) -> bool:
    if getattr(card, "rental", False):
        return True
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return bool(ability.get("rental"))
    return False


def _is_perishable(card: Card) -> bool:
    if getattr(card, "perishable", False):
        return True
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return bool(ability.get("perishable"))
    return False


def _get_perish_tally(card: Card) -> int:
    # Prefer card.ability.perish_tally (matches Lua) then card.perish_tally
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict) and "perish_tally" in ability:
        return ability["perish_tally"]
    return getattr(card, "perish_tally", 0)


def _set_perish_tally(card: Card, value: int) -> None:
    # Update both locations for consistency
    if isinstance(getattr(card, "ability", None), dict):
        card.ability["perish_tally"] = value
    card.perish_tally = value
