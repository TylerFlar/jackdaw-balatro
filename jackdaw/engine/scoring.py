"""Card evaluation wrapper for the scoring pipeline.

Ports ``eval_card`` from ``common_events.lua:580``.  This is the function
that calls the appropriate Card scoring methods based on context and assembles
the return dict consumed by the scoring pipeline.

Note: ``calculate_joker`` calls are stubbed — joker effects will be added
when the joker system is implemented (M5+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom


def eval_card(
    card: Card,
    context: dict[str, Any] | None = None,
    *,
    rng: PseudoRandom | None = None,
    probabilities_normal: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a single card's scoring contribution.

    Matches ``eval_card`` (common_events.lua:580).

    The return dict contains only fields with non-zero values (matching
    the source's ``if value > 0 then ret.field = value`` pattern).

    Context keys:
        - ``cardarea``: ``"play"`` or ``"hand"`` — determines which
          scoring methods to call.
        - ``repetition_only``: If true, only check for retrigger seals.
        - ``edition``: If true, only return edition info (for joker
          edition pass in Phase 9a).

    For played cards (``cardarea="play"``):
        Returns: chips, mult, x_mult, p_dollars, edition.

    For held cards (``cardarea="hand"``):
        Returns: h_mult, x_mult (from h_x_mult).

    For repetition check (``repetition_only=True``):
        Returns: seals (with repetitions count).

    Args:
        card: The card to evaluate.
        context: Context dict with cardarea and flags.
        rng: PseudoRandom instance for Lucky Card rolls.
        probabilities_normal: ``G.GAME.probabilities.normal`` (default 1).
    """
    ctx = context or {}
    ret: dict[str, Any] = {}

    # Repetition-only mode: just check seals
    if ctx.get("repetition_only"):
        seals = card.calculate_seal(repetition=True)
        if seals:
            ret["seals"] = seals
        return ret

    cardarea = ctx.get("cardarea")

    # Played cards (cardarea == G.play → "play")
    if cardarea == "play":
        chips = card.get_chip_bonus()
        if chips > 0:
            ret["chips"] = chips

        mult = card.get_chip_mult(
            rng=rng, probabilities_normal=probabilities_normal,
        )
        if mult > 0:
            ret["mult"] = mult

        x_mult = card.get_chip_x_mult()
        if x_mult > 0:
            ret["x_mult"] = x_mult

        p_dollars = card.get_p_dollars(
            rng=rng, probabilities_normal=probabilities_normal,
        )
        if p_dollars > 0:
            ret["p_dollars"] = p_dollars

        # Joker effects on played cards (calculate_joker stub)
        # Will be: jokers = card.calculate_joker(context)
        # if jokers: ret["jokers"] = jokers

        edition = card.get_edition()
        if edition:
            ret["edition"] = edition

    # Held-in-hand cards (cardarea == G.hand → "hand")
    elif cardarea == "hand":
        h_mult = card.get_chip_h_mult()
        if h_mult > 0:
            ret["h_mult"] = h_mult

        # Source maps h_x_mult to ret.x_mult (not ret.h_x_mult)
        h_x_mult = card.get_chip_h_x_mult()
        if h_x_mult > 0:
            ret["x_mult"] = h_x_mult

        # Joker effects on held cards (calculate_joker stub)

    # Joker area (cardarea == G.jokers → "jokers")
    elif cardarea == "jokers":
        if ctx.get("edition"):
            edition = card.get_edition()
            if edition:
                ret["jokers"] = edition
        # else: calculate_joker stub

    return ret
