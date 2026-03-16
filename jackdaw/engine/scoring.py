"""Scoring pipeline and eval_card wrapper.

Ports ``eval_card`` from ``common_events.lua:580`` and the base scoring
pipeline from ``state_events.lua:571-1065`` (Phases 1-4, 6-8, 12 without
joker effects).

Note: ``calculate_joker`` calls are stubbed — joker effects will be added
when the joker system is implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# ScoreResult
# ---------------------------------------------------------------------------


@dataclass
class ScoreResult:
    """Result of the base scoring pipeline (without joker effects)."""

    hand_type: str
    """Detected hand type (e.g. ``"Full House"``) or ``"NULL"``."""

    scoring_cards: list[Card]
    """Cards that scored (including Splash/Stone augmentation)."""

    chips: float
    """Final chip value (after all per-card and edition bonuses)."""

    mult: float
    """Final mult value (after all additive and multiplicative bonuses)."""

    total: int
    """``floor(chips * mult)`` — the score added to G.GAME.chips."""

    dollars_earned: int
    """Dollars earned from scoring (Gold Seal, Lucky Card, etc.)."""

    debuffed: bool
    """True if the hand was blocked by a boss blind."""

    breakdown: list[str] = field(default_factory=list)
    """Step-by-step log for debugging."""


# ---------------------------------------------------------------------------
# Base scoring pipeline (Phases 1-4, 6-8, 12 without joker effects)
# ---------------------------------------------------------------------------


def score_hand_base(
    played_cards: list[Card],
    held_cards: list[Card],
    hand_levels: Any,  # HandLevels
    blind: Any,  # Blind
    rng: PseudoRandom,
    *,
    probabilities_normal: float = 1.0,
    joker_flags: dict[str, bool] | None = None,
) -> ScoreResult:
    """Score a hand without joker effects.

    Implements Phases 1-4, 6-8, 12 of the scoring pipeline from
    ``state_events.lua:571-1065``.
    """
    from jackdaw.engine.hand_eval import evaluate_hand

    _ = joker_flags  # reserved for future joker flag passing
    dollars = 0
    breakdown: list[str] = []

    # === Phase 1-2: Hand detection ===
    eval_result = evaluate_hand(played_cards, jokers=None)
    hand_type = eval_result.detected_hand
    scoring_cards = eval_result.scoring_cards
    poker_hands = eval_result.poker_hands

    if hand_type == "NULL":
        return ScoreResult(
            hand_type="NULL", scoring_cards=[], chips=0, mult=0,
            total=0, dollars_earned=0, debuffed=False, breakdown=["No hand"],
        )

    # === Phase 3: Boss blind debuff check ===
    debuffed = blind.debuff_hand(scoring_cards, poker_hands, hand_type)
    if debuffed:
        return ScoreResult(
            hand_type=hand_type, scoring_cards=scoring_cards,
            chips=0, mult=0, total=0, dollars_earned=0, debuffed=True,
            breakdown=[f"Hand blocked by {blind.name}"],
        )

    # === Phase 4: Base chips/mult from hand level ===
    base_chips, base_mult = hand_levels.get(hand_type)
    hand_chips = float(base_chips)
    mult = float(base_mult)
    breakdown.append(
        f"Base: {hand_type} L{hand_levels[hand_type].level}"
        f" -> {int(hand_chips)} chips, {int(mult)} mult"
    )

    # Record play
    hand_levels.record_play(hand_type)

    # === Phase 6: Blind modify_hand (The Flint) ===
    new_mult, new_chips, modified = blind.modify_hand(mult, int(hand_chips))
    if modified:
        mult = float(new_mult)
        hand_chips = float(new_chips)
        breakdown.append(f"Blind modify: {int(hand_chips)} chips, {int(mult)} mult")

    # === Phase 7: Per scored card (with retriggers) ===
    for card in scoring_cards:
        if card.debuff:
            continue

        # Collect retriggers
        reps = [1]  # base evaluation
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for _rep_idx in range(len(reps)):
            ev = eval_card(
                card, {"cardarea": "play"},
                rng=rng, probabilities_normal=probabilities_normal,
            )

            # Apply effects in source order (state_events.lua:702-776)
            if "chips" in ev:
                hand_chips += ev["chips"]
            if "mult" in ev:
                mult += ev["mult"]
            if "p_dollars" in ev:
                dollars += ev["p_dollars"]
            if "x_mult" in ev:
                mult *= ev["x_mult"]
            if "edition" in ev:
                ed = ev["edition"]
                hand_chips += ed.get("chip_mod", 0)
                mult += ed.get("mult_mod", 0)
                mult *= ed.get("x_mult_mod", 1)

    # === Phase 8: Per held card (with retriggers) ===
    for card in held_cards:
        if card.debuff:
            continue

        reps = [1]
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for _rep_idx in range(len(reps)):
            ev = eval_card(
                card, {"cardarea": "hand"},
                rng=rng, probabilities_normal=probabilities_normal,
            )

            # Apply in source order (state_events.lua:845-862)
            if "h_mult" in ev:
                mult += ev["h_mult"]
            if "x_mult" in ev:
                mult *= ev["x_mult"]

    # === Phase 12: Final score ===
    total = math.floor(hand_chips * mult)
    breakdown.append(f"Final: {int(hand_chips)} x {mult:.1f} = {total}")

    return ScoreResult(
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        chips=hand_chips,
        mult=mult,
        total=total,
        dollars_earned=dollars,
        debuffed=False,
        breakdown=breakdown,
    )
