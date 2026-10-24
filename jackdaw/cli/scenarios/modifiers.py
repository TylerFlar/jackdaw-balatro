"""Card modifier validation scenarios — enhancements, editions, seals, combinations.

Tests that card modifiers (enhancements, editions, seals) score the same in the
live game as in the shadow run. ``add`` takes BalatroBot's names (``GLASS``,
``FOIL``, ``RED``).

Authoritative source: balatro_source/game.lua (P_CENTERS, Enhanced/Edition sets)
and balatro_source/card.lua (Card:eval_card).

Enhancement values (from game.lua):
  m_bonus:  +30 chips when scored
  m_mult:   +4 mult when scored
  m_wild:   counts as any suit
  m_glass:  x2 mult, 1 in 4 chance to destroy
  m_steel:  x1.5 mult while in hand (not played)
  m_stone:  +50 chips, no rank/suit, always scores
  m_gold:   +$3 at end of round
  m_lucky:  1 in 5 for +20 mult, 1 in 15 for +$20

Edition values (from game.lua):
  e_foil:        +50 chips
  e_holo:        +10 mult
  e_polychrome:  x1.5 mult
  e_negative:    +1 joker slot

Seal effects:
  Red:    retrigger card once
  Blue:   creates planet card if held at end of round
  Gold:   +$3 when held at end of round
  Purple: creates tarot when discarded
"""

from __future__ import annotations

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Lockstep,
    add_both,
    compare_state,
    discard,
    get_hand_count,
    play_hand,
    run_modifier_scenario,
    select_blind,
    start_both,
)

# ---------------------------------------------------------------------------
# Enhancement scenarios
# ---------------------------------------------------------------------------

# (BalatroBot's enhancement name, description)
_ENHANCEMENTS: list[tuple[str, str]] = [
    ("BONUS", "+30 chips when scored"),
    ("MULT", "+4 mult when scored"),
    ("WILD", "counts as any suit"),
    ("GLASS", "x2 mult, 1/4 chance to destroy"),
    ("STEEL", "x1.5 mult while held (not played)"),
    ("STONE", "+50 chips, always scores, no rank/suit"),
    ("GOLD", "+$3 at end of round when held"),
    ("LUCKY", "1/5 for +20 mult, 1/15 for +$20"),
]

for _enh, _desc in _ENHANCEMENTS:

    def _make_fn(enh: str = _enh):  # noqa: B023
        def fn(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
            return run_modifier_scenario(ls, card_key="H_A", enhancement=enh, delay=delay)

        return fn

    register(
        name=f"modifier_{_enh.lower()}",
        category="modifiers",
        description=f"Enhancement m_{_enh.lower()}: {_desc}",
    )(_make_fn(_enh))

# ---------------------------------------------------------------------------
# Edition scenarios
# ---------------------------------------------------------------------------

# (BalatroBot's edition name, description)
_EDITIONS: list[tuple[str, str]] = [
    ("FOIL", "+50 chips"),
    ("HOLO", "+10 mult"),
    ("POLYCHROME", "x1.5 mult"),
]

for _ed, _desc in _EDITIONS:

    def _make_fn(ed: str = _ed):  # noqa: B023
        def fn(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
            return run_modifier_scenario(ls, card_key="H_A", edition=ed, delay=delay)

        return fn

    register(
        name=f"modifier_{_ed.lower()}",
        category="modifiers",
        description=f"Edition {_ed.lower()}: {_desc}",
    )(_make_fn(_ed))


# Negative edition on joker (adds joker slot)
@register(
    name="modifier_negative_joker",
    category="modifiers",
    description="Negative edition on joker: +1 joker slot",
)
def _modifier_negative(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    start_both(ls, seed="M_NEGATIVE", delay=delay)
    select_blind(ls, delay=delay)
    add_both(ls, key="j_joker", edition="NEGATIVE")
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(ls, label="negative joker")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Negative joker: {'PASS' if not diffs else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Seal scenarios
# ---------------------------------------------------------------------------


@register(
    name="modifier_red_seal",
    category="modifiers",
    description="Red seal: retrigger card once when scored",
)
def _modifier_red_seal(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(ls, card_key="H_A", seal="RED", delay=delay)


@register(
    name="modifier_gold_seal",
    category="modifiers",
    description="Gold seal: +$3 when held at end of round",
)
def _modifier_gold_seal(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    start_both(ls, seed="M_GOLD_SEAL", delay=delay)
    select_blind(ls, delay=delay)
    # Add a gold sealed card — don't play it so it's held at end of round
    add_both(ls, key="H_A", seal="GOLD")
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)  # play without the gold card
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(ls, label="gold seal end of round")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Gold seal: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="modifier_blue_seal",
    category="modifiers",
    description="Blue seal: creates planet if held at end of round",
)
def _modifier_blue_seal(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    start_both(ls, seed="M_BLUE_SEAL", delay=delay)
    select_blind(ls, delay=delay)
    add_both(ls, key="H_A", seal="BLUE")
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(ls, label="blue seal end of round")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Blue seal: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="modifier_purple_seal",
    category="modifiers",
    description="Purple seal: creates tarot when discarded",
)
def _modifier_purple_seal(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    start_both(ls, seed="M_PURPLE_SEAL", delay=delay)
    select_blind(ls, delay=delay)
    add_both(ls, key="H_A", seal="PURPLE")
    # Discard the purple-sealed card to trigger tarot creation.
    # It was appended to the end of hand, so discard the last card.
    last_idx = get_hand_count(ls) - 1
    discard(ls, [last_idx], delay=delay)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(ls, label="purple seal discard")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Purple seal: {'PASS' if not diffs else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Combination scenarios
# ---------------------------------------------------------------------------


@register(
    name="modifier_glass_polychrome",
    category="modifiers",
    description="Glass + Polychrome: x2 * x1.5 = x3 mult",
)
def _modifier_glass_poly(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        ls, card_key="H_A", enhancement="GLASS", edition="POLYCHROME", delay=delay
    )


@register(
    name="modifier_steel_holo",
    category="modifiers",
    description="Steel + Holo: x1.5 mult + 10 mult while held",
)
def _modifier_steel_holo(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        ls, card_key="H_A", enhancement="STEEL", edition="HOLO", delay=delay
    )


@register(
    name="modifier_stone_foil",
    category="modifiers",
    description="Stone + Foil: +50 chips + 50 chips",
)
def _modifier_stone_foil(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        ls, card_key="H_A", enhancement="STONE", edition="FOIL", delay=delay
    )


@register(
    name="modifier_lucky_red_seal",
    category="modifiers",
    description="Lucky + Red seal: retrigger lucky roll",
)
def _modifier_lucky_red(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(ls, card_key="H_A", enhancement="LUCKY", seal="RED", delay=delay)


@register(
    name="modifier_bonus_gold_seal",
    category="modifiers",
    description="Bonus + Gold seal: +30 chips scored + $3 held",
)
def _modifier_bonus_gold(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(ls, card_key="H_A", enhancement="BONUS", seal="GOLD", delay=delay)


@register(
    name="modifier_mult_foil", category="modifiers", description="Mult + Foil: +4 mult + 50 chips"
)
def _modifier_mult_foil(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        ls, card_key="H_A", enhancement="MULT", edition="FOIL", delay=delay
    )
