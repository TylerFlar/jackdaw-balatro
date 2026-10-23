"""Consumable targeting system for the RL environment.

Given a consumable (a :class:`jackdaw.core.CardView` or its center key), determines:

- What card selection constraints apply (min/max targets, filters)
- Which hand cards are valid targets
- Whether a specific target selection is legal

The constraints come from the center's config in ``centers.json``
(``max_highlighted``, ``min_highlighted``, ``mod_num``; card.lua:1564-1568), with
overrides for the consumables whose targeting isn't in their config: Aura takes one
card without an edition (card.lua:1571-1574), and Sigil and Ouija work on the whole
hand without a selection.  :func:`jackdaw.env.action_space.get_consumable_target_info`
gives the Rust engine's ``(min, max, needs_targets)`` for a card of a run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jackdaw.engine.data import load

# ---------------------------------------------------------------------------
# ConsumableTargetSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumableTargetSpec:
    """Describes targeting requirements for a specific consumable.

    The RL agent uses this to know how many (and which) hand cards to
    select when using a consumable.
    """

    needs_card_targets: bool
    """True if the consumable requires hand card selection."""

    min_targets: int
    """Minimum cards to select (0 if no targets needed)."""

    max_targets: int
    """Maximum cards to select."""

    exact_targets: int | None
    """If not None, must select exactly this many cards."""

    target_filter: str
    """Filter for valid targets: "any", "no_edition", "all_hand", etc."""

    description: str
    """Human-readable description (for debugging)."""


# ---------------------------------------------------------------------------
# No-target sentinel
# ---------------------------------------------------------------------------

_NO_TARGETS = ConsumableTargetSpec(
    needs_card_targets=False,
    min_targets=0,
    max_targets=0,
    exact_targets=None,
    target_filter="any",
    description="no card targets",
)

# ---------------------------------------------------------------------------
# Special-case overrides (not derivable from config alone)
# ---------------------------------------------------------------------------

# Aura: needs exactly 1 highlighted card with no edition.
# Config is empty [] but can_use_consumeable has a hardcoded check.
_AURA_SPEC = ConsumableTargetSpec(
    needs_card_targets=True,
    min_targets=1,
    max_targets=1,
    exact_targets=1,
    target_filter="no_edition",
    description="Aura: 1 card without edition",
)

# Sigil/Ouija: affect ALL hand cards — no player selection.
_ALL_HAND_SPEC = ConsumableTargetSpec(
    needs_card_targets=False,
    min_targets=0,
    max_targets=0,
    exact_targets=None,
    target_filter="all_hand",
    description="targets all hand cards (no selection)",
)

_SPECIAL_SPECS: dict[str, ConsumableTargetSpec] = {
    "c_aura": _AURA_SPEC,
    "c_sigil": _ALL_HAND_SPEC,
    "c_ouija": _ALL_HAND_SPEC,
}


# ---------------------------------------------------------------------------
# get_consumable_target_spec
# ---------------------------------------------------------------------------


def _key(card: Any) -> str:
    return card if isinstance(card, str) else card.key


def get_consumable_target_spec(card: Any) -> ConsumableTargetSpec:
    """Determine targeting requirements for a consumable.

    Parameters
    ----------
    card:
        The consumable: a :class:`jackdaw.core.CardView` (``game.consumables[i]``,
        ``game.pack_cards[i]``) or its center key (``"c_death"``).

    Returns
    -------
    ConsumableTargetSpec
        The targeting constraints for this consumable.
    """
    key = _key(card)

    # 1. Check special-case overrides
    if key in _SPECIAL_SPECS:
        return _SPECIAL_SPECS[key]

    # 2. The center's config (an empty Lua table is a JSON list)
    cfg = load("centers").get(key, {}).get("config")
    if not isinstance(cfg, dict) or cfg.get("hand_type"):  # planets take no targets
        return _NO_TARGETS

    max_h = cfg.get("max_highlighted")
    if max_h:
        min_h = cfg.get("min_highlighted", 1)
        # mod_num overrides the effective max (used by can_use_consumeable)
        mod_num = cfg.get("mod_num", max_h)
        exact = mod_num if min_h == mod_num else None
        return ConsumableTargetSpec(
            needs_card_targets=True,
            min_targets=min_h,
            max_targets=mod_num,
            exact_targets=exact,
            target_filter="any",
            description=f"{key}: {min_h}-{mod_num} cards",
        )

    # 3. Fallback: no targets
    return _NO_TARGETS


# ---------------------------------------------------------------------------
# get_valid_target_cards
# ---------------------------------------------------------------------------


def get_valid_target_cards(card: Any, hand: Sequence[Any]) -> list[int]:
    """Return indices of hand cards that are valid targets for this consumable.

    Parameters
    ----------
    card:
        The consumable (a ``CardView`` or its center key).
    hand:
        Current hand cards (``game.hand``): anything with an ``edition``.

    Returns
    -------
    list[int]
        Indices into ``hand`` of valid target cards.
    """
    spec = get_consumable_target_spec(card)

    if not spec.needs_card_targets:
        return []

    if spec.target_filter == "no_edition":
        return [i for i, c in enumerate(hand) if not c.edition]

    # Default: all hand cards are valid
    return list(range(len(hand)))


# ---------------------------------------------------------------------------
# validate_card_targets
# ---------------------------------------------------------------------------


def validate_card_targets(card: Any, target_indices: tuple[int, ...], hand: Sequence[Any]) -> bool:
    """Validate that a specific target selection is legal for this consumable.

    Parameters
    ----------
    card:
        The consumable (a ``CardView`` or its center key).
    target_indices:
        Tuple of hand card indices selected as targets.
    hand:
        Current hand cards (``game.hand``).

    Returns
    -------
    bool
        True if the selection is valid.
    """
    spec = get_consumable_target_spec(card)

    # No-target consumables: must have empty selection
    if not spec.needs_card_targets:
        return len(target_indices) == 0

    n = len(target_indices)

    # Count check
    if n < spec.min_targets or n > spec.max_targets:
        return False

    if spec.exact_targets is not None and n != spec.exact_targets:
        return False

    # Bounds check: all indices must be valid
    if any(i < 0 or i >= len(hand) for i in target_indices):
        return False

    # Duplicates check
    if len(set(target_indices)) != n:
        return False

    # Filter check
    valid = set(get_valid_target_cards(card, hand))
    return all(i in valid for i in target_indices)
