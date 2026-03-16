"""Blind class — tracks chip target and boss effects for a round.

Mirrors the Lua ``Blind`` class from ``blind.lua``.  The Small and Big
blinds have no boss effects; boss blinds have unique debuff/modify/press
behaviors implemented as methods.

Source: blind.lua lines 5-751, blind chip target from misc_functions.lua:919.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jackdaw.engine.data.blind_scaling import get_blind_amount
from jackdaw.engine.data.prototypes import BLINDS


@dataclass
class Blind:
    """Active blind for the current round.

    Created via :meth:`create` which looks up the prototype and computes
    the chip target from ante, scaling, and ante_scaling.
    """

    key: str
    """P_BLINDS key (e.g. ``"bl_small"``, ``"bl_hook"``)."""

    name: str
    """Display name (e.g. ``"Small Blind"``, ``"The Hook"``)."""

    chips: int
    """Chip target to beat this blind."""

    mult: float
    """Blind multiplier (Small=1, Big=1.5, Boss=2+)."""

    dollars: int
    """Reward dollars for beating (0 if no_blind_reward)."""

    boss: bool
    """Whether this is a boss blind."""

    disabled: bool = False
    """Boss effect disabled (by Chicot or Luchador)."""

    triggered: bool = False
    """Whether the boss effect has activated this round."""

    # Boss-specific state
    debuff_config: dict[str, Any] = field(default_factory=dict)
    """Debuff config from prototype (suit, is_face, hand, etc.)."""

    hands_used: dict[str, bool] = field(default_factory=dict)
    """Tracks which hand types have been played (The Eye)."""

    only_hand: str | None = None
    """Locked hand type (The Mouth — only one hand type allowed)."""

    discards_sub: int | None = None
    """Discards subtracted by The Water."""

    hands_sub: int | None = None
    """Hands subtracted by The Needle."""

    @classmethod
    def create(
        cls,
        key: str,
        ante: int,
        scaling: int = 1,
        ante_scaling: float = 1.0,
        *,
        no_blind_reward: bool = False,
    ) -> Blind:
        """Create a blind with the correct chip target.

        Args:
            key: P_BLINDS prototype key.
            ante: Current ante number.
            scaling: Stake scaling level (1/2/3).
            ante_scaling: Deck ante scaling (1.0 normally, 2.0 Plasma).
            no_blind_reward: If True, dollars set to 0 (stake modifier).
        """
        proto = BLINDS[key]
        base = get_blind_amount(ante, scaling)
        chip_target = int(base * proto.mult * ante_scaling)

        dollars = proto.dollars
        if no_blind_reward:
            dollars = 0

        debuff = proto.debuff if isinstance(proto.debuff, dict) else {}

        return cls(
            key=key,
            name=proto.name,
            chips=chip_target,
            mult=proto.mult,
            dollars=dollars,
            boss=proto.boss is not None,
            debuff_config=debuff,
        )

    @classmethod
    def empty(cls) -> Blind:
        """Create an empty/null blind (between rounds)."""
        return cls(
            key="",
            name="",
            chips=0,
            mult=0,
            dollars=0,
            boss=False,
        )

    def debuff_card(
        self,
        card: Any,
        *,
        is_joker_area: bool = False,
        pareidolia: bool = False,
    ) -> None:
        """Set card.debuff based on boss blind effect.

        Matches ``Blind:debuff_card`` (blind.lua:624-652).

        The debuff is driven by ``self.debuff_config`` (from the prototype)
        plus special-case name checks for The Pillar and Verdant Leaf.

        Args:
            card: The Card to check and debuff.
            is_joker_area: If True, the card is in the joker area (not a
                playing card).  Most debuffs only apply to playing cards.
            pareidolia: If True, all cards count as face cards (for The Plant).
        """
        # Check prototype-driven debuffs (suit, face, pillar, value, nominal)
        # Note: in Lua, empty table {} is truthy.  All boss blinds have a
        # debuff table (even if empty), so this block runs for all bosses.
        # We check 'boss' to replicate this — non-boss blinds skip.
        if self.boss and not self.disabled and not is_joker_area:
            cfg = self.debuff_config

            # Suit debuff: The Club (Clubs), The Goad (Spades),
            # The Head (Hearts), The Window (Diamonds)
            if "suit" in cfg:
                if card.is_suit(cfg["suit"], bypass_debuff=True):
                    card.set_debuff(True)
                    return

            # Face card debuff: The Plant
            if cfg.get("is_face") == "face":
                if card.is_face(from_boss=True, pareidolia=pareidolia):
                    card.set_debuff(True)
                    return

            # The Pillar: debuffs cards played in previous hands this ante
            if self.name == "The Pillar":
                if card.ability.get("played_this_ante"):
                    card.set_debuff(True)
                    return

            # Value-based debuff (not used by vanilla blinds, but supported)
            if "value" in cfg and card.base is not None:
                if cfg["value"] == card.base.rank.value:
                    card.set_debuff(True)
                    return

            # Nominal-based debuff (not used by vanilla blinds, but supported)
            if "nominal" in cfg and card.base is not None:
                if cfg["nominal"] == card.base.nominal:
                    card.set_debuff(True)
                    return

        # Crimson Heart: debuffs a random joker (handled elsewhere, not here)
        if self.name == "Crimson Heart" and not self.disabled and is_joker_area:
            return  # joker debuff handled separately in drawn_to_hand

        # Verdant Leaf: debuffs ALL non-joker cards unconditionally
        if self.name == "Verdant Leaf" and not self.disabled and not is_joker_area:
            card.set_debuff(True)
            return

        # No debuff applies
        card.set_debuff(False)

    def get_type(self) -> str:
        """Return blind type string: ``'Small'``, ``'Big'``, or ``'Boss'``."""
        if self.name == "Small Blind":
            return "Small"
        if self.name == "Big Blind":
            return "Big"
        if self.name and self.name != "":
            return "Boss"
        return ""

    def __repr__(self) -> str:
        return (
            f"Blind({self.name!r}, chips={self.chips:,}, "
            f"boss={self.boss}, disabled={self.disabled})"
        )
