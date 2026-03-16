"""Card class — the central data structure for all game objects.

Every playing card, joker, tarot, planet, spectral, voucher, and booster
is a Card instance.  Mirrors the Lua ``Card`` object structure from
``card.lua``, keeping the ``ability`` dict untyped to match Lua's dynamic
table semantics.

Source: card.lua lines 5-77 (init), 97-145 (set_base), 223-342 (set_ability).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from jackdaw.engine.data.enums import Rank, Suit

# ---------------------------------------------------------------------------
# Module-level sort_id counter (matches G.sort_id in globals.lua)
# ---------------------------------------------------------------------------

_sort_id_counter: int = 0


def _next_sort_id() -> int:
    global _sort_id_counter  # noqa: PLW0603
    _sort_id_counter += 1
    return _sort_id_counter


def reset_sort_id_counter() -> None:
    """Reset to 0 (call at run start, matching G.sort_id = 0)."""
    global _sort_id_counter  # noqa: PLW0603
    _sort_id_counter = 0


# ---------------------------------------------------------------------------
# Face nominal values (from Card:set_base, card.lua:131-134)
# ---------------------------------------------------------------------------

_FACE_NOMINAL: dict[str, float] = {
    "Jack": 0.1,
    "Queen": 0.2,
    "King": 0.3,
    "Ace": 0.4,
}

_RANK_NOMINAL: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "Jack": 10, "Queen": 10, "King": 10, "Ace": 11,
}

_RANK_ID: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "Jack": 11, "Queen": 12, "King": 13, "Ace": 14,
}

_SUIT_NOMINAL: dict[str, float] = {
    "Diamonds": 0.01, "Clubs": 0.02, "Hearts": 0.03, "Spades": 0.04,
}

_SUIT_NOMINAL_ORIGINAL: dict[str, float] = {
    "Diamonds": 0.001, "Clubs": 0.002, "Hearts": 0.003, "Spades": 0.004,
}


# ---------------------------------------------------------------------------
# CardBase — playing card identity
# ---------------------------------------------------------------------------

@dataclass
class CardBase:
    """Base identity for a playing card (suit, rank, and derived numeric values).

    Corresponds to ``self.base`` in card.lua, populated by ``Card:set_base``.
    """

    suit: Suit
    rank: Rank  # the value string: "Ace", "King", ..., "2"
    id: int  # numeric rank id (2-14, Ace=14)
    nominal: int  # chip value
    suit_nominal: float  # suit tiebreaker (S=0.04, H=0.03, C=0.02, D=0.01)
    suit_nominal_original: float  # preserved original suit tiebreaker
    face_nominal: float  # 0.0 for non-face, 0.1-0.4 for J/Q/K/A
    original_value: Rank  # original rank before Strength tarot changes
    times_played: int = 0

    @staticmethod
    def from_card_key(card_key: str, suit: str, value: str) -> CardBase:
        """Build a CardBase from P_CARDS data, matching Card:set_base."""
        rank = Rank(value)
        suit_enum = Suit(suit)
        return CardBase(
            suit=suit_enum,
            rank=rank,
            id=_RANK_ID[value],
            nominal=_RANK_NOMINAL[value],
            suit_nominal=_SUIT_NOMINAL[suit],
            suit_nominal_original=_SUIT_NOMINAL_ORIGINAL[suit],
            face_nominal=_FACE_NOMINAL.get(value, 0.0),
            original_value=rank,
            times_played=0,
        )


# ---------------------------------------------------------------------------
# Card — the main object
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """A card in the game — playing card, joker, tarot, planet, spectral, etc.

    The ``ability`` dict is intentionally untyped (``dict[str, Any]``) to match
    Lua's dynamic table semantics.  It is populated by :meth:`set_ability` and
    freely mutated by joker effects during gameplay.
    """

    # Identity
    sort_id: int = field(default_factory=_next_sort_id)

    # Base (playing cards only — None for jokers/consumables/vouchers)
    base: CardBase | None = None

    # Center (prototype reference)
    center_key: str = "c_base"  # P_CENTERS key
    card_key: str | None = None  # P_CARDS key for playing cards

    # Mutable ability state
    ability: dict[str, Any] = field(default_factory=dict)

    # Modifiers
    edition: dict[str, bool] | None = None  # None, {"foil": True}, etc.
    seal: str | None = None  # None, "Red", "Blue", "Gold", "Purple"
    debuff: bool = False

    # Status
    playing_card: int | None = None  # index in playing_cards list
    facing: str = "front"  # "front" or "back"

    # Economy
    base_cost: int = 0
    cost: int = 0
    sell_cost: int = 0
    extra_cost: int = 0

    # Stickers
    eternal: bool = False
    perishable: bool = False
    perish_tally: int = 5  # rounds until perish
    rental: bool = False

    def set_base(self, card_key: str, suit: str, value: str) -> None:
        """Populate base fields from P_CARDS data, matching Card:set_base."""
        self.card_key = card_key
        self.base = CardBase.from_card_key(card_key, suit, value)

    def set_ability(self, center: dict[str, Any]) -> None:
        """Populate ability from a P_CENTERS prototype, matching Card:set_ability.

        Args:
            center: The prototype dict from P_CENTERS (or a loaded JSON entry).
                    Must have at least 'name', 'set', and 'config' fields.
        """
        config = center.get("config") or {}
        if isinstance(config, list):
            config = {}  # empty Lua table [] normalization

        old_perma_bonus = self.ability.get("perma_bonus", 0)
        old_forced_selection = self.ability.get("forced_selection")

        extra = config.get("extra")
        if isinstance(extra, (dict, list)):
            extra = copy.deepcopy(extra)

        self.ability = {
            "name": center.get("name", ""),
            "effect": center.get("effect", ""),
            "set": center.get("set", ""),
            "mult": config.get("mult", 0),
            "h_mult": config.get("h_mult", 0),
            "h_x_mult": config.get("h_x_mult", 0),
            "h_dollars": config.get("h_dollars", 0),
            "p_dollars": config.get("p_dollars", 0),
            "t_mult": config.get("t_mult", 0),
            "t_chips": config.get("t_chips", 0),
            "x_mult": config.get("Xmult", 1),
            "h_size": config.get("h_size", 0),
            "d_size": config.get("d_size", 0),
            "extra": extra,
            "extra_value": 0,
            "type": config.get("type", ""),
            "order": center.get("order"),
            "forced_selection": old_forced_selection,
            "perma_bonus": old_perma_bonus,
        }

        self.ability["bonus"] = self.ability.get("bonus", 0) + config.get("bonus", 0)

        if center.get("consumeable"):
            self.ability["consumeable"] = config

        self.center_key = center.get("key", self.center_key)
        self.base_cost = center.get("cost", 1)

    def set_edition(self, edition: dict[str, bool] | None) -> None:
        """Set the card's edition."""
        self.edition = edition

    def set_seal(self, seal: str | None) -> None:
        """Set the card's seal."""
        self.seal = seal

    def set_eternal(self, eternal: bool) -> None:
        self.eternal = eternal

    def set_perishable(self, perishable: bool) -> None:
        self.perishable = perishable
        if perishable:
            self.perish_tally = 5

    def set_rental(self, rental: bool) -> None:
        self.rental = rental

    def set_debuff(self, should_debuff: bool) -> None:
        self.debuff = should_debuff

    def is_face(self) -> bool:
        """Check if this is a face card (J/Q/K), matching Card:is_face.

        Note: Pareidolia joker makes all cards face cards — that check
        happens at the game logic layer, not here.
        """
        if self.debuff:
            return False
        if self.base is None:
            return False
        return self.base.id in (11, 12, 13)

    def get_id(self) -> int:
        """Get the rank id for hand evaluation, matching Card:get_id.

        Stone Cards return a random negative number in Lua; here we
        return -1 as a deterministic placeholder (the actual randomness
        is handled at the game logic layer).
        """
        if self.ability.get("effect") == "Stone Card":
            return -1
        if self.base is None:
            return 0
        return self.base.id

    def get_chip_bonus(self) -> int:
        """Chip bonus when scored, matching Card:get_chip_bonus."""
        if self.debuff:
            return 0
        if self.ability.get("effect") == "Stone Card":
            return self.ability.get("bonus", 0) + self.ability.get("perma_bonus", 0)
        if self.base is None:
            return 0
        return self.base.nominal + self.ability.get("bonus", 0) + self.ability.get("perma_bonus", 0)

    def get_chip_mult(self) -> float:
        """Mult bonus when scored, matching Card:get_chip_mult."""
        if self.debuff:
            return 0
        return self.ability.get("mult", 0)

    def get_chip_x_mult(self) -> float:
        """X-mult bonus when scored, matching Card:get_chip_x_mult."""
        if self.debuff:
            return 0
        xm = self.ability.get("x_mult", 1)
        if xm <= 1:
            return 0
        return xm

    def __repr__(self) -> str:
        if self.base:
            return f"Card({self.base.rank} of {self.base.suit}, center={self.center_key!r})"
        return f"Card(center={self.center_key!r}, name={self.ability.get('name', '?')!r})"
