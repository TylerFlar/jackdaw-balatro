"""Card creation functions matching the Balatro source's card factory patterns.

Provides typed constructors for playing cards, jokers, consumables, and
vouchers, plus the ``card_from_control`` function matching
``misc_functions.lua:1625`` used during deck building.

Source references:
  - card_from_control: misc_functions.lua:1625
  - create_card: common_events.lua:2082
  - Card:init, Card:set_base, Card:set_ability: card.lua:5-342
"""

from __future__ import annotations

from jackdaw.engine.card import Card
from jackdaw.engine.data.enums import Rank, Suit

# ---------------------------------------------------------------------------
# Suit / rank letter mappings (used by card_from_control and deck building)
# ---------------------------------------------------------------------------

SUIT_LETTER: dict[str, Suit] = {
    "H": Suit.HEARTS,
    "C": Suit.CLUBS,
    "D": Suit.DIAMONDS,
    "S": Suit.SPADES,
}

RANK_LETTER: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}

# Reverse mappings
SUIT_TO_LETTER: dict[Suit, str] = {v: k for k, v in SUIT_LETTER.items()}
RANK_TO_LETTER: dict[Rank, str] = {v: k for k, v in RANK_LETTER.items()}


def _card_key(suit: Suit, rank: Rank) -> str:
    """Build a P_CARDS key like ``'S_A'`` or ``'H_K'``."""
    return f"{SUIT_TO_LETTER[suit]}_{RANK_TO_LETTER[rank]}"


# ---------------------------------------------------------------------------
# Playing card creation
# ---------------------------------------------------------------------------

def create_playing_card(
    suit: Suit,
    rank: Rank,
    enhancement: str = "c_base",
    edition: dict[str, bool] | None = None,
    seal: str | None = None,
    *,
    playing_card_index: int | None = None,
    hands_played: int = 0,
) -> Card:
    """Create a playing card (goes into the deck).

    Mirrors the card creation in ``card_from_control`` (misc_functions.lua:1625)
    and ``Card:init`` (card.lua:5).

    Args:
        suit: Card suit.
        rank: Card rank.
        enhancement: P_CENTERS key for the enhancement center.
            ``"c_base"`` for a normal card, ``"m_glass"`` for Glass, etc.
        edition: Edition dict (``{"foil": True}``) or None.
        seal: Seal string (``"Gold"``, ``"Red"``, etc.) or None.
        playing_card_index: Index in ``G.playing_cards`` list.
        hands_played: Current ``G.GAME.hands_played`` for post-init fields.

    Returns:
        A fully initialised playing card.
    """
    card = Card()
    key = _card_key(suit, rank)
    card.set_base(key, suit.value, rank.value)
    card.set_ability(enhancement, hands_played=hands_played)
    card.playing_card = playing_card_index
    if edition:
        card.set_edition(edition)
    if seal:
        card.set_seal(seal)
    return card


# ---------------------------------------------------------------------------
# Joker creation
# ---------------------------------------------------------------------------

def create_joker(
    key: str,
    edition: dict[str, bool] | None = None,
    *,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
    hands_played: int = 0,
) -> Card:
    """Create a joker card from a P_CENTERS key (e.g. ``"j_joker"``).

    Args:
        key: P_CENTERS joker key.
        edition: Edition dict or None.
        eternal: Whether the joker has the Eternal sticker.
        perishable: Whether the joker has the Perishable sticker.
        rental: Whether the joker has the Rental sticker.
        hands_played: Current hands_played for post-init fields.
    """
    card = Card()
    card.set_ability(key, hands_played=hands_played)
    if edition:
        card.set_edition(edition)
    if eternal:
        card.set_eternal(True)
    if perishable:
        card.set_perishable(True)
    if rental:
        card.set_rental(True)
    return card


# ---------------------------------------------------------------------------
# Consumable creation (tarots, planets, spectrals)
# ---------------------------------------------------------------------------

def create_consumable(key: str, *, hands_played: int = 0) -> Card:
    """Create a tarot, planet, or spectral card from a P_CENTERS key.

    Args:
        key: P_CENTERS consumable key (e.g. ``"c_magician"``, ``"c_pluto"``).
        hands_played: Current hands_played for post-init fields.
    """
    card = Card()
    card.set_ability(key, hands_played=hands_played)
    return card


# ---------------------------------------------------------------------------
# Voucher creation
# ---------------------------------------------------------------------------

def create_voucher(key: str) -> Card:
    """Create a voucher card from a P_CENTERS key (e.g. ``"v_overstock_norm"``)."""
    card = Card()
    card.set_ability(key)
    return card


# ---------------------------------------------------------------------------
# card_from_control — deck building helper
# ---------------------------------------------------------------------------

def card_from_control(
    control: dict,
    *,
    playing_card_index: int | None = None,
    hands_played: int = 0,
) -> Card:
    """Create a playing card from a control dict.

    Matches ``card_from_control`` in ``misc_functions.lua:1625``.

    Control dict fields:
        - ``s``: suit letter (``'H'``/``'C'``/``'D'``/``'S'``)
        - ``r``: rank letter (``'2'``-``'9'``/``'T'``/``'J'``/``'Q'``/``'K'``/``'A'``)
        - ``e``: enhancement center key (e.g. ``'m_glass'``), defaults to ``'c_base'``
        - ``d``: edition key (e.g. ``'foil'``/``'holo'``/``'polychrome'``), or None
        - ``g``: seal (e.g. ``'Gold'``/``'Red'``), or None

    Args:
        control: The control dict.
        playing_card_index: Index in the playing_cards list.
        hands_played: Current hands_played for post-init fields.
    """
    suit = SUIT_LETTER[control["s"]]
    rank = RANK_LETTER[control["r"]]
    enhancement = control.get("e") or "c_base"
    edition_key = control.get("d")
    seal = control.get("g")

    edition = {edition_key: True} if edition_key else None

    return create_playing_card(
        suit=suit,
        rank=rank,
        enhancement=enhancement,
        edition=edition,
        seal=seal,
        playing_card_index=playing_card_index,
        hands_played=hands_played,
    )
