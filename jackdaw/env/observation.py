"""Entity-based observation encoding for neural network input (encoder v1).

A state is encoded as structured numeric arrays: a fixed-size global context and
variable-length entity lists (no artificial caps), following the AlphaStar entity
encoding pattern.  Each entity type gets a fixed-width feature vector; the number of
entities varies per timestep.  The Rust engine computes it
(``crates/jackdaw-env/src/observation.rs``); :func:`encode_observation` returns it for a
:class:`jackdaw.core.Game`.

Playing cards (hand and pack rows, ``D_PLAYING_CARD`` = 15): rank id /14, suit /3,
chip value /11, enhancement /8, edition /4, seal /4, debuffed, face down, face card,
scoring, bonus chips (log), times played (log), position /20, part of the best hand,
padding.  A face-down card encodes only ``face down`` and its position.

Jokers (``D_JOKER`` = 15): center id / ``NUM_CENTER_KEYS``, rarity proxy, edition /4,
sell value (log), eternal, perishable, perish tally /5, rental, debuffed, position /20,
mult (log), x_mult, chips (log), extra (log), condition met.  A face-down joker (The
Amber Acorn) encodes only its position.

Consumables (``D_CONSUMABLE`` = 7): center id, set /9, sell value (log), usable now,
needs targets, max targets /5, min targets /5.

Shop items (``D_SHOP`` = 9): center id, set /9, cost (log), affordable, has a slot,
edition /4, eternal, perishable, rental.

The global context (``D_GLOBAL`` = 235)::

    [0:6]      phase one-hot (6)
    [6:10]     blind_on_deck one-hot (4)
    [10:30]    scalar features (20)
    [30:90]    hand_levels 12×5 (60)
    [90:122]   vouchers_owned binary (32)
    [122:130]  blind_effect features (8)
    [130:133]  round_position one-hot (3)
    [133:135]  round_progress (2)
    [135:159]  tags binary (24)
    [159:211]  discard_pile histogram (52)
    [211:223]  hand_type_indicators (12) — binary, which hand types current hand forms
    [223:226]  base_score (3) — log(chips), log(mult), log(chips×mult) for best hand
    [226]      score_to_blind_ratio (1) — log(base_score)/log(blind_chips), clamped
    [227]      flush_proximity (1) — max suit count in hand / 5 (continuous)
    [228]      straight_proximity (1) — max consecutive run length / 5 (continuous)
    [229]      interest_earned (1) — current interest bracket, normalized
    [230]      flush_draw_outs (1) — P(drawing suit-completing card) from deck
    [231]      straight_draw_outs (1) — P(drawing rank-completing card) from deck
    [232]      debuff_hand_fraction (1) — face-up debuffed hand cards / hand size
    [233]      spendable_above_interest (1) — money above interest floor, log-scaled
    [234]      urgency (1) — blind_remaining / (base_score × hands_left), clamped

The only external dependency is numpy (standard for RL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from jackdaw.engine.data import load
from jackdaw.env.game_spec import GameObservation

if TYPE_CHECKING:
    from jackdaw.core import Game

# ---------------------------------------------------------------------------
# Center key → integer ID mapping
# ---------------------------------------------------------------------------

# Contiguous ids from 1 in sorted key order; 0 is unknown/padding.
_CENTER_KEY_TO_ID: dict[str, int] = {
    key: i for i, key in enumerate(sorted(load("centers")), start=1)
}

# Number of distinct center keys (for embedding table sizing)
NUM_CENTER_KEYS: int = len(_CENTER_KEY_TO_ID)  # ~299


def center_key_id(key: str) -> int:
    """Map a center_key to its integer ID (0 for unknown).  The center-id features hold
    ``center_key_id(key) / NUM_CENTER_KEYS``."""
    return _CENTER_KEY_TO_ID.get(key, 0)


# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

NUM_HAND_TYPES: int = len(load("hands"))  # 12
NUM_VOUCHERS: int = 32
NUM_TAGS: int = 24
NUM_SUITS: int = 4
NUM_RANKS: int = 13
D_DISCARD_HISTOGRAM: int = NUM_SUITS * NUM_RANKS  # 52

D_PLAYING_CARD: int = 15
D_JOKER: int = 15
D_CONSUMABLE: int = 7
D_SHOP: int = 9
_D_BASE: int = 6 + 4 + 20 + NUM_HAND_TYPES * 5  # 90
_D_OLD_GLOBAL: int = _D_BASE + NUM_VOUCHERS + 8 + 3 + 2 + NUM_TAGS + D_DISCARD_HISTOGRAM  # 211
_D_STRATEGIC: int = NUM_HAND_TYPES + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # 24
D_GLOBAL: int = _D_OLD_GLOBAL + _D_STRATEGIC  # 235


# ---------------------------------------------------------------------------
# Observation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """Entity-based observation — variable-length entity lists + fixed global.

    Each entity array has shape ``(N, D)`` where N varies per timestep.
    Empty areas produce shape ``(0, D)`` arrays.  All arrays are float32.
    """

    global_context: np.ndarray  # (D_GLOBAL,)
    hand_cards: np.ndarray  # (N_hand, D_PLAYING_CARD)
    jokers: np.ndarray  # (N_joker, D_JOKER)
    consumables: np.ndarray  # (N_cons, D_CONSUMABLE)
    shop_cards: np.ndarray  # (N_shop, D_SHOP)
    pack_cards: np.ndarray  # (N_pack, D_PLAYING_CARD)

    def to_game_observation(self) -> GameObservation:
        """Convert to a game-agnostic :class:`GameObservation`."""
        return GameObservation(
            global_context=self.global_context,
            entities={
                "hand_card": self.hand_cards,
                "joker": self.jokers,
                "consumable": self.consumables,
                "shop_item": self.shop_cards,
                "pack_card": self.pack_cards,
            },
        )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def encode_observation(game: Game) -> Observation:
    """The observation of a run's current state.

    The arrays are writable, C-contiguous views sharing one buffer per call
    (:func:`jackdaw.env.rust_observation.encode_game`).
    """
    from jackdaw.env.rust_observation import encode_game

    return encode_game(game)
