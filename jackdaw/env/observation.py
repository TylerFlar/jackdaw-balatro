"""Entity-based observation encoding for neural network input.

Converts the raw engine ``game_state`` dict into structured numeric arrays.
Uses variable-length entity lists (no artificial caps) following the AlphaStar
entity encoding pattern.  Each entity type gets a fixed-width feature vector;
the number of entities varies per timestep.

The only external dependency is numpy (standard for RL).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card import Card
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.hand_levels import HandLevels

# ---------------------------------------------------------------------------
# Center key → integer ID mapping (loaded once at module init)
# ---------------------------------------------------------------------------

_CENTERS_PATH = Path(__file__).resolve().parent.parent / "engine" / "data" / "centers.json"

_CENTER_KEY_TO_ID: dict[str, int] = {}  # 0 = unknown/padding


def _load_center_ids() -> dict[str, int]:
    """Build a contiguous int mapping from centers.json keys."""
    with open(_CENTERS_PATH) as f:
        data: dict[str, Any] = json.load(f)
    mapping: dict[str, int] = {}
    for i, key in enumerate(sorted(data.keys()), start=1):
        mapping[key] = i
    return mapping


_CENTER_KEY_TO_ID = _load_center_ids()

# Number of distinct center keys (for embedding table sizing)
NUM_CENTER_KEYS: int = len(_CENTER_KEY_TO_ID)  # ~299


def center_key_id(key: str) -> int:
    """Map a center_key to its integer ID (0 for unknown)."""
    return _CENTER_KEY_TO_ID.get(key, 0)


# ---------------------------------------------------------------------------
# Categorical index tables
# ---------------------------------------------------------------------------

_SUIT_IDX: dict[str, int] = {
    "Hearts": 0,
    "Diamonds": 1,
    "Clubs": 2,
    "Spades": 3,
}

_RANK_IDX: dict[str, int] = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "Jack": 9,
    "Queen": 10,
    "King": 11,
    "Ace": 12,
}

_RANK_CHIPS: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 10,
    "Queen": 10,
    "King": 10,
    "Ace": 11,
}

_ENHANCEMENT_IDX: dict[str, int] = {
    "none": 0,
    "c_base": 0,
    "m_bonus": 1,
    "m_mult": 2,
    "m_wild": 3,
    "m_glass": 4,
    "m_steel": 5,
    "m_stone": 6,
    "m_gold": 7,
    "m_lucky": 8,
}

_EDITION_IDX_MAP: dict[str, int] = {
    "none": 0,
    "foil": 1,
    "holo": 2,
    "polychrome": 3,
    "negative": 4,
}

_SEAL_IDX: dict[str | None, int] = {
    None: 0,
    "none": 0,
    "Gold": 1,
    "Red": 2,
    "Blue": 3,
    "Purple": 4,
}

_PHASE_IDX: dict[str, int] = {
    GamePhase.BLIND_SELECT: 0,
    GamePhase.SELECTING_HAND: 1,
    GamePhase.ROUND_EVAL: 2,
    GamePhase.SHOP: 3,
    GamePhase.PACK_OPENING: 4,
    GamePhase.GAME_OVER: 5,
}

_BLIND_ON_DECK_IDX: dict[str | None, int] = {
    None: 0,
    "Small": 1,
    "Big": 2,
    "Boss": 3,
}

_CARD_SET_IDX: dict[str, int] = {
    "": 0,
    "Default": 0,
    "Enhanced": 1,
    "Joker": 2,
    "Tarot": 3,
    "Planet": 4,
    "Spectral": 5,
    "Voucher": 6,
    "Booster": 7,
    "Back": 8,
    "Edition": 9,
}

# Hand types in canonical order for the global context vector
_HAND_TYPES: list[HandType] = list(HandType)
NUM_HAND_TYPES: int = len(_HAND_TYPES)  # 12

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

D_PLAYING_CARD: int = 14
D_JOKER: int = 15
D_CONSUMABLE: int = 7
D_SHOP: int = 9
# Global: 6 phase + 4 blind_on_deck + 20 scalars + 12*5 hand_levels = 90
D_GLOBAL: int = 6 + 4 + 20 + NUM_HAND_TYPES * 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LOG2 = math.log(2.0)


def _log_scale(x: float) -> float:
    """Log-scale large values: sign(x) * log2(1 + |x|)."""
    if x >= 0:
        return math.log2(1.0 + x)
    return -math.log2(1.0 - x)


def _log_scale_arr(arr: np.ndarray) -> np.ndarray:
    """Vectorized log-scale: sign(x) * log2(1 + |x|)."""
    signs = np.sign(arr)
    return signs * np.log2(1.0 + np.abs(arr))


def _edition_idx(edition: dict[str, Any] | None) -> int:
    """Map edition dict to integer index."""
    if not edition:
        return 0
    if edition.get("foil"):
        return 1
    if edition.get("holo"):
        return 2
    if edition.get("polychrome"):
        return 3
    if edition.get("negative"):
        return 4
    return 0


def _one_hot(idx: int, n: int) -> list[float]:
    """Return a one-hot list of length n with 1.0 at idx."""
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Entity encoding functions
# ---------------------------------------------------------------------------


def encode_playing_card(
    card: Card,
    position: int,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a playing card as a fixed-size feature vector.

    Returns shape ``(D_PLAYING_CARD,)`` float32 array.

    Features (14):
        0: rank_id (ordinal 2-14, normalized /14)
        1: suit (ordinal 0-3, normalized /3)
        2: chip_value (normalized /11)
        3: enhancement (ordinal 0-8, normalized /8)
        4: edition (ordinal 0-4, normalized /4)
        5: seal (ordinal 0-4, normalized /4)
        6: debuffed (0/1)
        7: face_down (0/1)
        8: is_face_card (0/1)
        9: is_scoring (0/1 — all cards if splash active)
       10: bonus_chips (log-scaled)
       11: times_played (log-scaled)
       12: position_in_hand (normalized /20)
       13: reserved/padding (0)
    """
    v = np.zeros(D_PLAYING_CARD, dtype=np.float32)

    face_down = card.facing == "back"

    if face_down or card.base is None:
        # Card exists but attributes hidden
        v[7] = 1.0
        v[12] = position / 20.0
        return v

    rank_str = card.base.rank.value
    suit_str = card.base.suit.value

    v[0] = card.base.id / 14.0
    v[1] = _SUIT_IDX.get(suit_str, 0) / 3.0
    v[2] = _RANK_CHIPS.get(rank_str, 0) / 11.0
    v[3] = _ENHANCEMENT_IDX.get(card.center_key, 0) / 8.0
    v[4] = _edition_idx(card.edition) / 4.0
    v[5] = _SEAL_IDX.get(card.seal, 0) / 4.0
    v[6] = float(card.debuff)
    v[7] = 0.0  # not face down
    v[8] = float(card.base.id in (11, 12, 13))  # J, Q, K
    v[9] = float(bool(gs.get("splash", 0)) or not card.debuff)
    v[10] = _log_scale(card.ability.get("bonus", 0) + card.ability.get("perma_bonus", 0))
    v[11] = _log_scale(card.base.times_played)
    v[12] = position / 20.0

    return v


def encode_joker(
    card: Card,
    position: int,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a joker as a fixed-size feature vector.

    Returns shape ``(D_JOKER,)`` float32 array.

    Features (15):
        0: center_key_id (normalized / NUM_CENTER_KEYS)
        1: rarity (ordinal 1-4, normalized /4)
        2: edition (ordinal 0-4, normalized /4)
        3: sell_value (log-scaled)
        4: eternal (0/1)
        5: perishable (0/1)
        6: perish_tally (normalized /5)
        7: rental (0/1)
        8: debuffed (0/1)
        9: position (normalized /20)
       10: ability_mult (log-scaled)
       11: ability_x_mult (raw — typically 1-5)
       12: ability_chips (log-scaled)
       13: ability_extra (log-scaled — meaning varies per joker)
       14: condition_met (0/1 — simplified: not debuffed)
    """
    v = np.zeros(D_JOKER, dtype=np.float32)

    v[0] = center_key_id(card.center_key) / max(NUM_CENTER_KEYS, 1)
    card.ability.get("extra", {})
    # Rarity from center data — ability dict doesn't store it, use cost heuristic
    # or look it up. For now use a simple cost-based proxy.
    v[1] = min(card.base_cost / 20.0, 1.0) if card.base_cost else 0.0
    v[2] = _edition_idx(card.edition) / 4.0
    v[3] = _log_scale(card.sell_cost)
    v[4] = float(card.eternal)
    v[5] = float(card.perishable)
    v[6] = card.perish_tally / 5.0 if card.perishable else 1.0
    v[7] = float(card.rental)
    v[8] = float(card.debuff)
    v[9] = position / 20.0
    v[10] = _log_scale(card.ability.get("mult", 0) + card.ability.get("t_mult", 0))
    v[11] = card.ability.get("x_mult", 1.0)
    v[12] = _log_scale(card.ability.get("t_chips", 0) + card.ability.get("bonus", 0))

    # ability_extra — flatten to a single float
    extra = card.ability.get("extra")
    if isinstance(extra, (int, float)):
        v[13] = _log_scale(extra)
    elif isinstance(extra, dict):
        # Sum numeric values as a generic signal
        total = 0.0
        for val in extra.values():
            if isinstance(val, (int, float)):
                total += val
        v[13] = _log_scale(total)

    v[14] = float(not card.debuff)

    return v


def encode_consumable(
    card: Card,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a consumable as a fixed-size feature vector.

    Returns shape ``(D_CONSUMABLE,)`` float32 array.

    Features (7):
        0: center_key_id (normalized)
        1: card_set (ordinal 0-9, normalized /9)
        2: sell_value (log-scaled)
        3: can_use (0/1)
        4: needs_targets (0/1)
        5: max_targets (normalized /5)
        6: min_targets (normalized /5)
    """
    v = np.zeros(D_CONSUMABLE, dtype=np.float32)

    v[0] = center_key_id(card.center_key) / max(NUM_CENTER_KEYS, 1)
    card_set = card.ability.get("set", "")
    v[1] = _CARD_SET_IDX.get(card_set, 0) / 9.0
    v[2] = _log_scale(card.sell_cost)

    # can_use — check without highlighting (conservative: some need targets)
    from jackdaw.engine.consumables import can_use_consumable

    hand_cards: list[Card] = gs.get("hand", [])
    jokers: list[Card] = gs.get("jokers", [])
    consumables: list[Card] = gs.get("consumables", [])
    can_use = can_use_consumable(
        card,
        hand_cards=hand_cards,
        jokers=jokers,
        consumables=consumables,
        consumable_limit=gs.get("consumable_slots", 2),
        joker_limit=gs.get("joker_slots", 5),
    )
    v[3] = float(can_use)

    # Targeting info from consumable config
    cfg = card.ability.get("consumeable", {})
    if not isinstance(cfg, dict):
        cfg = {}
    max_h = cfg.get("max_highlighted", 0)
    min_h = cfg.get("min_highlighted", 0)
    v[4] = float(max_h > 0)
    v[5] = min(max_h, 5) / 5.0
    v[6] = min(min_h, 5) / 5.0

    return v


def encode_shop_item(
    card: Card,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a shop item as a fixed-size feature vector.

    Returns shape ``(D_SHOP,)`` float32 array.

    Features (9):
        0: center_key_id (normalized)
        1: card_set (ordinal, normalized /9)
        2: cost (log-scaled)
        3: affordable (0/1)
        4: has_slot (0/1)
        5: edition (ordinal 0-4, normalized /4)
        6: eternal (0/1)
        7: perishable (0/1)
        8: rental (0/1)
    """
    v = np.zeros(D_SHOP, dtype=np.float32)

    v[0] = center_key_id(card.center_key) / max(NUM_CENTER_KEYS, 1)
    card_set = card.ability.get("set", "")
    v[1] = _CARD_SET_IDX.get(card_set, 0) / 9.0
    v[2] = _log_scale(card.cost)

    dollars = gs.get("dollars", 0)
    v[3] = float(card.cost <= dollars)

    # Has slot check
    jokers: list[Card] = gs.get("jokers", [])
    joker_slots: int = gs.get("joker_slots", 5)
    consumables: list[Card] = gs.get("consumables", [])
    consumable_slots: int = gs.get("consumable_slots", 2)

    if card_set == "Joker":
        is_negative = isinstance(card.edition, dict) and bool(card.edition.get("negative"))
        v[4] = float(len(jokers) < joker_slots or is_negative)
    elif card_set in ("Tarot", "Planet", "Spectral"):
        v[4] = float(len(consumables) < consumable_slots)
    else:
        v[4] = 1.0  # Vouchers, boosters, playing cards always "have slot"

    v[5] = _edition_idx(card.edition) / 4.0
    v[6] = float(card.eternal)
    v[7] = float(card.perishable)
    v[8] = float(card.rental)

    return v


def encode_global_context(gs: dict[str, Any]) -> np.ndarray:
    """Encode the fixed-size global context vector.

    Returns shape ``(D_GLOBAL,)`` float32 array.

    Layout:
        [0:6]    phase one-hot (6)
        [6:10]   blind_on_deck one-hot (4)
        [10:30]  scalar features (20)
        [30:90]  hand_levels 12 x (level, chips, mult, played, visible) = 60
    """
    v = np.zeros(D_GLOBAL, dtype=np.float32)

    # Phase one-hot [0:6]
    phase = gs.get("phase", GamePhase.GAME_OVER)
    if isinstance(phase, str):
        try:
            phase = GamePhase(phase)
        except ValueError:
            phase = GamePhase.GAME_OVER
    phase_idx = _PHASE_IDX.get(phase, 5)
    v[phase_idx] = 1.0

    # Blind on deck one-hot [6:10]
    bod = gs.get("blind_on_deck")
    bod_idx = _BLIND_ON_DECK_IDX.get(bod, 0)
    v[6 + bod_idx] = 1.0

    # Scalar features [10:30]
    rr = gs.get("round_resets", {})
    cr = gs.get("current_round", {})
    blind = gs.get("blind")
    blind_chips = getattr(blind, "chips", 0) if blind else 0
    chips = gs.get("chips", 0)

    i = 10
    v[i] = rr.get("ante", 1) / 8.0
    i += 1  # 10: ante
    v[i] = gs.get("round", 0) / 30.0
    i += 1  # 11: round
    v[i] = _log_scale(gs.get("dollars", 0))
    i += 1  # 12: dollars
    v[i] = cr.get("hands_left", 0) / 10.0
    i += 1  # 13: hands_left
    v[i] = cr.get("discards_left", 0) / 10.0
    i += 1  # 14: discards_left
    v[i] = gs.get("hand_size", 8) / 15.0
    i += 1  # 15: hand_size
    v[i] = gs.get("joker_slots", 5) / 10.0
    i += 1  # 16: joker_slots
    v[i] = gs.get("consumable_slots", 2) / 5.0
    i += 1  # 17: consumable_slots
    v[i] = _log_scale(blind_chips)
    i += 1  # 18: blind_chips_required
    v[i] = _log_scale(chips)
    i += 1  # 19: chips_scored
    # Score fraction (clamp to avoid div-by-zero)
    v[i] = min(chips / max(blind_chips, 1), 10.0) / 10.0
    i += 1  # 20: score_fraction
    v[i] = cr.get("reroll_cost", 5) / 10.0
    i += 1  # 21: reroll_cost
    v[i] = min(cr.get("free_rerolls", 0), 5) / 5.0
    i += 1  # 22: free_rerolls
    v[i] = gs.get("interest_cap", 25) / 100.0
    i += 1  # 23: interest_cap
    v[i] = gs.get("discount_percent", 0) / 50.0
    i += 1  # 24: discount_percent
    v[i] = gs.get("skips", 0) / 10.0
    i += 1  # 25: skips
    # Boss blind key ID
    blind_key = getattr(blind, "key", "") if blind else ""
    v[i] = center_key_id(blind_key) / max(NUM_CENTER_KEYS, 1)
    i += 1  # 26: boss_blind_key_id
    v[i] = _log_scale(len(gs.get("deck", [])))
    i += 1  # 27: deck_cards_remaining
    v[i] = _log_scale(len(gs.get("discard_pile", [])))
    i += 1  # 28: discard_pile_size
    # Meta joker flags
    flags = (
        float(bool(gs.get("four_fingers", 0)))
        + float(bool(gs.get("shortcut", 0))) * 2
        + float(bool(gs.get("smeared", 0))) * 4
        + float(bool(gs.get("splash", 0))) * 8
    )
    v[i] = flags / 15.0
    i += 1  # 29: meta_flags_packed

    # Hand levels [30:90] — 12 hand types × 5 features
    hand_levels: HandLevels | None = gs.get("hand_levels")
    base_idx = 30
    for j, ht in enumerate(_HAND_TYPES):
        offset = base_idx + j * 5
        if hand_levels is not None:
            hs = hand_levels.get_state(ht)
            v[offset + 0] = hs.level / 20.0
            v[offset + 1] = _log_scale(hs.chips)
            v[offset + 2] = _log_scale(hs.mult)
            v[offset + 3] = _log_scale(hs.played)
            v[offset + 4] = float(hs.visible)

    return v


# ---------------------------------------------------------------------------
# Batch encoding functions (hot path optimization)
# ---------------------------------------------------------------------------

# Pre-allocated buffers keyed by max entity count, reused across calls.
# Thread-local would be needed for multi-threaded use, but RL envs are
# single-threaded per worker.
_HAND_BUF: np.ndarray | None = None
_JOKER_BUF: np.ndarray | None = None

_EMPTY_PLAYING = np.zeros((0, D_PLAYING_CARD), dtype=np.float32)
_EMPTY_JOKER = np.zeros((0, D_JOKER), dtype=np.float32)
_EMPTY_CONSUMABLE = np.zeros((0, D_CONSUMABLE), dtype=np.float32)
_EMPTY_SHOP = np.zeros((0, D_SHOP), dtype=np.float32)


def _get_hand_buf(n: int) -> np.ndarray:
    """Return a pre-allocated (n, D_PLAYING_CARD) buffer, reusing when possible."""
    global _HAND_BUF
    if _HAND_BUF is None or _HAND_BUF.shape[0] < n:
        _HAND_BUF = np.zeros((max(n, 16), D_PLAYING_CARD), dtype=np.float32)
    buf = _HAND_BUF[:n]
    buf[:] = 0.0
    return buf


def _get_joker_buf(n: int) -> np.ndarray:
    """Return a pre-allocated (n, D_JOKER) buffer, reusing when possible."""
    global _JOKER_BUF
    if _JOKER_BUF is None or _JOKER_BUF.shape[0] < n:
        _JOKER_BUF = np.zeros((max(n, 8), D_JOKER), dtype=np.float32)
    buf = _JOKER_BUF[:n]
    buf[:] = 0.0
    return buf


def encode_playing_cards_batch(
    cards: list[Card],
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode multiple playing cards into a single array.

    Returns shape ``(len(cards), D_PLAYING_CARD)`` float32 array.
    Uses pre-allocated buffer to avoid per-call allocation.
    """
    n = len(cards)
    if n == 0:
        return _EMPTY_PLAYING
    buf = _get_hand_buf(n)
    splash = bool(gs.get("splash", 0))

    for i, card in enumerate(cards):
        row = buf[i]  # already zeroed
        face_down = card.facing == "back"
        if face_down or card.base is None:
            row[7] = 1.0
            row[12] = i / 20.0
            continue

        rank_str = card.base.rank.value
        suit_str = card.base.suit.value

        row[0] = card.base.id / 14.0
        row[1] = _SUIT_IDX.get(suit_str, 0) / 3.0
        row[2] = _RANK_CHIPS.get(rank_str, 0) / 11.0
        row[3] = _ENHANCEMENT_IDX.get(card.center_key, 0) / 8.0
        row[4] = _edition_idx(card.edition) / 4.0
        row[5] = _SEAL_IDX.get(card.seal, 0) / 4.0
        row[6] = float(card.debuff)
        # row[7] = 0.0 already
        row[8] = float(card.base.id in (11, 12, 13))
        row[9] = float(splash or not card.debuff)
        row[10] = _log_scale(
            card.ability.get("bonus", 0) + card.ability.get("perma_bonus", 0)
        )
        row[11] = _log_scale(card.base.times_played)
        row[12] = i / 20.0

    # Return a copy so the buffer can be reused next call
    return buf.copy()


def encode_jokers_batch(
    jokers: list[Card],
    gs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Encode multiple jokers into a single array.

    Parameters
    ----------
    jokers : list[Card]
        Joker cards to encode.
    gs : dict, optional
        Game state dict (reserved for future use).

    Returns shape ``(len(jokers), D_JOKER)`` float32 array.
    """
    n = len(jokers)
    if n == 0:
        return _EMPTY_JOKER
    buf = _get_joker_buf(n)
    nck = max(NUM_CENTER_KEYS, 1)

    for i, card in enumerate(jokers):
        row = buf[i]
        row[0] = center_key_id(card.center_key) / nck
        row[1] = min(card.base_cost / 20.0, 1.0) if card.base_cost else 0.0
        row[2] = _edition_idx(card.edition) / 4.0
        row[3] = _log_scale(card.sell_cost)
        row[4] = float(card.eternal)
        row[5] = float(card.perishable)
        row[6] = card.perish_tally / 5.0 if card.perishable else 1.0
        row[7] = float(card.rental)
        row[8] = float(card.debuff)
        row[9] = i / 20.0
        row[10] = _log_scale(
            card.ability.get("mult", 0) + card.ability.get("t_mult", 0)
        )
        row[11] = card.ability.get("x_mult", 1.0)
        row[12] = _log_scale(
            card.ability.get("t_chips", 0) + card.ability.get("bonus", 0)
        )
        extra = card.ability.get("extra")
        if isinstance(extra, (int, float)):
            row[13] = _log_scale(extra)
        elif isinstance(extra, dict):
            total = 0.0
            for val in extra.values():
                if isinstance(val, (int, float)):
                    total += val
            row[13] = _log_scale(total)
        row[14] = float(not card.debuff)

    return buf.copy()


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


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------


def encode_observation(gs: dict[str, Any]) -> Observation:
    """Encode a full game state into an :class:`Observation`.

    Parameters
    ----------
    gs : dict
        The engine game_state dict (from ``DirectAdapter.raw_state`` or
        equivalent).

    Returns
    -------
    Observation
        Entity-based observation with variable-length arrays.
    """
    # Global context
    global_ctx = encode_global_context(gs)

    # Hand cards — batch encode
    hand: list[Card] = gs.get("hand", [])
    hand_arr = encode_playing_cards_batch(hand, gs)

    # Jokers — batch encode
    jokers: list[Card] = gs.get("jokers", [])
    joker_arr = encode_jokers_batch(jokers, gs)

    # Consumables (small count, keep per-card)
    consumables: list[Card] = gs.get("consumables", [])
    if consumables:
        cons_arr = np.stack([encode_consumable(c, gs) for c in consumables])
    else:
        cons_arr = _EMPTY_CONSUMABLE

    # Shop cards (combine shop_cards + shop_vouchers + shop_boosters)
    shop_items: list[Card] = (
        gs.get("shop_cards", []) + gs.get("shop_vouchers", []) + gs.get("shop_boosters", [])
    )
    if shop_items:
        shop_arr = np.stack([encode_shop_item(c, gs) for c in shop_items])
    else:
        shop_arr = _EMPTY_SHOP

    # Pack cards — batch encode
    pack_cards: list[Card] = gs.get("pack_cards", [])
    pack_arr = encode_playing_cards_batch(pack_cards, gs)

    return Observation(
        global_context=global_ctx,
        hand_cards=hand_arr,
        jokers=joker_arr,
        consumables=cons_arr,
        shop_cards=shop_arr,
        pack_cards=pack_arr,
    )
