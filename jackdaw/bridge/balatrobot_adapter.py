"""Adapter between jackdaw Action types and balatrobot JSON-RPC calls.

Converts our frozen dataclass actions into balatrobot RPC payloads,
and converts balatrobot game state responses into our game_state dict.

See ``docs/balatrobot-action-mapping.md`` for the full mapping table.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import (
    Action,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
)
from jackdaw.engine.card import Card
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.hand_levels import HandLevels

# ---------------------------------------------------------------------------
# Action → balatrobot RPC
# ---------------------------------------------------------------------------

_STATE_MAP = {
    "BLIND_SELECT": GamePhase.BLIND_SELECT,
    "SELECTING_HAND": GamePhase.SELECTING_HAND,
    "ROUND_EVAL": GamePhase.ROUND_EVAL,
    "SHOP": GamePhase.SHOP,
    "SMODS_BOOSTER_OPENED": GamePhase.PACK_OPENING,
    "GAME_OVER": GamePhase.GAME_OVER,
}


def action_to_rpc(action: Action, game_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a jackdaw Action to a balatrobot JSON-RPC method + params.

    Returns ``{"method": str, "params": dict}`` suitable for sending
    as a JSON-RPC 2.0 request body (minus jsonrpc/id fields).

    *game_state* is required for swap actions so that the full
    permutation array can be constructed for the balatrobot RPC.
    """
    match action:
        case PlayHand(card_indices=indices):
            return {"method": "play", "params": {"cards": list(indices)}}

        case Discard(card_indices=indices):
            return {"method": "discard", "params": {"cards": list(indices)}}

        case SelectBlind():
            return {"method": "select", "params": {}}

        case SkipBlind():
            return {"method": "skip", "params": {}}

        case BuyCard(shop_index=idx):
            return {"method": "buy", "params": {"card": idx}}

        case SellCard(area=area, card_index=idx):
            if area == "jokers":
                return {"method": "sell", "params": {"joker": idx}}
            else:
                return {"method": "sell", "params": {"consumable": idx}}

        case UseConsumable(card_index=idx, target_indices=targets):
            params: dict[str, Any] = {"consumable": idx}
            if targets:
                params["cards"] = list(targets)
            return {"method": "use", "params": params}

        case RedeemVoucher(card_index=idx):
            return {"method": "buy", "params": {"voucher": idx}}

        case OpenBooster(card_index=idx):
            return {"method": "buy", "params": {"pack": idx}}

        case PickPackCard(card_index=idx, target_indices=targets):
            params = {"card": idx}
            if targets:
                params["targets"] = list(targets)
            return {"method": "pack", "params": params}

        case SkipPack():
            return {"method": "pack", "params": {"skip": True}}

        case Reroll():
            return {"method": "reroll", "params": {}}

        case NextRound():
            return {"method": "next_round", "params": {}}

        case CashOut():
            return {"method": "cash_out", "params": {}}

        case SortHand(mode=mode):
            assert game_state is not None, "game_state required for sort actions"
            hand = game_state.get("hand", [])
            n = len(hand)
            indices = list(range(n))

            def _card_id(c: Card) -> int:
                return getattr(c.base, "id", 0) if c.base else 0

            def _card_suit(c: Card) -> float:
                return getattr(c.base, "suit_nominal", 0) if c.base else 0

            if mode == "rank":
                indices.sort(key=lambda i: (_card_id(hand[i]), _card_suit(hand[i])))
            elif mode == "suit":
                indices.sort(key=lambda i: (_card_suit(hand[i]), _card_id(hand[i])))
            return {"method": "rearrange", "params": {"hand": indices}}

        case SwapHandLeft(idx=idx) | SwapHandRight(idx=idx):
            assert game_state is not None, "game_state required for swap actions"
            n = len(game_state.get("hand", []))
            order = list(range(n))
            other = idx - 1 if isinstance(action, SwapHandLeft) else idx + 1
            if 0 <= other < n:
                order[idx], order[other] = order[other], order[idx]
            return {"method": "rearrange", "params": {"hand": order}}

        case SwapJokersLeft(idx=idx) | SwapJokersRight(idx=idx):
            assert game_state is not None, "game_state required for swap actions"
            n = len(game_state.get("jokers", []))
            order = list(range(n))
            other = idx - 1 if isinstance(action, SwapJokersLeft) else idx + 1
            if 0 <= other < n:
                order[idx], order[other] = order[other], order[idx]
            return {"method": "rearrange", "params": {"jokers": order}}

        case _:
            raise ValueError(f"Unknown action type: {type(action).__name__}")


# ---------------------------------------------------------------------------
# Balatrobot state → game_state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reverse lookup tables for deserialization
# ---------------------------------------------------------------------------

_LETTER_TO_SUIT: dict[str, str] = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
_LETTER_TO_RANK: dict[str, str] = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "T": "10",
    "J": "Jack",
    "Q": "Queen",
    "K": "King",
    "A": "Ace",
}
_BOT_EDITION: dict[str, dict[str, bool]] = {
    "FOIL": {"foil": True},
    "HOLO": {"holo": True},
    "POLYCHROME": {"polychrome": True},
    "NEGATIVE": {"negative": True},
}
_BOT_SEAL: dict[str, str] = {"GOLD": "Gold", "RED": "Red", "BLUE": "Blue", "PURPLE": "Purple"}
_BOT_ENHANCEMENT: dict[str, str] = {
    "BONUS": "m_bonus",
    "MULT": "m_mult",
    "WILD": "m_wild",
    "GLASS": "m_glass",
    "STEEL": "m_steel",
    "STONE": "m_stone",
    "GOLD": "m_gold",
    "LUCKY": "m_lucky",
}
_BOT_SET: dict[str, str] = {
    "DEFAULT": "",
    "ENHANCED": "Enhanced",
    "JOKER": "Joker",
    "TAROT": "Tarot",
    "PLANET": "Planet",
    "SPECTRAL": "Spectral",
    "VOUCHER": "Voucher",
    "BOOSTER": "Booster",
}


# Rank → card_key prefix for playing card reconstruction
def _deserialize_card(data: dict[str, Any]) -> Card:
    """Reconstruct a Card object from balatrobot's JSON card schema."""
    card = Card()

    val = data.get("value") or {}
    mod = data.get("modifier") or {}
    state = data.get("state") or {}
    cost_info = data.get("cost") or {}
    if isinstance(val, list):
        val = {}
    if isinstance(mod, list):
        mod = {}
    if isinstance(state, list):
        state = {}
    if isinstance(cost_info, list):
        cost_info = {}
    card_set = _BOT_SET.get(data.get("set", ""), "")

    # Playing card base
    suit_letter = val.get("suit")
    rank_letter = val.get("rank")
    if suit_letter and rank_letter:
        suit = _LETTER_TO_SUIT.get(suit_letter, "Hearts")
        rank = _LETTER_TO_RANK.get(rank_letter, "2")
        card_key = data.get("key", "")
        card.set_base(card_key, suit, rank)

    # Center key (jokers, consumables, vouchers, boosters)
    key = data.get("key", "c_base")
    if not suit_letter:
        # Non-playing card — key IS the center key
        card.center_key = key
        try:
            card.set_ability(key)
        except (KeyError, TypeError):
            # Unknown center key — set basic ability
            card.ability = {"set": card_set, "name": val.get("effect", "")}
    else:
        # Playing card — set enhancement center if present
        enhancement = mod.get("enhancement")
        center = _BOT_ENHANCEMENT.get(enhancement, "c_base") if enhancement else "c_base"
        card.center_key = center

    # Edition
    edition_str = mod.get("edition")
    card.edition = _BOT_EDITION.get(edition_str) if edition_str else None

    # Seal
    seal_str = mod.get("seal")
    card.seal = _BOT_SEAL.get(seal_str) if seal_str else None

    # Stickers
    card.eternal = bool(mod.get("eternal", False))
    perish = mod.get("perishable")
    if perish is not None:
        card.perishable = True
        card.perish_tally = int(perish)
    card.rental = bool(mod.get("rental", False))

    # State
    card.debuff = bool(state.get("debuff", False))
    card.facing = "back" if state.get("hidden") else "front"

    # Cost
    card.sell_cost = cost_info.get("sell", 0)
    card.cost = cost_info.get("buy", 0)
    card.base_cost = card.cost

    return card


def _deserialize_area(area: dict[str, Any]) -> list[Card]:
    """Deserialize a balatrobot card area into a list of Card objects."""
    cards_data = area.get("cards", [])
    return [_deserialize_card(c) for c in cards_data]


def _deserialize_hand_levels(hands: dict[str, dict[str, Any]]) -> HandLevels:
    """Reconstruct HandLevels from balatrobot's hands schema."""
    hl = HandLevels()  # starts at defaults
    for ht_name, data in hands.items():
        try:
            ht = HandType(ht_name)
        except ValueError:
            continue
        hs = hl.get_state(ht)
        hs.level = data.get("level", 1)
        hs.chips = data.get("chips", hs.chips)
        hs.mult = data.get("mult", hs.mult)
        hs.played = data.get("played", 0)
        hs.played_this_round = data.get("played_this_round", 0)
        hs.visible = True  # if it's in the response, it's visible
    return hl


def bot_state_to_game_state(bot: dict[str, Any]) -> dict[str, Any]:
    """Convert a balatrobot gamestate response to a full game_state dict.

    Creates proper Card objects, HandLevels, and Blind objects so that
    ``get_action_mask()`` and ``encode_observation()`` work correctly.
    """
    from jackdaw.engine.blind import Blind

    gs: dict[str, Any] = {}

    # Phase
    state_str = bot.get("state", "")
    gs["phase"] = _STATE_MAP.get(state_str, state_str)

    # Economy
    gs["dollars"] = bot.get("money", 0)

    # Ante / round
    ante = bot.get("ante_num", 1)
    gs["round_resets"] = {"ante": ante}
    gs["round"] = bot.get("round_num", 0)

    # Round state
    br = bot.get("round", {})
    gs["current_round"] = {
        "hands_left": br.get("hands_left", 0),
        "discards_left": br.get("discards_left", 0),
        "hands_played": br.get("hands_played", 0),
        "discards_used": br.get("discards_used", 0),
        "reroll_cost": br.get("reroll_cost", 5),
        "free_rerolls": 0,
    }
    gs["chips"] = br.get("chips", 0)

    # Deserialize card areas into proper Card objects
    gs["hand"] = _deserialize_area(bot.get("hand", {}))
    gs["deck"] = _deserialize_area(bot.get("cards", {}))
    gs["jokers"] = _deserialize_area(bot.get("jokers", {}))
    gs["consumables"] = _deserialize_area(bot.get("consumables", {}))
    gs["discard_pile"] = []  # not sent by balatrobot

    # Shop areas
    gs["shop_cards"] = _deserialize_area(bot.get("shop", {}))
    gs["shop_vouchers"] = _deserialize_area(bot.get("vouchers", {}))
    gs["shop_boosters"] = _deserialize_area(bot.get("packs", {}))
    gs["pack_cards"] = _deserialize_area(bot.get("pack", {}))
    gs["pack_choices_remaining"] = bot.get("pack", {}).get("limit", 0)

    # Slot limits
    gs["joker_slots"] = bot.get("jokers", {}).get("limit", 5)
    gs["consumable_slots"] = bot.get("consumables", {}).get("limit", 2)
    gs["hand_size"] = bot.get("hand", {}).get("limit", 8)

    # Hand levels
    gs["hand_levels"] = _deserialize_hand_levels(bot.get("hands", {}))

    # Blinds
    blinds = bot.get("blinds", {})
    gs["blind_info"] = {}
    blind_states: dict[str, str] = {}
    for btype in ("small", "big", "boss"):
        bi = blinds.get(btype, {})
        gs["blind_info"][btype] = {
            "name": bi.get("name", ""),
            "status": bi.get("status", ""),
            "score": bi.get("score", 0),
            "tag_name": bi.get("tag_name", ""),
        }
        # Map balatrobot status → engine blind state
        status = bi.get("status", "")
        engine_status = {
            "SELECT": "Select",
            "CURRENT": "Current",
            "UPCOMING": "Select",
            "SKIPPED": "Skipped",
            "DEFEATED": "Defeated",
        }.get(status, status)
        blind_states[btype.capitalize()] = engine_status

    gs["round_resets"]["blind_states"] = blind_states

    # Derive blind_on_deck — first non-defeated/non-skipped blind
    if blind_states.get("Small") in ("", "Select", "Current"):
        gs["blind_on_deck"] = "Small"
    elif blind_states.get("Big") in ("", "Select", "Current"):
        gs["blind_on_deck"] = "Big"
    else:
        gs["blind_on_deck"] = "Boss"

    # Active blind object
    active_type = None
    for btype in ("boss", "big", "small"):
        if blind_states.get(btype.capitalize()) == "Current":
            active_type = btype
            break
    if active_type:
        bi = blinds.get(active_type, {})
        score = bi.get("score", 0)
        try:
            gs["blind"] = Blind.create(bi.get("name", "bl_small"), ante, 1, 1.0)
            gs["blind"].chips = score
        except Exception:
            gs["blind"] = None
    else:
        gs["blind"] = None

    # Game modifiers (defaults — balatrobot doesn't send all of these)
    gs["four_fingers"] = 0
    gs["shortcut"] = 0
    gs["smeared"] = 0
    gs["splash"] = 0
    gs["interest_cap"] = 25
    gs["discount_percent"] = 0
    gs["skips"] = 0
    uv = bot.get("used_vouchers", {})
    gs["used_vouchers"] = uv if isinstance(uv, dict) else {}
    gs["awarded_tags"] = []
    gs["deck_size"] = bot.get("cards", {}).get("count", len(gs["deck"]))

    # Backward compat keys
    gs["hand_keys"] = [c.get("key", "") for c in bot.get("hand", {}).get("cards", [])]
    gs["hand_count"] = len(gs["hand"])
    gs["joker_keys"] = [c.get("key", "") for c in bot.get("jokers", {}).get("cards", [])]
    gs["joker_count"] = len(gs["jokers"])
    gs["consumable_keys"] = [c.get("key", "") for c in bot.get("consumables", {}).get("cards", [])]
    gs["deck_keys"] = [c.get("key", "") for c in bot.get("cards", {}).get("cards", [])]

    # Metadata
    gs["seed"] = bot.get("seed", "")
    gs["deck_type"] = bot.get("deck", "")
    gs["stake_type"] = bot.get("stake", "")
    gs["won"] = bot.get("won", False)

    return gs


# ---------------------------------------------------------------------------
# Game state → balatrobot response (reverse direction)
# ---------------------------------------------------------------------------


def game_state_to_bot_response(gs: dict[str, Any]) -> dict[str, Any]:
    """Convert our game_state to balatrobot's JSON response format.

    Delegates to :func:`jackdaw.bridge.serializer.game_state_to_bot_response`.
    """
    from jackdaw.bridge.serializer import (
        game_state_to_bot_response as _serialize,
    )

    return _serialize(gs)


# ---------------------------------------------------------------------------
# Game state → comparison keys
# ---------------------------------------------------------------------------


def extract_comparison_keys(gs: dict[str, Any]) -> dict[str, Any]:
    """Extract the key fields from our game_state for comparison with bot state.

    Returns a flat dict suitable for field-by-field comparison with the
    output of :func:`bot_state_to_game_state`.
    """
    cr = gs.get("current_round", {})
    return {
        "dollars": gs.get("dollars", 0),
        "chips": gs.get("chips", 0),
        "ante": gs.get("round_resets", {}).get("ante", 1),
        "hands_left": cr.get("hands_left", 0),
        "discards_left": cr.get("discards_left", 0),
        "hand_count": len(gs.get("hand", [])),
        "deck_size": len(gs.get("deck", [])),
        "joker_count": len(gs.get("jokers", [])),
    }
