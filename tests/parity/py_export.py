"""Export the Python oracle's game state as a canonical snapshot.

The snapshot format is defined in ``docs/snapshot-schema.md``.  It holds the
*semantic* state of a run: everything that can change what happens next, and
nothing that is Python bookkeeping (scratch keys written before scoring,
write-only statistics, aliases).  ``py_import.import_state`` rebuilds a
playable Python state from it; the round-trip tests prove the snapshot is
complete by stepping both states and comparing.
"""

from __future__ import annotations

import enum
import math
from typing import Any

from jackdaw.engine import card as card_module
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.data.hands import HAND_BASE
from jackdaw.engine.data.prototypes import TAGS

SCHEMA = "jackdaw.snapshot/1"

# Ability keys that set_ability derives from the center; the center key
# carries them, so they are not part of the snapshot.
DERIVED_ABILITY_KEYS = frozenset({"name", "effect", "set", "order", "consumeable", "type"})

# Python area key -> snapshot area name (Lua CardArea names).
AREAS: tuple[tuple[str, str], ...] = (
    ("deck", "deck"),
    ("hand", "hand"),
    ("discard_pile", "discard"),
    ("played_cards_area", "play"),
    ("jokers", "jokers"),
    ("consumables", "consumables"),
    ("shop_cards", "shop_jokers"),
    ("shop_vouchers", "shop_vouchers"),
    ("shop_boosters", "shop_booster"),
    ("pack_cards", "pack_cards"),
)

GAME_SCALARS: tuple[str, ...] = (
    "dollars",
    "chips",
    "round",
    "skips",
    "hands_played",
    "unused_discards",
    "interest_cap",
    "interest_amount",
    "discount_percent",
    "inflation",
    "rental_rate",
    "bankrupt_at",
    "base_reroll_cost",
    "edition_rate",
    "joker_rate",
    "tarot_rate",
    "planet_rate",
    "spectral_rate",
    "playing_card_rate",
    "hand_size",
    "joker_slots",
    "consumable_slots",
    "starting_deck_size",
    "last_tarot_planet",
    "last_hand_played",
    "pack_size",
    "ecto_minus",
    "perishable_rounds",
    "blind_on_deck",
)

ROUND_RESETS_KEYS: tuple[str, ...] = (
    "hands",
    "discards",
    "reroll_cost",
    "temp_reroll_cost",
    "temp_handsize",
    "ante",
    "blind_ante",
    "blind_states",
    "blind_choices",
    "blind_tags",
    "boss_rerolled",
)

CURRENT_ROUND_KEYS: tuple[str, ...] = (
    "hands_left",
    "hands_played",
    "discards_left",
    "discards_used",
    "reroll_cost",
    "reroll_cost_increase",
    "free_rerolls",
    "jokers_purchased",
    "voucher",
    "most_played_poker_hand",
    "idol_card",
    "mail_card",
    "ancient_card",
    "castle_card",
)

# Tag context type -> the awarded_tags flag Python sets once a tag of that
# type has triggered.  ``immediate`` tags trigger when awarded; Double Tags
# (``tag_add``) are removed from the list when they trigger.
_TAG_FIRED_FLAG: dict[str, str | None] = {
    "eval": "eval_fired",
    "new_blind_choice": "nbc_fired",
    "round_start_bonus": "rsb_fired",
    "store_joker_create": "shop_fired",
    "store_joker_modify": "shop_fired",
    "shop_start": "shop_fired",
    "shop_final_pass": "shop_fired",
    "voucher_add": "shop_fired",
    "tag_add": None,
}


def tag_type(key: str) -> str:
    proto = TAGS[key]
    config = proto.config if isinstance(proto.config, dict) else {}
    return str(config.get("type", ""))


def plain(v: Any) -> Any:
    """Convert a Python value into a snapshot tree value."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, enum.Enum):
        return plain(v.value)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, dict):
        return {str(plain(k)): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(plain(x) for x in v)
    raise TypeError(f"export: unsupported value {type(v).__qualname__}: {v!r}")


def _truthy_keys(d: Any) -> list[str]:
    if not d:
        return []
    if isinstance(d, dict):
        return sorted(str(k) for k, v in d.items() if v)
    return sorted(str(k) for k in d)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def export_ability(ability: dict[str, Any]) -> dict[str, Any]:
    return {k: plain(v) for k, v in ability.items() if k not in DERIVED_ABILITY_KEYS}


def export_card(c: Any) -> dict[str, Any]:
    base = c.base
    return {
        "sort_id": c.sort_id,
        "center": c.center_key,
        "front": c.card_key,
        "base": None
        if base is None
        else {
            "suit": plain(base.suit),
            "value": plain(base.rank),
            "id": base.id,
            "nominal": base.nominal,
            "suit_nominal": base.suit_nominal,
            "suit_nominal_original": base.suit_nominal_original,
            "face_nominal": base.face_nominal,
            "original_value": plain(base.original_value),
            "times_played": base.times_played,
        },
        "ability": export_ability(c.ability),
        "edition": plain(c.edition) if c.edition else None,
        "seal": c.seal,
        "debuff": bool(c.debuff),
        "facing": c.facing,
        "playing_card": c.playing_card,
        "costs": {
            "base_cost": c.base_cost,
            "extra_cost": c.extra_cost,
            "cost": c.cost,
            "sell_cost": c.sell_cost,
        },
        "stickers": {
            "eternal": bool(c.eternal),
            "perishable": bool(c.perishable),
            "perish_tally": c.perish_tally if c.perishable else None,
            "rental": bool(c.rental),
            "pinned": bool(getattr(c, "pinned", False)),
        },
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _export_blind(blind: Any) -> dict[str, Any] | None:
    if blind is None:
        return None
    return {
        "key": blind.key,
        "chips": blind.chips,
        "mult": blind.mult,
        "dollars": blind.dollars,
        "boss": bool(blind.boss),
        "disabled": bool(blind.disabled),
        "triggered": bool(blind.triggered),
        "prepped": bool(getattr(blind, "prepped", False)),
        "eye_hands": sorted(str(h) for h, used in blind.hands_used.items() if used),
        "mouth_hand": blind.only_hand,
        "discards_sub": blind.discards_sub,
        "hands_sub": blind.hands_sub,
    }


def _export_hands(hand_levels: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ht in HAND_BASE:
        h = hand_levels.get_state(ht)
        out[ht.value] = {
            "level": h.level,
            "chips": h.chips,
            "mult": h.mult,
            "played": h.played,
            "played_this_round": h.played_this_round,
            "visible": bool(h.visible),
        }
    return out


def pending_tags(awarded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in awarded:
        key = entry.get("key", "")
        ttype = tag_type(key)
        if ttype == "immediate":
            continue
        flag = _TAG_FIRED_FLAG.get(ttype)
        if flag is not None and entry.get(flag):
            continue
        out.append({"key": key})
    return out


def _export_rng(rng: Any) -> dict[str, Any]:
    state = rng.state
    return {
        "seed": state["seed"],
        "hashed_seed": state["hashed_seed"],
        "streams": {k: v for k, v in state.items() if k not in ("seed", "hashed_seed")},
    }


def _export_pack(gs: dict[str, Any]) -> dict[str, Any] | None:
    if gs.get("phase") != GamePhase.PACK_OPENING:
        return None
    # Dealt cards destroyed by a pick (e.g. The Hanged Man) stay in
    # pack_hand but have left the hand; only those still held return to
    # the deck when the pack closes.
    in_hand = {id(c) for c in gs.get("hand", [])}
    return {
        "kind": gs.get("pack_type", ""),
        "choices_remaining": gs.get("pack_choices_remaining", 0),
        "dealt": [c.sort_id for c in gs.get("pack_hand", []) or [] if id(c) in in_hand],
        "return_phase": plain(gs.get("shop_return_phase", GamePhase.SHOP)),
    }


def _export_last_score(res: Any) -> dict[str, Any] | None:
    if res is None:
        return None
    return {
        "hand_type": plain(res.hand_type),
        "chips": res.chips,
        "mult": res.mult,
        "total": res.total,
        "dollars": res.dollars_earned,
        "debuffed": bool(res.debuffed),
    }


def export_state(gs: dict[str, Any]) -> dict[str, Any]:
    """Canonical snapshot of a Python engine state (read-only)."""
    phase = gs.get("phase")
    earnings = gs.get("round_earnings")
    usage = gs.get("consumable_usage") or {}
    return {
        "schema": SCHEMA,
        "phase": plain(phase),
        "run": {
            "seed": gs["rng"].seed_str,
            "back": gs.get("selected_back_key"),
            "stake": gs.get("stake", 1),
            "challenge": gs.get("challenge"),
            "win_ante": gs.get("win_ante", 8),
            "won": bool(gs.get("won", False)),
        },
        "game": {
            **{k: plain(gs.get(k)) for k in GAME_SCALARS},
            "probabilities_normal": gs.get("probabilities", {}).get("normal", 1),
            "shop_joker_max": gs.get("shop", {}).get("joker_max", 2),
            "first_shop_buffoon": bool(gs.get("first_shop_buffoon", False)),
        },
        "starting_params": plain(gs.get("starting_params", {})),
        "modifiers": plain(gs.get("modifiers", {})),
        "round_resets": {k: plain(gs["round_resets"].get(k)) for k in ROUND_RESETS_KEYS},
        "current_round": {k: plain(gs["current_round"].get(k)) for k in CURRENT_ROUND_KEYS},
        "round_bonus": plain(gs.get("round_bonus", {})),
        "blind": _export_blind(gs.get("blind")),
        "hands": _export_hands(gs["hand_levels"]),
        "tags": pending_tags(gs.get("awarded_tags", [])),
        "sets": {
            "used_jokers": _truthy_keys(gs.get("used_jokers")),
            "used_vouchers": _truthy_keys(gs.get("used_vouchers")),
            "banned_keys": _truthy_keys(gs.get("banned_keys")),
            "pool_flags": plain(gs.get("pool_flags", {})),
            "bosses_used": plain(gs.get("bosses_used", {})),
        },
        "consumeable_usage": {
            "by_key": {k: v.get("count", 0) for k, v in usage.items()},
            "totals": plain(gs.get("consumable_usage_total") or {}),
        },
        "areas": {name: [export_card(c) for c in gs.get(key, []) or []] for key, name in AREAS},
        "pack": _export_pack(gs),
        "rng": _export_rng(gs["rng"]),
        "sim": {
            "next_sort_id": card_module._sort_id_counter + 1,
            "pending_tag_packs": list(gs.get("pending_tag_packs") or []),
            "pending_deck_bottom": [export_card(c) for c in gs.get("pending_deck_bottom") or []],
            "shop_d6ed": bool(gs.get("shop_d6ed", False)),
            "last_blind_was_boss": bool(gs.get("last_blind_was_boss", False)),
            "pending_cashout": earnings.total
            if earnings is not None and phase == GamePhase.ROUND_EVAL
            else 0,
            "last_score": _export_last_score(gs.get("last_score_result")),
            "tag_history": [e.get("key", "") for e in gs.get("awarded_tags", [])],
        },
    }


def is_finite_tree(obj: Any) -> bool:
    """True when no float in the tree is inf/NaN (useful in assertions)."""
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(is_finite_tree(v) for v in obj.values())
    if isinstance(obj, list):
        return all(is_finite_tree(v) for v in obj)
    return True
