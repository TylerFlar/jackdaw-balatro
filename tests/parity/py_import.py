"""Rebuild a playable Python oracle state from a canonical snapshot.

The inverse of :mod:`tests.parity.py_export`.  Only the snapshot is used;
Python bookkeeping that the snapshot leaves out (scratch keys, statistics,
aliases) is either recomputed by the engine before it is read or never read.
``test_roundtrip`` checks that claim by stepping an original and a rebuilt
state side by side.
"""

from __future__ import annotations

import copy
from typing import Any

from jackdaw.engine import card as card_module
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, CardBase, _resolve_center
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.hands import HAND_BASE
from jackdaw.engine.data.prototypes import BLINDS
from jackdaw.engine.economy import RoundEarnings
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import init_game_object
from jackdaw.engine.scoring import ScoreResult
from tests.parity.py_export import (
    _TAG_FIRED_FLAG,
    AREAS,
    CURRENT_ROUND_KEYS,
    GAME_SCALARS,
    ROUND_RESETS_KEYS,
    SCHEMA,
    tag_type,
)


def import_card(cs: dict[str, Any]) -> Card:
    c = Card(sort_id=cs["sort_id"])  # explicit id: no counter increment
    c.center_key = cs["center"]
    c.card_key = cs["front"]
    base = cs["base"]
    if base is not None:
        c.base = CardBase(
            suit=Suit(base["suit"]),
            rank=Rank(base["value"]),
            id=base["id"],
            nominal=base["nominal"],
            suit_nominal=base["suit_nominal"],
            suit_nominal_original=base["suit_nominal_original"],
            face_nominal=base["face_nominal"],
            original_value=Rank(base["original_value"]),
            times_played=base["times_played"],
        )
    center = _resolve_center(cs["center"])
    config = center.get("config") or {}
    if isinstance(config, list):
        config = {}
    ability: dict[str, Any] = {
        "name": center.get("name", ""),
        "effect": center.get("effect", ""),
        "set": center.get("set", ""),
        "type": config.get("type", ""),
        "order": center.get("order"),
    }
    if center.get("consumeable"):
        ability["consumeable"] = config  # set_ability aliases the center config
    ability.update(copy.deepcopy(cs["ability"]))
    c.ability = ability
    c.edition = copy.deepcopy(cs["edition"]) if cs["edition"] else None
    c.seal = cs["seal"]
    c.debuff = cs["debuff"]
    c.facing = cs["facing"]
    c.playing_card = cs["playing_card"]
    costs = cs["costs"]
    c.base_cost = costs["base_cost"]
    c.extra_cost = costs["extra_cost"]
    c.cost = costs["cost"]
    c.sell_cost = costs["sell_cost"]
    st = cs["stickers"]
    c.eternal = st["eternal"]
    c.perishable = st["perishable"]
    if st["perish_tally"] is not None:
        c.perish_tally = st["perish_tally"]
    c.rental = st["rental"]
    if st.get("pinned"):
        c.pinned = True  # type: ignore[attr-defined]
    return c


def _import_blind(bs: dict[str, Any] | None) -> Blind | None:
    if bs is None:
        return None
    key = bs["key"]
    proto = BLINDS.get(key)
    debuff = proto.debuff if proto is not None and isinstance(proto.debuff, dict) else {}
    blind = Blind(
        key=key,
        name=proto.name if proto is not None else "",
        chips=bs["chips"],
        mult=bs["mult"],
        dollars=bs["dollars"],
        boss=bs["boss"],
        disabled=bs["disabled"],
        triggered=bs["triggered"],
        debuff_config=debuff,
        only_hand=bs["mouth_hand"],
        discards_sub=bs["discards_sub"],
        hands_sub=bs["hands_sub"],
    )
    if bs["prepped"]:
        blind.prepped = True  # type: ignore[attr-defined]
    blind.hands_used = {h: True for h in bs["eye_hands"]}
    return blind


def _import_tags(history: list[str], pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild ``awarded_tags``: pending tags are the leftmost subsequence of the history."""
    want = [t["key"] for t in pending]
    entries: list[dict[str, Any]] = []
    j = 0
    for key in history:
        entry: dict[str, Any] = {"key": key, "result": None, "blind": "import"}
        if j < len(want) and want[j] == key and tag_type(key) != "immediate":
            j += 1  # pending: no fired flag
        else:
            flag = _TAG_FIRED_FLAG.get(tag_type(key))
            if flag is not None:
                entry[flag] = True
        entries.append(entry)
    if j != len(want):
        raise ValueError(f"pending tags {want} are not a subsequence of history {history}")
    return entries


def import_state(snap: dict[str, Any]) -> dict[str, Any]:
    if snap.get("schema") != SCHEMA:
        raise ValueError(f"unsupported snapshot schema {snap.get('schema')!r}")
    gs = init_game_object()

    run = snap["run"]
    gs["selected_back_key"] = run["back"]
    gs["stake"] = run["stake"]
    gs["win_ante"] = run["win_ante"]
    gs["won"] = run["won"]
    gs["seed"] = run["seed"]
    gs["seeded"] = True
    if run["challenge"] is not None:
        gs["challenge"] = run["challenge"]

    game = snap["game"]
    for k in GAME_SCALARS:
        gs[k] = copy.deepcopy(game[k])
    gs["probabilities"] = {"normal": game["probabilities_normal"]}
    gs["shop"] = {"joker_max": game["shop_joker_max"]}
    if game["first_shop_buffoon"]:
        gs["first_shop_buffoon"] = True

    gs["starting_params"] = copy.deepcopy(snap["starting_params"])
    gs["modifiers"] = copy.deepcopy(snap["modifiers"])
    rr = gs["round_resets"]
    for k in ROUND_RESETS_KEYS:
        if snap["round_resets"][k] is not None or k in rr:
            rr[k] = copy.deepcopy(snap["round_resets"][k])
    cr = gs["current_round"]
    for k in CURRENT_ROUND_KEYS:
        cr[k] = copy.deepcopy(snap["current_round"][k])
    gs["round_bonus"] = copy.deepcopy(snap["round_bonus"])

    blind = _import_blind(snap["blind"])
    gs["blind"] = blind
    rr["blind"] = blind

    hl = HandLevels()
    for ht in HAND_BASE:
        h = snap["hands"][ht.value]
        st = hl.get_state(ht)
        st.level = h["level"]
        st.chips = h["chips"]
        st.mult = h["mult"]
        st.played = h["played"]
        st.played_this_round = h["played_this_round"]
        st.visible = h["visible"]
    gs["hand_levels"] = hl

    sim = snap["sim"]
    gs["awarded_tags"] = _import_tags(sim["tag_history"], snap["tags"])

    sets = snap["sets"]
    gs["used_jokers"] = {k: True for k in sets["used_jokers"]}
    gs["used_vouchers"] = {k: True for k in sets["used_vouchers"]}
    gs["banned_keys"] = {k: True for k in sets["banned_keys"]}
    gs["pool_flags"] = copy.deepcopy(sets["pool_flags"])
    gs["bosses_used"] = copy.deepcopy(sets["bosses_used"])

    usage = snap["consumeable_usage"]
    if usage["by_key"]:
        gs["consumable_usage"] = {
            k: {"count": n, "set": _resolve_center(k).get("set", "")}
            for k, n in usage["by_key"].items()
        }
    if usage["totals"]:
        gs["consumable_usage_total"] = copy.deepcopy(usage["totals"])

    for key, name in AREAS:
        gs[key] = [import_card(cs) for cs in snap["areas"][name]]

    pack = snap["pack"]
    if pack is not None:
        gs["pack_type"] = pack["kind"]
        gs["pack_choices_remaining"] = pack["choices_remaining"]
        by_id = {c.sort_id: c for c in gs["hand"]}
        gs["pack_hand"] = [by_id[i] for i in pack["dealt"]]
        gs["shop_return_phase"] = GamePhase(pack["return_phase"])
    else:
        gs["pack_choices_remaining"] = 0
        gs["pack_hand"] = []

    rng_snap = snap["rng"]
    rng = PseudoRandom(rng_snap["seed"])
    rng._state = {
        "seed": rng_snap["seed"],
        "hashed_seed": rng_snap["hashed_seed"],
        **rng_snap["streams"],
    }
    gs["rng"] = rng

    card_module._sort_id_counter = sim["next_sort_id"] - 1
    if sim["pending_tag_packs"]:
        gs["pending_tag_packs"] = list(sim["pending_tag_packs"])
    if sim["pending_deck_bottom"]:
        gs["pending_deck_bottom"] = [import_card(cs) for cs in sim["pending_deck_bottom"]]
    if sim["shop_d6ed"]:
        gs["shop_d6ed"] = True
    if sim["hand_last_poll_size"]:
        gs["hand_last_poll_size"] = sim["hand_last_poll_size"]
    if sim["last_blind_was_boss"]:
        gs["last_blind_was_boss"] = True
    if snap["phase"] == GamePhase.ROUND_EVAL.value:
        gs["round_earnings"] = RoundEarnings(total=sim["pending_cashout"])
    last = sim["last_score"]
    if last is not None:
        gs["last_score_result"] = ScoreResult(
            hand_type=last["hand_type"],
            scoring_cards=[],
            chips=last["chips"],
            mult=last["mult"],
            total=last["total"],
            dollars_earned=last["dollars"],
            debuffed=last["debuffed"],
        )

    gs["phase"] = GamePhase(snap["phase"])
    return gs
