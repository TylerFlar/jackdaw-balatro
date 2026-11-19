"""The player's view (``Game.player_view``): what a player sees, and nothing they can't.

The hidden-information tests follow ``tests/env/test_player_view.py``: they build states a player
can't tell apart (they differ only in what the player can't see), and check that the view is the
same for each, while the engine tells them apart (their state hashes differ).
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

from jackdaw import _core
from jackdaw.core import Game
from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
    SortHand,
)

Snapshot = dict[str, Any]
KEY = 0x5EC12E7

FACE_DOWN_BOSSES = ("bl_wheel", "bl_house", "bl_mark", "bl_fish")
IDENTITY = ("front", "base")
#: What a face-down card shows: its place, its id, whether it shakes or is forced.
SHOWN_FACE_DOWN = {"id", "face_down", "shaking", "forced"}


def _dealt(seed: str) -> Game:
    game = Game.new("b_red", 1, seed)
    game.step(SelectBlind())
    assert game.phase == GamePhase.SELECTING_HAND
    return game


def _view(snap: Snapshot) -> dict[str, Any]:
    return Game.from_dict(snap).player_view(KEY)


def _check_hidden(base: Snapshot, variants: Iterable[tuple[str, Snapshot]]) -> None:
    ref, ref_hash = _view(base), Game.from_dict(base).state_hash()
    told_apart = 0
    for label, snap in variants:
        assert _view(snap) == ref, f"the view sees {label}"
        told_apart += Game.from_dict(snap).state_hash() != ref_hash
    assert told_apart, "the variants should be other states"


def _swap(a: dict[str, Any], b: dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        a[key], b[key] = b[key], a[key]


def _swaps(snap: Snapshot, area: str, index: int) -> Iterator[tuple[str, Snapshot]]:
    for j in range(len(snap["areas"]["deck"])):
        s = copy.deepcopy(snap)
        _swap(s["areas"][area][index], s["areas"]["deck"][j], IDENTITY)
        yield f"{area}[{index}] <-> deck[{j}]", s


def _boss(snap: Snapshot, key: str, disabled: bool = False) -> None:
    snap["blind"]["key"] = key
    snap["blind"]["boss"] = True
    snap["blind"]["disabled"] = disabled


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def test_the_document() -> None:
    game = _dealt("PVIEW_DOC")
    v = game.player_view(KEY)
    assert v["format"] == "jackdaw.player_view/1"
    assert v["phase"] == "selecting_hand"
    assert len(v["hand"]) == 8 and len(v["deck"]) == 52
    assert v["deck_size"] == 44 and v["discard_size"] == 0
    assert v["blind"]["key"] == "bl_small" and v["blinds"]["Small"]["state"] == "Current"
    assert v["blinds"]["Boss"]["key"] == game.to_dict()["round_resets"]["blind_choices"]["Boss"]
    assert v["shop"] is None and v["pack"] is None and v["last_play"] is None
    text = json.dumps(v)
    # No seed, no RNG state, no sort ids.
    assert game.seed not in text
    assert "rng" not in v and "streams" not in text and "sort_id" not in text
    sort_ids = {c.sort_id for c in game.hand}
    assert not sort_ids & {c["id"] for c in v["hand"]}
    for c, card in zip(v["hand"], game.hand, strict=True):
        assert (c["front"], c["key"], c["face_down"]) == (card.front, card.key, False)


def test_the_shop_shows_in_the_shop_only() -> None:
    game = _dealt("PVIEW_SHOP")
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    assert game.phase == GamePhase.ROUND_EVAL
    assert game.player_view(KEY)["shop"] is None
    game.step(CashOut())
    v = game.player_view(KEY)
    assert v["phase"] == "shop"
    assert [c["key"] for c in v["shop"]["jokers"]] == [c.key for c in game.shop_jokers]
    assert [c["key"] for c in v["shop"]["boosters"]] == [c.key for c in game.shop_boosters]


def test_ids_depend_on_the_key() -> None:
    game = _dealt("PVIEW_KEY")
    a, b = game.player_view(1), game.player_view(2)
    assert [c["id"] for c in a["hand"]] != [c["id"] for c in b["hand"]]
    assert [c["front"] for c in a["hand"]] == [c["front"] for c in b["hand"]]
    assert all(0 <= c["id"] < 2**53 for c in a["hand"])


# ---------------------------------------------------------------------------
# Card ids
# ---------------------------------------------------------------------------


def test_ids_follow_the_cards() -> None:
    """Sorting moves the cards; each keeps its id, as a player follows it."""
    game = _dealt("PVIEW_SORT")
    before = {c["id"]: c["front"] for c in game.player_view(KEY)["hand"]}
    game.step(SortHand(mode="suit"))
    after = {c["id"]: c["front"] for c in game.player_view(KEY)["hand"]}
    assert before == after


def test_a_card_through_the_deck_comes_back_under_a_new_id() -> None:
    """A card that goes into the deck is one the player loses track of: its view epoch goes
    up, and with it its id."""
    game = _dealt("PVIEW_DECK")
    epochs = {sid: e for sid, e, _, _ in game._g._card_runtime("hand")}
    ids = [c["id"] for c in game.player_view(KEY)["hand"]]
    assert ids == [_core.Game._player_view_card_id(KEY, c.sort_id, epochs[c.sort_id])
                   for c in game.hand]  # fmt: skip
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    game.step(CashOut())
    back = {sid: e for sid, e, _, _ in game._g._card_runtime("deck")}
    assert all(back[sid] == e + 1 for sid, e in epochs.items())


def test_the_last_play_names_the_cards_by_their_ids() -> None:
    game = _dealt("PVIEW_PLAY")
    game.record_scoring = True
    hand = game.player_view(KEY)["hand"]
    game.step(PlayHand(card_indices=(0, 1, 2)))
    play = game.player_view(KEY)["last_play"]
    assert play["played"] == [hand[i]["id"] for i in (0, 1, 2)]
    assert set(play["scoring"]) <= set(play["played"])
    named = {e.get("card") for e in play["events"]} - {None}
    assert named <= set(play["played"])
    assert play["total"] == game.last_score.total


def test_a_destroyed_card_keeps_its_id_in_the_last_play() -> None:
    """A Glass Card that shatters is gone after the play; its events still name it."""
    for n in range(40):
        game = _dealt(f"PVIEW_GLASS_{n}")
        game.record_scoring = True
        sid = game.debug_add("H_A", enhancement="m_glass")
        game.debug_set_hand([sid] + [c.sort_id for c in game.hand[:4]])
        before = game.player_view(KEY)["hand"][0]["id"]
        game.step(PlayHand(card_indices=(0,)))
        play = game.player_view(KEY)["last_play"]
        if any(e["kind"] == "destroyed" for e in play["events"]):
            assert play["played"] == [before]
            assert any(e["kind"] == "destroyed" and e["card"] == before for e in play["events"])
            return
    pytest.fail("no Glass Card shattered in 40 plays")


# ---------------------------------------------------------------------------
# Hidden information
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boss", FACE_DOWN_BOSSES)
@pytest.mark.parametrize("index", [0, 5])
def test_a_face_down_hand_card_shows_nothing(boss: str, index: int) -> None:
    snap = _dealt("PVIEW_DOWN").to_dict()
    _boss(snap, boss)
    snap["areas"]["hand"][index]["facing"] = "back"
    card = _view(snap)["hand"][index]
    assert card["face_down"] and all(card[k] is None for k in set(card) - SHOWN_FACE_DOWN)
    _check_hidden(snap, _swaps(snap, "hand", index))


def test_face_down_hand_cards_against_each_other() -> None:
    snap = _dealt("PVIEW_DOWN2").to_dict()
    _boss(snap, "bl_house")
    for i in (1, 3):
        snap["areas"]["hand"][i]["facing"] = "back"
    s = copy.deepcopy(snap)
    _swap(s["areas"]["hand"][1], s["areas"]["hand"][3], IDENTITY)
    _check_hidden(snap, [("hand[1] <-> hand[3]", s)])


@pytest.mark.parametrize("boss", FACE_DOWN_BOSSES)
def test_a_card_discarded_face_down_shows_nothing(boss: str) -> None:
    snap = _dealt("PVIEW_DISCARD").to_dict()
    _boss(snap, boss)
    snap["areas"]["discard"].append(snap["areas"]["deck"].pop())
    _check_hidden(snap, _swaps(snap, "discard", -1))


def test_the_deck_order_doesnt_show() -> None:
    snap = _dealt("PVIEW_ORDER").to_dict()
    s = copy.deepcopy(snap)
    s["areas"]["deck"].reverse()
    _check_hidden(snap, [("the deck reversed", s)])


def test_the_rng_doesnt_show() -> None:
    snap = _dealt("PVIEW_RNG").to_dict()
    s = copy.deepcopy(snap)
    for k in s["rng"]["streams"]:
        s["rng"]["streams"][k] = 0.123
    _check_hidden(snap, [("other RNG streams", s)])


def test_face_down_jokers_show_nothing() -> None:
    """Amber Acorn's jokers: face down and shuffled, which is which can't show."""
    game = _dealt("PVIEW_ACORN")
    game.debug_add("j_joker")
    game.debug_add("j_greedy_joker", edition="foil")
    snap = game.to_dict()
    for c in snap["areas"]["jokers"]:
        c["facing"] = "back"
    s = copy.deepcopy(snap)
    a, b = s["areas"]["jokers"]
    _swap(a, b, ("center", "ability", "edition", "costs"))
    _check_hidden(snap, [("the jokers exchanged", s)])
    jokers = _view(snap)["jokers"]
    assert all(j["face_down"] and j["key"] is None and j["sell_cost"] is None for j in jokers)


def test_amber_acorn_deals_the_jokers_new_ids() -> None:
    """The player can't follow a joker through Amber Acorn's flip and shuffle, so its id
    changes; turning face up again keeps it."""
    game = Game.new("b_red", 1, "PVIEW_AMBER")
    game.apply_setup({"op": "force_boss", "key": "bl_final_acorn"})
    for key in ("j_joker", "j_greedy_joker", "j_lusty_joker"):
        game.debug_add(key)
    before = {c["id"] for c in game.player_view(KEY)["jokers"]}
    game.step(SelectBlind())  # force_boss puts the boss on deck
    v = game.player_view(KEY)
    assert v["blind"]["key"] == "bl_final_acorn"
    assert all(j["face_down"] for j in v["jokers"])
    down = [j["id"] for j in v["jokers"]]
    assert not set(down) & before
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    up = game.player_view(KEY)["jokers"]
    assert not any(j["face_down"] for j in up)
    assert sorted(j["id"] for j in up) == sorted(down)


# ---------------------------------------------------------------------------
# The deck view
# ---------------------------------------------------------------------------


def _deck_counts(v: dict[str, Any], remaining: bool) -> Counter[str]:
    return Counter(e["front"] for e in v["deck"] if e["remaining"] == remaining)


def test_the_deck_view() -> None:
    game = _dealt("PVIEW_DECKVIEW")
    v = game.player_view(KEY)
    assert _deck_counts(v, True) == Counter(c.front for c in game.deck)
    assert _deck_counts(v, False) == Counter(c.front for c in game.hand)
    assert all(set(e) == {"key", "front", "suit", "rank", "edition", "seal", "perma_bonus",
                          "remaining"} for e in v["deck"])  # fmt: skip


def test_the_deck_view_counts_face_down_cards_as_remaining() -> None:
    """The game's Remaining view shows a card dealt face down as remaining, wherever it is
    (UI_definitions.lua:3264), until the boss is disabled (blind.lua:370-372)."""
    for n in range(50):
        game = Game.new("b_red", 1, f"PVIEW_WHEEL_{n}")
        game.apply_setup({"op": "force_boss", "key": "bl_wheel"})
        game.step(SelectBlind())
        down = [i for i, c in enumerate(game.hand) if c.face_down]
        if down:
            break
    else:
        pytest.fail("no face-down card in 50 Wheel deals")
    game.step(Discard(card_indices=(down[0],)))
    v = game.player_view(KEY)
    remaining = _deck_counts(v, True)
    face_down = [c.front for c in game.hand if c.face_down]
    discarded = [game.discard_pile[0].front]
    assert remaining == Counter([c.front for c in game.deck] + face_down + discarded)


# ---------------------------------------------------------------------------
# Shaking jokers (#62)
# ---------------------------------------------------------------------------


def _joker(game: Game, key: str) -> dict[str, Any]:
    (j,) = [
        j for j, c in zip(game.player_view(KEY)["jokers"], game.jokers, strict=True) if c.key == key
    ]
    return j


def test_dna_shakes_until_a_hand_is_played() -> None:
    game = Game.new("b_red", 1, "PVIEW_DNA")
    game.debug_add("j_dna")
    assert not _joker(game, "j_dna")["shaking"]
    game.step(SelectBlind())
    assert _joker(game, "j_dna")["shaking"]
    game.step(Discard(card_indices=(0,)))
    assert _joker(game, "j_dna")["shaking"]
    game.step(PlayHand(card_indices=(0, 1)))
    assert not _joker(game, "j_dna")["shaking"]


def test_trading_card_shakes_until_the_first_discard() -> None:
    game = Game.new("b_red", 1, "PVIEW_TRADING")
    game.debug_add("j_trading")
    game.step(SelectBlind())
    assert _joker(game, "j_trading")["shaking"]
    game.step(PlayHand(card_indices=(0, 1)))
    assert _joker(game, "j_trading")["shaking"]
    game.step(Discard(card_indices=(0, 1)))
    assert not _joker(game, "j_trading")["shaking"]


def test_trading_card_stops_with_the_round() -> None:
    """``G.RESET_JIGGLES`` (state_events.lua:95) stops it at the round's end, discard or not."""
    game = Game.new("b_red", 1, "PVIEW_TRADING_END")
    game.debug_add("j_trading")
    game.step(SelectBlind())
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    assert game.phase == GamePhase.ROUND_EVAL
    assert not _joker(game, "j_trading")["shaking"]


def test_a_joker_added_after_the_first_draw_doesnt_shake() -> None:
    """The shake starts on the first hand drawn: a DNA that comes later stays still."""
    game = _dealt("PVIEW_DNA_LATE")
    game.debug_add("j_dna")
    assert not _joker(game, "j_dna")["shaking"]


def test_blueprint_doesnt_shake() -> None:
    game = Game.new("b_red", 1, "PVIEW_BLUEPRINT")
    game.debug_add("j_blueprint")
    game.debug_add("j_dna")
    game.step(SelectBlind())
    assert _joker(game, "j_dna")["shaking"]
    assert not _joker(game, "j_blueprint")["shaking"]


def test_loyalty_card_shakes_before_its_x_mult() -> None:
    game = Game.new("b_red", 1, "PVIEW_LOYALTY")
    game.debug_add("j_loyalty_card")
    game.step(SelectBlind())
    game.debug_set(hands=20)
    seen = 0
    for _ in range(12):
        game.step(PlayHand(card_indices=(0,)))
        loyalty = next(c for c in game.jokers if c.key == "j_loyalty_card")
        ready = loyalty.ability["loyalty_remaining"] == 0
        assert _joker(game, "j_loyalty_card")["shaking"] == ready
        seen += ready
        if game.phase != GamePhase.SELECTING_HAND:
            break
    assert seen >= 1


def test_invisible_joker_shakes_once_ready() -> None:
    game = Game.new("b_red", 1, "PVIEW_INVISIBLE")
    game.debug_add("j_invisible")
    for rounds in range(1, 4):
        game.step(SelectBlind())
        game.debug_set(chips=game.blind.chips)
        game.step(PlayHand(card_indices=(0,)))
        ready = rounds >= 2
        assert _joker(game, "j_invisible")["shaking"] == ready, rounds
        game.step(CashOut())
        assert _joker(game, "j_invisible")["shaking"] == ready
        game.step(NextRound())


def test_shaking_survives_neither_a_snapshot_nor_the_hash() -> None:
    """The shake is an event in the game, outside the state: not in snapshots or the hash."""
    game = Game.new("b_red", 1, "PVIEW_DNA_HASH")
    game.debug_add("j_dna")
    other = game.clone()
    game.step(SelectBlind())
    other.step(SelectBlind())
    assert _joker(game, "j_dna")["shaking"]
    assert game.state_hash() == Game.from_dict(game.to_dict()).state_hash()
    assert not _joker(Game.from_dict(game.to_dict()), "j_dna")["shaking"]
