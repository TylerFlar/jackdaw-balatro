"""The view differ: structural comparison, text fields left out, cards matched one to one."""

from __future__ import annotations

import copy

from jackdaw.bridge.viewdiff import ABSENT, FieldDiff, IdMap, diff_views, path_pattern
from jackdaw.core import Game
from jackdaw.engine.actions import SelectBlind
from tests.live.fakes import live_ids, renumber


def _view() -> dict:
    g = Game.new("b_red", 1, "VIEWDIFF")
    g.step(SelectBlind())
    return g.bot_view()


def test_a_view_equals_itself() -> None:
    v = _view()
    out = diff_views(v, copy.deepcopy(v), IdMap())
    assert out.diffs == []
    assert len(out.pairs) == 52  # every card paired, by position


def test_empty_tables_and_booleans() -> None:
    assert diff_views({"m": []}, {"m": {}}).diffs == []
    assert [d.path for d in diff_views({"won": False}, {"won": 0}).diffs] == ["won"]
    assert [d.path for d in diff_views({"a": 1}, {"a": 1.0, "b": 2}).diffs] == ["b"]
    (d,) = diff_views({"x": {"y": 1}}, {"x": {}}).diffs
    assert (d.path, d.sim, d.live) == ("x.y", 1, ABSENT)


def test_text_fields_are_not_compared() -> None:
    v = _view()
    live = copy.deepcopy(v)
    live["hand"]["cards"][0]["value"]["effect"] = "+4 Mult (a localized description)"
    live["blinds"]["boss"]["effect"] = "All Club cards are debuffed"
    live["blinds"]["small"]["tag_effect"] = "Next shop has a free Uncommon Joker"
    live["used_vouchers"] = {"v_grabber": "Permanently gain +1 hand per round"}
    v["used_vouchers"] = {"v_grabber": ""}
    assert diff_views(v, live).diffs == []
    # A voucher only one side redeemed is still a difference (the key is compared).
    live["used_vouchers"]["v_hone"] = "..."
    assert [d.path for d in diff_views(v, live).diffs] == ["used_vouchers.v_hone"]


def test_card_fields_and_count() -> None:
    v = _view()
    live = copy.deepcopy(v)
    live["hand"]["cards"][2]["modifier"] = {"seal": "RED"}
    live["cards"]["cards"].pop()
    live["cards"]["count"] -= 1
    paths = [d.path for d in diff_views(v, live).diffs]
    assert "hand.cards[2].modifier.seal" in paths
    assert "cards.count" in paths
    assert "cards.cards[43]" in paths  # the card the live deck lacks


def test_ids_are_matched_not_compared() -> None:
    v = _view()
    ids = IdMap()
    out = diff_views(v, renumber(v, live_ids), ids)
    assert out.diffs == []
    ids.commit(out.pairs)
    assert ids.live_to_sim[live_ids(1)] == 1
    # The same faces in the same places, but two identities crossed (two twin cards
    # the games keep in different orders): content alone can't tell, the pairing can.
    live = renumber(v, live_ids)
    deck = live["cards"]["cards"]
    deck[0]["id"], deck[1]["id"] = deck[1]["id"], deck[0]["id"]
    paths = [d.path for d in diff_views(v, live, ids).diffs]
    assert paths == ["cards.cards[0].id", "cards.cards[1].id"]


def test_path_patterns() -> None:
    p = path_pattern("hand.cards[*].state.debuff")
    assert p.fullmatch("hand.cards[3].state.debuff")
    assert not p.fullmatch("hands.Pair.level")
    whole = path_pattern("shop")
    assert whole.fullmatch("shop") and whole.fullmatch("shop.cards[0].key")
    assert not whole.fullmatch("shops")
    assert path_pattern("shop.cards[*].cost.**").fullmatch("shop.cards[1].cost.buy")


def test_field_diff_json_round_trip() -> None:
    d = FieldDiff("pack", "area with 3 cards", ABSENT, "x")
    assert FieldDiff.from_json(d.to_json()) == d
