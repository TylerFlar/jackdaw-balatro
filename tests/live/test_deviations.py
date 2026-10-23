"""The deviation catalog: every entry documented and well-formed; the predicates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jackdaw.bridge.backend import RPCError
from jackdaw.bridge.deviations import (
    CATALOG,
    Mode,
    RunMemory,
    StepContext,
    by_id,
    classify,
    refusal_entry,
    veto_reason,
)
from jackdaw.bridge.viewdiff import FieldDiff
from jackdaw.core import Game
from jackdaw.engine.actions import PlayHand, SelectBlind, SkipBlind

ROOT = Path(__file__).resolve().parents[2]


def _anchors(doc: Path) -> set[str]:
    text = doc.read_text(encoding="utf-8")
    ids = set(re.findall(r'<a id="([^"]+)"></a>', text))
    for heading in re.findall(r"^#+ (.+)$", text, re.MULTILINE):
        ids.add(re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip().replace(" ", "-"))
    return ids


def test_catalog_entries_are_documented_and_well_formed() -> None:
    ids = [d.id for d in CATALOG]
    assert len(ids) == len(set(ids))
    for d in CATALOG:
        doc, _, anchor = d.source.partition("#")
        assert (ROOT / doc).is_file(), d.id
        if anchor:
            assert anchor in _anchors(ROOT / doc), f"{d.id}: no #{anchor} in {doc}"
        assert d.title and d.note, d.id
        if d.mode is Mode.RECONCILE:
            assert d.reconcile is not None and d.paths, d.id
        elif d.mode is Mode.VETO:
            assert d.veto is not None or d.refusal is not None, d.id
            # A veto explains differences only where its trigger says it may have missed.
            assert not d.paths or d.trigger is not None, d.id
        else:
            assert d.reconcile is None and d.veto is None, d.id


def test_every_smods_entry_is_in_the_catalog() -> None:
    text = (ROOT / "docs/parity/smods-diffs.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| \[([SB]\d\d)\]", text, re.MULTILINE))
    assert documented >= {"S01", "S13", "B01", "B04"}
    assert documented <= set(by_id())


def _ctx(game: Game, method: str, params: dict, action=None) -> StepContext:
    return StepContext(
        method, params, action, pre=game, pre_view=game.bot_view(), memory=RunMemory()
    )


def _in_round(seed: str, *adds: dict) -> Game:
    g = Game.new("b_red", 1, seed)
    g.step(SelectBlind())
    for add in adds:
        g.debug_add(add.pop("key"), **add)
    return g


def _hand_positions(g: Game, enhancement: str) -> list[int]:
    hand = g.bot_view()["hand"]["cards"]
    return [
        i for i, c in enumerate(hand) if (c["modifier"] or {}).get("enhancement") == enhancement
    ]


def test_s07_vetoes_four_fingers_with_three_wild_cards() -> None:
    wild = {"key": "H_A", "enhancement": "m_wild"}
    g = _in_round("S07VETO", {"key": "j_four_fingers"}, dict(wild), dict(wild), dict(wild))
    wilds = _hand_positions(g, "WILD")
    assert len(wilds) == 3
    vetoed = veto_reason(_ctx(g, "play", {"cards": wilds}))
    assert vetoed is not None and vetoed[0] == "S07"
    assert veto_reason(_ctx(g, "play", {"cards": wilds[:2]})) is None


def test_s06_fires_only_for_a_lone_suited_wild_card() -> None:
    g = _in_round("S06TRIG", {"key": "j_seeing_double"}, {"key": "H_A", "enhancement": "m_wild"})
    (wild,) = _hand_positions(g, "WILD")
    s06 = by_id()["S06"]
    assert s06.active(_ctx(g, "play", {"cards": [wild]}))
    hand = g.bot_view()["hand"]["cards"]
    other = next(i for i, c in enumerate(hand) if i != wild)
    # A High Card: the Wild Card alone scores, the other card is a kicker.
    pair_like = [wild, other]
    scored = g.scoring_cards(pair_like)
    assert scored is not None
    expected = len(scored[1]) == 1
    assert s06.active(_ctx(g, "play", {"cards": pair_like})) is expected
    plain = _in_round("S06TRIG", {"key": "j_seeing_double"})
    assert not s06.active(_ctx(plain, "play", {"cards": [0]}))


def test_b02_vetoes_leaving_out_the_forced_card() -> None:
    g = Game.new("b_red", 1, "B02VETO")
    g.apply_setup({"op": "force_boss", "key": "bl_final_bell"})
    g.step(SelectBlind())
    hand = g.bot_view()["hand"]["cards"]
    forced = next(i for i, c in enumerate(hand) if (c["state"] or {}).get("highlight"))
    others = [i for i in range(len(hand)) if i != forced][:3]
    vetoed = veto_reason(_ctx(g, "play", {"cards": others}))
    assert vetoed is not None and vetoed[0] == "B02"
    assert veto_reason(_ctx(g, "play", {"cards": [*others, forced]})) is None
    assert veto_reason(_ctx(g, "discard", {"cards": others})) is not None


def test_s13_vetoes_crimson_heart_with_every_joker_debuffed() -> None:
    g = Game.new("b_red", 1, "S13VETO")
    g.apply_setup({"op": "force_boss", "key": "bl_final_heart"})
    g.step(SelectBlind())
    g.debug_add("j_joker")
    g.debug_add("j_greedy_joker")
    assert veto_reason(_ctx(g, "play", {"cards": [0]})) is None
    snap = g.to_dict()
    for card in snap["areas"]["jokers"]:
        card["debuff"] = True
    g.replace_state(snap)
    vetoed = veto_reason(_ctx(g, "play", {"cards": [0]}))
    assert vetoed is not None and vetoed[0] == "S13"


def _seed_with_small_tag(tags: set[str]) -> str:
    for n in range(2000):
        seed = f"PACKTAG{n}"
        if Game.new("b_red", 1, seed).to_dict()["round_resets"]["blind_tags"]["Small"] in tags:
            return seed
    raise AssertionError("no seed found")


def test_skip_for_a_pack_tag_is_vetoed() -> None:
    seed = _seed_with_small_tag({"tag_charm", "tag_meteor", "tag_buffoon", "tag_standard"})
    g = Game.new("b_red", 1, seed)
    vetoed = veto_reason(_ctx(g, "skip", {}, SkipBlind()))
    assert vetoed is not None and vetoed[0] == "BB-SKIP-PACK"
    plain = Game.new("b_red", 1, _seed_with_small_tag({"tag_economy", "tag_investment"}))
    assert veto_reason(_ctx(plain, "skip", {}, SkipBlind())) is None


def test_refusals_fall_under_their_entries() -> None:
    g = Game.new("b_red", 1, "REFUSAL")
    ctx = _ctx(g, "pack", {"card": 0})
    ctx.refusal = RPCError(
        -32003,
        "Cannot select 'Judgement' from the pack: vanilla greys this pick out "
        "(it cannot be used now)",
    )
    assert refusal_entry(ctx) == "B01"
    ctx.refusal = RPCError(
        -32003, "Cannot sell 'Joker': the card is eternal (vanilla's sell button refuses it)"
    )
    assert refusal_entry(ctx) == "BB-ETERNAL"
    ctx.refusal = RPCError(-32000, "Failed to encode response: unexpected number value 'inf'")
    assert refusal_entry(ctx) == "BB-NONFINITE"
    ctx.refusal = RPCError(-32001, "Invalid card index: 9")
    assert refusal_entry(ctx) is None


@pytest.mark.parametrize(
    ("diff", "state", "verdict"),
    [
        # S11: the unseeded variant of the first shop's Buffoon pack.
        (
            FieldDiff("packs.cards[0].key", "p_buffoon_normal_1", "p_buffoon_normal_2"),
            "SHOP",
            "ignored",
        ),
        (FieldDiff("packs.cards[0].key", "p_arcana_normal_1", "p_arcana_normal_2"), "SHOP", "new"),
        # BB-SHOP-AREAS: the removed shop's areas, outside the shop only.
        (FieldDiff("shop.count", 2, 0), "BLIND_SELECT", "ignored"),
        (FieldDiff("shop.count", 2, 0), "SHOP", "new"),
        (FieldDiff("money", 4, 5), "SHOP", "new"),
    ],
)
def test_ignore_entries(diff: FieldDiff, state: str, verdict: str) -> None:
    g = Game.new("b_red", 1, "IGNORES")
    view = g.bot_view()
    view["state"] = state
    ctx = StepContext("gamestate", {}, pre=g, pre_view=view, post=g, post_view=view)
    cls = classify([diff], ctx)
    assert (verdict == "ignored") == bool(cls.ignored)
    assert (verdict == "new") == bool(cls.unclassified)


def test_s10_reconciles_a_chip_but_not_more() -> None:
    g = _in_round("S10ACC")
    ctx = StepContext(
        "play", {"cards": [0]}, pre=g, pre_view=g.bot_view(), post=g, post_view=g.bot_view()
    )
    assert classify([FieldDiff("round.chips", 120, 121)], ctx).reconcile == {
        "S10": [FieldDiff("round.chips", 120, 121)]
    }
    assert classify([FieldDiff("round.chips", 120, 125)], ctx).unclassified
    other = StepContext(
        "discard", {"cards": [0]}, pre=g, pre_view=g.bot_view(), post=g, post_view=g.bot_view()
    )
    assert classify([FieldDiff("round.chips", 120, 121)], other).unclassified


def test_play_hand_marker_is_never_vetoed_in_the_legal_list() -> None:
    # Markers stand for any selection; the vetoes judge concrete ones.
    g = Game.new("b_red", 1, "B02VETO")
    g.apply_setup({"op": "force_boss", "key": "bl_final_bell"})
    g.step(SelectBlind())
    assert PlayHand(()) in g.legal_actions("balatrobot")
