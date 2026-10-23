"""Parity cases for the blind, round-flow and challenge-rule rows of the ledger audit.

Each case builds a state with the oracle (:class:`tests.parity.pyengine.PyEngine`)
and checks the steps that exercise one ledger row with
:func:`tests.parity.rust_step.check_step`: the oracle's snapshot goes into Rust,
both engines take the action, and the snapshots afterwards must match.  The
assertions on the oracle's state afterwards show that the branch under test
ran.  Line references are into ``balatro_source/``.
"""

from __future__ import annotations

from typing import Any

from tests.parity.pyengine import PyEngine
from tests.parity.rust_step import Report, check_step


def _start(seed: str, challenge: str | None = None) -> dict[str, Any]:
    back = "b_challenge" if challenge else "b_red"
    return {"back": back, "stake": 1, "seed": seed, "challenge": challenge}


def in_round(
    seed: str,
    jokers: tuple[str, ...] = (),
    *,
    boss: str | None = None,
    challenge: str | None = None,
    **sets: Any,
) -> PyEngine:
    """SELECTING_HAND of the first blind (or of ``boss``), with ``jokers`` added
    by BalatroBot's ``add``."""
    setup: list[dict[str, Any]] = []
    if boss:
        setup.append({"op": "force_boss", "key": boss})
    setup.append({"op": "action", "action": ["SelectBlind"]})
    setup += [{"op": "add", "key": k} for k in jokers]
    if sets:
        setup.append({"op": "set", **sets})
    return PyEngine(_start(seed, challenge), setup)


def at_blind_select(
    seed: str,
    jokers: tuple[str, ...] = (),
    *,
    boss: str | None = None,
    challenge: str | None = None,
) -> PyEngine:
    """The Big Blind's (or ``boss``'s) blind select, with ``jokers`` bought in the
    first shop: their ``setting_blind`` runs in the next, checked, SelectBlind."""
    setup: list[dict[str, Any]] = [
        {"op": "action", "action": ["SelectBlind"]},
        {"op": "set", "chips": 10**9},
        {"op": "action", "action": ["PlayHand", [0]]},
        {"op": "action", "action": ["CashOut"]},
    ]
    setup += [{"op": "add", "key": k} for k in jokers]
    setup.append({"op": "action", "action": ["NextRound"]})
    if boss:
        setup.append({"op": "force_boss", "key": boss})
    return PyEngine(_start(seed, challenge), setup)


def check(engine: PyEngine, *actions: list[Any]) -> dict[str, Any]:
    """Step both engines through ``actions``; every step must run in Rust and
    match.  Returns the oracle's snapshot afterwards."""
    report = Report()
    for i, action in enumerate(actions):
        assert check_step(engine, action, f"step {i} {action}", report), action
    found = [f"{w}: {'; '.join(d[:6])}" for items in report.mismatches.values() for w, d in items]
    assert not found, "\n".join(found[:5])
    assert sum(report.matched.values()) == len(actions), dict(report.unported)
    return engine.snapshot()


def jokers_of(snap: dict[str, Any]) -> list[str]:
    return [j["center"] for j in snap["areas"]["jokers"]]


# ---------------------------------------------------------------------------
# Blinds (blind.lua)
# ---------------------------------------------------------------------------


class TestTheSerpent:
    """state_events.lua:363-367: after the round's first play or discard, The
    Serpent draws min(#deck, 3) cards whatever the hand size."""

    def test_draws_three_after_a_discard(self):
        eng = in_round("AU_SERPENT_D", boss="bl_serpent")
        before = len(eng.snapshot()["areas"]["hand"])
        snap = check(eng, ["Discard", [0]])
        assert len(snap["areas"]["hand"]) == before - 1 + 3

    def test_draws_three_after_a_play(self):
        eng = in_round("AU_SERPENT_P", boss="bl_serpent")
        before = len(eng.snapshot()["areas"]["hand"])
        snap = check(eng, ["PlayHand", [0, 1]])
        assert snap["phase"] == "selecting_hand"
        assert len(snap["areas"]["hand"]) == before - 2 + 3

    def test_a_disabled_serpent_fills_the_hand(self):
        eng = at_blind_select("AU_SERPENT_CHICOT", ("j_chicot",), boss="bl_serpent")
        snap = check(eng, ["SelectBlind"], ["Discard", [0, 1, 2, 3]])
        assert snap["blind"]["disabled"]
        assert len(snap["areas"]["hand"]) == snap["game"]["hand_size"]


class TestDisable:
    """Blind:disable (blind.lua:356-415) through Chicot's setting_blind."""

    def test_the_needle_gives_the_hands_back(self):
        eng = at_blind_select("AU_NEEDLE", ("j_chicot",), boss="bl_needle")
        snap = check(eng, ["SelectBlind"])
        assert snap["blind"]["disabled"]
        assert snap["current_round"]["hands_left"] == snap["round_resets"]["hands"]

    def test_violet_vessel_is_divided_by_three(self):
        target = check(at_blind_select("AU_VESSEL", boss="bl_final_vessel"), ["SelectBlind"])
        eng = at_blind_select("AU_VESSEL", ("j_chicot",), boss="bl_final_vessel")
        snap = check(eng, ["SelectBlind"])
        assert snap["blind"]["chips"] == target["blind"]["chips"] / 3

    def test_cerulean_bell_forces_nothing(self):
        eng = at_blind_select("AU_BELL", ("j_chicot",), boss="bl_final_bell")
        snap = check(eng, ["SelectBlind"], ["Discard", [0]])
        assert not any(c["ability"].get("forced_selection") for c in snap["areas"]["hand"])

    def test_crimson_heart_debuffs_no_joker(self):
        eng = at_blind_select("AU_HEART", ("j_chicot", "j_joker"), boss="bl_final_heart")
        snap = check(eng, ["SelectBlind"], ["PlayHand", [0]])
        assert not any(j["debuff"] for j in snap["areas"]["jokers"])

    def test_face_down_bosses_deal_face_up(self):
        for boss in ("bl_wheel", "bl_house", "bl_mark", "bl_fish"):
            eng = at_blind_select(f"AU_FLIP_{boss}", ("j_chicot",), boss=boss)
            snap = check(eng, ["SelectBlind"], ["PlayHand", [0]])
            assert all(c["facing"] == "front" for c in snap["areas"]["hand"]), boss


class TestVerdantLeaf:
    def test_selling_a_joker_disables_it(self):
        """card.lua:1616-1621 (G08)."""
        eng = in_round("AU_LEAF", ("j_joker", "j_greedy_joker"), boss="bl_final_leaf")
        snap = check(eng, ["SellCard", "jokers", 0])
        assert snap["blind"]["disabled"]
        assert not any(c["debuff"] for c in snap["areas"]["hand"])

    def test_selling_a_consumable_keeps_it(self):
        eng = in_round("AU_LEAF_C", ("c_fool",), boss="bl_final_leaf")
        snap = check(eng, ["SellCard", "consumables", 0])
        assert not snap["blind"]["disabled"]


class TestThePlant:
    def test_debuffs_face_cards_when_set(self):
        eng = at_blind_select("AU_PLANT", boss="bl_plant")
        snap = check(eng, ["SelectBlind"])
        owned = snap["areas"]["deck"] + snap["areas"]["hand"]
        faces = [c for c in owned if c["base"]["value"] in ("Jack", "Queen", "King")]
        assert faces and all(c["debuff"] for c in faces)


class TestPinnedJoker:
    def test_stays_first_after_amber_acorns_shuffles(self):
        """G200: align_cards puts pinned jokers back in front every frame
        (cardarea.lua:528), so Amber Acorn's shuffles (blind.lua:196-203) leave a
        pinned joker leftmost."""
        for seed in ("PINACORN1", "PINACORN4"):
            eng = at_blind_select(
                seed,
                ("j_joker", "j_greedy_joker", "j_lusty_joker"),
                boss="bl_final_acorn",
                challenge="c_knife_1",
            )
            snap = check(eng, ["SelectBlind"])
            assert jokers_of(snap)[0] == "j_ceremonial", seed


# ---------------------------------------------------------------------------
# Round flow (cardarea.lua, state_events.lua)
# ---------------------------------------------------------------------------


class TestHandSizeGrowsMidRound:
    def test_selling_stuntman_draws_two(self):
        """cardarea.lua:94-111 (G37): a positive change_size in SELECTING_HAND
        draws the difference."""
        eng = in_round("AU_STUNTMAN", ("j_stuntman",))
        before = eng.snapshot()
        snap = check(eng, ["SellCard", "jokers", 0])
        assert snap["game"]["hand_size"] == before["game"]["hand_size"] + 2
        assert len(snap["areas"]["hand"]) == len(before["areas"]["hand"]) + 2


class TestRunningOutOfCards:
    def test_the_round_ends_when_hand_and_deck_are_empty(self):
        """game.lua:3056-3058, 3063-3064 (G36): with hand, deck and play empty the
        round ends; short of the target the run is lost."""
        eng = in_round("AU_EMPTY")
        eng.gs["deck"] = []
        check(eng, ["Discard", [0, 1, 2, 3, 4]])
        snap = check(eng, ["Discard", [0, 1, 2]])
        assert snap["areas"]["hand"] == [] and snap["areas"]["deck"] == []
        assert snap["phase"] == "game_over"

    def test_no_hand_size_is_game_over(self):
        """state_events.lua:356-360 (G36): drawing to an empty hand that can hold
        no cards ends the run."""
        eng = PyEngine(_start("AU_NO_HAND"))
        eng.gs["hand_size"] = 0
        snap = check(eng, ["SelectBlind"])
        assert snap["phase"] == "game_over"


class TestPlayedCardsScoreInHandOrder:
    def test_click_order_is_sorted_by_position(self):
        """state_events.lua:463 (G49): the highlighted cards are sorted by their
        position in the hand before they are played."""
        eng = in_round("AU_ORDER")
        hand = [c["sort_id"] for c in eng.snapshot()["areas"]["hand"]]
        snap = check(eng, ["PlayHand", [4, 0, 2]])
        played = [c["sort_id"] for c in snap["areas"]["discard"]]
        assert played == [hand[0], hand[2], hand[4]]


class TestWinning:
    def test_beating_the_final_boss_wins_and_a_later_loss_keeps_it(self):
        """state_events.lua:111-114 sets G.GAME.won; nothing clears it (G48)."""
        eng = at_blind_select("AU_WIN", boss="bl_wall")
        eng.gs["round_resets"]["ante"] = 8
        eng.gs["round_resets"]["blind_ante"] = 8
        check(eng, ["SelectBlind"])
        eng.gs["chips"] = 10**12
        snap = check(eng, ["PlayHand", [0]])
        assert snap["run"]["won"]
        snap = check(eng, ["CashOut"], ["NextRound"], ["SelectBlind"])
        eng.gs["current_round"]["hands_left"] = 1
        snap = check(eng, ["PlayHand", [0]])
        assert snap["phase"] == "game_over"
        assert snap["run"]["won"]


# ---------------------------------------------------------------------------
# Challenge rules
# ---------------------------------------------------------------------------


class TestChallengeRules:
    def test_golden_needle_discards_cost_a_dollar(self):
        """state_events.lua:433-435."""
        eng = in_round("AU_NEEDLE_COST", challenge="c_golden_needle_1")
        before = eng.snapshot()["game"]["dollars"]
        snap = check(eng, ["Discard", [0]])
        assert snap["game"]["dollars"] == before - 1

    def test_typecast_rules_apply_after_the_ante_four_boss(self):
        """state_events.lua:240-247: every joker turns Eternal and the joker
        slots drop to 0."""
        eng = at_blind_select(
            "AU_TYPECAST", ("j_joker", "j_greedy_joker"), boss="bl_wall", challenge="c_typecast_1"
        )
        eng.gs["round_resets"]["ante"] = 4
        check(eng, ["SelectBlind"])
        eng.gs["chips"] = 10**12
        snap = check(eng, ["PlayHand", [0]])
        assert snap["game"]["joker_slots"] == 0
        assert all(j["stickers"]["eternal"] for j in snap["areas"]["jokers"])
