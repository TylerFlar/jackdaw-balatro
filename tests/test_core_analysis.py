"""The analysis entry points: preview_play(rolls=...), Game.roll_bound and the analysis setters,
each against the route it replaces (replace_state of an edited to_dict())."""

from __future__ import annotations

import copy

import pytest

from jackdaw.core import Game, GameBuilder, IllegalActionError
from jackdaw.engine.actions import PlayHand, SelectBlind


def _round(seed: str) -> Game:
    g = Game.new("b_red", 1, seed)
    g.step(SelectBlind())
    return g


def _outcome(board: Game, cards: tuple[int, ...]):
    t = board.clone()
    try:
        t.step(PlayHand(card_indices=cards))
    except IllegalActionError as e:
        return ("refused", str(e))
    return (t.last_score, t.state_hash(), [c.sort_id for c in t.hand])


def _same_board(a: Game, b: Game) -> None:
    assert a.state_hash() == b.state_hash()
    assert a.bot_view() == b.bot_view()  # carries the highlights, which the snapshot does not
    n = len(a.hand)
    for cards in [(i,) for i in range(n)] + [(0, 1), (0, 1, 2), tuple(range(min(5, n)))]:
        assert _outcome(a, cards) == _outcome(b, cards), cards


def _replaced(g: Game, edit) -> Game:
    d = copy.deepcopy(g.to_dict())
    edit(d)
    b = g.clone()
    b.replace_state(d)
    return b


# ---------------------------------------- roll bounds
def test_lucky_card_low_and_high() -> None:
    g = (
        GameBuilder("b_red", seed="ANLY0001")
        .select_blind()
        .card("H_A", enhancement="m_lucky")
        .build()
    )
    lucky = (len(g.hand) - 1,)
    before = g.state_hash()
    low, real, high = (g.preview_play(lucky, rolls=r) for r in ("low", None, "high"))
    assert g.state_hash() == before
    assert low.total <= real.total <= high.total
    assert (high.mult - low.mult, high.dollars - low.dollars) == (
        20,
        20,
    )  # Lucky Card: +20 mult, $20


def test_misprint_low_and_high() -> None:
    g = GameBuilder("b_red", seed="ANLY0002").select_blind().joker("j_misprint").build()
    low, high = g.preview_play((0,), rolls="low"), g.preview_play((0,), rolls="high")
    assert high.mult - low.mult == 23  # Misprint: +0 to +23 mult


def test_bound_rolls_draw_nothing() -> None:
    g = (
        GameBuilder("b_red", seed="ANLY0003")
        .select_blind()
        .card("H_A", enhancement="m_lucky")
        .build()
    )
    lucky = (len(g.hand) - 1,)
    for bound in ("low", "high"):
        t = g.clone()
        t.roll_bound = bound
        t.step(PlayHand(card_indices=lucky))
        streams = t.to_dict()["rng"]["streams"]
        assert "lucky_mult" not in streams and "lucky_money" not in streams
    t = g.clone()
    t.step(PlayHand(card_indices=lucky))
    assert "lucky_mult" in t.to_dict()["rng"]["streams"]  # unbound: the play rolls


def test_roll_bound_is_runtime_only() -> None:
    g = _round("ANLY0004")
    assert g.roll_bound is None
    b = g.clone()
    b.roll_bound = "high"
    assert b.roll_bound == "high" and b.clone().roll_bound == "high"
    assert b.state_hash() == g.state_hash() and b.to_dict() == g.to_dict()
    b.replace_state(b.to_dict())
    assert b.roll_bound is None  # a snapshot carries no bound
    with pytest.raises(ValueError):
        b.roll_bound = "middle"
    # Unbound, a play is the game's own play.
    for cards in [(0,), (0, 1, 2)]:
        assert g.preview_play(cards) == g.preview_play(cards, rolls=None)


# ---------------------------------------- analysis setters
def test_set_hand_is_replace_state_of_the_edited_snapshot() -> None:
    g = _round("ANLY0005")
    hand, deck = [c.sort_id for c in g.hand], [c.sort_id for c in g.deck]
    want = [hand[1], deck[0], hand[3], deck[-1], hand[0], deck[7], hand[5], hand[6]]

    def edit(d):
        pool = {c["sort_id"]: c for c in d["areas"]["hand"] + d["areas"]["deck"]}
        rest = [c for c in d["areas"]["deck"] + d["areas"]["hand"] if c["sort_id"] not in want]
        d["areas"]["hand"] = [pool[i] for i in want]
        d["areas"]["deck"] = rest

    a = g.clone()
    a.debug_set_hand(want)
    assert [c.sort_id for c in a.hand] == want
    assert len(a.deck) == len(g.deck)
    assert not any(c.face_down for c in a.hand)
    _same_board(a, _replaced(g, edit))


def test_set_hand_refuses_and_leaves_the_run() -> None:
    g = _round("ANLY0006")
    hand = [c.sort_id for c in g.hand]
    before = g.state_hash()
    for bad in ([hand[0], hand[0]], [hand[0], 10_000]):
        with pytest.raises(IllegalActionError):
            g.debug_set_hand(bad)
        assert g.state_hash() == before
    idle = Game.new("b_red", 1, "ANLY0007")
    with pytest.raises(IllegalActionError):
        idle.debug_set_hand([])


def test_blind_disabled_is_replace_state_of_the_edited_snapshot() -> None:
    g = GameBuilder("b_red", seed="ANLY0008").boss("bl_hook").select_blind().build()
    assert g.blind is not None and g.blind.key == "bl_hook" and not g.blind.disabled
    a = g.clone()
    a.debug_set_blind_disabled(True)
    assert a.blind is not None and a.blind.disabled

    def edit(d):
        d["blind"]["disabled"] = True

    _same_board(a, _replaced(g, edit))
    # The Hook no longer discards two held cards on the play.
    on, off = g.clone(), a.clone()
    on.step(PlayHand(card_indices=(0,)))
    off.step(PlayHand(card_indices=(0,)))
    assert (len(on.discard_pile), len(off.discard_pile)) == (3, 1)


def test_clear_forced_selection_is_replace_state_of_the_edited_snapshot() -> None:
    g = GameBuilder("b_red", seed="ANLY0009").boss("bl_final_bell").select_blind().build()
    forced = [i for i, c in enumerate(g.hand) if c.ability.get("forced_selection")]
    assert len(forced) == 1
    free = (next(i for i in range(len(g.hand)) if i not in forced),)
    assert _outcome(g, free)[0] == "refused"  # the Bell's card must be played
    a = g.clone()
    a.debug_clear_forced_selection()

    def edit(d):
        for c in d["areas"]["hand"]:
            c["ability"]["forced_selection"] = None

    _same_board(a, _replaced(g, edit))
    assert _outcome(a, free)[0] != "refused"
