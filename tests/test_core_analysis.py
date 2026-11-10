"""The analysis entry points: preview_play(rolls=...), Game.roll_bound and the analysis setters,
each against the route it replaces (replace_state of an edited to_dict())."""

from __future__ import annotations

import copy

import pytest

from jackdaw.core import Game, GameBuilder, IllegalActionError
from jackdaw.engine.actions import (
    CashOut,
    Discard,
    OpenBooster,
    PlayHand,
    SelectBlind,
    UseConsumable,
)


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


def _at(g: Game, bound: str | None, action) -> Game:
    """A copy of the run after ``action``, taken under ``bound``."""
    t = g.clone()
    t.roll_bound = bound
    t.step(action)
    return t


def _with_jokers(seed: str, *jokers: str) -> Game:
    """A round with these jokers, and a King and an 8 of Hearts as the hand's last two cards."""
    b = GameBuilder("b_red", seed=seed).select_blind()
    for key in jokers:
        b.joker(key)
    g = b.card("H_K").card("H_8").build()
    # The debug add polls an edition for a joker given none; these seeds poll none.
    assert all(j.edition is None for j in g.jokers), [j.edition for j in g.jokers]
    return g


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
    # A snapshot carries no bound: a run made from one starts without it.
    assert Game.from_dict(b.to_dict()).roll_bound is None
    with pytest.raises(ValueError):
        b.roll_bound = "middle"
    assert b.roll_bound == "high"
    # Unbound, a play is the game's own play.
    for cards in [(0,), (0, 1, 2)]:
        assert g.preview_play(cards) == g.preview_play(cards, rolls=None)


def test_replace_state_keeps_the_roll_bound() -> None:
    """The bound is the run's mode, as record_scoring is: a board edited through its snapshot
    still rolls at the bound, like one edited with a setter."""
    g = (
        GameBuilder("b_red", seed="ANLY0010")
        .boss("bl_hook")
        .select_blind()
        .card("H_A", enhancement="m_lucky")
        .build()
    )
    lucky = (len(g.hand) - 1,)
    a, b = g.clone(), g.clone()
    a.roll_bound = b.roll_bound = "high"
    a.debug_set_blind_disabled(True)
    d = copy.deepcopy(b.to_dict())
    d["blind"]["disabled"] = True
    b.replace_state(d)
    assert b.roll_bound == "high"
    _same_board(a, b)
    # The Lucky Card pays out on both, as preview_play(rolls="high") says.
    high = b.preview_play(lucky, rolls="high")
    assert high != b.preview_play(lucky, rolls="low")
    assert _outcome(a, lucky)[0] == _outcome(b, lucky)[0] == b.preview_play(lucky) == high


# ---------------------------------------- every chance roll follows the bound
@pytest.mark.parametrize(
    ("joker", "card", "read", "low", "high"),
    [
        # 1 in 4 to level up the hand played.
        ("j_space", -2, lambda t: t.hand_levels["High Card"].level, 1, 2),
        # 1 in 2 for X1.5 mult on a scored Heart.
        ("j_bloodstone", -2, lambda t: t.last_score.mult, 1, 1.5),
        # 1 in 2 for $2 on a scored face card.
        ("j_business", -2, lambda t: t.last_score.dollars, 0, 2),
        # 1 in 4 for a Tarot on a scored 8.
        ("j_8_ball", -1, lambda t: len(t.consumables), 0, 1),
    ],
    ids=["space", "bloodstone", "business", "8_ball"],
)
def test_a_chance_joker_follows_the_bound(joker, card, read, low, high) -> None:
    g = _with_jokers("ANLY0011", joker)
    play = PlayHand(card_indices=(len(g.hand) + card,))
    assert (read(_at(g, "low", play)), read(_at(g, "high", play))) == (low, high)


def test_reserved_parking_follows_the_bound() -> None:
    """1 in 2 for $1 from each face card held in hand."""
    g = _with_jokers("ANLY0011", "j_reserved_parking")
    eight = len(g.hand) - 1
    faces = sum(c.rank in ("Jack", "Queen", "King") for c in g.hand[:eight])
    assert faces >= 1  # the King of Hearts, at least
    low, high = (g.preview_play((eight,), rolls=r) for r in ("low", "high"))
    assert (low.dollars, high.dollars) == (0, faces)


def test_low_fires_a_chance_that_cannot_fail() -> None:
    """Oops! All 6s doubles Bloodstone's 1 in 2: certain, so it fires under "low" too."""
    g = _with_jokers("ANLY0012", "j_bloodstone", "j_oops")
    king = (len(g.hand) - 2,)
    assert g.preview_play(king, rolls="low").mult == 1.5
    assert g.preview_play(king, rolls="high").mult == 1.5


@pytest.mark.parametrize("joker", ["j_gros_michel", "j_cavendish"])
def test_an_end_of_round_chance_follows_the_bound(joker) -> None:
    """Gros Michel (1 in 6) and Cavendish (1 in 1000) go extinct at the end of the round under
    "high" and stay under "low": the bounds are the play's score, not the board it leaves."""
    g = GameBuilder("b_red", seed="ANLY0013").select_blind().joker(joker).chips(10**9).build()
    assert g.jokers[0].edition is None
    play = PlayHand(card_indices=(0,))
    low, high = _at(g, "low", play), _at(g, "high", play)
    assert low.phase == high.phase == "round_eval"
    assert [j.key for j in low.jokers] == [joker] and not high.jokers
    assert low.last_score == high.last_score


def test_hallucination_follows_the_bound() -> None:
    """1 in 2 for a Tarot when a booster pack is opened."""
    g = (
        GameBuilder("b_red", seed="ANLY0014")
        .select_blind()
        .joker("j_hallucination")
        .chips(10**9)
        .step(PlayHand(card_indices=(0,)), CashOut())
        .build()
    )
    assert g.jokers[0].edition is None and not g.consumables
    opened = OpenBooster(card_index=0)
    low, high = _at(g, "low", opened), _at(g, "high", opened)
    assert low.phase == high.phase == "pack_opening"
    assert (len(low.consumables), len(high.consumables)) == (0, 1)


def test_a_glass_card_shatters_under_high() -> None:
    """A harmful chance fires under "high" too: the Glass Card scores the same, then shatters
    (1 in 4); under "low" it goes to the discard pile."""
    g = (
        GameBuilder("b_red", seed="ANLY0015")
        .select_blind()
        .card("H_A", enhancement="m_glass")
        .build()
    )
    glass = g.hand[-1].sort_id
    play = PlayHand(card_indices=(len(g.hand) - 1,))
    low, high = _at(g, "low", play), _at(g, "high", play)
    assert low.last_score == high.last_score
    assert glass in [c.sort_id for c in low.discard_pile]
    left = [c.sort_id for area in (high.hand, high.deck, high.discard_pile) for c in area]
    assert glass not in left


def test_the_wheel_flips_every_drawn_card_or_none() -> None:
    """The Wheel: 1 in 7 for each card drawn to be face down."""
    g = GameBuilder("b_red", seed="ANLY0016").boss("bl_wheel").select_blind().build()
    held = {c.sort_id for c in g.hand}
    for bound, face_down in (("low", False), ("high", True)):
        t = _at(g, bound, Discard(card_indices=(0, 1, 2)))
        drawn = [c for c in t.hand if c.sort_id not in held]
        assert len(drawn) == 3 and all(c.face_down == face_down for c in drawn), bound


def test_the_wheel_of_fortune_follows_the_bound() -> None:
    """The Wheel of Fortune's 1 in 4 is a chance check too. Its joker and its edition come
    from the same stream as that roll ('wheel_of_fortune'): a miss under "low" draws nothing,
    and a hit under "high" draws only the two choices."""
    g = (
        GameBuilder("b_red", seed="ANLY0017")
        .select_blind()
        .joker("j_joker")
        .consumable("c_wheel_of_fortune")
        .build()
    )
    assert g.jokers[0].edition is None

    def stream(t: Game) -> float | None:
        return t.to_dict()["rng"]["streams"].get("wheel_of_fortune")

    use = UseConsumable(card_index=0)
    low, high = _at(g, "low", use), _at(g, "high", use)
    assert low.jokers[0].edition is None and high.jokers[0].edition is not None
    assert stream(low) is None and stream(high) is not None
    # Two Oops! All 6s make it certain: it hits under "low" as well, and in play.
    sure = (
        GameBuilder("b_red", seed="ANLY0017")
        .select_blind()
        .joker("j_oops")
        .joker("j_oops")
        .consumable("c_wheel_of_fortune")
        .build()
    )
    assert all(j.edition is None for j in sure.jokers)
    sure_low, sure_real = _at(sure, "low", use), _at(sure, None, use)
    assert sum(j.edition is not None for j in sure_low.jokers) == 1
    assert sum(j.edition is not None for j in sure_real.jokers) == 1
    # A bound hit draws the joker and the edition and nothing else, so both leave the stream
    # (one seed, one key) in one state; the real use draws its chance roll first.
    assert stream(high) == stream(sure_low) != stream(sure_real)


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
