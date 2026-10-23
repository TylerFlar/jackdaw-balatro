"""The Game facade's read-only views, check() and preview_play() (jackdaw.core.views)."""

from __future__ import annotations

import pytest

from jackdaw.core import CardView, Game, IllegalActionError
from jackdaw.engine.actions import BuyCard, PlayHand, SelectBlind


@pytest.fixture
def game() -> Game:
    g = Game.new("b_red", 1, "VIEWS001")
    g.step(SelectBlind())
    return g


def test_card_views_match_the_snapshot(game: Game) -> None:
    game.debug_add("j_joker", edition="foil")
    snap = game.to_dict()
    for area, cards in (("hand", game.hand), ("jokers", game.jokers), ("deck", game.deck)):
        assert [c.sort_id for c in cards] == [c["sort_id"] for c in snap["areas"][area]], area
    joker = game.jokers[0]
    assert isinstance(joker, CardView)
    assert (joker.key, joker.edition, joker.front) == ("j_joker", "foil", None)
    card = game.hand[0]
    assert card.front and card.suit and card.rank and card.key == "c_base"
    with pytest.raises(TypeError):
        card.ability["mult"] = 1  # read-only


def test_blind_levels_and_tags(game: Game) -> None:
    blind = game.blind
    assert blind is not None and blind.key == "bl_small" and blind.chips > 0
    levels = game.hand_levels
    assert levels["Pair"].level == 1 and levels["Pair"].visible
    assert not levels["Flush Five"].visible
    assert game.tags == ()
    assert game.last_score is None


def test_check_leaves_the_run_alone(game: Game) -> None:
    before = game.state_hash()
    assert game.check(PlayHand(card_indices=(0,))) is None
    assert game.check(BuyCard(shop_index=0)) is not None  # no shop during a round
    assert game.state_hash() == before


def test_preview_play_is_the_real_play(game: Game) -> None:
    before = game.state_hash()
    preview = game.preview_play((0, 1))
    assert game.state_hash() == before
    game.step(PlayHand(card_indices=(0, 1)))
    assert game.last_score == preview
    with pytest.raises(IllegalActionError):
        game.clone().preview_play(())
