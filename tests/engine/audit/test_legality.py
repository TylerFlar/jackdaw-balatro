"""Legal actions and the steps that take them follow vanilla's button gates
(G201-G203) and drags (G206). Line references are into ``balatro_source/``."""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import (
    BuyCard,
    GamePhase,
    NextRound,
    OpenBooster,
    PlayHand,
    SelectBlind,
    SellCard,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run

from ._joker_helpers import (
    give_consumable,
    give_joker,
    new_run,
    select_blind,
    to_shop,
    win_round,
)


def _negative_joker_in_shop(seed: str) -> dict:
    gs = to_shop(new_run(seed))
    gs["dollars"] = 100
    card = create_joker("j_joker", edition={"negative": True}, hands_played=gs["hands_played"])
    card.set_cost(ante=gs["round_resets"]["ante"])
    gs["shop_cards"][0] = card
    return gs


class TestNegativeShopJokerNeedsRoom:
    """G201: check_for_buy_space (button_callbacks.lua:2392-2402) lets a Negative
    joker in only while the jokers number fewer than the limit plus one."""

    def test_refused_when_the_jokers_already_exceed_the_limit(self):
        gs = _negative_joker_in_shop("AUDIT_G201_FULL")
        give_joker(gs, "j_greedy_joker")
        gs["joker_slots"] = 0  # Typecast's set_joker_slots_ante (state_events.lua:245-247)
        assert BuyCard(shop_index=0) not in get_legal_actions(gs)
        with pytest.raises(IllegalActionError):
            step(gs, BuyCard(shop_index=0))

    def test_bought_into_the_extra_slot(self):
        gs = _negative_joker_in_shop("AUDIT_G201_ROOM")
        give_joker(gs, "j_greedy_joker")
        gs["joker_slots"] = 1
        assert BuyCard(shop_index=0) in get_legal_actions(gs)
        step(gs, BuyCard(shop_index=0))
        assert [j.center_key for j in gs["jokers"]] == ["j_greedy_joker", "j_joker"]


def _open_pack(
    seed: str,
    key: str = "p_arcana_normal_1",
    *,
    jokers: tuple[str, ...] = (),
    consumables: tuple[str, ...] = (),
) -> dict:
    """The first shop's ``key`` pack, opened with ``jokers`` and ``consumables`` owned."""
    gs = to_shop(new_run(seed))
    for k in jokers:
        give_joker(gs, k)
    for k in consumables:
        give_consumable(gs, k)
    pack = Card()
    pack.set_ability(key)
    pack.set_cost()
    gs["shop_boosters"] = [pack]
    gs["dollars"] = 50
    step(gs, OpenBooster(card_index=0))
    assert gs["phase"] == GamePhase.PACK_OPENING
    return gs


def _arcana_pack(seed: str) -> dict:
    return _open_pack(seed)


class TestSkipPackNeedsTheDealtHand:
    """G202: can_skip_booster (button_callbacks.lua:2132-2141) lets an Arcana or
    Spectral pack go only once its hand is dealt (or the hand holds no cards)."""

    def test_an_empty_hand_blocks_the_skip(self):
        gs = _arcana_pack("AUDIT_G202_EMPTY")
        gs["deck"].extend(gs["hand"])
        gs["hand"].clear()
        assert SkipPack() not in get_legal_actions(gs)
        with pytest.raises(IllegalActionError):
            step(gs, SkipPack())

    def test_a_dealt_hand_allows_it(self):
        gs = _arcana_pack("AUDIT_G202_DEALT")
        assert gs["hand"]
        assert SkipPack() in get_legal_actions(gs)
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP

    def test_no_hand_size_allows_it(self):
        gs = _arcana_pack("AUDIT_G202_NOSIZE")
        gs["deck"].extend(gs["hand"])
        gs["hand"].clear()
        gs["hand_size"] = 0
        assert SkipPack() in get_legal_actions(gs)
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP


class TestSellingAtRoundEval:
    """G203: Card:can_sell_card (card.lua:1640-1653) has no state gate, so a joker
    can be sold on the cash-out screen."""

    def test_a_joker_can_be_sold_before_cashing_out(self):
        gs = select_blind(new_run("AUDIT_G203"))
        joker = give_joker(gs, "j_joker")
        win_round(gs)
        assert gs["phase"] == GamePhase.ROUND_EVAL
        action = SellCard(area="jokers", card_index=0)
        assert action in get_legal_actions(gs)
        before = gs["dollars"]
        step(gs, action)
        assert gs["jokers"] == []
        assert gs["dollars"] == before + joker.sell_cost
        assert gs["phase"] == GamePhase.ROUND_EVAL


def _jokers(gs: dict) -> list[str]:
    return [j.center_key for j in gs["jokers"]]


_HAND_SWAPS = (SwapHandLeft, SwapHandRight)


class TestJokersReorderAtEveryDecisionPoint:
    """G206: the jokers' area has type 'joker' (game.lua:2241-2245), whose cards
    CardArea:set_ranks lets be dragged (cardarea.lua:214-227), and align_cards
    re-sorts it by position every frame in any state (cardarea.lua:509-529). The
    controller is free at every decision point, so a drag reorders the jokers at
    blind select, on the cash-out screen and in any pack too."""

    def test_at_blind_select(self):
        gs = new_run("AUDIT_G206_BLIND")
        give_joker(gs, "j_joker")
        give_joker(gs, "j_greedy_joker")
        assert SwapJokersRight(idx=0) in get_legal_actions(gs)
        step(gs, SwapJokersRight(idx=0))
        assert _jokers(gs) == ["j_greedy_joker", "j_joker"]
        assert gs["phase"] == GamePhase.BLIND_SELECT

    def test_the_order_at_blind_select_picks_the_daggers_victim(self):
        """Ceremonial Dagger slices the joker to its right when the blind is set
        (card.lua:2561-2580, from state_events.lua:334-337), so a joker that
        arrived since the shop (a skip's Buffoon pack, The Soul) can be moved
        into or out of its reach first."""
        gs = new_run("AUDIT_G206_DAGGER")
        for key in ("j_ceremonial", "j_joker", "j_greedy_joker"):
            give_joker(gs, key)
        step(gs, SwapJokersRight(idx=1))
        assert _jokers(gs) == ["j_ceremonial", "j_greedy_joker", "j_joker"]
        select_blind(gs)
        assert _jokers(gs) == ["j_ceremonial", "j_joker"]

    def test_on_the_cash_out_screen(self):
        gs = select_blind(new_run("AUDIT_G206_EVAL"))
        give_joker(gs, "j_joker")
        give_joker(gs, "j_greedy_joker")
        win_round(gs)
        assert SwapJokersLeft(idx=1) in get_legal_actions(gs)
        step(gs, SwapJokersLeft(idx=1))
        assert _jokers(gs) == ["j_greedy_joker", "j_joker"]
        assert gs["phase"] == GamePhase.ROUND_EVAL

    def test_in_a_pack_that_deals_no_hand(self):
        gs = _open_pack(
            "AUDIT_G206_BUFFOON", "p_buffoon_normal_1", jokers=("j_joker", "j_greedy_joker")
        )
        legal = get_legal_actions(gs)
        assert SwapJokersRight(idx=0) in legal
        assert not any(isinstance(a, _HAND_SWAPS) for a in legal)
        step(gs, SwapJokersRight(idx=0))
        assert _jokers(gs) == ["j_greedy_joker", "j_joker"]
        with pytest.raises(IllegalActionError):
            step(gs, SwapHandRight(idx=0))  # the hand is empty

    def test_not_once_the_run_is_over(self):
        gs = new_run("AUDIT_G206_OVER")
        give_joker(gs, "j_joker")
        give_joker(gs, "j_greedy_joker")
        gs["phase"] = GamePhase.GAME_OVER
        assert get_legal_actions(gs) == []
        with pytest.raises(IllegalActionError):
            step(gs, SwapJokersRight(idx=0))


class TestAPacksHandReorders:
    """G206: an Arcana or Spectral pack deals a hand (game.lua:3371-3375,
    3422-3426), which align_cards lays out and re-sorts by position in
    TAROT_PACK and SPECTRAL_PACK as in a round (cardarea.lua:436-449), so a drag
    reorders it. The sort buttons are built only while choosing a hand
    (game.lua:3046-3050) and removed in the packs (game.lua:3342, 3393)."""

    def test_the_dealt_hand_can_be_reordered_but_not_sorted(self):
        gs = _arcana_pack("AUDIT_G206_ARCANA")
        first, second = gs["hand"][:2]
        legal = get_legal_actions(gs)
        assert SwapHandRight(idx=0) in legal and SwapHandLeft(idx=1) in legal
        assert not any(isinstance(a, SortHand) for a in legal)
        step(gs, SwapHandRight(idx=0))
        assert gs["hand"][:2] == [second, first]
        assert gs["phase"] == GamePhase.PACK_OPENING
        with pytest.raises(IllegalActionError):
            step(gs, SortHand(mode="suit"))

    def test_death_copies_the_card_moved_right(self):
        """Death copies the rightmost of its two cards by position
        (card.lua:1112-1117): the pack's hand order picks the direction."""
        gs = _open_pack("AUDIT_G206_DEATH", consumables=("c_death",))
        first, second = gs["hand"][:2]
        key = first.card_key
        assert second.card_key != key
        step(gs, SwapHandRight(idx=0))
        step(gs, UseConsumable(card_index=0, target_indices=(0, 1)))
        assert (first.card_key, second.card_key) == (key, key)

    def test_the_hand_goes_back_to_the_deck_in_its_order(self):
        """end_consumeable draws the hand back from the left
        (state_events.lua:1121-1126; cardarea.lua:76-80), each card to the
        deck's bottom (cardarea.lua:33-34): later deals before a shuffle see it."""
        gs = _arcana_pack("AUDIT_G206_RETURN")
        step(gs, SwapHandRight(idx=0))
        hand = list(gs["hand"])
        step(gs, SkipPack())
        assert gs["deck"][: len(hand)] == hand[::-1]


def _observatory_score(planets: tuple[str, ...]) -> float:
    def two(key: str, suit: str) -> Card:
        c = Card()
        c.set_base(key, suit, "2")
        c.set_ability("c_base")
        return c

    gs = initialize_run("b_red", 1, "AUDIT_G206_OBSERVATORY")
    step(gs, SelectBlind())
    gs["blind"].chips = 10**9  # don't end the round
    gs["hand"] = [two("H_2", "Hearts"), two("S_2", "Spades")]
    gs["jokers"] = [create_joker("j_joker")]
    gs["consumables"] = [create_consumable(k) for k in planets]
    gs["used_vouchers"]["v_observatory"] = True
    step(gs, PlayHand(card_indices=(0, 1)))
    return gs["chips"]


def _perkeo_copy(seed: str, reverse: bool) -> str:
    gs = to_shop(new_run(seed))
    give_joker(gs, "j_perkeo")
    for key in ("c_mercury", "c_venus"):
        give_consumable(gs, key)
    if reverse:
        gs["consumables"].reverse()
    step(gs, NextRound())
    copies = [c for c in gs["consumables"] if (c.edition or {}).get("negative")]
    assert len(copies) == 1
    return copies[0].center_key


class TestConsumableOrderChangesNothing:
    """G206: the consumables' area is a 'joker' area too (game.lua:2235-2239), so
    a drag reorders it, but no rule reads that order, and the simulator has no
    move for it. Its cards come after the jokers in the joker_main pass
    (state_events.lua:877-878), where only a Planet answers, with Observatory's
    same X1.5 (card.lua:2293-2302); Baseball Card skips them (no rarity,
    card.lua:3397). Perkeo draws by sort_id (card.lua:2414-2420;
    pseudorandom_element sorts by it, misc_functions.lua:260-261); Gift Card and
    find_joker only visit them (card.lua:3000-3005; misc_functions.lua:903-917)."""

    def test_observatory_scores_the_same_in_any_order(self):
        orders = [
            ("c_mercury", "c_venus", "c_mercury"),
            ("c_venus", "c_mercury", "c_mercury"),
            ("c_mercury", "c_mercury", "c_venus"),
        ]
        assert {_observatory_score(o) for o in orders} == {14 * (2 + 4) * 1.5 * 1.5}

    def test_perkeo_copies_the_same_card_in_any_order(self):
        for seed in ("AUDIT_G206_PERKEO1", "AUDIT_G206_PERKEO2", "AUDIT_G206_PERKEO3"):
            assert _perkeo_copy(seed, False) == _perkeo_copy(seed, True), seed
