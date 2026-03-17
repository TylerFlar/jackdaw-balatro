"""Tests for jackdaw.engine.actions — Action types and GamePhase enum.

Coverage
--------
* Every action type constructs with correct fields.
* All action types are frozen (immutable).
* GamePhase enum has all expected members.
* GamePhase members are string-valued (str mixin).
* Action union type includes all 17 action types.
* Default values (target_indices=None on BuyAndUse/UseConsumable).
* Equality and hashing (frozen dataclasses are hashable).
"""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import (
    Action,
    BuyAndUse,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    Reroll,
    RedeemVoucher,
    ReorderJokers,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    UseConsumable,
)


# ---------------------------------------------------------------------------
# GamePhase
# ---------------------------------------------------------------------------


class TestGamePhase:
    def test_all_phases_present(self):
        expected = {
            "BLIND_SELECT",
            "SELECTING_HAND",
            "ROUND_EVAL",
            "SHOP",
            "PACK_OPENING",
            "GAME_OVER",
        }
        assert {p.name for p in GamePhase} == expected

    def test_string_values(self):
        assert GamePhase.BLIND_SELECT == "blind_select"
        assert GamePhase.SHOP == "shop"
        assert GamePhase.GAME_OVER == "game_over"

    def test_is_str(self):
        assert isinstance(GamePhase.BLIND_SELECT, str)

    def test_lookup_by_value(self):
        assert GamePhase("shop") is GamePhase.SHOP


# ---------------------------------------------------------------------------
# Construction and fields
# ---------------------------------------------------------------------------


class TestActionConstruction:
    def test_play_hand(self):
        a = PlayHand(card_indices=(0, 2, 4))
        assert a.card_indices == (0, 2, 4)

    def test_discard(self):
        a = Discard(card_indices=(1, 3))
        assert a.card_indices == (1, 3)

    def test_select_blind(self):
        a = SelectBlind()
        assert isinstance(a, SelectBlind)

    def test_skip_blind(self):
        a = SkipBlind()
        assert isinstance(a, SkipBlind)

    def test_buy_card(self):
        a = BuyCard(shop_index=2)
        assert a.shop_index == 2

    def test_buy_and_use_no_targets(self):
        a = BuyAndUse(shop_index=0)
        assert a.shop_index == 0
        assert a.target_indices is None

    def test_buy_and_use_with_targets(self):
        a = BuyAndUse(shop_index=1, target_indices=(0, 3))
        assert a.target_indices == (0, 3)

    def test_sell_card(self):
        a = SellCard(area="jokers", card_index=0)
        assert a.area == "jokers"
        assert a.card_index == 0

    def test_use_consumable_no_targets(self):
        a = UseConsumable(card_index=1)
        assert a.card_index == 1
        assert a.target_indices is None

    def test_use_consumable_with_targets(self):
        a = UseConsumable(card_index=0, target_indices=(2, 5))
        assert a.target_indices == (2, 5)

    def test_redeem_voucher(self):
        a = RedeemVoucher(card_index=0)
        assert a.card_index == 0

    def test_open_booster(self):
        a = OpenBooster(card_index=1)
        assert a.card_index == 1

    def test_pick_pack_card(self):
        a = PickPackCard(card_index=2)
        assert a.card_index == 2

    def test_skip_pack(self):
        a = SkipPack()
        assert isinstance(a, SkipPack)

    def test_reroll(self):
        a = Reroll()
        assert isinstance(a, Reroll)

    def test_next_round(self):
        a = NextRound()
        assert isinstance(a, NextRound)

    def test_cash_out(self):
        a = CashOut()
        assert isinstance(a, CashOut)

    def test_sort_hand(self):
        a = SortHand(mode="rank")
        assert a.mode == "rank"

    def test_reorder_jokers(self):
        a = ReorderJokers(new_order=(2, 0, 1))
        assert a.new_order == (2, 0, 1)


# ---------------------------------------------------------------------------
# Frozen (immutable)
# ---------------------------------------------------------------------------


class TestFrozen:
    @pytest.mark.parametrize(
        "action",
        [
            PlayHand(card_indices=(0,)),
            Discard(card_indices=(1,)),
            SelectBlind(),
            SkipBlind(),
            BuyCard(shop_index=0),
            BuyAndUse(shop_index=0),
            SellCard(area="jokers", card_index=0),
            UseConsumable(card_index=0),
            RedeemVoucher(card_index=0),
            OpenBooster(card_index=0),
            PickPackCard(card_index=0),
            SkipPack(),
            Reroll(),
            NextRound(),
            CashOut(),
            SortHand(mode="suit"),
            ReorderJokers(new_order=(0,)),
        ],
    )
    def test_immutable(self, action):
        with pytest.raises(AttributeError):
            action.frozen_test_field = 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Hashable (frozen → usable in sets/dicts)
# ---------------------------------------------------------------------------


class TestHashable:
    def test_play_hand_hashable(self):
        a = PlayHand(card_indices=(0, 1))
        assert hash(a) is not None
        assert a in {a}

    def test_select_blind_hashable(self):
        assert hash(SelectBlind()) is not None

    def test_equal_actions_same_hash(self):
        a1 = BuyCard(shop_index=3)
        a2 = BuyCard(shop_index=3)
        assert a1 == a2
        assert hash(a1) == hash(a2)

    def test_different_actions_not_equal(self):
        assert BuyCard(shop_index=0) != BuyCard(shop_index=1)
        assert PlayHand(card_indices=(0,)) != Discard(card_indices=(0,))


# ---------------------------------------------------------------------------
# Action union
# ---------------------------------------------------------------------------


class TestActionUnion:
    ALL_TYPES = (
        PlayHand,
        Discard,
        SelectBlind,
        SkipBlind,
        BuyCard,
        BuyAndUse,
        SellCard,
        UseConsumable,
        RedeemVoucher,
        OpenBooster,
        PickPackCard,
        SkipPack,
        Reroll,
        NextRound,
        CashOut,
        SortHand,
        ReorderJokers,
    )

    def test_seventeen_action_types(self):
        assert len(self.ALL_TYPES) == 17

    @pytest.mark.parametrize("cls", ALL_TYPES)
    def test_each_type_in_union(self, cls):
        """Every action class is part of the Action union type."""
        # Action is a PEP 604 union; check via __args__
        assert cls in Action.__args__

    def test_isinstance_check(self):
        """isinstance works with PEP 604 union on Python 3.10+."""
        assert isinstance(PlayHand(card_indices=(0,)), PlayHand)
        assert isinstance(CashOut(), CashOut)
