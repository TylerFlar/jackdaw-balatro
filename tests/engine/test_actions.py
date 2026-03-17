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
    get_legal_actions,
)
from jackdaw.engine.card import Card


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


# ===========================================================================
# get_legal_actions
# ===========================================================================


def _card(key: str = "c_base", cost: int = 0, **kw) -> Card:
    """Quick Card factory for testing."""
    c = Card(center_key=key, cost=cost)
    c.ability = kw.pop("ability", {"set": "", "effect": ""})
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _joker_card(key: str = "j_joker", cost: int = 5, **kw) -> Card:
    c = Card(center_key=key, cost=cost)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# BLIND_SELECT
# ---------------------------------------------------------------------------


class TestLegalBlindSelect:
    def test_small_blind_select_and_skip(self):
        gs = {"phase": GamePhase.BLIND_SELECT, "blind_on_deck": "Small"}
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert SelectBlind in types
        assert SkipBlind in types

    def test_big_blind_select_and_skip(self):
        gs = {"phase": GamePhase.BLIND_SELECT, "blind_on_deck": "Big"}
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert SelectBlind in types
        assert SkipBlind in types

    def test_boss_blind_no_skip(self):
        gs = {"phase": GamePhase.BLIND_SELECT, "blind_on_deck": "Boss"}
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert SelectBlind in types
        assert SkipBlind not in types


# ---------------------------------------------------------------------------
# SELECTING_HAND
# ---------------------------------------------------------------------------


class TestLegalSelectingHand:
    def test_play_and_discard_available(self):
        hand = [_card() for _ in range(8)]
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 3, "discards_left": 2},
            "hand": hand,
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert PlayHand in types
        assert Discard in types

    def test_no_hands_left_no_play(self):
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 0, "discards_left": 2},
            "hand": [_card()],
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert PlayHand not in types
        assert Discard in types

    def test_no_discards_left_no_discard(self):
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 3, "discards_left": 0},
            "hand": [_card()],
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert PlayHand in types
        assert Discard not in types

    def test_empty_hand_nothing(self):
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 3, "discards_left": 2},
            "hand": [],
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert PlayHand not in types
        assert Discard not in types

    def test_sort_available_with_multiple_cards(self):
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 1, "discards_left": 0},
            "hand": [_card(), _card()],
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        sort_actions = [a for a in actions if isinstance(a, SortHand)]
        assert len(sort_actions) == 2
        modes = {a.mode for a in sort_actions}
        assert modes == {"rank", "suit"}

    def test_reorder_jokers_with_multiple(self):
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 1, "discards_left": 0},
            "hand": [_card()],
            "jokers": [_joker_card(), _joker_card()],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        assert any(isinstance(a, ReorderJokers) for a in actions)


# ---------------------------------------------------------------------------
# SHOP
# ---------------------------------------------------------------------------


class TestLegalShop:
    def test_buy_affordable_joker(self):
        joker = _joker_card(cost=5)
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 10,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [joker],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        buys = [a for a in actions if isinstance(a, BuyCard)]
        assert len(buys) == 1
        assert buys[0].shop_index == 0

    def test_cannot_buy_too_expensive(self):
        joker = _joker_card(cost=15)
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 10,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [joker],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        buys = [a for a in actions if isinstance(a, BuyCard)]
        assert len(buys) == 0

    def test_full_joker_slots_blocks_buy(self):
        shop_joker = _joker_card(cost=3)
        owned = [_joker_card() for _ in range(5)]
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 20,
            "jokers": owned,
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [shop_joker],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        buys = [a for a in actions if isinstance(a, BuyCard)]
        assert len(buys) == 0

    def test_negative_edition_bypasses_slot_limit(self):
        shop_joker = _joker_card(cost=3, edition={"negative": True})
        owned = [_joker_card() for _ in range(5)]
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 20,
            "jokers": owned,
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [shop_joker],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        buys = [a for a in actions if isinstance(a, BuyCard)]
        assert len(buys) == 1

    def test_sell_non_eternal_joker(self):
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 0,
            "jokers": [_joker_card(eternal=False)],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        sells = [a for a in actions if isinstance(a, SellCard) and a.area == "jokers"]
        assert len(sells) == 1

    def test_eternal_joker_not_sellable(self):
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 0,
            "jokers": [_joker_card(eternal=True)],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        sells = [a for a in actions if isinstance(a, SellCard) and a.area == "jokers"]
        assert len(sells) == 0

    def test_reroll_affordable(self):
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 5,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        assert any(isinstance(a, Reroll) for a in actions)

    def test_reroll_too_expensive(self):
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 2,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        assert not any(isinstance(a, Reroll) for a in actions)

    def test_free_reroll_available(self):
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 0,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 1},
        }
        actions = get_legal_actions(gs)
        assert any(isinstance(a, Reroll) for a in actions)

    def test_next_round_always_available(self):
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 0,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        assert any(isinstance(a, NextRound) for a in actions)

    def test_buy_voucher(self):
        voucher = _card("v_grabber", cost=10, ability={"set": "Voucher"})
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 10,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [voucher],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        redeems = [a for a in actions if isinstance(a, RedeemVoucher)]
        assert len(redeems) == 1

    def test_open_booster(self):
        pack = _card("p_arcana_normal_1", cost=4, ability={"set": "Booster"})
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 5,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [pack],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        opens = [a for a in actions if isinstance(a, OpenBooster)]
        assert len(opens) == 1


# ---------------------------------------------------------------------------
# PACK_OPENING
# ---------------------------------------------------------------------------


class TestLegalPackOpening:
    def test_pick_and_skip(self):
        gs = {
            "phase": GamePhase.PACK_OPENING,
            "pack_cards": [_card(), _card(), _card()],
            "pack_choices_remaining": 1,
        }
        actions = get_legal_actions(gs)
        picks = [a for a in actions if isinstance(a, PickPackCard)]
        assert len(picks) == 3
        assert any(isinstance(a, SkipPack) for a in actions)

    def test_no_choices_remaining(self):
        gs = {
            "phase": GamePhase.PACK_OPENING,
            "pack_cards": [_card(), _card()],
            "pack_choices_remaining": 0,
        }
        actions = get_legal_actions(gs)
        picks = [a for a in actions if isinstance(a, PickPackCard)]
        assert len(picks) == 0
        assert any(isinstance(a, SkipPack) for a in actions)


# ---------------------------------------------------------------------------
# ROUND_EVAL
# ---------------------------------------------------------------------------


class TestLegalRoundEval:
    def test_cashout(self):
        gs = {"phase": GamePhase.ROUND_EVAL, "consumables": []}
        actions = get_legal_actions(gs)
        assert any(isinstance(a, CashOut) for a in actions)


# ---------------------------------------------------------------------------
# GAME_OVER
# ---------------------------------------------------------------------------


class TestLegalGameOver:
    def test_empty(self):
        gs = {"phase": GamePhase.GAME_OVER}
        assert get_legal_actions(gs) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestLegalEdgeCases:
    def test_no_phase_returns_empty(self):
        assert get_legal_actions({}) == []

    def test_string_phase_works(self):
        gs = {"phase": "game_over"}
        assert get_legal_actions(gs) == []

    def test_sell_consumable_in_shop(self):
        tarot = _card("c_fool", ability={"set": "Tarot"})
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 0,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [tarot],
            "consumable_slots": 2,
            "shop_cards": [],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        sells = [a for a in actions if isinstance(a, SellCard) and a.area == "consumables"]
        assert len(sells) == 1
