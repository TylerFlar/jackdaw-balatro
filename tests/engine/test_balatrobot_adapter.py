"""Tests for jackdaw.bridge.balatrobot_adapter — action and state conversion.

Coverage
--------
* Every action type converts to correct balatrobot RPC method + params.
* Bot state converts to game_state with correct field values.
* Round-trip: action → RPC → verify method and param structure.
* State mapping: all 6 phases map correctly.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import (
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
    ReorderHand,
    ReorderJokers,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    UseConsumable,
)
from jackdaw.bridge.balatrobot_adapter import (
    action_to_rpc,
    bot_state_to_game_state,
)


# ---------------------------------------------------------------------------
# Action → RPC
# ---------------------------------------------------------------------------


class TestActionToRpc:
    def test_play_hand(self):
        rpc = action_to_rpc(PlayHand(card_indices=(0, 2, 4)))
        assert rpc["method"] == "play"
        assert rpc["params"]["cards"] == [0, 2, 4]

    def test_discard(self):
        rpc = action_to_rpc(Discard(card_indices=(1, 3)))
        assert rpc["method"] == "discard"
        assert rpc["params"]["cards"] == [1, 3]

    def test_select_blind(self):
        rpc = action_to_rpc(SelectBlind())
        assert rpc["method"] == "select"

    def test_skip_blind(self):
        rpc = action_to_rpc(SkipBlind())
        assert rpc["method"] == "skip"

    def test_buy_card(self):
        rpc = action_to_rpc(BuyCard(shop_index=1))
        assert rpc["method"] == "buy"
        assert rpc["params"]["card"] == 1

    def test_sell_joker(self):
        rpc = action_to_rpc(SellCard(area="jokers", card_index=2))
        assert rpc["method"] == "sell"
        assert rpc["params"]["joker"] == 2

    def test_sell_consumable(self):
        rpc = action_to_rpc(SellCard(area="consumables", card_index=0))
        assert rpc["method"] == "sell"
        assert rpc["params"]["consumable"] == 0

    def test_use_consumable_no_targets(self):
        rpc = action_to_rpc(UseConsumable(card_index=0))
        assert rpc["method"] == "use"
        assert rpc["params"]["consumable"] == 0
        assert "cards" not in rpc["params"]

    def test_use_consumable_with_targets(self):
        rpc = action_to_rpc(UseConsumable(card_index=1, target_indices=(0, 3)))
        assert rpc["method"] == "use"
        assert rpc["params"]["consumable"] == 1
        assert rpc["params"]["cards"] == [0, 3]

    def test_redeem_voucher(self):
        rpc = action_to_rpc(RedeemVoucher(card_index=0))
        assert rpc["method"] == "buy"
        assert rpc["params"]["voucher"] == 0

    def test_open_booster(self):
        rpc = action_to_rpc(OpenBooster(card_index=1))
        assert rpc["method"] == "buy"
        assert rpc["params"]["pack"] == 1

    def test_pick_pack_card(self):
        rpc = action_to_rpc(PickPackCard(card_index=2))
        assert rpc["method"] == "pack"
        assert rpc["params"]["card"] == 2

    def test_pick_pack_with_targets(self):
        rpc = action_to_rpc(PickPackCard(card_index=0, target_indices=(1, 2)))
        assert rpc["method"] == "pack"
        assert rpc["params"]["card"] == 0
        assert rpc["params"]["targets"] == [1, 2]

    def test_skip_pack(self):
        rpc = action_to_rpc(SkipPack())
        assert rpc["method"] == "pack"
        assert rpc["params"]["skip"] is True

    def test_reroll(self):
        rpc = action_to_rpc(Reroll())
        assert rpc["method"] == "reroll"

    def test_next_round(self):
        rpc = action_to_rpc(NextRound())
        assert rpc["method"] == "next_round"

    def test_cash_out(self):
        rpc = action_to_rpc(CashOut())
        assert rpc["method"] == "cash_out"

    def test_reorder_hand(self):
        rpc = action_to_rpc(ReorderHand(new_order=(2, 0, 1)))
        assert rpc["method"] == "rearrange"
        assert rpc["params"]["hand"] == [2, 0, 1]

    def test_reorder_jokers(self):
        rpc = action_to_rpc(ReorderJokers(new_order=(1, 0)))
        assert rpc["method"] == "rearrange"
        assert rpc["params"]["jokers"] == [1, 0]

    def test_sort_hand_maps_to_rearrange(self):
        rpc = action_to_rpc(SortHand(mode="rank"))
        assert rpc["method"] == "rearrange"


# ---------------------------------------------------------------------------
# Bot state → game_state
# ---------------------------------------------------------------------------

_SAMPLE_BOT_STATE = {
    "state": "SELECTING_HAND",
    "money": 12,
    "ante_num": 2,
    "round_num": 3,
    "seed": "TESTSEED",
    "deck": "RED",
    "stake": "WHITE",
    "won": False,
    "round": {
        "hands_left": 3,
        "discards_left": 2,
        "hands_played": 1,
        "discards_used": 1,
        "reroll_cost": 7,
        "chips": 150,
    },
    "hand": {
        "count": 8,
        "cards": [
            {"key": "H_A", "value": {"rank": "A", "suit": "H"}},
            {"key": "S_K", "value": {"rank": "K", "suit": "S"}},
        ],
    },
    "cards": {
        "count": 44,
        "cards": [{"key": "C_2"}, {"key": "D_3"}],
    },
    "jokers": {
        "count": 2,
        "cards": [
            {"key": "j_joker"},
            {"key": "j_banner"},
        ],
    },
    "consumables": {
        "count": 1,
        "cards": [{"key": "c_fool"}],
    },
    "blinds": {
        "small": {"name": "Small Blind", "status": "DEFEATED", "score": 300, "tag_name": "Economy Tag"},
        "big": {"name": "Big Blind", "status": "CURRENT", "score": 450, "tag_name": "Skip Tag"},
        "boss": {"name": "The Hook", "status": "UPCOMING", "score": 600, "tag_name": ""},
    },
}


class TestBotStateToGameState:
    def test_phase(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["phase"] == GamePhase.SELECTING_HAND

    def test_dollars(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["dollars"] == 12

    def test_ante(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["round_resets"]["ante"] == 2

    def test_chips(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["chips"] == 150

    def test_hands_left(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["current_round"]["hands_left"] == 3

    def test_hand_keys(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["hand_keys"] == ["H_A", "S_K"]

    def test_deck_size(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["deck_size"] == 44

    def test_joker_keys(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["joker_keys"] == ["j_joker", "j_banner"]

    def test_blind_info(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["blind_info"]["boss"]["name"] == "The Hook"
        assert gs["blind_info"]["small"]["tag_name"] == "Economy Tag"

    def test_seed(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["seed"] == "TESTSEED"

    def test_won(self):
        gs = bot_state_to_game_state(_SAMPLE_BOT_STATE)
        assert gs["won"] is False


class TestStateMapping:
    @pytest.mark.parametrize("bot_state,expected", [
        ("BLIND_SELECT", GamePhase.BLIND_SELECT),
        ("SELECTING_HAND", GamePhase.SELECTING_HAND),
        ("ROUND_EVAL", GamePhase.ROUND_EVAL),
        ("SHOP", GamePhase.SHOP),
        ("SMODS_BOOSTER_OPENED", GamePhase.PACK_OPENING),
        ("GAME_OVER", GamePhase.GAME_OVER),
    ])
    def test_phase_mapping(self, bot_state, expected):
        gs = bot_state_to_game_state({"state": bot_state})
        assert gs["phase"] == expected
