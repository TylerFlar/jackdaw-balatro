"""Tests for jackdaw.engine.game — step function (trimmed).

Coverage: one test per action handler + key edge cases + integration.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.bridge.balatrobot_adapter import action_to_rpc
from jackdaw.engine.actions import (
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapJokersLeft,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.card import Card
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_gs(seed: str = "GAME_TEST") -> dict[str, Any]:
    """Create a fully initialised game_state ready for blind selection."""
    return initialize_run("b_red", 1, seed)


def _joker_card(key: str = "j_joker", **kw) -> Card:
    c = Card(center_key=key)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    c.sell_cost = kw.pop("sell_cost", 3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _setup_shop(seed="SHOP_TEST"):
    """Set up a game state in the SHOP phase after beating Small Blind."""
    gs = _init_gs(seed)
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    return gs


def _make_consumable(key: str, set_name: str = "Tarot", **kw) -> Card:
    c = Card(center_key=key, cost=0)
    ability = {"set": set_name, "effect": ""}
    ability.update(kw.pop("extra_ability", {}))
    c.ability = ability
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# SelectBlind
# ---------------------------------------------------------------------------


class TestSelectBlind:
    def test_phase_transitions_to_selecting_hand(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND

    def test_blind_created(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["blind"] is not None
        assert gs["blind"].chips > 0


# ---------------------------------------------------------------------------
# SkipBlind
# ---------------------------------------------------------------------------


class TestSkipBlind:
    def test_skip_small_advances_to_big(self):
        gs = _init_gs()
        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Big"
        assert gs["round_resets"]["blind_states"]["Small"] == "Skipped"
        assert gs["round_resets"]["blind_states"]["Big"] == "Select"
        assert gs["phase"] == GamePhase.BLIND_SELECT

    def test_skip_boss_raises(self):
        gs = _init_gs()
        gs["blind_on_deck"] = "Boss"
        with pytest.raises(IllegalActionError, match="Cannot skip Boss"):
            step(gs, SkipBlind())


# ---------------------------------------------------------------------------
# PlayHand
# ---------------------------------------------------------------------------


class TestPlayHand:
    def _setup_playing(self, seed="PLAY_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_chips_accumulate(self):
        gs = self._setup_playing()
        hand = gs["hand"]
        step(gs, PlayHand(card_indices=tuple(range(min(5, len(hand))))))
        assert gs["chips"] > 0

    def test_hands_left_decremented(self):
        gs = self._setup_playing()
        initial = gs["current_round"]["hands_left"]
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["current_round"]["hands_left"] == initial - 1

    def test_round_won_transitions_to_round_eval(self):
        gs = self._setup_playing()
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL

    def test_game_over_when_no_hands_and_not_beaten(self):
        gs = self._setup_playing()
        gs["current_round"]["hands_left"] = 1
        gs["blind"].chips = 999_999_999
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["phase"] == GamePhase.GAME_OVER


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


class TestDiscard:
    def _setup_playing(self, seed="DISC_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_discards_left_decremented(self):
        gs = self._setup_playing()
        initial = gs["current_round"]["discards_left"]
        step(gs, Discard(card_indices=(0,)))
        assert gs["current_round"]["discards_left"] == initial - 1


# ---------------------------------------------------------------------------
# CashOut
# ---------------------------------------------------------------------------


class TestCashOut:
    def _setup_round_eval(self):
        gs = _init_gs("CASHOUT_TEST")
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        return gs

    def test_dollars_increase(self):
        gs = self._setup_round_eval()
        before = gs["dollars"]
        step(gs, CashOut())
        assert gs["dollars"] >= before

    def test_phase_transitions_to_shop(self):
        gs = self._setup_round_eval()
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP


# ---------------------------------------------------------------------------
# SellCard
# ---------------------------------------------------------------------------


class TestSellCard:
    def test_sell_joker(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card(sell_cost=5)]
        before = gs["dollars"]
        step(gs, SellCard(area="jokers", card_index=0))
        assert gs["dollars"] == before + 5
        assert len(gs["jokers"]) == 0

    def test_eternal_blocks_sale(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card(eternal=True)]
        with pytest.raises(IllegalActionError, match="eternal"):
            step(gs, SellCard(area="jokers", card_index=0))


# ---------------------------------------------------------------------------
# NextRound
# ---------------------------------------------------------------------------


class TestNextRound:
    def test_phase_transitions_to_blind_select(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT


# ---------------------------------------------------------------------------
# SortHand
# ---------------------------------------------------------------------------


class TestSortHand:
    def test_hand_reordered(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        step(gs, SortHand(mode="rank"))
        ids = [c.base.id for c in gs["hand"] if c.base]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# SwapJokers
# ---------------------------------------------------------------------------


class TestSwapJokers:
    def test_swap_joker_left(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        j0 = _joker_card("j_a")
        j1 = _joker_card("j_b")
        j2 = _joker_card("j_c")
        gs["jokers"] = [j0, j1, j2]
        step(gs, SwapJokersLeft(idx=2))
        assert gs["jokers"] == [j0, j2, j1]


# ---------------------------------------------------------------------------
# Reroll
# ---------------------------------------------------------------------------


class TestReroll:
    def test_dollars_deducted(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 10
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["dollars"] == 5


# ---------------------------------------------------------------------------
# Full mini-game (integration)
# ---------------------------------------------------------------------------


class TestMiniGame:
    def test_select_blind_play_hand_cash_out_next_round(self):
        """Full loop: blind select -> play hand -> cash out -> next round."""
        gs = _init_gs("MINI_GAME")

        assert gs["phase"] == GamePhase.BLIND_SELECT
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND

        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL

        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        assert gs["dollars"] > 0

        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT


# ---------------------------------------------------------------------------
# Extra edge cases: pack opening, buy_and_use, full ante
# ---------------------------------------------------------------------------


class TestPackOpening:
    def test_open_booster_transitions_to_pack(self):
        gs = _setup_shop("OPEN_PACK")
        pack = Card(center_key="p_arcana_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        step(gs, OpenBooster(card_index=0))
        assert gs["phase"] == GamePhase.PACK_OPENING

    def test_skip_pack_returns_to_shop(self):
        gs = _setup_shop("SKIP_PACK_TEST")
        pack = Card(center_key="p_arcana_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        step(gs, OpenBooster(card_index=0))
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP
        assert gs["pack_cards"] == []


class TestUseConsumable:
    def test_use_planet_in_shop(self):
        gs = _setup_shop("MERCURY_TEST")
        from jackdaw.engine.data.hands import HandType

        mercury = _make_consumable("c_mercury", set_name="Planet")
        mercury.ability["consumeable"] = {"hand_type": "Pair"}
        gs["consumables"] = [mercury]
        hl = gs["hand_levels"]
        level_before = hl[HandType.PAIR].level
        step(gs, UseConsumable(card_index=0))
        assert hl[HandType.PAIR].level == level_before + 1


class TestFullAnteProgression:
    def test_full_ante_cycle(self):
        gs = _init_gs("FULL_ANTE")

        # Small Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["blind_on_deck"] == "Big"

        # Big Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["blind_on_deck"] == "Boss"

        # Boss Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["round_resets"]["ante"] == 2
        assert gs["blind_on_deck"] == "Small"


# ============================================================================
# Legal actions (merged from test_actions.py)
# ============================================================================


def _action_card(key: str = "c_base", cost: int = 0, **kw) -> Card:
    c = Card(center_key=key, cost=cost)
    c.ability = kw.pop("ability", {"set": "", "effect": ""})
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _action_joker_card(key: str = "j_joker", cost: int = 5, **kw) -> Card:
    c = Card(center_key=key, cost=cost)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestLegalBlindSelect:
    def test_boss_blind_no_skip(self):
        gs = {"phase": GamePhase.BLIND_SELECT, "blind_on_deck": "Boss"}
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert SelectBlind in types
        assert SkipBlind not in types


class TestLegalSelectingHand:
    def test_play_and_discard_available(self):
        hand = [_action_card() for _ in range(8)]
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
            "hand": [_action_card()],
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert PlayHand not in types
        assert Discard in types


class TestLegalShop:
    def test_buy_affordable_joker(self):
        joker = _action_joker_card(cost=5)
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


class TestLegalPackOpening:
    def test_pick_and_skip(self):
        gs = {
            "phase": GamePhase.PACK_OPENING,
            "pack_cards": [_action_card(), _action_card(), _action_card()],
            "pack_choices_remaining": 1,
        }
        actions = get_legal_actions(gs)
        picks = [a for a in actions if isinstance(a, PickPackCard)]
        assert len(picks) == 3
        assert any(isinstance(a, SkipPack) for a in actions)


class TestLegalRoundEval:
    def test_cashout(self):
        gs = {"phase": GamePhase.ROUND_EVAL, "consumables": []}
        actions = get_legal_actions(gs)
        assert any(isinstance(a, CashOut) for a in actions)


class TestLegalGameOver:
    def test_empty(self):
        gs = {"phase": GamePhase.GAME_OVER}
        assert get_legal_actions(gs) == []


# ============================================================================
# Balatrobot adapter (merged from test_balatrobot_adapter.py)
# ============================================================================


class TestActionToRpc:
    def test_play_hand(self):
        rpc = action_to_rpc(PlayHand(card_indices=(0, 2, 4)))
        assert rpc["method"] == "play"
        assert rpc["params"]["cards"] == [0, 2, 4]

    def test_buy_card(self):
        rpc = action_to_rpc(BuyCard(shop_index=1))
        assert rpc["method"] == "buy"
        assert rpc["params"]["card"] == 1

    def test_sell_joker(self):
        rpc = action_to_rpc(SellCard(area="jokers", card_index=2))
        assert rpc["method"] == "sell"
        assert rpc["params"]["joker"] == 2

    def test_reroll(self):
        rpc = action_to_rpc(Reroll())
        assert rpc["method"] == "reroll"

    def test_pick_pack_card(self):
        rpc = action_to_rpc(PickPackCard(card_index=2))
        assert rpc["method"] == "pack"
        assert rpc["params"]["card"] == 2


# ============================================================================
# Mechanics checklist (merged from test_mechanics_checklist.py)
# ============================================================================


def _mech_init(seed: str = "MECH") -> dict[str, Any]:
    return initialize_run("b_red", 1, seed)


def _mech_joker(key: str = "j_joker", **kw) -> Card:
    c = Card(center_key=key)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    c.sell_cost = 3
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestCardFlipping:
    def test_card_has_facing_attribute(self):
        c = Card()
        assert hasattr(c, "facing")
        assert c.facing == "front"

    def test_facing_can_be_set_to_back(self):
        c = Card()
        c.facing = "back"
        assert c.facing == "back"

    def test_the_fish_flips_cards(self):
        """The Fish boss flips hand cards after play."""
        gs = _mech_init("FISH_FLIP")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_fish"
        step(gs, SelectBlind())
        # Play a hand
        gs["blind"].chips = 999999
        held_before = list(gs["hand"])
        played_ids = {id(gs["hand"][i]) for i in (0, 1, 2, 3, 4)}
        held_ids = {id(c) for c in held_before} - played_ids
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        # Vanilla flips ONLY the replacement draws (blind.lua:611), not
        # the cards already held (the old whole-hand flip was a bug).
        if gs["phase"] == GamePhase.SELECTING_HAND:
            for card in gs["hand"]:
                if id(card) in held_ids:
                    assert card.facing == "front", f"{card.card_key} wrongly flipped"
                else:
                    assert card.facing == "back", f"{card.card_key} not flipped"


class TestBossPressPlay:
    def test_the_hook_discards(self):
        """The Hook discards 2 random hand cards on play."""
        gs = _mech_init("HOOK_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_hook"
        step(gs, SelectBlind())
        len(gs["hand"])
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0,)))
        # Hand should have fewer cards (1 played + 2 hooked + replacements drawn)
        # The net effect depends on deck size, but discard_pile should have entries
        assert len(gs.get("discard_pile", [])) >= 2

    def test_the_tooth_costs_dollars(self):
        """The Tooth costs $1 per card played."""
        gs = _mech_init("TOOTH_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_tooth"
        step(gs, SelectBlind())
        dollars_before = gs["dollars"]
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0, 1, 2)))
        # Should lose $3 (ignoring scoring dollars)
        result = gs["last_score_result"]
        expected = dollars_before - 3 + result.dollars_earned
        assert gs["dollars"] == expected


class TestSealEffects:
    def test_purple_seal_creates_tarot_on_discard(self):
        """Discarding a Purple Seal card creates a Tarot consumable."""
        gs = _mech_init("PURPLE_SEAL")
        step(gs, SelectBlind())
        gs["hand"][0].seal = "Purple"
        initial_cons = len(gs.get("consumables", []))
        step(gs, Discard(card_indices=(0,)))
        assert len(gs.get("consumables", [])) == initial_cons + 1


class TestDoubleTag:
    def test_double_tag_duplicates(self):
        """Double Tag duplicates a newly awarded tag."""
        gs = _mech_init("DOUBLE_TAG")
        # A Double Tag acquired from an earlier skip lives in awarded_tags
        # (gs["tags"] was a phantom key nothing ever wrote).
        gs.setdefault("awarded_tags", []).append({"key": "tag_double", "result": None})
        # Force Small tag to be tag_economy
        gs["round_resets"]["blind_tags"]["Small"] = "tag_economy"
        gs["dollars"] = 10
        step(gs, SkipBlind())
        # Should have received tag_economy + a duplicate
        awarded = gs.get("awarded_tags", [])
        economy_awards = [a for a in awarded if a["key"] == "tag_economy"]
        assert len(economy_awards) >= 2, f"Expected 2 economy tags, got {len(economy_awards)}"


# ---------------------------------------------------------------------------
# Joker passives on acquisition/removal (card.lua:564/648 wiring)
# ---------------------------------------------------------------------------


class TestJokerPassivesOnAcquisition:
    """add_to_deck/remove_from_deck must fire on the step() path.

    Regression: buy/pack-pick/create appended jokers without applying
    passive effects, so Juggler granted no hand size and Negative
    editions granted no extra slot (the bridge path was correct)."""

    def test_buy_joker_applies_passive(self):
        gs = _setup_shop()
        before = gs["hand_size"]
        juggler = _joker_card("j_juggler", cost=0)
        juggler.ability["h_size"] = 1
        gs["shop_cards"] = [juggler]
        step(gs, BuyCard(shop_index=0))
        assert gs["hand_size"] == before + 1

    def test_sell_joker_reverts_passive(self):
        gs = _setup_shop()
        before = gs["hand_size"]
        juggler = _joker_card("j_juggler", cost=0, sell_cost=1)
        juggler.ability["h_size"] = 1
        gs["shop_cards"] = [juggler]
        step(gs, BuyCard(shop_index=0))
        step(gs, SellCard(area="jokers", card_index=len(gs["jokers"]) - 1))
        assert gs["hand_size"] == before

    def test_buy_negative_joker_grants_and_revokes_slot(self):
        gs = _setup_shop()
        neg = _joker_card("j_neg_test", cost=0, sell_cost=1)
        neg.edition = {"negative": True, "type": "negative"}
        gs["shop_cards"] = [neg]
        step(gs, BuyCard(shop_index=0))
        assert gs["joker_slots"] == 6
        step(gs, SellCard(area="jokers", card_index=len(gs["jokers"]) - 1))
        assert gs["joker_slots"] == 5

    def test_negative_on_board_allows_extra_buy(self):
        """Five board entries incl. one Negative must still offer BuyCard."""
        gs = _setup_shop()
        gs["jokers"] = [_joker_card(f"j_filler_{i}") for i in range(4)]
        neg = _joker_card("j_neg_test", cost=0)
        neg.edition = {"negative": True, "type": "negative"}
        gs["shop_cards"] = [neg]
        step(gs, BuyCard(shop_index=0))
        assert len(gs["jokers"]) == 5
        gs["shop_cards"] = [_joker_card("j_sixth", cost=0)]
        legal = get_legal_actions(gs)
        assert any(isinstance(a, BuyCard) for a in legal)
        step(gs, BuyCard(shop_index=0))
        assert len(gs["jokers"]) == 6

    def test_pack_pick_joker_applies_passive(self):
        gs = _setup_shop()
        before = gs["hand_size"]
        juggler = _joker_card("j_juggler")
        juggler.ability["h_size"] = 1
        gs["phase"] = GamePhase.PACK_OPENING
        gs["pack_cards"] = [juggler]
        gs["pack_choices_remaining"] = 1
        step(gs, PickPackCard(card_index=0))
        assert gs["hand_size"] == before + 1


class TestBuySpaceGuard:
    """The executor must refuse no-room buys/picks itself, not just rely
    on the legality mask hiding them (hand-built actions bypass the mask)."""

    def test_buy_joker_full_board_refused(self):
        gs = _setup_shop()
        gs["jokers"] = [_joker_card(f"j_full_{i}") for i in range(5)]
        gs["shop_cards"] = [_joker_card("j_sixth", cost=0)]
        with pytest.raises(IllegalActionError, match="room"):
            step(gs, BuyCard(shop_index=0))
        assert len(gs["jokers"]) == 5
        assert len(gs["shop_cards"]) == 1

    def test_buy_negative_joker_full_board_allowed(self):
        gs = _setup_shop()
        gs["jokers"] = [_joker_card(f"j_full_{i}") for i in range(5)]
        neg = _joker_card("j_neg_test", cost=0)
        neg.edition = {"negative": True, "type": "negative"}
        gs["shop_cards"] = [neg]
        step(gs, BuyCard(shop_index=0))
        assert len(gs["jokers"]) == 6
        assert gs["joker_slots"] == 6

    def test_buy_consumable_full_slots_refused(self):
        gs = _setup_shop()
        gs["consumables"] = [_make_consumable("c_fool"), _make_consumable("c_magician")]
        gs["shop_cards"] = [_make_consumable("c_temperance")]
        with pytest.raises(IllegalActionError, match="room"):
            step(gs, BuyCard(shop_index=0))
        assert len(gs["consumables"]) == 2

    def test_pack_pick_joker_full_board_refused(self):
        gs = _setup_shop()
        gs["jokers"] = [_joker_card(f"j_full_{i}") for i in range(5)]
        gs["phase"] = GamePhase.PACK_OPENING
        gs["pack_cards"] = [_joker_card("j_sixth")]
        gs["pack_choices_remaining"] = 1
        with pytest.raises(IllegalActionError, match="room"):
            step(gs, PickPackCard(card_index=0))
        assert len(gs["jokers"]) == 5
        assert gs["pack_choices_remaining"] == 1


# ---------------------------------------------------------------------------
# used_jokers lifecycle (card.lua:4739-4749) and store_joker_create tags
# ---------------------------------------------------------------------------


class TestUsedKeyRelease:
    """Removed cards must free their center key for future pool draws.

    Regression: keys registered at creation were never released, so
    pools exhausted permanently — 3-4 celestial packs marked all 12
    planets and later packs collapsed to the all-Pluto fallback."""

    def test_sell_releases_key(self):
        gs = _setup_shop()
        gs["jokers"] = [_joker_card("j_release_probe")]
        gs.setdefault("used_jokers", {})["j_release_probe"] = True
        step(gs, SellCard(area="jokers", card_index=0))
        assert "j_release_probe" not in gs["used_jokers"]

    def test_sell_keeps_key_while_copy_in_play(self):
        gs = _setup_shop()
        gs["jokers"] = [_joker_card("j_release_probe"), _joker_card("j_release_probe")]
        gs.setdefault("used_jokers", {})["j_release_probe"] = True
        step(gs, SellCard(area="jokers", card_index=0))
        assert "j_release_probe" in gs["used_jokers"]

    def test_pack_close_releases_unpicked(self):
        gs = _setup_shop()
        c = _make_consumable("c_probe_tarot")
        c.center_key = "c_probe_tarot"
        gs.setdefault("used_jokers", {})["c_probe_tarot"] = True
        gs["phase"] = GamePhase.PACK_OPENING
        gs["pack_cards"] = [c]
        gs["pack_choices_remaining"] = 1
        step(gs, SkipPack())
        assert "c_probe_tarot" not in gs["used_jokers"]

    def test_shop_close_releases_unsold(self):
        gs = _setup_shop()
        gs["shop_cards"] = [_joker_card("j_unsold_probe", cost=5)]
        gs.setdefault("used_jokers", {})["j_unsold_probe"] = True
        step(gs, NextRound())
        assert "j_unsold_probe" not in gs["used_jokers"]


class TestStoreJokerCreateTag:
    """Rare/Uncommon Tags must deliver a free forced-rarity shop joker
    (previously documented-inert on the step() path)."""

    @staticmethod
    def _skip_for_tag(tag_key):
        gs = _init_gs(f"TAGT_{tag_key}")
        gs["round_resets"]["blind_tags"]["Small"] = tag_key
        step(gs, SkipBlind())
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        return gs

    def test_rare_tag_free_rare_in_next_shop(self):
        from jackdaw.engine.pools import JOKER_RARITY_POOLS

        gs = self._skip_for_tag("tag_rare")
        rares = set(JOKER_RARITY_POOLS[3])
        free_rares = [c for c in gs["shop_cards"]
                      if c.center_key in rares and c.cost == 0]
        assert free_rares, [(c.center_key, c.cost) for c in gs["shop_cards"]]
        assert gs["awarded_tags"][0].get("shop_fired") is True

    def test_uncommon_tag_free_uncommon_in_next_shop(self):
        from jackdaw.engine.pools import JOKER_RARITY_POOLS

        gs = self._skip_for_tag("tag_uncommon")
        uncommons = set(JOKER_RARITY_POOLS[2])
        free_unc = [c for c in gs["shop_cards"]
                    if c.center_key in uncommons and c.cost == 0]
        assert free_unc, [(c.center_key, c.cost) for c in gs["shop_cards"]]


class TestChaosMidShop:
    """Chaos the Clown bought mid-shop grants its free reroll immediately
    (card.lua:601-603) — regression for the wrong-slot write AND the
    calculate_reroll_cost signature crash the first fix introduced."""

    def test_add_grants_free_reroll_and_zeroes_cost(self):
        from jackdaw.engine.card_factory import create_joker

        gs = _init_gs("CHAOS_MS")
        gs["phase"] = GamePhase.SHOP
        gs["current_round"]["free_rerolls"] = 0
        c = create_joker("j_chaos")
        c.add_to_deck(gs)
        assert gs["current_round"]["free_rerolls"] == 1
        assert gs["current_round"]["reroll_cost"] == 0
        c.remove_from_deck(gs)
        assert gs["current_round"]["free_rerolls"] == 0


class TestHandDetectionJokerFlags:
    """Bug #27: score_hand called evaluate_hand(jokers=None), so Four
    Fingers / Shortcut / Smeared never affected hand DETECTION on the
    organic play path (live-verified: LS9X3EVM smeared flush, LSIMZGYW
    shortcut straight)."""

    def _score(self, cards, jokers):
        from jackdaw.engine.blind import Blind
        from jackdaw.engine.hand_levels import HandLevels
        from jackdaw.engine.rng import PseudoRandom
        from jackdaw.engine.scoring import score_hand

        return score_hand(
            played_cards=cards,
            held_cards=[],
            jokers=jokers,
            hand_levels=HandLevels(),
            blind=Blind.create("bl_small", 1),
            rng=PseudoRandom("FLAGTEST"),
        )

    @staticmethod
    def _pc(suit, rank):
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit

        return create_playing_card(Suit(suit), Rank(rank))

    def test_smeared_enables_mixed_red_flush(self):
        from jackdaw.engine.card_factory import create_joker

        cards = [self._pc("Hearts", "2"), self._pc("Diamonds", "5"),
                 self._pc("Hearts", "7"), self._pc("Diamonds", "9"),
                 self._pc("Hearts", "King")]
        assert self._score(cards, []).hand_type == "High Card"
        smeared = create_joker("j_smeared")
        assert self._score(cards, [smeared]).hand_type == "Flush"

    def test_shortcut_enables_gapped_straight(self):
        from jackdaw.engine.card_factory import create_joker

        cards = [self._pc("Hearts", "3"), self._pc("Spades", "4"),
                 self._pc("Clubs", "6"), self._pc("Hearts", "7"),
                 self._pc("Diamonds", "9")]
        assert self._score(cards, []).hand_type == "High Card"
        shortcut = create_joker("j_shortcut")
        assert self._score(cards, [shortcut]).hand_type == "Straight"


class TestBurglarOnBlindSelect:
    """Bug #28: setting_blind fired BEFORE start_round, whose counter
    reset wiped Burglar's +3 hands / 0 discards (vanilla order:
    state_events.lua:296-297 reset, then line 336 joker loop).
    Live-verified: LS4GK4C4."""

    def test_burglar_hands_and_discards_survive_select(self):
        from jackdaw.engine.card_factory import create_joker

        gs = _init_gs("BURGLAR1")
        j = create_joker("j_burglar")
        gs["jokers"] = [j]
        j.add_to_deck(gs)
        step(gs, SelectBlind())
        assert gs["current_round"]["hands_left"] == 7
        assert gs["current_round"]["discards_left"] == 0


class TestVoucherTagAddsShopVoucher:
    """Bug #29: voucher_add was unwired — Voucher Tag never added its
    extra shop voucher ('Voucher_fromtag' stream, tag.lua:302-318).
    Live-verified: LS1UBIP7."""

    def test_voucher_tag_second_voucher_in_shop(self):
        gs = _init_gs("VTAG1")
        gs.setdefault("awarded_tags", []).append({"key": "tag_voucher"})
        step(gs, SelectBlind())
        gs["chips"] = 10**9
        step(gs, PlayHand(card_indices=(0,)))
        step(gs, CashOut())
        assert len(gs["shop_vouchers"]) == 2
        # not free — vanilla tag vouchers are purchasable at full price
        assert all(v.cost > 0 for v in gs["shop_vouchers"])
        assert gs["awarded_tags"][0].get("shop_fired") is True


class TestHallucinationOnPackOpen:
    """Bug #30: Hallucination rolled a nonexistent 'hallucination'
    stream (vanilla: 'halu'+ante, card.lua:2335) and the open_booster
    mutations were discarded. Live-verified: LS6TA2OU (c_hanged_man)."""

    def test_hallucination_roll_and_create(self):
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.rng import PseudoRandom

        # find a seed whose first 'halu1' roll passes (< 0.5)
        seed = next(s for s in ("HAL1", "HAL2", "HAL3", "HAL4", "HAL5")
                    if PseudoRandom(s).random("halu1") < 0.5)
        gs = _setup_shop(seed)
        j = create_joker("j_hallucination")
        gs["jokers"] = [j]
        j.add_to_deck(gs)
        gs["shop_boosters"] = gs.get("shop_boosters") or []
        if not gs["shop_boosters"]:
            pytest.skip("no booster in shop for this seed")
        gs["dollars"] = 50
        step(gs, OpenBooster(card_index=0))
        assert len(gs.get("consumables", [])) == 1
        assert gs["consumables"][0].ability.get("set") == "Tarot"

    def test_full_slots_consume_no_halu_roll(self):
        """Bug #55 (LSVRQFEU): vanilla room-gates BEFORE the halu roll
        (outer if card.lua:2336, roll card.lua:2337) — a pack opened at
        full consumable slots must not advance the halu stream."""
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.jokers import (
            GameSnapshot,
            JokerContext,
            calculate_joker,
        )
        from jackdaw.engine.rng import PseudoRandom

        j = create_joker("j_hallucination")
        rng = PseudoRandom("HALGATE")
        snap = GameSnapshot(consumable_count=2, consumable_slots=2, ante=1)
        ctx = JokerContext(open_booster=True, rng=rng, game=snap, jokers=[j])
        assert calculate_joker(j, ctx) is None
        # stream untouched: next halu1 pull matches a fresh rng's first
        fresh = PseudoRandom("HALGATE")
        assert rng.random("halu1") == fresh.random("halu1")


class TestConsumableUsageTotals:
    """Bug #31: consumable_usage_total['tarot'] had no writer — Fortune
    Teller always read 0 (vanilla set_consumeable_usage,
    misc_functions.lua:1184). Live-verified: LSVBDNCC (off-by-one mult)."""

    def test_tarot_use_increments_totals(self):
        from jackdaw.engine.card_factory import create_consumable

        gs = _init_gs("USAGE1")
        step(gs, SelectBlind())
        tarot = create_consumable("c_temperance")  # no targets, pays $
        gs["consumables"] = [tarot]
        step(gs, UseConsumable(card_index=0))
        totals = gs["consumable_usage_total"]
        assert totals["tarot"] == 1
        assert totals["tarot_planet"] == 1
        assert totals["all"] == 1
        assert gs["consumable_usage"]["c_temperance"]["count"] == 1


class TestPillarPlayedThisAnte:
    """Bug #32: played_this_ante was never cleared at boss defeat
    (vanilla state_events.lua:266) — The Pillar debuffed cards played
    in PREVIOUS antes. Live-verified: LSOL9RKQ."""

    def test_flags_cleared_at_boss_defeat_only(self):
        gs = _init_gs("PILLAR1")
        gs["chips_target_override"] = None
        # Small blind: play one hand, win, check flags survive
        step(gs, SelectBlind())
        gs["chips"] = 10**9
        played_cards = [gs["hand"][i] for i in range(3)]
        step(gs, PlayHand(card_indices=(0, 1, 2)))
        assert all(c.ability.get("played_this_ante") for c in played_cards)
        step(gs, CashOut())
        step(gs, NextRound())
        # Big blind: flags still set (cleared at BOSS defeat only)
        step(gs, SelectBlind())
        assert all(c.ability.get("played_this_ante") for c in played_cards)
        gs["chips"] = 10**9
        step(gs, PlayHand(card_indices=(0, 1, 2)))
        step(gs, CashOut())
        step(gs, NextRound())
        # Boss: win it -> every flag cleared
        step(gs, SelectBlind())
        gs["chips"] = 10**9
        step(gs, PlayHand(card_indices=(0, 1, 2)))
        deck_all = (gs["deck"] + gs.get("hand", []) + gs.get("discard_pile", []))
        assert not any(c.ability.get("played_this_ante") for c in deck_all)


class TestMarbleStoneShuffleTiming:
    """Bug #33: Marble Joker's stone joined the deck BEFORE the per-round
    'nr' shuffle, changing the whole permutation. Vanilla emplaces it via
    a queued event AFTER the shuffle; it lands at the pile bottom
    (live-verified: LS86UJ9R — dealt hand matches a stone-less 52-card
    shuffle, stone at serialized deck index 0)."""

    def test_stone_added_after_shuffle_at_bottom(self):
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.rng import PseudoRandom

        gs = _init_gs("MARBLE1")
        j = create_joker("j_marble")
        gs["jokers"] = [j]
        j.add_to_deck(gs)

        # Expected deal: shuffle a stone-less copy with the same stream.
        ref = list(gs["deck"])
        ref_rng = PseudoRandom("MARBLE1")
        ref_rng.shuffle(ref, ref_rng.seed("nr1"))
        expected_hand = sorted(c.card_key for c in ref[-8:])

        step(gs, SelectBlind())
        dealt = sorted(c.card_key for c in gs["hand"])
        assert dealt == expected_hand

        deck = gs["deck"]
        stones = [i for i, c in enumerate(deck)
                  if c.ability.get("effect") == "Stone Card"]
        assert stones == [0]  # pile bottom, drawn last


class TestDiscountRepricesOwnedCards:
    """Bug #34: buying Clearance Sale repriced only shop areas — vanilla's
    per-frame set_cost also drops OWNED jokers'/consumables' buy AND sell
    values immediately (live-verified: LSPNZ98T, -25% across the board)."""

    def test_clearance_sale_lowers_owned_sell_values(self):
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.shop import reprice_shop

        gs = _init_gs("DISC1")
        gs["phase"] = GamePhase.SHOP
        j = create_joker("j_joker")  # base_cost 2? use set_cost baseline
        j.base_cost = 8
        j.set_cost()
        gs["jokers"] = [j]
        assert (j.cost, j.sell_cost) == (8, 4)

        gs["discount_percent"] = 25
        reprice_shop(gs)
        assert j.cost == 6  # floor(8.5 * 0.75) = 6
        assert j.sell_cost == 3


class TestHookDiscardJokerContexts:
    """Bug #35: The Hook's forced discards silently moved cards to the
    pile — vanilla routes them through discard_cards_from_highlighted
    (hook flag), firing per-card seal + joker discard contexts: Green
    Joker LOSES mult on hook discards (live-verified: LSN6STIA, mult
    13 vs 12 on the first Hook hand), while Burnt Joker is hook-gated
    and the discard counters stay untouched."""

    def _hook_gs(self, seed="HOOK1"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        gs["blind"].name = "The Hook"
        gs["blind"].boss = True
        gs["blind"].chips = 10**9  # keep the round going
        return gs

    def test_green_joker_loses_mult_on_hook_discard(self):
        from jackdaw.engine.card_factory import create_joker

        # Baseline: same play WITHOUT the hook gains +1 (hand played).
        gs0 = self._hook_gs("HOOKG")
        gs0["blind"].name = "The Wall"
        g0 = create_joker("j_green_joker")
        g0.ability["mult"] = 5
        gs0["jokers"] = [g0]
        step(gs0, PlayHand(card_indices=(0,)))
        assert g0.ability["mult"] == 6

        # With The Hook: the forced discard costs 1 → net unchanged.
        gs = self._hook_gs("HOOKG")
        green = create_joker("j_green_joker")
        green.ability["mult"] = 5
        gs["jokers"] = [green]
        discards_before = gs["current_round"]["discards_left"]
        step(gs, PlayHand(card_indices=(0,)))
        assert green.ability["mult"] == 5  # +1 play, -1 hook discard
        # hook discards must NOT consume a discard
        assert gs["current_round"]["discards_left"] == discards_before

    def test_burnt_joker_hook_gated(self):
        from jackdaw.engine.card_factory import create_joker

        gs = self._hook_gs("HOOK2")
        burnt = create_joker("j_burnt")
        gs["jokers"] = [burnt]
        hl = gs["hand_levels"]
        levels_before = {ht: hl.get_level(ht) for ht in hl.hands} if hasattr(hl, "hands") else None
        step(gs, PlayHand(card_indices=(0,)))
        if levels_before is not None:
            assert {ht: hl.get_level(ht) for ht in hl.hands} == levels_before


class TestD6TagRerollCost:
    """Bug #41: D6 Tag was modeled as one Chaos-style free reroll —
    vanilla sets round_resets.temp_reroll_cost = 0 so rerolls START at
    $0 and climb +1 each (tag.lua:383-391), once per shop (shop_d6ed,
    cleared at cash_out).  A Python `or` also dropped temp == 0 (0 is
    truthy in Lua).  Live-verified: LSKWQS7C entered the shop at $0."""

    def test_d6_shop_starts_at_zero_and_climbs(self):
        from jackdaw.engine.shop import calculate_reroll_cost

        gs = _init_gs("D6TAG1")
        gs.setdefault("awarded_tags", []).append({"key": "tag_d_six"})
        step(gs, SelectBlind())
        gs["chips"] = 10**9
        step(gs, PlayHand(card_indices=(0,)))
        step(gs, CashOut())
        assert gs["current_round"]["reroll_cost"] == 0
        gs["dollars"] = 20
        step(gs, Reroll())
        assert gs["current_round"]["reroll_cost"] == 1
        step(gs, Reroll())
        assert gs["current_round"]["reroll_cost"] == 2

    def test_temp_zero_not_dropped_by_or(self):
        from jackdaw.engine.shop import calculate_reroll_cost

        gs = _init_gs("D6TAG2")
        gs["round_resets"]["temp_reroll_cost"] = 0
        gs["current_round"]["reroll_cost_increase"] = 0
        assert calculate_reroll_cost(gs) == 0


# ---------------------------------------------------------------------------
# Round-end seal effects (bugs #51/#52, LSGLNPN9)
# ---------------------------------------------------------------------------


class TestRoundEndSealEffects:
    """Gold Seal has NO held effect at round end — vanilla pays it on
    play+score only (get_p_dollars, card.lua:1071-73); the held-card
    round-end effects are h_dollars + Blue Seal only (card.lua:1033-65).
    Blue Seal's planet matches the LAST hand played
    (G.GAME.last_hand_played, card.lua:1047-53), not the most-played."""

    def _setup_win(self, seed="SEAL_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        return gs

    def test_gold_seal_held_pays_nothing_at_round_end(self):
        gs = self._setup_win()
        gs["hand"][7].seal = "Gold"  # stays in hand through the play
        dollars_before = gs["dollars"]
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        assert gs["dollars"] == dollars_before

    def test_gold_seal_pays_on_played_scoring_card(self):
        gs = self._setup_win()
        gs["hand"][0].seal = "Gold"
        dollars_before = gs["dollars"]
        step(gs, PlayHand(card_indices=(0,)))  # High Card always scores
        assert gs["dollars"] == dollars_before + 3

    def test_blue_seal_creates_planet_for_last_hand_played(self):
        gs = self._setup_win()
        gs["hand"][7].seal = "Blue"
        step(gs, PlayHand(card_indices=(0,)))  # last hand = High Card
        assert gs["phase"] == GamePhase.ROUND_EVAL
        cons = gs["consumables"]
        assert [c.center_key for c in cons] == ["c_pluto"]

    def test_blue_seal_no_planet_at_full_consumable_slots(self):
        gs = self._setup_win()
        gs["hand"][7].seal = "Blue"
        from jackdaw.engine.card import Card as _C

        filler = []
        for _ in range(gs.get("consumable_slots", 2)):
            f = _C(center_key="c_pluto")
            f.ability = {"set": "Planet", "effect": ""}
            filler.append(f)
        gs["consumables"] = filler
        step(gs, PlayHand(card_indices=(0,)))
        assert len(gs["consumables"]) == gs.get("consumable_slots", 2)


# ---------------------------------------------------------------------------
# d_size passives adjust the CURRENT round (bug #53, LSH8NR94)
# ---------------------------------------------------------------------------


class TestDrunkardCurrentRoundDiscards:
    """d_size passives ease_discard the CURRENT round immediately on
    add/remove (card.lua:591/653), not just round_resets — a Riff-Raff-
    created Drunkard at setting_blind affects the round being dealt."""

    def test_buy_drunkard_bumps_current_round_discards(self):
        gs = _setup_shop()
        rr_before = gs["round_resets"]["discards"]
        cr_before = gs["current_round"]["discards_left"]
        drunkard = _joker_card("j_drunkard", cost=0)
        drunkard.ability["d_size"] = 1
        gs["shop_cards"] = [drunkard]
        step(gs, BuyCard(shop_index=0))
        assert gs["round_resets"]["discards"] == rr_before + 1
        assert gs["current_round"]["discards_left"] == cr_before + 1

    def test_sell_drunkard_clamps_current_round_at_zero(self):
        gs = _setup_shop()
        drunkard = _joker_card("j_drunkard", cost=0, sell_cost=1)
        drunkard.ability["d_size"] = 1
        gs["shop_cards"] = [drunkard]
        step(gs, BuyCard(shop_index=0))
        gs["current_round"]["discards_left"] = 0  # all spent this round
        rr_before = gs["round_resets"]["discards"]
        step(gs, SellCard(area="jokers", card_index=len(gs["jokers"]) - 1))
        assert gs["round_resets"]["discards"] == rr_before - 1
        assert gs["current_round"]["discards_left"] == 0  # ease_discard clamp


# ---------------------------------------------------------------------------
# Reroll shop slots are not soulable (bug #56, LSMORS4J)
# ---------------------------------------------------------------------------


class TestRerollNotSoulable:
    """Shop cards are never soulable (UI_definitions.lua:776 passes nil)
    — the reroll refill left create_card's True default, burning a
    phantom soul_* roll per slot; LSMORS4J's hit 0.997+ and forced a
    c_black_hole into the rerolled shop."""

    def test_reroll_consumes_no_soul_rolls(self):
        from jackdaw.engine.rng import PseudoRandom

        seed = "SOULR1"
        gs = _setup_shop(seed)
        gs["dollars"] = 50
        step(gs, Reroll())
        rng = gs["rng"]
        for pool in ("Tarot", "Planet", "Spectral"):
            fresh = PseudoRandom(seed)
            key = f"soul_{pool}1"
            assert rng.random(key) == fresh.random(key), key
        assert all(
            getattr(c, "center_key", "") not in ("c_soul", "c_black_hole")
            for c in gs.get("shop_cards", [])
        )


# ---------------------------------------------------------------------------
# Ceremonial Dagger destroys its right neighbor (bug #57, LSL9ZZUW)
# ---------------------------------------------------------------------------


class TestCeremonialDaggerOnBlindSelect:
    """The destroy_joker mutation from Ceremonial Dagger was never
    processed by _apply_setting_blind_mutations — the dagger gained
    mult while its victim survived (card.lua:2561)."""

    def test_dagger_eats_right_neighbor(self):
        gs = _init_gs("DAGGER1")
        dagger = _joker_card("j_ceremonial")
        dagger.ability["name"] = "j_ceremonial"
        victim = _joker_card("j_flower_pot", sell_cost=3)
        gs["jokers"] = [dagger, victim]
        step(gs, SelectBlind())
        assert [j.center_key for j in gs["jokers"]] == ["j_ceremonial"]
        assert dagger.ability.get("mult", 0) == 6  # 2x sell_cost

    def test_dagger_spares_eternal_neighbor(self):
        gs = _init_gs("DAGGER2")
        dagger = _joker_card("j_ceremonial")
        victim = _joker_card("j_flower_pot", sell_cost=3, eternal=True)
        gs["jokers"] = [dagger, victim]
        step(gs, SelectBlind())
        assert len(gs["jokers"]) == 2
        assert dagger.ability.get("mult", 0) == 0

    def test_dagger_alone_no_effect(self):
        gs = _init_gs("DAGGER3")
        dagger = _joker_card("j_ceremonial")
        gs["jokers"] = [dagger]
        step(gs, SelectBlind())
        assert len(gs["jokers"]) == 1


# ---------------------------------------------------------------------------
# hands_played counters are pre-increment during scoring (bug #58)
# ---------------------------------------------------------------------------


class TestHandsPlayedPreIncrementDuringScoring:
    """Vanilla increments BOTH hands_played counters in the event AFTER
    evaluate_play (state_events.lua:523-24) — scoring contexts see
    pre-increment values.  The sim incremented before score_hand, which
    killed DNA / Sixth Sense (== 0 never true) and shifted Loyalty
    Card's window (LSHACSAC)."""

    def test_dna_copies_first_single_card_play(self):
        gs = _init_gs("DNAPRE1")
        dna = _joker_card("j_dna")
        gs["jokers"] = [dna]
        step(gs, SelectBlind())
        owned_before = (
            len(gs["deck"]) + len(gs["hand"]) + len(gs.get("discard_pile", []))
        )
        step(gs, PlayHand(card_indices=(0,)))
        owned_after = (
            len(gs["deck"]) + len(gs["hand"]) + len(gs.get("discard_pile", []))
            + len(gs.get("played_cards_area", []))
        )
        assert owned_after == owned_before + 1  # DNA copy created

    def test_second_play_no_dna(self):
        gs = _init_gs("DNAPRE2")
        dna = _joker_card("j_dna")
        gs["jokers"] = [dna]
        step(gs, SelectBlind())
        step(gs, PlayHand(card_indices=(0, 1)))  # 2 cards: DNA needs 1
        owned_before = (
            len(gs["deck"]) + len(gs["hand"]) + len(gs.get("discard_pile", []))
        )
        step(gs, PlayHand(card_indices=(0,)))  # 2nd hand of round: no DNA
        owned_after = (
            len(gs["deck"]) + len(gs["hand"]) + len(gs.get("discard_pile", []))
            + len(gs.get("played_cards_area", []))
        )
        assert owned_after == owned_before


# ---------------------------------------------------------------------------
# Oops! All 6s doubles the NESTED probabilities table (bug #60, LSPV15ZH)
# ---------------------------------------------------------------------------


class TestOopsAllSixesProbabilities:
    """Vanilla doubles/halves every key of G.GAME.probabilities
    (card.lua:608-11/665-68).  The old top-level probabilities_normal
    write was a dead slot — scoring reads gs['probabilities']['normal']."""

    def test_buy_doubles_nested_normal(self):
        gs = _setup_shop("OOPS1")
        oops = _joker_card("j_oops", cost=0)
        oops.ability["name"] = "Oops! All 6s"
        gs["shop_cards"] = [oops]
        step(gs, BuyCard(shop_index=0))
        assert gs["probabilities"]["normal"] == 2.0

    def test_sell_halves_back(self):
        gs = _setup_shop("OOPS2")
        oops = _joker_card("j_oops", cost=0, sell_cost=1)
        oops.ability["name"] = "Oops! All 6s"
        gs["shop_cards"] = [oops]
        step(gs, BuyCard(shop_index=0))
        step(gs, SellCard(area="jokers", card_index=len(gs["jokers"]) - 1))
        assert gs["probabilities"]["normal"] == 1.0


# ---------------------------------------------------------------------------
# Rocket bumps before the payout is read (bug #62, LSZ6YOW6)
# ---------------------------------------------------------------------------


class TestRocketBossBump:
    """Rocket's +$2-per-boss bump fires in the end_of_round pass
    (card.lua:2896, needs the blind) BEFORE calc_dollar_bonus builds
    the payout rows — a boss-round cash-out pays the bumped value."""

    def test_boss_round_pays_bumped_value(self):
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.game import _joker_end_of_round_effects
        from jackdaw.engine.blind import Blind
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "ROCKET1")
        rocket = create_joker("j_rocket")
        gs["jokers"] = [rocket]
        gs["blind"] = Blind.create("bl_hook", ante=1)  # boss
        eor = _joker_end_of_round_effects(gs)
        assert rocket.ability["extra"]["dollars"] == 3  # 1 + 2
        assert eor["dollars_earned"] == 3  # post-bump value paid

    def test_non_boss_round_no_bump(self):
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.game import _joker_end_of_round_effects
        from jackdaw.engine.blind import Blind
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "ROCKET2")
        rocket = create_joker("j_rocket")
        gs["jokers"] = [rocket]
        gs["blind"] = Blind.create("bl_small", ante=1)
        eor = _joker_end_of_round_effects(gs)
        assert rocket.ability["extra"]["dollars"] == 1
        assert eor["dollars_earned"] == 1
