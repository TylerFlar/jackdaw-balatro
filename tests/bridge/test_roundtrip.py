"""BalatroBot's API on the simulator, end to end: full runs, the gamestate view's
schema at each phase, blind scores and money (``SimBackend``)."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import NOT_ALLOWED, RPCError, SimBackend
from tests.bot.schema import schemas, validate, validate_gamestate

# ============================================================================
# Helpers
# ============================================================================

_VALID_STATES = {
    "BLIND_SELECT",
    "SELECTING_HAND",
    "ROUND_EVAL",
    "SHOP",
    "SMODS_BOOSTER_OPENED",
    "GAME_OVER",
}


def _start(sim: SimBackend, seed: str = "M15_ROUNDTRIP") -> dict:
    return sim.handle("start", {"deck": "RED", "stake": "WHITE", "seed": seed})


def _beat_blind(sim: SimBackend) -> dict:
    """``set`` the round's chips to the blind's score, then play one card."""
    view = sim.handle("gamestate", None)
    current = next(b for b in view["blinds"].values() if b["status"] == "CURRENT")
    sim.handle("set", {"chips": current["score"]})
    return sim.handle("play", {"cards": [0]})


def _assert_valid_gamestate(gs: dict) -> None:
    """The view fits BalatroBot's GameState schema, and its areas count their cards."""
    validate_gamestate(gs)
    assert gs["state"] in _VALID_STATES
    assert isinstance(gs["seed"], str) and gs["seed"]
    for btype in ("small", "big", "boss"):
        assert gs["blinds"][btype]["type"] == btype.upper()
    for area_key in ("jokers", "consumables", "cards", "hand"):
        area = gs[area_key]
        assert area["count"] == len(area["cards"]), area_key
    assert len(gs["hands"]) == 12  # every hand, secret ones included


def _assert_valid_card(card: dict) -> None:
    """A card fits BalatroBot's Card schema."""
    validate(card, schemas()["Card"])


# ============================================================================
# 1. Full run through backend
# ============================================================================


class TestFullRunThroughBackend:
    def test_three_antes_or_game_over(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_FULL_RUN")
        _assert_valid_gamestate(gs)
        assert gs["state"] == "BLIND_SELECT"

        max_steps = 200
        steps = 0

        while steps < max_steps:
            state = gs["state"]

            if state == "GAME_OVER":
                break

            if state == "BLIND_SELECT":
                gs = sim.handle("select", {})
                steps += 1

            elif state == "SELECTING_HAND":
                hand_count = gs["hand"]["count"]
                n = min(5, hand_count)
                if n == 0 or gs["round"]["hands_left"] <= 0:
                    break
                gs = sim.handle("play", {"cards": list(range(n))})
                steps += 1

            elif state == "ROUND_EVAL":
                gs = sim.handle("cash_out", {})
                steps += 1

            elif state == "SHOP":
                gs = sim.handle("next_round", {})
                steps += 1

            else:
                pytest.fail(f"Unexpected state: {state}")

            _assert_valid_gamestate(gs)

            # Ante 4 reached = 3 antes beaten, good enough
            if gs["ante_num"] >= 4:
                break

        # Should have progressed past ante 1
        assert gs["ante_num"] >= 2 or gs["state"] == "GAME_OVER"
        assert steps > 0

    def test_state_transitions_are_valid(self):
        """Verify each step produces a valid next state."""
        sim = SimBackend()
        gs = _start(sim, seed="M15_TRANSITIONS")

        valid_next = {
            "BLIND_SELECT": {"SELECTING_HAND"},
            "SELECTING_HAND": {"SELECTING_HAND", "ROUND_EVAL", "GAME_OVER"},
            "ROUND_EVAL": {"SHOP"},
            "SHOP": {"BLIND_SELECT", "SMODS_BOOSTER_OPENED"},
        }

        prev_state = gs["state"]
        for _ in range(50):
            state = gs["state"]
            if state == "GAME_OVER":
                break

            if state == "BLIND_SELECT":
                gs = sim.handle("select", {})
            elif state == "SELECTING_HAND":
                n = min(5, gs["hand"]["count"])
                if n == 0:
                    break
                gs = sim.handle("play", {"cards": list(range(n))})
            elif state == "ROUND_EVAL":
                gs = sim.handle("cash_out", {})
            elif state == "SHOP":
                gs = sim.handle("next_round", {})
            else:
                break

            new_state = gs["state"]
            allowed = valid_next.get(prev_state, set())
            assert new_state in allowed, (
                f"Invalid transition {prev_state} → {new_state} (expected one of {allowed})"
            )
            prev_state = new_state


# ============================================================================
# 2. Gamestate schema completeness at each phase
# ============================================================================


class TestGamestateSchemaCompleteness:
    def test_blind_select_phase(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_SCHEMA_1")
        assert gs["state"] == "BLIND_SELECT"
        _assert_valid_gamestate(gs)

    def test_selecting_hand_phase(self):
        sim = SimBackend()
        _start(sim, seed="M15_SCHEMA_2")
        gs = sim.handle("select", {})
        assert gs["state"] == "SELECTING_HAND"
        _assert_valid_gamestate(gs)
        assert gs["hand"]["count"] > 0
        assert gs["round"]["hands_left"] > 0

    def test_round_eval_phase(self):
        sim = SimBackend()
        _start(sim, seed="M15_SCHEMA_3")
        sim.handle("select", {})
        gs = _beat_blind(sim)
        assert gs["state"] == "ROUND_EVAL"
        _assert_valid_gamestate(gs)

    def test_shop_phase(self):
        sim = SimBackend()
        _start(sim, seed="M15_SCHEMA_4")
        sim.handle("select", {})
        _beat_blind(sim)
        gs = sim.handle("cash_out", {})
        assert gs["state"] == "SHOP"
        _assert_valid_gamestate(gs)
        for area_key in ("shop", "vouchers", "packs"):
            assert gs[area_key]["count"] == len(gs[area_key]["cards"]) > 0

    def test_hands_dict_structure(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_SCHEMA_5")
        hands = gs["hands"]
        assert len(hands) == 12
        for name, h in hands.items():
            assert isinstance(name, str)
            assert isinstance(h["order"], int)
            assert isinstance(h["level"], int) and h["level"] >= 1
            assert isinstance(h["chips"], int) and h["chips"] > 0
            assert isinstance(h["mult"], int) and h["mult"] >= 1
            assert isinstance(h["played"], int)
            assert isinstance(h["played_this_round"], int)


# ============================================================================
# 3. Card schema in context
# ============================================================================


class TestCardSchemaInContext:
    def test_hand_card_has_full_schema(self):
        sim = SimBackend()
        _start(sim, seed="M15_CARD_SCHEMA")
        gs = sim.handle("select", {})
        assert gs["state"] == "SELECTING_HAND"
        assert gs["hand"]["count"] > 0

        card = gs["hand"]["cards"][0]
        _assert_valid_card(card)

        # Playing card should have suit/rank
        assert card["value"]["suit"] in ("H", "D", "C", "S")
        assert card["value"]["rank"] in (
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "T",
            "J",
            "Q",
            "K",
            "A",
        )

    def test_deck_cards_have_full_schema(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_CARD_SCHEMA_2")
        assert gs["cards"]["count"] > 0
        for card in gs["cards"]["cards"][:5]:
            _assert_valid_card(card)


# ============================================================================
# 4. Blind scores increase with ante
# ============================================================================


class TestBlindScoresIncreaseWithAnte:
    def test_boss_score_increases(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_SCALING")
        ante1_boss_score = gs["blinds"]["boss"]["score"]
        assert ante1_boss_score > 0

        # Play through 3 blinds of ante 1
        for _ in range(3):
            sim.handle("select", {})
            gs = _beat_blind(sim)
            # May need multiple hands if somehow didn't beat
            for _ in range(5):
                if gs["state"] != "SELECTING_HAND":
                    break
                n = min(5, gs["hand"]["count"])
                if n == 0:
                    break
                gs = sim.handle("play", {"cards": list(range(n))})
            if gs["state"] == "GAME_OVER":
                pytest.skip("Game over before reaching ante 2")
            gs = sim.handle("cash_out", {})
            if gs["state"] == "SHOP":
                gs = sim.handle("next_round", {})

        assert gs["ante_num"] >= 2
        ante2_boss_score = gs["blinds"]["boss"]["score"]
        assert ante2_boss_score > ante1_boss_score


# ============================================================================
# 5. Money changes on actions
# ============================================================================


class TestMoneyChangesOnActions:
    def test_reroll_costs_money(self):
        sim = SimBackend()
        _start(sim, seed="M15_MONEY")

        # Get to shop
        sim.handle("select", {})
        _beat_blind(sim)
        gs = sim.handle("cash_out", {})
        assert gs["state"] == "SHOP"

        money_before = gs["money"]
        reroll_cost = gs["round"]["reroll_cost"]

        if money_before >= reroll_cost:
            gs = sim.handle("reroll", {})
            assert gs["money"] == money_before - reroll_cost
        else:
            # Can't afford reroll: verify it fails
            with pytest.raises(RPCError, match="Not enough dollars to reroll") as exc_info:
                sim.handle("reroll", {})
            assert exc_info.value.code == NOT_ALLOWED

    def test_money_after_cashout_increased(self):
        sim = SimBackend()
        _start(sim, seed="M15_MONEY_2")

        money_at_start = sim.handle("gamestate", None)["money"]

        sim.handle("select", {})
        _beat_blind(sim)
        gs = sim.handle("cash_out", {})

        # After beating a blind and cashing out, money should increase
        assert gs["money"] > money_at_start
