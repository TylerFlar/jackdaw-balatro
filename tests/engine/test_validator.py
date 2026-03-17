"""Tests for jackdaw.engine.validator — state comparison.

Coverage
--------
* validate_step detects dollar mismatch.
* validate_step detects ante mismatch.
* validate_step detects chip mismatch.
* validate_step detects hand size mismatch.
* validate_step returns empty list when states match.
* validate_hand_cards detects suit/rank mismatch per card.
* validate_jokers detects count and key mismatches.
* format_report produces readable output.
* Integration: validate_step on actual sim state vs mock live.
"""

from __future__ import annotations

from jackdaw.engine.validator import (
    format_report,
    validate_hand_cards,
    validate_jokers,
    validate_step,
)


def _sim_state(**overrides):
    base = {
        "dollars": 10,
        "chips": 500,
        "round_resets": {"ante": 1},
        "current_round": {"hands_left": 4, "discards_left": 3},
        "hand": [],
        "deck": [None] * 44,
        "jokers": [],
        "blind": None,
    }
    base.update(overrides)
    return base


def _live_state(**overrides):
    base = {
        "money": 10,
        "ante": 1,
        "round": {"chips": 500, "hands_left": 4, "discards_left": 3},
        "hand": [],
        "jokers": [],
        "deck_size": 44,
        "blind": {"chips": 0},
    }
    base.update(overrides)
    return base


class TestValidateStep:
    def test_matching_states_no_diffs(self):
        diffs = validate_step(_sim_state(), _live_state())
        assert diffs == []

    def test_dollar_mismatch(self):
        diffs = validate_step(_sim_state(dollars=10), _live_state(money=15))
        assert any("dollars" in d for d in diffs)

    def test_ante_mismatch(self):
        sim = _sim_state()
        sim["round_resets"]["ante"] = 3
        diffs = validate_step(sim, _live_state(ante=2))
        assert any("ante" in d for d in diffs)

    def test_chip_mismatch(self):
        diffs = validate_step(_sim_state(chips=100), _live_state())
        assert any("chips" in d for d in diffs)

    def test_hands_left_mismatch(self):
        sim = _sim_state()
        sim["current_round"]["hands_left"] = 2
        diffs = validate_step(sim, _live_state())
        assert any("hands_left" in d for d in diffs)

    def test_deck_size_mismatch(self):
        diffs = validate_step(
            _sim_state(deck=[None] * 30),
            _live_state(deck_size=32),
        )
        assert any("deck_size" in d for d in diffs)

    def test_joker_count_mismatch(self):
        class FakeJoker:
            center_key = "j_joker"
        diffs = validate_step(
            _sim_state(jokers=[FakeJoker()]),
            _live_state(jokers=[]),
        )
        assert any("joker_count" in d for d in diffs)


class TestValidateHandCards:
    def test_matching_hands(self):
        class FakeCard:
            class base:
                class suit:
                    value = "Spades"
                class rank:
                    value = "Ace"
        diffs = validate_hand_cards(
            [FakeCard()],
            [{"suit": "Spades", "rank": "Ace"}],
        )
        assert diffs == []

    def test_suit_mismatch(self):
        class FakeCard:
            class base:
                class suit:
                    value = "Hearts"
                class rank:
                    value = "Ace"
        diffs = validate_hand_cards(
            [FakeCard()],
            [{"suit": "Spades", "rank": "Ace"}],
        )
        assert len(diffs) == 1
        assert "hand[0]" in diffs[0]

    def test_length_mismatch(self):
        diffs = validate_hand_cards([], [{"suit": "Spades", "rank": "Ace"}])
        assert any("missing in sim" in d for d in diffs)


class TestValidateJokers:
    def test_matching(self):
        class FJ:
            center_key = "j_joker"
        diffs = validate_jokers([FJ()], [{"key": "j_joker"}])
        assert diffs == []

    def test_key_mismatch(self):
        class FJ:
            center_key = "j_joker"
        diffs = validate_jokers([FJ()], [{"key": "j_banner"}])
        assert any("joker[0].key" in d for d in diffs)


class TestFormatReport:
    def test_clean_report(self):
        report = format_report([[], []], "SEED")
        assert "Clean steps: 2/2" in report
        assert "Total discrepancies: 0" in report

    def test_report_with_diffs(self):
        report = format_report([["dollars: sim=10 live=15"], []], "SEED")
        assert "Clean steps: 1/2" in report
        assert "Total discrepancies: 1" in report
        assert "dollars: sim=10 live=15" in report


class TestIntegration:
    def test_sim_matches_mock(self):
        """When mock derives from sim state, no discrepancies."""
        from jackdaw.engine.run_init import initialize_run
        from jackdaw.engine.actions import GamePhase

        gs = initialize_run("b_red", 1, "VALID_INT")
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"

        # Build a mock live state from the sim state
        live = {
            "money": gs["dollars"],
            "ante": gs["round_resets"]["ante"],
            "round": {
                "chips": gs.get("chips", 0),
                "hands_left": gs["current_round"]["hands_left"],
                "discards_left": gs["current_round"]["discards_left"],
            },
            "hand": [],
            "jokers": [],
            "deck_size": len(gs["deck"]),
            "blind": {"chips": 0},
        }
        diffs = validate_step(gs, live)
        assert diffs == []
