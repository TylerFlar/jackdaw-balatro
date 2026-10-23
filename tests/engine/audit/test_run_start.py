"""Run start order (G129)."""

from __future__ import annotations

from jackdaw.engine.data.enums import Suit

from ._joker_helpers import new_run


class TestStartRun:
    """``start_run`` fills the first round's hands and discards (game.lua:2373-2374)
    and rolls the round targets from the deck it built; the Checkered Deck's
    suit change is an event ``Back:apply_to_run`` queued (back.lua:233-247), so it
    lands after those rolls."""

    def test_the_first_round_has_its_hands_and_discards(self):
        gs = new_run("AUDIT_START_COUNTS")
        assert gs["current_round"]["hands_left"] == gs["round_resets"]["hands"]
        assert gs["current_round"]["discards_left"] == gs["round_resets"]["discards"]

    def test_checkered_rolls_its_targets_before_the_suit_change(self):
        seen = set()
        for i in range(40):
            gs = new_run(f"AUDIT_CHECKERED_{i}", back="b_checkered")
            assert all(c.base.suit in (Suit.SPADES, Suit.HEARTS) for c in gs["deck"])
            seen.add(gs["current_round"]["castle_card"]["suit"])
        # The deck had Clubs and Diamonds when the Castle's suit was rolled.
        assert seen & {"Clubs", "Diamonds"}
