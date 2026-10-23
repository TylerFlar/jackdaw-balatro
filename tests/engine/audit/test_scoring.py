"""Scoring-pipeline fidelity (G20, G34, G32 and related)."""

from __future__ import annotations

from ._helpers import discard, give_card, give_joker, new_run, play, rng_twin, select_blind


class TestDestructionPass:
    """state_events.lua:950-973: every scoring card, debuffed or not, is
    offered to the jokers' ``destroying_card``; then a non-debuffed Glass Card
    rolls 'glass' whether or not a joker already destroyed it."""

    def test_sixth_sense_destroys_a_debuffed_six(self):
        """Sixth Sense checks full_hand[1]:get_id() == 6 (card.lua:2604), which
        ignores the debuff."""
        gs = select_blind(new_run("AUDIT_SIXTH"))
        give_joker(gs, "j_sixth_sense")
        six = give_card(gs, "H_6")
        six.debuff = True
        play(gs, six)
        owned = gs["deck"] + gs["hand"] + gs["discard_pile"]
        assert six not in owned
        assert [c.ability["set"] for c in gs["consumables"]] == ["Spectral"]

    def test_glass_rolls_even_when_a_joker_destroyed_the_card(self):
        gs = select_blind(new_run("AUDIT_SIXTH_GLASS"))
        give_joker(gs, "j_sixth_sense")
        six = give_card(gs, "H_6", enhancement="m_glass")
        oracle = rng_twin(gs)
        oracle.seed("glass")
        play(gs, six)
        assert six.shattered
        assert gs["rng"].state["glass"] == oracle.state["glass"]

    def test_debuffed_glass_does_not_roll(self):
        gs = select_blind(new_run("AUDIT_GLASS_DEBUFF"))
        glass = give_card(gs, "S_9", enhancement="m_glass")
        glass.debuff = True
        play(gs, glass)
        assert "glass" not in gs["rng"].state


class TestPurpleSeal:
    def test_debuffed_purple_seal_makes_no_tarot(self):
        """Card:calculate_seal returns at once for a debuffed card
        (card.lua:2242-2243), so a debuffed Purple Seal discard creates
        nothing."""
        gs = select_blind(new_run("AUDIT_PURPLE"))
        card = give_card(gs, "H_5", seal="Purple")
        card.debuff = True
        discard(gs, card)
        assert gs["consumables"] == []

    def test_purple_seal_makes_a_tarot(self):
        gs = select_blind(new_run("AUDIT_PURPLE2"))
        card = give_card(gs, "H_5", seal="Purple")
        discard(gs, card)
        assert [c.ability["set"] for c in gs["consumables"]] == ["Tarot"]
