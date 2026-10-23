"""Scoring-pipeline fidelity (G20, G34, G32 and related)."""

from __future__ import annotations

from ._helpers import (
    discard,
    give_card,
    give_consumable,
    give_joker,
    new_run,
    play,
    rng_twin,
    select_blind,
)


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


class TestMimeWhileScoring:
    """Held cards are retriggered (Red Seal, Mime) only when their first pass
    produced something: next(effects[1]) or #effects > 1
    (state_events.lua:809-829, card.lua:3386-3394)."""

    def test_reserved_parking_is_only_retriggered_after_a_hit(self):
        gs = select_blind(new_run("AUDIT_MIME_PARKING"))
        give_joker(gs, "j_mime")
        give_joker(gs, "j_reserved_parking")
        for key in ("H_K", "S_Q", "D_J", "C_K"):
            give_card(gs, key)
        played = give_card(gs, "S_2")
        oracle = rng_twin(gs)
        held = [c for c in gs["hand"] if c is not played]
        hits = misses = 0
        for card in held:
            if card.debuff or card.get_id() not in (11, 12, 13):
                continue
            if oracle.random("parking") < 1 / 2:  # paid: Mime retriggers it
                hits += 1
                hits += oracle.random("parking") < 1 / 2
            else:
                misses += 1  # nothing to repeat
        assert misses, "seed must include a missed roll"
        dollars = gs["dollars"]
        play(gs, played)
        assert gs["rng"].state["parking"] == oracle.state["parking"]
        assert gs["dollars"] == dollars + hits


class TestRoundEndRetriggers:
    """end_round's held-card loop (state_events.lua:171-233) recomputes
    get_end_of_round_effect (Gold Card $, Blue Seal Planet, card.lua:1033-1065)
    for every repetition; Red Seal and Mime (card.lua:2877-2886) add
    repetitions when the first pass had an effect."""

    def test_mime_repeats_a_held_gold_card(self):
        gs = select_blind(new_run("AUDIT_EOR_GOLD"))
        give_joker(gs, "j_mime")
        give_card(gs, "S_3", enhancement="m_gold")
        played = give_card(gs, "S_2")
        gs["blind"].chips = 1
        dollars = gs["dollars"]
        play(gs, played)
        assert gs["dollars"] == dollars + 2 * 3

    def test_mime_repeats_a_held_blue_seal(self):
        gs = select_blind(new_run("AUDIT_EOR_BLUE"))
        give_joker(gs, "j_mime")
        give_card(gs, "S_3", seal="Blue")
        played = give_card(gs, "S_2")
        gs["blind"].chips = 1
        play(gs, played)
        assert [c.center_key for c in gs["consumables"]] == ["c_pluto", "c_pluto"]
        assert gs["consumeable_buffer"] == 0

    def test_blue_seal_and_mime_share_the_buffer(self):
        """One free slot: the first pass reserves it, the Mime repeat finds no
        room, so one Planet."""
        gs = select_blind(new_run("AUDIT_EOR_BLUE1"))
        give_joker(gs, "j_mime")
        give_consumable(gs, "c_mercury")
        give_card(gs, "S_3", seal="Blue")
        played = give_card(gs, "S_2")
        gs["blind"].chips = 1
        play(gs, played)
        assert [c.center_key for c in gs["consumables"]] == ["c_mercury", "c_pluto"]


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
