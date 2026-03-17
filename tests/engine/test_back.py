"""Tests for jackdaw.engine.back — Back.apply_to_run and Back.trigger_effect.

Coverage
--------
* apply_to_run: all 16 backs produce the correct mutation keys/values.
* trigger_effect('final_scoring_step'): Plasma Deck averages chips + mult.
* trigger_effect('eval'): Anaglyph Deck creates Double Tag after boss.
* Scoring integration: score_hand Phase 10 refactored from hardcode to Back.
"""

from __future__ import annotations

from jackdaw.engine.back import Back
from jackdaw.engine.data.prototypes import BACKS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def back(key: str) -> Back:
    return Back(key)


# ---------------------------------------------------------------------------
# Smoke: all backs construct without error
# ---------------------------------------------------------------------------


class TestBackConstruction:
    def test_all_backs_construct(self):
        for key in BACKS:
            b = Back(key)
            assert b.key == key
            assert b.name != ""
            assert isinstance(b.config, dict)

    def test_back_with_list_config_gets_empty_dict(self):
        """Backs with config=[] (e.g. b_anaglyph, b_checkered) get {} config."""
        for key in ("b_anaglyph", "b_checkered", "b_challenge"):
            if key in BACKS:
                b = Back(key)
                assert b.config == {}


# ---------------------------------------------------------------------------
# apply_to_run — individual deck assertions
# ---------------------------------------------------------------------------


class TestApplyToRun:
    """Each deck's apply_to_run returns the correct mutation dict."""

    def test_red_deck_discards_delta(self):
        m = back("b_red").apply_to_run({})
        assert m["discards_delta"] == 1
        assert "hands_delta" not in m

    def test_blue_deck_hands_delta(self):
        m = back("b_blue").apply_to_run({})
        assert m["hands_delta"] == 1
        assert "discards_delta" not in m

    def test_black_deck_hands_minus_joker_slot_plus(self):
        m = back("b_black").apply_to_run({})
        assert m["hands_delta"] == -1
        assert m["joker_slots_delta"] == 1

    def test_yellow_deck_dollars_delta(self):
        m = back("b_yellow").apply_to_run({})
        assert m["dollars_delta"] == 10

    def test_green_deck_hand_bonus_discard_bonus_no_interest(self):
        m = back("b_green").apply_to_run({})
        assert m["money_per_hand"] == 2
        assert m["money_per_discard"] == 1
        assert m["no_interest"] is True
        assert "dollars_delta" not in m

    def test_magic_deck_consumables_and_voucher(self):
        m = back("b_magic").apply_to_run({})
        # Two Fools as starting consumables
        assert m["starting_consumables"] == ["c_fool", "c_fool"]
        # Crystal Ball voucher
        assert m["starting_vouchers"] == ["v_crystal_ball"]

    def test_nebula_deck_consumable_slot_minus_and_telescope(self):
        m = back("b_nebula").apply_to_run({})
        assert m["consumable_slots_delta"] == -1
        assert m["starting_vouchers"] == ["v_telescope"]

    def test_ghost_deck_consumable_and_spectral_rate(self):
        m = back("b_ghost").apply_to_run({})
        assert m["starting_consumables"] == ["c_hex"]
        assert m["spectral_rate"] == 2

    def test_plasma_deck_ante_scaling(self):
        m = back("b_plasma").apply_to_run({})
        assert m["ante_scaling"] == 2

    def test_painted_deck_hand_size_and_joker_slot(self):
        m = back("b_painted").apply_to_run({})
        assert m["hand_size_delta"] == 2
        assert m["joker_slots_delta"] == -1

    def test_zodiac_deck_three_vouchers(self):
        m = back("b_zodiac").apply_to_run({})
        assert set(m["starting_vouchers"]) == {
            "v_tarot_merchant",
            "v_planet_merchant",
            "v_overstock_norm",
        }
        assert len(m["starting_vouchers"]) == 3

    def test_anaglyph_deck_no_apply_to_run_mutations(self):
        """Anaglyph has empty config — no starting mutations."""
        m = back("b_anaglyph").apply_to_run({})
        assert m == {}

    def test_checkered_deck_no_apply_to_run_mutations(self):
        """Checkered is deck-builder-only — no starting mutations."""
        m = back("b_checkered").apply_to_run({})
        assert m == {}

    def test_abandoned_deck_no_apply_to_run_mutations(self):
        """Abandoned is deck-builder-only — no starting mutations."""
        m = back("b_abandoned").apply_to_run({})
        assert m == {}

    def test_erratic_deck_no_economy_or_slot_mutations(self):
        """Erratic config has only randomize_rank_suit — no economy mutations."""
        m = back("b_erratic").apply_to_run({})
        assert "hands_delta" not in m
        assert "dollars_delta" not in m
        assert "starting_vouchers" not in m

    def test_apply_to_run_does_not_mutate_game_state(self):
        """apply_to_run must not mutate the passed game_state dict."""
        gs = {"dollars": 5, "hands": 3}
        original = dict(gs)
        back("b_black").apply_to_run(gs)
        assert gs == original

    def test_mutations_are_independent_instances(self):
        """Each apply_to_run call returns a fresh dict."""
        b = back("b_red")
        m1 = b.apply_to_run({})
        m2 = b.apply_to_run({})
        m1["extra"] = True
        assert "extra" not in m2


# ---------------------------------------------------------------------------
# trigger_effect — Plasma Deck
# ---------------------------------------------------------------------------


class TestTriggerEffectPlasma:
    """Plasma Deck averages chips and mult in final_scoring_step."""

    def test_plasma_100_chips_20_mult_gives_60_each(self):
        result = back("b_plasma").trigger_effect("final_scoring_step", chips=100.0, mult=20.0)
        assert result == {"chips": 60, "mult": 60}

    def test_plasma_floors_odd_total(self):
        # 99 + 20 = 119 → floor(119/2) = 59
        result = back("b_plasma").trigger_effect("final_scoring_step", chips=99.0, mult=20.0)
        assert result == {"chips": 59, "mult": 59}

    def test_plasma_zero_total(self):
        result = back("b_plasma").trigger_effect("final_scoring_step", chips=0.0, mult=0.0)
        assert result == {"chips": 0, "mult": 0}

    def test_plasma_equal_chips_and_mult(self):
        # 50 + 50 = 100 → 50 each
        result = back("b_plasma").trigger_effect("final_scoring_step", chips=50.0, mult=50.0)
        assert result == {"chips": 50, "mult": 50}

    def test_plasma_large_values(self):
        # 10000 + 240 = 10240 → 5120 each
        result = back("b_plasma").trigger_effect("final_scoring_step", chips=10000.0, mult=240.0)
        assert result == {"chips": 5120, "mult": 5120}

    def test_non_plasma_final_scoring_step_returns_none(self):
        for key in ("b_red", "b_blue", "b_black", "b_green", "b_anaglyph"):
            if key in BACKS:
                result = back(key).trigger_effect("final_scoring_step", chips=100.0, mult=20.0)
                assert result is None, f"{key} should return None for final_scoring_step"


# ---------------------------------------------------------------------------
# trigger_effect — Anaglyph Deck
# ---------------------------------------------------------------------------


class TestTriggerEffectAnaglyph:
    """Anaglyph Deck creates Double Tag on boss defeat."""

    def test_anaglyph_boss_defeated_creates_double_tag(self):
        result = back("b_anaglyph").trigger_effect("eval", boss_defeated=True)
        assert result == {"create_tag": "tag_double"}

    def test_anaglyph_non_boss_returns_none(self):
        result = back("b_anaglyph").trigger_effect("eval", boss_defeated=False)
        assert result is None

    def test_anaglyph_no_boss_kwarg_returns_none(self):
        result = back("b_anaglyph").trigger_effect("eval")
        assert result is None

    def test_non_anaglyph_eval_returns_none(self):
        for key in ("b_red", "b_plasma", "b_blue"):
            if key in BACKS:
                result = back(key).trigger_effect("eval", boss_defeated=True)
                assert result is None, f"{key} should return None for eval"

    def test_unknown_context_returns_none(self):
        for key in BACKS:
            result = back(key).trigger_effect("unknown_context")
            assert result is None


# ---------------------------------------------------------------------------
# Scoring integration — Phase 10 uses Back.trigger_effect
# ---------------------------------------------------------------------------


class TestScoringPhase10Integration:
    """score_hand Phase 10 now delegates to Back.trigger_effect, not hardcode."""

    def _make_minimal_scoring_fixtures(self):
        """Return minimal objects needed to call score_hand."""
        from jackdaw.engine.blind import Blind
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit
        from jackdaw.engine.hand_levels import HandLevels
        from jackdaw.engine.rng import PseudoRandom

        card = create_playing_card(Suit.SPADES, Rank.ACE)
        hand_levels = HandLevels()
        rng = PseudoRandom("PHASE10_TEST")
        blind = Blind.create("bl_small", ante=1)
        return card, hand_levels, rng, blind

    def test_plasma_deck_scoring_averages_chips_mult(self):
        """score_hand with back_key='b_plasma' averages chips and mult."""
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit
        from jackdaw.engine.scoring import score_hand

        card, hand_levels, rng, blind = self._make_minimal_scoring_fixtures()
        # Five Aces for a trivially hand-evaluable set
        aces = [create_playing_card(Suit.SPADES, Rank.ACE) for _ in range(5)]

        result = score_hand(
            played_cards=aces,
            held_cards=[],
            jokers=[],
            hand_levels=hand_levels,
            blind=blind,
            rng=rng,
            back_key="b_plasma",
        )
        # Chips and mult must be equal (plasma averaging)
        assert result.chips == result.mult, (
            f"Plasma: chips={result.chips}, mult={result.mult} should be equal"
        )

    def test_non_plasma_scoring_unaffected(self):
        """score_hand with back_key='b_red' does not average chips/mult."""
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit
        from jackdaw.engine.scoring import score_hand

        card, hand_levels, rng, blind = self._make_minimal_scoring_fixtures()
        aces = [create_playing_card(Suit.SPADES, Rank.ACE) for _ in range(5)]

        result_red = score_hand(
            played_cards=aces,
            held_cards=[],
            jokers=[],
            hand_levels=hand_levels,
            blind=blind,
            rng=rng,
            back_key="b_red",
        )
        result_none = score_hand(
            played_cards=aces,
            held_cards=[],
            jokers=[],
            hand_levels=hand_levels,
            blind=blind,
            rng=rng,
            back_key=None,
        )
        # Red Deck has no scoring effect; result should match no-back result
        assert result_red.chips == result_none.chips
        assert result_red.mult == result_none.mult

    def test_no_back_key_no_effect(self):
        """score_hand without back_key runs Phase 10 without mutation."""
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit
        from jackdaw.engine.scoring import score_hand

        card, hand_levels, rng, blind = self._make_minimal_scoring_fixtures()
        aces = [create_playing_card(Suit.SPADES, Rank.ACE) for _ in range(5)]

        result = score_hand(
            played_cards=aces,
            held_cards=[],
            jokers=[],
            hand_levels=hand_levels,
            blind=blind,
            rng=rng,
            back_key=None,
        )
        # Should not crash; chips > mult for standard hand (more chips than mult)
        assert result.chips > 0
        assert result.mult > 0

    def test_plasma_breakdown_message_present(self):
        """Phase 10 breakdown message is appended for Plasma Deck."""
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit
        from jackdaw.engine.scoring import score_hand

        card, hand_levels, rng, blind = self._make_minimal_scoring_fixtures()
        aces = [create_playing_card(Suit.SPADES, Rank.ACE) for _ in range(5)]

        result = score_hand(
            played_cards=aces,
            held_cards=[],
            jokers=[],
            hand_levels=hand_levels,
            blind=blind,
            rng=rng,
            back_key="b_plasma",
        )
        assert any("Plasma" in step for step in result.breakdown), (
            f"No 'Plasma' in breakdown: {result.breakdown}"
        )
