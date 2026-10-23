"""Joker fidelity (G16, G17, G18, G21, G24, G25 and related)."""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase, NextRound, UseConsumable
from jackdaw.engine.game import step

from ._helpers import (
    discard,
    give_card,
    give_consumable,
    give_joker,
    new_run,
    play,
    select_blind,
    to_shop,
)


class TestBlueprintGuards:
    """Blueprint/Brainstorm copy a joker by calling its calculate_joker with
    context.blueprint set (card.lua:2304-2334); only branches guarded by
    ``not context.blueprint`` refuse.  Hiker, DNA, Perkeo and Burnt Joker have
    no guard; Trading Card has one."""

    def test_hiker_is_copied(self):
        """Hiker: individual/play, no guard (card.lua:3067-3074)."""
        gs = select_blind(new_run("AUDIT_BP_HIKER"))
        give_joker(gs, "j_blueprint")
        give_joker(gs, "j_hiker")
        card = give_card(gs, "S_9")
        play(gs, card)
        assert card.ability["perma_bonus"] == 10

    def test_hiker_is_copied_by_brainstorm(self):
        gs = select_blind(new_run("AUDIT_BS_HIKER"))
        give_joker(gs, "j_hiker")
        give_joker(gs, "j_brainstorm")
        card = give_card(gs, "S_9")
        play(gs, card)
        assert card.ability["perma_bonus"] == 10

    def test_dna_is_copied(self):
        """DNA: before, no guard (card.lua:3501-3523)."""
        gs = select_blind(new_run("AUDIT_BP_DNA"))
        give_joker(gs, "j_blueprint")
        give_joker(gs, "j_dna")
        card = give_card(gs, "S_9")
        play(gs, card)
        copies = [c for c in gs["hand"] if c.card_key == "S_9" and c is not card]
        assert len(copies) == 2

    def test_burnt_joker_is_copied(self):
        """Burnt Joker: pre_discard, no guard (card.lua:2748-2755): the first
        discard's hand levels up once per copy."""
        gs = select_blind(new_run("AUDIT_BP_BURNT"))
        give_joker(gs, "j_blueprint")
        give_joker(gs, "j_burnt")
        a = give_card(gs, "S_9")
        b = give_card(gs, "H_9")
        discard(gs, a, b)
        assert gs["hand_levels"]["Pair"].level == 3

    def test_perkeo_is_copied(self):
        """Perkeo: ending_shop, no guard (card.lua:2412-2426)."""
        gs = to_shop(new_run("AUDIT_BP_PERKEO"))
        give_joker(gs, "j_blueprint")
        give_joker(gs, "j_perkeo")
        give_consumable(gs, "c_pluto")
        step(gs, NextRound())
        assert len(gs["consumables"]) == 3

    def test_trading_card_is_not_copied(self):
        """Trading Card: discard, guarded (card.lua:2802-2812): $3 once."""
        gs = select_blind(new_run("AUDIT_BP_TRADING"))
        give_joker(gs, "j_blueprint")
        give_joker(gs, "j_trading")
        card = give_card(gs, "S_9")
        dollars = gs["dollars"]
        discard(gs, card)
        assert gs["dollars"] == dollars + 3
        assert card not in gs["discard_pile"]


class TestMrBones:
    def test_save_pays_no_blind_reward(self):
        """evaluate_round pays the blind only when G.GAME.chips >=
        G.GAME.blind.chips; a Mr. Bones save shows the blind row with $0
        (state_events.lua:1139-1146).  The other rows are unchanged."""
        gs = select_blind(new_run("AUDIT_BONES"))
        give_joker(gs, "j_mr_bones")
        card = give_card(gs, "S_2")
        gs["current_round"]["hands_left"] = 1
        gs["chips"] = int(gs["blind"].chips * 0.3)
        play(gs, card)
        assert gs["phase"] == GamePhase.ROUND_EVAL
        assert not any(j.center_key == "j_mr_bones" for j in gs["jokers"])
        earnings = gs["round_earnings"]
        assert earnings.blind_reward == 0
        assert earnings.total == (
            earnings.unused_hands_bonus
            + earnings.unused_discards_bonus
            + earnings.joker_dollars
            + earnings.interest
            - earnings.rental_cost
        )

    def test_beaten_blind_still_pays(self):
        gs = select_blind(new_run("AUDIT_BONES2"))
        gs["blind"].chips = 1
        play(gs, gs["hand"][0])
        assert gs["round_earnings"].blind_reward == gs["blind"].dollars > 0


class TestSatellite:
    def test_pays_per_distinct_planet_used(self):
        """calculate_dollar_bonus pays extra x the number of distinct Planet
        keys in G.GAME.consumeable_usage (card.lua:1667-1674)."""
        gs = select_blind(new_run("AUDIT_SATELLITE"))
        give_joker(gs, "j_satellite")
        for key in ("c_pluto", "c_mercury", "c_pluto"):
            give_consumable(gs, key)
            step(gs, UseConsumable(card_index=0))
        gs["blind"].chips = 1
        play(gs, gs["hand"][0])
        assert gs["round_earnings"].joker_dollars == 2
