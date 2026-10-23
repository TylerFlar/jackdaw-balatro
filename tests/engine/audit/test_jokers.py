"""Joker fidelity (G16, G17, G18, G21, G24, G25 and related)."""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase, NextRound, UseConsumable
from jackdaw.engine.game import step

from ._helpers import (
    discard,
    give_card,
    give_consumable,
    give_joker,
    indices,
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


class TestEffectFlags:
    """Card:is_face and Card:is_suit consult find_joker('Pareidolia') /
    find_joker('Smeared Joker') themselves (card.lua:964-970, 4064-4089), and
    evaluate_poker_hand reads Four Fingers / Shortcut / Smeared, so every
    caller sees them."""

    def test_ride_the_bus_sees_pareidolia(self):
        """Ride the Bus resets when a scoring card is_face() (card.lua:3525-3541)."""
        gs = select_blind(new_run("AUDIT_BUS"))
        give_joker(gs, "j_pareidolia")
        bus = give_joker(gs, "j_ride_the_bus")
        play(gs, give_card(gs, "S_2"))
        assert bus.ability["mult"] == 0

    def test_caino_sees_pareidolia(self):
        """Caino counts destroyed cards that is_face() (card.lua:2672-2684)."""
        gs = select_blind(new_run("AUDIT_CAINO"))
        give_joker(gs, "j_pareidolia")
        caino = give_joker(gs, "j_caino")
        card = give_card(gs, "S_2")
        give_consumable(gs, "c_hanged_man")
        step(gs, UseConsumable(card_index=0, target_indices=indices(gs, card)))
        assert caino.ability["caino_xmult"] == 2

    def test_flower_pot_sees_smeared(self):
        """Flower Pot's is_suit (card.lua:3807-3839): with Smeared a Heart also
        fills Diamonds and a Spade also fills Clubs."""
        gs = select_blind(new_run("AUDIT_FLOWER"))
        give_joker(gs, "j_smeared")
        give_joker(gs, "j_flower_pot")
        cards = [give_card(gs, k) for k in ("H_2", "H_3", "S_2", "S_3")]
        play(gs, *cards)
        # Two Pair L1: 20 chips x2; 2+3+2+3 chips; Flower Pot x3
        assert gs["last_score_result"].total == (20 + 10) * 2 * 3

    def test_seeing_double_sees_smeared(self):
        """Seeing Double (card.lua:3840-3872): with Smeared a Spade counts as a
        Club."""
        gs = select_blind(new_run("AUDIT_DOUBLE"))
        give_joker(gs, "j_smeared")
        give_joker(gs, "j_seeing_double")
        cards = [give_card(gs, "S_5"), give_card(gs, "H_5")]
        play(gs, *cards)
        # Pair L1: 10 chips x2; 5+5 chips; Seeing Double x2
        assert gs["last_score_result"].total == (10 + 10) * 2 * 2

    def test_faceless_joker_sees_pareidolia(self):
        """Faceless Joker counts discarded cards that is_face() (card.lua:2858-2872)."""
        gs = select_blind(new_run("AUDIT_FACELESS"))
        give_joker(gs, "j_pareidolia")
        give_joker(gs, "j_faceless")
        cards = [give_card(gs, k) for k in ("S_2", "S_3", "S_4")]
        dollars = gs["dollars"]
        discard(gs, *cards)
        assert gs["dollars"] == dollars + 5

    def test_castle_sees_smeared(self):
        """Castle's discard check is_suit(castle_card.suit) (card.lua:2814-2824)."""
        gs = select_blind(new_run("AUDIT_CASTLE"))
        give_joker(gs, "j_smeared")
        castle = give_joker(gs, "j_castle")
        suit = gs["current_round"]["castle_card"]["suit"]
        twin = {"Spades": "C", "Clubs": "S", "Hearts": "D", "Diamonds": "H"}[suit]
        discard(gs, give_card(gs, f"{twin}_2"))
        assert castle.ability["extra"]["chips"] == 3

    def test_burnt_joker_detection_sees_four_fingers(self):
        """Burnt Joker levels the hand get_poker_hand_info finds
        (card.lua:2750), which honours Four Fingers."""
        gs = select_blind(new_run("AUDIT_BURNT_FF"))
        give_joker(gs, "j_four_fingers")
        give_joker(gs, "j_burnt")
        cards = [give_card(gs, k) for k in ("H_2", "H_5", "H_9", "H_K")]
        discard(gs, *cards)
        assert gs["hand_levels"]["Flush"].level == 2


class TestDollarBuffer:
    """Money earned while a hand is evaluated is eased in later events, so
    G.GAME.dollars is frozen during evaluate_play; every scoring payout also
    adds itself to G.GAME.dollar_buffer (get_p_dollars card.lua:1084-1087;
    Golden Ticket 3152, Business Card 3178, Rough Gem 3226, Reserved Parking
    3312, To Do List 3493, Matador 2739/3722) until an event after the hand
    zeroes it.  Bull and Bootstraps read dollars + dollar_buffer (card.lua:3936,
    4046)."""

    def test_bull_sees_money_scored_earlier_in_the_hand(self):
        gs = select_blind(new_run("AUDIT_BULL"))
        give_joker(gs, "j_bull")
        card = give_card(gs, "S_2", seal="Gold")
        dollars = gs["dollars"]
        play(gs, card)
        # High Card L1 5 chips + 2; Bull +2 per $ of (dollars + the Gold Seal's $3)
        assert gs["last_score_result"].total == 5 + 2 + 2 * (dollars + 3)
        assert gs["dollars"] == dollars + 3

    def test_bootstraps_sees_money_scored_earlier_in_the_hand(self):
        gs = select_blind(new_run("AUDIT_BOOTSTRAPS"))
        give_joker(gs, "j_bootstraps")
        card = give_card(gs, "S_2", seal="Gold")
        assert gs["dollars"] == 4  # 4 // 5 == 0, (4 + 3) // 5 == 1
        play(gs, card)
        assert gs["last_score_result"].total == (5 + 2) * (1 + 2)

    def test_to_do_list_pays_before_scoring(self):
        """To Do List pays in the ``before`` pass (card.lua:3491-3499), so a Bull
        to its left already sees the $4."""
        gs = select_blind(new_run("AUDIT_TODO_BEFORE"))
        give_joker(gs, "j_bull")
        todo = give_joker(gs, "j_todo_list")
        todo.ability["to_do_poker_hand"] = "High Card"
        card = give_card(gs, "S_2")
        dollars = gs["dollars"]
        play(gs, card)
        assert gs["last_score_result"].total == 5 + 2 + 2 * (dollars + 4)
        assert gs["dollars"] == dollars + 4


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
