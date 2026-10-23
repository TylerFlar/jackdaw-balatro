"""Targeted parity cases for joker batch C: the branches seeded play rarely reaches.

Each case builds a board (jokers, playing cards, a forced boss), then takes its steps; each
step must match the oracle's (:mod:`tests.parity.cases`), and each case also asserts that the
branch it is about fired.

Jokers join without BalatroBot's edition poll (a stray edition would change the scores
asserted here); the oracle added them with its own helper, which the fixture records. Playing
cards join the end of the hand through BalatroBot's ``add``; most cases first move the dealt
hand to the bottom of the deck so that only the cards they add are held. A boss checks cards
created mid-round only on its next debuff pass, so cases that need a boss's debuffs run that
pass (``set_blind(nil, true)``) after adding their cards. Rolls are forced by raising
``G.GAME.probabilities.normal``.

Hand scores: High Card 5 chips x1, Pair 10 x2, Two Pair 20 x2, Flush 35 x4 at level 1.
"""

from __future__ import annotations

from typing import Any

from tests.parity.cases import CardRef, Run, front_key


class Board:
    """A run in its first round, whose steps are checked against the oracle's."""

    def __init__(self, name: str, *jokers: str, boss: str | None = None) -> None:
        self.name = name
        start = {"back": "b_red", "stake": 1, "seed": f"BATCHC_{name}", "challenge": None}
        setup: list[dict[str, Any]] = []
        if boss:
            setup.append({"op": "force_boss", "key": boss})
        setup.append({"op": "action", "action": ["SelectBlind"]})
        self.run = Run(start, setup)
        self.jokers = [self.joker(key) for key in jokers]

    # -- setup ----------------------------------------------------------------

    def joker(self, key: str) -> CardRef:
        """A joker added without BalatroBot's edition poll."""
        self.run.oracle(f"{key} added without the edition poll")
        return self.run.add(key)

    def consumable(self, key: str) -> CardRef:
        return self.run.add(key)

    def card(self, key: str, **params: Any) -> CardRef:
        """A playing card added to the end of the hand."""
        return self.run.add(key, **params)

    def cards(self, *keys: str) -> list[CardRef]:
        return [self.card(key) for key in keys]

    def clear_hand(self) -> None:
        """Move the dealt hand to the bottom of the deck."""
        self.run.move("hand", "deck", to_front=True)

    def recheck(self) -> None:
        """The boss's debuff pass over every playing card (``set_blind(nil, true)``)."""
        self.run.recheck()

    def odds(self, normal: float) -> None:
        """``G.GAME.probabilities.normal``."""
        self.run.write("game.probabilities_normal", normal)

    def to_win(self) -> None:
        """One chip short of the blind: the next scoring hand wins the round."""
        self.run.op({"op": "set", "chips": self.run.read("blind.chips") - 1})

    # -- checked steps ----------------------------------------------------------

    def play(self, *cards: CardRef) -> None:
        action = ["PlayHand", self.run.indices(cards)]
        assert self.run.step(action), f"{self.name}: {action} is illegal"

    # -- results ----------------------------------------------------------------

    @property
    def total(self) -> float:
        score = self.run.game.last_score
        assert score is not None
        return score.total

    @property
    def dollars(self) -> float:
        return self.run.game.dollars

    def streams(self) -> dict[str, Any]:
        """The RNG streams drawn so far (``G.GAME.pseudorandom``)."""
        return self.run.read("rng.streams")


# ---------------------------------------------------------------------------
# Scored cards: +chips, +Mult and XMult by rank, suit or face
# ---------------------------------------------------------------------------


def test_fibonacci_counts_aces_and_fibonacci_ranks():
    b = Board("fibonacci", "j_fibonacci")
    b.play(*b.cards("S_A", "S_2", "S_3", "S_5", "S_8"))
    # Flush: 35 + 11+2+3+5+8 chips, x(4 + 5*8)
    assert b.total == 64 * 44


def test_fibonacci_skips_a_stone_card():
    """A Stone Card's id is negative (card.lua get_id), so it is not an 8."""
    b = Board("fibonacci_stone", "j_fibonacci")
    b.play(b.card("S_8"), b.card("H_8", enhancement="m_stone"))
    # High Card 8 plus the stone: 5 + 8 + 50 chips, x(1 + 8)
    assert b.total == 63 * 9


def test_even_steven_and_odd_todd():
    b = Board("even_steven", "j_even_steven")
    b.play(*b.cards("S_2", "S_4", "S_6", "S_8", "S_T"))
    assert b.total == 65 * (4 + 5 * 4)
    b = Board("odd_todd", "j_odd_todd")
    b.play(*b.cards("H_A", "H_3", "H_5", "H_7", "H_9"))
    assert b.total == (35 + 35 + 5 * 31) * 4


def test_scholar_walkie_talkie_and_triboulet():
    b = Board("scholar", "j_scholar")
    b.play(b.card("S_A"))
    assert b.total == (5 + 11 + 20) * (1 + 4)
    b = Board("walkie_talkie", "j_walkie_talkie")
    b.play(*b.cards("S_T", "H_T", "S_4", "H_4"))
    # Two Pair: 20 + 28 chips + 4 * 10, x(2 + 4 * 4)
    assert b.total == 88 * 18
    b = Board("triboulet", "j_triboulet")
    b.play(*b.cards("S_K", "H_K"))
    assert b.total == 30 * 8
    b = Board("triboulet_queens", "j_triboulet")
    b.play(*b.cards("S_Q", "H_Q"))
    assert b.total == 30 * 8


def test_scary_and_smiley_faces():
    b = Board("faces", "j_scary_face", "j_smiley")
    b.play(b.card("S_K"))
    assert b.total == (5 + 10 + 30) * (1 + 5)


def test_pareidolia_makes_every_card_a_face():
    b = Board("faces_pareidolia", "j_pareidolia", "j_scary_face", "j_smiley")
    b.play(b.card("S_2"))
    assert b.total == (5 + 2 + 30) * (1 + 5)


def test_faces_debuffed_by_the_plant_score_nothing():
    b = Board("faces_plant", "j_scary_face", "j_smiley", boss="bl_plant")
    king = b.card("S_K")
    b.recheck()
    assert king.debuff
    b.play(king)
    assert b.total == 5 * 1


def test_suit_jokers_with_smeared_joker():
    b = Board("suits_smeared", "j_smeared", "j_arrowhead", "j_onyx_agate")
    b.play(*b.cards("S_9", "C_9"))
    # Smeared: each card is both a Spade and a Club. 10 + 18 + 2 * 50 chips, x(2 + 2 * 7)
    assert b.total == 128 * 16


def test_arrowhead_skips_a_spade_debuffed_by_the_goad():
    b = Board("arrowhead_goad", "j_arrowhead", boss="bl_goad")
    spade, heart = b.cards("S_9", "H_9")
    b.recheck()
    assert spade.debuff
    b.play(spade, heart)
    assert b.total == (10 + 9) * 2


def test_rough_gem_pays_for_diamonds():
    b = Board("rough_gem", "j_rough_gem", "j_smeared")
    b.clear_hand()
    dollars = b.dollars
    b.play(*b.cards("D_2", "H_2"))
    assert b.dollars == dollars + 2
    assert b.run.game.last_score.dollars == 2


def test_bloodstone_rolls_only_for_hearts():
    b = Board("bloodstone", "j_bloodstone")
    b.odds(2)  # 2 in 2: every roll hits
    b.play(*b.cards("H_K", "S_K"))
    assert b.total == 30 * 3


def test_bloodstone_misses_and_smeared_diamonds():
    b = Board("bloodstone_smeared", "j_bloodstone", "j_smeared")
    b.play(*b.cards("D_K", "C_K"))
    assert b.streams().get("bloodstone") is not None


def test_ancient_joker_matches_its_suit_and_wild_cards():
    b = Board("ancient", "j_ancient")
    suit = b.run.read("current_round.ancient_card.suit")
    other = next(s for s in ("Spades", "Hearts", "Clubs", "Diamonds") if s != suit)
    b.play(b.card(front_key(suit, "9")), b.card(front_key(other, "9"), enhancement="m_wild"))
    assert b.total == 28 * 2 * 1.5 * 1.5


def test_the_idol_matches_rank_and_suit():
    b = Board("idol", "j_idol")
    idol = b.run.read("current_round.idol_card")
    other = next(s for s in ("Spades", "Hearts", "Clubs", "Diamonds") if s != idol["suit"])
    first = b.card(front_key(idol["suit"], idol["rank"]))
    wild = b.card(front_key(other, idol["rank"]), enhancement="m_wild")
    miss = b.card(front_key(other, idol["rank"]))
    nominal = first.base["nominal"]
    b.play(first, wild, miss)
    # Three of a Kind (30 x3): the idol card and the Wild Card match, the third does not.
    assert b.total == (30 + 3 * nominal) * 3 * 2 * 2


def test_the_idol_with_smeared_joker():
    b = Board("idol_smeared", "j_idol", "j_smeared")
    idol = b.run.read("current_round.idol_card")
    twin = {"Spades": "Clubs", "Clubs": "Spades", "Hearts": "Diamonds", "Diamonds": "Hearts"}
    first = b.card(front_key(twin[idol["suit"]], idol["rank"]))
    nominal = first.base["nominal"]
    b.play(first)
    assert b.total == (5 + nominal) * 2


def test_photograph_skips_a_debuffed_first_face():
    b = Board("photograph_club", "j_photograph", boss="bl_club")
    club, heart = b.cards("C_K", "H_K")
    b.recheck()
    assert club.debuff
    b.play(club, heart)
    assert b.total == 20 * 2 * 2


def test_photograph_with_pareidolia_and_blueprint():
    b = Board("photograph_bp", "j_blueprint", "j_photograph", "j_pareidolia")
    b.play(*b.cards("S_2", "H_2"))
    assert b.total == 14 * 2 * 2 * 2


def test_golden_ticket_pays_for_gold_cards():
    b = Board("golden_ticket", "j_blueprint", "j_ticket")
    b.clear_hand()
    dollars = b.dollars
    b.play(b.card("S_K", enhancement="m_gold", seal="Red"))
    # Two evaluations (Red Seal), each paid by Golden Ticket and its Blueprint copy.
    assert b.dollars == dollars + 16


def test_golden_ticket_skips_a_gold_card_debuffed_by_the_club():
    b = Board("ticket_club", "j_ticket", boss="bl_club")
    b.clear_hand()
    club = b.card("C_K", enhancement="m_gold")
    heart = b.card("H_K", enhancement="m_gold")
    b.recheck()
    dollars = b.dollars
    b.play(club, heart)
    assert b.dollars == dollars + 4


def test_business_card_pays_for_faces():
    b = Board("business", "j_business")
    b.odds(2)
    dollars = b.dollars
    b.play(b.card("S_K"))
    assert b.dollars == dollars + 2


def test_business_card_skips_faces_debuffed_by_the_plant():
    b = Board("business_plant", "j_business", boss="bl_plant")
    b.odds(2)
    king = b.card("S_K")
    b.recheck()
    dollars = b.dollars
    b.play(king)
    assert b.dollars == dollars
    assert b.streams().get("business") is None


# ---------------------------------------------------------------------------
# Scored cards: scaling and creation
# ---------------------------------------------------------------------------


def test_hiker_with_a_red_seal():
    b = Board("hiker", "j_hiker")
    card = b.card("S_9", seal="Red")
    b.play(card)
    # The second evaluation already sees the first +5.
    assert b.total == 5 + 9 + 14
    assert card.ability["perma_bonus"] == 10


def test_hiker_is_copied_by_blueprint_and_brainstorm():
    b = Board("hiker_copies", "j_blueprint", "j_hiker", "j_brainstorm")
    card = b.card("S_9")
    b.play(card)
    assert card.ability["perma_bonus"] == 15


def test_lucky_cat_grows_on_lucky_triggers():
    b = Board("lucky_cat", "j_blueprint", "j_lucky_cat")
    b.odds(15)  # both Lucky Card rolls always hit
    cat = b.jokers[1]
    dollars = b.dollars
    b.play(b.card("H_5", enhancement="m_lucky", seal="Red"))
    # Two triggers, the Blueprint copy is guarded: X1.5, scored twice (shared x_mult rule).
    assert cat.ability["x_mult"] == 1.5
    assert b.total == int(15 * 41 * 1.5 * 1.5)
    assert b.dollars == dollars + 40


def test_lucky_cat_and_a_lucky_card_debuffed_by_the_head():
    b = Board("lucky_cat_head", "j_lucky_cat", boss="bl_head")
    b.odds(15)
    lucky = b.card("H_5", enhancement="m_lucky")
    b.recheck()
    b.play(lucky, b.card("S_5", enhancement="m_lucky"))
    # Only the Spade scores and triggers.
    assert b.jokers[0].ability["x_mult"] == 1.25


def test_wee_joker_counts_retriggered_twos():
    b = Board("wee_hack", "j_hack", "j_wee")
    b.play(*b.cards("S_2", "H_2"))
    assert b.jokers[1].ability["extra"]["chips"] == 32
    assert b.total == (10 + 4 * 2 + 32) * 2


def test_wee_joker_grows_on_twos():
    b = Board("wee", "j_blueprint", "j_wee")
    b.play(*b.cards("S_2", "H_2"))
    assert b.jokers[1].ability["extra"]["chips"] == 16
    # The copy doesn't grow it but scores its chips too.
    assert b.total == (14 + 2 * 16) * 2


def test_eight_ball_makes_a_tarot():
    b = Board("eight_ball", "j_8_ball")
    b.odds(4)
    b.play(b.card("S_8"))
    assert len(b.run.cards("consumables")) == 1


def test_eight_ball_does_not_roll_with_full_slots():
    """G22: no room (cards plus the buffer), no roll."""
    b = Board("eight_ball_full", "j_8_ball")
    b.consumable("c_pluto")
    b.consumable("c_pluto")
    b.play(b.card("S_8"))
    assert b.streams().get("8ball") is None
    assert len(b.run.cards("consumables")) == 2


def test_eight_ball_buffer_reserves_the_last_slot():
    b = Board("eight_ball_buffer", "j_8_ball")
    b.odds(4)
    b.consumable("c_pluto")
    b.play(*b.cards("S_8", "H_8"))
    assert len(b.run.cards("consumables")) == 2


def test_eight_ball_copies_and_red_seal():
    b = Board("eight_ball_copies", "j_blueprint", "j_8_ball")
    b.odds(4)
    b.play(b.card("S_8", seal="Red"))
    # The copy and 8 Ball fill both slots on the first evaluation; the Red Seal's finds none.
    assert len(b.run.cards("consumables")) == 2


# ---------------------------------------------------------------------------
# Held cards (card.lua:3271-3341)
# ---------------------------------------------------------------------------


def test_shoot_the_moon_and_baron():
    b = Board("moon_baron", "j_shoot_the_moon", "j_baron")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.cards("S_Q", "H_Q", "S_K")
    b.play(*pair)
    assert b.total == 14 * (2 + 13 + 13) * 1.5


def test_baron_with_a_red_seal_king():
    b = Board("baron_red", "j_baron")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.card("S_K", seal="Red")
    b.play(*pair)
    assert b.total == 14 * 2 * 1.5 * 1.5


def test_held_jokers_meet_cards_debuffed_by_the_plant():
    b = Board("held_plant", "j_shoot_the_moon", "j_baron", "j_mime", boss="bl_plant")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    queen = b.card("S_Q")
    king = b.card("S_K", seal="Red")
    b.recheck()
    assert queen.debuff and king.debuff
    b.play(*pair)
    assert b.total == 14 * 2


def test_reserved_parking_pays_for_held_faces():
    b = Board("parking", "j_blueprint", "j_reserved_parking")
    b.odds(2)
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.cards("S_K", "H_Q")
    dollars = b.dollars
    b.play(*pair)
    assert b.dollars == dollars + 4


def test_reserved_parking_with_pareidolia():
    b = Board("parking_pareidolia", "j_reserved_parking", "j_pareidolia")
    b.odds(2)
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.card("S_3")
    dollars = b.dollars
    b.play(*pair)
    assert b.dollars == dollars + 1


def test_reserved_parking_skips_faces_debuffed_by_the_plant():
    """A debuffed card is no face, so it never rolls: the debuffed arm is unreachable."""
    b = Board("parking_plant", "j_reserved_parking", boss="bl_plant")
    b.odds(2)
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    king = b.card("S_K")
    b.recheck()
    assert king.debuff
    dollars = b.dollars
    b.play(*pair)
    assert b.dollars == dollars
    assert b.streams().get("parking") is None


def test_the_plant_with_pareidolia_debuffs_every_card():
    """``is_face(true)`` is true for every card with Pareidolia (card.lua:964-970)."""
    b = Board("plant_pareidolia", "j_pareidolia", "j_baron", "j_mime", boss="bl_plant")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    king = b.card("S_K", enhancement="m_steel")
    b.recheck()
    assert all(c.debuff for c in (*pair, king))
    b.play(*pair)
    assert b.total == 10 * 2


def test_red_seals_repeat_held_joker_effects():
    """A held card with an effect is repeated by its Red Seal (state_events.lua:812-818)."""
    b = Board("held_red", "j_shoot_the_moon", "j_raised_fist", "j_reserved_parking")
    b.odds(2)
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.card("S_Q", seal="Red")
    b.card("H_3", seal="Red")
    dollars = b.dollars
    b.play(*pair)
    # The Queen: +13 and $1, twice. The 3 (Raised Fist): +6, twice.
    assert b.total == 14 * (2 + 26 + 12)
    assert b.dollars == dollars + 2


def test_raised_fist_takes_the_last_lowest_card():
    b = Board("raised_fist", "j_raised_fist")
    b.clear_hand()
    king = b.card("S_K")
    b.cards("S_7", "H_3", "D_3", "C_9")
    b.play(king)
    assert b.total == 15 * (1 + 2 * 3)


def test_raised_fist_lowest_card_debuffed_by_the_club():
    b = Board("raised_fist_club", "j_raised_fist", boss="bl_club")
    b.clear_hand()
    king = b.card("S_K")
    three = b.card("H_3")
    club = b.card("C_3")
    b.recheck()
    assert club.debuff and not three.debuff
    b.play(king)
    assert b.total == 15


def test_raised_fist_skips_stone_cards_and_counts_aces_high():
    b = Board("raised_fist_stone", "j_blueprint", "j_raised_fist")
    b.clear_hand()
    king = b.card("S_K")
    b.card("S_2", enhancement="m_stone")
    b.card("H_A")
    b.play(king)
    assert b.total == 15 * (1 + 2 * 22)


# ---------------------------------------------------------------------------
# Repetitions (card.lua:3342-3395)
# ---------------------------------------------------------------------------


def test_sock_and_buskin_with_a_red_seal():
    b = Board("sock", "j_sock_and_buskin")
    b.play(b.card("S_K", seal="Red"), b.card("H_K"))
    assert b.total == (10 + 3 * 10 + 2 * 10) * 2


def test_sock_and_buskin_with_pareidolia_and_brainstorm():
    b = Board("sock_copies", "j_sock_and_buskin", "j_pareidolia", "j_brainstorm")
    b.play(*b.cards("S_2", "H_2"))
    assert b.total == (10 + 3 * 2 + 3 * 2) * 2


def test_sock_and_buskin_and_faces_debuffed_by_the_plant():
    b = Board("sock_plant", "j_sock_and_buskin", boss="bl_plant")
    cards = b.cards("S_K", "H_K")
    b.recheck()
    b.play(*cards)
    assert b.total == 10 * 2


def test_hanging_chad_retriggers_a_leading_stone():
    b = Board("chad_stone", "j_hanging_chad")
    b.play(b.card("S_4", enhancement="m_stone"), b.card("S_K"), b.card("H_K"))
    # Pair of Kings; the stone joins the scoring hand and sorts first by position.
    assert b.total == (10 + 3 * 50 + 20) * 2


def test_hanging_chad_and_a_debuffed_first_card():
    b = Board("chad_club", "j_blueprint", "j_hanging_chad", boss="bl_club")
    club, heart = b.cards("C_K", "H_K")
    b.recheck()
    b.play(club, heart)
    assert b.total == 20 * 2


def test_hanging_chad_with_splash():
    b = Board("chad_splash", "j_splash", "j_hanging_chad")
    b.play(*b.cards("S_3", "H_K", "D_9"))
    # Splash: every played card scores, the 3 first (three evaluations).
    assert b.total == 5 + 3 * 3 + 10 + 9


def test_repetitions_stack_on_the_first_face():
    b = Board("repetitions", "j_sock_and_buskin", "j_hanging_chad", "j_dusk")
    b.run.op({"op": "set", "hands_left": 1})
    b.play(b.card("S_K", seal="Red"), b.card("H_K"))
    # The first King: once, Red Seal, Sock and Buskin, Hanging Chad twice and Dusk. The second:
    # once, Sock and Buskin and Dusk.
    assert b.total == (10 + 6 * 10 + 3 * 10) * 2


def test_dusk_retriggers_the_last_hand():
    b = Board("dusk", "j_blueprint", "j_dusk")
    b.run.op({"op": "set", "hands_left": 1})
    b.play(b.card("S_K"))
    assert b.total == 5 + 3 * 10


def test_dusk_waits_for_the_last_hand():
    b = Board("dusk_early", "j_dusk")
    b.run.op({"op": "set", "hands_left": 2})
    b.play(b.card("S_K"))
    assert b.total == 5 + 10


def test_seltzer_counts_down_and_is_drunk():
    b = Board("seltzer", "j_blueprint", "j_selzer")
    seltzer = b.jokers[1]
    with b.run.edit_card(seltzer) as card:
        card["ability"]["extra"] = 2
    b.play(b.card("S_K"))
    assert b.total == 5 + 3 * 10
    assert seltzer.ability["extra"] == 1
    b.play(b.card("H_K"))
    assert seltzer not in b.run.cards("jokers")


def test_seltzer_counts_a_hand_debuffed_by_the_psychic():
    b = Board("seltzer_psychic", "j_selzer", boss="bl_psychic")
    b.play(b.card("S_K"))
    assert b.run.game.last_score.debuffed
    assert b.jokers[0].ability["extra"] == 9


def test_mime_retriggers_held_cards_with_an_effect():
    """G20: the plain held card has no effect, so Mime leaves it alone."""
    b = Board("mime", "j_mime")
    b.clear_hand()
    pair = b.cards("S_9", "H_9")
    b.card("S_K", enhancement="m_steel")
    b.card("C_5")
    b.play(*pair)
    assert b.total == 28 * 2 * 1.5 * 1.5


def test_mime_repeats_only_the_parking_rolls_that_paid():
    """G20: a held face whose Reserved Parking roll misses has no effect, so Mime doesn't
    repeat it; one that paid is repeated and rolls again. Six held faces roll at least six
    times; the oracle's verdict, which an RNG twin confirmed roll by roll, pins the outcome."""
    b = Board("mime_parking", "j_mime", "j_reserved_parking")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.cards("S_K", "H_K", "D_K", "C_K", "S_Q", "H_Q")
    dollars = b.dollars
    b.play(*pair)
    paid = b.dollars - dollars
    assert 0 < paid < 12  # some rolls paid (each repeated once), not all
    assert "parking" in b.streams()


def test_mime_red_seal_and_copies_on_a_steel_card():
    b = Board("mime_copies", "j_blueprint", "j_mime", "j_brainstorm")
    b.clear_hand()
    pair = b.cards("S_9", "H_9")
    b.card("S_K", enhancement="m_steel", seal="Red")
    b.play(*pair)
    # Once, then the Red Seal, Blueprint's Mime, Mime and Brainstorm's Mime.
    assert b.total == int(28 * 2 * 1.5**5)


def test_mime_retriggers_joker_effects_on_held_cards():
    b = Board("mime_jokers", "j_mime", "j_shoot_the_moon", "j_raised_fist")
    b.clear_hand()
    pair = b.cards("S_9", "H_9")
    b.cards("S_Q", "H_3")
    b.play(*pair)
    # The Queen: +13 twice. The 3 (Raised Fist): +6 twice.
    assert b.total == 28 * (2 + 26 + 12)


def test_mime_at_round_end_repeats_gold_cards():
    b = Board("mime_gold", "j_mime")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.card("S_K", enhancement="m_gold")
    b.card("H_K", enhancement="m_gold", seal="Red")
    b.card("C_5")
    b.to_win()
    dollars = b.dollars
    b.play(*pair)
    assert b.run.game.phase == "round_eval"
    assert b.dollars == dollars + 2 * 3 + 3 * 3


def test_copied_mime_at_round_end():
    """The end-of-round repetition has no Blueprint guard (card.lua:2877-2887)."""
    b = Board("mime_gold_copies", "j_blueprint", "j_mime", "j_brainstorm")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.card("S_K", enhancement="m_gold")
    b.to_win()
    dollars = b.dollars
    b.play(*pair)
    assert b.dollars == dollars + 4 * 3


def test_mime_skips_a_gold_card_debuffed_by_the_club():
    """A debuffed card has no end-of-round effect (card.lua:1033-1036): nothing to repeat."""
    b = Board("mime_gold_club", "j_mime", boss="bl_club")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    club = b.card("C_K", enhancement="m_gold", seal="Red")
    b.card("H_K", enhancement="m_gold")
    b.recheck()
    assert club.debuff
    b.to_win()
    dollars = b.dollars
    b.play(*pair)
    assert b.run.game.phase == "round_eval"
    assert b.dollars == dollars + 2 * 3


def test_mime_and_blue_seals_without_room():
    """A Blue Seal without room makes nothing and has no effect, so Mime doesn't repeat it."""
    b = Board("mime_blue_full", "j_mime")
    b.consumable("c_pluto")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.card("S_K", seal="Blue")
    b.card("H_K", seal="Blue")
    b.to_win()
    b.play(*pair)
    # The first Blue Seal takes the last slot; Mime's repeat and the second card find none.
    assert b.run.keys("consumables") == ["c_pluto", "c_mercury"]


def test_mime_at_round_end_repeats_blue_seals():
    b = Board("mime_blue", "j_mime")
    b.clear_hand()
    pair = b.cards("S_2", "H_2")
    b.card("S_K", seal="Blue")
    b.to_win()
    b.play(*pair)
    assert b.run.keys("consumables") == ["c_mercury", "c_mercury"]


# ---------------------------------------------------------------------------
# Joker on joker (card.lua:3396-3408)
# ---------------------------------------------------------------------------


def test_baseball_card_and_uncommon_jokers():
    b = Board("baseball", "j_baseball", "j_mime", "j_hiker", "j_scholar")
    b.consumable("c_pluto")
    b.play(b.card("S_9"))
    assert b.total == int(14 * 1.5 * 1.5)


def test_baseball_card_counts_a_debuffed_uncommon_and_is_copied():
    b = Board("baseball_copies", "j_blueprint", "j_baseball", "j_mime")
    b.run.oracle("set_debuff(True) on Mime")
    b.play(b.card("S_9"))
    assert b.jokers[2].debuff
    assert b.total == int(14 * 1.5 * 1.5)


# ---------------------------------------------------------------------------
# G80: the boss's re-check sees the cards in play
# ---------------------------------------------------------------------------


def test_g80_pillar_recheck_after_seltzer():
    b = Board("g80_seltzer", "j_selzer", boss="bl_pillar")
    with b.run.edit_card(b.jokers[0]) as card:
        card["ability"]["extra"] = 1
    cards = b.cards("S_9", "H_9")
    b.play(*cards)
    assert all(c.debuff for c in cards)


def test_g80_pillar_recheck_after_eight_ball():
    b = Board("g80_8ball", "j_8_ball", boss="bl_pillar")
    b.odds(4)
    cards = b.cards("S_8", "H_8")
    b.play(*cards)
    assert len(b.run.cards("consumables")) == 2
    assert all(c.debuff for c in cards)
