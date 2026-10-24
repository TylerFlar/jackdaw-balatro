"""Targeted parity cases for joker batch B (main step, ``before`` and ``after``).

The seeded scenarios (``test_joker_scenarios.py``) rarely reach some branches: a Straight Flush
for Seance, a repeated hand for Card Sharp, a full board of Stencils, a melting Ice Cream. Each
case builds a run (BalatroBot ``add`` for the jokers and the playing cards, which join the
hand), writes the few values play would take long to reach straight into the state, and takes
its steps; each step must match the oracle's (:mod:`tests.parity.cases`). The assertions after
a step confirm that the branch under test was taken.
"""

from __future__ import annotations

from typing import Any

from jackdaw.core import Score
from tests.parity.cases import CardRef, Run

# Jokers with no effect while a hand scores (dollar bonus, interest, bankruptcy): board filler.
FILLER = ("j_golden", "j_credit_card", "j_to_the_moon", "j_cloud_9")


class Board:
    """A targeted run whose steps are checked against the oracle's one at a time."""

    def __init__(self, seed: str, *jokers: str, boss: str | None = None, edition: str = "foil"):
        setup: list[dict[str, Any]] = []
        if boss is not None:
            setup.append({"op": "force_boss", "key": boss})
        setup.append({"op": "action", "action": ["SelectBlind"]})
        # A fixed edition keeps the board's slots and mult predictable (foil adds chips only).
        setup += [{"op": "add", "key": key, "edition": edition} for key in jokers]
        start = {"back": "b_red", "stake": 1, "seed": seed, "challenge": None}
        self.seed = seed
        self.run = Run(start, setup)
        # Out of reach: a hand ends the round only when a case wins it on purpose.
        self.run.write("blind.chips", 10**9)

    def add(self, key: str, **params: Any) -> CardRef:
        """BalatroBot ``add``: a joker, a consumable, or a playing card into the hand. A playing
        card meets the boss as a card dealt this round did at set_blind (blind.lua:207-210)."""
        card = self.run.add(key, **params)
        if card.area == "hand":
            self.run.boss_check(card)
        return card

    def cards(self, *keys: str, **params: Any) -> list[CardRef]:
        return [self.add(k, **params) for k in keys]

    def joker(self, key: str, n: int = 0) -> CardRef:
        return [j for j in self.run.cards("jokers") if j.key == key][n]

    def has(self, key: str) -> bool:
        return key in self.run.keys("jokers")

    def hold_only(self, *keys: str, **params: Any) -> list[CardRef]:
        """Return the dealt hand to the deck and hold exactly these cards."""
        self.run.move("hand", "deck")
        return self.cards(*keys, **params)

    def step(self, action: list[Any]) -> None:
        assert self.run.step(action), f"{self.seed}: the oracle refused {action}"

    def play(self, *cards: CardRef) -> None:
        self.step(["PlayHand", self.run.indices(cards)])

    def discard(self, *cards: CardRef) -> None:
        self.step(["Discard", self.run.indices(cards)])

    def win_with(self, *cards: CardRef) -> None:
        """Play the cards with the blind one chip from beaten."""
        self.run.write("game.chips", self.run.read("blind.chips") - 1)
        self.play(*cards)
        assert self.run.game.phase == "round_eval"

    @property
    def last(self) -> Score:
        score = self.run.game.last_score
        assert score is not None
        return score

    def level(self, hand: str) -> float:
        return self.run.game.hand_levels[hand].level


# ---------------------------------------------------------------------------
# Main step only
# ---------------------------------------------------------------------------


def test_half_joker_counts_the_played_cards() -> None:
    """Half Joker: +20 Mult with at most 3 cards played (card.lua:3672-3677); Blueprint copies
    it."""
    b = Board("BB_HALF", "j_blueprint", "j_half")
    b.play(*b.cards("S_2", "H_2", "D_9"))
    assert b.last.mult == 2 + 20 + 20
    b.play(*b.cards("S_3", "H_3", "D_4", "C_6"))
    assert b.last.mult == 2


def test_two_stencils_score_on_a_full_board() -> None:
    """G58: x_mult = empty slots + Stencils = 0 + 2, and the shared x_mult > 1 rule scores it
    (card.lua:3653-3659, 4203-4208), however full the board."""
    b = Board("BB_STENCIL_FULL", "j_stencil", "j_stencil", *FILLER[:3])
    assert len(b.run.cards("jokers")) == b.run.read("game.joker_slots") == 5
    b.play(*b.cards("S_5", "H_5"))
    assert b.last.mult == 2 * 2 * 2


def test_one_stencil_on_a_full_board_gives_nothing() -> None:
    """One Stencil, no empty slot: x_mult = 1, and its own branch needs an empty slot
    (card.lua:3966-3972)."""
    b = Board("BB_STENCIL_ONE", "j_stencil", *FILLER)
    b.play(*b.cards("S_5", "H_5"))
    assert b.last.mult == 2


def test_stencil_with_empty_slots_and_its_copies() -> None:
    """Stencil leftmost, Blueprint and Brainstorm copy it: three jokers, two empty slots, X3
    each."""
    b = Board("BB_STENCIL_COPY", "j_stencil", "j_blueprint", "j_brainstorm")
    b.play(*b.cards("S_5", "H_5"))
    assert b.last.mult == 2 * 3 * 3 * 3


def test_banner_and_mystic_summit_read_the_discards_left() -> None:
    """Banner: +30 Chips per discard left (card.lua:3707-3712); Mystic Summit: +15 Mult with none
    left (3694-3699); Brainstorm copies the leftmost."""
    b = Board("BB_DISCARDS", "j_banner", "j_mystic_summit", "j_brainstorm")
    played = b.cards("S_5", "H_5")
    b.play(*played)
    assert b.last.mult == 2
    b.run.write("current_round.discards_left", 0)
    b.play(*b.cards("S_6", "H_6"))
    assert b.last.mult == 2 + 15


def test_misprint_rolls_for_each_copy() -> None:
    """Misprint draws pseudorandom('misprint', 0, 23) per call (card.lua:3700-3706)."""
    b = Board("BB_MISPRINT", "j_misprint", "j_blueprint", "j_misprint")
    b.play(*b.cards("S_5", "H_5"))
    b.play(*b.cards("S_7", "H_7"))


def test_steel_and_stone_jokers_count_the_deck() -> None:
    """Steel Joker: X(1 + 0.2 per Steel Card); Stone Joker: +25 Chips per Stone Card; both from
    the per-frame tallies over G.playing_cards (card.lua:3922-3935, 4185-4202)."""
    b = Board("BB_STEEL_STONE", "j_steel_joker", "j_stone", "j_brainstorm")
    b.cards("S_K", "D_4", enhancement="m_steel")
    b.cards("C_2", "H_3", enhancement="m_stone")
    b.play(*b.cards("S_5", "H_5"))
    b.play(*b.cards("S_9", "H_9"))


def test_abstract_and_supernova() -> None:
    """Abstract Joker: +3 per Joker-set card in G.jokers (card.lua:3678-3687); Supernova: +Mult
    equal to the scored hand's play count, this play included (3731-3736)."""
    b = Board("BB_ABSTRACT_NOVA", "j_abstract", "j_blueprint", "j_supernova")
    b.play(*b.cards("S_5", "H_5"))
    b.play(*b.cards("S_8", "H_8"))
    assert b.run.game.hand_levels["Pair"].played == 2


def test_blue_joker_with_an_empty_deck() -> None:
    """Blue Joker: +2 Chips per card in the deck, nothing once it is empty (card.lua:3887-3893)."""
    b = Board("BB_BLUE", "j_blueprint", "j_blue_joker")
    b.play(*b.cards("S_5", "H_5"))
    b.run.move("deck", "discard")
    b.play(*b.cards("S_6", "H_6"))


def test_card_sharp_on_a_repeated_hand() -> None:
    """Card Sharp: X3 once the scored hand was played earlier this round (card.lua:4040-4045)."""
    b = Board("BB_SHARP", "j_blueprint", "j_card_sharp")
    b.play(*b.cards("S_5", "H_5"))
    assert b.last.mult == 2
    b.play(*b.cards("S_6", "H_6"))
    assert b.last.mult == 2 * 3 * 3


def test_erosion_below_the_starting_deck() -> None:
    """Erosion: +4 Mult per card below the starting deck size (card.lua:3894-3900)."""
    b = Board("BB_EROSION", "j_erosion", "j_brainstorm")
    b.play(*b.cards("S_5", "H_5"))  # 52 + 2 added cards: none missing
    assert b.last.mult == 2
    with b.run.edit() as snap:
        del snap["areas"]["deck"][:8]
    b.play(*b.cards("S_6", "H_6"))  # 54 - 8 + 2 = 48 cards: 4 missing
    assert b.last.mult == 2 + 16 + 16


def test_bull_and_bootstraps_see_the_dollar_buffer() -> None:
    """Bull and Bootstraps read G.GAME.dollars + G.GAME.dollar_buffer, which a Gold Seal scored
    earlier in the hand feeds (card.lua:1084-1087, 3936-3942, 4046-4051)."""
    b = Board("BB_BULL", "j_bull", "j_blueprint", "j_bootstraps")
    b.run.write("game.dollars", 4)
    b.play(*b.cards("S_5", "H_5", seal="Gold"))
    b.run.write("game.dollars", 0)
    b.play(*b.cards("S_6", "H_6"))
    b.run.write("game.dollars", -3)
    b.play(*b.cards("S_7", "H_7"))


def test_acrobat_on_the_last_hand() -> None:
    """Acrobat: X3 when no hand is left (card.lua:3688-3693)."""
    b = Board("BB_ACROBAT", "j_blueprint", "j_acrobat")
    b.play(*b.cards("S_5", "H_5"))
    assert b.last.mult == 2
    b.run.write("current_round.hands_left", 1)
    b.win_with(*b.cards("S_6", "H_6"))
    assert b.last.mult == 2 * 3 * 3


def test_acrobat_on_a_lost_last_hand() -> None:
    """The last hand fails the blind: the round ends in a game over."""
    b = Board("BB_ACROBAT_LOSS", "j_acrobat")
    b.run.write("current_round.hands_left", 1)
    b.play(*b.cards("S_6", "H_6"))
    assert b.run.game.phase == "game_over"


def test_swashbuckler_sums_the_other_jokers() -> None:
    """Swashbuckler: +Mult = the other jokers' sell value, debuffed ones included
    (card.lua:3974-3979, 4240-4248)."""
    b = Board("BB_SWASH", "j_blueprint", "j_swashbuckler", "j_golden", "j_credit_card")
    with b.run.edit_card(b.joker("j_credit_card")) as card:
        card["debuff"] = True
    b.play(*b.cards("S_5", "H_5"))


def test_stuntman_hand_size_follows_its_debuff() -> None:
    """Stuntman: +250 Chips (card.lua:3713-3718); Crimson Heart's debuff toggles its -2 hand
    size through remove_from_deck / add_to_deck (card.lua:627-629, 684-686)."""
    b = Board("BB_STUNTMAN", "j_stuntman", "j_golden", boss="bl_final_heart")
    for _ in range(3):
        b.play(*b.cards("S_5", "H_5"))


def test_drivers_license_needs_sixteen_enhanced_cards() -> None:
    """Driver's License: X3 with at least 16 enhanced playing cards (card.lua:3943-3950,
    4179-4184)."""
    b = Board("BB_DRIVER", "j_blueprint", "j_drivers_license")
    b.run.oracle("set_ability('m_bonus') on the first 15 deck cards")
    b.play(*b.cards("S_5", "H_5"))
    assert b.last.mult == 2
    b.run.oracle("set_ability('m_mult') on deck card 20")
    b.play(*b.cards("S_6", "H_6"))
    assert b.last.mult == 2 * 3 * 3


def test_blackboard_holds_only_black_suits() -> None:
    """Blackboard: X3 when every held card is a Spade or a Club, with is_suit's flush rules
    (card.lua:3951-3965): a Wild Card counts, a Stone Card does not."""
    b = Board("BB_BLACKBOARD", "j_blueprint", "j_blackboard")
    played = b.hold_only("H_5", "D_5")
    b.cards("S_2", "C_9")
    b.add("H_3", enhancement="m_wild")
    b.play(*played)
    assert b.last.mult == 2 * 3 * 3
    played = b.hold_only("H_6", "D_6")
    b.cards("S_2", "C_9")
    b.add("C_3", enhancement="m_stone")
    b.play(*played)
    assert b.last.mult == 2


def test_blackboard_under_the_goad() -> None:
    """With flush_calc a debuffed Spade is still a Spade, but a debuffed Wild Card is only its
    own suit (card.lua:4064-4075): a held Wild Heart under The Goad breaks Blackboard."""
    b = Board("BB_BLACKBOARD_GOAD", "j_blackboard", boss="bl_goad")
    played = b.hold_only("H_5", "D_5")
    spade = b.add("S_2")
    b.add("C_9")
    assert spade.debuff
    b.play(*played)
    assert b.last.mult == 2 * 3
    played = b.hold_only("H_6", "D_6")
    b.add("S_3")
    wild = b.add("H_4", enhancement="m_wild")
    assert wild.debuff
    b.play(*played)
    assert b.last.mult == 2


def test_blackboard_with_an_empty_hand() -> None:
    """No held card: every held card is black, so X3."""
    b = Board("BB_BLACKBOARD_EMPTY", "j_blackboard")
    b.play(*b.hold_only("H_5", "D_5"))
    assert b.last.mult == 2 * 3


def test_flower_pot_needs_four_suits() -> None:
    """Flower Pot (card.lua:3807-3839): each non-Wild card fills the first empty suit it
    belongs to, then each Wild Card fills one more; Blueprint copies it."""
    b = Board("BB_FLOWER", "j_blueprint", "j_flower_pot")
    b.play(*b.cards("H_2", "D_3", "S_4", "D_5", "H_6"))  # no Club
    assert b.last.mult == 4
    straight = b.cards("H_7", "D_8", "S_9")
    straight.append(b.add("H_T", enhancement="m_wild"))
    straight.append(b.add("D_J"))
    b.play(*straight)  # Hearts, Diamonds, Spades, then the Wild Card fills Clubs
    assert b.last.mult == 4 * 3 * 3


def test_flower_pot_under_the_goad() -> None:
    """The non-Wild pass reads suits with bypass_debuff, so a debuffed Spade still counts; a
    debuffed Wild Card (The Goad debuffs it too) fills nothing."""
    b = Board("BB_FLOWER_GOAD", "j_flower_pot", boss="bl_goad")
    b.play(*b.cards("H_2", "D_3", "S_4", "C_5", "H_6"))
    assert b.last.mult == 4 * 3
    straight = b.cards("H_7", "D_8", "S_9")
    straight.append(b.add("H_T", enhancement="m_wild"))
    straight.append(b.add("D_J"))
    assert straight[3].debuff
    b.play(*straight)
    assert b.last.mult == 4


def test_seeing_double_needs_a_club_and_another_suit() -> None:
    """Seeing Double (card.lua:3840-3872): a Club and another suit; a Wild Card fills Clubs
    first."""
    b = Board("BB_DOUBLE", "j_blueprint", "j_seeing_double")
    b.play(*b.cards("D_7", "H_7"))
    assert b.last.mult == 2
    b.play(*b.cards("C_7", "S_7"))
    assert b.last.mult == 2 * 2 * 2
    pair = [b.add("H_8", enhancement="m_wild"), b.add("D_8")]
    b.play(*pair)  # Diamonds, then the Wild Card fills Clubs
    assert b.last.mult == 2 * 2 * 2


def test_seeing_double_under_the_club() -> None:
    """Without bypass_debuff a debuffed card belongs to no suit: The Club's Clubs, and a Wild
    Card it debuffs, give no Club."""
    b = Board("BB_DOUBLE_CLUB", "j_seeing_double", boss="bl_club")
    b.play(*b.cards("C_7", "H_7"))
    assert b.last.mult == 2
    pair = [b.add("H_8", enhancement="m_wild"), b.add("D_8")]
    b.play(*pair)
    assert b.last.mult == 2


# ---------------------------------------------------------------------------
# Consumable creators (main step)
# ---------------------------------------------------------------------------


def test_superposition_needs_an_ace_and_a_straight() -> None:
    """Superposition: a Tarot for an Ace in a hand containing a Straight (card.lua:3762-3786);
    its Blueprint copy makes one more while there is room."""
    b = Board("BB_SUPERPOSITION", "j_blueprint", "j_superposition")
    b.play(*b.cards("H_A", "D_2", "S_3", "C_4", "H_5"))
    assert len(b.run.cards("consumables")) == 2
    b.run.write("areas.consumables", [])
    b.play(*b.cards("H_6", "D_7", "S_8", "C_9", "H_T"))  # no Ace
    b.play(*b.cards("H_A", "D_A"))  # no Straight
    assert not b.run.cards("consumables")


def test_seance_on_a_straight_flush() -> None:
    """Seance: a Spectral for a hand containing a Straight Flush (card.lua:3787-3806)."""
    b = Board("BB_SEANCE", "j_seance", "j_brainstorm")
    b.play(*b.cards("H_5", "H_6", "H_7", "H_8", "H_9"))
    assert [c.set for c in b.run.cards("consumables")] == ["Spectral", "Spectral"]
    b.play(*b.cards("S_5", "S_6", "S_7", "S_8", "H_T"))  # a Straight, no flush
    assert len(b.run.cards("consumables")) == 2


def test_vagabond_at_four_dollars_or_less() -> None:
    """Vagabond: a Tarot when the hand is played with $4 or less (card.lua:3743-3761). The copies
    share G.GAME.consumeable_buffer: with one free slot only the first makes a card."""
    b = Board("BB_VAGABOND", "j_blueprint", "j_vagabond")
    b.run.write("game.dollars", 4)
    b.add("c_pluto")
    b.play(*b.cards("S_5", "H_5"))
    assert len(b.run.cards("consumables")) == 2
    b.run.write("areas.consumables", [])
    b.run.write("game.dollars", 5)
    b.play(*b.cards("S_6", "H_6"))
    assert not b.run.cards("consumables")
    b.run.write("game.dollars", 0)
    b.play(*b.cards("S_7", "H_7"))
    assert len(b.run.cards("consumables")) == 2


def test_superposition_counts_a_debuffed_ace() -> None:
    """get_id ignores the debuff (card.lua:957-962): The Goad's debuffed Ace of Spades still
    counts."""
    b = Board("BB_SUPER_GOAD", "j_superposition", boss="bl_goad")
    straight = b.cards("S_A", "D_2", "H_3", "C_4", "D_5")
    assert straight[0].debuff
    b.play(*straight)
    assert len(b.run.cards("consumables")) == 1


def test_vagabond_ignores_the_dollar_buffer() -> None:
    """Vagabond reads G.GAME.dollars alone (card.lua:3744): the Gold Seals scored earlier in the
    hand do not lift it over $4."""
    b = Board("BB_VAGABOND_BUFFER", "j_vagabond")
    b.run.write("game.dollars", 4)
    b.play(*b.cards("S_5", "H_5", seal="Gold"))
    assert len(b.run.cards("consumables")) == 1


def test_creators_need_a_free_slot() -> None:
    """With the consumable slots full no creator makes a card (or draws a pool roll)."""
    b = Board("BB_FULL_SLOTS", "j_superposition", "j_seance", "j_vagabond")
    b.add("c_pluto")
    b.add("c_mercury")
    b.run.write("game.dollars", 0)
    b.play(*b.cards("H_A", "H_2", "H_3", "H_4", "H_5"))
    assert b.run.keys("consumables") == ["c_pluto", "c_mercury"]


# ---------------------------------------------------------------------------
# Scaling: before / after / discard / end of round, then the main step
# ---------------------------------------------------------------------------


def test_ride_the_bus_grows_and_resets() -> None:
    """Ride the Bus: +1 per hand without a scoring face card, back to 0 with one
    (card.lua:3525-3542, 3992-3997)."""
    b = Board("BB_BUS", "j_blueprint", "j_ride_the_bus")
    bus = b.joker("j_ride_the_bus")
    b.play(*b.cards("S_5", "H_5"))
    b.play(*b.cards("S_6", "H_6"))
    assert bus.ability["mult"] == 2
    b.play(*b.cards("S_K", "H_K"))
    assert bus.ability["mult"] == 0
    b.play(*b.cards("S_Q", "H_Q"))
    assert bus.ability["mult"] == 0


def test_ride_the_bus_with_pareidolia() -> None:
    """With Pareidolia every card is a face card: the bus never grows."""
    b = Board("BB_BUS_PAREIDOLIA", "j_pareidolia", "j_ride_the_bus")
    b.play(*b.cards("S_5", "H_5"))
    assert b.joker("j_ride_the_bus").ability["mult"] == 0


def test_space_joker_levels_up_the_hand() -> None:
    """Space Joker: a 1 in 4 chance to level up the played hand (card.lua:3420-3426), copies
    included; two Oops! All 6s make it certain."""
    b = Board("BB_SPACE", "j_oops", "j_oops", "j_blueprint", "j_space")
    b.play(*b.cards("S_5", "H_5"))
    assert b.level("Pair") == 3


def test_square_joker_on_four_cards() -> None:
    """Square Joker: +4 Chips before each four-card hand (card.lua:3427-3434, 3901-3907)."""
    b = Board("BB_SQUARE", "j_blueprint", "j_square")
    square = b.joker("j_square")
    b.play(*b.cards("S_5", "H_5", "D_9", "C_3"))
    assert square.ability["extra"]["chips"] == 4
    b.play(*b.cards("S_6", "H_6"))
    assert square.ability["extra"]["chips"] == 4


def test_runner_on_a_straight() -> None:
    """Runner: +15 Chips before each hand containing a Straight (card.lua:3435-3442, 3908-3914)."""
    b = Board("BB_RUNNER", "j_runner", "j_brainstorm")
    b.play(*b.cards("S_5", "H_5"))
    b.play(*b.cards("H_6", "D_7", "S_8", "C_9", "H_T"))
    assert b.joker("j_runner").ability["extra"]["chips"] == 15


def test_spare_trousers_on_two_pair_and_full_house() -> None:
    """Spare Trousers: +2 Mult before a hand containing a Two Pair (a Full House does)
    (card.lua:3412-3419, 3986-3991)."""
    b = Board("BB_TROUSERS", "j_blueprint", "j_trousers")
    trousers = b.joker("j_trousers")
    b.play(*b.cards("S_5", "H_5", "S_9", "H_9"))
    b.play(*b.cards("S_6", "H_6", "D_6", "S_2", "H_2"))
    b.play(*b.cards("C_7", "H_7"))
    assert trousers.ability["mult"] == 4


def test_green_joker_hands_and_discards() -> None:
    """Green Joker: +1 Mult per hand (card.lua:3563-3569), -1 per discard on its last card, not
    below 0 (2846-2856)."""
    b = Board("BB_GREEN", "j_blueprint", "j_green_joker")
    green = b.joker("j_green_joker")
    b.discard(*b.cards("S_2", "H_3"))
    assert green.ability["mult"] == 0
    b.play(*b.cards("S_5", "H_5"))
    b.play(*b.cards("S_6", "H_6"))
    assert green.ability["mult"] == 2
    b.discard(*b.cards("S_7", "D_8", "C_9"))
    assert green.ability["mult"] == 1


def test_the_hook_discards_reach_green_joker() -> None:
    """The Hook discards two held cards before the hand scores, through
    discard_cards_from_highlighted (blind.lua:466-487): Green Joker loses its Mult on the last
    of them, then gains one in the before pass."""
    b = Board("BB_HOOK_GREEN", "j_green_joker", "j_blueprint", boss="bl_hook")
    green = b.joker("j_green_joker")
    b.play(*b.cards("S_5", "H_5"))
    b.play(*b.cards("S_6", "H_6"))
    assert green.ability["mult"] == 1


def test_a_debuffed_hand_runs_only_the_after_pass() -> None:
    """The Psychic debuffs a hand of fewer than five cards: no before pass (Green Joker) and no
    main step, but the after pass runs (Ice Cream decays, state_events.lua:1068-1075), and the
    hand still counts for Card Sharp (state_events.lua:574-575)."""
    b = Board("BB_PSYCHIC", "j_green_joker", "j_ice_cream", "j_card_sharp", boss="bl_psychic")
    b.play(*b.cards("S_5", "H_5"))
    assert b.last.debuffed
    assert b.joker("j_green_joker").ability["mult"] == 0
    assert b.joker("j_ice_cream").ability["extra"]["chips"] == 95
    b.play(*b.cards("S_6", "H_6", "D_2", "C_9", "H_K"))
    assert b.last.mult == (2 + 1) * 3
    assert b.joker("j_ice_cream").ability["extra"]["chips"] == 90


def test_obelisk_grows_and_resets() -> None:
    """Obelisk (card.lua:3543-3562): +0.2 when another visible hand was played as often, back
    to X1 when the played hand is the most played; Brainstorm copies its X."""
    b = Board("BB_OBELISK", "j_obelisk", "j_brainstorm")
    b.run.op({"op": "set", "hands_left": 6})
    obelisk = b.joker("j_obelisk")
    b.play(*b.cards("S_5", "H_5"))  # Pair 1: the most played, already X1
    assert obelisk.ability["x_mult"] == 1
    b.play(b.add("S_9"))  # High Card 1, Pair 1
    assert obelisk.ability["x_mult"] == 1 + 0.2
    b.play(b.add("H_8"))  # High Card 2: the most played
    assert obelisk.ability["x_mult"] == 1
    b.play(*b.cards("S_7", "H_7"))  # Pair 2, High Card 2
    b.play(*b.cards("S_6", "H_6", "S_4", "H_4"))  # Two Pair 1
    assert obelisk.ability["x_mult"] == 1 + 0.2 + 0.2
    b.play(*b.cards("S_3", "H_3"))  # Pair 3: the most played
    assert obelisk.ability["x_mult"] == 1


def test_ice_cream_melts() -> None:
    """Ice Cream: -5 Chips after each hand, melting when that reaches 0 (card.lua:3571-3600);
    its Blueprint copy scores the chips but never decays it."""
    b = Board("BB_ICE_CREAM", "j_blueprint", "j_ice_cream")
    with b.run.edit_card(b.joker("j_ice_cream")) as card:
        card["ability"]["extra"]["chips"] = 10
    b.play(*b.cards("S_5", "H_5"))
    assert b.joker("j_ice_cream").ability["extra"]["chips"] == 5
    b.play(*b.cards("S_6", "H_6"))
    assert not b.has("j_ice_cream")


def test_popcorn_is_eaten_at_the_end_of_the_round() -> None:
    """Popcorn: -4 Mult at the end of each round, eaten when that reaches 0
    (card.lua:2945-2974)."""
    b = Board("BB_POPCORN", "j_popcorn", "j_blueprint", "j_popcorn")
    with b.run.edit_card(b.joker("j_popcorn", 1)) as card:
        card["ability"]["mult"] = 4
    b.win_with(*b.cards("S_5", "H_5"))
    assert [j.ability["mult"] for j in b.run.cards("jokers") if j.key == "j_popcorn"] == [16]


# ---------------------------------------------------------------------------
# Enhancement changes (before)
# ---------------------------------------------------------------------------


def test_midas_mask_gilds_the_scoring_face_cards() -> None:
    """Midas Mask: every scoring face card becomes a Gold Card (card.lua:3443-3464); set_ability
    stamps hands_played_at_create (card.lua:337, G63). Blueprint's copy does nothing."""
    b = Board("BB_MIDAS", "j_blueprint", "j_midas_mask")
    b.play(*b.cards("S_5", "H_5"))
    faces = b.cards("S_K", "H_K", "D_K", "C_2")
    b.play(*faces)
    assert [c.key for c in faces] == ["m_gold", "m_gold", "m_gold", "c_base"]
    assert faces[0].ability["hands_played_at_create"] == 1


def test_midas_mask_skips_debuffed_faces() -> None:
    """Under The Plant face cards are debuffed, so is_face() is false for them."""
    b = Board("BB_MIDAS_PLANT", "j_midas_mask", boss="bl_plant")
    faces = b.cards("S_Q", "H_Q")
    b.play(*faces)
    assert [c.key for c in faces] == ["c_base", "c_base"]


def test_midas_mask_with_pareidolia_gilds_a_stone() -> None:
    """With Pareidolia every scoring card is a face, a Stone Card included."""
    b = Board("BB_MIDAS_STONE", "j_pareidolia", "j_midas_mask")
    played = b.cards("S_5", "H_5")
    played.append(b.add("D_9", enhancement="m_stone"))
    b.play(*played)
    assert [c.key for c in played] == ["m_gold", "m_gold", "m_gold"]


def test_midas_mask_under_the_pillar() -> None:
    """set_ability builds a new ability table (card.lua:277-297), so a gilded card loses
    played_this_ante, and the boss re-check that follows (card.lua:365) finds nothing for The
    Pillar to debuff; the ungilded pair keeps the flag."""
    b = Board("BB_MIDAS_PILLAR", "j_midas_mask", boss="bl_pillar")
    jacks, twos = b.cards("S_J", "H_J"), b.cards("S_2", "H_2")
    b.play(*jacks, *twos)
    assert [c.key for c in jacks] == ["m_gold", "m_gold"]
    assert not any(c.debuff or c.ability.get("played_this_ante") for c in jacks)
    assert all(c.ability.get("played_this_ante") for c in twos)


def test_vampire_drains_the_enhanced_cards() -> None:
    """Vampire: each enhanced, non-debuffed scoring card returns to c_base and Vampire gains
    X0.1 (card.lua:3465-3490); the shared x_mult rule scores it, copies included."""
    b = Board("BB_VAMPIRE", "j_vampire", "j_brainstorm")
    played = [
        b.add("S_5", enhancement="m_bonus"),
        b.add("H_5", enhancement="m_mult"),
        b.add("D_5", enhancement="m_glass"),
        b.add("C_5"),
    ]
    b.play(*played)
    assert b.joker("j_vampire").ability["x_mult"] == 1 + 0.1 * 3
    assert all(c.key == "c_base" for c in played)
    assert played[0].ability["hands_played_at_create"] == 0
    b.play(*b.cards("S_9", "H_9"))


def test_midas_mask_then_vampire() -> None:
    """Midas Mask gilds the Kings in its before turn; Vampire, later in the same pass, drains the
    new Gold Cards."""
    b = Board("BB_MIDAS_VAMPIRE", "j_midas_mask", "j_vampire")
    kings = b.cards("S_K", "H_K")
    b.play(*kings)
    assert [c.key for c in kings] == ["c_base", "c_base"]
    assert b.joker("j_vampire").ability["x_mult"] == 1 + 0.1 * 2


def test_vampire_skips_debuffed_cards() -> None:
    """Vampire leaves a debuffed card's enhancement (card.lua:3468): under The Goad the Spade
    stays a Bonus Card."""
    b = Board("BB_VAMPIRE_DEBUFF", "j_vampire", boss="bl_goad")
    spade, heart = b.add("S_5", enhancement="m_bonus"), b.add("H_5", enhancement="m_bonus")
    assert spade.debuff
    b.play(spade, heart)
    assert (spade.key, heart.key) == ("m_bonus", "c_base")


def test_vampire_drains_a_stone_under_the_goad() -> None:
    """A Stone Card belongs to no suit; drained, it is a Spade again and The Goad's re-check
    debuffs it (card.lua:365)."""
    b = Board("BB_VAMPIRE_GOAD", "j_vampire", boss="bl_goad")
    stone = b.add("S_2", enhancement="m_stone")
    b.play(stone, *b.cards("H_7", "D_7"))
    assert stone.key == "c_base"
    assert stone.debuff


def test_vampire_and_the_steel_joker_tally() -> None:
    """Steel Joker reads the per-frame steel_tally, counted before Vampire drained the card."""
    b = Board("BB_VAMPIRE_STEEL", "j_vampire", "j_steel_joker")
    b.play(*[b.add("S_5", enhancement="m_steel"), b.add("H_5")])
    b.play(*b.cards("S_6", "H_6"))


# ---------------------------------------------------------------------------
# The Arm's preview (G64)
# ---------------------------------------------------------------------------


def test_the_arm_preview_sets_triggered_on_a_discard() -> None:
    """Selecting a hand above level 1 under The Arm sets blind.triggered in the preview
    (parse_highlighted -> debuff_hand(check), blind.lua:550-559), and a discard keeps it."""
    b = Board("BB_ARM_PREVIEW", "j_space", boss="bl_arm")
    b.run.level_up("Pair")
    b.discard(*b.cards("S_5", "H_5"))
    assert b.run.read("blind.triggered")
    b.discard(b.add("S_9"))
    assert not b.run.read("blind.triggered")
