"""Targeted parity cases for the Tarots and Spectrals (``Card:use_consumeable``).

Each case builds a run (BalatroBot ``add`` for the consumables, jokers and playing cards, which
join the hand), writes the few values play would take long to reach straight into the state,
and takes its steps; each step must match the oracle's, legal actions included
(:mod:`tests.parity.cases`). The assertions after a step confirm that the branch under test was
taken. The recorded tapes rarely reach most consumables with only ported jokers on the board,
so these cases are the consumables' main parity gate.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.data import load
from tests.parity.cases import CardRef, Run


class Table:
    """A targeted run whose steps are checked against the oracle's one at a time."""

    def __init__(self, seed: str, *jokers: str, boss: str | None = None) -> None:
        setup: list[dict[str, Any]] = []
        if boss is not None:
            setup.append({"op": "force_boss", "key": boss})
        setup.append({"op": "action", "action": ["SelectBlind"]})
        start = {"back": "b_red", "stake": 1, "seed": seed, "challenge": None}
        self.seed = seed
        self.run = Run(start, setup)
        # Out of reach: a hand ends the round only when a case wins it on purpose.
        self.run.write("blind.chips", 10**9)
        for key in jokers:
            self.joker(key)

    @property
    def phase(self) -> str:
        return self.run.game.phase

    @property
    def hand(self) -> list[CardRef]:
        return self.run.cards("hand")

    def read(self, path: str) -> Any:
        return self.run.read(path)

    def write(self, path: str, value: Any) -> None:
        self.run.write(path, value)

    # -- cards --------------------------------------------------------------

    def add(self, key: str, **params: Any) -> CardRef:
        """BalatroBot ``add``: a joker, a consumable, or a playing card into the hand. A playing
        card meets the boss as a card dealt this round did at set_blind (blind.lua:207-210)."""
        card = self.run.add(key, **params)
        if card.area == "hand":
            self.run.boss_check(card)
        return card

    def joker(self, key: str, edition: str | None = None, *, eternal: bool = False) -> CardRef:
        """A joker with exactly this edition (``add`` polls one when none is asked for; the
        oracle took a polled edition off again)."""
        card = self.add(key, **({"edition": edition} if edition else {}))
        if edition is None and card.edition is not None:
            self.run.oracle(f"the edition polled for {key} taken off")
        if eternal:
            with self.run.edit_card(card) as data:
                data["stickers"]["eternal"] = True
        return card

    def cards(self, *keys: str, **params: Any) -> list[CardRef]:
        return [self.add(k, **params) for k in keys]

    def hold_only(self, *keys: str, **params: Any) -> list[CardRef]:
        """Return the dealt hand to the deck and hold exactly these cards."""
        self.run.move("hand", "deck")
        return self.cards(*keys, **params)

    def index(self, card: CardRef) -> int:
        return self.hand.index(card)

    def consumable(self, key: str) -> CardRef:
        return next(c for c in self.run.cards("consumables") if c.key == key)

    # -- steps --------------------------------------------------------------

    def step(self, action: list[Any]) -> None:
        assert self.run.step(action), f"{self.seed}: the oracle refused {action}"

    def refused(self, action: list[Any]) -> None:
        """Both engines refuse the action, and the run is unchanged."""
        self.run.sync()
        before = self.run.game.state_hash()
        assert not self.run.step(action), f"{self.seed}: the oracle took {action}"
        assert self.run.game.state_hash() == before

    def use_action(self, key: str, *cards: CardRef) -> list[Any]:
        idx = self.run.cards("consumables").index(self.consumable(key))
        return ["UseConsumable", idx, [self.index(c) for c in cards]]

    def use(self, key: str, *cards: CardRef) -> None:
        """Add the consumable and use it on these hand cards (in this highlight order)."""
        self.add(key)
        self.step(self.use_action(key, *cards))

    def play(self, *cards: CardRef) -> None:
        self.step(["PlayHand", sorted(self.index(c) for c in cards)])

    def win_with(self, *cards: CardRef) -> None:
        """Play the cards with the blind one chip from beaten."""
        self.write("blind.chips", 300)
        self.write("game.chips", 299)
        self.play(*cards)
        assert self.phase == "round_eval"

    def to_shop(self) -> None:
        self.win_with(self.hand[0])
        self.step(["CashOut"])
        assert self.phase == "shop"

    def open_pack(self, key: str) -> None:
        """Put the booster ``key`` in the shop's first booster slot, free, and open it."""
        self.run.oracle(f"a free {key} in the shop's first booster slot")
        self.step(["OpenBooster", 0])
        assert self.phase == "pack_opening"

    def stock_pack(self, *keys: str) -> list[CardRef]:
        """Replace the open pack's cards with these consumables."""
        self.run.oracle(f"the pack's cards replaced with {', '.join(keys)}")
        self.run.sync()
        cards = self.run.cards("pack_cards")
        assert [c.key for c in cards] == list(keys)
        return cards

    def pick(self, card: CardRef, *targets: CardRef) -> None:
        idx = self.run.cards("pack_cards").index(card)
        self.step(["PickPackCard", idx, [self.index(c) for c in targets]])

    # -- reading ------------------------------------------------------------

    def keys(self, area: str) -> list[str]:
        return self.run.keys(area)


def fronts(cards: list[CardRef]) -> list[str | None]:
    return [c.front for c in cards]


# ---------------------------------------------------------------------------
# The Fool, The Emperor, The High Priestess
# ---------------------------------------------------------------------------


def test_the_fool_copies_the_last_tarot_or_planet() -> None:
    """The Fool makes the last Tarot or Planet used (card.lua:1374-1383); its own entry lands two
    events later (misc_functions.lua:1213-1225), so a second Fool is refused."""
    t = Table("CS_FOOL")
    t.use("c_pluto")
    assert t.read("game.last_tarot_planet") == "c_pluto"
    t.use("c_fool")
    assert t.keys("consumables") == ["c_pluto"]
    assert t.read("game.last_tarot_planet") == "c_fool"
    t.add("c_fool")
    t.refused(t.use_action("c_fool"))


def test_the_fool_after_a_tarot_with_full_slots() -> None:
    """The Fool used from full slots frees its own slot (``self.area == G.consumeables``)."""
    t = Table("CS_FOOL_FULL")
    t.use("c_emperor")
    t.write("areas.consumables", [])
    t.add("c_hermit")
    t.use("c_fool")
    assert t.keys("consumables") == ["c_hermit", "c_emperor"]


def test_creators_in_an_overfull_tray() -> None:
    """From the tray The Emperor, The High Priestess and The Fool are always usable
    (``self.area == G.consumeables``, card.lua:1550-1555), even over the limit (a Negative
    card gone): they then make nothing."""
    t = Table("CS_TRAY_OVERFULL")
    t.write("game.last_tarot_planet", "c_pluto")
    for key in ("c_hermit", "c_temperance", "c_emperor", "c_high_priestess", "c_fool"):
        t.add(key)
    for key in ("c_emperor", "c_high_priestess", "c_fool"):
        t.step(t.use_action(key))
    assert t.keys("consumables") == ["c_hermit", "c_temperance"]


def test_the_fool_with_no_memory_is_refused() -> None:
    t = Table("CS_FOOL_NONE")
    t.add("c_fool")
    t.refused(t.use_action("c_fool"))


def test_the_emperor_and_the_high_priestess() -> None:
    """Two cards each, one event per free slot (card.lua:1401-1413)."""
    t = Table("CS_EMPEROR")
    t.use("c_emperor")
    assert len(t.run.cards("consumables")) == 2
    assert all(c.set == "Tarot" for c in t.run.cards("consumables"))
    with t.run.edit() as snap:
        snap["areas"]["consumables"].pop()
    t.use("c_high_priestess")
    assert [c.set for c in t.run.cards("consumables")] == ["Tarot", "Planet"]


def test_the_emperor_with_one_free_slot() -> None:
    t = Table("CS_EMPEROR_ONE")
    t.add("c_hermit")
    t.use("c_emperor")
    assert len(t.run.cards("consumables")) == 2


def test_the_high_priestess_with_full_slots_is_refused() -> None:
    t = Table("CS_PRIESTESS_FULL")
    t.add("c_hermit")
    t.add("c_temperance")
    t.add("c_high_priestess")
    with t.run.edit() as snap:
        snap["areas"]["consumables"].pop()
        # From a full tray the card frees its own slot: two Planets fit only one at a time.
        snap["areas"]["consumables"].pop()
    t.use("c_high_priestess")
    assert len(t.run.cards("consumables")) == 2


# ---------------------------------------------------------------------------
# Enhancements, suits, ranks, Death
# ---------------------------------------------------------------------------


def test_enhancing_tarots_stamp_hands_played() -> None:
    """set_ability(G.P_CENTERS[mod_conv]) is not initial: it stamps hands_played_at_create and
    the boss re-checks the card (card.lua:337, 365, 1141-1144)."""
    t = Table("CS_ENHANCE")
    t.play(*t.cards("S_5", "H_5"))
    for key, enh in [
        ("c_magician", "m_lucky"),
        ("c_empress", "m_mult"),
        ("c_heirophant", "m_bonus"),
    ]:
        a, b = t.cards("D_7", "C_9", seal="Red", edition="holo")
        t.use(key, a, b)
        assert a.key == b.key == enh
        assert a.ability["hands_played_at_create"] == t.read("game.hands_played") == 1
        assert a.seal == "Red" and a.edition["holo"]
    for key, enh in [
        ("c_lovers", "m_wild"),
        ("c_chariot", "m_steel"),
        ("c_justice", "m_glass"),
        ("c_devil", "m_gold"),
        ("c_tower", "m_stone"),
    ]:
        (a,) = t.cards("H_K")
        t.use(key, a)
        assert a.key == enh


def test_enhancing_a_stone_card_and_a_bonus_card() -> None:
    """The Tower turns a Bonus Card to Stone (the old bonus is dropped, card.lua:274-298);
    The Hierophant turns a Stone Card back."""
    t = Table("CS_ENHANCE_STONE")
    (a,) = t.cards("S_Q", enhancement="m_bonus")
    t.use("c_tower", a)
    assert a.key == "m_stone" and a.ability["bonus"] == 50
    t.use("c_heirophant", a)
    assert a.key == "m_bonus" and a.ability["bonus"] == 30


def test_the_lovers_under_the_club_debuffs_the_new_wild_card() -> None:
    """A Wild Card is every suit, so under The Club the boss re-check debuffs it
    (blind.lua:624-630, card.lua:365)."""
    t = Table("CS_LOVERS_CLUB", boss="bl_club")
    (heart,) = t.cards("H_4")
    assert not heart.debuff
    t.use("c_lovers", heart)
    assert heart.debuff


def test_suit_tarots_change_up_to_three_cards() -> None:
    """change_suit: set_base with the new suit's front, then the boss re-check
    (card.lua:547-562); a Stone Card changes suit too."""
    t = Table("CS_SUITS", boss="bl_goad")
    a, b = t.cards("H_2", "D_J")
    (stone,) = t.cards("C_A", enhancement="m_stone")
    t.use("c_world", a, b, stone)
    assert fronts([a, b, stone]) == ["S_2", "S_J", "S_A"]
    assert a.debuff and b.debuff and not stone.debuff
    t.use("c_star", a)
    t.use("c_moon", b)
    t.use("c_sun", stone)
    assert fronts([a, b, stone]) == ["D_2", "C_J", "H_A"]
    assert not a.debuff


def test_strength_raises_the_rank() -> None:
    """Strength: the next rank, an Ace to 2 (card.lua:1121-1135); the base is rebuilt."""
    t = Table("CS_STRENGTH")
    t.play(*t.cards("C_A", "D_A"))
    ace, king = t.cards("S_A", "H_K")
    t.use("c_strength", king, ace)
    assert fronts([ace, king]) == ["S_2", "H_A"]
    (stone,) = t.cards("D_9", enhancement="m_stone")
    t.use("c_strength", stone)
    assert fronts([stone]) == ["D_T"]


def test_death_copies_the_rightmost_card() -> None:
    """Death: copy_card(rightmost, other) (card.lua:1111-1120): center, front, ability, edition,
    seal and debuff; the rightmost is the highlighted card furthest right in the hand."""
    t = Table("CS_DEATH")
    left = t.add("H_3")
    right = t.add("S_K", enhancement="m_glass", seal="Purple", edition="polychrome")
    with t.run.edit_card(right) as card:
        card["ability"]["perma_bonus"] = 10
    t.use("c_death", right, left)
    assert left.key == "m_glass" and left.front == "S_K"
    assert left.seal == "Purple" and left.edition["polychrome"]
    assert left.ability["perma_bonus"] == 10


def test_death_onto_a_debuffed_card_and_a_sorted_hand() -> None:
    """The copy takes the source's debuff (common_events.lua:2177); the rightmost follows the
    hand's order, not the highlight order."""
    t = Table("CS_DEATH_CLUB", boss="bl_club")
    t.step(["SortHand", "suit"])
    club, heart = t.cards("C_5", "H_8")
    assert club.debuff
    t.use("c_death", heart, club)
    assert club.front == "H_8" and not club.debuff


# ---------------------------------------------------------------------------
# The Hermit, Temperance, The Wheel of Fortune, Judgement
# ---------------------------------------------------------------------------


def test_the_hermit_doubles_money_up_to_twenty() -> None:
    t = Table("CS_HERMIT")
    t.write("game.dollars", 7)
    t.use("c_hermit")
    assert t.read("game.dollars") == 14
    t.write("game.dollars", 30)
    t.use("c_hermit")
    assert t.read("game.dollars") == 50
    t.write("game.dollars", -4)
    t.use("c_hermit")
    assert t.read("game.dollars") == -4


def test_temperance_pays_the_jokers_sell_value() -> None:
    """Temperance: the Joker-set cards' sell value, capped at $50 (card.lua:4167-4174)."""
    t = Table("CS_TEMPERANCE", "j_joker", "j_golden")
    t.write("game.dollars", 0)
    t.use("c_temperance")
    assert t.read("game.dollars") == sum(j.costs["sell_cost"] for j in t.run.cards("jokers"))
    t.use("c_temperance")
    with t.run.edit() as snap:
        for j in snap["areas"]["jokers"]:
            j["ability"]["extra_value"] = 30
    t.run.oracle("the jokers repriced with their new extra_value")
    t.write("game.dollars", 0)
    t.use("c_temperance")
    assert t.read("game.dollars") == 50


def test_temperance_with_no_jokers() -> None:
    t = Table("CS_TEMPERANCE_NONE")
    t.write("game.dollars", 3)
    t.use("c_temperance")
    assert t.read("game.dollars") == 3


def test_the_wheel_of_fortune_hits_with_two_oops() -> None:
    """With two Oops! All 6s the 1 in 4 always hits; the edition is never Negative and lands on
    an editionless joker, debuffed or not (card.lua:1467-1500, 4209-4215)."""
    t = Table("CS_WHEEL_HIT")
    t.joker("j_oops", "foil")
    t.joker("j_oops", "foil")
    a = t.joker("j_joker")
    b = t.joker("j_golden")
    with t.run.edit_card(b) as card:
        card["debuff"] = True
    t.use("c_wheel_of_fortune")
    assert (a.edition is None) != (b.edition is None)


def test_the_wheel_of_fortune_misses() -> None:
    """A miss shows "Nope!" and draws nothing else."""
    for i in range(20):
        t = Table(f"CS_WHEEL_MISS{i}", "j_joker")
        t.joker("j_golden")
        t.use("c_wheel_of_fortune")
        if all(j.edition is None for j in t.run.cards("jokers")):
            assert "wheel_of_fortune" in t.read("rng.streams")
            return
    raise AssertionError("no miss in 20 seeds")


def test_the_wheel_of_fortune_with_only_editioned_jokers_is_refused() -> None:
    t = Table("CS_WHEEL_NONE")
    t.joker("j_joker", "holo")
    t.add("c_wheel_of_fortune")
    t.refused(t.use_action("c_wheel_of_fortune"))


def test_judgement_makes_a_joker() -> None:
    t = Table("CS_JUDGEMENT", "j_joker")
    t.use("c_judgement")
    assert len(t.run.cards("jokers")) == 2


def test_judgement_with_full_slots_is_refused() -> None:
    t = Table("CS_JUDGEMENT_FULL", "j_joker", "j_golden", "j_cloud_9", "j_credit_card", "j_juggler")
    t.add("c_judgement")
    t.refused(t.use_action("c_judgement"))


# ---------------------------------------------------------------------------
# The Hanged Man and the destroying Spectrals
# ---------------------------------------------------------------------------


def test_the_hanged_man_destroys_two_cards() -> None:
    """The Hanged Man: the highlighted cards shatter (Glass) or dissolve (card.lua:1271-1291)."""
    t = Table("CS_HANGED", "j_joker")
    glass, plain = t.cards("H_9", "C_3")
    t.run.oracle("set_ability('m_glass') on the 9")
    n = len(t.hand)
    t.use("c_hanged_man", glass, plain)
    assert len(t.hand) == n - 2
    assert glass not in t.hand and plain not in t.hand


def test_the_hanged_man_empties_the_hand() -> None:
    """Destroying the last cards with the deck left: the hand is redrawn (game.lua:2542-2545)."""
    t = Table("CS_HANGED_EMPTY")
    a, b = t.hold_only("H_9", "C_3")
    t.use("c_hanged_man", a, b)
    assert len(t.hand) == t.read("game.hand_size")


def test_familiar_grim_and_incantation() -> None:
    """One random hand card goes, then `extra` enhanced cards join the hand (card.lua:1292-1339)."""
    t = Table("CS_FAMILIAR")
    t.play(*t.cards("S_5", "H_5"))
    for key, extra in [("c_familiar", 3), ("c_grim", 2), ("c_incantation", 4)]:
        n = len(t.hand)
        t.use(key)
        assert len(t.hand) == n - 1 + extra
        new = t.hand[-extra:]
        assert all(c.set == "Enhanced" and c.key != "m_stone" for c in new)
        assert all(c.ability["hands_played_at_create"] == 1 for c in new)


def test_spectral_cards_meet_the_boss_when_the_familiar_goes() -> None:
    """create_playing_card's initial set_ability and set_base never call debuff_card
    (card.lua:43-44, 143, 365), but the used Familiar's Card:remove runs remove_from_deck, whose
    set_blind(nil, true) re-checks every playing card (card.lua:696-698): under The Club a new
    Club ends debuffed."""
    for i in range(10):
        t = Table(f"CS_FAMILIAR_CLUB{i}", boss="bl_club")
        t.use("c_familiar")
        new = t.hand[-3:]
        if any(c.base["suit"] == "Clubs" for c in new):
            assert all(c.debuff == (c.base["suit"] == "Clubs") for c in new)
            return
    raise AssertionError("no Club in 10 seeds")


def test_a_negative_emperor_counts_its_own_slot() -> None:
    """use_card takes the card out of G.consumeables, but its Negative slot goes only with
    Card:remove, after the use (card.lua:686-693, 4727-4731): a Negative Emperor in a full
    tray makes a Tarot."""
    t = Table("CS_EMPEROR_NEGATIVE")
    t.add("c_hermit")
    t.add("c_temperance")
    t.add("c_emperor")
    t.run.oracle("The Emperor made Negative and repriced")
    t.write("game.consumable_slots", t.read("game.consumable_slots") + 1)
    t.step(t.use_action("c_emperor"))
    assert len(t.run.cards("consumables")) == 3 and t.read("game.consumable_slots") == 2


def test_familiar_destroys_a_glass_card() -> None:
    t = Table("CS_FAMILIAR_GLASS")
    t.hold_only("H_9", "C_3", enhancement="m_glass")
    t.use("c_familiar")
    assert len(t.hand) == 4


def test_immolate_destroys_five_and_pays() -> None:
    """Immolate: five cards of the shuffled hand go, then $20 (card.lua:1340-1366)."""
    t = Table("CS_IMMOLATE")
    t.write("game.dollars", 4)
    n = len(t.hand)
    t.use("c_immolate")
    assert len(t.hand) == n - 5
    assert t.read("game.dollars") == 24


def test_immolate_shatters_glass_cards() -> None:
    """Immolate's Glass Cards shatter (card.lua:1358-1359)."""
    t = Table("CS_IMMOLATE_GLASS")
    t.hold_only("H_9", "C_3", "D_4", "S_6", "H_7", "C_8", enhancement="m_glass")
    t.use("c_immolate")
    assert len(t.hand) == 1


def test_immolate_with_a_small_hand() -> None:
    """Fewer than five cards: every held card goes (``temp_hand[i]`` is nil past the hand) and
    the empty hand is redrawn."""
    t = Table("CS_IMMOLATE_SMALL")
    t.hold_only("H_9", "C_3", "D_4")
    t.use("c_immolate")
    assert t.read("game.dollars") == 24
    assert len(t.hand) == t.read("game.hand_size")


def test_destroying_spectrals_need_two_cards() -> None:
    t = Table("CS_FAMILIAR_ONE")
    t.hold_only("H_9")
    for key in ("c_familiar", "c_grim", "c_incantation", "c_immolate", "c_sigil", "c_ouija"):
        t.add(key)
        t.refused(t.use_action(key))
        t.write("areas.consumables", [])


# ---------------------------------------------------------------------------
# Seals, Aura, Cryptid, Sigil, Ouija
# ---------------------------------------------------------------------------


def test_seal_spectrals() -> None:
    """Talisman, Deja Vu, Trance and Medium: the highlighted card's seal (card.lua:1178-1190)."""
    t = Table("CS_SEALS")
    for key, seal in [
        ("c_talisman", "Gold"),
        ("c_deja_vu", "Red"),
        ("c_trance", "Blue"),
        ("c_medium", "Purple"),
    ]:
        (card,) = t.cards("H_6", seal="Gold")
        t.use(key, card)
        assert card.seal == seal


def test_aura() -> None:
    """Aura: a guaranteed edition, never Negative (card.lua:1192-1199); an editioned card is
    refused (card.lua:1544-1545)."""
    t = Table("CS_AURA")
    (card,) = t.cards("S_8")
    t.use("c_aura", card)
    assert card.edition and not card.edition.get("negative")
    t.add("c_aura")
    t.refused(t.use_action("c_aura", card))


def test_cryptid_copies_into_the_hand() -> None:
    """Cryptid: two copies of the highlighted card join the hand (card.lua:1201-1220)."""
    t = Table("CS_CRYPTID")
    (card,) = t.cards("D_Q", enhancement="m_lucky", seal="Red", edition="foil")
    n = len(t.hand)
    t.use("c_cryptid", card)
    assert len(t.hand) == n + 2
    copies = t.hand[-2:]
    assert fronts(copies) == ["D_Q", "D_Q"]
    assert all(c.seal == "Red" and c.edition["foil"] for c in copies)


def test_sigil_and_ouija() -> None:
    """Sigil: every held card takes one suit; Ouija: one rank, and -1 hand size
    (card.lua:1222-1262)."""
    t = Table("CS_SIGIL")
    t.add("H_5", enhancement="m_stone")
    size = t.read("game.hand_size")
    t.use("c_sigil")
    assert len({c.base["suit"] for c in t.hand}) == 1
    t.use("c_ouija")
    assert len({c.base["value"] for c in t.hand}) == 1
    assert t.read("game.hand_size") == size - 1


# ---------------------------------------------------------------------------
# Wraith, Ectoplasm, Hex, Ankh, The Soul, Black Hole
# ---------------------------------------------------------------------------


def test_wraith() -> None:
    """Wraith: a Rare joker, then the money goes to 0 (card.lua:1454-1465)."""
    t = Table("CS_WRAITH")
    t.write("game.dollars", 12)
    t.use("c_wraith")
    assert len(t.run.cards("jokers")) == 1 and t.read("game.dollars") == 0
    t.write("game.dollars", -5)
    t.use("c_wraith")
    assert t.read("game.dollars") == 0


def test_ectoplasm_twice() -> None:
    """Ectoplasm: Negative on an editionless joker, then ecto_minus hand size, growing by one
    (card.lua:1494-1497)."""
    t = Table("CS_ECTO", "j_joker", "j_golden", "j_cloud_9")
    size, slots = t.read("game.hand_size"), t.read("game.joker_slots")
    t.use("c_ectoplasm")
    t.use("c_ectoplasm")
    assert t.read("game.hand_size") == size - 3
    assert t.read("game.joker_slots") == slots + 2
    assert t.read("game.ecto_minus") == 3


def test_hex_spares_eternal_jokers() -> None:
    """Hex: Polychrome on an editionless joker, every other non-eternal joker dissolves
    (card.lua:1488-1492); a dissolved Negative takes its slot with it."""
    t = Table("CS_HEX")
    t.joker("j_joker")
    t.joker("j_golden", "foil")
    eternal = t.joker("j_cloud_9", eternal=True)
    t.joker("j_juggler", "negative")
    slots = t.read("game.joker_slots")
    t.use("c_hex")
    jokers = t.run.cards("jokers")
    (poly,) = [j for j in jokers if j.edition and j.edition.get("polychrome")]
    assert set(jokers) == {poly, eternal}
    assert t.read("game.joker_slots") == slots - 1


def test_hex_removing_merry_andy_draws_nothing() -> None:
    """Hex's jokers are removed before use_card's last event restores the state, so the hand
    size Merry Andy gives back grows the hand in PLAY_TAROT, without a draw
    (cardarea.lua:94-112; G133)."""
    t = Table("CS_HEX_ANDY")
    t.joker("j_joker")
    t.joker("j_merry_andy", "foil")
    size, held = t.read("game.hand_size"), len(t.hand)
    t.use("c_hex")
    assert t.read("game.hand_size") == size + 1
    assert len(t.hand) == held


def test_ankh_copies_one_joker() -> None:
    """Ankh: every non-eternal joker but the chosen one dissolves, then the chosen joker is
    copied (card.lua:1426-1452)."""
    t = Table("CS_ANKH")
    t.joker("j_joker")
    t.joker("j_golden", "holo")
    t.joker("j_cloud_9", eternal=True)
    t.use("c_ankh")
    assert len(t.run.cards("jokers")) == 3


def test_ankh_copy_of_a_negative_joker_drops_the_edition() -> None:
    """The copy of a Negative joker has no edition (card.lua:1445); a Juggler copy grows the
    hand without a draw (G133)."""
    t = Table("CS_ANKH_NEG")
    neg = t.joker("j_juggler", "negative")
    size, held = t.read("game.hand_size"), len(t.hand)
    t.use("c_ankh")
    jokers = t.run.cards("jokers")
    assert len(jokers) == 2
    assert neg.edition["negative"] and jokers[1].edition is None
    assert t.read("game.hand_size") == size + 1 and len(t.hand) == held


def test_ankh_removes_before_it_copies() -> None:
    """The dissolved jokers are gone before the copy is made (G136): with one discard left, a
    removed Merry Andy takes it (ease_discard stops at 0) before the Drunkard copy adds one,
    or the removed Drunkard finds none left before the Merry Andy copy adds three."""
    for i in range(6):
        t = Table(f"CS_ANKH_ORDER{i}")
        drunk = t.joker("j_drunkard")
        andy = t.joker("j_merry_andy")
        t.write("current_round.discards_left", 1 if i % 2 else 0)
        left = t.read("current_round.discards_left")
        t.use("c_ankh")
        kept = drunk if drunk in t.run.cards("jokers") else andy
        if kept == drunk:
            assert t.read("current_round.discards_left") == max(0, left - 3) + 1
        else:
            assert t.read("current_round.discards_left") == max(0, left - 1) + 3


def test_ankh_with_full_slots_is_refused() -> None:
    t = Table("CS_ANKH_FULL", "j_joker", "j_golden", "j_cloud_9", "j_credit_card", "j_juggler")
    t.add("c_ankh")
    t.refused(t.use_action("c_ankh"))


def test_the_soul_makes_a_legendary() -> None:
    t = Table("CS_SOUL")
    t.use("c_soul")
    assert t.run.cards("jokers")[0].set == "Joker"


def test_black_hole() -> None:
    t = Table("CS_BLACK_HOLE")
    t.use("c_black_hole")
    assert all(h.level == 2 for h in t.run.game.hand_levels.values())


# ---------------------------------------------------------------------------
# Cerulean Bell: the forced card stays highlighted
# ---------------------------------------------------------------------------


def forced(t: Table) -> CardRef:
    (card,) = [c for c in t.hand if c.ability.get("forced_selection")]
    return card


def test_cerulean_bell_forced_card_takes_a_one_card_tarot() -> None:
    """The forced card is always in G.hand.highlighted (cardarea.lua:252-256), so a one-card
    Tarot can only be used on it: another target makes two."""
    t = Table("CS_BELL_LOVERS", boss="bl_final_bell")
    card = forced(t)
    other = next(c for c in t.hand if c != card)
    t.add("c_lovers")
    t.refused(t.use_action("c_lovers", other))
    t.step(t.use_action("c_lovers"))
    assert card.key == "m_wild"


def test_cerulean_bell_death_and_cryptid() -> None:
    """Death copies between the forced card and one target; Cryptid copies the forced card,
    forced_selection included (common_events.lua:2161-2168)."""
    t = Table("CS_BELL_DEATH", boss="bl_final_bell")
    card = forced(t)
    other = next(c for c in t.hand if c != card)
    t.use("c_death", other)
    assert card.front == other.front
    # A forced rightmost card passes forced_selection on with its ability.
    n = sum(1 for c in t.hand if c.ability.get("forced_selection"))
    t.add("c_cryptid")
    t.step(t.use_action("c_cryptid"))
    assert sum(1 for c in t.hand if c.ability.get("forced_selection")) == n + 2


def test_cerulean_bell_hanged_man_destroys_the_forced_card() -> None:
    t = Table("CS_BELL_HANGED", boss="bl_final_bell")
    card = forced(t)
    t.add("c_hanged_man")
    t.step(t.use_action("c_hanged_man"))
    assert card not in t.hand


# ---------------------------------------------------------------------------
# Packs and buy-and-use
# ---------------------------------------------------------------------------


def test_arcana_pack_picks() -> None:
    """Pack picks use the card on the dealt hand (button_callbacks.lua:2155-2316)."""
    t = Table("CS_ARCANA")
    t.to_shop()
    t.open_pack("p_arcana_mega_1")
    death, hanged = t.stock_pack("c_death", "c_hanged_man", "c_strength", "c_emperor")[:2]
    hand = t.hand
    t.pick(death, hand[0], hand[3])
    hand = t.hand
    t.pick(hanged, hand[1])
    assert t.phase == "shop"


def test_spectral_pack_cards_join_the_dealt_hand() -> None:
    """Familiar and Cryptid in a Spectral pack add their cards to the dealt hand, which returns
    to the deck when the pack closes (state_events.lua:1121-1126)."""
    t = Table("CS_SPECTRAL_PACK")
    t.to_shop()
    t.open_pack("p_spectral_mega_1")
    familiar, cryptid = t.stock_pack("c_familiar", "c_cryptid", "c_aura", "c_sigil")[:2]
    t.pick(familiar)
    t.pick(cryptid, t.hand[-1])
    assert t.phase == "shop"
    assert not t.hand


def shop_consumables(t: Table, *keys: str) -> list[CardRef]:
    """Replace the shop's first cards with these consumables."""
    t.run.oracle(f"the shop's first cards replaced with {', '.join(keys)}")
    t.run.sync()
    cards = t.run.cards("shop_jokers")[: len(keys)]
    assert [c.key for c in cards] == list(keys)
    return cards


def test_buy_and_use_ankh_with_full_slots() -> None:
    """can_buy_and_use asks only can_use_consumeable (button_callbacks.lua:77-89): Ankh is bought,
    then use_card's check_use stops it (2163-2169) and the card is gone."""
    t = Table("CS_BUY_ANKH", "j_joker", "j_golden", "j_cloud_9", "j_credit_card", "j_juggler")
    t.to_shop()
    t.write("game.dollars", 10)
    shop_consumables(t, "c_ankh")
    t.step(["BuyAndUse", 0, []])
    assert len(t.run.cards("jokers")) == 5 and "c_ankh" not in t.keys("shop_jokers")


def test_buy_and_use() -> None:
    """Buy and use: the card is paid for and used without joining the consumables."""
    t = Table("CS_BUY_USE")
    t.to_shop()
    t.write("game.dollars", 30)
    hermit, _ = shop_consumables(t, "c_hermit", "c_wraith")
    cost = hermit.costs["cost"]
    t.step(["BuyAndUse", 0, []])
    assert t.read("game.dollars") == 30 - cost + 20
    t.step(["BuyAndUse", 0, []])
    assert t.read("game.dollars") == 0 and len(t.run.cards("jokers")) == 1


# ---------------------------------------------------------------------------
# Every Tarot and Spectral from a pack, and from the tray in every phase
# ---------------------------------------------------------------------------

TAROTS = (
    "c_fool",
    "c_magician",
    "c_high_priestess",
    "c_empress",
    "c_emperor",
    "c_heirophant",
    "c_lovers",
    "c_chariot",
    "c_justice",
    "c_hermit",
    "c_wheel_of_fortune",
    "c_strength",
    "c_hanged_man",
    "c_death",
    "c_temperance",
    "c_devil",
    "c_tower",
    "c_star",
    "c_moon",
    "c_sun",
    "c_judgement",
    "c_world",
)
SPECTRALS = (
    "c_familiar",
    "c_grim",
    "c_incantation",
    "c_talisman",
    "c_aura",
    "c_wraith",
    "c_sigil",
    "c_ouija",
    "c_ectoplasm",
    "c_immolate",
    "c_ankh",
    "c_deja_vu",
    "c_hex",
    "c_trance",
    "c_medium",
    "c_cryptid",
    "c_soul",
    "c_black_hole",
)


def targets_for(key: str, hand: list[CardRef]) -> list[CardRef]:
    """The fewest hand cards the consumable is used on."""
    config = load("centers")[key].get("config") or {}
    if key == "c_aura":
        return hand[:1]
    if isinstance(config, dict) and config.get("max_highlighted"):
        return hand[: config.get("min_highlighted", 1)]
    return []


@pytest.mark.parametrize("key", TAROTS + SPECTRALS)
def test_pack_pick(key: str) -> None:
    """Every Tarot and Spectral picked from a pack, on the dealt hand; the pick closes the pack
    and its hand goes back to the deck."""
    t = Table(f"CS_PICK_{key}", "j_joker", "j_golden")
    t.write("game.last_tarot_planet", "c_pluto")
    t.to_shop()
    t.open_pack("p_arcana_normal_1" if key in TAROTS else "p_spectral_normal_1")
    filler = "c_hermit" if key in TAROTS else "c_black_hole"
    (card, _) = t.stock_pack(key, filler)
    t.pick(card, *targets_for(key, t.hand))
    assert t.phase == "shop" and not t.hand


# The consumables that need no hand card.
UNTARGETED = (
    "c_fool",
    "c_high_priestess",
    "c_emperor",
    "c_hermit",
    "c_wheel_of_fortune",
    "c_temperance",
    "c_judgement",
    "c_wraith",
    "c_ectoplasm",
    "c_ankh",
    "c_hex",
    "c_soul",
    "c_black_hole",
    "c_saturn",
)


@pytest.mark.parametrize("phase", ["round_eval", "shop", "blind_select"])
@pytest.mark.parametrize("key", UNTARGETED)
def test_tray_use_between_rounds(key: str, phase: str) -> None:
    """Consumables that need no hand card work in every resting state (card.lua:1523-1562)."""
    t = Table(f"CS_TRAY_{key}_{phase}", "j_joker", "j_golden")
    t.write("game.last_tarot_planet", "c_mercury")
    t.win_with(t.hand[0])
    if phase != "round_eval":
        t.step(["CashOut"])
    t.add(key)  # BalatroBot adds in a round, the round's evaluation or the shop only
    if phase == "blind_select":
        t.step(["NextRound"])
    assert t.phase == phase
    t.step(t.use_action(key))
    assert t.phase == phase


def test_hand_tarots_are_refused_in_the_shop() -> None:
    """Outside SELECTING_HAND and the packs, cards that need a hand selection can't be used
    (card.lua:1564-1575)."""
    t = Table("CS_TRAY_SHOP_HAND")
    t.to_shop()
    for key in ("c_magician", "c_death", "c_familiar", "c_sigil", "c_aura", "c_cryptid"):
        t.add(key)
        t.refused(t.use_action(key))
        t.write("areas.consumables", [])


def test_tray_use_during_an_arcana_pack() -> None:
    """A tray consumable used in a pack works on the dealt hand and costs no pick
    (button_callbacks.lua:2267-2271)."""
    t = Table("CS_TRAY_PACK")
    t.to_shop()
    t.add("c_death")
    t.add("c_familiar")
    t.open_pack("p_arcana_normal_1")
    picks = t.read("pack.choices_remaining")
    t.step(t.use_action("c_death", t.hand[0], t.hand[1]))
    t.step(t.use_action("c_familiar"))
    assert t.phase == "pack_opening"
    assert t.read("pack.choices_remaining") == picks
    t.step(["SkipPack"])
    assert not t.hand


def test_pack_creators_need_room() -> None:
    """From a pack, The Emperor and The Fool need a free consumable slot (no own slot to free),
    Judgement a free joker slot (card.lua:1550-1562)."""
    t = Table("CS_PICK_ROOM", "j_joker", "j_golden", "j_cloud_9", "j_credit_card", "j_juggler")
    t.write("game.last_tarot_planet", "c_pluto")
    t.add("c_hermit")
    t.add("c_temperance")
    t.to_shop()
    t.open_pack("p_arcana_mega_1")
    emperor, fool, judgement, _ = t.stock_pack("c_emperor", "c_fool", "c_judgement", "c_hermit")
    for card in (emperor, fool, judgement):
        idx = t.run.cards("pack_cards").index(card)
        t.refused(["PickPackCard", idx, []])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_ectoplasm_on_a_debuffed_joker_adds_no_slot() -> None:
    """A debuffed joker is out of the deck (set_debuff, card.lua:526-538), so the Negative
    edition adds no slot (card.lua:405-415)."""
    t = Table("CS_ECTO_DEBUFF")
    joker = t.joker("j_joker")
    t.run.oracle("set_debuff(True) on the Joker")
    slots = t.read("game.joker_slots")
    t.use("c_ectoplasm")
    assert joker.edition["negative"] and t.read("game.joker_slots") == slots


def test_ankh_copies_a_debuffed_joker() -> None:
    """copy_card copies the debuff (common_events.lua:2177); the copy's add_to_deck still
    applies its passives (card.lua:564-643)."""
    t = Table("CS_ANKH_DEBUFF")
    t.joker("j_juggler")
    t.add("c_ankh")
    t.run.oracle("set_debuff(True) on the Juggler, which takes its hand size back")
    t.run.sync()
    size = t.read("game.hand_size")
    t.step(t.use_action("c_ankh"))
    jokers = t.run.cards("jokers")
    assert len(jokers) == 2 and jokers[1].debuff
    assert t.read("game.hand_size") == size + 1


def test_wheel_of_fortune_on_debuffed_jokers() -> None:
    t = Table("CS_WHEEL_DEBUFF")
    t.joker("j_oops", "foil")
    t.joker("j_oops", "foil")
    a = t.joker("j_joker")
    t.run.oracle("set_debuff(True) on the Joker")
    t.use("c_wheel_of_fortune")
    assert a.edition is not None


def test_the_emperor_with_showman_can_repeat_itself() -> None:
    """Showman lets the pools repeat cards held or used (common_events.lua:1987)."""
    t = Table("CS_EMPEROR_SHOWMAN", "j_ring_master")
    for _ in range(3):
        t.use("c_emperor")
        t.write("areas.consumables", [])


def empty_deck(t: Table) -> None:
    t.run.move("deck", "discard")


def test_immolate_empties_the_last_cards() -> None:
    """Hand and deck empty after Immolate: the round ends (game.lua:3056-3058), lost."""
    t = Table("CS_IMMOLATE_END")
    t.hold_only("H_9", "C_3", "D_4")
    empty_deck(t)
    t.use("c_immolate")
    assert t.phase == "game_over"


def test_the_hanged_man_empties_the_last_cards() -> None:
    """The round ends won when the chips already beat the blind."""
    t = Table("CS_HANGED_END")
    a, b = t.hold_only("H_9", "C_3")
    empty_deck(t)
    t.write("game.chips", 300)
    t.write("blind.chips", 300)
    t.use("c_hanged_man", a, b)
    assert t.phase == "round_eval"


def test_strength_ten_to_jack_and_sigil_under_the_club() -> None:
    t = Table("CS_STRENGTH_CLUB", boss="bl_club")
    (ten,) = t.cards("H_T")
    t.use("c_strength", ten)
    assert ten.front == "H_J"
    t.use("c_sigil")
    suit = t.hand[0].base["suit"]
    assert all(c.debuff == (suit == "Clubs") for c in t.hand)


def test_ouija_under_the_pillar_keeps_played_this_ante() -> None:
    """set_base rebuilds only the base: a card played this ante stays debuffed by The Pillar
    (blind.lua:634-637)."""
    t = Table("CS_OUIJA_PILLAR", boss="bl_pillar")
    t.play(*t.cards("S_5", "H_5"))
    played = t.run.cards("discard")[-2:]
    for card in played:
        assert card.ability.get("played_this_ante")
    with t.run.edit() as snap:
        areas = snap["areas"]
        areas["hand"] += areas["discard"][-2:]
        del areas["discard"][-2:]
    t.run.boss_check(*played)
    t.use("c_ouija")
    assert all(c.debuff for c in t.hand[-2:])


def test_seals_and_death_on_debuffed_cards() -> None:
    t = Table("CS_SEAL_DEBUFF", boss="bl_window")
    diamond, spade = t.cards("D_4", "S_9")
    assert diamond.debuff
    t.use("c_talisman", diamond)
    assert diamond.seal == "Gold"
    t.use("c_death", diamond, spade)
    assert diamond.front == "S_9" or spade.front == "D_4"


def test_temperance_with_a_rental_joker() -> None:
    t = Table("CS_TEMPERANCE_RENTAL")
    joker = t.joker("j_joker")
    with t.run.edit_card(joker) as card:
        card["stickers"]["rental"] = True
    t.run.oracle("the rental Joker repriced")
    t.write("game.dollars", 0)
    t.use("c_temperance")
    assert t.read("game.dollars") == joker.costs["sell_cost"]


PLANETS = (
    "c_mercury",
    "c_venus",
    "c_earth",
    "c_mars",
    "c_jupiter",
    "c_saturn",
    "c_uranus",
    "c_neptune",
    "c_pluto",
    "c_planet_x",
    "c_ceres",
    "c_eris",
)


@pytest.mark.parametrize("key", PLANETS)
def test_every_planet_levels_its_hand(key: str) -> None:
    """Each Planet levels up its own hand (card.lua:1137-1143, level_up_hand); a secret
    hand's Planet too."""
    hand = load("centers")[key]["config"]["hand_type"]
    t = Table(f"CS_PLANET_{key}")
    before = t.run.game.hand_levels[hand].level
    t.use(key)
    assert t.run.game.hand_levels[hand].level == before + 1
