"""Targeted parity cases for joker batch E (the shop and consumable contexts).

The seeded scenarios (``test_joker_scenarios.py``) never buy, sell, reroll, open or skip a pack,
skip a blind or use a consumable, so most branches of these jokers are reached only here. Each
case builds a run (BalatroBot ``add`` for the jokers and cards), writes the few values play
would take long to reach straight into the state, and takes its steps; each step must match the
oracle's, legal actions included (these jokers change slot counts, sell values and pack and shop
options; :mod:`tests.parity.cases`). The assertions after a step confirm that the branch under
test was taken.
"""

from __future__ import annotations

from typing import Any

from jackdaw.core import Score
from tests.parity.cases import CardRef, Run


class Board:
    """A targeted run whose steps are checked against the oracle's one at a time."""

    def __init__(
        self,
        seed: str,
        *jokers: str,
        boss: str | None = None,
        edition: str = "foil",
        challenge: str | None = None,
    ):
        setup: list[dict[str, Any]] = []
        if boss is not None:
            setup.append({"op": "force_boss", "key": boss})
        setup.append({"op": "action", "action": ["SelectBlind"]})
        # A fixed edition keeps the board's slots and mult predictable (foil adds chips only).
        setup += [{"op": "add", "key": key, "edition": edition} for key in jokers]
        back = "b_challenge" if challenge else "b_red"
        start = {"back": back, "stake": 1, "seed": seed, "challenge": challenge}
        self.seed = seed
        self.run = Run(start, setup)
        self.out_of_reach()

    @property
    def phase(self) -> str:
        return self.run.game.phase

    def out_of_reach(self) -> None:
        """A hand ends the round only when a case wins it on purpose."""
        self.run.write("blind.chips", 10**9)

    def add(self, key: str, **params: Any) -> CardRef:
        """BalatroBot ``add``: a joker, a consumable, or a playing card (into the hand during a
        round, else the deck). A playing card in the hand meets the boss as a dealt one did."""
        card = self.run.add(key, **params)
        if card.area == "hand":
            self.run.boss_check(card)
        return card

    def cards(self, *keys: str, **params: Any) -> list[CardRef]:
        return [self.add(k, **params) for k in keys]

    def joker(self, key: str, n: int = 0) -> CardRef:
        return [j for j in self.run.cards("jokers") if j.key == key][n]

    def keys(self, area: str) -> list[str]:
        return self.run.keys(area)

    def step(self, action: list[Any]) -> None:
        assert self.run.step(action), f"{self.seed}: the oracle refused {action}"

    # -- actions ------------------------------------------------------------

    def play(self, *cards: CardRef) -> None:
        self.step(["PlayHand", self.run.indices(cards)])

    def play_pair(self) -> None:
        self.play(*self.cards("S_5", "H_5"))

    def win_with(self, *cards: CardRef) -> None:
        """Play the cards with the blind one chip from beaten."""
        self.run.write("game.chips", self.run.read("blind.chips") - 1)
        self.play(*cards)
        assert self.phase == "round_eval"

    def to_shop(self, dollars: int = 100) -> None:
        """Win the round with a pair, cash out, and hold ``dollars`` in the shop."""
        self.win_with(*self.cards("S_5", "H_5"))
        self.step(["CashOut"])
        assert self.phase == "shop"
        self.run.write("game.dollars", dollars)

    def next_blind(self) -> None:
        """Leave the shop and select the blind on deck."""
        self.step(["NextRound"])
        assert self.phase == "blind_select"
        self.step(["SelectBlind"])
        self.out_of_reach()

    def sell(self, card: CardRef) -> None:
        for area in ("jokers", "consumables"):
            cards = self.run.cards(area)
            if card in cards:
                self.step(["SellCard", area, cards.index(card)])
                return
        raise AssertionError("the card is not held")

    def use(self, card: CardRef) -> None:
        self.step(["UseConsumable", self.run.cards("consumables").index(card), None])

    def reprice(self, card: CardRef) -> None:
        """``card:set_cost()`` with the run's prices."""
        self.run.oracle(f"card {card.sort_id} repriced")

    def set_booster(self, index: int, key: str) -> None:
        """Make the shop's booster at ``index`` a ``key`` pack, as populate_shop builds one."""
        self.run.oracle(f"shop booster {index} made a {key}")

    def shop_rates(self, kind: str) -> None:
        """Before the shop is built: every shop card is of ``kind`` (a ``*_rate`` key)."""
        with self.run.edit() as snap:
            game = snap["game"]
            for rate in ("joker_rate", "tarot_rate", "planet_rate", "spectral_rate"):
                game[rate] = 0
            game["playing_card_rate"] = 0
            game[kind] = 4

    def open_booster(self, key: str | None = None, index: int = 0) -> None:
        if key is not None:
            self.set_booster(index, key)
        self.step(["OpenBooster", index])
        assert self.phase == "pack_opening"

    @property
    def last(self) -> Score:
        score = self.run.game.last_score
        assert score is not None
        return score

    def double_tags(self) -> int:
        return sum(1 for t in self.run.read("tags") if t["key"] == "tag_double")


def negative(card: CardRef) -> bool:
    return bool(card.edition and card.edition.get("negative"))


# ---------------------------------------------------------------------------
# Rerolls, skipped packs and skipped blinds
# ---------------------------------------------------------------------------


def test_flash_card_grows_with_each_reroll() -> None:
    """Flash Card: +2 Mult per reroll (card.lua:2403-2411), not from a Blueprint copy; both
    score it in the main step (card.lua:3998-4003)."""
    b = Board("BE_FLASH", "j_blueprint", "j_flash")
    b.to_shop()
    b.step(["Reroll"])
    b.step(["Reroll"])
    assert b.joker("j_flash").ability["mult"] == 4
    b.next_blind()
    b.play_pair()
    assert b.last.mult == 2 + 4 + 4


def test_flash_card_counts_a_free_reroll() -> None:
    """Chaos the Clown's free reroll is still a reroll (button_callbacks.lua:2855-2909)."""
    b = Board("BE_FLASH_FREE", "j_chaos", "j_flash")
    b.to_shop(dollars=0)
    assert b.run.read("current_round.reroll_cost") == 0
    b.step(["Reroll"])
    assert b.joker("j_flash").ability["mult"] == 2


def test_flash_card_scores_nothing_before_a_reroll() -> None:
    b = Board("BE_FLASH_ZERO", "j_flash", "j_brainstorm")
    b.play_pair()
    assert b.last.mult == 2


def test_red_card_grows_when_a_pack_is_skipped() -> None:
    """Red Card: +3 Mult per booster pack skipped (card.lua:2441-2455), not from a Blueprint
    copy, nor for a pack whose cards were picked."""
    b = Board("BE_RED_CARD", "j_blueprint", "j_red_card")
    b.to_shop()
    b.open_booster("p_buffoon_normal_1")
    b.step(["SkipPack"])
    assert b.joker("j_red_card").ability["mult"] == 3
    b.open_booster("p_celestial_normal_1")
    b.step(["PickPackCard", 0, None])
    assert b.phase == "shop"
    assert b.joker("j_red_card").ability["mult"] == 3
    b.next_blind()
    b.play_pair()
    assert b.last.mult == 2 + 3 + 3


def test_red_card_skipping_an_arcana_pack() -> None:
    """An Arcana pack deals a hand first (game.lua:3345-3388); skipping it counts too."""
    b = Board("BE_RED_ARCANA", "j_red_card", "j_brainstorm")
    b.to_shop()
    b.open_booster("p_arcana_normal_1")
    assert b.run.cards("hand")
    b.step(["SkipPack"])
    assert b.joker("j_red_card").ability["mult"] == 3


def test_throwback_counts_the_skipped_blinds() -> None:
    """Throwback: Card:update keeps x_mult at 1 + 0.25 per blind skipped this run
    (card.lua:4176-4178); the shared x_mult rule scores it, Blueprint's copy too. Its
    skip_blind branch only shows the value (card.lua:2428-2440)."""
    b = Board("BE_THROWBACK", "j_blueprint", "j_throwback")
    b.play_pair()
    assert b.last.mult == 2
    b.to_shop()
    b.step(["NextRound"])
    b.step(["SkipBlind"])
    if b.phase == "pack_opening":
        b.step(["SkipPack"])
    assert b.run.read("game.skips") == 1
    b.step(["SelectBlind"])
    b.out_of_reach()
    b.play_pair()
    assert b.last.mult == 2 * 1.25 * 1.25


# ---------------------------------------------------------------------------
# Selling
# ---------------------------------------------------------------------------


def test_campfire_grows_for_each_card_sold() -> None:
    """Campfire: +0.25 XMult for every other card sold, jokers and consumables alike
    (card.lua:2396-2402); Blueprint copies the XMult but not the growth."""
    b = Board("BE_CAMPFIRE", "j_blueprint", "j_campfire", "j_golden")
    b.sell(b.joker("j_golden"))
    pluto = b.add("c_pluto")
    b.sell(pluto)
    assert b.joker("j_campfire").ability["x_mult"] == 1.5
    b.play_pair()
    assert b.last.mult == 2 * 1.5 * 1.5


def test_campfire_sees_sales_in_the_shop_and_blind_select() -> None:
    b = Board("BE_CAMPFIRE_SHOP", "j_campfire", "j_campfire", "j_golden", "j_credit_card")
    b.to_shop()
    b.sell(b.joker("j_golden"))
    b.step(["NextRound"])
    b.sell(b.joker("j_campfire", 1))
    assert b.joker("j_campfire").ability["x_mult"] == 1.5


def test_campfire_resets_after_a_boss() -> None:
    """At the end of a boss round Campfire returns to X1 (card.lua:2889-2895)."""
    b = Board("BE_CAMPFIRE_BOSS", "j_campfire", "j_brainstorm", boss="bl_club")
    with b.run.edit_card(b.joker("j_campfire")) as card:
        card["ability"]["x_mult"] = 2
    b.win_with(*b.cards("S_5", "H_5"))
    assert b.joker("j_campfire").ability["x_mult"] == 1


def test_campfire_keeps_its_xmult_after_a_small_blind() -> None:
    b = Board("BE_CAMPFIRE_SMALL", "j_campfire")
    with b.run.edit_card(b.joker("j_campfire")) as card:
        card["ability"]["x_mult"] = 2
    b.win_with(*b.cards("S_5", "H_5"))
    assert b.joker("j_campfire").ability["x_mult"] == 2


def test_campfire_resets_after_a_disabled_boss() -> None:
    """G.GAME.blind.boss stays set when the boss is disabled (blind.lua:356-415)."""
    b = Board("BE_CAMPFIRE_LUCHADOR", "j_campfire", "j_luchador", boss="bl_club")
    b.sell(b.joker("j_luchador"))
    assert b.run.read("blind.disabled")
    assert b.joker("j_campfire").ability["x_mult"] == 1.25
    b.win_with(*b.cards("S_5", "H_5"))
    assert b.joker("j_campfire").ability["x_mult"] == 1


def test_luchador_disables_the_wall() -> None:
    """Luchador: sold during an active boss, G.GAME.blind:disable() (card.lua:2355-2360); The
    Wall's target is halved (blind.lua:377-380)."""
    b = Board("BE_LUCHADOR_WALL", "j_luchador", boss="bl_wall")
    target = b.run.read("blind.chips")
    b.sell(b.joker("j_luchador"))
    assert b.run.read("blind.disabled")
    assert b.run.read("blind.chips") == target / 2


def test_luchador_ends_the_round_the_disabled_wall_lost() -> None:
    """Disabled, a boss whose new target the score already meets ends the round
    (blind.lua:397-406). The sale's money lands first: at $4 it is what earns interest."""
    b = Board("BE_LUCHADOR_WALL_WIN", "j_luchador", "j_golden", boss="bl_wall")
    b.run.write("blind.chips", 2000)
    b.run.write("game.chips", 1200)
    b.run.write("game.dollars", 4)
    b.sell(b.joker("j_luchador"))
    assert b.phase == "round_eval"


def test_a_sale_that_ends_the_round_follows_the_event_order() -> None:
    """G161: the selling_card loop runs first (Campfire grows, then end_round resets it after
    the boss), end_round still finds the sold rental Luchador (its rent is charged), and the
    payout lands before the interest is counted (blind.lua:397-406, card.lua:1601-1637,
    button_callbacks.lua:2318-2326)."""
    b = Board("BE_LUCHADOR_WALL_RENTAL", "j_campfire", "j_luchador", boss="bl_wall")
    luchador = b.joker("j_luchador")
    with b.run.edit_card(luchador) as card:
        card["stickers"]["rental"] = True
    b.reprice(luchador)
    b.run.write("blind.chips", 2000)
    b.run.write("game.chips", 1200)
    b.run.write("game.dollars", 6)
    b.sell(luchador)
    assert b.phase == "round_eval"
    assert b.keys("jokers") == ["j_campfire"]
    assert b.joker("j_campfire").ability["x_mult"] == 1
    assert b.run.read("game.dollars") == 6 + 1 - 3
    b.step(["CashOut"])


def test_luchador_gives_back_the_needles_hands() -> None:
    b = Board("BE_LUCHADOR_NEEDLE", "j_luchador", boss="bl_needle")
    assert b.run.read("current_round.hands_left") == 1
    b.sell(b.joker("j_luchador"))
    assert b.run.read("current_round.hands_left") == 4


def test_luchador_frees_the_manacle() -> None:
    """The Manacle disabled: +1 hand size, whose change_size draws, then one more card drawn
    (blind.lua:386-390, cardarea.lua:100-105)."""
    b = Board("BE_LUCHADOR_MANACLE", "j_luchador", boss="bl_manacle")
    held = len(b.run.cards("hand"))
    b.sell(b.joker("j_luchador"))
    assert b.run.read("blind.disabled")
    assert len(b.run.cards("hand")) > held


def test_luchador_lifts_the_plants_debuffs() -> None:
    b = Board("BE_LUCHADOR_PLANT", "j_luchador", boss="bl_plant")
    king = b.add("S_K")
    assert king.debuff
    b.sell(b.joker("j_luchador"))
    assert not king.debuff


def test_a_sold_blueprint_copying_luchador_disables_the_boss() -> None:
    """selling_self has no Blueprint guard: the copy disables the boss (card.lua:2354-2360)."""
    b = Board("BE_LUCHADOR_COPY", "j_blueprint", "j_luchador", boss="bl_water")
    assert b.run.read("current_round.discards_left") == 0
    b.sell(b.joker("j_blueprint"))
    assert b.run.read("blind.disabled")
    assert b.run.read("current_round.discards_left") == b.run.read("round_resets.discards")


def test_luchador_outside_a_boss_disables_nothing() -> None:
    """A Small blind is not a boss (Blind:get_type, blind.lua:346-354)."""
    b = Board("BE_LUCHADOR_SMALL", "j_luchador")
    b.sell(b.joker("j_luchador"))
    assert not b.run.read("blind.disabled")


def test_luchador_sold_in_the_shop_after_a_boss_disables_nothing() -> None:
    """The defeated boss was reset to the empty blind (G12), whose type is no boss."""
    b = Board("BE_LUCHADOR_SHOP", "j_luchador", "j_golden", boss="bl_manacle")
    b.to_shop()
    size = b.run.read("game.hand_size")
    b.sell(b.joker("j_luchador"))
    assert b.run.read("game.hand_size") == size


def test_a_debuffed_luchador_disables_nothing() -> None:
    """After a hand Crimson Heart debuffs a joker, the only one here (blind.lua:588-602); a
    debuffed card's calculate_joker returns at once (card.lua:2292)."""
    b = Board("BE_LUCHADOR_HEART", "j_luchador", boss="bl_final_heart")
    b.play_pair()
    luchador = b.joker("j_luchador")
    assert luchador.debuff
    b.sell(luchador)
    assert not b.run.read("blind.disabled")


def test_luchador_under_verdant_leaf() -> None:
    """Luchador disables Verdant Leaf in selling_self; selling a joker under it queues another
    disable (card.lua:1616-1620), which changes nothing more."""
    b = Board("BE_LUCHADOR_VERDANT", "j_luchador", "j_golden", boss="bl_final_leaf")
    card = b.add("S_5")
    assert card.debuff
    b.sell(b.joker("j_luchador"))
    assert b.run.read("blind.disabled")
    assert not card.debuff


def test_luchador_after_the_boss_was_disabled() -> None:
    """A disabled boss is not disabled again."""
    b = Board("BE_LUCHADOR_TWICE", "j_luchador", "j_luchador", boss="bl_wall")
    target = b.run.read("blind.chips")
    b.sell(b.joker("j_luchador"))
    b.sell(b.joker("j_luchador"))
    assert b.run.read("blind.chips") == target / 2


def test_diet_cola_sold_gives_a_double_tag() -> None:
    """Diet Cola: sold, an event adds a Double Tag (card.lua:2361-2370); it then doubles the
    next skip's tag (tag.lua:319-333)."""
    b = Board("BE_DIET_COLA", "j_diet_cola", "j_golden")
    b.to_shop()
    b.sell(b.joker("j_diet_cola"))
    assert b.double_tags() == 1
    b.step(["NextRound"])
    b.step(["SkipBlind"])
    assert b.double_tags() == 0


def test_diet_colas_double_tag_doubles_a_pack_tag() -> None:
    """G106: the Double Tag's copy of a Buffoon Tag waits while the first pack is open
    (button_callbacks.lua:2775-2777) and opens its pack when that one closes (2612-2619). Tag
    packs run open_booster (G43): Hallucination rolls for each; Red Card counts each skip."""
    b = Board("BE_DIET_COLA_PACKS", "j_diet_cola", "j_red_card", "j_oops", "j_hallucination")
    b.to_shop()
    b.sell(b.joker("j_diet_cola"))
    b.step(["NextRound"])
    b.run.write("round_resets.blind_tags.Big", "tag_buffoon")
    b.step(["SkipBlind"])
    assert b.phase == "pack_opening"
    b.step(["SkipPack"])
    assert b.phase == "pack_opening"
    b.step(["SkipPack"])
    assert b.phase == "blind_select"
    assert b.joker("j_red_card").ability["mult"] == 6
    assert [c.set for c in b.run.cards("consumables")] == ["Tarot", "Tarot"]


def test_diet_colas_double_tag_doubles_a_boss_tag() -> None:
    """Each Boss Tag rerolls through G.FUNCS.reroll_boss, whose event runs the next pass
    (button_callbacks.lua:2847-2849)."""
    b = Board("BE_DIET_COLA_BOSS", "j_diet_cola")
    b.to_shop()
    b.sell(b.joker("j_diet_cola"))
    b.step(["NextRound"])
    b.run.write("round_resets.blind_tags.Big", "tag_boss")
    b.step(["SkipBlind"])
    assert b.run.read("round_resets.boss_rerolled")
    assert b.double_tags() == 0


def test_a_sold_brainstorm_copying_diet_cola_gives_a_double_tag() -> None:
    """selling_self has no Blueprint guard; a held Double Tag never doubles another
    (tag.lua:320)."""
    b = Board("BE_DIET_COLA_COPY", "j_diet_cola", "j_brainstorm")
    b.sell(b.joker("j_brainstorm"))
    b.sell(b.joker("j_diet_cola"))
    assert b.double_tags() == 2


def test_invisible_joker_counts_its_rounds() -> None:
    """Invisible Joker: +1 invis_rounds at the end of each round (card.lua:2934-2944); a
    Blueprint copy counts nothing."""
    b = Board("BE_INVISIBLE_ROUNDS", "j_blueprint", "j_invisible")
    b.win_with(*b.cards("S_5", "H_5"))
    assert b.joker("j_invisible").ability["invis_rounds"] == 1


def _invis_rounds(b: Board, rounds: int, n: int = 0) -> None:
    with b.run.edit_card(b.joker("j_invisible", n)) as card:
        card["ability"]["invis_rounds"] = rounds


def test_invisible_joker_sold_too_early_copies_nothing() -> None:
    b = Board("BE_INVISIBLE_EARLY", "j_invisible", "j_golden")
    _invis_rounds(b, 1)
    b.sell(b.joker("j_invisible"))
    assert b.keys("jokers") == ["j_golden"]


def test_invisible_joker_copies_another_joker() -> None:
    """Sold after two rounds, it copies a random other joker at once (card.lua:2371-2394); a
    Negative joker's copy loses the edition (G28)."""
    b = Board("BE_INVISIBLE_COPY", "j_invisible", "j_golden", "j_credit_card")
    _invis_rounds(b, 2)
    b.sell(b.joker("j_invisible"))
    assert len(b.run.cards("jokers")) == 3


def test_invisible_joker_copy_of_a_negative_joker_drops_the_edition() -> None:
    b = Board("BE_INVISIBLE_NEGATIVE", "j_invisible")
    b.add("j_golden", edition="negative")
    _invis_rounds(b, 2)
    b.sell(b.joker("j_invisible"))
    assert [negative(j) for j in b.run.cards("jokers")] == [True, False]


def test_invisible_joker_copy_sees_the_sale() -> None:
    """The copy joins G.jokers during selling_self, so a copied Campfire sees the sale in the
    selling_card loop that follows (button_callbacks.lua:2318-2326)."""
    b = Board("BE_INVISIBLE_CAMPFIRE", "j_invisible", "j_campfire")
    _invis_rounds(b, 2)
    b.sell(b.joker("j_invisible"))
    assert [j.ability["x_mult"] for j in b.run.cards("jokers")] == [1.25, 1.25]


def test_invisible_joker_copy_of_an_invisible_joker_starts_over() -> None:
    b = Board("BE_INVISIBLE_TWO", "j_invisible", "j_invisible")
    first = b.joker("j_invisible")
    _invis_rounds(b, 2)
    _invis_rounds(b, 3, n=1)
    b.sell(first)
    assert [j.ability["invis_rounds"] for j in b.run.cards("jokers")] == [3, 0]


def test_invisible_joker_copy_joins_the_deck() -> None:
    """The copy runs add_to_deck: a copied Juggler's +1 hand size draws a card mid-round
    (card.lua:586-588, cardarea.lua:94-111)."""
    b = Board("BE_INVISIBLE_JUGGLER", "j_invisible", "j_juggler")
    _invis_rounds(b, 2)
    held, size = len(b.run.cards("hand")), b.run.read("game.hand_size")
    b.sell(b.joker("j_invisible"))
    assert b.keys("jokers") == ["j_juggler", "j_juggler"]
    assert b.run.read("game.hand_size") == size + 1
    assert len(b.run.cards("hand")) == held + 1


def test_invisible_joker_copies_a_debuffed_joker() -> None:
    """Crimson Heart debuffed the Juggler, which took its hand size back (card.lua:535). The
    copy keeps the debuff (common_events.lua:2178), yet its add_to_deck gives the hand size
    (card.lua:564-643 checks no debuff)."""
    b = Board("BE_INVISIBLE_HEART_1", "j_invisible", "j_juggler", boss="bl_final_heart")
    b.play_pair()
    assert [j.key for j in b.run.cards("jokers") if j.debuff] == ["j_juggler"]
    _invis_rounds(b, 2)
    size = b.run.read("game.hand_size")
    b.sell(b.joker("j_invisible"))
    copy = b.joker("j_juggler", 1)
    assert copy.debuff
    assert b.run.read("game.hand_size") == size + 1


def test_invisible_joker_alone_copies_nothing() -> None:
    b = Board("BE_INVISIBLE_ALONE", "j_invisible")
    _invis_rounds(b, 2)
    b.sell(b.joker("j_invisible"))
    assert not b.run.cards("jokers")


def test_invisible_joker_needs_the_jokers_to_fit() -> None:
    """``#G.jokers.cards <= G.jokers.config.card_limit``, Invisible Joker included."""
    b = Board("BE_INVISIBLE_FULL", "j_invisible", "j_golden", "j_credit_card")
    _invis_rounds(b, 2)
    b.run.write("game.joker_slots", 2)
    b.sell(b.joker("j_invisible"))
    assert b.keys("jokers") == ["j_golden", "j_credit_card"]


def test_a_sold_blueprint_copying_invisible_joker_copies_nothing() -> None:
    b = Board("BE_INVISIBLE_BLUEPRINT", "j_blueprint", "j_invisible", "j_golden")
    _invis_rounds(b, 2)
    b.sell(b.joker("j_blueprint"))
    assert b.keys("jokers") == ["j_invisible", "j_golden"]


# ---------------------------------------------------------------------------
# Packs and the end of the shop
# ---------------------------------------------------------------------------


def test_hallucination_makes_a_tarot_when_a_pack_opens() -> None:
    """Hallucination: a 1 in 2 Tarot when a pack opens (card.lua:2335-2351), two Oops! All 6s
    make it certain; the Tarot comes after the pack's cards (G266)."""
    b = Board("BE_HALLUCINATION", "j_oops", "j_hallucination")
    b.to_shop()
    b.open_booster("p_buffoon_normal_1")
    assert [c.set for c in b.run.cards("consumables")] == ["Tarot"]
    tarot = b.run.cards("consumables")[0]
    assert tarot.sort_id > max(c.sort_id for c in b.run.cards("pack_cards"))


def test_hallucination_rolls_for_each_copy() -> None:
    """No Blueprint guard: each copy rolls 'halu' and reserves a slot."""
    b = Board("BE_HALLUCINATION_COPIES", "j_oops", "j_blueprint", "j_hallucination")
    b.to_shop()
    b.open_booster("p_standard_normal_1")
    assert [c.set for c in b.run.cards("consumables")] == ["Tarot", "Tarot"]


def test_hallucination_under_inflation() -> None:
    """Inflation: Card:open raises G.GAME.inflation after the open_booster contexts and queues
    a repricing (card.lua:1796-1807); the Tarot's event runs after the contents, with the
    raised value."""
    b = Board("BE_HALLUCINATION_INFLATION", "j_oops", "j_hallucination", challenge="c_inflation_1")
    b.to_shop()
    inflation = b.run.read("game.inflation")
    b.open_booster("p_buffoon_normal_1")
    assert b.run.read("game.inflation") == inflation + 1
    assert [c.set for c in b.run.cards("consumables")] == ["Tarot"]


def test_hallucination_copies_share_the_last_slot() -> None:
    """G22: the copy's room check sees the first hit's reservation in
    G.GAME.consumeable_buffer, so it does not roll."""
    b = Board("BE_HALLUCINATION_BUFFER", "j_hallucination", "j_oops", "j_brainstorm")
    b.to_shop()
    b.add("c_pluto")
    b.open_booster("p_celestial_normal_1")
    assert len(b.run.cards("consumables")) == 2
    assert "halu1" in b.run.read("rng.streams")


def test_hallucination_with_full_slots_rolls_nothing() -> None:
    b = Board("BE_HALLUCINATION_FULL", "j_hallucination", "j_blueprint")
    b.to_shop()
    b.add("c_pluto")
    b.add("c_mercury")
    b.open_booster("p_arcana_normal_1")
    assert "halu1" not in b.run.read("rng.streams")


def test_hallucination_rolls_with_the_odds() -> None:
    """Without Oops! All 6s the roll can miss: two packs, whatever they roll."""
    b = Board("BE_HALLUCINATION_ODDS", "j_hallucination", "j_blueprint")
    with b.run.edit() as snap:
        snap["areas"]["jokers"].reverse()  # Blueprint copies Hallucination
    b.to_shop()
    for _ in range(2):
        b.open_booster("p_buffoon_normal_1")
        b.step(["SkipPack"])
    assert b.run.read("rng.streams")["halu1"]


def test_perkeo_copies_a_consumable_when_the_shop_ends() -> None:
    """Perkeo: leaving the shop, an event copies a random consumable as a Negative
    (card.lua:2412-2426): one more consumable slot (card.lua:630-639)."""
    b = Board("BE_PERKEO", "j_perkeo")
    b.to_shop()
    b.add("c_pluto")
    b.add("c_mars")
    b.step(["NextRound"])
    assert len(b.run.cards("consumables")) == 3
    assert negative(b.run.cards("consumables")[2])
    assert b.run.read("game.consumable_slots") == 3


def test_perkeo_without_a_consumable_copies_nothing() -> None:
    b = Board("BE_PERKEO_EMPTY", "j_perkeo", "j_blueprint")
    b.to_shop()
    b.step(["NextRound"])
    assert not b.run.cards("consumables")


def test_perkeo_and_its_copies_each_copy() -> None:
    """G16: no Blueprint guard. Each copy queues its own event, and a later event may pick an
    earlier one's card."""
    b = Board("BE_PERKEO_COPIES", "j_blueprint", "j_perkeo", "j_brainstorm")
    b.to_shop()
    b.add("c_pluto")
    b.step(["NextRound"])
    assert len(b.run.cards("consumables")) == 4
    assert b.run.read("game.consumable_slots") == 5


def test_perkeo_copies_a_negative_consumable() -> None:
    """Over two shops: the first copy is Negative, and the second shop's pick may take it; the
    copy is Negative either way."""
    b = Board("BE_PERKEO_TWICE", "j_perkeo")
    b.to_shop()
    b.add("c_pluto")
    b.step(["NextRound"])
    b.step(["SelectBlind"])
    b.out_of_reach()
    b.to_shop()
    b.step(["NextRound"])
    assert len(b.run.cards("consumables")) == 3
    assert b.run.read("game.consumable_slots") == 4


# ---------------------------------------------------------------------------
# Consumables and playing cards
# ---------------------------------------------------------------------------


def test_constellation_grows_with_each_planet() -> None:
    """Constellation: +0.1 XMult per Planet used (card.lua:2727-2733), not for Black Hole (a
    Spectral); Blueprint copies the XMult but not the growth."""
    b = Board("BE_CONSTELLATION", "j_blueprint", "j_constellation")
    b.use(b.add("c_pluto"))
    b.use(b.add("c_black_hole"))
    assert b.joker("j_constellation").ability["x_mult"] == 1.1
    b.play_pair()


def test_constellation_counts_a_planet_bought_and_used() -> None:
    """Buy and use goes through G.FUNCS.use_card (button_callbacks.lua:2472-2474)."""
    b = Board("BE_CONSTELLATION_BUY", "j_constellation", "j_blueprint")
    b.shop_rates("planet_rate")
    b.to_shop()
    b.step(["BuyAndUse", 0, None])
    assert b.joker("j_constellation").ability["x_mult"] == 1.1


def test_constellation_counts_a_planet_picked_from_a_pack() -> None:
    b = Board("BE_CONSTELLATION_PACK", "j_constellation", "j_brainstorm")
    b.to_shop()
    b.open_booster("p_celestial_mega_1")
    b.step(["PickPackCard", 0, None])
    b.step(["PickPackCard", 0, None])
    assert b.joker("j_constellation").ability["x_mult"] == 1 + 0.1 + 0.1


def test_fortune_teller_counts_the_tarots_used() -> None:
    """Fortune Teller: +Mult equal to consumeable_usage_total.tarot (card.lua:4016-4021),
    nothing while it is nil or 0; Blueprint copies it."""
    b = Board("BE_FORTUNE", "j_blueprint", "j_fortune_teller")
    b.play_pair()
    assert b.last.mult == 2
    b.use(b.add("c_pluto"))
    assert b.run.read("consumeable_usage.totals.tarot") == 0
    b.play_pair()
    assert b.last.mult == 2
    b.run.write("consumeable_usage.totals.tarot", 3)
    b.play_pair()
    assert b.last.mult == 2 + 3 + 3


def test_hologram_grows_for_each_card_bought() -> None:
    """Hologram: +0.25 XMult per playing card added to the deck (card.lua:2456-2461), here a
    shop purchase (button_callbacks.lua:2424-2430); Blueprint copies the XMult."""
    b = Board("BE_HOLOGRAM", "j_blueprint", "j_hologram")
    b.shop_rates("playing_card_rate")
    b.to_shop()
    assert all(c.set in ("Default", "Enhanced") for c in b.run.cards("shop_jokers"))
    b.step(["BuyCard", 0])
    b.step(["BuyCard", 0])
    assert b.joker("j_hologram").ability["x_mult"] == 1.5
    b.next_blind()
    b.play_pair()
    assert b.last.mult == 2 * 1.5 * 1.5


def test_hologram_counts_a_standard_pack_pick() -> None:
    b = Board("BE_HOLOGRAM_PACK", "j_hologram", "j_brainstorm")
    b.to_shop()
    b.open_booster("p_standard_mega_1")
    b.step(["PickPackCard", 0, None])
    b.step(["PickPackCard", 0, None])
    assert b.joker("j_hologram").ability["x_mult"] == 1.5
