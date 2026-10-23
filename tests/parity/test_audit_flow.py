"""Parity cases for the blind, round-flow and challenge-rule rows of the ledger audit.

Each case builds a state and takes the steps that exercise one ledger row; each step must
match the oracle's (:mod:`tests.parity.cases`), and the assertions on the state afterwards
show that the branch under test ran. Line references are into ``balatro_source/``.
"""

from __future__ import annotations

from typing import Any

from tests.parity.cases import Run


def _start(seed: str, challenge: str | None = None) -> dict[str, Any]:
    back = "b_challenge" if challenge else "b_red"
    return {"back": back, "stake": 1, "seed": seed, "challenge": challenge}


def in_round(
    seed: str,
    jokers: tuple[str, ...] = (),
    *,
    boss: str | None = None,
    challenge: str | None = None,
    **sets: Any,
) -> Run:
    """SELECTING_HAND of the first blind (or of ``boss``), with ``jokers`` added
    by BalatroBot's ``add``."""
    setup: list[dict[str, Any]] = []
    if boss:
        setup.append({"op": "force_boss", "key": boss})
    setup.append({"op": "action", "action": ["SelectBlind"]})
    setup += [{"op": "add", "key": k} for k in jokers]
    if sets:
        setup.append({"op": "set", **sets})
    return Run(_start(seed, challenge), setup)


def at_blind_select(
    seed: str,
    jokers: tuple[str, ...] = (),
    *,
    boss: str | None = None,
    challenge: str | None = None,
) -> Run:
    """The Big Blind's (or ``boss``'s) blind select, with ``jokers`` bought in the
    first shop: their ``setting_blind`` runs in the next, checked, SelectBlind."""
    setup: list[dict[str, Any]] = [
        {"op": "action", "action": ["SelectBlind"]},
        {"op": "set", "chips": 10**9},
        {"op": "action", "action": ["PlayHand", [0]]},
        {"op": "action", "action": ["CashOut"]},
    ]
    setup += [{"op": "add", "key": k} for k in jokers]
    setup.append({"op": "action", "action": ["NextRound"]})
    if boss:
        setup.append({"op": "force_boss", "key": boss})
    return Run(_start(seed, challenge), setup)


def check(run: Run, *actions: list[Any]) -> dict[str, Any]:
    """Take ``actions``, each where the oracle took it. Returns the snapshot afterwards."""
    for action in actions:
        assert run.step(action), action
    return run.snapshot()


def jokers_of(snap: dict[str, Any]) -> list[str]:
    return [j["center"] for j in snap["areas"]["jokers"]]


# ---------------------------------------------------------------------------
# Blinds (blind.lua)
# ---------------------------------------------------------------------------


class TestTheSerpent:
    """state_events.lua:363-367: after the round's first play or discard, The
    Serpent draws min(#deck, 3) cards whatever the hand size."""

    def test_draws_three_after_a_discard(self):
        run = in_round("AU_SERPENT_D", boss="bl_serpent")
        before = len(run.view("hand"))
        snap = check(run, ["Discard", [0]])
        assert len(snap["areas"]["hand"]) == before - 1 + 3

    def test_draws_three_after_a_play(self):
        run = in_round("AU_SERPENT_P", boss="bl_serpent")
        before = len(run.view("hand"))
        snap = check(run, ["PlayHand", [0, 1]])
        assert snap["phase"] == "selecting_hand"
        assert len(snap["areas"]["hand"]) == before - 2 + 3

    def test_a_disabled_serpent_fills_the_hand(self):
        run = at_blind_select("AU_SERPENT_CHICOT", ("j_chicot",), boss="bl_serpent")
        snap = check(run, ["SelectBlind"], ["Discard", [0, 1, 2, 3]])
        assert snap["blind"]["disabled"]
        assert len(snap["areas"]["hand"]) == snap["game"]["hand_size"]


class TestDisable:
    """Blind:disable (blind.lua:356-415) through Chicot's setting_blind."""

    def test_the_needle_gives_the_hands_back(self):
        run = at_blind_select("AU_NEEDLE", ("j_chicot",), boss="bl_needle")
        snap = check(run, ["SelectBlind"])
        assert snap["blind"]["disabled"]
        assert snap["current_round"]["hands_left"] == snap["round_resets"]["hands"]

    def test_violet_vessel_is_divided_by_three(self):
        target = check(at_blind_select("AU_VESSEL", boss="bl_final_vessel"), ["SelectBlind"])
        run = at_blind_select("AU_VESSEL", ("j_chicot",), boss="bl_final_vessel")
        snap = check(run, ["SelectBlind"])
        assert snap["blind"]["chips"] == target["blind"]["chips"] / 3

    def test_cerulean_bell_forces_nothing(self):
        run = at_blind_select("AU_BELL", ("j_chicot",), boss="bl_final_bell")
        snap = check(run, ["SelectBlind"], ["Discard", [0]])
        assert not any(c["ability"].get("forced_selection") for c in snap["areas"]["hand"])

    def test_crimson_heart_debuffs_no_joker(self):
        run = at_blind_select("AU_HEART", ("j_chicot", "j_joker"), boss="bl_final_heart")
        snap = check(run, ["SelectBlind"], ["PlayHand", [0]])
        assert not any(j["debuff"] for j in snap["areas"]["jokers"])

    def test_face_down_bosses_deal_face_up(self):
        for boss in ("bl_wheel", "bl_house", "bl_mark", "bl_fish"):
            run = at_blind_select(f"AU_FLIP_{boss}", ("j_chicot",), boss=boss)
            snap = check(run, ["SelectBlind"], ["PlayHand", [0]])
            assert all(c["facing"] == "front" for c in snap["areas"]["hand"]), boss


class TestVerdantLeaf:
    def test_selling_a_joker_disables_it(self):
        """card.lua:1616-1621 (G08)."""
        run = in_round("AU_LEAF", ("j_joker", "j_greedy_joker"), boss="bl_final_leaf")
        snap = check(run, ["SellCard", "jokers", 0])
        assert snap["blind"]["disabled"]
        assert not any(c["debuff"] for c in snap["areas"]["hand"])

    def test_selling_a_consumable_keeps_it(self):
        run = in_round("AU_LEAF_C", ("c_fool",), boss="bl_final_leaf")
        snap = check(run, ["SellCard", "consumables", 0])
        assert not snap["blind"]["disabled"]


class TestThePlant:
    def test_debuffs_face_cards_when_set(self):
        run = at_blind_select("AU_PLANT", boss="bl_plant")
        snap = check(run, ["SelectBlind"])
        owned = snap["areas"]["deck"] + snap["areas"]["hand"]
        faces = [c for c in owned if c["base"]["value"] in ("Jack", "Queen", "King")]
        assert faces and all(c["debuff"] for c in faces)


class TestPinnedJoker:
    def test_stays_first_after_amber_acorns_shuffles(self):
        """G200: align_cards puts pinned jokers back in front every frame
        (cardarea.lua:528), so Amber Acorn's shuffles (blind.lua:196-203) leave a
        pinned joker leftmost."""
        for seed in ("PINACORN1", "PINACORN4"):
            run = at_blind_select(
                seed,
                ("j_joker", "j_greedy_joker", "j_lusty_joker"),
                boss="bl_final_acorn",
                challenge="c_knife_1",
            )
            snap = check(run, ["SelectBlind"])
            assert jokers_of(snap)[0] == "j_ceremonial", seed


# ---------------------------------------------------------------------------
# Round flow (cardarea.lua, state_events.lua)
# ---------------------------------------------------------------------------


class TestHandSizeGrowsMidRound:
    def test_selling_stuntman_draws_two(self):
        """cardarea.lua:94-111 (G37): a positive change_size in SELECTING_HAND
        draws the difference."""
        run = in_round("AU_STUNTMAN", ("j_stuntman",))
        before = run.snapshot()
        snap = check(run, ["SellCard", "jokers", 0])
        assert snap["game"]["hand_size"] == before["game"]["hand_size"] + 2
        assert len(snap["areas"]["hand"]) == len(before["areas"]["hand"]) + 2


class TestRunningOutOfCards:
    def test_the_round_ends_when_hand_and_deck_are_empty(self):
        """game.lua:3056-3058, 3063-3064 (G36): with hand, deck and play empty the
        round ends; short of the target the run is lost."""
        run = in_round("AU_EMPTY")
        run.write("areas.deck", [])
        check(run, ["Discard", [0, 1, 2, 3, 4]])
        snap = check(run, ["Discard", [0, 1, 2]])
        assert snap["areas"]["hand"] == [] and snap["areas"]["deck"] == []
        assert snap["phase"] == "game_over"

    def test_no_hand_size_is_game_over(self):
        """state_events.lua:356-360 (G36): drawing to an empty hand that can hold
        no cards ends the run."""
        run = Run(_start("AU_NO_HAND"))
        run.write("game.hand_size", 0)
        snap = check(run, ["SelectBlind"])
        assert snap["phase"] == "game_over"


class TestPlayedCardsScoreInHandOrder:
    def test_click_order_is_sorted_by_position(self):
        """state_events.lua:463 (G49): the highlighted cards are sorted by their
        position in the hand before they are played."""
        run = in_round("AU_ORDER")
        hand = [c["sort_id"] for c in run.view("hand")]
        snap = check(run, ["PlayHand", [4, 0, 2]])
        played = [c["sort_id"] for c in snap["areas"]["discard"]]
        assert played == [hand[0], hand[2], hand[4]]


class TestWinning:
    def test_beating_the_final_boss_wins_and_a_later_loss_keeps_it(self):
        """state_events.lua:111-114 sets G.GAME.won; nothing clears it (G48)."""
        run = at_blind_select("AU_WIN", boss="bl_wall")
        run.write("round_resets.ante", 8)
        run.write("round_resets.blind_ante", 8)
        check(run, ["SelectBlind"])
        run.write("game.chips", 10**12)
        snap = check(run, ["PlayHand", [0]])
        assert snap["run"]["won"]
        snap = check(run, ["CashOut"], ["NextRound"], ["SelectBlind"])
        run.write("current_round.hands_left", 1)
        snap = check(run, ["PlayHand", [0]])
        assert snap["phase"] == "game_over"
        assert snap["run"]["won"]


# ---------------------------------------------------------------------------
# Challenge rules
# ---------------------------------------------------------------------------


class TestChallengeRules:
    def test_golden_needle_discards_cost_a_dollar(self):
        """state_events.lua:433-435."""
        run = in_round("AU_NEEDLE_COST", challenge="c_golden_needle_1")
        before = run.read("game.dollars")
        snap = check(run, ["Discard", [0]])
        assert snap["game"]["dollars"] == before - 1

    def test_typecast_rules_apply_after_the_ante_four_boss(self):
        """state_events.lua:240-247: every joker turns Eternal and the joker
        slots drop to 0."""
        run = at_blind_select(
            "AU_TYPECAST", ("j_joker", "j_greedy_joker"), boss="bl_wall", challenge="c_typecast_1"
        )
        run.write("round_resets.ante", 4)
        check(run, ["SelectBlind"])
        run.write("game.chips", 10**12)
        snap = check(run, ["PlayHand", [0]])
        assert snap["game"]["joker_slots"] == 0
        assert all(j["stickers"]["eternal"] for j in snap["areas"]["jokers"])


# ---------------------------------------------------------------------------
# Legal actions against vanilla's button gates (G201-G203)
# ---------------------------------------------------------------------------


def refused(run: Run, action: list[Any]) -> bool:
    """Both engines refuse ``action``."""
    return not run.step(action)


def to_big_blind_shop(seed: str, jokers: tuple[str, ...] = ()) -> Run:
    """The shop after the Big Blind, checking the steps that lead there."""
    run = at_blind_select(seed, jokers)
    check(run, ["SelectBlind"])
    run.write("game.chips", 10**12)
    check(run, ["PlayHand", [0]], ["CashOut"])
    return run


class TestLegality:
    def test_a_negative_shop_joker_needs_the_extra_slot(self):
        """G201: check_for_buy_space (button_callbacks.lua:2392-2402) raises the
        limit by one for a Negative card; it doesn't lift it."""
        run = to_big_blind_shop("AU_G201", ("j_greedy_joker",))
        run.oracle("a Negative Joker, priced, as the shop's first card")
        run.write("game.dollars", 100)
        run.write("game.joker_slots", 0)
        assert ["BuyCard", 0] not in run.legal()
        assert refused(run, ["BuyCard", 0])
        run.write("game.joker_slots", 1)
        assert ["BuyCard", 0] in run.legal()
        check(run, ["BuyCard", 0])

    def test_an_arcana_pack_waits_for_its_hand(self):
        """G202: can_skip_booster (button_callbacks.lua:2132-2141)."""
        run = to_big_blind_shop("AU_G202")
        run.oracle("an Arcana pack as the shop's only booster")
        run.write("game.dollars", 50)
        check(run, ["OpenBooster", 0])
        assert ["SkipPack"] in run.legal()
        run.move("hand", "deck")
        assert ["SkipPack"] not in run.legal()
        assert refused(run, ["SkipPack"])

    def test_a_joker_can_be_sold_at_round_eval(self):
        """G203: Card:can_sell_card (card.lua:1640-1653) has no state gate."""
        run = in_round("AU_G203", ("j_joker",))
        run.write("game.chips", 10**12)
        check(run, ["PlayHand", [0]])
        assert ["SellCard", "jokers", 0] in run.legal()
        snap = check(run, ["SellCard", "jokers", 0])
        assert snap["phase"] == "round_eval"
        assert snap["areas"]["jokers"] == []

    def test_jokers_reorder_at_blind_select_and_round_eval(self):
        """G206: a drag reorders the jokers in every state (cardarea.lua:214-227,
        509-529). BalatroBot's rearrange needs SELECTING_HAND, SHOP or a pack
        (rearrange.lua's requires_state), so BALATROBOT legality leaves these out."""
        run = at_blind_select("AU_G206_BLIND", ("j_ceremonial", "j_joker", "j_greedy_joker"))
        assert ["SwapJokersRight", 1] in run.legal()
        assert not any(a[0].startswith("Swap") for a in run.bot_legal())
        check(run, ["SwapJokersRight", 1])
        # Ceremonial Dagger slices its new right neighbour (card.lua:2561-2580).
        snap = check(run, ["SelectBlind"])
        assert jokers_of(snap) == ["j_ceremonial", "j_joker"]
        run.write("game.chips", 10**12)
        check(run, ["PlayHand", [0]])
        assert ["SwapJokersLeft", 1] in run.legal()
        assert not any(a[0].startswith("Swap") for a in run.bot_legal())
        snap = check(run, ["SwapJokersLeft", 1])
        assert snap["phase"] == "round_eval"
        assert jokers_of(snap) == ["j_joker", "j_ceremonial"]

    @staticmethod
    def _arcana_pack(seed: str, owned: tuple[str, ...]) -> Run:
        run = to_big_blind_shop(seed, owned)
        run.oracle("an Arcana pack as the shop's only booster")
        run.write("game.dollars", 50)
        check(run, ["OpenBooster", 0])
        return run

    def test_a_packs_dealt_hand_and_the_jokers_reorder(self):
        """G206: an Arcana pack's dealt hand is laid out and re-sorted by position
        like a round's (cardarea.lua:436-449). Death then copies the card moved
        rightmost (card.lua:1112-1117). BalatroBot's rearrange takes the hand and
        the jokers in SMODS_BOOSTER_OPENED."""
        run = self._arcana_pack("AU_G206_PACK", ("j_joker", "j_greedy_joker", "c_death"))
        legal = run.legal()
        bot = run.bot_legal()
        for action in (["SwapHandRight", 0], ["SwapHandLeft", 1], ["SwapJokersRight", 0]):
            assert action in legal and action in bot, action
        assert not any(a[0] == "SortHand" for a in legal)
        first, second = (c["front"] for c in run.view("hand")[:2])
        assert first != second
        check(run, ["SwapHandRight", 0], ["SwapJokersRight", 0])
        snap = check(run, ["UseConsumable", 0, [0, 1]])
        assert [c["front"] for c in snap["areas"]["hand"][:2]] == [first, first]
        assert jokers_of(snap) == ["j_greedy_joker", "j_joker"]

    def test_a_reordered_pack_hand_goes_back_to_the_deck_in_order(self):
        """G206: end_consumeable draws the hand back to the deck from the left,
        each card to the bottom (state_events.lua:1121-1126; cardarea.lua:33-34,
        76-80), so the reorder reaches the next deal before a shuffle."""
        run = self._arcana_pack("AU_G206_RETURN", ())
        check(run, ["SwapHandRight", 0], ["SwapHandLeft", 3])
        hand = [c["sort_id"] for c in run.view("hand")]
        snap = check(run, ["SkipPack"])
        assert [c["sort_id"] for c in snap["areas"]["deck"][: len(hand)]] == hand[::-1]


# ---------------------------------------------------------------------------
# Playing cards that went through add_to_deck (G205)
# ---------------------------------------------------------------------------


class TestDeckAddedCards:
    def test_cryptid_copies_are_added_to_the_deck(self):
        """card.lua:1206-1213 and 568-575: each copy runs add_to_deck, which
        flags a playing card and returns."""
        run = in_round("AU_G205_CRYPTID", ("c_cryptid",))
        before = {c["sort_id"] for c in run.view("hand")}
        snap = check(run, ["UseConsumable", 0, [0]])
        copies = [c for c in snap["areas"]["hand"] if c["sort_id"] not in before]
        assert len(copies) == 2 and all(c["added_to_deck"] for c in copies)

    def test_a_destroyed_deck_added_card_reruns_the_boss_debuffs(self):
        """card.lua:698 and 4733: removing a flagged card queues
        set_blind(nil, true), whose pass debuffs the card played earlier in
        The Pillar's blind (blind.lua:207-210, 634-636)."""
        run = in_round("AU_G205_PILLAR", boss="bl_pillar")
        run.write("blind.chips", 10**9)
        run.write("game.probabilities_normal", 4)  # every Glass Card breaks
        glass = run.cards("hand")[0]
        run.oracle("set_ability('m_glass') on the first hand card")
        with run.edit_card(glass) as card:
            card["added_to_deck"] = True  # as a Standard-pack pick leaves it
        snap = check(run, ["PlayHand", [1]])
        assert not snap["areas"]["discard"][0]["debuff"]
        snap = check(run, ["PlayHand", [run.cards("hand").index(glass)]])
        assert snap["areas"]["discard"][0]["debuff"]
