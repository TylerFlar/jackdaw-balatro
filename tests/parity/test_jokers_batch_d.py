"""Joker batch D: a targeted case for every branch.

Each case builds a board (BalatroBot ``add`` and ``set``, a forced boss) and takes the steps
that matter; each step must match the oracle's (:mod:`tests.parity.cases`), and the assertions
on the state afterwards show that the branch under test fired. Riff-raff's seeds are chosen so
that it creates jokers the port has.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.data import load
from tests.parity.cases import CardRef, Run

# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------


def _start(seed: str) -> dict[str, Any]:
    return {"back": "b_red", "stake": 1, "seed": seed, "challenge": None}


def in_round(
    seed: str,
    jokers: tuple[str, ...] = (),
    cards: tuple[dict[str, Any], ...] = (),
    *,
    boss: str | None = None,
    **sets: Any,
) -> Run:
    """SELECTING_HAND of the first blind (or of ``boss``) with the jokers and
    cards added by BalatroBot's ``add`` (cards join the hand, at its end)."""
    setup: list[dict[str, Any]] = []
    if boss:
        setup.append({"op": "force_boss", "key": boss})
    setup.append({"op": "action", "action": ["SelectBlind"]})
    setup += [{"op": "add", "key": k} for k in jokers]
    setup += [{"op": "add", **c} for c in cards]
    if sets:
        setup.append({"op": "set", **sets})
    return Run(_start(seed), setup)


def at_blind_select(seed: str, jokers: tuple[str, ...] = (), *, boss: str | None = None) -> Run:
    """The Big Blind's (or ``boss``'s) blind select, with the jokers added in
    the shop: their setting_blind runs in the next, checked, SelectBlind."""
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
    return Run(_start(seed), setup)


def check(run: Run, *actions: list[Any]) -> dict[str, Any]:
    """Take ``actions``, each where the oracle took it. Returns the snapshot afterwards."""
    for action in actions:
        assert run.step(action), action
    return run.snapshot()


def last(run: Run, n: int = 1) -> list[int]:
    """Hand indices of the last ``n`` hand cards (the ones ``add`` put there)."""
    size = len(run.view("hand"))
    return list(range(size - n, size))


def last_ids(run: Run, n: int = 1) -> set[int]:
    """Sort ids of the last ``n`` hand cards."""
    return {c["sort_id"] for c in run.view("hand")[-n:]}


def at(snap: dict[str, Any], ids: set[int]) -> list[int]:
    """Hand indices of the cards with these sort ids."""
    return [i for i, c in enumerate(snap["areas"]["hand"]) if c["sort_id"] in ids]


def debuff_last(run: Run, n: int = 1) -> None:
    """Debuff the last ``n`` hand cards, as the boss's set_blind pass does to
    cards already owned (``add`` makes them after it)."""
    with run.edit() as snap:
        for card in snap["areas"]["hand"][-n:]:
            card["debuff"] = True


def joker(run: Run, key: str) -> CardRef:
    return next(j for j in run.cards("jokers") if j.key == key)


def snap_joker(snap: dict[str, Any], key: str) -> dict[str, Any]:
    return next(j for j in snap["areas"]["jokers"] if j["center"] == key)


def owned(snap: dict[str, Any]) -> list[dict[str, Any]]:
    areas = snap["areas"]
    return areas["deck"] + areas["hand"] + areas["discard"] + areas["play"]


def stones(snap: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in owned(snap) if c["center"] == "m_stone"]


def center_set(key: str) -> str:
    return load("centers")[key]["set"]


def rarity(key: str) -> int:
    return load("centers")[key]["rarity"]


BIG = 10**9  # round chips that beat any early blind

# ---------------------------------------------------------------------------
# setting_blind
# ---------------------------------------------------------------------------


class TestCeremonialDagger:
    def test_slices_its_neighbour_then_scores(self):
        run = at_blind_select("PD_DAGGER", ("j_ceremonial", "j_joker"))
        sell = joker(run, "j_joker").costs["sell_cost"]
        snap = check(run, ["SelectBlind"])
        assert [j["center"] for j in snap["areas"]["jokers"]] == ["j_ceremonial"]
        assert snap_joker(snap, "j_ceremonial")["ability"]["mult"] == 2 * sell
        check(run, ["PlayHand", [0]])

    def test_spares_an_eternal_neighbour(self):
        run = at_blind_select("PD_DAGGER_ET", ("j_ceremonial", "j_joker"))
        with run.edit_card(joker(run, "j_joker")) as card:
            card["stickers"]["eternal"] = True
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["jokers"]) == 2
        assert snap_joker(snap, "j_ceremonial")["ability"]["mult"] == 0

    def test_blueprint_copies_the_mult_not_the_slice(self):
        run = at_blind_select("PD_DAGGER_BP", ("j_blueprint", "j_ceremonial", "j_joker"))
        snap = check(run, ["SelectBlind"], ["PlayHand", [0]])
        assert [j["center"] for j in snap["areas"]["jokers"]] == ["j_blueprint", "j_ceremonial"]
        assert snap_joker(snap, "j_ceremonial")["ability"]["mult"] > 0

    @pytest.mark.parametrize(("victim", "change"), [("j_juggler", -1), ("j_merry_andy", 1)])
    def test_the_deal_counts_before_the_victims_hand_size_goes(self, victim, change):
        """G105: the victim's change_size runs after the deal is counted."""
        run = at_blind_select(f"PD_DAGGER_{victim}", ("j_ceremonial", victim))
        size = run.read("game.hand_size")
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["hand"]) == size
        assert snap["game"]["hand_size"] == size + change

    def test_madness_and_dagger_slice_one_joker(self):
        run = at_blind_select("PD_MAD_DAGGER", ("j_madness", "j_ceremonial", "j_joker"))
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["jokers"]) == 2


class TestMadness:
    def test_grows_and_slices_outside_boss_blinds(self):
        run = at_blind_select("PD_MADNESS", ("j_madness", "j_joker", "j_greedy_joker"))
        snap = check(run, ["SelectBlind"], ["PlayHand", [0]])
        assert snap_joker(snap, "j_madness")["ability"]["x_mult"] == 1.5
        assert len(snap["areas"]["jokers"]) == 2

    def test_rests_at_a_boss(self):
        run = at_blind_select("PD_MADNESS_BOSS", ("j_madness", "j_joker"), boss="bl_club")
        snap = check(run, ["SelectBlind"])
        assert snap_joker(snap, "j_madness")["ability"]["x_mult"] == 1
        assert len(snap["areas"]["jokers"]) == 2

    def test_spares_eternal_jokers(self):
        run = at_blind_select("PD_MADNESS_ET", ("j_madness", "j_joker"))
        with run.edit_card(joker(run, "j_joker")) as card:
            card["stickers"]["eternal"] = True
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["jokers"]) == 2
        assert snap_joker(snap, "j_madness")["ability"]["x_mult"] == 1.5

    def test_blueprint_copy_does_nothing_at_setting_blind(self):
        run = at_blind_select("PD_MADNESS_BP", ("j_blueprint", "j_madness"))
        snap = check(run, ["SelectBlind"], ["PlayHand", [0]])
        # Madness slices Blueprint (its only other joker); nothing copies its growth.
        assert [j["center"] for j in snap["areas"]["jokers"]] == ["j_madness"]


class TestBurglar:
    def test_trades_discards_for_hands(self):
        run = at_blind_select("PD_BURGLAR", ("j_burglar",))
        snap = check(run, ["SelectBlind"])
        cr, rr = snap["current_round"], snap["round_resets"]
        assert (cr["discards_left"], cr["hands_left"]) == (0, rr["hands"] + 3)

    def test_blueprint_adds_three_more(self):
        run = at_blind_select("PD_BURGLAR_BP", ("j_blueprint", "j_burglar"))
        snap = check(run, ["SelectBlind"])
        assert snap["current_round"]["hands_left"] == snap["round_resets"]["hands"] + 6

    @pytest.mark.parametrize(
        "board", [("j_burglar",), ("j_burglar", "j_chicot"), ("j_chicot", "j_burglar")]
    )
    def test_the_water_counts_first(self, board):
        """G97: The Water's count is taken in set_blind, before the jokers."""
        run = at_blind_select("PD_BURGLAR_WATER", board, boss="bl_water")
        snap = check(run, ["SelectBlind"])
        assert snap["blind"]["discards_sub"] == snap["round_resets"]["discards"]


class TestRiffRaff:
    def test_makes_two_commons(self):
        run = at_blind_select("PD_RIFF_9", ("j_riff_raff",))
        snap = check(run, ["SelectBlind"])
        made = snap["areas"]["jokers"][1:]
        assert len(made) == 2
        assert all(rarity(j["center"]) == 1 for j in made)

    def test_fills_only_the_room_left(self):
        run = at_blind_select("PD_RIFF_ROOM_1", ("j_riff_raff", "j_joker", "j_joker", "j_joker"))
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["jokers"]) == snap["game"]["joker_slots"]

    def test_counts_the_daggers_slot(self):
        """G24: the Dagger's joker_buffer -1 frees the slot of its victim."""
        run = at_blind_select(
            "PD_RIFF_DAGGER_1", ("j_ceremonial", "j_joker", "j_riff_raff", "j_joker", "j_joker")
        )
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["jokers"]) == 5
        assert snap_joker(snap, "j_ceremonial")["ability"]["mult"] > 0

    def test_blueprint_makes_more_while_there_is_room(self):
        run = at_blind_select("PD_RIFF_BP_14", ("j_blueprint", "j_riff_raff"))
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["jokers"]) == 5


class TestCartomancer:
    def test_makes_a_tarot(self):
        run = at_blind_select("PD_CARTO", ("j_cartomancer",))
        snap = check(run, ["SelectBlind"])
        (tarot,) = snap["areas"]["consumables"]
        assert center_set(tarot["center"]) == "Tarot"

    def test_needs_room(self):
        run = at_blind_select("PD_CARTO_FULL", ("j_blueprint", "j_cartomancer", "j_cartomancer"))
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["consumables"]) == snap["game"]["consumable_slots"]

    def test_tarot_comes_after_the_stone_and_riff_raffs_jokers(self):
        """G100: Cartomancer's Tarot is made by a nested event."""
        run = at_blind_select("PD_CARTO_ORDER_5", ("j_cartomancer", "j_marble", "j_riff_raff"))
        snap = check(run, ["SelectBlind"])
        tarot = snap["areas"]["consumables"][0]["sort_id"]
        others = [c["sort_id"] for c in stones(snap)] + [
            j["sort_id"] for j in snap["areas"]["jokers"][3:]
        ]
        assert len(others) == 3
        assert max(others) < tarot


class TestMarbleJoker:
    def test_stone_is_shuffled_in(self):
        """G55: the Stone joins the deck before the 'nr' shuffle."""
        run = at_blind_select("PD_MARBLE", ("j_marble",))
        snap = check(run, ["SelectBlind"])
        assert len(stones(snap)) == 1
        assert snap["sim"]["pending_deck_bottom"] == []

    def test_copies_add_stones(self):
        run = at_blind_select("PD_MARBLE_BP", ("j_blueprint", "j_marble", "j_brainstorm"))
        snap = check(run, ["SelectBlind"])
        assert len(stones(snap)) == 3

    def test_a_sliced_marble_makes_no_stone(self):
        run = at_blind_select("PD_MARBLE_SLICED", ("j_ceremonial", "j_marble"))
        snap = check(run, ["SelectBlind"])
        assert not stones(snap)

    @pytest.mark.parametrize(
        ("seed", "board"),
        [
            ("PD_MARBLE_LEAF_RM_4", ("j_riff_raff", "j_marble")),
            ("PD_MARBLE_LEAF_MR_0", ("j_marble", "j_riff_raff")),
            ("PD_MARBLE_LEAF_C", ("j_cartomancer", "j_marble")),
        ],
    )
    def test_a_later_redebuff_reaches_the_stone(self, seed, board):
        """G100: an add_to_deck queues set_blind(nil, true) after the Stone."""
        run = at_blind_select(seed, board, boss="bl_final_leaf")
        snap = check(run, ["SelectBlind"])
        (stone,) = stones(snap)
        assert stone["debuff"]

    def test_a_lone_stone_keeps_its_state(self):
        run = at_blind_select("PD_MARBLE_LONE", ("j_marble",), boss="bl_final_leaf")
        snap = check(run, ["SelectBlind"])
        (stone,) = stones(snap)
        assert not stone["debuff"]


class TestChicot:
    def test_disables_a_boss(self):
        run = at_blind_select("PD_CHICOT", ("j_chicot",), boss="bl_wall")
        snap = check(run, ["SelectBlind"])
        assert snap["blind"]["disabled"]

    def test_two_chicots_disable_twice(self):
        run = at_blind_select("PD_CHICOT_2", ("j_chicot", "j_chicot"), boss="bl_wall")
        target = check(at_blind_select("PD_CHICOT_2", (), boss="bl_wall"), ["SelectBlind"])
        snap = check(run, ["SelectBlind"])
        assert snap["blind"]["chips"] == target["blind"]["chips"] / 4

    def test_leaves_small_and_big_blinds(self):
        run = at_blind_select("PD_CHICOT_BIG", ("j_blueprint", "j_chicot"))
        snap = check(run, ["SelectBlind"])
        assert not snap["blind"]["disabled"]

    def test_the_manacles_disable_draws(self):
        run = at_blind_select("PD_CHICOT_MANACLE", ("j_chicot",), boss="bl_manacle")
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["hand"]) == snap["game"]["hand_size"]

    def test_amber_acorn_shuffles_the_daggers_victim(self):
        """G100: Amber Acorn's shuffles run before the sliced joker leaves."""
        board = ("j_ceremonial", "j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker")
        run = at_blind_select("PD_ACORN_DAGGER", board, boss="bl_final_acorn")
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["jokers"]) == 4


# ---------------------------------------------------------------------------
# first_hand_drawn
# ---------------------------------------------------------------------------


class TestCertificate:
    def test_deals_a_sealed_card(self):
        run = at_blind_select("PD_CERT", ("j_certificate",))
        snap = check(run, ["SelectBlind"])
        hand = snap["areas"]["hand"]
        assert len(hand) == snap["game"]["hand_size"] + 1
        new = max(hand, key=lambda c: c["sort_id"])
        assert new["seal"] is not None
        assert (new["costs"]["cost"], new["costs"]["sell_cost"]) == (1, 1)

    def test_blueprint_deals_another(self):
        run = at_blind_select("PD_CERT_BP", ("j_blueprint", "j_certificate"))
        snap = check(run, ["SelectBlind"])
        assert len(snap["areas"]["hand"]) == snap["game"]["hand_size"] + 2

    def test_deals_before_cerulean_bell_picks(self):
        """G14: first_hand_drawn runs before the blind's drawn_to_hand."""
        run = at_blind_select("PD_CERT_BELL", ("j_certificate",), boss="bl_final_bell")
        snap = check(run, ["SelectBlind"])
        assert sum(1 for c in snap["areas"]["hand"] if c["ability"].get("forced_selection")) == 1

    def test_the_boss_debuffs_the_new_card(self):
        run = at_blind_select("PD_CERT_PLANT_1", ("j_certificate", "j_pareidolia"), boss="bl_plant")
        snap = check(run, ["SelectBlind"])
        new = max(snap["areas"]["hand"], key=lambda c: c["sort_id"])
        assert new["debuff"]


# ---------------------------------------------------------------------------
# The played hand
# ---------------------------------------------------------------------------


class TestDNA:
    def test_copies_a_single_card_first_hand(self):
        card = {"key": "H_K", "enhancement": "m_glass", "seal": "Red", "edition": "foil"}
        run = in_round("PD_DNA", ("j_dna",), (card,))
        snap = check(run, ["PlayHand", last(run)])
        copy = max(owned(snap), key=lambda c: c["sort_id"])
        assert (copy["front"], copy["center"], copy["seal"]) == ("H_K", "m_glass", "Red")
        assert copy["edition"]["type"] == "foil"

    def test_blueprint_copies_twice(self):
        run = in_round("PD_DNA_BP", ("j_blueprint", "j_dna"), ({"key": "S_A", "seal": "Gold"},))
        snap = check(run, ["PlayHand", last(run)])
        assert sum(1 for c in snap["areas"]["hand"] if c["seal"] == "Gold") == 2
        assert len(owned(snap)) == 52 + 1 + 2

    def test_only_the_first_hand(self):
        run = in_round("PD_DNA_2ND", ("j_dna",))
        snap = check(run, ["PlayHand", [0, 1]], ["PlayHand", [0]])
        assert len(owned(snap)) == 52

    def test_copies_a_debuffed_card_debuffed(self):
        run = in_round("PD_DNA_GOAD", ("j_dna",), ({"key": "S_K"},), boss="bl_goad")
        debuff_last(run)
        snap = check(run, ["PlayHand", last(run)])
        copy = max(snap["areas"]["hand"], key=lambda c: c["sort_id"])
        assert copy["front"] == "S_K" and copy["debuff"]


class TestSixthSense:
    def test_destroys_a_lone_six_for_a_spectral(self):
        run = in_round("PD_SIXTH", ("j_sixth_sense",), ({"key": "H_6"},))
        six = last_ids(run)
        snap = check(run, ["PlayHand", last(run)])
        assert not any(c["sort_id"] in six for c in owned(snap))
        (spectral,) = snap["areas"]["consumables"]
        assert center_set(spectral["center"]) == "Spectral"

    def test_destroys_without_room_too(self):
        run = in_round("PD_SIXTH_FULL", ("j_sixth_sense", "c_pluto", "c_mars"), ({"key": "H_6"},))
        snap = check(run, ["PlayHand", last(run)])
        assert len(owned(snap)) == 52
        assert [c["center"] for c in snap["areas"]["consumables"]] == ["c_pluto", "c_mars"]

    def test_blueprint_copy_does_nothing(self):
        run = in_round("PD_SIXTH_BP", ("j_blueprint", "j_sixth_sense"), ({"key": "H_6"},))
        snap = check(run, ["PlayHand", last(run)])
        assert len(snap["areas"]["consumables"]) == 1

    def test_not_after_the_first_hand(self):
        run = in_round("PD_SIXTH_2ND", ("j_sixth_sense",), ({"key": "H_6"},))
        six = last_ids(run)
        snap = check(run, ["PlayHand", [0]])
        snap = check(run, ["PlayHand", at(snap, six)])
        assert any(c["sort_id"] in six for c in owned(snap))
        assert not snap["areas"]["consumables"]

    def test_caino_counts_a_destroyed_face(self):
        """With Pareidolia the six is a face card (card.lua:964-970)."""
        run = in_round(
            "PD_SIXTH_CAINO", ("j_sixth_sense", "j_caino", "j_pareidolia"), ({"key": "H_6"},)
        )
        snap = check(run, ["PlayHand", last(run)])
        assert snap_joker(snap, "j_caino")["ability"]["caino_xmult"] == 2

    def test_a_glass_six_shatters_for_glass_joker(self):
        """A destroyed Glass Card is marked shattered (state_events.lua:966-967)."""
        six = ({"key": "H_6", "enhancement": "m_glass"},)
        run = in_round("PD_SIXTH_GLASS", ("j_sixth_sense", "j_glass"), six)
        snap = check(run, ["PlayHand", last(run)])
        assert snap_joker(snap, "j_glass")["ability"]["x_mult"] == 1.75


class TestToDoList:
    def _set_hand(self, run: Run, hand: str) -> None:
        with run.edit() as snap:
            for j in snap["areas"]["jokers"]:
                if j["center"] == "j_todo_list":
                    j["ability"]["to_do_poker_hand"] = hand

    def test_pays_for_its_hand_before_scoring(self):
        run = in_round("PD_TODO", ("j_todo_list",))
        self._set_hand(run, "High Card")
        dollars = run.read("game.dollars")
        snap = check(run, ["PlayHand", [0]])
        assert snap["game"]["dollars"] == dollars + 4
        assert snap["sim"]["last_score"]["dollars"] == 4

    def test_blueprint_pays_again(self):
        run = in_round("PD_TODO_BP", ("j_blueprint", "j_todo_list"))
        self._set_hand(run, "High Card")
        dollars = run.read("game.dollars")
        snap = check(run, ["PlayHand", [0]])
        assert snap["game"]["dollars"] == dollars + 8

    def test_rerolls_its_hand_at_round_end(self):
        run = in_round("PD_TODO_EOR", ("j_todo_list",), chips=BIG)
        self._set_hand(run, "Flush Five")
        snap = check(run, ["PlayHand", [0]])
        assert snap["phase"] == "round_eval"
        assert snap_joker(snap, "j_todo_list")["ability"]["to_do_poker_hand"] != "Flush Five"


class TestMatador:
    def test_pays_on_a_debuffed_hand(self):
        run = in_round("PD_MATADOR_PSY", ("j_matador",), boss="bl_psychic")
        dollars = run.read("game.dollars")
        snap = check(run, ["PlayHand", [0]])
        assert snap["sim"]["last_score"]["debuffed"]
        assert snap["game"]["dollars"] == dollars + 8

    def test_pays_for_a_debuffed_scoring_card(self):
        run = in_round(
            "PD_MATADOR_GOAD", ("j_blueprint", "j_matador"), ({"key": "S_2"},), boss="bl_goad"
        )
        debuff_last(run)
        dollars = run.read("game.dollars")
        snap = check(run, ["PlayHand", last(run)])
        assert snap["game"]["dollars"] == dollars + 8 * 2

    @pytest.mark.parametrize(
        ("boss", "change"),
        [
            # press_play sets triggered for The Hook and The Tooth, but debuff_hand
            # clears it for every blind before scoring (blind.lua:519-522).
            ("bl_hook", lambda d: d),
            ("bl_tooth", lambda d: d - 1),
            # modify_hand sets it after debuff_hand (blind.lua:510-517).
            ("bl_flint", lambda d: d + 8),
            # The Ox takes the money for the most played hand (High Card) first.
            ("bl_ox", lambda d: 8),
        ],
    )
    def test_pays_when_the_boss_triggers(self, boss, change):
        run = in_round(f"PD_MATADOR_{boss}", ("j_matador",), boss=boss)
        dollars = run.read("game.dollars")
        snap = check(run, ["PlayHand", [0]])
        assert snap["game"]["dollars"] == change(dollars)

    def test_no_pay_without_a_trigger(self):
        run = in_round("PD_MATADOR_ARM", ("j_matador",), boss="bl_arm")
        dollars = run.read("game.dollars")
        snap = check(run, ["PlayHand", [0]])
        assert snap["game"]["dollars"] == dollars


class TestGlassJoker:
    def test_grows_with_shattered_glass(self):
        # Two Oops! All 6s make every Glass Card shatter (4 in 4).
        cards = tuple({"key": k, "enhancement": "m_glass"} for k in ("H_2", "D_2", "C_2", "S_2"))
        run = in_round("PD_GLASS_1", ("j_glass", "j_oops", "j_oops"), cards)
        snap = check(run, ["PlayHand", last(run, 4)])
        assert snap_joker(snap, "j_glass")["ability"]["x_mult"] == 1 + 0.75 * 4

    def test_counts_a_glass_card_trading_card_destroys(self):
        run = in_round(
            "PD_GLASS_TRADE", ("j_glass", "j_trading"), ({"key": "H_K", "enhancement": "m_glass"},)
        )
        snap = check(run, ["Discard", last(run)])
        assert snap_joker(snap, "j_glass")["ability"]["x_mult"] == 1.75

    def test_counts_the_glass_cards_the_hanged_man_destroys(self):
        # using_consumeable counts the highlighted Glass Cards (card.lua:2709-2720).
        cards = ({"key": "H_K", "enhancement": "m_glass"}, {"key": "D_K", "enhancement": "m_glass"})
        run = in_round("PD_GLASS_HANGED", ("j_glass", "c_hanged_man"), cards)
        snap = check(run, ["UseConsumable", 0, last(run, 2)])
        assert snap_joker(snap, "j_glass")["ability"]["x_mult"] == 1 + 0.75 * 2

    def test_the_hanged_man_counts_only_glass_cards(self):
        cards = ({"key": "H_K", "enhancement": "m_glass"}, {"key": "D_K"})
        run = in_round("PD_GLASS_HANGED_2", ("j_glass", "c_hanged_man"), cards)
        snap = check(run, ["UseConsumable", 0, last(run, 2)])
        assert snap_joker(snap, "j_glass")["ability"]["x_mult"] == 1.75


# ---------------------------------------------------------------------------
# Discards
# ---------------------------------------------------------------------------


class TestTradingCard:
    def test_first_lone_discard_pays_and_destroys(self):
        run = in_round("PD_TRADE", ("j_trading", "j_caino"), ({"key": "H_K"},))
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run)])
        assert snap["game"]["dollars"] == dollars + 3
        assert len(owned(snap)) == 52
        assert snap_joker(snap, "j_caino")["ability"]["caino_xmult"] == 2
        # Caino's X2 in the main step (card.lua:4052-4057).
        snap = check(run, ["PlayHand", [0]])
        last_score = snap["sim"]["last_score"]
        assert last_score["mult"] == 2 * 1  # High Card's mult 1, then X2

    def test_blueprint_copy_does_nothing(self):
        run = in_round("PD_TRADE_BP", ("j_blueprint", "j_trading"), ({"key": "H_K"},))
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run)])
        assert snap["game"]["dollars"] == dollars + 3

    def test_only_the_first_discard_of_one_card(self):
        run = in_round("PD_TRADE_2", ("j_trading",))
        snap = check(run, ["Discard", [0, 1]], ["Discard", [0]])
        assert len(owned(snap)) == 52


class TestFacelessJoker:
    FACES = ({"key": "H_J"}, {"key": "S_Q"}, {"key": "D_K"})

    def test_pays_for_three_faces(self):
        run = in_round("PD_FACELESS", ("j_faceless",), self.FACES)
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run, 3)])
        assert snap["game"]["dollars"] == dollars + 5

    def test_blueprint_pays_again(self):
        run = in_round("PD_FACELESS_BP", ("j_blueprint", "j_faceless"), self.FACES)
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run, 3)])
        assert snap["game"]["dollars"] == dollars + 10

    def test_counts_pareidolia_faces(self):
        cards = ({"key": "H_2"}, {"key": "S_3"}, {"key": "D_4"})
        run = in_round("PD_FACELESS_PAR", ("j_faceless", "j_pareidolia"), cards)
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run, 3)])
        assert snap["game"]["dollars"] == dollars + 5

    def test_debuffed_faces_do_not_count(self):
        run = in_round("PD_FACELESS_PLANT", ("j_faceless",), self.FACES, boss="bl_plant")
        debuff_last(run, 3)
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run, 3)])
        assert snap["game"]["dollars"] == dollars


class TestMailInRebate:
    def test_pays_for_each_card_of_the_rank(self):
        run = in_round("PD_MAIL", ("j_blueprint", "j_mail"))
        rank = run.read("current_round.mail_card.rank")
        letter = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}.get(rank, rank)
        for suit in "HS":
            run.op({"op": "add", "key": f"{suit}_{letter}"})
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run, 2)])
        assert snap["game"]["dollars"] == dollars + 5 * 2 * 2


class TestCastle:
    def test_gains_chips_for_its_suit_and_scores_them(self):
        run = in_round("PD_CASTLE", ("j_blueprint", "j_castle"))
        suit = run.read("current_round.castle_card.suit")
        for rank in "234":
            run.op({"op": "add", "key": f"{suit[0]}_{rank}"})
        snap = check(run, ["Discard", last(run, 3)], ["PlayHand", [0]])
        assert snap_joker(snap, "j_castle")["ability"]["extra"]["chips"] == 9

    def test_smeared_joker_widens_the_suit(self):
        run = in_round("PD_CASTLE_SMEAR", ("j_castle", "j_smeared"))
        before = run.snapshot()
        red = {"Hearts", "Diamonds"}
        castle_red = before["current_round"]["castle_card"]["suit"] in red
        same = sum(
            1 for c in before["areas"]["hand"][:5] if (c["base"]["suit"] in red) == castle_red
        )
        snap = check(run, ["Discard", [0, 1, 2, 3, 4]], ["PlayHand", [0]])
        assert snap_joker(snap, "j_castle")["ability"]["extra"]["chips"] == 3 * same

    def test_castle_and_mail_skip_debuffed_cards(self):
        run = in_round("PD_CASTLE_MAIL_DEBUFF", ("j_castle", "j_mail"))
        cr = run.read("current_round")
        rank = cr["mail_card"]["rank"]
        letter = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}.get(rank, rank)
        run.op({"op": "add", "key": f"{cr['castle_card']['suit'][0]}_{letter}"})
        debuff_last(run)
        dollars = run.read("game.dollars")
        snap = check(run, ["Discard", last(run)])
        assert snap["game"]["dollars"] == dollars
        assert snap_joker(snap, "j_castle")["ability"]["extra"]["chips"] == 0


class TestHitTheRoad:
    def test_grows_per_jack_and_resets(self):
        jacks = ({"key": "H_J"}, {"key": "S_J"}, {"key": "D_J"})
        run = in_round("PD_ROAD", ("j_blueprint", "j_hit_the_road"), jacks)
        snap = check(run, ["Discard", last(run, 3)])
        assert snap_joker(snap, "j_hit_the_road")["ability"]["x_mult"] == 2.5
        run.op({"op": "set", "chips": BIG})
        snap = check(run, ["PlayHand", [0]])
        assert snap_joker(snap, "j_hit_the_road")["ability"]["x_mult"] == 1

    def test_a_debuffed_jack_does_not_count(self):
        run = in_round("PD_ROAD_GOAD", ("j_hit_the_road",), ({"key": "S_J"},), boss="bl_goad")
        debuff_last(run)
        snap = check(run, ["Discard", last(run)])
        assert snap_joker(snap, "j_hit_the_road")["ability"]["x_mult"] == 1


class TestBurntJoker:
    def test_levels_the_first_discard(self):
        run = in_round("PD_BURNT", ("j_burnt", "j_brainstorm"), ({"key": "H_9"}, {"key": "S_9"}))
        snap = check(run, ["Discard", last(run, 2)], ["Discard", [0, 1]])
        assert snap["hands"]["Pair"]["level"] == 3

    def test_not_the_hooks_discard(self):
        run = in_round("PD_BURNT_HOOK", ("j_burnt",), boss="bl_hook")
        snap = check(run, ["PlayHand", [0]], ["Discard", [0, 1]])
        assert sum(h["level"] for h in snap["hands"].values()) == 13

    def test_the_arm_sees_the_levels(self):
        """G98: highlighting a levelled hand under The Arm sets triggered."""
        cards = ({"key": "H_9"}, {"key": "S_9"}, {"key": "D_9"}, {"key": "C_9"})
        run = in_round("PD_BURNT_ARM", ("j_blueprint", "j_burnt"), cards, boss="bl_arm")
        later = last_ids(run, 2)
        snap = check(run, ["Discard", last(run, 4)[:2]])
        assert snap["hands"]["Pair"]["level"] == 3
        assert not snap["blind"]["triggered"]
        nines = [i for i, c in enumerate(snap["areas"]["hand"]) if c["sort_id"] in later]
        snap = check(run, ["Discard", nines])
        assert snap["blind"]["triggered"]


class TestYorick:
    def test_grows_every_23_discarded_cards(self):
        run = in_round("PD_YORICK", ("j_blueprint", "j_yorick"))
        with run.edit_card(joker(run, "j_yorick")) as card:
            card["ability"]["yorick_discards"] = 2
        snap = check(run, ["Discard", [0, 1, 2]], ["PlayHand", [0]])
        ab = snap_joker(snap, "j_yorick")["ability"]
        assert (ab["x_mult"], ab["yorick_discards"]) == (2, 22)


class TestRamen:
    def test_loses_xmult_per_card(self):
        run = in_round("PD_RAMEN", ("j_blueprint", "j_ramen"))
        snap = check(run, ["Discard", [0, 1, 2]], ["PlayHand", [0]])
        assert snap_joker(snap, "j_ramen")["ability"]["x_mult"] == 2 - 0.01 - 0.01 - 0.01

    def test_is_eaten(self):
        run = in_round("PD_RAMEN_EATEN", ("j_ramen",))
        with run.edit_card(joker(run, "j_ramen")) as card:
            card["ability"]["x_mult"] = 1.02
        snap = check(run, ["Discard", [0, 1, 2]])
        assert not snap["areas"]["jokers"]


# ---------------------------------------------------------------------------
# end_of_round
# ---------------------------------------------------------------------------


class TestEggAndGiftCard:
    def test_raise_sell_values(self):
        run = in_round("PD_EGG_GIFT", ("j_egg", "j_gift", "c_pluto"), chips=BIG)
        snap = check(run, ["PlayHand", [0]])
        assert snap_joker(snap, "j_egg")["ability"]["extra_value"] == 3 + 1
        assert snap_joker(snap, "j_gift")["ability"]["extra_value"] == 1
        assert snap["areas"]["consumables"][0]["ability"]["extra_value"] == 1

    def test_keep_the_coupon_flag(self):
        """G99."""
        run = in_round("PD_EGG_COUPON", ("j_egg", "j_gift", "j_joker"), chips=BIG)
        with run.edit() as snap:
            for j in snap["areas"]["jokers"]:
                j["ability"]["couponed"] = True
        snap = check(run, ["PlayHand", [0]])
        assert all(j["ability"].get("couponed") for j in snap["areas"]["jokers"])

    def test_gift_card_prices_a_planet_with_astronomer(self):
        """G99."""
        run = in_round("PD_GIFT_ASTRO", ("j_gift", "j_astronomer", "c_pluto"), chips=BIG)
        snap = check(run, ["PlayHand", [0]])
        assert snap["areas"]["consumables"][0]["costs"]["cost"] == 0

    def test_blueprint_copies_neither(self):
        run = in_round("PD_EGG_BP", ("j_blueprint", "j_egg", "j_brainstorm"), chips=BIG)
        snap = check(run, ["PlayHand", [0]])
        assert snap_joker(snap, "j_egg")["ability"]["extra_value"] == 3


class TestRocket:
    def test_grows_at_a_boss(self):
        run = in_round("PD_ROCKET", ("j_blueprint", "j_rocket"), boss="bl_club", chips=BIG)
        snap = check(run, ["PlayHand", [0]])
        assert snap_joker(snap, "j_rocket")["ability"]["extra"]["dollars"] == 3

    def test_rests_after_a_small_blind(self):
        run = in_round("PD_ROCKET_SMALL", ("j_rocket",), chips=BIG)
        snap = check(run, ["PlayHand", [0]])
        assert snap_joker(snap, "j_rocket")["ability"]["extra"]["dollars"] == 1


class TestTurtleBean:
    def test_shrinks_each_round(self):
        run = in_round("PD_TURTLE", ("j_turtle_bean",), chips=BIG)
        snap = check(run, ["PlayHand", [0]])
        assert snap_joker(snap, "j_turtle_bean")["ability"]["extra"]["h_size"] == 4
        assert snap["game"]["hand_size"] == 8 + 4

    def test_is_eaten(self):
        run = in_round("PD_TURTLE_EATEN", ("j_turtle_bean",), chips=BIG)
        with run.edit() as snap:
            snap_joker(snap, "j_turtle_bean")["ability"]["extra"]["h_size"] = 1
            snap["game"]["hand_size"] -= 4
        snap = check(run, ["PlayHand", [0]])
        assert not snap["areas"]["jokers"]
        assert snap["game"]["hand_size"] == 8


class TestMrBones:
    def _losing(self, seed: str, jokers: tuple[str, ...], chips: float) -> Run:
        return in_round(seed, jokers, chips=chips, hands_left=1)

    def test_saves_a_quarter_of_the_blind(self):
        run = self._losing("PD_BONES", ("j_mr_bones",), 100)
        snap = check(run, ["PlayHand", [0]])
        assert snap["phase"] == "round_eval"
        assert not snap["areas"]["jokers"]

    def test_lets_a_poor_round_go(self):
        run = self._losing("PD_BONES_LOW", ("j_mr_bones",), 0)
        snap = check(run, ["PlayHand", [0]])
        assert snap["phase"] == "game_over"

    def test_one_of_two_saves(self):
        run = self._losing("PD_BONES_2", ("j_mr_bones", "j_mr_bones"), 100)
        snap = check(run, ["PlayHand", [0]])
        assert [j["center"] for j in snap["areas"]["jokers"]] == ["j_mr_bones"]

    def test_blueprint_cannot_save(self):
        run = self._losing("PD_BONES_BP", ("j_mr_bones", "j_brainstorm"), 100)
        snap = check(run, ["PlayHand", [0]])
        assert [j["center"] for j in snap["areas"]["jokers"]] == ["j_brainstorm"]

    def test_pays_its_rent_when_it_saves(self):
        """G101."""
        run = self._losing("PD_BONES_RENT", ("j_mr_bones",), 100)
        with run.edit_card(joker(run, "j_mr_bones")) as card:
            card["stickers"]["rental"] = True
        dollars = run.read("game.dollars")
        snap = check(run, ["PlayHand", [0]])
        assert snap["game"]["dollars"] == dollars - 3

    def test_saves_a_debuffed_hand(self):
        run = in_round("PD_BONES_PSY", ("j_mr_bones",), boss="bl_psychic", chips=1000, hands_left=1)
        snap = check(run, ["PlayHand", [0]])
        assert snap["phase"] == "round_eval"
