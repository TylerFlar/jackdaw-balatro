"""Parity cases for points of the python-fixed gaps that no tape or targeted case reached.

Found by the gap status sweep: Rust implemented every point, but these sub-cases had no test
that ran Rust through them. Each step is checked through the batch helpers: it must match the
oracle's (:mod:`tests.parity.cases`), and the assertions show the point under test decided the
outcome. Line references are into ``balatro_source/``.
"""

from __future__ import annotations

import sys
from typing import Any

from tests.parity.cases import Run
from tests.parity.test_consumables_rust import Table
from tests.parity.test_jokers_batch_b import Board
from tests.parity.test_jokers_batch_d import (
    BIG,
    at_blind_select,
    check,
    debuff_last,
    in_round,
    joker,
    last,
    last_ids,
    owned,
    snap_joker,
)

_FACES = ("Jack", "Queen", "King")


def _hologram(snap: dict[str, Any]) -> float:
    return snap_joker(snap, "j_hologram")["ability"]["x_mult"]


def _hand(snap: dict[str, Any]) -> list[dict[str, Any]]:
    return snap["areas"]["hand"]


# ---------------------------------------------------------------------------
# G13: Pareidolia or Smeared Joker joining or leaving re-runs the boss's debuffs
# ---------------------------------------------------------------------------


def test_selling_pareidolia_under_the_plant_rechecks_every_card():
    """card.lua:698 (remove_from_deck's set_blind(nil, true)); blind.lua:630."""
    run = in_round("AS_G13_PAR_SELL", ("j_pareidolia",), boss="bl_plant")
    run.recheck()
    assert all(c["debuff"] for c in _hand(run.snapshot()))  # every card is a face
    snap = check(run, ["SellCard", "jokers", 0])
    assert any(c["base"]["value"] not in _FACES for c in _hand(snap))
    for c in _hand(snap):
        assert c["debuff"] == (c["base"]["value"] in _FACES), c["front"]


def test_selling_smeared_joker_under_the_club_rechecks_every_card():
    """card.lua:698; blind.lua:626: with Smeared a Spade is a Club."""
    run = in_round("AS_G13_SMEAR_SELL", ("j_smeared",), boss="bl_club")
    run.recheck()
    spades = [c for c in _hand(run.snapshot()) if c["base"]["suit"] == "Spades"]
    assert spades and all(c["debuff"] for c in spades)
    snap = check(run, ["SellCard", "jokers", 0])
    for c in _hand(snap):
        assert c["debuff"] == (c["base"]["suit"] == "Clubs"), c["front"]


def test_judgement_making_pareidolia_under_the_plant_debuffs_every_card():
    """card.lua:641 (add_to_deck's set_blind(nil, true))."""
    t = Table("AS_G13_JUD_bl_plant_10", boss="bl_plant")
    assert not all(c.debuff for c in t.hand)
    t.use("c_judgement")
    assert t.keys("jokers") == ["j_pareidolia"]
    assert all(c.debuff for c in t.hand)


def test_judgement_making_smeared_joker_under_the_club_debuffs_the_spades():
    t = Table("AS_G13_JUD_bl_club_79", boss="bl_club")
    spades = [c for c in t.hand if c.base["suit"] == "Spades"]
    assert spades and not any(c.debuff for c in spades)
    t.use("c_judgement")
    assert t.keys("jokers") == ["j_smeared"]
    assert all(c.debuff for c in spades)


def test_strength_making_a_face_under_the_plant_debuffs_it():
    """card.lua:143: a rank change re-checks the card."""
    t = Table("AS_G13_STRENGTH_PLANT", boss="bl_plant")
    (ten,) = t.cards("H_T")
    assert not ten.debuff
    t.use("c_strength", ten)
    assert ten.front == "H_J" and ten.debuff


# ---------------------------------------------------------------------------
# G15: a perished joker stays debuffed, after the tick its dollar bonus is gone,
# and a Negative keeps its slot until it leaves
# ---------------------------------------------------------------------------


def _board(seed: str, key: str, what: str) -> Run:
    """The first round with ``key`` added without the edition poll, as ``what`` says."""
    run = Run(
        {"back": "b_red", "stake": 1, "seed": seed, "challenge": None},
        [{"op": "action", "action": ["SelectBlind"]}],
    )
    run.add(key)
    run.oracle(f"{key} added without the edition poll, {what}")
    return run


def _win(run: Run) -> dict[str, Any]:
    run.write("game.chips", BIG)
    return check(run, ["PlayHand", [0]])


def test_a_perished_golden_joker_pays_nothing_and_stays_debuffed():
    """card.lua:526-533, 2283; state_events.lua:99-110: the perishable tick runs
    before the dollar bonus, and set_debuff(false) leaves a spent perishable alone."""
    cash = {}
    for label, perishable in (("plain", False), ("perished", True)):
        what = "perishable with one round left" if perishable else "with no sticker"
        run = _board("AS_G15_GOLD", "j_golden", what)
        snap = _win(run)
        assert snap["phase"] == "round_eval"
        cash[label] = snap["sim"]["pending_cashout"]
        assert snap["areas"]["jokers"][0]["debuff"] == perishable
    assert cash["plain"] == cash["perished"] + 4
    check(run, ["CashOut"], ["NextRound"])
    snap = check(run, ["SelectBlind"])
    assert snap["areas"]["jokers"][0]["debuff"]


def test_a_perished_negative_juggler_keeps_its_slot_until_sold():
    """card.lua:687-690, 4733-4739: the debuff queues the slot's removal; selling
    frees it."""
    run = _board("AS_G15_NEG", "j_juggler", "Negative and perishable with one round left")
    run.write("game.chips", BIG)
    run.sync()
    slots = run.read("game.joker_slots")
    hand_size = run.read("game.hand_size")
    snap = check(run, ["PlayHand", [0]])
    card = snap["areas"]["jokers"][0]
    assert card["debuff"] and card["ability"].get("queue_negative_removal")
    assert snap["game"]["joker_slots"] == slots
    assert snap["game"]["hand_size"] == hand_size - 1
    snap = check(run, ["CashOut"])
    assert snap["phase"] == "shop"
    snap = check(run, ["SellCard", "jokers", 0])
    assert snap["game"]["joker_slots"] == slots - 1


# ---------------------------------------------------------------------------
# G34: destroying_card runs for a debuffed scoring card too
# ---------------------------------------------------------------------------


def test_sixth_sense_destroys_a_debuffed_six():
    """state_events.lua:950-960: every scoring card is offered to destroying_card."""
    run = in_round("AS_G34_SIXTH_GOAD", ("j_sixth_sense",), ({"key": "S_6"},), boss="bl_goad")
    debuff_last(run)
    six = last_ids(run)
    snap = check(run, ["PlayHand", last(run)])
    assert not any(c["sort_id"] in six for c in owned(snap))
    assert [c["center"][:2] for c in snap["areas"]["consumables"]] == ["c_"]


# ---------------------------------------------------------------------------
# G35: a secret hand becomes visible when it is played
# ---------------------------------------------------------------------------


def test_playing_five_of_a_kind_reveals_it():
    """state_events.lua:578: evaluate_play sets visible."""
    aces = tuple({"key": k} for k in ("S_A", "H_A", "D_A", "C_A", "S_A"))
    run = in_round("AS_G35_FIVE", (), aces)
    assert not run.read("hands")["Five of a Kind"]["visible"]
    snap = check(run, ["PlayHand", last(run, 5)])
    assert snap["hands"]["Five of a Kind"]["visible"]
    assert snap["hands"]["Five of a Kind"]["played"] == 1


def test_black_hole_leaves_five_of_a_kind_hidden_until_played():
    """common_events.lua:464: level_up_hand doesn't reveal a hand."""
    t = Table("AS_G35_BH")
    t.use("c_black_hole")
    assert not t.run.game.hand_levels["Five of a Kind"].visible
    t.play(*t.hold_only("S_A", "H_A", "D_A", "C_A", "S_A"))
    assert t.run.game.hand_levels["Five of a Kind"].visible


# ---------------------------------------------------------------------------
# B12: chips are doubles
# ---------------------------------------------------------------------------


def test_the_round_score_saturates_at_the_largest_double():
    run = in_round("AS_B12_MAX")
    run.write("game.chips", sys.float_info.max)
    snap = check(run, ["PlayHand", [0, 1, 2, 3, 4]])
    assert snap["game"]["chips"] == sys.float_info.max
    assert snap["phase"] == "round_eval"


def test_an_ante_39_blind_target_is_nan_and_the_round_goes_on():
    """misc_functions.lua:927-929: by ante 39 the amount is infinite and the rounding
    turns it into NaN, a target no score reaches."""
    run = Run(
        {"back": "b_red", "stake": 1, "seed": "AS_B12_ANTE39", "challenge": None},
        [{"op": "set", "ante": 39}],
    )
    snap = check(run, ["SelectBlind"])
    assert snap["blind"]["chips"] != snap["blind"]["chips"]
    snap = check(run, ["PlayHand", [0, 1, 2, 3, 4]])
    assert snap["phase"] == "selecting_hand"


# ---------------------------------------------------------------------------
# G18: To Do List pays in `before`, so Bull and Bootstraps see it in dollar_buffer
# ---------------------------------------------------------------------------


def test_to_do_list_pays_before_bull_and_bootstraps_read_the_buffer():
    """card.lua:3491-3499 (To Do List in context.before), 3936, 4046: the $4
    sits in G.GAME.dollar_buffer when Bull and Bootstraps score."""
    b = Board("AS_TODO_BULL", "j_bull", "j_bootstraps", "j_todo_list")
    with b.run.edit_card(b.joker("j_todo_list")) as card:
        card["ability"]["to_do_poker_hand"] = "Pair"
    b.run.write("game.dollars", 1)
    b.play(*b.cards("S_5", "H_5"))
    # Bootstraps: floor((1 + 4) / 5) = 1, so +2 Mult on Pair's 2.
    assert b.last.mult == 2 + 2


# ---------------------------------------------------------------------------
# G19: every way a playing card joins the deck grows Hologram
# ---------------------------------------------------------------------------


def test_certificate_and_its_copy_grow_hologram():
    """card.lua:2600 (Certificate's playing_card_joker_effects); Blueprint copies it."""
    run = at_blind_select("AS_CERT_HOLO", ("j_blueprint", "j_certificate", "j_hologram"))
    snap = check(run, ["SelectBlind"])
    assert _hologram(snap) == 1.5


def test_marble_joker_grows_hologram():
    """card.lua:2481: Marble's Stone joins through playing_card_joker_effects."""
    run = at_blind_select("AS_MARBLE_HOLO", ("j_marble", "j_hologram"))
    snap = check(run, ["SelectBlind"])
    assert _hologram(snap) == 1.25


def test_dna_grows_hologram_before_it_scores():
    """card.lua:3501-3523: DNA's copy reports in the before pass, so the hand it
    makes already scores the larger x_mult."""
    run = in_round("AS_DNA_HOLO", ("j_dna", "j_hologram"), ({"key": "H_K"},))
    snap = check(run, ["PlayHand", last(run)])
    assert _hologram(snap) == 1.25


def test_cryptid_grows_hologram():
    """card.lua:1217: playing_card_joker_effects(new_cards) after the two copies."""
    t = Table("AS_CRYPTID_HOLO", "j_hologram")
    (card,) = t.cards("D_Q")
    t.use("c_cryptid", card)
    assert t.run.cards("jokers")[0].ability["x_mult"] == 1.5


def test_familiar_grim_and_incantation_grow_hologram():
    """card.lua:1338: each creates its cards, then playing_card_joker_effects."""
    t = Table("AS_SPECTRAL_HOLO", "j_hologram")
    grown = []
    for key in ("c_familiar", "c_grim", "c_incantation"):
        t.use(key)
        grown.append(t.run.cards("jokers")[0].ability["x_mult"])
    assert grown == [1.75, 2.25, 3.25]


# ---------------------------------------------------------------------------
# G21: Smeared, Four Fingers and Shortcut reach the jokers that read them
# ---------------------------------------------------------------------------


def test_flower_pot_with_smeared_joker():
    """card.lua:3840: Two Pair of Hearts and Spades covers Diamonds and Clubs too."""
    b = Board("AS_FLOWER_SMEARED", "j_smeared", "j_flower_pot")
    b.play(*b.cards("H_7", "S_7", "H_9", "S_9"))
    assert b.last.mult == 2 * 3


def test_seeing_double_with_smeared_joker():
    """card.lua:3807: the Spade counts as a Club beside the Heart."""
    b = Board("AS_DOUBLE_SMEARED", "j_smeared", "j_seeing_double")
    b.play(*b.cards("S_7", "H_7"))
    assert b.last.mult == 2 * 2


def test_castle_counts_the_partner_suit_with_smeared_joker():
    """card.lua:2858: a discarded card of the castle suit's colour partner counts."""
    run = in_round("AS_CASTLE_SMEAR", ("j_castle", "j_smeared"))
    suit = run.read("current_round.castle_card.suit")
    partner = {"Clubs": "S", "Spades": "C", "Hearts": "D", "Diamonds": "H"}[suit]
    run.op({"op": "add", "key": f"{partner}_2"})
    snap = check(run, ["Discard", last(run)])
    assert snap_joker(snap, "j_castle")["ability"]["extra"]["chips"] == 3


def _burnt(seed: str, flag: str, keys: tuple[str, ...], hand: str) -> None:
    run = in_round(seed, ("j_burnt", flag), tuple({"key": k} for k in keys))
    snap = check(run, ["Discard", last(run, len(keys))])
    assert snap["hands"][hand]["level"] == 2


def test_burnt_joker_with_four_fingers():
    """card.lua:2748-2760: the first discard's hand is found with Four Fingers."""
    _burnt("AS_BURNT_FF", "j_four_fingers", ("H_2", "H_5", "H_8", "H_J"), "Flush")


def test_burnt_joker_with_shortcut():
    _burnt("AS_BURNT_SC", "j_shortcut", ("H_2", "S_4", "D_6", "C_8", "H_T"), "Straight")


def test_burnt_joker_with_smeared_joker():
    _burnt("AS_BURNT_SM", "j_smeared", ("H_2", "D_5", "H_8", "D_J", "H_K"), "Flush")


# ---------------------------------------------------------------------------
# G23: Oops! All 6s doubles the end-of-round banana roll
# ---------------------------------------------------------------------------


def _goes_extinct_only_with_two_oops(key: str, seed: str, odds: float | None = None) -> None:
    """card.lua:3019-3045: extinct when pseudorandom < normal / odds. The seeds were chosen,
    with the oracle's RNG, so that the roll survives without the Oops and not with both."""
    run = in_round(seed, (key, "j_oops", "j_oops"), chips=BIG)
    if odds is not None:
        with run.edit_card(joker(run, key)) as card:
            card["ability"]["extra"]["odds"] = odds
    snap = check(run, ["PlayHand", [0]])
    assert snap["phase"] == "round_eval"
    assert all(j["center"] != key for j in snap["areas"]["jokers"])


def test_gros_michel_goes_extinct_with_two_oops():
    _goes_extinct_only_with_two_oops("j_gros_michel", "AS_j_gros_michel_OOPS1")


def test_cavendish_goes_extinct_with_two_oops():
    _goes_extinct_only_with_two_oops("j_cavendish", "AS_j_cavendish_OOPS1", odds=6)


# ---------------------------------------------------------------------------
# G32: a debuffed Purple Seal makes no Tarot
# ---------------------------------------------------------------------------


def test_a_debuffed_purple_seal_makes_no_tarot():
    """card.lua:2242-2243: calculate_seal returns nothing for a debuffed card."""
    t = Table("CS_PURPLE_HEAD", boss="bl_head")
    (heart,) = t.cards("H_5", seal="Purple")
    assert heart.debuff
    t.step(["Discard", [t.index(heart)]])
    assert t.keys("consumables") == []
    (spade,) = t.cards("S_5", seal="Purple")  # undebuffed: makes a Tarot
    t.step(["Discard", [t.index(spade)]])
    assert len(t.run.cards("consumables")) == 1
