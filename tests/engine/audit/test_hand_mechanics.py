"""Audit regressions: hand mechanics (play/discard order, sorting, deck emplace,
card bases, hand visibility, blind amounts).

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

from jackdaw.engine.actions import (
    BuyCard,
    Discard,
    OpenBooster,
    PickPackCard,
    PlayHand,
    SelectBlind,
    UseConsumable,
)
from jackdaw.engine.card import Card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from tests.engine.audit._boards import (
    give_consumable,
    give_joker,
    ids,
    index_of,
    new_run,
    playing_card,
    to_shop,
)

# ---------------------------------------------------------------------------
# G49 — played and discarded cards are processed in hand order
# ---------------------------------------------------------------------------


def test_g49_played_cards_score_in_hand_order():
    """play_cards_from_highlighted sorts G.hand.highlighted by T.x — the hand
    position — before moving the cards to G.play (state_events.lua:463), and
    evaluate_play scores them in play-area order (state_events.lua:600).  The
    order the indices were clicked in does not matter."""
    gs = new_run("AUDIT_G49")
    step(gs, SelectBlind())
    give_joker(gs, "j_splash")  # every played card scores
    hand = list(gs["hand"])
    first, second = hand[0], hand[3]

    step(gs, PlayHand(card_indices=(3, 0)))

    assert ids(gs["last_score_result"].scoring_cards) == ids([first, second])
    # draw_from_play_to_discard moves G.play.cards in order (state_events.lua:1091)
    assert ids(gs["discard_pile"]) == ids([first, second])


def test_g49_discarded_cards_follow_hand_order():
    """discard_cards_from_highlighted sorts the highlight by T.x too
    (state_events.lua:392) before the per-card discard loop."""
    gs = new_run("AUDIT_G49D")
    step(gs, SelectBlind())
    hand = list(gs["hand"])

    step(gs, Discard(card_indices=(4, 1)))

    assert ids(gs["discard_pile"]) == ids([hand[1], hand[4]])


# ---------------------------------------------------------------------------
# G50 — Card:set_base keeps suit_nominal_original; original_value is initial-only
# ---------------------------------------------------------------------------


def test_g50_fresh_card_records_original_value():
    """Card:init calls set_base(card, true) (card.lua:44): the initial base
    records original_value and its own suit_nominal_original (card.lua:136-141)."""
    card = playing_card("Hearts", "7")
    assert card.base.original_value is Rank.SEVEN
    assert card.base.suit_nominal_original == 0.003


def test_g50_suit_change_keeps_original_suit_tiebreak():
    """The Star calls change_suit -> set_base(new_card) (card.lua:547-561).  A
    non-initial set_base carries the old base's suit_nominal_original over
    (card.lua:109-110, 138-141) and never sets original_value (card.lua:136),
    so the rebuilt base has none."""
    gs = new_run("AUDIT_G50")
    step(gs, SelectBlind())
    target = next(c for c in gs["hand"] if c.base.suit is not Suit.DIAMONDS)
    original = target.base.suit_nominal_original
    give_consumable(gs, "c_star")

    step(gs, UseConsumable(card_index=0, target_indices=(index_of(gs["hand"], target),)))

    assert target.base.suit is Suit.DIAMONDS
    assert target.base.suit_nominal == 0.01
    assert target.base.suit_nominal_original == original
    assert target.base.original_value is None
    assert target.card_key == "D_" + target.card_key[2:]


def test_g50_rank_change_keeps_original_suit_tiebreak():
    """Strength rebuilds the base with set_base(G.P_CARDS[...]) (card.lua:1121-1134):
    the same non-initial set_base rules apply."""
    gs = new_run("AUDIT_G50S")
    step(gs, SelectBlind())
    target = next(c for c in gs["hand"] if c.base.id < 14)
    old_id, original = target.base.id, target.base.suit_nominal_original
    give_consumable(gs, "c_strength")

    step(gs, UseConsumable(card_index=0, target_indices=(index_of(gs["hand"], target),)))

    assert target.base.id == old_id + 1
    assert target.base.suit_nominal_original == original
    assert target.base.original_value is None


def test_g50_checkered_deck_keeps_original_suit_tiebreak():
    """Checkered Deck converts with change_suit (back.lua:239-253), so a Spade
    that started as a Club keeps the Club's 0.002 tiebreak and sorts below a
    native Spade of the same rank (get_nominal, card.lua:950-955)."""
    gs = new_run("AUDIT_G50C", back="b_checkered")
    cards = gs["deck"]
    assert all(c.base.suit in (Suit.SPADES, Suit.HEARTS) for c in cards)
    aces = [c for c in cards if c.base.id == 14 and c.base.suit is Suit.SPADES]
    assert sorted(c.base.suit_nominal_original for c in aces) == [0.002, 0.004]
    converted = next(c for c in aces if c.base.suit_nominal_original == 0.002)
    native = next(c for c in aces if c.base.suit_nominal_original == 0.004)
    assert converted.card_key == "S_A"
    assert converted.base.original_value is None
    assert native.base.original_value is Rank.ACE
    assert converted.get_nominal() < native.get_nominal()


# ---------------------------------------------------------------------------
# G35 — a hand becomes visible when played, never by leveling
# ---------------------------------------------------------------------------


def test_g35_leveling_a_secret_hand_keeps_it_hidden():
    """level_up_hand (common_events.lua:464-468) only moves level, mult and
    chips; it never touches ``visible``.  A Planet X used before any Five of
    a Kind is played levels the hand but leaves it out of the visible pools
    (To Do List card.lua:314, Telescope card.lua:1740, Obelisk card.lua:3547,
    Orbital Tag UI_definitions.lua:1512)."""
    gs = new_run("AUDIT_G35")
    step(gs, SelectBlind())
    give_consumable(gs, "c_planet_x")

    step(gs, UseConsumable(card_index=0))

    five = gs["hand_levels"].get_state("Five of a Kind")
    assert five.level == 2
    assert not five.visible


def test_g35_black_hole_keeps_secret_hands_hidden():
    """Black Hole levels every hand through level_up_hand (card.lua:1179-1181)."""
    gs = new_run("AUDIT_G35B")
    step(gs, SelectBlind())
    give_consumable(gs, "c_black_hole")

    step(gs, UseConsumable(card_index=0))

    for secret in ("Flush Five", "Flush House", "Five of a Kind"):
        state = gs["hand_levels"].get_state(secret)
        assert state.level == 2
        assert not state.visible


def test_g35_playing_a_secret_hand_reveals_it():
    """evaluate_play sets ``G.GAME.hands[text].visible = true`` for the hand
    it just counted (state_events.lua:574-578)."""
    gs = new_run("AUDIT_G35P")
    step(gs, SelectBlind())
    gs["hand"][:5] = [playing_card("Spades", "Ace") for _ in range(5)]

    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))

    assert gs["last_score_result"].hand_type == "Flush Five"
    assert gs["hand_levels"].get_state("Flush Five").visible


# ---------------------------------------------------------------------------
# G38 — cards emplaced into the deck go to the bottom
# ---------------------------------------------------------------------------


def test_g38_bought_playing_card_goes_to_the_bottom_of_the_deck():
    """buy_from_shop emplaces a bought playing card with G.deck:emplace
    (button_callbacks.lua:2427), and a 'deck' CardArea inserts at index 1
    (cardarea.lua:33-34) — the bottom, drawn last (remove_card takes the
    last card, cardarea.lua:76-77)."""
    gs = to_shop(new_run("AUDIT_G38"))
    card = playing_card("Hearts", "Ace")
    card.set_cost()
    gs["shop_cards"].insert(0, card)
    gs["dollars"] = 50
    top = gs["deck"][-1]

    step(gs, BuyCard(shop_index=0))

    assert gs["deck"][0] is card
    assert gs["deck"][-1] is top


def test_g38_standard_pack_pick_goes_to_the_bottom_of_the_deck():
    """Picking a playing card from a Standard pack: use_card emplaces it with
    G.deck:emplace (button_callbacks.lua:2222-2224)."""
    gs = to_shop(new_run("AUDIT_G38P"))
    pack = Card()
    pack.set_ability("p_standard_normal_1")
    pack.set_cost()
    gs["shop_boosters"] = [pack]
    gs["dollars"] = 50
    step(gs, OpenBooster(card_index=0))
    picked = gs["pack_cards"][0]
    top = gs["deck"][-1]

    step(gs, PickPackCard(card_index=0))

    assert gs["deck"][0] is picked
    assert gs["deck"][-1] is top
