"""Consumable fidelity found by the Rust port (G130-G137).

Each test drives the oracle through its public ``step`` and checks the behaviour the vanilla
Lua source fixes; the Lua lines are cited in each docstring.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import (
    BuyAndUse,
    OpenBooster,
    PickPackCard,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_consumable
from jackdaw.engine.game import IllegalActionError, _reprice, step

from ._joker_helpers import (
    give_card,
    give_consumable,
    give_joker,
    indices,
    new_run,
    play,
    rng_twin,
    select_blind,
    to_shop,
)


def _use(gs, card, *targets):
    step(gs, UseConsumable(card_index=gs["consumables"].index(card), target_indices=targets))


class TestEnhancingTarotsStampHandsPlayed:
    def test_magician_stamps_hands_played_at_create(self):
        """G130: set_ability(G.P_CENTERS[mod_conv]) (card.lua:1141-1144) stamps
        ``hands_played_at_create = G.GAME.hands_played`` like every set_ability
        (card.lua:337)."""
        gs = select_blind(new_run("AUDIT_G130"))
        gs["blind"].chips = 10**9
        play(gs, give_card(gs, "S_5"), give_card(gs, "H_5"))
        assert gs["hands_played"] == 1
        target = give_card(gs, "D_7")
        _use(gs, give_consumable(gs, "c_magician"), *indices(gs, target))
        assert target.center_key == "m_lucky"
        assert target.ability["hands_played_at_create"] == 1


class TestSpectralCardsStampHandsPlayed:
    @pytest.mark.parametrize("key", ["c_familiar", "c_grim", "c_incantation"])
    def test_created_cards_stamp_hands_played(self, key):
        """G131: create_playing_card builds each card with Card:init, whose
        set_ability stamps G.GAME.hands_played (common_events.lua:1927-1942,
        card.lua:337; card.lua:1336)."""
        gs = select_blind(new_run(f"AUDIT_G131_{key}"))
        gs["blind"].chips = 10**9
        play(gs, give_card(gs, "S_5"), give_card(gs, "H_5"))
        consumable = give_consumable(gs, key)
        extra = consumable.ability["extra"]
        _use(gs, consumable)
        new = gs["hand"][-extra:]
        assert all(c.ability["set"] == "Enhanced" for c in new)
        assert [c.ability["hands_played_at_create"] for c in new] == [1] * extra


def _open_pack(gs, pack_key: str, *consumables: str) -> list:
    """Open ``pack_key`` from the shop and replace its cards with these consumables."""
    pack = Card()
    pack.set_ability(pack_key, hands_played=gs["hands_played"])
    _reprice(gs, pack)
    pack.cost = 0
    gs["shop_boosters"][0] = pack
    step(gs, OpenBooster(card_index=0))
    cards = []
    for key in consumables:
        card = create_consumable(key, hands_played=gs["hands_played"])
        _reprice(gs, card)
        cards.append(card)
    gs["pack_cards"] = cards
    return cards


class TestPackCardsJoinTheDealtHand:
    def test_familiar_cards_join_the_dealt_hand_and_return_with_it(self):
        """G132: Familiar makes its cards with create_playing_card(..., G.hand)
        (card.lua:1336), the pack's dealt hand, and end_consumeable's
        draw_from_hand_to_deck returns every held card to the deck
        (button_callbacks.lua:2592, state_events.lua:1121-1126): the new cards
        end up at the bottom of the deck, the last made lowest."""
        gs = to_shop(new_run("AUDIT_G132"))
        (familiar,) = _open_pack(gs, "p_spectral_normal_1", "c_familiar")
        deck_before = len(gs["deck"])
        dealt = len(gs["hand"])
        step(gs, PickPackCard(card_index=0, target_indices=()))
        assert gs["hand"] == [] and gs["pack_cards"] == []
        assert len(gs["deck"]) == deck_before + dealt - 1 + familiar.ability["extra"]
        newest = sorted(gs["deck"], key=lambda c: c.sort_id)[-3:]
        assert gs["deck"][:3] == newest[::-1]

    def test_cryptid_copies_join_the_dealt_hand(self):
        """G132: Cryptid's copies go to G.hand (card.lua:1210-1213) in a pack too."""
        gs = to_shop(new_run("AUDIT_G132_CRYPTID"))
        _open_pack(gs, "p_spectral_mega_1", "c_cryptid", "c_aura")
        source = gs["hand"][2]
        dealt = len(gs["hand"])
        step(gs, PickPackCard(card_index=0, target_indices=(2,)))
        assert len(gs["hand"]) == dealt + 2
        assert [c.card_key for c in gs["hand"][-2:]] == [source.card_key] * 2


class TestHandSizeDuringAUse:
    def test_ankh_copy_of_a_juggler_draws_nothing(self):
        """G133: use_card sets G.STATE = PLAY_TAROT until its last event
        (button_callbacks.lua:2181-2186, 2253-2256); the change_size event the
        copy's add_to_deck queues runs before that, and CardArea:change_size
        draws only in DRAW_TO_HAND or SELECTING_HAND (cardarea.lua:94-112)."""
        gs = select_blind(new_run("AUDIT_G133_ANKH"))
        give_joker(gs, "j_juggler")
        size, held = gs["hand_size"], len(gs["hand"])
        _use(gs, give_consumable(gs, "c_ankh"))
        assert gs["hand_size"] == size + 1
        assert len(gs["hand"]) == held

    def test_hex_removing_merry_andy_draws_nothing(self):
        """G133: Hex's dissolved jokers are removed before use_card's last event
        (card.lua:1488-1493): Merry Andy's hand size comes back in PLAY_TAROT."""
        gs = select_blind(new_run("AUDIT_G133_HEX"))
        give_joker(gs, "j_joker")
        give_joker(gs, "j_merry_andy", edition={"foil": True})
        size, held = gs["hand_size"], len(gs["hand"])
        _use(gs, give_consumable(gs, "c_hex"))
        assert gs["hand_size"] == size + 1
        assert len(gs["hand"]) == held


class TestTheSoulRollsItsRarity:
    def test_the_soul_draws_the_rarity_roll(self):
        """G134: get_current_pool draws pseudorandom('rarity'..ante.._append) whenever
        no _rarity is passed (common_events.lua:1969-1970); The Soul passes none
        (card.lua:1418), so 'rarity1sou' advances before Legendary is forced."""
        gs = select_blind(new_run("AUDIT_G134"))
        twin = rng_twin(gs)
        _use(gs, give_consumable(gs, "c_soul"))
        twin.random("rarity1sou")
        assert gs["rng"].state["rarity1sou"] == twin.state["rarity1sou"]
        assert gs["jokers"][0].ability["set"] == "Joker"


class TestBuyAndUseEveryConsumable:
    def test_wraith_can_be_bought_and_used(self):
        """G135: the shop gives every consumable a buy-and-use button
        (UI_definitions.lua:335: ``card.ability.consumeable`` is the center's
        config, a table, truthy even when empty)."""
        gs = to_shop(new_run("AUDIT_G135"))
        wraith = create_consumable("c_wraith", hands_played=gs["hands_played"])
        _reprice(gs, wraith)
        gs["shop_cards"][0] = wraith
        gs["dollars"] = 20
        action = BuyAndUse(shop_index=0, target_indices=None)
        assert action in get_legal_actions(gs)
        step(gs, action)
        assert len(gs["jokers"]) == 1 and gs["dollars"] == 0


class TestBuyAndUseAnkhWithFullSlots:
    def test_ankh_is_bought_then_check_use_stops_it(self):
        """G138: can_buy_and_use asks can_use_consumeable alone
        (button_callbacks.lua:77-89), which for Ankh needs a joker and a joker
        limit above 1 (card.lua:1536-1541).  buy_from_shop pays and calls
        use_card (2404-2477), whose check_use stops the use with every joker slot
        taken (2163-2169, card.lua:1574-1581): the card is paid for and gone."""
        gs = to_shop(new_run("AUDIT_G138"))
        for key in ("j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker", "j_mad"):
            give_joker(gs, key)
        ankh = create_consumable("c_ankh", hands_played=gs["hands_played"])
        _reprice(gs, ankh)
        gs["shop_cards"][0] = ankh
        gs["used_jokers"]["c_ankh"] = True
        gs["dollars"] = 10
        action = BuyAndUse(shop_index=0, target_indices=None)
        assert action in get_legal_actions(gs)
        step(gs, action)
        assert gs["dollars"] == 10 - ankh.cost
        assert ankh not in gs["shop_cards"] and ankh not in gs["consumables"]
        assert len(gs["jokers"]) == 5
        assert gs["used_jokers"].get("c_ankh")


class TestAnkhRemovesBeforeItCopies:
    def test_removed_merry_andy_takes_the_discard_before_the_copy_adds_one(self):
        """G136: Ankh's first event dissolves the jokers and blocks for 0.75s,
        longer than the dissolve (card.lua:1435-1443, 2130-2181), so Card:remove
        runs before the copy event (card.lua:1444-1453).  ease_discard stops at
        0 (common_events.lua:113-116): removing Merry Andy first takes the one
        discard left, then the Drunkard copy adds one."""
        for i in range(30):
            gs = select_blind(new_run(f"AUDIT_G136_{i}"))
            drunk = give_joker(gs, "j_drunkard")
            give_joker(gs, "j_merry_andy")
            twin = rng_twin(gs)
            order = sorted(gs["jokers"], key=lambda c: c.sort_id)
            chosen, _ = twin.element(order, twin.seed("ankh_choice"))
            if chosen is drunk:
                break
        assert chosen is drunk
        gs["current_round"]["discards_left"] = 1
        _use(gs, give_consumable(gs, "c_ankh"))
        assert [j.center_key for j in gs["jokers"]] == ["j_drunkard", "j_drunkard"]
        assert gs["current_round"]["discards_left"] == 1


class TestCeruleanBellForcedCard:
    def test_a_one_card_tarot_takes_only_the_forced_card(self):
        """G137: the forced card is always highlighted (CardArea:update,
        cardarea.lua:252-256; remove_from_highlighted keeps it, 187-188), so The
        Lovers (max_highlighted 1) works on it alone (card.lua:1564-1567)."""
        gs = new_run("AUDIT_G137")
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_final_bell"
        gs["blind_on_deck"] = "Boss"
        select_blind(gs)
        forced = next(c for c in gs["hand"] if c.ability.get("forced_selection"))
        other = next(i for i, c in enumerate(gs["hand"]) if c is not forced)
        lovers = give_consumable(gs, "c_lovers")
        with pytest.raises(IllegalActionError):
            _use(gs, lovers, other)
        _use(gs, lovers)
        assert forced.center_key == "m_wild"
