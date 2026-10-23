"""copy_card fidelity (G29, G30, G28 copy half).

Vanilla duplicates cards through ``copy_card`` (common_events.lua:2156-2181):
a *new* ``Card`` (Card:init assigns the next ``G.sort_id``, card.lua:24-25),
``set_ability`` + ``set_base`` from the source, then every ``ability`` entry
deep-copied, the edition (unless stripped), the seal, ``debuff`` and
``pinned``.
"""

from __future__ import annotations

from jackdaw.engine import card as card_module
from jackdaw.engine.actions import NextRound, SellCard, UseConsumable
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker, roll_to_do_hand
from jackdaw.engine.game import step

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


def _next_sort_id() -> int:
    return card_module._sort_id_counter + 1


class TestCardIdentity:
    def test_cards_compare_by_identity(self):
        """Lua cards are tables compared by reference: two cards with equal
        fields are still different cards (common_events.lua:2157 creates a new
        Card for every copy).  Value equality let ``list.remove`` take the wrong
        twin."""
        a = Card(sort_id=7)
        b = Card(sort_id=7)
        assert a != b
        cards = [a, b]
        cards.remove(b)
        assert cards[0] is a
        assert len({a, b}) == 2


class TestFreshSortId:
    def test_perkeo_copy_is_a_new_card(self):
        """Perkeo's copy (card.lua:2417) is a fresh Card with the next sort_id."""
        gs = to_shop(new_run("AUDIT_PERKEO"))
        give_joker(gs, "j_perkeo")
        source = give_consumable(gs, "c_pluto")
        expected_id = _next_sort_id()
        step(gs, NextRound())
        copies = [c for c in gs["consumables"] if c is not source]
        assert len(copies) == 1
        copy = copies[0]
        assert copy.sort_id == expected_id
        assert copy.sort_id != source.sort_id
        assert copy.edition and copy.edition.get("negative")
        assert not source.edition

    def test_invisible_joker_copy_is_a_new_card(self):
        """Invisible Joker's copy (card.lua:2384) is a fresh Card."""
        gs = to_shop(new_run("AUDIT_INVIS"))
        invis = give_joker(gs, "j_invisible")
        target = give_joker(gs, "j_joker")
        invis.ability["invis_rounds"] = 2
        expected_id = _next_sort_id()
        step(gs, SellCard(area="jokers", card_index=0))
        copies = [j for j in gs["jokers"] if j is not target]
        assert len(copies) == 1
        assert copies[0].center_key == "j_joker"
        assert copies[0].sort_id == expected_id

    def test_ankh_copy_is_a_new_card(self):
        """Ankh's copy (card.lua:1445) is a fresh Card."""
        gs = select_blind(new_run("AUDIT_ANKH"))
        target = give_joker(gs, "j_joker")
        give_consumable(gs, "c_ankh")
        expected_id = _next_sort_id()
        step(gs, UseConsumable(card_index=0))
        assert len(gs["jokers"]) == 2
        copy = gs["jokers"][1]
        assert copy is not target and copy.center_key == "j_joker"
        assert copy.sort_id == expected_id


class TestCopySemantics:
    def test_copying_to_do_list_pulls_the_to_do_stream(self):
        """copy_card runs set_ability on the new card (common_events.lua:2158),
        and To Do List's set_ability rolls a target hand on 'to_do'
        (card.lua:311-322) before the ability copy restores the source's."""
        gs = to_shop(new_run("AUDIT_TODO"))
        invis = give_joker(gs, "j_invisible")
        todo = give_joker(gs, "j_todo_list")
        roll_to_do_hand(todo, gs["rng"], gs["hand_levels"])
        invis.ability["invis_rounds"] = 2
        oracle = rng_twin(gs)
        oracle.seed("invisible")  # the copy target pick
        oracle.seed("to_do")  # one pull: a fresh card has no old hand to repeat
        step(gs, SellCard(area="jokers", card_index=0))
        copy = gs["jokers"][1]
        assert copy is not todo
        assert copy.ability["to_do_poker_hand"] == todo.ability["to_do_poker_hand"]
        assert gs["rng"].state["to_do"] == oracle.state["to_do"]

    def test_cryptid_copy_has_a_fresh_base(self):
        """copy_card's set_base (common_events.lua:2160) builds a new base
        table, so ``times_played`` starts at 0 (card.lua:111-120); the ability
        is a deep copy."""
        gs = select_blind(new_run("AUDIT_CRYPTID"))
        source = give_card(gs, "H_K", enhancement="m_bonus")
        source.base.times_played = 3
        source.ability["perma_bonus"] = 15
        give_consumable(gs, "c_cryptid")
        hand_index = gs["hand"].index(source)
        expected_ids = [_next_sort_id(), _next_sort_id() + 1]
        step(gs, UseConsumable(card_index=0, target_indices=(hand_index,)))
        copies = [c for c in gs["hand"] if c is not source and c.card_key == "H_K"]
        assert sorted(c.sort_id for c in copies) == expected_ids
        for c in copies:
            assert c.base.times_played == 0
            assert c.ability["perma_bonus"] == 15
            assert c.center_key == "m_bonus"
        copies[0].ability["perma_bonus"] = 99
        assert source.ability["perma_bonus"] == 15
        assert source.base.times_played == 3

    def test_death_copies_the_whole_card_onto_the_target(self):
        """Death runs copy_card(rightmost, target) (card.lua:1111-1119): the
        target takes the source's center, base, *every* ability entry
        (perma_bonus and played_this_ante included, common_events.lua:2161-2167),
        edition (with its scoring values), seal and debuff — but keeps its
        own sort_id."""
        gs = select_blind(new_run("AUDIT_DEATH"))
        target = give_card(gs, "H_3")
        target.ability["perma_bonus"] = 7
        source = give_card(gs, "S_A", enhancement="m_bonus", seal="Red", edition="foil")
        source.ability["perma_bonus"] = 20
        source.ability["played_this_ante"] = True
        source.debuff = True
        target_id = target.sort_id
        give_consumable(gs, "c_death")
        step(gs, UseConsumable(card_index=0, target_indices=indices(gs, target, source)))
        assert target.sort_id == target_id
        assert target.card_key == "S_A" and target.base.id == 14
        assert target.center_key == "m_bonus"
        assert target.ability["bonus"] == source.ability["bonus"]
        assert target.ability["perma_bonus"] == 20
        assert target.ability["played_this_ante"] is True
        assert target.debuff is True
        assert target.seal == "Red"
        assert target.edition == {"foil": True, "chips": 50, "type": "foil"}
        assert target.edition is not source.edition

    def test_dna_copy_is_a_held_card_of_the_same_hand(self):
        """DNA copies the played card with copy_card in the ``before`` pass and
        emplaces the copy into G.hand at once (card.lua:3501-3508), so it is a
        held card for the rest of that evaluation (state_events.lua:784): a
        copied Steel card's x1.5 applies to the hand that created it.  The
        copy keeps the source's perma_bonus and played_this_ante."""
        gs = select_blind(new_run("AUDIT_DNA"))
        give_joker(gs, "j_dna")
        source = give_card(gs, "S_K", enhancement="m_steel")
        source.ability["perma_bonus"] = 10
        hand_before = list(gs["hand"])
        play(gs, source)
        # High Card L1 (5 chips, x1) + King 10 + perma_bonus 10, held copy x1.5
        assert gs["last_score_result"].total == int((5 + 10 + 10) * 1.5)
        copies = [c for c in gs["hand"] if c not in hand_before]
        copies = [c for c in copies if c.card_key == "S_K" and c is not source]
        assert len(copies) == 1
        copy = copies[0]
        assert copy.center_key == "m_steel"
        assert copy.ability["perma_bonus"] == 10
        assert copy.ability["played_this_ante"] is True

    def test_ankh_copy_drops_negative(self):
        """Ankh calls copy_card(chosen, nil, nil, nil, chosen.edition and
        chosen.edition.negative) (card.lua:1445): a Negative joker's copy has
        no edition, so it does not add a joker slot."""
        gs = select_blind(new_run("AUDIT_ANKH_NEG"))
        neg = give_joker(gs, "j_joker", edition={"negative": True})
        slots = gs["joker_slots"]
        give_consumable(gs, "c_ankh")
        step(gs, UseConsumable(card_index=0))
        copy = gs["jokers"][1]
        assert copy is not neg and copy.center_key == "j_joker"
        assert copy.edition is None
        assert gs["joker_slots"] == slots

    def test_ankh_copy_keeps_other_editions(self):
        """Only Negative is stripped (card.lua:1445); copy_card copies any other
        edition through set_edition (common_events.lua:2169-2171)."""
        gs = select_blind(new_run("AUDIT_ANKH_FOIL"))
        give_joker(gs, "j_joker", edition={"foil": True})
        give_consumable(gs, "c_ankh")
        step(gs, UseConsumable(card_index=0))
        assert gs["jokers"][1].edition == {"foil": True, "chips": 50, "type": "foil"}

    def test_invisible_joker_copy_drops_negative(self):
        """Invisible Joker strips Negative the same way (card.lua:2384)."""
        gs = to_shop(new_run("AUDIT_INVIS_NEG"))
        invis = give_joker(gs, "j_invisible")
        give_joker(gs, "j_joker", edition={"negative": True})
        invis.ability["invis_rounds"] = 2
        slots_after_sale = gs["joker_slots"]
        step(gs, SellCard(area="jokers", card_index=0))
        copy = gs["jokers"][1]
        assert copy.edition is None
        assert gs["joker_slots"] == slots_after_sale

    def test_copy_does_not_share_ability_tables(self):
        """Tables are copied with copy_table (common_events.lua:2161-2166)."""
        gs = to_shop(new_run("AUDIT_INVIS2"))
        invis = give_joker(gs, "j_invisible")
        bean = give_joker(gs, "j_turtle_bean")
        invis.ability["invis_rounds"] = 2
        step(gs, SellCard(area="jokers", card_index=0))
        copy = gs["jokers"][1]
        copy.ability["extra"]["h_size"] = 1
        assert (
            bean.ability["extra"]["h_size"]
            == create_joker("j_turtle_bean").ability["extra"]["h_size"]
        )


class TestCopyBaseIsNotInitial:
    """copy_card starts from Card:init's empty base (common_events.lua:2157)
    and then calls a non-initial ``set_base`` (2160): the copy never records
    ``original_value`` (card.lua:136), and its suit tiebreak is its suit's own
    (card.lua:138-141)."""

    def test_cryptid_copy_has_no_original_value(self):
        gs = select_blind(new_run("AUDIT_CRYPTID_ORIG"))
        source = give_card(gs, "S_7")
        assert source.base.original_value is not None
        give_consumable(gs, "c_cryptid")
        step(gs, UseConsumable(card_index=0, target_indices=(gs["hand"].index(source),)))
        copies = [c for c in gs["hand"] if c is not source and c.card_key == "S_7"]
        assert copies
        for c in copies:
            assert c.base.original_value is None
            assert c.base.suit_nominal_original == source.base.suit_nominal_original
