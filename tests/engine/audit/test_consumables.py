"""Consumable fidelity (G26, G27, G28, G31, G23 Wheel of Fortune)."""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import UseConsumable, get_legal_actions
from jackdaw.engine.game import IllegalActionError, step

from ._helpers import (
    give_card,
    give_consumable,
    give_joker,
    indices,
    new_run,
    rng_twin,
    select_blind,
)

# Card:set_edition fills in the scoring value (card.lua:390-418) and the
# set_cost surcharge per edition (card.lua:372-373).
_EDITION_TABLES = {
    "foil": {"foil": True, "chips": 50, "type": "foil"},
    "holo": {"holo": True, "mult": 10, "type": "holo"},
    "polychrome": {"polychrome": True, "x_mult": 1.5, "type": "polychrome"},
    "negative": {"negative": True, "type": "negative"},
}
_SURCHARGE = {"foil": 2, "holo": 3, "polychrome": 5, "negative": 5}


def _expected_costs(base_cost: int, edition_type: str) -> tuple[int, int]:
    """Card:set_cost with no inflation/discount (card.lua:369-385)."""
    cost = max(1, int(base_cost + _SURCHARGE[edition_type] + 0.5))
    return cost, max(1, cost // 2)


class TestConsumableEditions:
    """The Wheel of Fortune, Aura, Ectoplasm and Hex all end in
    ``card:set_edition(edition, true)`` (card.lua:1197, 1486), which builds
    the edition table with its scoring value and runs set_cost (card.lua:461)."""

    def test_hex_polychrome_scores_and_reprices(self):
        gs = select_blind(new_run("AUDIT_HEX"))
        target = give_joker(gs, "j_joker")
        give_consumable(gs, "c_hex")
        step(gs, UseConsumable(card_index=0))
        assert target.edition == _EDITION_TABLES["polychrome"]
        assert target.get_edition()["x_mult_mod"] == 1.5
        assert (target.cost, target.sell_cost) == _expected_costs(2, "polychrome")

    def test_ectoplasm_negative_reprices_and_adds_a_slot(self):
        gs = select_blind(new_run("AUDIT_ECTO"))
        target = give_joker(gs, "j_joker")
        slots = gs["joker_slots"]
        give_consumable(gs, "c_ectoplasm")
        step(gs, UseConsumable(card_index=0))
        assert target.edition == _EDITION_TABLES["negative"]
        assert (target.cost, target.sell_cost) == _expected_costs(2, "negative")
        assert gs["joker_slots"] == slots + 1

    def test_wheel_of_fortune_edition_scores_and_reprices(self):
        gs = select_blind(new_run("AUDIT_WHEEL"))
        give_joker(gs, "j_oops")
        give_joker(gs, "j_oops")  # probabilities.normal = 4: the 1-in-4 roll always hits
        target = give_joker(gs, "j_joker")
        for oops in gs["jokers"][:2]:
            oops.edition = {"foil": True, "chips": 50, "type": "foil"}  # not eligible
        give_consumable(gs, "c_wheel_of_fortune")
        step(gs, UseConsumable(card_index=0))
        assert target.edition is not None
        kind = target.edition["type"]
        assert target.edition == _EDITION_TABLES[kind]
        assert (target.cost, target.sell_cost) == _expected_costs(2, kind)

    def test_aura_edition_scores_and_reprices(self):
        gs = select_blind(new_run("AUDIT_AURA"))
        target = give_card(gs, "H_7")
        give_consumable(gs, "c_aura")
        step(gs, UseConsumable(card_index=0, target_indices=indices(gs, target)))
        kind = target.edition["type"]
        assert target.edition == _EDITION_TABLES[kind]
        assert target.get_edition()
        assert (target.cost, target.sell_cost) == _expected_costs(1, kind)


class TestEctoplasm:
    def test_hand_size_penalty_grows_each_use(self):
        """G.GAME.ecto_minus starts at 1 (game.lua:1896); each Ectoplasm costs
        ecto_minus hand size and then increments it (card.lua:1494-1498)."""
        gs = select_blind(new_run("AUDIT_ECTO2"))
        give_joker(gs, "j_joker")
        give_joker(gs, "j_greedy_joker")
        size = gs["hand_size"]
        give_consumable(gs, "c_ectoplasm")
        step(gs, UseConsumable(card_index=0))
        assert gs["hand_size"] == size - 1
        assert gs["ecto_minus"] == 2
        give_consumable(gs, "c_ectoplasm")
        step(gs, UseConsumable(card_index=0))
        assert gs["hand_size"] == size - 1 - 2
        assert gs["ecto_minus"] == 3


class TestTheSoul:
    def test_edition_poll_uses_the_sou_append(self):
        """The Soul creates with create_card('Joker', G.jokers, true, nil, nil,
        nil, nil, 'sou') (card.lua:1418), so its edition is polled on
        'edi'..'sou'..ante (common_events.lua:2149)."""
        gs = select_blind(new_run("AUDIT_SOUL"))
        give_consumable(gs, "c_soul")
        step(gs, UseConsumable(card_index=0))
        assert gs["jokers"] and gs["jokers"][0].ability["set"] == "Joker"
        assert "edisou1" in gs["rng"].state
        assert "edisoul1" not in gs["rng"].state


class TestDebuffedJokersAreEligible:
    """The Wheel's eligible_strength_jokers and Ectoplasm/Hex's
    eligible_editionless_jokers are every editionless G.jokers card, debuffed
    or not (Card:update, card.lua:4209-4224)."""

    def test_hex_targets_a_debuffed_joker(self):
        gs = select_blind(new_run("AUDIT_HEX_DEBUFF"))
        target = give_joker(gs, "j_joker")
        target.debuff = True
        give_consumable(gs, "c_hex")
        assert UseConsumable(card_index=0) in get_legal_actions(gs)
        step(gs, UseConsumable(card_index=0))
        assert target.edition == _EDITION_TABLES["polychrome"]

    def test_ectoplasm_pool_includes_debuffed_jokers(self):
        """With one debuffed and one clean joker the 'ectoplasm' pick draws from
        both (pseudorandom_element over the sort_id-ordered pool)."""
        for i in range(50):  # a seed whose pick is the debuffed joker
            gs = select_blind(new_run(f"AUDIT_ECTO_DEBUFF{i}"))
            first = give_joker(gs, "j_joker")
            second = give_joker(gs, "j_greedy_joker")
            first.debuff = True
            oracle = rng_twin(gs)
            expected, _ = oracle.element([first, second], oracle.seed("ectoplasm"))
            if expected is first:
                break
        assert expected is first
        give_consumable(gs, "c_ectoplasm")
        step(gs, UseConsumable(card_index=0))
        assert first.edition == _EDITION_TABLES["negative"]
        assert second.edition is None


class TestAnkhCheckUse:
    def test_ankh_cannot_be_used_with_full_joker_slots(self):
        """use_card returns early when Card:check_use says so
        (button_callbacks.lua:2163-2169); Ankh's check_use is true while
        #G.jokers.cards >= G.jokers.config.card_limit (card.lua:1581-1588).
        The card is not consumed and nothing happens."""
        gs = select_blind(new_run("AUDIT_ANKH_FULL"))
        for key in ("j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker", "j_mad"):
            give_joker(gs, key)
        assert len(gs["jokers"]) == gs["joker_slots"]
        give_consumable(gs, "c_ankh")
        assert UseConsumable(card_index=0) not in get_legal_actions(gs)
        with pytest.raises(IllegalActionError):
            step(gs, UseConsumable(card_index=0))
        assert [c.center_key for c in gs["consumables"]] == ["c_ankh"]
        assert len(gs["jokers"]) == 5

    def test_ankh_usable_with_a_free_slot(self):
        gs = select_blind(new_run("AUDIT_ANKH_ROOM"))
        for key in ("j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker"):
            give_joker(gs, key)
        give_consumable(gs, "c_ankh")
        assert UseConsumable(card_index=0) in get_legal_actions(gs)
