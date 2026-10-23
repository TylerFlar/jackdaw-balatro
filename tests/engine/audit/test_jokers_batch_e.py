"""Joker batch E fidelity: the new_blind_choice pass (G160) and a sale that ends the round
(G161), both found by the batch E parity runs."""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import CashOut, GamePhase, NextRound, SellCard, SkipBlind, SkipPack
from jackdaw.engine.game import _award_tag, _close_pack, _fire_new_blind_choice_tags, step
from tests.parity.py_export import pending_tags

from ._helpers import force_boss
from ._joker_helpers import give_joker, new_run, select_blind


def _pending(gs: dict[str, Any]) -> list[str]:
    return [t["key"] for t in pending_tags(gs.get("awarded_tags", []))]


class TestNewBlindChoicePass:
    """G160: every new_blind_choice pass stops at the first tag that fires, ``if
    G.GAME.tags[i]:apply_to_run({type = 'new_blind_choice'}) then break end``
    (button_callbacks.lua:2775-2777 after a skip, 2847-2849 after a boss reroll,
    2617-2619 when a pack closes), and every tag that fires returns true (tag.lua:206-301).
    A second tag waits for the next pass."""

    def test_a_double_tag_copy_of_a_pack_tag_waits_for_the_first_pack(self):
        gs = new_run(seed="AUDIT_G160_DOUBLE")
        _award_tag(gs, "tag_double", source="test")
        gs["round_resets"]["blind_tags"]["Small"] = "tag_buffoon"
        step(gs, SkipBlind())
        assert gs["phase"] == GamePhase.PACK_OPENING
        assert _pending(gs) == ["tag_buffoon"]
        assert not gs.get("pending_tag_packs")
        first = [c.sort_id for c in gs["pack_cards"]]
        # end_consumeable's pass opens the copy's pack once the first one closes.
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.PACK_OPENING
        assert _pending(gs) == []
        assert min(c.sort_id for c in gs["pack_cards"]) > max(first)
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.BLIND_SELECT

    def test_a_boss_tag_behind_a_pack_tag_waits_for_the_pack(self):
        gs = new_run(seed="AUDIT_G160_BOSS")
        gs["awarded_tags"] = [
            {"key": "tag_charm", "result": None, "blind": "test"},
            {"key": "tag_boss", "result": None, "blind": "test"},
        ]
        boss = gs["round_resets"]["blind_choices"]["Boss"]
        _fire_new_blind_choice_tags(gs)
        assert gs["phase"] == GamePhase.PACK_OPENING
        assert gs["round_resets"]["blind_choices"]["Boss"] == boss
        assert not gs["round_resets"]["boss_rerolled"]
        _close_pack(gs)
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert gs["round_resets"]["boss_rerolled"]
        assert _pending(gs) == []

    def test_a_boss_tag_reroll_runs_its_own_pass(self):
        """G.FUNCS.reroll_boss's event runs a pass after the new boss (2847-2849): two Boss
        Tags reroll twice in one skip."""
        gs = new_run(seed="AUDIT_G160_TWO_BOSSES")
        _award_tag(gs, "tag_double", source="test")
        gs["round_resets"]["blind_tags"]["Small"] = "tag_boss"
        step(gs, SkipBlind())
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert _pending(gs) == []
        assert sum(gs["bosses_used"].values()) == 3


def _wall_round(seed: str, *jokers: str) -> dict[str, Any]:
    """A round against The Wall, scored past half its target."""
    gs = new_run(seed=seed)
    force_boss(gs, "bl_wall")
    select_blind(gs)
    for key in jokers:
        give_joker(gs, key)
    gs["blind"].chips = 2000
    gs["chips"] = 1200
    return gs


def _sell(gs: dict[str, Any], key: str) -> None:
    index = next(i for i, j in enumerate(gs["jokers"]) if j.center_key == key)
    step(gs, SellCard(area="jokers", card_index=index))


class TestSaleThatEndsTheRound:
    """G161: Luchador's Blind:disable only queues the beaten-boss check (blind.lua:397-406),
    and Card:sell_card queues its payout and dissolve (card.lua:1601-1637). G.FUNCS.sell_card
    runs the selling_card loop first (button_callbacks.lua:2318-2326); the check's NEW_ROUND
    state then queues end_round (game.lua:3248-3256), which runs while the sold card is still
    held (its dissolve removes it on a timer) and before the payout's own ease_dollars
    event; evaluate_round's interest counts the payout."""

    def test_the_payout_counts_for_interest(self):
        gs = _wall_round("AUDIT_G161_INTEREST", "j_luchador")
        luchador = gs["jokers"][0]
        gs["dollars"] = 5 - luchador.sell_cost
        _sell(gs, "j_luchador")
        assert gs["phase"] == GamePhase.ROUND_EVAL
        assert gs["round_earnings"].interest == 1
        assert gs["jokers"] == []

    def test_campfire_grows_before_the_boss_resets_it(self):
        """Campfire takes the sale (card.lua:2396-2402), then end_round resets it after the
        boss (card.lua:2889-2895)."""
        gs = _wall_round("AUDIT_G161_CAMPFIRE", "j_campfire", "j_luchador")
        _sell(gs, "j_luchador")
        assert gs["phase"] == GamePhase.ROUND_EVAL
        assert gs["jokers"][0].ability["x_mult"] == 1

    def test_a_sold_rental_pays_its_rent(self):
        """end_round's joker loop still finds the sold card: its calculate_rental charges
        (state_events.lua:99-110, card.lua:2271-2276)."""
        gs = _wall_round("AUDIT_G161_RENTAL")
        luchador = give_joker(gs, "j_luchador", rental=True)
        dollars = gs["dollars"]
        _sell(gs, "j_luchador")
        assert gs["dollars"] == dollars + luchador.sell_cost - gs["rental_rate"]

    def test_the_round_ends_once(self):
        """The next shop follows as usual."""
        gs = _wall_round("AUDIT_G161_SHOP", "j_luchador", "j_golden")
        _sell(gs, "j_luchador")
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        step(gs, NextRound())
        assert gs["round_resets"]["ante"] == 2
