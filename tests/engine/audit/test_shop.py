"""Audit regressions: packs and the shop (G43, G44, G47)."""

from __future__ import annotations

import copy

from jackdaw.engine import GamePhase, RedeemVoucher, SkipBlind, step
from jackdaw.engine.card_factory import create_card, create_joker, create_voucher
from jackdaw.engine.data.prototypes import JOKERS
from jackdaw.engine.packs import generate_pack_cards
from tests.engine.audit._helpers import new_run, play_round


def _with_hallucination(gs: dict) -> dict:
    halu = create_joker("j_hallucination")
    gs["jokers"].append(halu)
    halu.add_to_deck(gs)
    return gs


# ---------------------------------------------------------------------------
# G43 — packs opened by a tag
# ---------------------------------------------------------------------------


def test_tag_pack_runs_open_booster_and_inflation():
    """A tag's pack opens through Card:open like a shop pack.

    tag.lua:209-282: pack tags open a free booster through
    ``G.FUNCS.use_card`` -> ``Card:open``, which runs every joker's
    ``open_booster`` context (card.lua:1796-1798: Hallucination's 'halu'
    roll) and, under Inflation, raises ``G.GAME.inflation``
    (card.lua:1800-1807) even though the pack was free.
    """
    gs = _with_hallucination(new_run(challenge="c_inflation_1", seed="AUDIT_G43"))
    gs["round_resets"]["blind_tags"]["Small"] = "tag_charm"
    step(gs, SkipBlind())
    assert gs["phase"] == GamePhase.PACK_OPENING
    assert gs["inflation"] == 1
    assert "halu1" in gs["rng"].state


# ---------------------------------------------------------------------------
# G44 — Hallucination's Tarot comes before the pack's contents
# ---------------------------------------------------------------------------


def test_hallucination_tarot_is_made_before_the_pack_contents():
    """Hallucination's Tarot exists (and is in used_jokers) before the pack draws.

    card.lua:1721-1798 (Card:open): its event queues the contents' creation
    as a non-blocking event 1.3s later (card.lua:1725), then runs the
    jokers' open_booster context; Hallucination queues the Tarot's creation
    as a blocking event (card.lua:2339-2348) that runs on the next frame.
    So the Tarot pool ('hal') is drawn first and the pack's Tarot pool
    already excludes it.  Seed chosen so that the two orders differ.
    """
    seed = "AUDIT_G44_6"
    gs = _with_hallucination(new_run(seed=seed))
    gs["round_resets"]["blind_tags"]["Small"] = "tag_charm"  # p_arcana_mega_1

    # Reference: vanilla order on a copy, with the engine's own primitives.
    ref = copy.deepcopy(gs)
    assert ref["rng"].random("halu1") < 1 / 2  # Hallucination fires
    want_hal = create_card(
        "Tarot", ref["rng"], 1, area="", soulable=False, append="hal", game_state=ref
    ).center_key
    pack, _ = generate_pack_cards("p_arcana_mega_1", ref["rng"], 1, ref)
    want_pack = [c.center_key for c in pack]

    # Pack-first order gives something else for this seed.
    other = copy.deepcopy(gs)
    other_pack = generate_pack_cards("p_arcana_mega_1", other["rng"], 1, other)[0]
    assert want_hal in {c.center_key for c in other_pack}

    step(gs, SkipBlind())
    assert gs["consumables"][-1].center_key == want_hal
    assert [c.center_key for c in gs["pack_cards"]] == want_pack
    assert want_hal not in want_pack


# ---------------------------------------------------------------------------
# G47 — Overstock's refill is a normal create_card_for_shop
# ---------------------------------------------------------------------------


def _shop_offering_overstock(seed: str) -> dict:
    gs = new_run(seed=seed)
    play_round(gs)
    voucher = create_voucher("v_overstock_norm")
    voucher.set_cost()
    gs["shop_vouchers"] = [voucher]
    gs["dollars"] = 50
    return gs


def test_overstock_refill_rolls_illusion():
    """With Illusion, the new slot's type roll includes Illusion's pull.

    card.lua:1885-1889 (Card:apply_to_run, Overstock) queues
    change_shop_size(1) (common_events.lua:1097-1118), which fills the new
    slot with create_card_for_shop (UI_definitions.lua:742-800): the rate
    table evaluates ``pseudorandom(pseudoseed('illusion')) > 0.6`` for the
    playing-card slot on every call (:772).
    """
    gs = _shop_offering_overstock("AUDIT_G47_ILL")
    gs["used_vouchers"]["v_illusion"] = True
    gs["playing_card_rate"] = 4
    before = gs["rng"].state.get("illusion")
    step(gs, RedeemVoucher(card_index=0))
    assert len(gs["shop_cards"]) == 3
    assert gs["rng"].state.get("illusion") != before


def test_overstock_refill_honours_a_pending_store_joker_create_tag():
    """create_card_for_shop gives the slot to a pending Uncommon/Rare Tag first.

    UI_definitions.lua:753-763: a pending ``store_joker_create`` tag makes the
    slot's card (forced rarity, its own 'uta'/'rta' stream, couponed).
    """
    from jackdaw.engine.game import _award_tag

    gs = _shop_offering_overstock("AUDIT_G47_TAG")
    _award_tag(gs, "tag_uncommon", source="test")
    step(gs, RedeemVoucher(card_index=0))
    new = gs["shop_cards"][-1]
    assert new.ability["set"] == "Joker" and new.cost == 0
    assert JOKERS[new.center_key].rarity == 2
    assert "Joker2uta1" in gs["rng"].state
