"""Audit regressions: challenge custom rules (G04) and challenge bans (G05)."""

from __future__ import annotations

import math

import pytest

from jackdaw.engine import (
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PlayHand,
    SelectBlind,
    SkipPack,
    step,
)
from jackdaw.engine.card_factory import create_card, create_joker
from jackdaw.engine.challenges import CHALLENGES
from jackdaw.engine.packs import generate_pack_cards
from jackdaw.engine.rng import PseudoRandom
from tests.engine.audit._helpers import new_run, play_round, win_blind

# ---------------------------------------------------------------------------
# G04 — every custom rule a 1.0.1o challenge uses is implemented
# ---------------------------------------------------------------------------

# Custom rule ids used by challenges.lua (1.0.1o) and where vanilla reads them.
_RULE_SITES = {
    "no_reward": "game.lua:2109",
    "no_reward_specific": "game.lua:2114",
    "no_extra_hand_money": "state_events.lua:1165",
    "no_interest": "state_events.lua:1191",
    "no_shop_jokers": "game.lua:2119",
    "discard_cost": "state_events.lua:433",
    "chips_dollar_cap": "misc_functions.lua:684",
    "flipped_cards": "cardarea.lua:601; common_events.lua:403",
    "minus_hand_size_per_X_dollar": "cardarea.lua:245",
    "all_eternal": "common_events.lua:2134",
    "debuff_played_cards": "state_events.lua:1077",
    "set_eternal_ante": "state_events.lua:240",
    "set_joker_slots_ante": "state_events.lua:245",
    "inflation": "button_callbacks.lua:2455; card.lua:1800/1858",
}


def test_every_challenge_rule_is_known():
    """challenges.lua: the custom rule ids of all 20 challenges."""
    used = {r["id"] for ch in CHALLENGES.values() for r in ch["rules"]["custom"]}
    assert used == set(_RULE_SITES)


def test_chips_dollar_cap():
    """Rich get Richer: chips are capped at the dollars (never below 0).

    misc_functions.lua:684-689: ``mod_chips`` returns
    ``min(chips, max(G.GAME.dollars, 0))`` under ``chips_dollar_cap``, and
    evaluate_play passes every chip value through it (state_events.lua:616,
    641, 646, 706, 760, 884, 911, 924, 948).
    """
    gs = new_run(challenge="c_rich_1", seed="AUDIT_G04_RICH")
    step(gs, SelectBlind())
    gs["dollars"] = 7
    step(gs, PlayHand(card_indices=(0,)))  # High Card: 5 + rank chips >= 7
    res = gs["last_score_result"]
    assert res.chips == 7 and res.total == 7

    gs["dollars"] = -3
    step(gs, PlayHand(card_indices=(0,)))
    res = gs["last_score_result"]
    assert res.chips == 0 and res.total == 0


def test_flipped_cards_rolls_for_every_card_drawn_to_hand():
    """X-ray Vision: each card drawn to the hand stays face down 1 time in 4.

    cardarea.lua:601-605 (CardArea:draw_card_from): after the boss check,
    every card drawn into G.hand rolls ``pseudorandom(pseudoseed(
    'flipped_card')) < 1/flipped_cards`` and stays flipped below it.
    """
    seed = "AUDIT_G04_XRAY"
    gs = new_run(challenge="c_xray_1", seed=seed)
    step(gs, SelectBlind())
    ref = PseudoRandom(seed)
    expected = sum(ref.random("flipped_card") < 1 / 4 for _ in range(len(gs["hand"])))
    assert expected > 0
    assert sum(c.facing == "back" for c in gs["hand"]) == expected

    # The redraw after a discard rolls for the replacements too.
    flipped_before = sum(c.facing == "back" for c in gs["hand"])
    face_up = [i for i, c in enumerate(gs["hand"]) if c.facing == "front"][:3]
    step(gs, Discard(card_indices=tuple(face_up)))
    expected_new = sum(ref.random("flipped_card") < 1 / 4 for _ in range(3))
    assert sum(c.facing == "back" for c in gs["hand"]) == flipped_before + expected_new


def test_minus_hand_size_per_x_dollar():
    """Luxury Tax: -1 hand size per $5.

    cardarea.lua:245-251 (CardArea:update for G.hand): whenever
    ``floor(dollars/5)`` differs from ``last_poll_size`` the hand size
    changes by the difference and the poll is stored.
    """
    gs = new_run(challenge="c_luxury_1", seed="AUDIT_G04_LUX")
    assert gs["hand_size"] == 10  # $4 -> floor(4/5) = 0
    gs["dollars"] = 12
    step(gs, SelectBlind())
    assert gs["hand_size"] == 8
    assert len(gs["hand"]) == 8

    gs["dollars"] = 3  # back under $5: the two slots return
    step(gs, Discard(card_indices=(0,)))
    assert gs["hand_size"] == 10
    assert len(gs["hand"]) == 10


def test_all_eternal_makes_created_jokers_eternal():
    """Non-Perishable: every Joker create_card makes is Eternal.

    common_events.lua:2134-2136 (create_card): ``if _type == 'Joker' and
    G.GAME.modifiers.all_eternal then card:set_eternal(true)``.
    """
    for i in range(6):
        gs = new_run(challenge="c_non_perishable_1", seed=f"AUDIT_G04_NP{i}")
        play_round(gs)
        shop_jokers = [c for c in gs["shop_cards"] if c.ability.get("set") == "Joker"]
        if shop_jokers:
            assert all(c.eternal for c in shop_jokers)
            return
    pytest.fail("no seed rolled a shop joker")


def test_debuff_played_cards_is_permanent():
    """Double or Nothing: every scoring card is debuffed for the rest of the run.

    state_events.lua:1077-1084: after scoring, ``ability.perma_debuff = true``
    on each card of the scoring hand; Card:update (card.lua:4158) then keeps
    ``debuff = true`` every frame, through round ends and later blinds.
    """
    gs = new_run(challenge="c_double_nothing_1", seed="AUDIT_G04_DON")
    step(gs, SelectBlind())
    played = gs["hand"][0]
    step(gs, PlayHand(card_indices=(0,)))
    assert played.ability.get("perma_debuff") is True
    assert played.debuff

    win_blind(gs)
    step(gs, CashOut())
    step(gs, NextRound())
    step(gs, SelectBlind())
    assert played.debuff  # the blind's debuff pass does not clear it


def test_typecast_ante_rules():
    """Typecast: at the ante-4 boss, jokers become Eternal and slots drop to 0.

    state_events.lua:240-247 (end_round, boss defeated, before ease_ante):
    ``set_eternal_ante`` runs set_eternal(true) on every joker (only
    eternal_compat ones take it, card.lua:506-511) and
    ``set_joker_slots_ante`` sets the joker card_limit to 0.
    """
    gs = new_run(challenge="c_typecast_1", seed="AUDIT_G04_TYPE")
    gs["round_resets"]["ante"] = 4
    gs["blind_on_deck"] = "Boss"
    step(gs, SelectBlind())
    joker, bean = create_joker("j_joker"), create_joker("j_ice_cream")  # Ice Cream: not compat
    for card in (joker, bean):
        gs["jokers"].append(card)
        card.add_to_deck(gs)
    win_blind(gs)
    assert joker.eternal and not bean.eternal
    assert gs["joker_slots"] == 0
    assert gs["round_resets"]["ante"] == 5


def test_typecast_rules_wait_for_their_ante():
    """state_events.lua:240/245 compare against the current ante exactly."""
    gs = new_run(challenge="c_typecast_1", seed="AUDIT_G04_TYPE3")
    gs["round_resets"]["ante"] = 3
    gs["blind_on_deck"] = "Boss"
    step(gs, SelectBlind())
    joker = create_joker("j_joker")
    gs["jokers"].append(joker)
    joker.add_to_deck(gs)
    win_blind(gs)
    assert not joker.eternal and gs["joker_slots"] == 5


def test_inflation_raises_prices_on_every_purchase():
    """Inflation: each purchase adds $1 to every price.

    button_callbacks.lua:2455-2462 (buy_from_shop), card.lua:1800 (Card:open)
    and card.lua:1858 (Card:redeem): ``G.GAME.inflation += 1`` and a
    set_cost pass over every card (card.lua:370 adds the inflation to
    extra_cost).
    """
    gs = new_run(challenge="c_inflation_1", seed="AUDIT_G04_INFL")
    play_round(gs)
    assert gs["phase"] == GamePhase.SHOP
    other = gs["shop_boosters"][1]
    cost_before = other.cost
    step(gs, OpenBooster(card_index=0))
    assert gs["inflation"] == 1
    assert other.cost == cost_before + 1
    step(gs, SkipPack())
    assert gs["shop_boosters"][0].cost == cost_before + 1
    assert math.isfinite(gs["dollars"])


# ---------------------------------------------------------------------------
# G05 — challenge bans reach the boss, tag and voucher pools and the Soul roll
# ---------------------------------------------------------------------------


def _banned(cid: str) -> set[str]:
    out: set[str] = set()
    for group in CHALLENGES[cid]["restrictions"].values():
        for entry in group:
            out.add(entry["id"])
            out.update(entry.get("ids", ()))
    return out


def test_banned_tags_never_offered():
    """Jokerless bans 8 tags; get_next_tag_key never offers them.

    common_events.lua:2030 (get_current_pool): ``if add and not
    G.GAME.banned_keys[v.key]`` applies to the Tag pool too, and
    get_next_tag_key (common_events.lua:1914) resamples UNAVAILABLE slots.
    """
    banned = _banned("c_jokerless_1")
    for i in range(30):
        gs = new_run(challenge="c_jokerless_1", seed=f"AUDIT_G05_TAG{i}")
        tags = gs["round_resets"]["blind_tags"]
        assert not ({tags["Small"], tags["Big"]} & banned), (i, tags)


def test_banned_vouchers_never_offered():
    """get_next_voucher_key (common_events.lua:1901) draws from the same
    banned-key-filtered pool (common_events.lua:2030)."""
    for cid in ("c_golden_needle_1", "c_inflation_1", "c_omelette_1"):
        banned = _banned(cid)
        for i in range(25):
            gs = new_run(challenge=cid, seed=f"AUDIT_G05_V{cid}{i}")
            assert gs["current_round"]["voucher"] not in banned, (cid, i)


def test_banned_bosses_never_drawn():
    """get_new_boss drops banned blinds before picking (common_events.lua:2359-2361).

    Jokerless bans three of the five showdown bosses, so the ante-8 boss is
    The Violet Vessel or The Cerulean Bell.
    """
    for i in range(8):
        gs = new_run(challenge="c_jokerless_1", seed=f"AUDIT_G05_BOSS{i}")
        gs["round_resets"]["ante"] = 7
        gs["blind_on_deck"] = "Boss"
        step(gs, SelectBlind())
        win_blind(gs)
        assert gs["round_resets"]["ante"] == 8
        step(gs, CashOut())  # cash_out's reset_blinds draws the next boss
        assert gs["round_resets"]["blind_choices"]["Boss"] in {"bl_final_vessel", "bl_final_bell"}


def test_banned_soul_skips_the_soul_rolls():
    """With c_soul banned there is no Soul / Black Hole roll at all.

    common_events.lua:2088: the forced-key block of create_card runs only
    ``if not forced_key and soulable and (not G.GAME.banned_keys['c_soul'])``
    — both the Soul and the Black Hole roll sit inside it.
    """
    gs = new_run(challenge="c_jokerless_1", seed="AUDIT_G05_SOUL")
    generate_pack_cards("p_arcana_normal_1", gs["rng"], 1, gs)
    generate_pack_cards("p_spectral_normal_1", gs["rng"], 1, gs)
    assert not any(k.startswith("soul_") for k in gs["rng"].state)

    control = new_run(seed="AUDIT_G05_SOUL")
    generate_pack_cards("p_arcana_normal_1", control["rng"], 1, control)
    assert "soul_Tarot1" in control["rng"].state


def test_banned_forced_key_falls_back_to_the_pool():
    """common_events.lua:2109: ``if forced_key and not G.GAME.banned_keys[forced_key]``
    — a banned forced key is ignored and the pool is drawn instead."""
    gs = new_run(challenge="c_jokerless_1", seed="AUDIT_G05_FORCED")
    card = create_card(
        "Tarot", gs["rng"], 1, area="", soulable=False, forced_key="c_judgement", game_state=gs
    )
    assert card.center_key != "c_judgement"
    assert card.ability["set"] == "Tarot"
