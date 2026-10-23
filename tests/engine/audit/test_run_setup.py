"""Audit regressions: run setup, decks and challenge starting state (G01-G03, G06)."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import SimBackend
from jackdaw.engine import (
    CashOut,
    GamePhase,
    IllegalActionError,
    SelectBlind,
    SwapJokersLeft,
    SwapJokersRight,
    get_legal_actions,
    step,
)
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.challenges import CHALLENGES
from jackdaw.env.action_space import ActionType, get_action_mask
from tests.engine.audit._helpers import force_boss, new_run, play_round, win_blind

# ---------------------------------------------------------------------------
# G01 — Green Deck economy
# ---------------------------------------------------------------------------


def test_green_deck_pays_per_hand_and_discard_and_no_interest():
    """Green Deck: $2 per unused hand, $1 per unused discard, no interest.

    back.lua:269-277 (Back:apply_to_run) stores the three values in
    ``G.GAME.modifiers``; state_events.lua:1165-1191 (evaluate_round) reads
    them from there: ``hands_left*(modifiers.money_per_hand or 1)``,
    ``discards_left*modifiers.money_per_discard`` and no interest row when
    ``modifiers.no_interest`` is set.
    """
    gs = new_run("b_green", "AUDIT_G01")
    assert gs["modifiers"]["money_per_hand"] == 2
    assert gs["modifiers"]["money_per_discard"] == 1
    assert gs["modifiers"]["no_interest"] is True

    step(gs, SelectBlind())
    gs["dollars"] = 50  # interest would be $5 on a normal deck
    win_blind(gs)

    cr = gs["current_round"]
    earnings = gs["round_earnings"]
    assert cr["hands_left"] == 3 and cr["discards_left"] == 3
    assert earnings.unused_hands_bonus == 3 * 2
    assert earnings.unused_discards_bonus == 3 * 1
    assert earnings.interest == 0

    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    # blind reward $3 + $6 hands + $3 discards, nothing else
    assert gs["dollars"] == 50 + 3 + 6 + 3


# ---------------------------------------------------------------------------
# G02 — Magic Deck / Ghost Deck starting consumables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("back", "keys"), [("b_magic", ["c_fool", "c_fool"]), ("b_ghost", ["c_hex"])]
)
def test_back_starting_consumables_are_created(back, keys):
    """Magic Deck starts with 2x The Fool, Ghost Deck with Hex.

    back.lua:184-196: Back:apply_to_run queues an event that, for each
    ``config.consumables`` key, calls ``create_card('Tarot', G.consumeables,
    nil, nil, nil, nil, key, 'deck')`` (a forced key: no soul roll, no pool
    pick, no edition poll), ``add_to_deck`` and ``emplace``, with no slot
    check.  The event runs after start_run's synchronous setup, so the cards'
    sort_ids follow the deck's; set_ability marks them in used_jokers
    (card.lua:349-354).
    """
    gs = new_run(back, "AUDIT_G02")
    assert [c.center_key for c in gs["consumables"]] == keys
    assert all(gs["used_jokers"].get(k) for k in keys)
    last_deck_id = max(c.sort_id for c in gs["deck"])
    assert all(c.sort_id > last_deck_id for c in gs["consumables"])
    # Forced keys draw nothing: no soul roll, pool pick or edition poll.
    assert not any(k.startswith(("soul_", "Tarot", "Spectral", "edi")) for k in gs["rng"].state)
    if back == "b_magic":
        assert gs["consumable_slots"] == 3  # Crystal Ball; both Fools fit


# ---------------------------------------------------------------------------
# G03 — Challenge starting jokers / consumables, pinned
# ---------------------------------------------------------------------------

_WITH_STARTING_CARDS = sorted(
    cid for cid, ch in CHALLENGES.items() if ch["jokers"] or ch["consumeables"]
)


def test_twelve_challenges_start_with_jokers():
    """challenges.lua (1.0.1o): 12 of the 20 challenges list starting jokers."""
    assert len([c for c in CHALLENGES.values() if c["jokers"]]) == 12
    assert len(_WITH_STARTING_CARDS) == 12


@pytest.mark.parametrize("cid", _WITH_STARTING_CARDS)
def test_challenge_starting_cards_are_created(cid):
    """Challenge jokers and consumeables exist at run start, in definition order.

    game.lua:2067-2088 (Game:start_run) queues one event per entry calling
    ``add_joker(v.id, v.edition, ...)`` (common_events.lua:372-384: a bare
    Card for the center, ``add_to_deck``, then ``set_edition``) and then
    ``set_eternal(true)`` / ``pinned = true``; consumeables go through
    ``add_joker(v.id)`` into G.consumeables.  No RNG is drawn.
    """
    ch = CHALLENGES[cid]
    gs = new_run(challenge=cid, seed="AUDIT_G03")
    jokers = gs["jokers"]
    assert [j.center_key for j in jokers] == [e["id"] for e in ch["jokers"]]
    for card, entry in zip(jokers, ch["jokers"], strict=True):
        edition = entry.get("edition")
        assert bool(card.edition and card.edition.get(edition)) == bool(edition)
        assert card.eternal == bool(entry.get("eternal"))
        assert card.pinned == bool(entry.get("pinned"))
        assert gs["used_jokers"].get(card.center_key)
        assert card.sell_cost >= 1  # Card:init runs set_cost
    assert [c.center_key for c in gs["consumables"]] == [e["id"] for e in ch["consumeables"]]
    assert not any(k.startswith("edi") for k in gs["rng"].state)


def test_challenge_starting_jokers_apply_their_passives():
    """add_joker runs add_to_deck, then set_edition on the added card.

    card.lua:405-415: a Negative edition set on a card that is already
    added_to_deck raises the joker slot count; card.lua:593 / 608 apply
    Credit Card (bankrupt_at -20) and Oops! All 6s (probabilities x2).
    """
    assert new_run(challenge="c_mad_world_1")["joker_slots"] == 6  # Negative Pareidolia
    fragile = new_run(challenge="c_fragile_1")
    assert fragile["joker_slots"] == 7  # two Negative Oops! All 6s
    assert fragile["probabilities"]["normal"] == 4
    assert new_run(challenge="c_inflation_1")["bankrupt_at"] == -20


def _knife_run_in_shop(seed: str = "AUDIT_G03_PIN") -> dict:
    """On a Knife's Edge (pinned Ceremonial Dagger) in the shop with 3 jokers."""
    gs = new_run(challenge="c_knife_1", seed=seed)
    play_round(gs)
    backend = SimBackend()
    backend._gs = gs
    backend.handle("add", {"key": "j_joker"})
    backend.handle("add", {"key": "j_egg"})
    assert [j.center_key for j in gs["jokers"]] == ["j_ceremonial", "j_joker", "j_egg"]
    return gs


def test_pinned_joker_cannot_be_swapped():
    """A pinned joker cannot be moved.

    cardarea.lua:528: align_cards sorts the joker area by
    ``T.x - 100*(pinned and sort_id or 0)``, so a pinned card always snaps
    back to the left whatever is dragged past it.  Swaps that would move it
    are refused, and neither the legal actions nor the env mask offer them.
    """
    gs = _knife_run_in_shop()
    legal = get_legal_actions(gs)
    assert SwapJokersRight(idx=0) not in legal
    assert SwapJokersLeft(idx=1) not in legal
    assert SwapJokersRight(idx=1) in legal and SwapJokersLeft(idx=2) in legal

    mask = get_action_mask(gs)
    assert not mask.entity_masks[ActionType.SwapJokersRight][0]
    assert not mask.entity_masks[ActionType.SwapJokersLeft][1]
    assert mask.entity_masks[ActionType.SwapJokersRight][1]

    for action in (SwapJokersRight(idx=0), SwapJokersLeft(idx=1)):
        with pytest.raises(IllegalActionError):
            step(gs, action)
    step(gs, SwapJokersRight(idx=1))
    assert [j.center_key for j in gs["jokers"]] == ["j_ceremonial", "j_egg", "j_joker"]


def test_pinned_joker_stays_leftmost_after_amber_acorn():
    """Amber Acorn shuffles the jokers; a pinned joker snaps back to the left.

    blind.lua:196 shuffles G.jokers ('aajk'); the next align_cards
    (cardarea.lua:528) sorts the pinned card first, the others keep the
    shuffled order.
    """
    gs = new_run(challenge="c_knife_1", seed="AUDIT_G03_ACORN1")
    added = ["j_joker", "j_egg", "j_banner", "j_mystic_summit", "j_jolly"]
    for key in added:
        card = create_joker(key)
        gs["jokers"].append(card)
        card.add_to_deck(gs)
    force_boss(gs, "bl_final_acorn")
    step(gs, SelectBlind())
    jokers = gs["jokers"]
    assert jokers[0].center_key == "j_ceremonial" and jokers[0].pinned
    # The Dagger sliced its right neighbour at setting_blind (card.lua:2552).
    assert len(jokers) == len(added) and {j.center_key for j in jokers[1:]} < set(added)
