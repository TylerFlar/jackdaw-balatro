"""Stake initialization must enable shop/pack stickers through the normal state."""

from __future__ import annotations

import pytest

from jackdaw.engine.card_factory import create_card
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run


@pytest.mark.parametrize("stake", range(1, 9))
@pytest.mark.parametrize(
    "area,ep_key,rental_key",
    [
        ("shop", "etperpoll1", "ssjr1"),
        ("pack", "packetper1", "packssjr1"),
    ],
)
@pytest.mark.parametrize(
    "ep_roll,rental_roll",
    [
        (0.8, 0.8),
        (0.5, 0.8),
        (0.4, 0.7),
        (0.7, 0.7),
    ],
)
def test_initialized_stakes_enable_stickers(
    monkeypatch,
    stake,
    area,
    ep_key,
    rental_key,
    ep_roll,
    rental_roll,
):
    gs = initialize_run("b_red", stake, "STAKEWIRING")
    rng = gs["rng"]
    original_random = rng.random
    calls = []

    def controlled_random(key, *args, **kwargs):
        calls.append(key)
        # Advance the actual stream even when controlling its returned roll.
        value = original_random(key, *args, **kwargs)
        return {ep_key: ep_roll, rental_key: rental_roll}.get(key, value)

    monkeypatch.setattr(rng, "random", controlled_random)
    card = create_card("Joker", rng, 1, area=area, forced_key="j_joker", game_state=gs)

    assert card.eternal is (stake >= 4 and ep_roll > 0.7)
    assert card.perishable is (stake >= 7 and 0.4 < ep_roll <= 0.7)
    assert card.rental is (stake >= 8 and rental_roll > 0.7)
    # `enable_rentals_in_shop and pseudorandom(...)` short-circuits: the rental
    # stream is only drawn once rentals are on (common_events.lua:2144).
    assert calls == [ep_key, *([rental_key] if stake >= 8 else []), "edi1"]

    # Compare the next draw on every touched stream to an unmodified reference.
    reference = initialize_run("b_red", stake, "STAKEWIRING")["rng"]
    for key in calls:
        reference.random(key)
    for key in tuple(calls):
        assert original_random(key) == reference.random(key)


@pytest.mark.parametrize(
    "flag,attribute,ep_roll",
    [
        ("enable_eternals_in_shop", "eternal", 0.8),
        ("enable_perishables_in_shop", "perishable", 0.5),
        ("enable_rentals_in_shop", "rental", 0.8),
    ],
)
@pytest.mark.parametrize(
    "nested,flat,expected",
    [
        (None, True, True),
        ({}, True, True),
        (True, False, True),
        (False, True, False),
    ],
)
def test_sticker_flag_compatibility(monkeypatch, flag, attribute, ep_roll, nested, flat, expected):
    gs = {flag: flat}
    if nested is not None:
        gs["modifiers"] = {} if nested == {} else {flag: nested}
    rng = PseudoRandom("FLAGINPUT")
    monkeypatch.setattr(rng, "random", lambda key: ep_roll if key == "etperpoll1" else 0.8)

    card = create_card("Joker", rng, 1, forced_key="j_joker", game_state=gs)
    assert getattr(card, attribute) is expected


def test_gold_flags_do_not_add_stickers_outside_shop_or_pack(monkeypatch):
    gs = initialize_run("b_red", 8, "NOSTICKERS")
    calls = []

    def roll(key):
        calls.append(key)
        return 0.8

    monkeypatch.setattr(gs["rng"], "random", roll)
    card = create_card("Joker", gs["rng"], 1, area="", forced_key="j_joker", game_state=gs)
    assert not card.eternal
    assert not card.perishable
    assert not card.rental
    assert calls == ["edi1"]
