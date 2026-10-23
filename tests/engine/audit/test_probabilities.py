"""Oops! All 6s reaches every G.GAME.probabilities.normal reader (G23)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jackdaw.engine.actions import UseConsumable
from jackdaw.engine.game import step

from ._helpers import (
    give_consumable,
    give_joker,
    new_run,
    rng_twin,
    select_blind,
    win_round,
)


def _find_seed(prefix: str, ready: Callable[[str], dict[str, Any]], key: str, lo: float, hi: float):
    """First seed whose next ``pseudorandom(key)`` lands in [lo, hi)."""
    for i in range(200):
        seed = f"{prefix}{i}"
        gs = ready(seed)
        if lo <= rng_twin(gs).random(key) < hi:
            return gs
    raise AssertionError(f"no seed found for {key} in [{lo}, {hi})")


def test_wheel_of_fortune_reads_oops():
    """The Wheel rolls pseudorandom('wheel_of_fortune') <
    G.GAME.probabilities.normal/extra (card.lua:1470); Oops! All 6s doubles
    probabilities.normal (card.lua:608-611).  A roll in [1/4, 2/4) hits only
    with Oops."""

    def ready(seed: str) -> dict[str, Any]:
        gs = select_blind(new_run(seed))
        give_joker(gs, "j_oops")
        give_joker(gs, "j_joker")
        give_consumable(gs, "c_wheel_of_fortune")
        return gs

    gs = _find_seed("AUDIT_WHEEL_OOPS", ready, "wheel_of_fortune", 0.25, 0.5)
    assert gs["probabilities"]["normal"] == 2
    step(gs, UseConsumable(card_index=0))
    assert any(j.edition for j in gs["jokers"])


def test_gros_michel_end_of_round_roll_reads_oops():
    """Gros Michel's end-of-round roll is pseudorandom('gros_michel') <
    G.GAME.probabilities.normal/odds (card.lua:3019-3020).  A roll in
    [1/6, 2/6) makes it extinct only with Oops."""

    def ready(seed: str) -> dict[str, Any]:
        gs = select_blind(new_run(seed))
        give_joker(gs, "j_oops")
        give_joker(gs, "j_gros_michel")
        return gs

    gs = _find_seed("AUDIT_GROS_OOPS", ready, "gros_michel", 1 / 6, 2 / 6)
    win_round(gs)
    assert [j.center_key for j in gs["jokers"]] == ["j_oops"]
    assert gs["pool_flags"].get("gros_michel_extinct")
