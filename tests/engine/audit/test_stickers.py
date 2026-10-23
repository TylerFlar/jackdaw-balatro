"""Sticker eligibility (G33).

Card:set_eternal / Card:set_perishable (card.lua:506-519) only apply a sticker
the center is compatible with (``eternal_compat`` / ``perishable_compat``)
and never put Eternal and Perishable on the same card.  create_card's shop /
pack roll (common_events.lua:2137-2146) gives Eternal above 0.7 and
Perishable in (0.4, 0.7], and only rolls for Rental when rentals are on.
"""

from __future__ import annotations

from jackdaw.engine.card_factory import create_card, create_joker
from jackdaw.engine.rng import PseudoRandom


def _seed_with_roll(key: str, lo: float, hi: float) -> str:
    for i in range(500):
        seed = f"AUDIT_STICKER{i}"
        if lo < PseudoRandom(seed).random(key) <= hi:
            return seed
    raise AssertionError(f"no seed rolls {key} in ({lo}, {hi}]")


def _shop_joker(seed: str, key: str, **modifiers: bool):
    return create_card(
        "Joker",
        PseudoRandom(seed),
        1,
        area="shop",
        forced_key=key,
        game_state={"modifiers": dict(modifiers)},
    )


class TestSetters:
    def test_set_eternal_needs_eternal_compat(self):
        banana = create_joker("j_gros_michel")
        banana.set_eternal(True)
        assert not banana.eternal
        joker = create_joker("j_joker")
        joker.set_eternal(True)
        assert joker.eternal

    def test_set_perishable_needs_perishable_compat(self):
        bus = create_joker("j_ride_the_bus")
        bus.set_perishable(True)
        assert not bus.perishable
        joker = create_joker("j_joker")
        joker.set_perishable(True)
        assert joker.perishable and joker.perish_tally == 5

    def test_eternal_and_perishable_exclude_each_other(self):
        eternal = create_joker("j_joker")
        eternal.set_eternal(True)
        eternal.set_perishable(True)
        assert eternal.eternal and not eternal.perishable
        perishable = create_joker("j_joker")
        perishable.set_perishable(True)
        perishable.set_eternal(True)
        assert perishable.perishable and not perishable.eternal


class TestShopRoll:
    def test_eternal_roll_skips_incompatible_jokers(self):
        seed = _seed_with_roll("etperpoll1", 0.7, 1.0)
        assert _shop_joker(seed, "j_joker", enable_eternals_in_shop=True).eternal
        assert not _shop_joker(seed, "j_gros_michel", enable_eternals_in_shop=True).eternal

    def test_perishable_roll_skips_incompatible_jokers(self):
        seed = _seed_with_roll("etperpoll1", 0.4, 0.7)
        assert _shop_joker(seed, "j_joker", enable_perishables_in_shop=True).perishable
        assert not _shop_joker(seed, "j_hologram", enable_perishables_in_shop=True).perishable

    def test_perishable_roll_is_capped_at_0_7(self):
        """With eternals off, a roll above 0.7 gives no sticker at all."""
        seed = _seed_with_roll("etperpoll1", 0.7, 1.0)
        card = _shop_joker(seed, "j_joker", enable_perishables_in_shop=True)
        assert not card.perishable and not card.eternal

    def test_rental_is_only_rolled_when_rentals_are_on(self):
        """``enable_rentals_in_shop and pseudorandom('ssjr'..ante) > 0.7``
        short-circuits: without rentals the stream is never drawn."""
        rng = PseudoRandom("AUDIT_RENTAL")
        create_card("Joker", rng, 1, area="shop", forced_key="j_joker", game_state={})
        assert "ssjr1" not in rng.state
        assert "etperpoll1" in rng.state
        rng = PseudoRandom("AUDIT_RENTAL")
        create_card(
            "Joker",
            rng,
            1,
            area="shop",
            forced_key="j_joker",
            game_state={"modifiers": {"enable_rentals_in_shop": True}},
        )
        assert "ssjr1" in rng.state
