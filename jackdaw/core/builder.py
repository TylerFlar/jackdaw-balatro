"""Build a run in a chosen state: :class:`GameBuilder`.

For tests, experiments and agent debugging: start a run, force a boss, select
the blind, add jokers, consumables and playing cards, set money or the ante,
take actions, then :meth:`GameBuilder.build` the :class:`~jackdaw.core.Game`.
The additions go through the same debug operations as BalatroBot's ``add`` and
``set`` (``Game.debug_add`` / ``Game.debug_set``), so a built state is one the
live game can reach with those commands too.

    game = (
        GameBuilder("b_red", stake=2, seed="DEMO0001")
        .boss("bl_hook")
        .select_blind()
        .joker("j_blueprint")
        .joker("j_joker", edition="polychrome")
        .card("H_A", enhancement="m_glass", seal="Red")
        .money(50)
        .build()
    )
"""

from __future__ import annotations

from typing import Any

from jackdaw.core.game import Game
from jackdaw.engine.actions import SelectBlind


class GameBuilder:
    """A run under construction. Every method returns the builder."""

    def __init__(
        self,
        back: str = "b_red",
        stake: int = 1,
        seed: str = "BUILDER1",
        challenge: str | None = None,
    ) -> None:
        self._game = Game.new(back, stake, seed, challenge)

    def boss(self, key: str) -> GameBuilder:
        """Make ``key`` (``"bl_hook"``...) this ante's boss and the next blind to play."""
        self._game.apply_setup({"op": "force_boss", "key": key})
        return self

    def select_blind(self) -> GameBuilder:
        """Select the blind on deck: the round starts and the hand is dealt."""
        self._game.step(SelectBlind())
        return self

    def joker(
        self,
        key: str,
        *,
        edition: str | None = None,
        eternal: bool = False,
        perishable: int | None = None,
        rental: bool = False,
    ) -> GameBuilder:
        """Add a joker (``"j_blueprint"``); ``edition`` ``"foil"``, ``"holo"``,
        ``"polychrome"`` or ``"negative"``, ``perishable`` in rounds."""
        self._game.debug_add(
            key, edition=edition, eternal=eternal, perishable=perishable, rental=rental
        )
        return self

    def consumable(self, key: str, *, edition: str | None = None) -> GameBuilder:
        """Add a Tarot, Planet or Spectral (``"c_fool"``)."""
        self._game.debug_add(key, edition=edition)
        return self

    def card(
        self,
        key: str,
        *,
        enhancement: str | None = None,
        seal: str | None = None,
        edition: str | None = None,
    ) -> GameBuilder:
        """Add a playing card (``"H_A"``: suit letter, rank letter): to the hand during a
        round, else to the deck. ``enhancement`` ``"m_glass"``..., ``seal`` ``"Red"``..."""
        self._game.debug_add(key, enhancement=enhancement, seal=seal, edition=edition)
        return self

    def shop_item(self, key: str) -> GameBuilder:
        """Put a voucher (``"v_overstock_norm"``) or booster (``"p_arcana_normal_1"``) in
        the shop (SHOP only)."""
        self._game.debug_add(key)
        return self

    def money(self, dollars: float) -> GameBuilder:
        self._game.debug_set(money=dollars)
        return self

    def chips(self, chips: float) -> GameBuilder:
        """The round's score so far."""
        self._game.debug_set(chips=chips)
        return self

    def ante(self, ante: int) -> GameBuilder:
        self._game.debug_set(ante=ante)
        return self

    def hands(self, hands: int) -> GameBuilder:
        """The current round's hands left."""
        self._game.debug_set(hands=hands)
        return self

    def discards(self, discards: int) -> GameBuilder:
        """The current round's discards left."""
        self._game.debug_set(discards=discards)
        return self

    def step(self, *actions: Any) -> GameBuilder:
        """Take actions (dataclasses from :mod:`jackdaw.engine.actions`, or tape forms)."""
        for action in actions:
            self._game.step(action)
        return self

    def build(self) -> Game:
        """The run as built. The builder keeps working on its own copy."""
        return self._game.clone()
