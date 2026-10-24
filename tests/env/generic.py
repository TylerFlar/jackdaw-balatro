"""An adapter that sends the environments down their generic path.

``BalatroEnvironment`` steps a ``DirectAdapter`` run in one call into ``jackdaw._core``, and
``BalatroGymnasiumEnv`` runs it whole in Rust (``_core.GymEnv``). Any other adapter with a
``game`` (``LiveAdapter``) takes the generic path: the adapter's own ``step``, the env view read
afterwards, and the Gym wrapper's Python action table and rewards. :class:`GenericAdapter` runs
the same Rust run as ``DirectAdapter`` down that path, and the test can edit its run.
"""

from __future__ import annotations

from typing import Any

from jackdaw.core import Game
from jackdaw.env.game_interface import DirectAdapter, GameState


class GenericAdapter:
    """``DirectAdapter``'s run behind a plain adapter."""

    def __init__(self) -> None:
        self._direct = DirectAdapter()

    def reset(self, back_key: str, stake: int, seed: str, *, challenge: Any = None) -> GameState:
        return self._direct.reset(back_key, stake, seed, challenge=challenge)

    def step(self, action: Any) -> GameState:
        return self._direct.step(action)

    def get_legal_actions(self) -> list[Any]:
        return self._direct.get_legal_actions()

    @property
    def game(self) -> Game:
        return self._direct.game

    @property
    def done(self) -> bool:
        return self._direct.done

    @property
    def won(self) -> bool:
        return self._direct.won
