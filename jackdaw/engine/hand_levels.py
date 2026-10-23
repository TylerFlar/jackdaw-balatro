"""Per-run poker hand level tracking.

Tracks the current level, chips, mult, and play counts for each of the
12 poker hand types.  Levels start at 1 and increase from Planet cards
(or Space Joker, Orbital Tag, etc.).

Source: ``G.GAME.hands`` in game.lua ``init_game_object`` (lines 2001-2014),
``level_up_hand`` in common_events.lua:464.
"""

from __future__ import annotations

from dataclasses import dataclass

from jackdaw.engine.data.hands import HAND_BASE, HANDS_PAIRS_ORDER, HandType


def most_played_poker_hand(levels: HandLevels) -> str:
    """end_round's tally of ``current_round.most_played_poker_hand``
    (state_events.lua:129-138), run after a boss is beaten; The Ox reads it.

    The loop starts from 'High Card' with ``_played = -1, _order = 100`` and
    takes a hand when ``v.played > _played or (v.played == _played and
    _order > v.order)`` — but never updates ``_order``, so the second test
    is always true: of the hands tied on the most plays, the one pairs()
    visits LAST wins.  pairs() order is :data:`HANDS_PAIRS_ORDER`.
    """
    name, played, order = "High Card", -1, 100
    for ht in HANDS_PAIRS_ORDER:
        h = levels.get_state(ht)
        if h.played > played or (h.played == played and order > HAND_BASE[ht].order):
            played = h.played
            name = ht.value
    return name


@dataclass
class HandState:
    """Mutable state for a single poker hand type during a run."""

    level: int
    chips: int
    mult: int
    played: int
    played_this_round: int
    visible: bool


class HandLevels:
    """Tracks hand levels and play counts for a run.

    Mirrors ``G.GAME.hands`` from ``init_game_object``.
    """

    __slots__ = ("_hands",)

    def __init__(self) -> None:
        self._hands: dict[HandType, HandState] = {}
        for ht, base in HAND_BASE.items():
            self._hands[ht] = HandState(
                level=1,
                chips=base.s_chips,
                mult=base.s_mult,
                played=0,
                played_this_round=0,
                visible=base.visible,
            )

    def get(self, hand_type: HandType | str) -> tuple[int, int]:
        """Return ``(chips, mult)`` for *hand_type* at its current level."""
        ht = HandType(hand_type) if isinstance(hand_type, str) else hand_type
        h = self._hands[ht]
        return h.chips, h.mult

    def get_state(self, hand_type: HandType | str) -> HandState:
        """Return the full mutable ``HandState`` for *hand_type*."""
        # HandType is a StrEnum, so members also pass isinstance(str); skip
        # the (identity) HandType() round-trip for them.
        if type(hand_type) is not HandType and isinstance(hand_type, str):
            hand_type = HandType(hand_type)
        return self._hands[hand_type]

    def level_up(self, hand_type: HandType | str, amount: int = 1) -> None:
        """Level up *hand_type* by *amount* (from Planet card, Space Joker, etc.).

        Recomputes chips and mult from the base formula::

            chips = s_chips + l_chips * (level - 1)
            mult  = s_mult  + l_mult  * (level - 1)

        Matches ``level_up_hand`` in common_events.lua:464.
        """
        ht = HandType(hand_type) if isinstance(hand_type, str) else hand_type
        h = self._hands[ht]
        base = HAND_BASE[ht]

        h.level = max(0, h.level + amount)
        h.chips = max(0, base.s_chips + base.l_chips * (h.level - 1))
        h.mult = max(1, base.s_mult + base.l_mult * (h.level - 1))
        # Leveling never reveals a hand: level_up_hand only sets level, mult
        # and chips (common_events.lua:464-468).  Playing it does, below.

    def record_play(self, hand_type: HandType | str) -> None:
        """Record that *hand_type* was played once.

        evaluate_play bumps both play counters and reveals the hand
        (``G.GAME.hands[text].visible = true``, state_events.lua:574-578) —
        a secret hand becomes visible only by being played.
        """
        ht = HandType(hand_type) if isinstance(hand_type, str) else hand_type
        h = self._hands[ht]
        h.played += 1
        h.played_this_round += 1
        h.visible = True

    def reset_round_counts(self) -> None:
        """Reset ``played_this_round`` for all types (called at round start)."""
        for h in self._hands.values():
            h.played_this_round = 0

    def most_played(self) -> HandType:
        """Return the most-played hand type by the live counts.

        Ties broken by hand priority order (highest hand wins).
        If nothing has been played, returns High Card.  The Ox does not use
        this: it reads the tally frozen at the last boss defeat (see
        :func:`most_played_poker_hand`).
        """
        best: HandType = HandType.HIGH_CARD
        best_count = 0
        for ht, h in self._hands.items():
            if h.played > best_count:
                best = ht
                best_count = h.played
        return best

    def level_up_all(self, amount: int = 1) -> None:
        """Level up ALL hand types by *amount* (Black Hole effect)."""
        for ht in self._hands:
            self.level_up(ht, amount)

    def __getitem__(self, hand_type: HandType | str) -> HandState:
        """Dict-like access: ``levels[HandType.PAIR]`` or ``levels["Pair"]``."""
        # HandType is a StrEnum, so members also pass isinstance(str); skip
        # the (identity) HandType() round-trip for them.
        if type(hand_type) is not HandType and isinstance(hand_type, str):
            hand_type = HandType(hand_type)
        return self._hands[hand_type]

    def __contains__(self, hand_type: HandType | str) -> bool:
        try:
            ht = HandType(hand_type) if isinstance(hand_type, str) else hand_type
            return ht in self._hands
        except ValueError:
            return False

    def __repr__(self) -> str:
        levels = {ht.value: h.level for ht, h in self._hands.items() if h.level > 1}
        if not levels:
            return "HandLevels(all at level 1)"
        return f"HandLevels({levels})"
