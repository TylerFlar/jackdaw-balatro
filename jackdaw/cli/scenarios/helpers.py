"""Shared helpers for validation scenarios.

Every helper takes the lockstep ``ls``, a
:class:`~jackdaw.bridge.live.LiveAdapter`: each request goes to the shadow run
(the simulator) first, then to the live game, and every live response is
compared with the shadow's view on every structural field. "Both" in a helper's
name means the shadow and the live game. A difference the deviation catalog
doesn't explain raises :class:`~jackdaw.bridge.live.DivergenceError`, which
``jackdaw validate`` reports as a failure with its repro bundle.

Parameters are BalatroBot's: ``enhancement="GLASS"``, ``edition="FOIL"``,
``seal="RED"``, playing cards as ``"H_A"``.
"""

from __future__ import annotations

import time
from typing import Any

from jackdaw.bridge.live import LiveAdapter
from jackdaw.cli.scenarios import ScenarioResult

#: The lockstep a scenario plays on.
Lockstep = LiveAdapter

# Default delay between actions (overridden per-run via ``delay`` kwarg)
_DEFAULT_DELAY = 0.3


def pause(delay: float, at_least: float = 0.0) -> None:
    """Wait between actions to watch the live game (``delay`` 0: never)."""
    if delay > 0:
        time.sleep(max(delay, at_least))


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def start_both(
    ls: Lockstep,
    *,
    seed: str = "SCENARIO",
    deck: str = "RED",
    stake: str = "WHITE",
    delay: float = _DEFAULT_DELAY,
) -> None:
    """Start a run on the live game (``menu`` + ``start``) and its shadow."""
    ls.start(deck, stake, seed)
    pause(delay)


def select_blind(ls: Lockstep, *, delay: float = _DEFAULT_DELAY) -> None:
    """Select the current blind."""
    ls.request("select")
    pause(delay)


def skip_blind(ls: Lockstep, *, delay: float = _DEFAULT_DELAY) -> None:
    """Skip the current blind."""
    ls.request("skip")
    pause(delay)


def add_both(ls: Lockstep, **params: Any) -> None:
    """BalatroBot's ``add`` (debug) in lockstep."""
    ls.request("add", params)


def set_both(ls: Lockstep, **params: Any) -> None:
    """BalatroBot's ``set`` (debug) in lockstep."""
    ls.request("set", params)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def play_hand(ls: Lockstep, cards: list[int], *, delay: float = _DEFAULT_DELAY) -> None:
    """Play the hand cards at these indices."""
    ls.request("play", {"cards": cards})
    pause(delay, 1.5)  # scoring animation


def discard(ls: Lockstep, cards: list[int], *, delay: float = _DEFAULT_DELAY) -> None:
    """Discard the hand cards at these indices."""
    ls.request("discard", {"cards": cards})
    pause(delay)


def use_consumable(
    ls: Lockstep,
    consumable: int,
    cards: list[int] | None = None,
    *,
    delay: float = _DEFAULT_DELAY,
) -> None:
    """Use a consumable, on these hand cards if given."""
    params: dict[str, Any] = {"consumable": consumable}
    if cards is not None:
        params["cards"] = cards
    ls.request("use", params)
    pause(delay)


def cash_out(ls: Lockstep, *, delay: float = _DEFAULT_DELAY) -> None:
    ls.request("cash_out")
    pause(delay, 1.5)


def next_round(ls: Lockstep, *, delay: float = _DEFAULT_DELAY) -> None:
    """Leave the shop."""
    ls.request("next_round")
    pause(delay)


def sell_joker(ls: Lockstep, index: int, *, delay: float = _DEFAULT_DELAY) -> None:
    ls.request("sell", {"joker": index})
    pause(delay)


def sell_consumable(ls: Lockstep, index: int, *, delay: float = _DEFAULT_DELAY) -> None:
    ls.request("sell", {"consumable": index})
    pause(delay)


def buy_card(ls: Lockstep, index: int, *, delay: float = _DEFAULT_DELAY) -> None:
    """Buy a shop card."""
    ls.request("buy", {"card": index})
    pause(delay)


def reroll_shop(ls: Lockstep, *, delay: float = _DEFAULT_DELAY) -> None:
    ls.request("reroll")
    pause(delay)


def gamestate(ls: Lockstep) -> dict[str, Any]:
    """The live game's last view (equal to the shadow's, or the step would have
    raised)."""
    return ls.view


def get_state(ls: Lockstep) -> str:
    """The game state's name (``SELECTING_HAND``, ``ROUND_EVAL``, ...)."""
    return str(ls.view.get("state", ""))


def get_hand_count(ls: Lockstep) -> int:
    """The number of cards in hand."""
    return int((ls.view.get("hand") or {}).get("count", 0))


def current_blind_score(ls: Lockstep, default: float = 600) -> float:
    """The score the current blind needs."""
    for b in (ls.view.get("blinds") or {}).values():
        if isinstance(b, dict) and b.get("status") == "CURRENT":
            return b.get("score", default)
    return default


def play_through_blind(
    ls: Lockstep,
    *,
    max_hands: int = 4,
    delay: float = _DEFAULT_DELAY,
) -> bool:
    """Play hands until the blind is beaten or the game ends.

    Returns True if the blind was beaten (state becomes ROUND_EVAL),
    False if the game ended (GAME_OVER) or hands ran out.
    """
    for _ in range(max_hands):
        if get_state(ls) != "SELECTING_HAND":
            break
        play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    return get_state(ls) == "ROUND_EVAL"


def advance_past_blind(ls: Lockstep, *, delay: float = _DEFAULT_DELAY) -> bool:
    """Select blind, play through it, cash out, advance to next round.

    Returns True if successful, False if game ended.
    """
    select_blind(ls, delay=delay)
    if not play_through_blind(ls, delay=delay):
        return False
    cash_out(ls, delay=delay)
    next_round(ls, delay=delay)
    return True


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_state(ls: Lockstep, *, label: str = "", **_: Any) -> list[str]:
    """Compare the live ``gamestate`` with the shadow once more; the differences the
    deviation catalog doesn't drop, as strings (each request was compared already)."""
    return [f"{label}: {d}" if label else str(d) for d in ls.check(raise_on_divergence=False)]


def outcome(name: str, diffs: list[str]) -> ScenarioResult:
    """A scenario's result from its remaining differences."""
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"{name}: {'PASS' if not diffs else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Scenario runner pattern
# ---------------------------------------------------------------------------


def run_joker_scenario(
    ls: Lockstep,
    *,
    joker_key: str,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
    edition: str | None = None,
) -> ScenarioResult:
    """Standard pattern: start game, select blind, add joker, play hand[0:5], compare.

    This is the workhorse for most joker scenarios.
    """
    if not seed:
        seed = f"J_{joker_key.upper()}"

    start_both(ls, seed=seed, delay=delay)
    select_blind(ls, delay=delay)

    add_kwargs: dict[str, Any] = {"key": joker_key}
    if edition:
        add_kwargs["edition"] = edition
    add_both(ls, **add_kwargs)

    # Play the first 5 cards from hand
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)

    return outcome(f"Joker {joker_key}", compare_state(ls, label=f"after play with {joker_key}"))


# ---------------------------------------------------------------------------
# Hand presets — known card sets for triggering specific joker conditions
# ---------------------------------------------------------------------------

HAND_PRESETS: dict[str, list[str]] = {
    # Poker hand types
    "PAIR": ["H_A", "D_A", "S_K", "C_Q", "H_J"],
    "THREE_KIND": ["H_A", "D_A", "S_A", "C_K", "H_Q"],
    "FOUR_KIND": ["H_A", "D_A", "S_A", "C_A", "H_K"],
    "TWO_PAIR": ["H_A", "D_A", "S_K", "C_K", "H_Q"],
    "FULL_HOUSE": ["H_A", "D_A", "S_A", "C_K", "H_K"],
    "STRAIGHT": ["H_5", "D_6", "S_7", "C_8", "H_9"],
    "FLUSH_HEARTS": ["H_2", "H_5", "H_8", "H_J", "H_A"],
    "FLUSH_DIAMONDS": ["D_2", "D_5", "D_8", "D_J", "D_A"],
    "FLUSH_CLUBS": ["C_2", "C_5", "C_8", "C_J", "C_A"],
    "FLUSH_SPADES": ["S_2", "S_5", "S_8", "S_J", "S_A"],
    # Rank-specific
    "FACE_CARDS": ["H_J", "D_Q", "S_K", "C_J", "H_Q"],
    "EVEN_RANKS": ["H_2", "D_4", "S_6", "C_8", "H_T"],
    "ODD_RANKS": ["H_3", "D_5", "S_7", "C_9", "H_A"],
    "FIBONACCI": ["H_A", "D_2", "S_3", "C_5", "H_8"],
    "WITH_ACES": ["H_A", "D_A", "S_3", "C_5", "H_8"],
    "WITH_TENS_FOURS": ["H_T", "D_4", "S_T", "C_4", "H_7"],
    "KINGS_QUEENS": ["H_K", "D_K", "S_Q", "C_Q", "H_J"],
    "HACK_RANKS": ["H_2", "D_3", "S_4", "C_5", "H_2"],
    "WITH_EIGHTS": ["H_8", "D_8", "S_3", "C_5", "H_A"],
    "WITH_SIXES": ["H_6", "D_6", "S_8", "C_9", "H_T"],
    "WITH_WEE_TWOS": ["H_2", "D_2", "S_3", "C_5", "H_A"],
    # Multi-condition
    "ALL_SUITS": ["H_A", "D_K", "C_Q", "S_J", "H_T"],
    "STRAIGHT_ACE": ["H_A", "D_T", "S_J", "C_Q", "H_K"],
    "STRAIGHT_FLUSH": ["H_5", "H_6", "H_7", "H_8", "H_9"],
    "SPADES_CLUBS": ["S_2", "S_5", "C_8", "C_J", "S_A"],
    "HIGH_CARD": ["H_2", "D_5", "S_8", "C_J", "H_A"],
    # Variable-count hands
    "THREE_CARDS": ["H_A", "D_K", "S_Q"],
    "FOUR_CARDS": ["H_A", "D_K", "S_Q", "C_J"],
}


def run_joker_with_setup(
    ls: Lockstep,
    *,
    joker_key: str,
    hand_preset: str | list[str] | None = None,
    play_count: int | None = None,
    pre_discard: list[str] | None = None,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
    edition: str | None = None,
) -> ScenarioResult:
    """Enhanced joker scenario: inject specific cards, optionally discard, play.

    Args:
        joker_key: The joker center key (e.g. "j_jolly").
        hand_preset: Preset name from HAND_PRESETS, or a list of card keys.
            If None, falls back to standard play [0..4].
        play_count: Number of injected cards to play. Defaults to len(cards).
        pre_discard: Card keys to inject and discard before the main hand.
        seed: Run seed.
        delay: Action delay.
        edition: Optional joker edition (BalatroBot's name: ``FOIL``, ...).
    """
    if not seed:
        seed = f"J_{joker_key.upper()}"

    start_both(ls, seed=seed, delay=delay)
    select_blind(ls, delay=delay)

    # Add the joker
    add_kwargs: dict[str, Any] = {"key": joker_key}
    if edition:
        add_kwargs["edition"] = edition
    add_both(ls, **add_kwargs)

    # No preset — fall back to standard behavior
    if hand_preset is None:
        play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
        return outcome(
            f"Joker {joker_key}", compare_state(ls, label=f"after play with {joker_key}")
        )

    # Resolve preset
    cards = HAND_PRESETS[hand_preset] if isinstance(hand_preset, str) else hand_preset
    if play_count is None:
        play_count = len(cards)

    # Pre-discard phase (for discard-triggered jokers)
    if pre_discard:
        for card_key in pre_discard:
            add_both(ls, key=card_key)
        count = get_hand_count(ls)
        discard(ls, list(range(count - len(pre_discard), count)), delay=delay)

    # Inject hand cards
    for card_key in cards:
        add_both(ls, key=card_key)

    # Play the last N injected cards
    count = get_hand_count(ls)
    play_hand(ls, list(range(count - play_count, count)), delay=delay)

    return outcome(f"Joker {joker_key}", compare_state(ls, label=f"after play with {joker_key}"))


def run_consumable_scenario(
    ls: Lockstep,
    *,
    consumable_key: str,
    targets: list[int] | None = None,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
) -> ScenarioResult:
    """Standard pattern: start game, select blind, add consumable, use it, compare."""
    if not seed:
        seed = f"C_{consumable_key.upper()}"

    start_both(ls, seed=seed, delay=delay)
    select_blind(ls, delay=delay)
    add_both(ls, key=consumable_key)
    use_consumable(ls, 0, cards=targets, delay=delay)

    return outcome(
        f"Consumable {consumable_key}", compare_state(ls, label=f"after using {consumable_key}")
    )


def run_modifier_scenario(
    ls: Lockstep,
    *,
    card_key: str,
    enhancement: str | None = None,
    edition: str | None = None,
    seal: str | None = None,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
) -> ScenarioResult:
    """Standard pattern: start game, select blind, add modified card, play it, compare.

    ``enhancement``, ``edition`` and ``seal`` are BalatroBot's names (``GLASS``,
    ``FOIL``, ``RED``)."""
    label = enhancement or "BASE"
    if not seed:
        # The seeds the scenarios have always used (the enhancement's center key).
        center = f"m_{enhancement.lower()}" if enhancement else "c_base"
        seed = f"M_{center}_{card_key}"

    start_both(ls, seed=seed, delay=delay)
    select_blind(ls, delay=delay)

    add_kwargs: dict[str, Any] = {"key": card_key}
    if enhancement:
        add_kwargs["enhancement"] = enhancement
    if edition:
        add_kwargs["edition"] = edition
    if seal:
        add_kwargs["seal"] = seal
    add_both(ls, **add_kwargs)

    # Play hand including the added card (it should be at the end of hand)
    # We play indices 0-4 which will include the dealt hand
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)

    if edition:
        label += f"+{edition}"
    if seal:
        label += f"+{seal}"

    return outcome(
        f"Modifier {label} on {card_key}",
        compare_state(ls, label=f"after play with {label} {card_key}"),
    )
