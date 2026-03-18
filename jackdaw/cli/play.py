"""Interactive terminal play — play Balatro in your terminal.

Uses SimBackend in-process (no server, no HTTP). Reads game state
directly from ``backend._gs`` for display, executes actions via
``step()`` for speed.

Usage::

    from jackdaw.cli.play import run_play
    run_play("ABCD1234", "RED", "WHITE")
"""

from __future__ import annotations

import sys
from typing import Any

from jackdaw.bridge.backend import SimBackend
from jackdaw.engine.actions import (
    Action,
    BuyAndUse,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    ReorderHand,
    ReorderJokers,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.card import Card
from jackdaw.engine.game import IllegalActionError, step

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_USE_COLOR: bool | None = None


def _color_enabled() -> bool:
    global _USE_COLOR  # noqa: PLW0603
    if _USE_COLOR is None:
        _USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return _USE_COLOR


def _c(code: str, text: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _c("1", text)


def _red(text: str) -> str:
    return _c("31", text)


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _dim(text: str) -> str:
    return _c("2", text)


# ---------------------------------------------------------------------------
# Card display
# ---------------------------------------------------------------------------

_RANK_SHORT: dict[str, str] = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "Jack": "J",
    "Queen": "Q",
    "King": "K",
    "Ace": "A",
}


def _suit_symbols() -> dict[str, str]:
    """Return suit symbols, falling back to letters if stdout can't encode them."""
    fancy = {"Hearts": "\u2665", "Diamonds": "\u2666", "Clubs": "\u2663", "Spades": "\u2660"}
    try:
        for v in fancy.values():
            v.encode(sys.stdout.encoding or "utf-8")
        return fancy
    except (UnicodeEncodeError, LookupError):
        return {"Hearts": "h", "Diamonds": "d", "Clubs": "c", "Spades": "s"}


_SUIT_SYMBOL: dict[str, str] = _suit_symbols()


def _fmt_playing_card(card: Card) -> str:
    """Format a playing card: 'A♠' with color and modifiers."""
    if card.base is None:
        return _fmt_non_playing_card(card)
    rank = _RANK_SHORT.get(card.base.rank.value, "?")
    suit_str = card.base.suit.value
    symbol = _SUIT_SYMBOL.get(suit_str, "?")
    text = f"{rank}{symbol}"
    if suit_str in ("Hearts", "Diamonds"):
        text = _red(text)

    # Enhancement
    enh = _get_enhancement(card)
    if enh:
        text = f"{text}{_dim(enh)}"

    # Edition
    ed = _get_edition_tag(card)
    if ed:
        text = f"{text}{_dim(ed)}"

    # Seal
    if card.seal:
        text = f"{text}{_dim(f'[{card.seal[0]}]')}"

    if card.debuff:
        text = f"{_dim('X')}{text}"

    return text


def _get_enhancement(card: Card) -> str:
    """Return short enhancement tag or empty string."""
    ck = card.center_key
    if not ck or ck == "c_base":
        return ""
    _map = {
        "m_bonus": "+",
        "m_mult": "*",
        "m_wild": "W",
        "m_glass": "G",
        "m_steel": "S",
        "m_stone": "#",
        "m_gold": "$",
        "m_lucky": "?",
    }
    return _map.get(ck, "")


def _get_edition_tag(card: Card) -> str:
    """Return short edition tag or empty string."""
    if not card.edition:
        return ""
    if card.edition.get("foil"):
        return "f"
    if card.edition.get("holo"):
        return "h"
    if card.edition.get("polychrome"):
        return "p"
    if card.edition.get("negative"):
        return "n"
    return ""


def _fmt_non_playing_card(card: Card) -> str:
    """Format a joker, consumable, voucher, or booster."""
    name = card.ability.get("name", card.center_key or "?")
    card_set = card.ability.get("set", "")
    if card_set == "Joker":
        return _yellow(name)
    return name


def _fmt_joker(card: Card) -> str:
    """Format a joker with key ability info."""
    name = _yellow(card.ability.get("name", "?"))
    extras: list[str] = []
    ed = _get_edition_tag(card)
    if ed:
        extras.append(f"ed:{ed}")
    if card.eternal:
        extras.append("eternal")
    if card.perishable:
        extras.append(f"perish:{card.perish_tally}")
    if card.rental:
        extras.append("rental")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{name}{suffix}"


def _fmt_shop_card(card: Card) -> str:
    """Format a shop card with cost."""
    name = _fmt_non_playing_card(card)
    cost = _green(f"${card.cost}")
    return f"{name} {cost}"


# ---------------------------------------------------------------------------
# State display
# ---------------------------------------------------------------------------


def _show_header(gs: dict[str, Any]) -> None:
    phase = gs.get("phase", "?")
    rr = gs["round_resets"]
    ante = rr["ante"]
    rnd = gs.get("round", 0)
    money = gs.get("dollars", 0)
    seed = ""
    rng = gs.get("rng")
    if rng:
        seed = rng._state.get("seed", "?") if hasattr(rng, "_state") else "?"
    print()
    print(
        _bold(f"=== {phase} === ")
        + f"Ante {ante} | Round {rnd} | {_green(f'${money}')} | Seed: {seed}"
    )


def _show_blind_select(gs: dict[str, Any]) -> None:
    from jackdaw.engine.blind import get_blind_amount
    from jackdaw.engine.data.prototypes import BLINDS, TAGS

    rr = gs["round_resets"]
    ante = rr["ante"]
    scaling = gs.get("modifiers", {}).get("scaling", 1)
    ante_scaling = gs["starting_params"].get("ante_scaling", 1.0)
    blind_on_deck = gs.get("blind_on_deck", "Small")
    blind_tags = rr.get("blind_tags", {})
    choices = rr["blind_choices"]

    for label in ("Small", "Big", "Boss"):
        key = choices.get(label, f"bl_{label.lower()}")
        proto = BLINDS.get(key)
        if proto is None:
            continue
        base = get_blind_amount(ante, scaling)
        chips = int(base * proto.mult * ante_scaling)
        marker = " <<" if label == blind_on_deck else ""
        tag_key = blind_tags.get(label)
        tag_name = ""
        if tag_key:
            tp = TAGS.get(tag_key)
            tag_name = f" | Tag: {tp.name}" if tp else f" | Tag: {tag_key}"
        print(f"  {label:5s}: {proto.name:<20s} chips: {chips:>8,}{tag_name}{_bold(marker)}")


def _show_selecting_hand(gs: dict[str, Any]) -> None:
    hand: list[Card] = gs.get("hand", [])
    jokers: list[Card] = gs.get("jokers", [])
    consumables: list[Card] = gs.get("consumables", [])
    cr = gs.get("current_round", {})
    blind = gs.get("blind")

    # Blind target
    if blind:
        chips = gs.get("chips", 0)
        target = blind.chips
        pct = (chips / target * 100) if target else 0
        boss_info = f" ({blind.name})" if blind.boss else ""
        print(f"  Blind: {chips:,}/{target:,} chips ({pct:.0f}%){boss_info}")

    print(f"  Hands: {cr.get('hands_left', 0)}  Discards: {cr.get('discards_left', 0)}")

    # Hand
    print(f"  Hand ({len(hand)}):")
    parts: list[str] = []
    for i, c in enumerate(hand):
        parts.append(f"{i}:{_fmt_playing_card(c)}")
    # Print in rows of 10
    for row_start in range(0, len(parts), 10):
        print(f"    {' '.join(parts[row_start : row_start + 10])}")

    # Jokers
    if jokers:
        print(f"  Jokers ({len(jokers)}):")
        for i, j in enumerate(jokers):
            print(f"    {i}: {_fmt_joker(j)}")

    # Consumables
    if consumables:
        cons_parts = [f"{i}:{_fmt_non_playing_card(c)}" for i, c in enumerate(consumables)]
        print(f"  Consumables: {' '.join(cons_parts)}")


def _show_shop(gs: dict[str, Any]) -> None:
    shop_cards: list[Card] = gs.get("shop_cards", [])
    shop_vouchers: list[Card] = gs.get("shop_vouchers", [])
    shop_boosters: list[Card] = gs.get("shop_boosters", [])
    jokers: list[Card] = gs.get("jokers", [])
    consumables: list[Card] = gs.get("consumables", [])
    money = gs.get("dollars", 0)

    print(f"  Money: {_green(f'${money}')}")

    if shop_cards:
        parts = [f"{i}:{_fmt_shop_card(c)}" for i, c in enumerate(shop_cards)]
        print(f"  Shop cards: {' | '.join(parts)}")
    if shop_vouchers:
        parts = [f"{i}:{_fmt_shop_card(c)}" for i, c in enumerate(shop_vouchers)]
        print(f"  Vouchers: {' | '.join(parts)}")
    if shop_boosters:
        parts = [f"{i}:{_fmt_shop_card(c)}" for i, c in enumerate(shop_boosters)]
        print(f"  Boosters: {' | '.join(parts)}")
    if jokers:
        j_parts = [_fmt_joker(j) for j in jokers]
        print(f"  Jokers: {', '.join(j_parts)}")
    if consumables:
        c_parts = [f"{_fmt_non_playing_card(c)}" for c in consumables]
        print(f"  Consumables: {', '.join(c_parts)}")


def _show_pack_opening(gs: dict[str, Any]) -> None:
    pack_cards: list[Card] = gs.get("pack_cards", [])
    remaining = gs.get("pack_choices_remaining", 0)
    print(f"  Picks remaining: {remaining}")
    for i, c in enumerate(pack_cards):
        if c.base:
            print(f"    {i}: {_fmt_playing_card(c)}")
        else:
            print(f"    {i}: {_fmt_shop_card(c)}")


def _show_round_eval(gs: dict[str, Any]) -> None:
    blind = gs.get("blind")
    chips = gs.get("chips", 0)
    target = blind.chips if blind else 0
    last = gs.get("last_score_result")

    if chips >= target:
        print(_bold("  You beat the blind!"))
    print(f"  Chips: {chips:,} / {target:,}")

    if last:
        chips_str = int(last.chips)
        mult_str = last.mult
        total_str = last.total
        print(f"  Last hand: {last.hand_type} — {chips_str} x {mult_str:.1f} = {total_str:,}")

    # Reward
    if blind:
        print(f"  Reward: ${blind.dollars}")


def _show_game_over(gs: dict[str, Any]) -> None:
    won = gs.get("won", False)
    rr = gs["round_resets"]
    ante = rr["ante"]
    rnd = gs.get("round", 0)
    money = gs.get("dollars", 0)

    if won:
        print(_bold(_green("  YOU WIN!")))
    else:
        print(_bold(_red("  GAME OVER")))
    print(f"  Ante: {ante}  Round: {rnd}  Money: ${money}")


def _show_state(gs: dict[str, Any]) -> None:
    """Display the current game state."""
    _show_header(gs)
    phase = gs.get("phase")
    if isinstance(phase, str):
        phase = GamePhase(phase)

    if phase == GamePhase.BLIND_SELECT:
        _show_blind_select(gs)
    elif phase == GamePhase.SELECTING_HAND:
        _show_selecting_hand(gs)
    elif phase == GamePhase.SHOP:
        _show_shop(gs)
    elif phase == GamePhase.PACK_OPENING:
        _show_pack_opening(gs)
    elif phase == GamePhase.ROUND_EVAL:
        _show_round_eval(gs)
    elif phase == GamePhase.GAME_OVER:
        _show_game_over(gs)


# ---------------------------------------------------------------------------
# Action display + selection
# ---------------------------------------------------------------------------


def _action_label(action: Action, gs: dict[str, Any]) -> str:
    """Human-readable label for an action."""
    match action:
        case SelectBlind():
            return "Select Blind"
        case SkipBlind():
            return "Skip Blind"
        case PlayHand(card_indices=idx):
            return "Play Hand" if not idx else f"Play {list(idx)}"
        case Discard(card_indices=idx):
            return "Discard" if not idx else f"Discard {list(idx)}"
        case CashOut():
            return "Cash Out"
        case NextRound():
            return "Next Round"
        case Reroll():
            cr = gs.get("current_round", {})
            cost = cr.get("reroll_cost", 5)
            return f"Reroll (${cost})"
        case BuyCard(shop_index=i):
            cards = gs.get("shop_cards", [])
            name = cards[i].ability.get("name", "?") if i < len(cards) else "?"
            cost = cards[i].cost if i < len(cards) else "?"
            return f"Buy {name} (${cost})"
        case BuyAndUse(shop_index=i):
            cards = gs.get("shop_cards", [])
            name = cards[i].ability.get("name", "?") if i < len(cards) else "?"
            return f"Buy+Use {name}"
        case SellCard(area=area, card_index=i):
            card_list = gs.get(area, [])
            name = "?"
            sell = 0
            if i < len(card_list):
                c = card_list[i]
                name = c.ability.get("name", "?")
                sell = c.sell_cost
            return f"Sell {name} (${sell})"
        case UseConsumable(card_index=i):
            cons = gs.get("consumables", [])
            name = cons[i].ability.get("name", "?") if i < len(cons) else "?"
            return f"Use {name}"
        case RedeemVoucher(card_index=i):
            vouchers = gs.get("shop_vouchers", [])
            name = vouchers[i].ability.get("name", "?") if i < len(vouchers) else "?"
            return f"Redeem {name}"
        case OpenBooster(card_index=i):
            boosters = gs.get("shop_boosters", [])
            name = boosters[i].ability.get("name", "?") if i < len(boosters) else "?"
            return f"Open {name}"
        case PickPackCard(card_index=i):
            cards = gs.get("pack_cards", [])
            if i < len(cards):
                c = cards[i]
                name = _fmt_playing_card(c) if c.base else _fmt_non_playing_card(c)
            else:
                name = "?"
            return f"Pick {name}"
        case SkipPack():
            return "Skip Pack"
        case SortHand(mode=mode):
            return f"Sort by {mode}"
        case ReorderHand():
            return "Reorder Hand"
        case ReorderJokers():
            return "Reorder Jokers"
        case _:
            return type(action).__name__


def _read_indices(
    prompt: str, hand_size: int, min_cards: int = 1, max_cards: int = 5
) -> tuple[int, ...] | None:
    """Prompt for space-separated card indices. Returns None on cancel."""
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "cancel", "back"):
            return None
        try:
            indices = [int(x) for x in raw.split()]
        except ValueError:
            print("  Enter space-separated numbers (e.g. 0 1 2). 'q' to cancel.")
            continue
        if not (min_cards <= len(indices) <= max_cards):
            print(f"  Select {min_cards}-{max_cards} cards.")
            continue
        if any(i < 0 or i >= hand_size for i in indices):
            print(f"  Indices must be 0-{hand_size - 1}.")
            continue
        if len(set(indices)) != len(indices):
            print("  No duplicates.")
            continue
        return tuple(sorted(indices))


def _prompt_action(actions: list[Action], gs: dict[str, Any]) -> Action | None:
    """Display actions and get player choice. Returns None on quit."""
    print()
    for i, action in enumerate(actions):
        label = _action_label(action, gs)
        print(f"  {_bold(str(i)):>4s}) {label}")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        try:
            choice = int(raw)
        except ValueError:
            print("  Enter a number. 'q' to quit.")
            continue
        if not (0 <= choice < len(actions)):
            print(f"  Choose 0-{len(actions) - 1}.")
            continue

        action = actions[choice]
        hand: list[Card] = gs.get("hand", [])

        # PlayHand marker — need card indices
        if isinstance(action, PlayHand) and not action.card_indices:
            indices = _read_indices(
                f"  Cards to play (0-{len(hand) - 1}, 1-5 cards): ",
                len(hand),
            )
            if indices is None:
                continue
            return PlayHand(card_indices=indices)

        # Discard marker — need card indices
        if isinstance(action, Discard) and not action.card_indices:
            indices = _read_indices(
                f"  Cards to discard (0-{len(hand) - 1}, 1-5 cards): ",
                len(hand),
            )
            if indices is None:
                continue
            return Discard(card_indices=indices)

        # UseConsumable — may need targets
        if isinstance(action, UseConsumable) and action.target_indices is None:
            cons = gs.get("consumables", [])
            if action.card_index < len(cons):
                c = cons[action.card_index]
                cfg = c.ability.get("consumeable", {})
                max_highlighted = cfg.get("max_highlighted") if isinstance(cfg, dict) else None
                if max_highlighted and hand:
                    indices = _read_indices(
                        f"  Target cards (0-{len(hand) - 1}, up to {max_highlighted}): ",
                        len(hand),
                        min_cards=1,
                        max_cards=max_highlighted,
                    )
                    if indices is None:
                        continue
                    return UseConsumable(card_index=action.card_index, target_indices=indices)

        return action


# ---------------------------------------------------------------------------
# Score display after play
# ---------------------------------------------------------------------------


def _show_score_result(gs: dict[str, Any]) -> None:
    """Show scoring result after a PlayHand action."""
    result = gs.get("last_score_result")
    if result is None:
        return
    blind = gs.get("blind")
    chips = gs.get("chips", 0)
    target = blind.chips if blind else 0

    hand_str = _bold(result.hand_type)
    score_str = f"{int(result.chips)} x {result.mult:.1f} = {_bold(f'{result.total:,}')}"
    total_str = f"{chips:,}/{target:,}"
    print(f"  {hand_str}: {score_str}  [{total_str}]")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_play(seed: str, deck: str, stake: str) -> None:
    """Run an interactive terminal game."""
    backend = SimBackend()
    backend.handle("start", {"deck": deck, "stake": stake, "seed": seed})
    gs = backend._gs
    assert gs is not None

    print(_bold("\nJackdaw - Balatro Simulator"))
    print(f"Seed: {seed}  Deck: {deck}  Stake: {stake}")
    print(_dim("Enter action number to play. 'q' to quit.\n"))

    try:
        while True:
            phase = gs.get("phase")
            if isinstance(phase, str):
                phase = GamePhase(phase)

            if phase == GamePhase.GAME_OVER:
                _show_state(gs)
                break

            if gs.get("won") and phase == GamePhase.SHOP:
                _show_state(gs)
                print(_bold(_green("\n  YOU WIN!")))
                break

            _show_state(gs)

            legal = get_legal_actions(gs)
            if not legal:
                print("  No legal actions available.")
                break

            action = _prompt_action(legal, gs)
            if action is None:
                print("\nGoodbye!")
                break

            was_play = isinstance(action, PlayHand) and action.card_indices

            try:
                step(gs, action)
            except IllegalActionError as exc:
                print(f"  {_red('Illegal action')}: {exc}")
                continue

            if was_play:
                _show_score_result(gs)

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
