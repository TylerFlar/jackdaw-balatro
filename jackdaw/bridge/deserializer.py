"""Deserialize balatrobot JSON-RPC calls into engine Actions.

The reverse of ``action_to_rpc()`` in ``balatrobot_adapter.py`` — given a
JSON-RPC method name and params dict, produce the correct engine Action.
"""

from __future__ import annotations

from jackdaw.engine.actions import (
    Action,
    BuyCard,
    CashOut,
    Discard,
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
    UseConsumable,
)

# Methods that are queries or lifecycle commands, not game actions.
_QUERY_METHODS = frozenset({
    "gamestate",
    "health",
    "start",
    "menu",
    "save",
    "load",
    "screenshot",
    "set",
    "add",
    "rpc.discover",
})


def rpc_to_action(method: str, params: dict | None = None) -> Action | None:
    """Convert a balatrobot JSON-RPC method + params to an engine Action.

    Returns ``None`` for query-only methods (gamestate, health, start, etc.).
    Raises ``ValueError`` for unknown methods or malformed params.
    """
    if params is None:
        params = {}

    if method in _QUERY_METHODS:
        return None

    if method == "play":
        return PlayHand(card_indices=tuple(params.get("cards", ())))

    if method == "discard":
        return Discard(card_indices=tuple(params.get("cards", ())))

    if method == "select":
        return SelectBlind()

    if method == "skip":
        return SkipBlind()

    if method == "buy":
        if "card" in params:
            return BuyCard(shop_index=params["card"])
        if "voucher" in params:
            return RedeemVoucher(card_index=params["voucher"])
        if "pack" in params:
            return OpenBooster(card_index=params["pack"])
        raise ValueError(f"buy: unrecognized params {params!r}")

    if method == "sell":
        if "joker" in params:
            return SellCard(area="jokers", card_index=params["joker"])
        if "consumable" in params:
            return SellCard(area="consumables", card_index=params["consumable"])
        raise ValueError(f"sell: unrecognized params {params!r}")

    if method == "use":
        if "consumable" not in params:
            raise ValueError(f"use: missing 'consumable' in params {params!r}")
        targets = params.get("cards")
        return UseConsumable(
            card_index=params["consumable"],
            target_indices=tuple(targets) if targets else None,
        )

    if method == "reroll":
        return Reroll()

    if method == "next_round":
        return NextRound()

    if method == "cash_out":
        return CashOut()

    if method == "pack":
        if params.get("skip"):
            return SkipPack()
        if "card" in params:
            targets = params.get("targets")
            return PickPackCard(
                card_index=params["card"],
                target_indices=tuple(targets) if targets else None,
            )
        raise ValueError(f"pack: unrecognized params {params!r}")

    if method == "rearrange":
        if "hand" in params:
            return ReorderHand(new_order=tuple(params["hand"]))
        if "jokers" in params:
            return ReorderJokers(new_order=tuple(params["jokers"]))
        raise ValueError(f"rearrange: unrecognized params {params!r}")

    raise ValueError(f"Unknown RPC method: {method!r}")
