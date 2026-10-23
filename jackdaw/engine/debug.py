"""Debug operations on an oracle game state: BalatroBot's ``add`` and ``set``.

These write state that no player action reaches.  The parity harness and the
golden v1 trajectories set up their boards with them, and the Rust core's
``Game.debug_add`` / ``debug_set`` / ``debug_restock_shop``
(``crates/jackdaw-core/src/debug.rs``) follow them, so a setup gives the same
canonical snapshot in both engines.

They were ``SimBackend._handle_add`` / ``_handle_set`` / ``_restock_shop``;
``SimBackend`` now runs the Rust BalatroBot layer (``jackdaw-bot``), whose
``add`` and ``set`` endpoints validate their arguments the way BalatroBot
v1.5.2 does (third_party/balatrobot/v1.5.2/src/lua/endpoints/add.lua,
set.lua).  Here arguments use the engine's names: ``seal="Red"``,
``edition="foil"``, ``enhancement="m_steel"``.

Known differences from the live game are rows in ``docs/parity/gaps.md``
(B15, B18-B21).
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import GamePhase


def price_and_mark(gs: dict[str, Any], card: Any) -> None:
    """What Card:init does for a card ``add`` creates: set_cost with the run's
    prices (card.lua:72) and the key's used_jokers mark (card.lua:349-354)."""
    from jackdaw.engine.card_utils import astronomer_active

    card.set_cost(
        inflation=gs.get("inflation", 0),
        discount_percent=gs.get("discount_percent", 0),
        ante=gs.get("round_resets", {}).get("ante", 1),
        has_astronomer=astronomer_active(gs),
    )
    gs.setdefault("used_jokers", {})[card.center_key] = True


def add_card(
    gs: dict[str, Any],
    key: str,
    *,
    seal: str | None = None,
    edition: str | None = None,
    enhancement: str = "c_base",
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
) -> Any:
    """BalatroBot's ``add`` (``SMODS.add_card``): create the card and put it in play.

    * a joker (``j_*``): create, roll To Do List's hand, poll ``'edi'..ante``
      (also when an edition is given, B15), price and mark, append to the
      jokers, ``add_to_deck``; a bigger hand mid-round is filled at once;
    * a consumable (``c_*``): create, price and mark, ``add_to_deck``, append;
    * a playing card (``H_A``): into the hand in SELECTING_HAND, else onto
      the end of the deck list (the top of the draw pile).

    Returns the card.  Raises ``ValueError`` for a key it cannot add.
    """
    from jackdaw.engine.card_factory import (
        RANK_LETTER,
        SUIT_LETTER,
        create_consumable,
        create_joker,
        create_playing_card,
    )
    from jackdaw.engine.card_utils import poll_edition

    edition_flags = {edition: True} if edition else None
    phase = gs.get("phase")

    if key.startswith("j_"):
        card = create_joker(
            key,
            edition=edition_flags,
            eternal=eternal,
            perishable=perishable,
            rental=rental,
            hands_played=gs.get("hands_played", 0),
        )
        # To Do List rolls its target at creation, before the edition poll
        # (card.lua:311): ``add`` goes through the full create_card pipeline.
        if key == "j_todo_list":
            from jackdaw.engine.card_factory import roll_to_do_hand

            roll_to_do_hand(card, gs.get("rng"), gs.get("hand_levels"))
        # create_card's poll_edition keeps the RNG stream where the live
        # game's is.
        rng = gs.get("rng")
        if rng is not None:
            ante = gs.get("round_resets", {}).get("ante", 1)
            polled = poll_edition("edi" + str(ante), rng, rate=gs.get("edition_rate", 1.0))
            if edition_flags is None:
                card.set_edition(polled)
        price_and_mark(gs, card)
        gs["jokers"].append(card)
        # Passive effects (hand_size, discards, joker_slots, ...).  add_to_deck
        # also gives the current round its extra discards (ease_discard,
        # card.lua:589-592); hand count changes (h_plays) take effect on
        # future rounds via round_resets.
        old_hand_size = gs.get("hand_size", 8)
        card.add_to_deck(gs)
        # A bigger hand mid-round fills up to the new size at once.
        if gs.get("hand_size", 8) > old_hand_size and phase == GamePhase.SELECTING_HAND:
            from jackdaw.engine.game import _draw_hand

            _draw_hand(gs)
        return card

    if key.startswith("c_"):
        # SMODS.add_card: create_card, add_to_deck, then the card lands.
        card = create_consumable(key, hands_played=gs.get("hands_played", 0))
        price_and_mark(gs, card)
        card.add_to_deck(gs)
        gs["consumables"].append(card)
        return card

    if len(key) == 3 and key[1] == "_":
        suit_letter, rank_letter = key[0], key[2]
        if suit_letter not in SUIT_LETTER or rank_letter not in RANK_LETTER:
            raise ValueError(f"Invalid playing card key: {key!r}")
        card = create_playing_card(
            suit=SUIT_LETTER[suit_letter],
            rank=RANK_LETTER[rank_letter],
            enhancement=enhancement,
            edition=edition_flags,
            seal=seal,
            inflation=gs.get("inflation", 0),
            discount_percent=gs.get("discount_percent", 0),
        )
        if phase == GamePhase.SELECTING_HAND:
            gs["hand"].append(card)
        else:
            gs["deck"].append(card)
        return card

    raise ValueError(f"Unrecognised card key prefix: {key!r}")


# ``set`` fields and the game-state slot each one writes (set.lua:98-168).
_SET_FIELDS = ("money", "hands", "discards", "ante", "round", "chips")


def set_values(gs: dict[str, Any], **values: Any) -> None:
    """BalatroBot's ``set``: write ``money``, ``hands``, ``discards``, ``ante``,
    ``round`` and ``chips`` directly."""
    for name in values:
        if name not in _SET_FIELDS:
            raise ValueError(f"unknown set field {name!r}")
    if "money" in values:
        gs["dollars"] = values["money"]
    if "hands" in values:
        gs["current_round"]["hands_left"] = values["hands"]
    if "discards" in values:
        gs["current_round"]["discards_left"] = values["discards"]
    if "ante" in values:
        gs["round_resets"]["ante"] = values["ante"]
    if "round" in values:
        gs["round"] = values["round"]
    if "chips" in values:
        gs["chips"] = values["chips"]


def restock_shop(gs: dict[str, Any]) -> None:
    """BalatroBot ``set`` with ``shop=true`` (endpoints/set.lua:170-189).

    Only in SHOP.  It removes G.shop (the shop cards dissolve, freeing their
    keys: Card:remove, card.lua:4741-4748), nils ``current_round.used_packs``
    so the packs are rolled again, and reruns ``G:update_shop()``
    (game.lua:3072-3181): new jokers, the same voucher
    (``current_round.voucher``), new packs and the shop tag hooks, as on
    arrival.
    """
    from jackdaw.engine.game import IllegalActionError, _populate_shop, _release_used_key

    if gs.get("phase") != GamePhase.SHOP:
        raise IllegalActionError("Can re-stock shop only in SHOP state")
    for card in gs.get("shop_cards", []) + gs.get("shop_boosters", []):
        _release_used_key(gs, card)
    gs.setdefault("current_round", {})["used_packs"] = []
    _populate_shop(gs)
