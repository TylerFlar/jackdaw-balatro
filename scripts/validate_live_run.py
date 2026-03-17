#!/usr/bin/env python3
"""Play the same action sequence in both simulator and live Balatro.

Compares state at every decision point. Requires balatrobot on :12346.

Usage::

    uv run python scripts/validate_live_run.py --seed LIVETEST --steps 20
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx

sys.path.insert(0, ".")

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
    SkipBlind,
)
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run

BOT_URL = "http://127.0.0.1:12346"

DECK_MAP = {"b_red": "RED", "b_blue": "BLUE", "b_black": "BLACK"}
STAKE_MAP = {1: "WHITE", 2: "RED"}


def rpc(method: str, params: dict | None = None) -> dict:
    payload: dict = {"jsonrpc": "2.0", "method": method, "id": 1}
    if params:
        payload["params"] = params
    resp = httpx.post(BOT_URL, json=payload, timeout=10.0)
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC {method}: {data['error']}")
    return data.get("result", {})


def get_live_state() -> dict:
    return rpc("gamestate")


def compare(sim: dict, live: dict, label: str) -> list[str]:
    diffs: list[str] = []

    def _cmp(name: str, s: Any, l: Any) -> None:
        if s != l:
            diffs.append(f"{name}: sim={s} live={l}")

    _cmp("money", sim.get("dollars", 0), live.get("money", 0))
    _cmp("ante", sim["round_resets"]["ante"], live.get("ante_num", 1))

    cr = sim.get("current_round", {})
    lr = live.get("round", {})
    _cmp("chips", sim.get("chips", 0), lr.get("chips", 0))
    _cmp("hands_left", cr.get("hands_left", 0), lr.get("hands_left", 0))
    _cmp("discards_left", cr.get("discards_left", 0), lr.get("discards_left", 0))

    sim_hand = [c.card_key for c in sim.get("hand", []) if hasattr(c, "card_key") and c.card_key]
    live_hand = [c["key"] for c in live.get("hand", {}).get("cards", [])]
    _cmp("hand_count", len(sim_hand), len(live_hand))
    if sim_hand and live_hand and sim_hand != live_hand:
        diffs.append(f"hand_cards: sim={sim_hand[:5]} live={live_hand[:5]}")

    _cmp("deck_size", len(sim.get("deck", [])), live.get("cards", {}).get("count", 0))

    if diffs:
        print(f"  [{label}] {len(diffs)} diffs:")
        for d in diffs:
            print(f"    {d}")
    else:
        print(f"  [{label}] OK")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="LIVETEST")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    seed = args.seed

    # Check health
    try:
        rpc("health")
    except Exception as e:
        print(f"Cannot reach balatrobot: {e}")
        sys.exit(1)

    # Start both
    rpc("menu")
    time.sleep(1.5)
    rpc("start", {"deck": "RED", "stake": "WHITE", "seed": seed})
    time.sleep(0.5)

    sim = initialize_run("b_red", 1, seed)
    sim["phase"] = GamePhase.BLIND_SELECT
    sim["blind_on_deck"] = "Small"
    sim["jokers"] = []
    sim["consumables"] = []

    all_diffs: list[list[str]] = []

    # Compare initial state
    live = get_live_state()
    all_diffs.append(compare(sim, live, "init"))

    # Scripted action sequence
    actions = [
        ("select", SelectBlind(), {"sim_pre": lambda gs: None}),
    ]

    # Select Small Blind
    print("\n--- Select Small Blind ---")
    step(sim, SelectBlind())
    rpc("select")
    time.sleep(0.3)
    live = get_live_state()
    all_diffs.append(compare(sim, live, "after_select_small"))

    # Play first 5 cards
    print("\n--- Play hand (0-4) ---")
    hand_cards = sim.get("hand", [])
    if len(hand_cards) >= 5:
        step(sim, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        rpc("play", {"cards": [0, 1, 2, 3, 4]})
        time.sleep(0.5)
        live = get_live_state()
        all_diffs.append(compare(sim, live, "after_play"))

    # Check if we beat the blind
    if sim.get("phase") == GamePhase.ROUND_EVAL:
        print("\n--- Cash out ---")
        step(sim, CashOut())
        rpc("cash_out")
        time.sleep(0.3)
        live = get_live_state()
        all_diffs.append(compare(sim, live, "after_cashout"))

        # Compare shop
        if sim.get("phase") == GamePhase.SHOP:
            shop_cards_sim = len(sim.get("shop_cards", []))
            shop_cards_live = len(live.get("shop", {}).get("cards", []))
            print(f"  Shop cards: sim={shop_cards_sim} live={shop_cards_live}")

            print("\n--- Next round ---")
            step(sim, NextRound())
            rpc("next_round")
            time.sleep(0.3)
            live = get_live_state()
            all_diffs.append(compare(sim, live, "after_next_round"))

            # Select Big Blind
            if sim.get("phase") == GamePhase.BLIND_SELECT:
                print("\n--- Select Big Blind ---")
                step(sim, SelectBlind())
                rpc("select")
                time.sleep(0.3)
                live = get_live_state()
                all_diffs.append(compare(sim, live, "after_select_big"))
    elif sim.get("phase") == GamePhase.SELECTING_HAND:
        print("  (blind not beaten, still playing)")
    elif sim.get("phase") == GamePhase.GAME_OVER:
        print("  (game over)")

    # Summary
    print("\n" + "=" * 60)
    total_steps = len(all_diffs)
    clean = sum(1 for d in all_diffs if not d)
    total_diffs = sum(len(d) for d in all_diffs)
    print(f"Steps compared: {total_steps}")
    print(f"Clean: {clean}/{total_steps}")
    print(f"Total discrepancies: {total_diffs}")


if __name__ == "__main__":
    main()
