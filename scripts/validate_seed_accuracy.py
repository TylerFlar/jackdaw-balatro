#!/usr/bin/env python3
"""Definitive seed-accuracy validation: play the same actions in sim + live.

Drives both the simulator and balatrobot through identical action sequences,
comparing state at every decision point.

Usage::

    uv run python scripts/validate_seed_accuracy.py --seed TESTSEED
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
)
from jackdaw.bridge.balatrobot_adapter import action_to_rpc
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run

BOT_URL = "http://127.0.0.1:12346"


def rpc(method: str, params: dict | None = None) -> dict:
    payload: dict = {"jsonrpc": "2.0", "method": method, "id": 1}
    if params:
        payload["params"] = params
    resp = httpx.post(BOT_URL, json=payload, timeout=10.0)
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC {method}: {data['error']}")
    return data.get("result", {})


def bot_hand_keys(bot: dict) -> list[str]:
    return [c["key"] for c in bot.get("hand", {}).get("cards", [])]


def bot_deck_keys(bot: dict) -> list[str]:
    return [c["key"] for c in bot.get("cards", {}).get("cards", [])]


def sim_hand_keys(gs: dict) -> list[str]:
    return [c.card_key for c in gs.get("hand", []) if hasattr(c, "card_key") and c.card_key]


def sim_deck_keys(gs: dict) -> list[str]:
    return [c.card_key for c in gs.get("deck", []) if hasattr(c, "card_key") and c.card_key]


def compare_states(sim: dict, bot: dict, label: str) -> list[str]:
    diffs: list[str] = []

    def cmp(name: str, s: Any, l: Any) -> None:
        if s != l:
            diffs.append(f"{name}: sim={s} live={l}")

    cmp("money", sim.get("dollars", 0), bot.get("money", 0))
    cmp("ante", sim["round_resets"]["ante"], bot.get("ante_num", 1))

    cr = sim.get("current_round", {})
    br = bot.get("round", {})
    cmp("chips", sim.get("chips", 0), br.get("chips", 0))
    # Only compare hands/discards when in a round (not at init/blind_select)
    sim_phase = sim.get("phase")
    if sim_phase not in (GamePhase.BLIND_SELECT, None):
        cmp("hands_left", cr.get("hands_left", 0), br.get("hands_left", 0))
        cmp("discards_left", cr.get("discards_left", 0), br.get("discards_left", 0))

    # Hand cards (as sets — display order may differ)
    sh = set(sim_hand_keys(sim))
    lh = set(bot_hand_keys(bot))
    cmp("hand_cards", sh, lh)

    # Deck size
    cmp("deck_size", len(sim.get("deck", [])), bot.get("cards", {}).get("count", 0))

    status = "OK" if not diffs else f"{len(diffs)} diffs"
    print(f"  [{label}] {status}")
    for d in diffs:
        print(f"    {d}")
    return diffs


def run_validation(seed: str) -> dict[str, Any]:
    """Play through Small→Big→Boss with the same actions in both."""
    print(f"\n{'='*60}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

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
    step_count = 0

    # Compare initial state
    bot = rpc("gamestate")
    all_diffs.append(compare_states(sim, bot, "init"))

    blinds = ["Small", "Big", "Boss"]
    for blind_name in blinds:
        # Select blind
        print(f"\n--- Select {blind_name} Blind ---")
        step(sim, SelectBlind())
        rpc("select")
        time.sleep(0.3)
        bot = rpc("gamestate")
        all_diffs.append(compare_states(sim, bot, f"after_select_{blind_name.lower()}"))
        step_count += 1

        # Play hands until blind beaten or game over
        hand_num = 0
        while sim.get("phase") == GamePhase.SELECTING_HAND:
            hand = sim.get("hand", [])
            n = min(5, len(hand))
            if n == 0:
                break

            # Get balatrobot's hand order and play the same cards
            bot_now = rpc("gamestate")
            bot_hand = bot_hand_keys(bot_now)
            bot_play_keys = bot_hand[:min(5, len(bot_hand))]

            # Find those same card keys in our sim hand
            sim_indices = []
            used = set()
            for bk in bot_play_keys:
                for i, c in enumerate(hand):
                    if i not in used and hasattr(c, "card_key") and c.card_key == bk:
                        sim_indices.append(i)
                        used.add(i)
                        break

            if len(sim_indices) != len(bot_play_keys):
                # Fallback: play first 5
                sim_indices = list(range(n))
                print(f"  Play hand {hand_num}: FALLBACK (card mismatch)")
            else:
                print(f"  Play hand {hand_num}: {bot_play_keys}")

            bot_indices = list(range(len(bot_play_keys)))
            step(sim, PlayHand(card_indices=tuple(sim_indices)))
            rpc("play", {"cards": bot_indices})
            time.sleep(0.3)
            bot = rpc("gamestate")
            all_diffs.append(compare_states(sim, bot, f"after_play_{blind_name.lower()}_{hand_num}"))
            step_count += 1
            hand_num += 1

            if sim.get("phase") == GamePhase.GAME_OVER:
                print("  GAME OVER")
                return _summary(seed, all_diffs, step_count, "game_over")

        if sim.get("phase") != GamePhase.ROUND_EVAL:
            print(f"  Unexpected phase: {sim.get('phase')}")
            return _summary(seed, all_diffs, step_count, "unexpected_phase")

        # Cash out
        print(f"  Cash out")
        step(sim, CashOut())
        rpc("cash_out")
        time.sleep(0.3)
        bot = rpc("gamestate")
        all_diffs.append(compare_states(sim, bot, f"after_cashout_{blind_name.lower()}"))
        step_count += 1

        # Next round (except after Boss)
        if blind_name != "Boss":
            print(f"  Next round")
            step(sim, NextRound())
            rpc("next_round")
            time.sleep(0.3)
            bot = rpc("gamestate")
            all_diffs.append(compare_states(sim, bot, f"after_nextround_{blind_name.lower()}"))
            step_count += 1

    return _summary(seed, all_diffs, step_count, "complete")


def _summary(seed: str, all_diffs: list, step_count: int, status: str) -> dict:
    total_diffs = sum(len(d) for d in all_diffs)
    clean = sum(1 for d in all_diffs if not d)
    return {
        "seed": seed,
        "steps": step_count,
        "total_diffs": total_diffs,
        "clean_steps": clean,
        "total_steps": len(all_diffs),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="TESTSEED")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    try:
        rpc("health")
    except Exception as e:
        print(f"Cannot reach balatrobot: {e}")
        sys.exit(1)

    results = []
    for i in range(args.seeds):
        seed = f"{args.seed}{i}" if args.seeds > 1 else args.seed
        result = run_validation(seed)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["total_diffs"] == 0 else "FAIL"
        print(f"  {r['seed']}: {status} ({r['clean_steps']}/{r['total_steps']} clean, "
              f"{r['total_diffs']} diffs, {r['status']})")

    total_clean = sum(r["total_diffs"] == 0 for r in results)
    print(f"\nResult: {total_clean}/{len(results)} seeds fully clean")


if __name__ == "__main__":
    main()
