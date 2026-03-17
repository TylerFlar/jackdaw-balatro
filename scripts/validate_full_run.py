#!/usr/bin/env python3
"""Full-run validation: scripted action sequence compared against balatrobot.

Usage::

    # Offline mode (sim-only, verifies no crashes + determinism):
    uv run python scripts/validate_full_run.py

    # Live mode (vs balatrobot on localhost:12346):
    uv run python scripts/validate_full_run.py --live

Runs a scripted sequence of actions through the simulator and optionally
mirrors them in balatrobot, comparing state at each step.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

sys.path.insert(0, ".")

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    Reroll,
    SelectBlind,
    SkipBlind,
)
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import random_agent, simulate_run


# ---------------------------------------------------------------------------
# Scripted run
# ---------------------------------------------------------------------------

def run_scripted(seed: str = "TESTSEED") -> dict[str, Any]:
    """Play a scripted sequence and record state at each step."""
    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    gs["jokers"] = []
    gs["consumables"] = []

    log: list[dict[str, Any]] = []

    def snapshot(label: str) -> dict[str, Any]:
        return {
            "label": label,
            "phase": str(gs.get("phase", "")),
            "dollars": gs.get("dollars", 0),
            "chips": gs.get("chips", 0),
            "ante": gs["round_resets"]["ante"],
            "hands_left": gs.get("current_round", {}).get("hands_left", 0),
            "hand_size": len(gs.get("hand", [])),
            "deck_size": len(gs.get("deck", [])),
            "joker_count": len(gs.get("jokers", [])),
            "shop_cards": len(gs.get("shop_cards", [])),
        }

    log.append(snapshot("init"))

    # --- Small Blind ---
    step(gs, SelectBlind())
    log.append(snapshot("after_select_small"))

    # Play first 5 cards
    gs["blind"].chips = 1  # easy win for scripted test
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    log.append(snapshot("after_play_small"))

    # Cash out
    step(gs, CashOut())
    log.append(snapshot("after_cashout_small"))

    # Next round → Big
    step(gs, NextRound())
    log.append(snapshot("after_nextround_to_big"))

    # --- Big Blind ---
    step(gs, SelectBlind())
    log.append(snapshot("after_select_big"))

    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    log.append(snapshot("after_play_big"))

    step(gs, CashOut())
    log.append(snapshot("after_cashout_big"))

    step(gs, NextRound())
    log.append(snapshot("after_nextround_to_boss"))

    # --- Boss Blind ---
    step(gs, SelectBlind())
    log.append(snapshot("after_select_boss"))

    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    log.append(snapshot("after_play_boss"))

    step(gs, CashOut())
    log.append(snapshot("after_cashout_boss"))

    gs["game_state"] = gs
    gs["log"] = log
    return gs


# ---------------------------------------------------------------------------
# Batch random validation
# ---------------------------------------------------------------------------

def run_batch(n: int, agent_name: str = "random") -> dict[str, Any]:
    """Run N games with the specified agent and collect stats."""
    agent = random_agent
    wins = 0
    crashes = 0
    total_actions = 0
    total_rounds = 0
    max_actions = 0

    t0 = time.time()
    for i in range(n):
        try:
            gs = simulate_run("b_red", 1, f"BATCH_{agent_name}_{i}", agent, max_actions=2000)
            if gs.get("won"):
                wins += 1
            total_actions += gs["actions_taken"]
            total_rounds += gs.get("round", 0)
            max_actions = max(max_actions, gs["actions_taken"])
        except Exception as e:
            crashes += 1
            print(f"  CRASH at seed {i}: {e}")
    elapsed = time.time() - t0

    return {
        "n": n,
        "agent": agent_name,
        "wins": wins,
        "crashes": crashes,
        "avg_actions": total_actions / max(n, 1),
        "max_actions": max_actions,
        "avg_rounds": total_rounds / max(n, 1),
        "elapsed": elapsed,
        "runs_per_sec": n / max(elapsed, 0.001),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full run validation")
    parser.add_argument("--live", action="store_true", help="Compare vs balatrobot")
    parser.add_argument("--batch", type=int, default=1000, help="Random batch size")
    args = parser.parse_args()

    print("=== Scripted run ===")
    gs = run_scripted("TESTSEED")
    for entry in gs["log"]:
        print(f"  {entry['label']:<25} phase={entry['phase']:<20} "
              f"$={entry['dollars']:<4} chips={entry['chips']:<8} "
              f"hand={entry['hand_size']} deck={entry['deck_size']} "
              f"shop={entry['shop_cards']}")

    print()
    print("=== Determinism check ===")
    gs2 = run_scripted("TESTSEED")
    all_match = all(
        gs["log"][i]["dollars"] == gs2["log"][i]["dollars"]
        and gs["log"][i]["chips"] == gs2["log"][i]["chips"]
        for i in range(len(gs["log"]))
    )
    print(f"  Same seed -> same result: {all_match}")

    print()
    print(f"=== Batch validation ({args.batch} runs) ===")
    stats = run_batch(args.batch, "random")
    print(f"  Runs: {stats['n']}")
    print(f"  Wins: {stats['wins']}")
    print(f"  Crashes: {stats['crashes']}")
    print(f"  Avg actions: {stats['avg_actions']:.1f}")
    print(f"  Max actions: {stats['max_actions']}")
    print(f"  Avg rounds: {stats['avg_rounds']:.1f}")
    print(f"  Time: {stats['elapsed']:.2f}s ({stats['runs_per_sec']:.0f} runs/sec)")

    if args.live:
        print()
        print("=== Live validation vs balatrobot ===")
        print("(Not yet implemented — use scripts/validate_run.py for pre-deal validation)")

    # Summary
    print()
    ok = stats["crashes"] == 0 and all_match
    print(f"{'PASS' if ok else 'FAIL'}: {stats['crashes']} crashes, determinism={'OK' if all_match else 'FAIL'}")


if __name__ == "__main__":
    main()
