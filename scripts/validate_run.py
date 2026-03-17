#!/usr/bin/env python3
"""Validate the simulator against a live Balatro game via balatrobot.

Usage::

    python scripts/validate_run.py --seed TUTORIAL --back b_red --stake 1

This script:
1. Initializes the simulator with the given parameters
2. Reads live state from balatrobot at each decision point
3. Compares simulator state to live state
4. Logs discrepancies
5. Prints a summary report

Requires balatrobot to be running (https://coder.github.io/balatrobot/).
The live state is read via the balatrobot HTTP API.

NOTE: This is a placeholder — actual balatrobot integration requires
the bridge module (jackdaw/bridge/) to be connected.  For now, this
script demonstrates the validation flow with mock data.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, ".")

from jackdaw.engine.actions import GamePhase, SelectBlind
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.validator import format_report, validate_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simulator against live Balatro")
    parser.add_argument("--seed", default="TUTORIAL", help="Run seed")
    parser.add_argument("--back", default="b_red", help="Deck back key")
    parser.add_argument("--stake", type=int, default=1, help="Stake level")
    parser.add_argument("--max-steps", type=int, default=100, help="Max validation steps")
    parser.add_argument("--mock", action="store_true", help="Use mock live data (for testing)")
    return parser.parse_args()


def mock_live_state(sim_state: dict[str, Any]) -> dict[str, Any]:
    """Generate mock live state from simulator state (for testing the flow)."""
    return {
        "money": sim_state.get("dollars", 0),
        "ante": sim_state.get("round_resets", {}).get("ante", 1),
        "round": {
            "chips": sim_state.get("chips", 0),
            "hands_left": sim_state.get("current_round", {}).get("hands_left", 0),
            "discards_left": sim_state.get("current_round", {}).get("discards_left", 0),
        },
        "hand": [
            {"suit": c.base.suit.value, "rank": c.base.rank.value}
            for c in sim_state.get("hand", [])
            if c.base is not None
        ],
        "jokers": [
            {"key": j.center_key}
            for j in sim_state.get("jokers", [])
        ],
        "deck_size": len(sim_state.get("deck", [])),
        "blind": {
            "chips": getattr(sim_state.get("blind"), "chips", 0)
            if sim_state.get("blind")
            else 0,
        },
        "phase": str(sim_state.get("phase", "")),
    }


def main() -> None:
    args = parse_args()
    print(f"Validating: seed={args.seed!r} back={args.back} stake={args.stake}")

    gs = initialize_run(args.back, args.stake, args.seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"

    step_diffs: list[list[str]] = []

    for step_num in range(args.max_steps):
        phase = gs.get("phase")
        if phase == GamePhase.GAME_OVER:
            break

        # Get live state
        if args.mock:
            live = mock_live_state(gs)
        else:
            print("ERROR: Live balatrobot integration not yet implemented.")
            print("Use --mock for testing the validation flow.")
            sys.exit(1)

        # Compare
        diffs = validate_step(gs, live)
        step_diffs.append(diffs)

        if diffs:
            print(f"  Step {step_num}: {len(diffs)} discrepancies")
            for d in diffs:
                print(f"    {d}")

        # Advance (using greedy agent for mock mode)
        if phase == GamePhase.BLIND_SELECT:
            step(gs, SelectBlind())
        else:
            break

    report = format_report(step_diffs, args.seed)
    print()
    print(report)


if __name__ == "__main__":
    main()
