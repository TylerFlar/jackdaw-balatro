#!/usr/bin/env python3
"""Validate the simulator against a live Balatro game via balatrobot.

Usage::

    # With balatrobot serve already running:
    uv run python scripts/validate_run.py --seed VALIDATE --back b_red --stake 1

Requires balatrobot to be serving on localhost:12346.
Start it with: uvx balatrobot serve --fast --no-audio --love-path <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx

sys.path.insert(0, ".")

from jackdaw.engine.run_init import initialize_run


BOT_URL = "http://127.0.0.1:12346"

# Mapping from balatrobot tag names to our tag keys
TAG_NAME_TO_KEY = {
    "Economy Tag": "tag_economy",
    "Investment Tag": "tag_investment",
    "Double Tag": "tag_double",
    "Skip Tag": "tag_skip",
    "Garbage Tag": "tag_garbage",
    "Handy Tag": "tag_handy",
    "Orbital Tag": "tag_orbital",
    "Top-up Tag": "tag_top_up",
    "Boss Tag": "tag_boss",
    "Buffoon Tag": "tag_buffoon",
    "Charm Tag": "tag_charm",
    "Ethereal Tag": "tag_ethereal",
    "Meteor Tag": "tag_meteor",
    "Standard Tag": "tag_standard",
    "Juggle Tag": "tag_juggle",
    "D6 Tag": "tag_d_six",
    "Uncommon Tag": "tag_uncommon",
    "Rare Tag": "tag_rare",
    "Foil Tag": "tag_foil",
    "Holographic Tag": "tag_holo",
    "Polychrome Tag": "tag_polychrome",
    "Negative Tag": "tag_negative",
    "Voucher Tag": "tag_voucher",
    "Coupon Tag": "tag_coupon",
}

# Mapping from balatrobot boss names to our blind keys
BOSS_NAME_TO_KEY = {
    "The Hook": "bl_hook",
    "The Club": "bl_club",
    "The Goad": "bl_goad",
    "The Head": "bl_head",
    "The Window": "bl_window",
    "The Wall": "bl_wall",
    "The Wheel": "bl_wheel",
    "The Arm": "bl_arm",
    "The Eye": "bl_eye",
    "The Mouth": "bl_mouth",
    "The Plant": "bl_plant",
    "The Serpent": "bl_serpent",
    "The Pillar": "bl_pillar",
    "The Needle": "bl_needle",
    "The Water": "bl_water",
    "The Manacle": "bl_manacle",
    "The Tooth": "bl_tooth",
    "The Fish": "bl_fish",
    "The Mark": "bl_mark",
    "The Flint": "bl_flint",
    "Amber Acorn": "bl_final_acorn",
    "Verdant Leaf": "bl_final_leaf",
    "Violet Vessel": "bl_final_vessel",
    "Crimson Heart": "bl_final_heart",
    "Cerulean Bell": "bl_final_bell",
    "The Ox": "bl_ox",
    "The House": "bl_house",
    "Psychic": "bl_psychic",
}

DECK_MAP = {
    "b_red": "RED",
    "b_blue": "BLUE",
    "b_yellow": "YELLOW",
    "b_green": "GREEN",
    "b_black": "BLACK",
    "b_magic": "MAGIC",
    "b_nebula": "NEBULA",
    "b_ghost": "GHOST",
    "b_abandoned": "ABANDONED",
    "b_checkered": "CHECKERED",
    "b_zodiac": "ZODIAC",
    "b_painted": "PAINTED",
    "b_anaglyph": "ANAGLYPH",
    "b_plasma": "PLASMA",
    "b_erratic": "ERRATIC",
}

STAKE_MAP = {1: "WHITE", 2: "RED", 3: "GREEN", 4: "BLACK", 5: "BLUE", 6: "PURPLE", 7: "ORANGE", 8: "GOLD"}


def rpc(method: str, params: dict | None = None) -> dict:
    """Send a JSON-RPC request to balatrobot."""
    payload = {"jsonrpc": "2.0", "method": method, "id": 1}
    if params:
        payload["params"] = params
    resp = httpx.post(BOT_URL, json=payload, timeout=10.0)
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")
    return data.get("result", {})


def compare_init(seed: str, back_key: str, stake: int) -> list[str]:
    """Start a run in both simulator and live, compare initial state."""
    diffs: list[str] = []

    # Start live run
    deck_name = DECK_MAP.get(back_key, "RED")
    stake_name = STAKE_MAP.get(stake, "WHITE")
    live = rpc("start", {"deck": deck_name, "stake": stake_name, "seed": seed})

    # Wait for state to settle
    time.sleep(0.5)
    live = rpc("gamestate")

    # Start simulator
    sim = initialize_run(back_key, stake, seed)

    print(f"=== Comparing seed={seed!r} back={back_key} stake={stake} ===")
    print()

    # --- Money ---
    sim_money = sim["dollars"]
    live_money = live.get("money", 0)
    status = "MATCH" if sim_money == live_money else "DIFF"
    print(f"  money:      sim={sim_money:>4}  live={live_money:>4}  [{status}]")
    if sim_money != live_money:
        diffs.append(f"money: sim={sim_money} live={live_money}")

    # --- Ante ---
    sim_ante = sim["round_resets"]["ante"]
    live_ante = live.get("ante_num", 1)
    status = "MATCH" if sim_ante == live_ante else "DIFF"
    print(f"  ante:       sim={sim_ante:>4}  live={live_ante:>4}  [{status}]")
    if sim_ante != live_ante:
        diffs.append(f"ante: sim={sim_ante} live={live_ante}")

    # --- Deck size ---
    sim_deck = len(sim["deck"])
    live_deck = live.get("cards", {}).get("count", 0)
    status = "MATCH" if sim_deck == live_deck else "DIFF"
    print(f"  deck_size:  sim={sim_deck:>4}  live={live_deck:>4}  [{status}]")
    if sim_deck != live_deck:
        diffs.append(f"deck_size: sim={sim_deck} live={live_deck}")

    # --- Boss ---
    sim_boss = sim["round_resets"]["blind_choices"]["Boss"]
    live_boss_name = live.get("blinds", {}).get("boss", {}).get("name", "")
    live_boss_key = BOSS_NAME_TO_KEY.get(live_boss_name, f"?{live_boss_name}")
    status = "MATCH" if sim_boss == live_boss_key else "DIFF"
    print(f"  boss:       sim={sim_boss:<16}  live={live_boss_key:<16}  [{status}]")
    if sim_boss != live_boss_key:
        diffs.append(f"boss: sim={sim_boss} live={live_boss_key} ({live_boss_name})")

    # --- Tags ---
    rr = sim["round_resets"]
    sim_small_tag = rr.get("blind_tags", {}).get("Small", "?")
    sim_big_tag = rr.get("blind_tags", {}).get("Big", "?")
    live_blinds = live.get("blinds", {})
    live_small_name = live_blinds.get("small", {}).get("tag_name", "")
    live_big_name = live_blinds.get("big", {}).get("tag_name", "")
    live_small_tag = TAG_NAME_TO_KEY.get(live_small_name, f"?{live_small_name}")
    live_big_tag = TAG_NAME_TO_KEY.get(live_big_name, f"?{live_big_name}")

    status = "MATCH" if sim_small_tag == live_small_tag else "DIFF"
    print(f"  small_tag:  sim={sim_small_tag:<20}  live={live_small_tag:<20}  [{status}]")
    if sim_small_tag != live_small_tag:
        diffs.append(f"small_tag: sim={sim_small_tag} live={live_small_tag}")

    status = "MATCH" if sim_big_tag == live_big_tag else "DIFF"
    print(f"  big_tag:    sim={sim_big_tag:<20}  live={live_big_tag:<20}  [{status}]")
    if sim_big_tag != live_big_tag:
        diffs.append(f"big_tag: sim={sim_big_tag} live={live_big_tag}")

    # --- Voucher ---
    sim_voucher = sim["current_round"].get("voucher", "?")
    # Live doesn't expose voucher directly at blind select; skip for now

    # --- Blind scores ---
    live_small_score = live_blinds.get("small", {}).get("score", 0)
    live_big_score = live_blinds.get("big", {}).get("score", 0)
    live_boss_score = live_blinds.get("boss", {}).get("score", 0)
    print(f"  blind targets: small={live_small_score} big={live_big_score} boss={live_boss_score}")

    # --- Deck order (first 10) ---
    print()
    print("  Deck order (first 10):")
    sim_deck_keys = [c.card_key for c in sim["deck"][:10]]
    live_cards = live.get("cards", {}).get("cards", [])
    live_deck_keys = [c["key"] for c in live_cards[:10]]
    for i in range(min(10, max(len(sim_deck_keys), len(live_deck_keys)))):
        sk = sim_deck_keys[i] if i < len(sim_deck_keys) else "?"
        lk = live_deck_keys[i] if i < len(live_deck_keys) else "?"
        status = "MATCH" if sk == lk else "DIFF"
        print(f"    [{i}] sim={sk:<6}  live={lk:<6}  [{status}]")
    deck_match = sim_deck_keys == live_deck_keys
    if not deck_match:
        diffs.append("deck_order: first 10 cards differ")

    print()
    print(f"  Total discrepancies: {len(diffs)}")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simulator vs balatrobot")
    parser.add_argument("--seed", default="VALIDATE", help="Run seed")
    parser.add_argument("--back", default="b_red", help="Deck back key")
    parser.add_argument("--stake", type=int, default=1, help="Stake level")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to test")
    args = parser.parse_args()

    # Check health
    try:
        rpc("health")
    except Exception as e:
        print(f"ERROR: Cannot reach balatrobot at {BOT_URL}: {e}")
        print("Start it with: uvx balatrobot serve --fast --no-audio --love-path <path>")
        sys.exit(1)

    all_diffs: list[list[str]] = []

    # Always start from menu
    try:
        rpc("menu")
        time.sleep(1.5)
    except Exception:
        pass

    for i in range(args.seeds):
        seed = f"{args.seed}{i}" if args.seeds > 1 else args.seed

        # Return to menu between runs
        if i > 0:
            try:
                rpc("menu")
                time.sleep(1.5)
            except Exception:
                pass

        diffs = compare_init(seed, args.back, args.stake)
        all_diffs.append(diffs)
        print()

    # Summary
    total_seeds = len(all_diffs)
    clean = sum(1 for d in all_diffs if not d)
    total_diffs = sum(len(d) for d in all_diffs)
    print("=" * 60)
    print(f"Seeds tested: {total_seeds}")
    print(f"Clean matches: {clean}/{total_seeds}")
    print(f"Total discrepancies: {total_diffs}")

    # Categorize discrepancies
    cats: dict[str, int] = {}
    for diffs in all_diffs:
        for d in diffs:
            cat = d.split(":")[0]
            cats[cat] = cats.get(cat, 0) + 1
    if cats:
        print("\nDiscrepancy breakdown:")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}/{total_seeds}")


if __name__ == "__main__":
    main()
