#!/usr/bin/env python3
"""Comprehensive evaluation harness for trained Balatro agents.

Usage:
    # Full evaluation across decks and stakes
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt

    # Quick check with fewer episodes
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 20

    # Specific decks only, skip baselines
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --decks b_red,b_blue --skip-baselines
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from jackdaw.env.agents import (
    EvalResult,
    HeuristicAgent,
    RandomAgent,
    evaluate_agent,
)
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.observation import encode_observation
from jackdaw.env.policy.policy import BalatroPolicy, PolicyInput, collate_policy_inputs
from jackdaw.env.training.ppo import ALL_DECKS, _compute_shop_splits

# ---------------------------------------------------------------------------
# Deck / stake display names
# ---------------------------------------------------------------------------

DECK_NAMES = {
    "b_red": "Red",
    "b_blue": "Blue",
    "b_yellow": "Yellow",
    "b_green": "Green",
    "b_black": "Black",
    "b_magic": "Magic",
    "b_nebula": "Nebula",
    "b_ghost": "Ghost",
    "b_abandoned": "Abandoned",
    "b_checkered": "Checkered",
    "b_zodiac": "Zodiac",
    "b_painted": "Painted",
    "b_anaglyph": "Anaglyph",
    "b_plasma": "Plasma",
    "b_erratic": "Erratic",
}

STAKE_NAMES = {
    1: "White",
    2: "Red",
    3: "Green",
    4: "Black",
    5: "Blue",
    6: "Purple",
    7: "Orange",
    8: "Gold",
}

# Default subset of decks for the sweep (the most commonly played)
DEFAULT_DECKS = ["b_red", "b_blue", "b_yellow", "b_green", "b_black"]


# ---------------------------------------------------------------------------
# Policy agent wrapper (mirrors ppo._PolicyAgent)
# ---------------------------------------------------------------------------


class PolicyAgent:
    """Wraps a BalatroPolicy as an Agent for evaluate_agent."""

    def __init__(self, policy: BalatroPolicy, device: torch.device) -> None:
        self._policy = policy
        self._device = device

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: Any, info: dict) -> Any:
        from jackdaw.env.action_space import ActionMask

        gs = info["raw_state"]
        encoded_obs = encode_observation(gs)
        policy_input = PolicyInput(
            obs=encoded_obs,
            action_mask=action_mask,
            shop_splits=_compute_shop_splits(gs),
        )
        batch = collate_policy_inputs([policy_input], balatro_game_spec(), device=self._device)

        self._policy.eval()
        with torch.no_grad():
            actions, _, _ = self._policy.sample_action(batch)
        return actions[0]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    device: str = "auto",
) -> tuple[BalatroPolicy, torch.device, dict[str, Any]]:
    """Load a trained policy from a checkpoint.

    Returns (policy, device, checkpoint_dict).
    """
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = checkpoint["config"]

    policy = BalatroPolicy(
        game_spec=balatro_game_spec(),
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
    ).to(dev)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    return policy, dev, checkpoint


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


def _eval_summary(result: EvalResult) -> dict[str, Any]:
    """Extract summary stats from an EvalResult."""
    antes = [e.ante_reached for e in result.episodes]
    return {
        "n_episodes": result.n_episodes,
        "avg_ante": result.avg_ante,
        "max_ante": max(antes) if antes else 0,
        "win_rate": result.win_rate,
        "avg_actions": result.avg_actions,
        "ante_distribution": dict(Counter(antes)),
    }


def run_deck_sweep(
    agent: Any,
    decks: list[str],
    stake: int,
    n_episodes: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate agent across multiple decks."""
    results: dict[str, dict[str, Any]] = {}
    for deck in decks:
        name = DECK_NAMES.get(deck, deck)
        print(f"  {name} Deck ({n_episodes} episodes)...", end="", flush=True)
        t0 = time.time()
        result = evaluate_agent(
            agent, n_episodes=n_episodes, back_key=deck, stake=stake,
        )
        elapsed = time.time() - t0
        summary = _eval_summary(result)
        results[deck] = summary
        print(
            f" avg_ante={summary['avg_ante']:.2f}, "
            f"win={summary['win_rate']:.1%} ({elapsed:.1f}s)"
        )
    return results


def run_stake_escalation(
    agent: Any,
    deck: str,
    stakes: list[int],
    n_episodes: int,
    min_avg_ante: float = 1.5,
) -> dict[str, dict[str, Any]]:
    """Evaluate agent at increasing stakes, stopping if too weak."""
    results: dict[str, dict[str, Any]] = {}
    for stake in stakes:
        name = STAKE_NAMES.get(stake, f"Stake {stake}")
        print(f"  {name} Stake ({n_episodes} episodes)...", end="", flush=True)
        t0 = time.time()
        result = evaluate_agent(
            agent, n_episodes=n_episodes, back_key=deck, stake=stake,
        )
        elapsed = time.time() - t0
        summary = _eval_summary(result)
        results[str(stake)] = summary
        print(
            f" avg_ante={summary['avg_ante']:.2f}, "
            f"win={summary['win_rate']:.1%} ({elapsed:.1f}s)"
        )

        # Skip higher stakes if agent can't survive
        if summary["avg_ante"] < min_avg_ante and stake < max(stakes):
            print(f"  (skipping higher stakes — avg_ante < {min_avg_ante})")
            break
    return results


def run_baselines(
    deck: str,
    stake: int,
    n_episodes: int,
) -> dict[str, dict[str, Any]]:
    """Run baseline agents for comparison."""
    results: dict[str, dict[str, Any]] = {}

    for name, agent in [("HeuristicAgent", HeuristicAgent()), ("RandomAgent", RandomAgent())]:
        print(f"  {name} ({n_episodes} episodes)...", end="", flush=True)
        t0 = time.time()
        result = evaluate_agent(
            agent, n_episodes=n_episodes, back_key=deck, stake=stake,
        )
        elapsed = time.time() - t0
        summary = _eval_summary(result)
        results[name] = summary
        print(
            f" avg_ante={summary['avg_ante']:.2f}, "
            f"win={summary['win_rate']:.1%} ({elapsed:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


def _table_row(cells: list[str], widths: list[int]) -> str:
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(f" {cell:>{w}s} ")
    return "|" + "|".join(parts) + "|"


def _table_sep(widths: list[int]) -> str:
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def print_deck_table(deck_results: dict[str, dict[str, Any]]) -> None:
    headers = ["Deck", "Avg Ante", "Max Ante", "Win Rate", "Avg Acts"]
    widths = [13, 8, 8, 8, 8]

    print(_table_sep(widths))
    print(_table_row(headers, widths))
    print(_table_sep(widths))

    for deck_key, stats in deck_results.items():
        name = DECK_NAMES.get(deck_key, deck_key)
        row = [
            name,
            f"{stats['avg_ante']:.2f}",
            str(stats["max_ante"]),
            f"{stats['win_rate']:.1%}",
            f"{stats['avg_actions']:.0f}",
        ]
        print(_table_row(row, widths))

    print(_table_sep(widths))


def print_baseline_table(
    trained_stats: dict[str, Any],
    baseline_results: dict[str, dict[str, Any]],
) -> None:
    headers = ["Agent", "Avg Ante", "Max Ante", "Win Rate"]
    widths = [17, 8, 8, 8]

    print(_table_sep(widths))
    print(_table_row(headers, widths))
    print(_table_sep(widths))

    # Trained first
    row = [
        "Trained (ours)",
        f"{trained_stats['avg_ante']:.2f}",
        str(trained_stats["max_ante"]),
        f"{trained_stats['win_rate']:.1%}",
    ]
    print(_table_row(row, widths))

    for agent_name, stats in baseline_results.items():
        row = [
            agent_name,
            f"{stats['avg_ante']:.2f}",
            str(stats["max_ante"]),
            f"{stats['win_rate']:.1%}",
        ]
        print(_table_row(row, widths))

    print(_table_sep(widths))


def print_ante_distribution(
    ante_dist: dict[int, int],
    n_episodes: int,
    label: str = "",
) -> None:
    if not ante_dist:
        return
    max_ante = max(ante_dist.keys())
    max_bar = 40

    # Find max count for scaling
    max_count = max(ante_dist.values()) if ante_dist else 1

    for ante in range(1, max_ante + 1):
        count = ante_dist.get(ante, 0)
        pct = count / n_episodes * 100 if n_episodes > 0 else 0
        bar_len = int(count / max_count * max_bar) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  Ante {ante}: {bar:<{max_bar}s} {pct:4.0f}% ({count}/{n_episodes})")


def print_stake_table(stake_results: dict[str, dict[str, Any]], deck: str) -> None:
    deck_name = DECK_NAMES.get(deck, deck)
    headers = ["Stake", "Avg Ante", "Max Ante", "Win Rate", "Avg Acts"]
    widths = [10, 8, 8, 8, 8]

    print(_table_sep(widths))
    print(_table_row(headers, widths))
    print(_table_sep(widths))

    for stake_str, stats in stake_results.items():
        stake = int(stake_str)
        name = STAKE_NAMES.get(stake, f"Stake {stake}")
        row = [
            name,
            f"{stats['avg_ante']:.2f}",
            str(stats["max_ante"]),
            f"{stats['win_rate']:.1%}",
            f"{stats['avg_actions']:.0f}",
        ]
        print(_table_row(row, widths))

    print(_table_sep(widths))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Balatro agent")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--decks", type=str, default="default",
                    help="Comma-separated deck keys, 'all', or 'default'")
    p.add_argument("--stakes", type=str, default="1,2,3",
                    help="Comma-separated stake levels")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--output", type=str, default="eval_results.json")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse deck list
    if args.decks == "all":
        decks = list(ALL_DECKS)
    elif args.decks == "default":
        decks = list(DEFAULT_DECKS)
    else:
        decks = [d.strip() for d in args.decks.split(",")]

    stakes = [int(s.strip()) for s in args.stakes.split(",")]

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    policy, device, checkpoint = load_model(args.checkpoint, args.device)

    cfg = checkpoint["config"]
    update_count = checkpoint.get("update_count", "?")
    global_step = checkpoint.get("global_step", "?")
    n_params = sum(p.numel() for p in policy.parameters())

    # Header
    ckpt_name = Path(args.checkpoint).name
    print()
    print("=" * 65)
    print(f"  Jackdaw Evaluation Report")
    print(f"  Checkpoint: {ckpt_name} (update {update_count}, step {global_step:,})")
    print(
        f"  Model: {cfg['embed_dim']}d/{cfg['num_heads']}h/{cfg['num_layers']}L "
        f"({n_params:,} params)"
    )
    print(f"  Device: {device}")
    print("=" * 65)

    agent = PolicyAgent(policy, device)
    all_results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "checkpoint_name": ckpt_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": {
            "embed_dim": cfg["embed_dim"],
            "num_heads": cfg["num_heads"],
            "num_layers": cfg["num_layers"],
            "n_params": n_params,
        },
        "training_steps": global_step,
        "update_count": update_count,
        "episodes_per_condition": args.episodes,
        "evaluations": {},
    }

    # --- Deck sweep ---
    primary_stake = stakes[0]
    print(
        f"\nDECK COMPARISON ({STAKE_NAMES.get(primary_stake, '?')} Stake, "
        f"{args.episodes} episodes each)"
    )
    deck_results = run_deck_sweep(agent, decks, primary_stake, args.episodes)
    print()
    print_deck_table(deck_results)
    all_results["evaluations"]["deck_sweep"] = deck_results

    # --- Ante distribution for primary deck ---
    primary_deck = decks[0]
    primary_stats = deck_results.get(primary_deck)
    if primary_stats:
        deck_name = DECK_NAMES.get(primary_deck, primary_deck)
        print(f"\nANTE DISTRIBUTION ({deck_name} Deck, trained agent)")
        ante_dist = {int(k): v for k, v in primary_stats["ante_distribution"].items()}
        print_ante_distribution(ante_dist, args.episodes)

    # --- Stake escalation ---
    if len(stakes) > 1:
        deck_name = DECK_NAMES.get(primary_deck, primary_deck)
        print(
            f"\nSTAKE ESCALATION ({deck_name} Deck, "
            f"{args.episodes} episodes each)"
        )
        stake_results = run_stake_escalation(
            agent, primary_deck, stakes, args.episodes,
        )
        print()
        print_stake_table(stake_results, primary_deck)
        all_results["evaluations"]["stake_escalation"] = stake_results

    # --- Baselines ---
    if not args.skip_baselines:
        print(
            f"\nBASELINE COMPARISON ({DECK_NAMES.get(primary_deck, primary_deck)} Deck, "
            f"{STAKE_NAMES.get(primary_stake, '?')} Stake)"
        )
        baseline_results = run_baselines(primary_deck, primary_stake, args.episodes)
        print()

        trained_stats = deck_results.get(primary_deck, {})
        print_baseline_table(trained_stats, baseline_results)
        all_results["evaluations"]["baselines"] = baseline_results

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
