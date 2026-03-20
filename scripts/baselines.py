"""Baseline agent evaluation for M18.

Runs RandomAgent and HeuristicAgent through the full env pipeline and
records performance baselines that the RL agent needs to beat.

Usage:
    uv run python scripts/baselines.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jackdaw.env.agents import (
    EvalResult,
    HeuristicAgent,
    RandomAgent,
    evaluate_agent,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAIN_N_EPISODES = 200
DECK_N_EPISODES = 50
MAIN_BACK = "b_red"
MAIN_STAKE = 1
MAX_STEPS = 5000
EPISODE_TIMEOUT_S = 30.0  # per-episode wall-clock timeout
EXTRA_DECKS = ["b_blue", "b_yellow", "b_green"]

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


@dataclass
class RunReport:
    """Formatted report for a single evaluation run."""

    label: str
    result: EvalResult
    wall_time: float
    timeouts: int = 0


def _ante_distribution(result: EvalResult) -> dict[int, int]:
    """Count episodes that reached each ante."""
    counter: Counter[int] = Counter()
    for ep in result.episodes:
        counter[ep.ante_reached] += 1
    return dict(sorted(counter.items()))


def _print_report(report: RunReport) -> None:
    """Print a detailed report to stdout."""
    r = report.result
    antes = [ep.ante_reached for ep in r.episodes]
    actions = [ep.actions_taken for ep in r.episodes]

    print(f"\n{'=' * 60}")
    print(f"  {report.label}")
    print(f"{'=' * 60}")
    print(f"  Episodes:       {r.n_episodes}")
    print(f"  Win rate:       {r.win_rate:.1%}")
    print(f"  Avg ante:       {r.avg_ante:.2f}")
    print(f"  Avg rounds:     {r.avg_rounds:.1f}")
    print(f"  Avg actions:    {r.avg_actions:.0f}")
    if antes:
        print(f"  Min ante:       {min(antes)}")
        print(f"  Max ante:       {max(antes)}")
    if actions:
        print(f"  Min actions:    {min(actions)}")
        print(f"  Max actions:    {max(actions)}")
    print(f"  Wall time:      {report.wall_time:.1f}s")
    if report.timeouts > 0:
        print(f"  TIMEOUTS:       {report.timeouts}")

    dist = _ante_distribution(r)
    print("\n  Ante distribution:")
    for ante, count in dist.items():
        bar = "#" * count
        print(f"    Ante {ante:>2d}: {count:>4d}  {bar}")


def _run_eval(label: str, agent, **kwargs) -> RunReport:
    """Run evaluate_agent with timing and timeout detection."""
    print(f"\nRunning: {label} ...", flush=True)
    t0 = time.monotonic()
    result = evaluate_agent(agent, **kwargs)
    wall = time.monotonic() - t0

    # Detect likely timeouts: episodes that hit max_steps
    max_steps = kwargs.get("max_steps", MAX_STEPS)
    timeouts = sum(1 for ep in result.episodes if ep.actions_taken >= max_steps)

    report = RunReport(label=label, result=result, wall_time=wall, timeouts=timeouts)
    _print_report(report)
    return report


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def _write_markdown(reports: list[RunReport], path: Path) -> None:
    """Write results to a markdown file."""
    lines: list[str] = []
    lines.append("# M18 Baseline Agent Performance")
    lines.append("")
    lines.append("Baselines recorded by `scripts/baselines.py`. These are the numbers")
    lines.append("the RL agent needs to beat.")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Agent | Episodes | Win Rate | Avg Ante "
        "| Avg Rounds | Avg Actions | Min Ante | Max Ante |"
    )
    lines.append(
        "|-------|----------|----------|----------|"
        "------------|-------------|----------|----------|"
    )

    for rep in reports:
        r = rep.result
        antes = [ep.ante_reached for ep in r.episodes]
        min_ante = min(antes) if antes else 0
        max_ante = max(antes) if antes else 0

        lines.append(
            f"| {rep.label} | {r.n_episodes} | "
            f"{r.win_rate:.1%} | {r.avg_ante:.2f} | "
            f"{r.avg_rounds:.1f} | {r.avg_actions:.0f} | "
            f"{min_ante} | {max_ante} |"
        )

    lines.append("")

    # Detailed ante distributions
    lines.append("## Ante Distributions")
    lines.append("")
    for rep in reports:
        lines.append(f"### {rep.label}")
        lines.append("")
        dist = _ante_distribution(rep.result)
        lines.append("| Ante | Count | % |")
        lines.append("|------|-------|---|")
        n = rep.result.n_episodes
        for ante, count in dist.items():
            pct = count / n * 100 if n > 0 else 0
            lines.append(f"| {ante} | {count} | {pct:.1f}% |")
        lines.append("")

    # Timeout warnings
    timeout_reports = [r for r in reports if r.timeouts > 0]
    if timeout_reports:
        lines.append("## Warnings")
        lines.append("")
        for rep in timeout_reports:
            lines.append(
                f"- **{rep.label}**: {rep.timeouts} episodes hit max_steps "
                f"({MAX_STEPS}) — possible infinite loop"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    reports: list[RunReport] = []

    # 1. RandomAgent — White Stake, Red Deck
    rep = _run_eval(
        "RandomAgent / b_red / stake 1",
        RandomAgent(),
        n_episodes=MAIN_N_EPISODES,
        back_key=MAIN_BACK,
        stake=MAIN_STAKE,
        max_steps=MAX_STEPS,
    )
    reports.append(rep)

    # 2. HeuristicAgent — White Stake, Red Deck
    rep = _run_eval(
        "HeuristicAgent / b_red / stake 1",
        HeuristicAgent(),
        n_episodes=MAIN_N_EPISODES,
        back_key=MAIN_BACK,
        stake=MAIN_STAKE,
        max_steps=MAX_STEPS,
    )
    reports.append(rep)

    if rep.result.win_rate == 0.0:
        # 0% win rate is expected: Balatro requires beating ante 8 boss
        # (100K chips), which is beyond the heuristic's simple strategy.
        # But verify it's at least progressing past ante 1.
        if rep.result.avg_ante <= 1.0:
            print("\n*** WARNING: HeuristicAgent stuck at ante 1! ***")
            print("    This indicates a bug: the agent should progress beyond ante 1.")
            print("    Check: agent stuck in loop, factored action mapping, phase handling.")

    # 3. HeuristicAgent on other decks
    for deck in EXTRA_DECKS:
        rep = _run_eval(
            f"HeuristicAgent / {deck} / stake 1",
            HeuristicAgent(),
            n_episodes=DECK_N_EPISODES,
            back_key=deck,
            stake=MAIN_STAKE,
            max_steps=MAX_STEPS,
        )
        reports.append(rep)

    # Write markdown
    docs_path = Path(__file__).resolve().parent.parent / "docs" / "baselines.md"
    _write_markdown(reports, docs_path)

    # Final summary
    print(f"\n{'=' * 60}")
    print("  BASELINE SUMMARY")
    print(f"{'=' * 60}")
    for rep in reports:
        r = rep.result
        timeout_str = f" ({rep.timeouts} timeouts!)" if rep.timeouts else ""
        print(
            f"  {rep.label:<40s}  "
            f"win={r.win_rate:5.1%}  ante={r.avg_ante:.2f}{timeout_str}"
        )

    # Sanity check: HeuristicAgent should at least progress past ante 1.
    # 0% win rate is expected (ante 8 requires ~100K chips, beyond simple heuristics),
    # but being stuck at ante 1 would indicate a pipeline bug.
    main_heuristic = reports[1]
    if main_heuristic.result.avg_ante <= 1.0:
        print("\nFAILED: HeuristicAgent stuck at ante 1 — pipeline bug.")
        sys.exit(1)

    # Verify HeuristicAgent clearly beats RandomAgent
    main_random = reports[0]
    if main_heuristic.result.avg_ante <= main_random.result.avg_ante:
        print("\nFAILED: HeuristicAgent not better than RandomAgent — likely a bug.")
        sys.exit(1)


if __name__ == "__main__":
    main()
