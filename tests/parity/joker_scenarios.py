"""Per-joker parity scenarios: the Rust engine against the oracle with chosen jokers.

A scenario starts a run (seeded by the jokers and an index), selects the first
blind, adds the jokers through BalatroBot's ``add``, and plays with a seeded
policy:
it selects every blind, plays and discards with the golden mixed policy,
cashes out, and leaves the shop and packs without buying.  Every step is
checked like a tape step (:mod:`tests.parity.rust_step`): the oracle's
snapshot goes into Rust, both engines take the action, and the results must
match.  Steps Rust refuses as unported are counted and skipped; the oracle
steps on.

Run as a script::

    PYTHONPATH=. .venv/Scripts/python.exe -m tests.parity.joker_scenarios j_joker j_hack [--seeds 4]
"""

from __future__ import annotations

import argparse
import random
from typing import Any

from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
    SkipPack,
    get_legal_actions,
)
from tests.golden.trajectories import _mixed_policy, encode_action
from tests.parity.pyengine import PyEngine
from tests.parity.rust_step import Report, check_step


def _policy(gs: dict[str, Any], legal: list[Any], rng: random.Random) -> Any:
    phase = gs["phase"]
    if phase == GamePhase.BLIND_SELECT and SelectBlind() in legal:
        return SelectBlind()
    if phase == GamePhase.ROUND_EVAL and CashOut() in legal:
        return CashOut()
    if phase == GamePhase.SHOP and NextRound() in legal:
        return NextRound()
    if phase == GamePhase.PACK_OPENING and SkipPack() in legal:
        return SkipPack()
    hand_actions = [a for a in legal if isinstance(a, (PlayHand, Discard))]
    return _mixed_policy(gs, hand_actions or legal, rng, eps=0.3)


def scenario(
    jokers: tuple[str, ...],
    index: int = 0,
    *,
    steps: int = 80,
    report: Report | None = None,
    back: str = "b_red",
    stake: int = 1,
) -> Report:
    """Play one seeded scenario with ``jokers`` on the board; return the step report."""
    report = report if report is not None else Report()
    name = "+".join(jokers)
    start = {"back": back, "stake": stake, "seed": f"JK{index}_{name}", "challenge": None}
    # BalatroBot's add works from SELECTING_HAND on, so the first blind is selected first.
    setup = [{"op": "action", "action": ["SelectBlind"]}]
    setup += [{"op": "add", "key": key} for key in jokers]
    engine = PyEngine(start, setup)
    rng = random.Random(f"{name}:{index}")
    for pos in range(steps):
        snap = engine.snapshot()
        if snap["phase"] == "game_over" or snap["run"]["won"]:
            break
        with engine._ids():
            legal = get_legal_actions(engine.gs)
            if not legal:
                break
            action = encode_action(_policy(engine.gs, legal, rng))
        if not check_step(engine, action, f"{name}#{index}@{pos}", report):
            break
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jokers", nargs="+", help="joker keys; joined with + to share a board")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--together", action="store_true", help="all jokers on one board")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    groups = [tuple(args.jokers)] if args.together else [(j,) for j in args.jokers]
    for group in groups:
        report = Report()
        for i in range(args.seeds):
            scenario(group, i, steps=args.steps, report=report)
        checked = sum(report.matched.values())
        bad = sum(len(v) for v in report.mismatches.values())
        unported = sum(report.unported.values())
        label = "+".join(group)
        print(f"{label:<40} matched {checked:>4}  mismatched {bad:>3}  unported {unported:>4}")
        for kind, items in report.mismatches.items():
            for where, diff in items[: args.show]:
                print(f"    {kind} {where}")
                for line in diff[:10]:
                    print(f"        {line}")
        if report.unported:
            print("    unported:", dict(report.unported.most_common(6)))


if __name__ == "__main__":
    main()
