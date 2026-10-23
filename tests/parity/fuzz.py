"""Lockstep fuzzing: random runs played by both engines, compared at every step.

Each case is a random start (back, stake, seed, challenge), an optional setup
(a forced boss, a joker board added with BalatroBot ``add``, money, a later
ante) and a seeded policy over the oracle's legal actions: random choices
weighted by action kind, mixed with the smart agent so runs get past the
early antes.

The Rust engine keeps one ``jackdaw._core.Game`` from the start, with no
re-import between steps, so state a snapshot doesn't carry is exercised too.
At every step the legal-action lists, the step outcome and the snapshot
digests must agree. A case ends at game over (a won run plays on into
endless mode), ``max_steps``, or a step Rust refuses as not ported yet. A
divergence is reported with its first snapshot diff and can be saved as a
parity spec (start, setup, tape) that :func:`replay` and
``tests/parity/test_fuzz.py`` play again. The exit status is 1 when any case
diverged (the nightly workflow runs it)::

    PYTHONPATH=. .venv/Scripts/python.exe -m tests.parity.fuzz --cases 200 [--seed 1] [--save DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jackdaw import _core
from jackdaw.core.game import random_seed
from jackdaw.engine.actions import GamePhase, get_legal_actions
from jackdaw.engine.challenges import CHALLENGES
from tests.golden.trajectories import (
    _BACKS,
    _BOSSES,
    _resolve,
    _with_forced_card,
    encode_action,
)
from tests.parity import canon
from tests.parity.harness import ROOT
from tests.parity.pyengine import PyEngine
from tests.parity.rust_step import plain

#: Relative weights of the random policy by action kind (default 1). Moves that
#: only reorder cards, and selling, would otherwise dominate long shop visits.
_WEIGHTS = {
    "SwapJokersLeft": 0.05,
    "SwapJokersRight": 0.05,
    "SwapHandLeft": 0.05,
    "SwapHandRight": 0.05,
    "SortHand": 0.1,
    "SellCard": 0.2,
    "Reroll": 0.5,
}
_EDITIONS = ("foil", "holo", "polychrome", "negative")
_SEALS = ("Red", "Blue", "Gold", "Purple")


def _centers(*sets: str) -> list[str]:
    data = json.loads((ROOT / "jackdaw" / "engine" / "data" / "centers.json").read_text("utf-8"))
    return sorted(k for k, v in data.items() if v.get("set") in sets)


_CONSUMABLES = _centers("Tarot", "Planet", "Spectral")
_ENHANCEMENTS = _centers("Enhanced")
_FRONTS = [f"{s}_{r}" for s in "HDCS" for r in "23456789TJQKA"]


@dataclass
class Case:
    name: str
    start: dict[str, Any]
    setup: list[dict[str, Any]]
    policy_seed: int
    smart: float  # chance that the smart agent picks the action
    max_steps: int

    def spec(self, tape: list[list[Any]]) -> dict[str, Any]:
        """The case and the tape it played, as a parity spec."""
        return {
            "name": self.name,
            "start": self.start,
            "setup": self.setup,
            "tape": tape,
            "max_steps": self.max_steps,
            "check": None,
            "check_after": 0,
        }


@dataclass
class Outcome:
    case: Case
    steps: int = 0
    ante: int = 1  # the ante the run reached
    stop: str = ""
    divergence: str | None = None
    diff: list[str] = field(default_factory=list)
    tape: list[list[Any]] = field(default_factory=list)


def random_case(index: int, rng: random.Random, max_steps: int = 400) -> Case:
    r = random.Random(rng.getrandbits(64))
    if r.random() < 0.2:
        back, challenge = "b_challenge", r.choice(sorted(CHALLENGES))
    else:
        back, challenge = r.choice(_BACKS), None
    start = {"back": back, "stake": r.randint(1, 8), "seed": random_seed(r), "challenge": challenge}
    setup: list[dict[str, Any]] = []
    if r.random() < 0.25:
        setup.append({"op": "force_boss", "key": r.choice(_BOSSES)})
    if r.random() < 0.5:
        setup.append({"op": "action", "action": ["SelectBlind"]})
        ported = sorted(_core._ported_jokers())
        for key in r.sample(ported, r.randint(1, 5)):
            op: dict[str, Any] = {"op": "add", "key": key}
            if r.random() < 0.2:
                op["edition"] = r.choice(_EDITIONS)
            setup.append(op)
        # Consumables and modified playing cards, which shops alone rarely combine with a
        # given board. Added cards join the hand (BalatroBot add in SELECTING_HAND).
        if r.random() < 0.5:
            for key in r.sample(_CONSUMABLES, r.randint(1, 2)):
                setup.append({"op": "add", "key": key})
        if r.random() < 0.5:
            for _ in range(r.randint(1, 4)):
                op = {"op": "add", "key": r.choice(_FRONTS)}
                if r.random() < 0.7:
                    op["enhancement"] = r.choice(_ENHANCEMENTS)
                if r.random() < 0.4:
                    op["seal"] = r.choice(_SEALS)
                if r.random() < 0.3:
                    op["edition"] = r.choice(_EDITIONS[:3])
                setup.append(op)
        setup.append({"op": "set", "dollars": r.choice([4, 15, 40, 100])})
        # Random play rarely survives ante 4: start some boards later, so late
        # antes (blind sizes, showdown bosses, pools) get played too.
        if r.random() < 0.4:
            setup.append({"op": "set", "ante": r.randint(3, 9)})
    return Case(
        name=f"fuzz_{index}_{start['seed']}",
        start=start,
        setup=setup,
        policy_seed=r.getrandbits(32),
        smart=r.choice([0.3, 0.6, 0.9]),
        max_steps=max_steps,
    )


def _choose(engine: PyEngine, rng: random.Random, smart: float) -> list[Any] | None:
    from jackdaw.cli.smart_agent import smart_agent

    with engine._ids():
        gs = engine.gs
        legal = get_legal_actions(gs)
        if not legal:
            return None
        # Random hands lose the first blind; the smart agent plays most of them
        # so runs reach later antes, and ``smart`` sets how often it shops.
        in_round = gs.get("phase") == GamePhase.SELECTING_HAND
        if rng.random() < (0.9 if in_round else smart):
            action = smart_agent(gs, legal)
        else:
            weights = [_WEIGHTS.get(type(a).__name__, 1.0) for a in legal]
            action = rng.choices(legal, weights)[0]
        return encode_action(_with_forced_card(_resolve(action, gs, rng), gs))


def _rust_view(game: Any) -> dict[str, Any]:
    return canon.from_json_compatible(json.loads(game.snapshot()))


def run_case(case: Case, tape: list[list[Any]] | None = None) -> Outcome:
    """Play ``case`` in both engines: with its policy, or the given ``tape``."""
    out = Outcome(case)
    py = PyEngine(case.start, case.setup)
    # The setup ops run natively in Rust too (BalatroBot's add/set, Game.apply_setup).
    s = case.start
    game = _core.Game.new(s["back"], s["stake"], s["seed"], s.get("challenge"))
    for op in case.setup:
        game.apply_setup(json.dumps(op))
    found = canon.diff(plain(py.snapshot()), _rust_view(game))
    if found:
        out.stop, out.divergence = "divergence", "setup" if case.setup else "new run"
        out.diff = found
        return out
    rng = random.Random(case.policy_seed)
    for pos in range(case.max_steps):
        snap = plain(py.snapshot())
        out.ante = int(snap["round_resets"]["ante"])
        # A won run goes on into endless mode, as the game offers.
        if snap["phase"] == GamePhase.GAME_OVER:
            out.stop = "game over (won)" if snap["run"]["won"] else "game over"
            return out
        py_legal = py.legal_actions()
        try:
            rs_legal = json.loads(game.legal_actions())
        except _core.NotPortedError as exc:
            out.stop = f"unported: {exc}"
            return out
        if rs_legal != py_legal:
            pj = [json.dumps(a) for a in py_legal]
            rj = [json.dumps(a) for a in rs_legal]
            out.stop, out.divergence = "divergence", f"legal actions at step {pos}"
            out.diff = [
                f"missing {[a for a in pj if a not in rj][:6]}",
                f"extra {[a for a in rj if a not in pj][:6]}",
            ]
            return out
        if tape is not None:
            if pos == len(tape):
                out.stop = "tape end"
                return out
            action = tape[pos]
        else:
            action = _choose(py, rng, case.smart)
        if action is None:
            out.stop = "no legal action"
            return out
        out.tape.append(action)
        try:
            game.step(json.dumps(action))
            rs_ok = True
        except _core.NotPortedError as exc:
            out.tape.pop()
            out.stop = f"unported: {exc}"
            return out
        except _core.IllegalActionError:
            rs_ok = False
        try:
            py.step(action)
            py_ok = True
        except Exception:
            py_ok = False
        if not py_ok:
            out.stop, out.divergence = "divergence", f"oracle refused its own legal {action}"
            return out
        if not rs_ok:
            out.stop, out.divergence = "divergence", f"rust refused {action} at step {pos}"
            return out
        out.steps += 1
        py_snap = plain(py.snapshot())
        if game.state_hash() != canon.digest(py_snap):
            found = canon.diff(py_snap, _rust_view(game))
            out.stop, out.divergence = "divergence", f"state after step {pos} {action}"
            out.diff = found or ["digests differ, no field diff"]
            return out
    out.stop = "max steps"
    return out


def replay(spec: dict[str, Any]) -> Outcome:
    """Play a saved case (a parity spec written by ``--save``) in both engines."""
    case = Case(spec["name"], spec["start"], spec["setup"], 0, 0.0, len(spec["tape"]) + 1)
    return run_case(case, spec["tape"])


def _reason(stop: str) -> str:
    """Group unported stops by what is missing (drop per-case detail)."""
    return stop.split(" (")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save", type=Path, help="write each divergence as a parity spec here")
    ap.add_argument("--show", type=int, default=8, help="diff lines shown per divergence")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stops: Counter[str] = Counter()
    antes: Counter[int] = Counter()
    steps = 0
    divergent: list[Outcome] = []
    t0 = time.perf_counter()
    for i in range(args.cases):
        case = random_case(i, rng, args.max_steps)
        out = run_case(case)
        steps += out.steps
        antes[out.ante] += 1
        stops[_reason(out.stop)] += 1
        if out.divergence:
            divergent.append(out)
            print(f"DIVERGENCE {case.name} ({case.start}): {out.divergence}")
            for line in out.diff[: args.show]:
                print(f"    {line}")
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                path = args.save / f"{case.name}.json"
                path.write_text(json.dumps(case.spec(out.tape)), encoding="utf-8")
    took = time.perf_counter() - t0
    print(f"\n{args.cases} cases, {steps} matched steps in {took:.1f}s")
    for stop, n in stops.most_common():
        print(f"  {n:4d}  {stop}")
    print("antes reached: " + ", ".join(f"{a}: {n}" for a, n in sorted(antes.items())))
    print(f"{len(divergent)} divergence(s)")
    raise SystemExit(1 if divergent else 0)


if __name__ == "__main__":
    main()
