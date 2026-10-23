"""Single-step transplant parity: the Rust engine against the Python oracle.

For every state along the engine tapes, the Python snapshot is imported into
Rust, both engines take the tape's action, and the results are compared.  Rust
refuses steps that need logic it doesn't have yet ("unported"); those are
counted, not compared.

Run as a script for a report::

    PYTHONPATH=. .venv/Scripts/python.exe -m tests.parity.rust_step [--spec NAME] [--show N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from jackdaw import _core
from tests.parity import canon
from tests.parity.harness import ParitySpec, load_specs
from tests.parity.pyengine import PyEngine


def plain(snap: dict[str, Any]) -> dict[str, Any]:
    """The snapshot as JSON text reads back (plain str instead of str enums)."""
    return canon.from_json_compatible(json.loads(json.dumps(canon.to_json_compatible(snap))))


def rust_step(snap: dict[str, Any], action: list[Any]) -> tuple[str, dict[str, Any] | None]:
    """Step the Rust engine from ``snap``: (status, snapshot or None)."""
    try:
        status, text = _core._step_snapshot(
            json.dumps(canon.to_json_compatible(snap)), json.dumps(action)
        )
    except ValueError as exc:  # a snapshot Rust can't hold counts as unported
        return f"unported: import: {exc}", None
    if status != "ok":
        return status, None
    return status, canon.from_json_compatible(json.loads(text))


@dataclass
class Report:
    steps: Counter = field(default_factory=Counter)
    matched: Counter = field(default_factory=Counter)
    unported: Counter = field(default_factory=Counter)
    mismatches: dict[str, list[tuple[str, list[str]]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_mismatch(self, kind: str, where: str, diff: list[str]) -> None:
        self.mismatches[kind].append((where, diff))


def check_step(engine: PyEngine, action: list[Any], where: str, report: Report) -> bool:
    """Step both engines; record the outcome. False when the tape can't continue."""
    kind = action[0]
    report.steps[kind] += 1
    snap = engine.snapshot()
    status, rust_snap = rust_step(snap, action)
    try:
        engine.step(action)
        py_ok = True
    except Exception:  # an illegal action ends the tape
        py_ok = False
    if status.startswith("unported"):
        report.unported[status.split(":", 1)[1].strip().split(" (")[0]] += 1
        return py_ok
    rust_ok = status == "ok"
    if rust_ok != py_ok:
        report.add_mismatch(kind, where, [f"outcome: rust {status!r}, python ok={py_ok}"])
        return py_ok
    if not rust_ok:
        report.matched[kind] += 1
        return False
    diff = canon.diff(plain(engine.snapshot()), rust_snap)
    if diff:
        report.add_mismatch(kind, where, diff)
    else:
        report.matched[kind] += 1
    return True


def run_spec(spec: ParitySpec, report: Report) -> None:
    """Check every step of one tape."""
    engine = PyEngine(spec.start, spec.setup)
    for pos, action in enumerate(spec.tape):
        snap = engine.snapshot()
        if snap["phase"] == "game_over" or snap["run"]["won"]:
            break
        if not check_step(engine, action, f"{spec.name}@{pos}", report):
            break


def run(spec_filter: str | None = None) -> Report:
    report = Report()
    for spec in load_specs():
        if spec_filter and spec_filter not in spec.name:
            continue
        run_spec(spec, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    r = run(args.spec)
    print(f"{'action':<16}{'steps':>8}{'matched':>9}{'mismatch':>10}")
    for kind, n in sorted(r.steps.items()):
        print(f"{kind:<16}{n:>8}{r.matched[kind]:>9}{len(r.mismatches[kind]):>10}")
    print("\nunported:", dict(r.unported.most_common(20)))
    for kind, items in r.mismatches.items():
        print(f"\n== {kind}: {len(items)} mismatches")
        for where, diff in items[: args.show]:
            print(f"  {where}")
            for line in diff[:12]:
                print(f"      {line}")


if __name__ == "__main__":
    main()
