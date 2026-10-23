"""Regenerate the golden v2 fixtures (parity specs + canonical digests).

Writes tests/fixtures/parity_specs.json (setup ops and action tapes, derived
from the golden v1 engine trajectories) and tests/fixtures/golden_v2.json
(engine-agnostic behaviour digests, produced by the Python oracle).  Only
regenerate when a behaviour change is intended, and say so in the PR.

    uv run python scripts/generate_golden_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.parity.golden_v2 import run_all, write_fixture  # noqa: E402
from tests.parity.harness import (  # noqa: E402
    load_specs,
    resolve_last_hand,
    specs_from_golden_v1,
    write_specs,
)
from tests.parity.pyengine import PyEngine  # noqa: E402


def main() -> None:
    specs = resolve_last_hand(specs_from_golden_v1())
    write_specs(specs)
    specs = load_specs()  # round-trip through JSON, as the tests read them
    results = run_all(specs, lambda s: PyEngine(s.start, s.setup))
    failed = [f"{n}:{k}" for n, r in results.items() for k, ok in r["checks"].items() if not ok]
    if failed:
        raise SystemExit(f"scenario checks failed, fixtures not written: {failed}")
    write_fixture(results)
    for name, r in results.items():
        print(f"{name:34s} records={r['records']:5d} actions={r['actions']}")
    print(f"wrote {len(results)} trajectories")


if __name__ == "__main__":
    main()
