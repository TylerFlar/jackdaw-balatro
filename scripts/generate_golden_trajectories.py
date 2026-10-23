"""Regenerate the golden behaviour-equivalence fixtures.

Writes tests/fixtures/golden_trajectories.json (digests) and
tests/fixtures/golden_tapes.json (the recorded action tapes the tests
replay).  The fixtures pin the simulator's exact behaviour (see
tests/golden/); only regenerate them when a behaviour change is intended,
and say so in the PR.

    uv run python scripts/generate_golden_trajectories.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.golden.trajectories import CHECKPOINT_EVERY, all_cases  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def main() -> None:
    digests: dict[str, dict] = {}
    tapes: dict[str, list] = {}
    failed_checks: list[str] = []
    for name, runner, spec in all_cases():
        res = runner(spec)
        digests[name] = res.as_json()
        if res.tape:
            tapes[name] = res.tape
        failed_checks += [f"{name}:{k}" for k, ok in res.checks.items() if not ok]
        print(f"{name:34s} records={res.steps:5d} actions={len(res.tape)}")
    if failed_checks:
        raise SystemExit(f"scenario checks failed, fixtures not written: {failed_checks}")
    data = {"checkpoint_every": CHECKPOINT_EVERY, "trajectories": digests}
    (FIXTURES / "golden_trajectories.json").write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n"
    )
    (FIXTURES / "golden_tapes.json").write_text(
        "{\n"
        + ",\n".join(
            f"{json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}"
            for k, v in sorted(tapes.items())
        )
        + "\n}\n"
    )
    print(f"wrote {len(digests)} trajectories")


if __name__ == "__main__":
    main()
