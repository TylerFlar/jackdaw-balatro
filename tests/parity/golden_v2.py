"""Golden v2: engine-agnostic behaviour digests for the recorded engine tapes.

Golden v2 hashes canonical snapshots (``docs/snapshot-schema.md``) plus the
legal action list at every step, so any engine that implements the snapshot
schema reproduces the same digests.  The Python oracle produced them; the
fixture outlived it.

Fixtures:

* ``tests/fixtures/parity_specs.json`` - start, setup ops and action tape
  for every trajectory (see :mod:`tests.parity.harness`).
* ``tests/fixtures/golden_v2.json`` - per-trajectory digest chain,
  checkpoints every ``CHECKPOINT_EVERY`` records, and scenario checks.

Rewrite the digests from the Rust engine with
``.venv/Scripts/python.exe scripts/refreeze.py golden_v2`` only when a
behaviour change is intended, and say so in the PR.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.parity.harness import (
    CHECKPOINT_EVERY,
    FIXTURES,
    Engine,
    ParitySpec,
    Replay,
    replay,
)

GOLDEN_V2_PATH = FIXTURES / "golden_v2.json"


def run_all(
    specs: list[ParitySpec], make_engine: Callable[[ParitySpec], Engine]
) -> dict[str, dict[str, Any]]:
    return {spec.name: replay(spec, make_engine(spec)).as_json() for spec in specs}


def write_fixture(results: dict[str, dict[str, Any]], path: Path = GOLDEN_V2_PATH) -> None:
    data = {"schema": 1, "checkpoint_every": CHECKPOINT_EVERY, "trajectories": results}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_fixture(path: Path = GOLDEN_V2_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_divergence(expected: dict[str, Any], got: Replay) -> str:
    """Describe where a replay first departs from its golden record."""
    exp_cp: list[str] = expected["checkpoints"]
    for i, (a, b) in enumerate(zip(exp_cp, got.checkpoints, strict=False)):
        if a != b:
            lo = i * CHECKPOINT_EVERY
            return f"records {lo + 1}..{lo + CHECKPOINT_EVERY} (checkpoint {i})"
    if len(exp_cp) != len(got.checkpoints):
        return f"checkpoint count {len(exp_cp)} != {len(got.checkpoints)}"
    last = len(exp_cp) * CHECKPOINT_EVERY
    return f"after record {last} (records {expected['records']} vs {got.records})"
