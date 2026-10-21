"""Regenerate tests/fixtures/golden_trajectories.json.

The fixture pins the simulator's exact behaviour (see tests/golden/).  Only
regenerate it when a behaviour change is intended, and say so in the PR.

    uv run python scripts/generate_golden_trajectories.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.golden.trajectories import CHECKPOINT_EVERY, run_all  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "golden_trajectories.json"


def main() -> None:
    data = {"checkpoint_every": CHECKPOINT_EVERY, "trajectories": run_all()}
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(data['trajectories'])} trajectories to {OUT}")


if __name__ == "__main__":
    main()
