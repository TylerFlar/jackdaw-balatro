"""BalatroBot's own endpoint tests, run against ``jackdaw serve`` (:mod:`tests.bot.upstream`).

Every test must pass, except the ones in ``upstream_xfail.json``, which must fail (strict
xfail, each with its reason).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.bot.upstream import MANIFEST, run


@pytest.mark.slow
def test_upstream_suite_passes_against_jackdaw_serve(tmp_path: Path) -> None:
    out = run(tmp_path / "suite")
    counts = out.counts()
    assert out.outcomes, f"the suite did not run:\n{out.output}"
    bad = {k: v for k, v in out.outcomes.items() if v not in ("passed", "xfailed")}
    details = "\n".join(f"{v} {k}: {out.messages.get(k, '')[:200]}" for k, v in bad.items())
    assert not bad, f"{dict(counts)}\n{details}"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    xfailed = {k for k, v in out.outcomes.items() if v == "xfailed"}
    assert xfailed == set(manifest), "the manifest lists tests that didn't run"
    assert counts["passed"] >= 455
