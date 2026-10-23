"""Bot path ≡ native path on every engine tape (:mod:`tests.bot.bot_native`)."""

from __future__ import annotations

import pytest

from tests.bot.bot_native import run_all


@pytest.mark.slow
def test_bot_path_matches_native_path_on_all_tapes() -> None:
    report = run_all()
    assert not report.mismatches, "\n".join(report.mismatches[:10])
    assert report.tapes >= 111 and report.steps >= 7800
    # Nearly every step goes through the bot's API; the rest are documented fallbacks.
    assert report.via_bot + report.sort_mode + sum(report.fallback.values()) == report.steps
    assert report.via_bot >= 0.95 * report.steps
    allowed = ("sim-only ", "SellCard -> sell: ", "UseConsumable -> use: ")
    unexpected = [r for r in report.fallback if not r.startswith(allowed)]
    assert not unexpected, unexpected
