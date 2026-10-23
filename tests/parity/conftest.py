"""After each test, the runs it built (:class:`tests.parity.cases.Run`) made every check the
oracle's runs made."""

from __future__ import annotations

from typing import Any

import pytest

from tests.parity import cases


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.parity_call_passed = report.passed


@pytest.fixture(autouse=True)
def _every_frozen_check_is_made(request: Any) -> Any:
    yield
    runs = cases.finish_test(request.node.nodeid)
    if getattr(request.node, "parity_call_passed", False):
        for run in runs:
            assert run.pos == len(run.events), (
                f"{run.where}: {len(run.events) - run.pos} of the oracle's checks were not made"
            )
