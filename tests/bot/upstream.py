"""Run BalatroBot's own endpoint test suite against ``jackdaw serve``.

The vendored suite (``third_party/balatrobot/v1.5.2/tests``) starts real
Balatro instances through ``balatrobot.manager.BalatroInstance``.  This module
copies the suite into a scratch directory (the vendored tree is never touched)
next to a small ``balatrobot`` shim whose ``BalatroInstance`` starts
``jackdaw serve --debug`` on the port the suite picked, and runs it with pytest
in a subprocess.  Tests listed in ``upstream_xfail.json`` are marked
``xfail(strict=True)`` with their reason, so a fixed one shows up as XPASS.

Run directly for a report (and to refresh the manifest by hand)::

    .venv/Scripts/python.exe -m tests.bot.upstream [pytest args...]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDORED_TESTS = ROOT / "third_party" / "balatrobot" / "v1.5.2" / "tests"
MANIFEST = Path(__file__).with_name("upstream_xfail.json")

_CONFIG = '''"""jackdaw shim: balatrobot.config."""


class Config:
    @classmethod
    def from_env(cls) -> "Config":
        return cls()
'''

_MANAGER = '''"""jackdaw shim: a BalatroInstance is ``jackdaw serve --debug`` on the given port."""

import asyncio
import os
import subprocess
import sys

import httpx


class BalatroInstance:
    def __init__(self, config, port: int = 12346) -> None:
        self.port = port
        self.proc = None

    def _healthy(self) -> bool:
        body = {"jsonrpc": "2.0", "method": "health", "params": {}, "id": 1}
        try:
            r = httpx.post(f"http://127.0.0.1:{self.port}", json=body, timeout=2.0)
            return r.json().get("result", {}).get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False

    async def start(self) -> None:
        cmd = [sys.executable, "-m", "jackdaw.cli.main", "serve", "--port", str(self.port)]
        self.proc = subprocess.Popen(
            cmd + ["--debug"],
            cwd=os.environ["JACKDAW_ROOT"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(200):
            if self._healthy():
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"jackdaw serve exited ({self.proc.returncode})")
            await asyncio.sleep(0.05)
        raise RuntimeError(f"jackdaw serve did not answer on port {self.port}")

    async def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
'''

# The suite's ``balatro_server`` is an async session fixture (pytest-asyncio upstream); this
# one waits for the server synchronously instead.
_LOCAL_CONFTEST = '''"""jackdaw: a synchronous ``balatro_server`` (upstream's is async)."""

import pytest

from tests.lua.conftest import HOST, _check_health


@pytest.fixture(scope="session")
def balatro_server(port):
    if not _check_health(HOST, port, timeout=10.0):
        pytest.fail(f"jackdaw serve on port {port} not responding")
    yield None
'''

_PLUGIN = '''"""jackdaw plugin for the upstream suite: no xdist, and the xfail manifest."""

import json
import os
import types

import pytest


@pytest.fixture(scope="session")
def worker_id():
    return "master"


@pytest.fixture
def instance(host, port):
    """The server's address (test_server.py's concurrency test asks for an ``instance``
    fixture that the vendored conftest doesn't define)."""
    return types.SimpleNamespace(host=host, port=port)


def pytest_collection_modifyitems(config, items):
    manifest = json.loads(open(os.environ["JACKDAW_XFAIL"], encoding="utf-8").read())
    for item in items:
        reason = manifest.get(item.nodeid)
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
'''


def prepare(dest: Path) -> Path:
    """Copy the suite and write the shim into ``dest``; return ``dest``."""
    shutil.copytree(
        VENDORED_TESTS,
        dest / "tests",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    shim = dest / "shim" / "balatrobot"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text("", encoding="utf-8")
    (shim / "config.py").write_text(_CONFIG, encoding="utf-8")
    (shim / "manager.py").write_text(_MANAGER, encoding="utf-8")
    for sub in ("endpoints", "core"):
        (dest / "tests" / "lua" / sub / "conftest.py").write_text(_LOCAL_CONFTEST, "utf-8")
    (dest / "jackdaw_upstream_plugin.py").write_text(_PLUGIN, encoding="utf-8")
    (dest / "pytest.ini").write_text("[pytest]\naddopts = -p no:cacheprovider\n", "utf-8")
    return dest


@dataclass
class Outcome:
    outcomes: dict[str, str] = field(default_factory=dict)  # nodeid -> outcome
    messages: dict[str, str] = field(default_factory=dict)  # nodeid -> failure text
    returncode: int = 0
    output: str = ""

    def counts(self) -> Counter[str]:
        return Counter(self.outcomes.values())


def _nodeid(case: ET.Element) -> str:
    classname = case.get("classname", "")
    parts = classname.split(".")
    # tests.lua.endpoints.test_add.TestAddEndpoint -> tests/lua/endpoints/test_add.py::TestAdd...
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("test_"):
            path = "/".join(parts[:i]) + ".py"
            rest = parts[i:]
            return "::".join([path, *rest, case.get("name", "")])
    return f"{classname}::{case.get('name')}"


def _parse(report: Path) -> Outcome:
    out = Outcome()
    for case in ET.parse(report).getroot().iter("testcase"):
        nodeid = _nodeid(case)
        outcome = "passed"
        for child in case:
            if child.tag == "skipped":
                kind = child.get("type", "")
                outcome = "xfailed" if kind == "pytest.xfail" else "skipped"
            elif child.tag in ("failure", "error"):
                text = child.get("message", "") or ""
                outcome = "xpassed" if "XPASS(strict)" in text else "failed"
                out.messages[nodeid] = text
        out.outcomes[nodeid] = outcome
    return out


def run(work: Path, args: list[str] | None = None, timeout: float = 900) -> Outcome:
    """Run the suite in ``work`` (a fresh directory) and return every test's outcome."""
    dest = prepare(work)
    report = dest / "report.xml"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(dest), str(dest / "shim")])
    env["JACKDAW_ROOT"] = str(ROOT)
    env["JACKDAW_XFAIL"] = str(MANIFEST)
    env.pop("PYTEST_ADDOPTS", None)
    cmd = [sys.executable, "-m", "pytest", "tests/lua", "-p", "jackdaw_upstream_plugin"]
    cmd += ["-q", "--tb=line", "-rN", f"--junitxml={report}", *(args or [])]
    proc = subprocess.run(cmd, cwd=dest, env=env, capture_output=True, text=True, timeout=timeout)
    out = _parse(report) if report.exists() else Outcome()
    out.returncode = proc.returncode
    out.output = proc.stdout[-4000:] + proc.stderr[-4000:]
    return out


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="bot_upstream_") as tmp:
        out = run(Path(tmp), sys.argv[1:])
    counts = out.counts()
    print(dict(counts))
    for nodeid, outcome in sorted(out.outcomes.items()):
        if outcome in ("failed", "xpassed"):
            print(f"{outcome.upper():8} {nodeid}\n         {out.messages.get(nodeid, '')[:300]}")
    if not out.outcomes:
        print(out.output)


if __name__ == "__main__":
    main()
