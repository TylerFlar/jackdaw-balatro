"""Run ``jackdaw serve`` in a subprocess for tests."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def jackdaw_serve(*args: str) -> Iterator[int]:
    """Start ``jackdaw serve --port 0 *args`` and yield the port it bound."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "jackdaw.cli.main", "serve", "--port", "0", *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"jackdaw serve did not start: {proc.stderr.read()}")  # type: ignore[union-attr]
        yield int(line.rsplit(":", 1)[1])
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
