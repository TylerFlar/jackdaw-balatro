"""Smoke tests for the compiled ``jackdaw._core`` extension module."""

from __future__ import annotations

from importlib.metadata import version

import jackdaw._core as core


def test_version_matches_distribution() -> None:
    assert core.__version__ == version("jackdaw")


def test_build_info() -> None:
    info = core._build_info()
    assert info["version"] == core.__version__
    assert info["profile"] in {"debug", "release"}
    assert isinstance(info["debug_assertions"], bool)
    assert "-" in info["target"]
    assert info["rustc"].startswith("rustc ")
