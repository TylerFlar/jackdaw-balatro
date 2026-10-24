"""Check that the release version is the same everywhere.

    uv run python scripts/check_version.py                 # the manifests and CHANGELOG.md
    uv run python scripts/check_version.py --tag v0.2.0    # and a release tag
    uv run python scripts/check_version.py --notes         # print the version's changelog section

The version lives in two manifests: ``pyproject.toml`` (the Python distribution) and
``Cargo.toml``'s ``[workspace.package]`` (the crates, and ``jackdaw._core.__version__``), and
CHANGELOG.md needs a ``## [X.Y.Z]`` section for it. CI runs this; the Wheels workflow also
checks the tag it builds. Exits 1 on a mismatch.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest_versions() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return {
        "pyproject.toml": pyproject["project"]["version"],
        "Cargo.toml": cargo["workspace"]["package"]["version"],
    }


def changelog_section(version: str) -> str | None:
    """The body of CHANGELOG.md's section for ``version``, or None."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n", re.MULTILINE)
    found = heading.search(text)
    if found is None:
        return None
    rest = text[found.end() :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    return (rest[: following.start()] if following else rest).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", help="a release tag (vX.Y.Z) that must name the version")
    p.add_argument("--notes", action="store_true", help="print the version's changelog section")
    args = p.parse_args(argv)

    found = manifest_versions()
    version = found["pyproject.toml"]
    errors = []
    if len(set(found.values())) != 1:
        errors.append("the manifests disagree: " + ", ".join(f"{k} {v}" for k, v in found.items()))
    notes = changelog_section(version)
    if notes is None:
        errors.append(f"CHANGELOG.md has no '## [{version}]' section")
    if args.tag is not None and args.tag != f"v{version}":
        errors.append(f"the tag {args.tag} doesn't name the version {version} (v{version})")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(notes if args.notes else f"version {version}", end="" if args.notes else "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
