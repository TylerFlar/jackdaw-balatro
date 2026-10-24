"""The vendored BalatroBot sources must match their pinned manifest byte for byte.

``third_party/balatrobot/v1.5.2/`` is the authority for the BalatroBot 1:1 layer, so it must
never drift: every file listed in ``MANIFEST.sha256`` has to exist with the listed SHA-256, and
nothing else may sit next to it. See ``VENDORED.md`` in that directory. Stdlib only.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "third_party" / "balatrobot"
VENDOR_DIR = VENDOR_ROOT / "v1.5.2"
MANIFEST = VENDOR_DIR / "MANIFEST.sha256"
PINNED_COMMIT = "9052d76f14723293f6c6b2cecaa791a5c4ae68f3"

# Files that describe the vendored tree rather than belonging to it.
SELF_FILES = {"VENDORED.md", "MANIFEST.sha256"}
# Byproducts of running Python in place; the root .gitignore keeps them out of git.
IGNORED_DIRS = {"__pycache__", ".pytest_cache"}

# sha256sum format: "<hex>  <path>" (text mode) or "<hex> *<path>" (binary mode).
_LINE = re.compile(r"^(?P<digest>[0-9a-f]{64}) [ *](?P<path>\S.*)$")


def _manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for lineno, line in enumerate(MANIFEST.read_bytes().decode("ascii").split("\n"), 1):
        if not line:
            continue
        match = _LINE.match(line)
        assert match, f"MANIFEST.sha256:{lineno}: not in sha256sum format: {line!r}"
        path = match["path"]
        assert path not in entries, f"MANIFEST.sha256:{lineno}: duplicate entry {path}"
        entries[path] = match["digest"]
    return entries


def _vendored_files() -> set[str]:
    found: set[str] = set()
    for path in VENDOR_DIR.rglob("*"):
        rel = path.relative_to(VENDOR_DIR)
        if any(part in IGNORED_DIRS for part in rel.parts) or not path.is_file():
            continue
        if rel.as_posix() not in SELF_FILES:
            found.add(rel.as_posix())
    return found


def test_manifest_is_well_formed_and_sorted():
    raw = MANIFEST.read_bytes()
    assert b"\r" not in raw, "MANIFEST.sha256 must use LF line endings"
    assert raw.endswith(b"\n"), "MANIFEST.sha256 must end with a newline"
    paths = list(_manifest())
    assert paths, "MANIFEST.sha256 is empty"
    assert paths == sorted(paths), "MANIFEST.sha256 entries must be sorted by path"
    for path in paths:
        assert not path.startswith(("/", "./")) and ".." not in path.split("/"), path
        assert path not in SELF_FILES, f"{path} must not list itself"


def test_every_manifest_entry_exists_with_its_hash():
    missing: list[str] = []
    changed: list[str] = []
    for path, expected in _manifest().items():
        target = VENDOR_DIR / path
        if not target.is_file():
            missing.append(path)
        elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            changed.append(path)
    assert not missing, f"vendored files missing: {missing}"
    assert not changed, (
        f"vendored files differ from the pinned upstream copy: {changed}. They must not be "
        "edited; put patches in jackdaw's own code (see VENDORED.md)."
    )


def test_no_unlisted_vendored_files():
    unlisted = sorted(_vendored_files() - set(_manifest()))
    assert not unlisted, f"files not in MANIFEST.sha256: {unlisted}"


def test_vendored_md_pins_the_commit():
    text = (VENDOR_DIR / "VENDORED.md").read_text(encoding="utf-8")
    assert PINNED_COMMIT in text
    assert "v1.5.2" in text


def test_line_endings_are_never_converted():
    attributes = (VENDOR_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "* -text" in [line.strip() for line in attributes]
