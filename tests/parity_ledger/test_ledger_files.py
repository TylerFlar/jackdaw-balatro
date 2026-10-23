"""Offline consistency checks for the parity ledger (``docs/parity/``).

These run without the Lua source or luaparser, so they hold in CI. Whether the ledger is in sync
with ``balatro_source/`` is checked locally by ``scripts/parity_ledger.py check``. The parsing
here is deliberately independent of that script.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = ROOT / "docs" / "parity"
LEDGER_DIR = PARITY_DIR / "ledger"

COLUMNS = ("id", "lua", "kind", "subject", "context", "status", "rust", "test", "gap", "notes")
GAP_COLUMNS = ("id", "area", "summary", "lua", "python fix", "rust", "test", "status")
KIND_KEYS = {
    "function": "fn",
    "branch": "branch",
    "rng": "rng",
    "context_call": "ctx",
    "event": "event",
    "entity": "entity",
}
STATUSES = ("unreviewed", "ported", "verified", "gap", "smods-diff", "n/a-ui", "n/a-meta")
GAP_STATUSES = {"open", "in-progress", "python-fixed", "fixed", "wontfix"}
SCOPE_FILES = (
    "card.lua",
    "blind.lua",
    "back.lua",
    "tag.lua",
    "cardarea.lua",
    "challenges.lua",
    "game.lua",
    "state_events.lua",
    "common_events.lua",
    "misc_functions.lua",
    "button_callbacks.lua",
    "UI_definitions.lua",
)
SUMMARY_BEGIN = "<!-- BEGIN GENERATED SUMMARY"
SUMMARY_END = "<!-- END GENERATED SUMMARY -->"
QUOTE_LIMIT = 100  # the ledger may quote at most 100 characters of Lua in one place

_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_CODE_SPAN = re.compile(r"``\s(.*?)\s``|`([^`]*)`")
_LUA_REF = re.compile(r"^(?P<file>[A-Za-z_]+\.lua):(?P<start>\d+)(?:-(?P<end>\d+))?$")
_GAP_ID = re.compile(r"^[GB]\d{2}$")


def split_cells(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") and not inner.endswith("\\|") else inner
    return [cell.strip().replace("\\|", "|") for cell in _CELL_SPLIT.split(inner)]


def tables(text: str, header: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """(section heading, cells) for every data row of every table with this header."""
    rows: list[tuple[str, list[str]]] = []
    section, in_table = "", False
    for line in text.splitlines():
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        if not line.lstrip().startswith("|"):
            in_table = False
            continue
        cells = split_cells(line)
        if tuple(cells) == header:
            in_table = True
        elif in_table and not all(set(cell) <= set("-: ") for cell in cells):
            assert len(cells) == len(header), f"malformed table row: {line[:120]}"
            rows.append((section, cells))
    return rows


@cache
def ledgers() -> dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]]:
    """Scope file name -> (rows, stale rows)."""
    out = {}
    for path in sorted(LEDGER_DIR.glob("*.md")):
        main: list[dict[str, str]] = []
        stale: list[dict[str, str]] = []
        for section, cells in tables(path.read_text(encoding="utf-8"), COLUMNS):
            target = stale if section.lower().startswith("stale rows") else main
            target.append(dict(zip(COLUMNS, cells, strict=True)))
        out[path.name.removesuffix(".md")] = (main, stale)
    return out


@cache
def gaps() -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    text = (PARITY_DIR / "gaps.md").read_text(encoding="utf-8")
    for _, cells in tables(text, GAP_COLUMNS):
        row = dict(zip(GAP_COLUMNS, cells, strict=True))
        assert row["id"] not in found, f"gaps.md: duplicate id {row['id']}"
        found[row["id"]] = row
    return found


def all_rows() -> list[tuple[str, dict[str, str]]]:
    return [(name, row) for name, (main, _) in ledgers().items() for row in main]


def gap_refs(cell: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", cell) if part]


def test_every_scope_file_has_a_ledger():
    assert set(ledgers()) == set(SCOPE_FILES)
    for name, (main, _) in ledgers().items():
        assert main, f"{name}.md has no rows"


def test_ids_are_unique():
    counts = Counter(row["id"] for _, row in all_rows())
    duplicates = sorted(row_id for row_id, n in counts.items() if n > 1)
    assert not duplicates, f"duplicate ids: {duplicates[:10]}"


def test_rows_have_valid_generated_fields():
    problems = []
    for name, row in all_rows():
        stem = name.removesuffix(".lua")
        kind_key = KIND_KEYS.get(row["kind"])
        if kind_key is None:
            problems.append(f"{row['id']}: unknown kind {row['kind']!r}")
        elif not row["id"].startswith(f"{stem}/{kind_key}/"):
            problems.append(f"{row['id']}: id does not start with {stem}/{kind_key}/")
        ref = _LUA_REF.match(row["lua"])
        if not ref or ref["file"] != name:
            problems.append(f"{row['id']}: lua ref {row['lua']!r} is not {name}:<line>[-<line>]")
        elif ref["end"] and int(ref["end"]) < int(ref["start"]):
            problems.append(f"{row['id']}: lua range {row['lua']} runs backwards")
        if not row["subject"]:
            problems.append(f"{row['id']}: empty subject")
    assert not problems, "\n".join(problems[:20])


def test_statuses_are_valid():
    bad = [
        f"{row['id']}: {row['status']!r}" for _, row in all_rows() if row["status"] not in STATUSES
    ]
    assert not bad, f"invalid statuses (allowed: {', '.join(STATUSES)}): {bad[:10]}"


def test_no_stale_rows():
    stale = [f"{name}: {row['id']}" for name, (_, rows) in ledgers().items() for row in rows]
    assert not stale, f"resolve the stale rows (see docs/parity/README.md): {stale[:10]}"


def test_gap_rows_reference_known_gaps():
    known = set(gaps())
    problems = []
    for _, row in all_rows():
        refs = gap_refs(row["gap"])
        if row["status"] == "gap" and not refs:
            problems.append(f"{row['id']}: status gap without a gap id")
        problems += [f"{row['id']}: unknown gap {ref}" for ref in refs if ref not in known]
    assert not problems, "\n".join(problems[:20])


def test_ported_and_verified_rows_name_their_rust_and_test():
    problems = []
    for _, row in all_rows():
        if row["status"] in ("ported", "verified") and not row["rust"]:
            problems.append(f"{row['id']}: {row['status']} without a rust location")
        if row["status"] == "verified" and not row["test"]:
            problems.append(f"{row['id']}: verified without a test id")
    assert not problems, "\n".join(problems[:20])


def test_gap_register_is_well_formed():
    register = gaps()
    for gap_id, row in register.items():
        assert _GAP_ID.match(gap_id), f"gaps.md: bad id {gap_id!r}"
        assert row["status"] in GAP_STATUSES, f"gaps.md: {gap_id} has status {row['status']!r}"
        assert row["summary"], f"gaps.md: {gap_id} has no summary"
    expected = {f"G{n:02d}" for n in range(1, 52)}
    assert expected <= set(register), f"missing gaps: {sorted(expected - set(register))}"


def test_every_planning_gap_is_linked_from_the_ledger():
    linked = {ref for _, row in all_rows() for ref in gap_refs(row["gap"])}
    unlinked = sorted(
        gap_id for gap_id in gaps() if gap_id.startswith("G") and gap_id not in linked
    )
    assert not unlinked, f"gaps with no ledger row: {unlinked}"


def test_summary_matches_rows():
    readme = (PARITY_DIR / "README.md").read_text(encoding="utf-8")
    assert SUMMARY_BEGIN in readme and SUMMARY_END in readme, "README.md lacks the summary block"
    block = readme.split(SUMMARY_BEGIN, 1)[1].split(SUMMARY_END, 1)[0]
    header = ("file", "kind", "rows", *STATUSES)
    reported = {(c[0], c[1]): [int(n) for n in c[2:]] for _, c in tables(block, header)}
    assert reported, "the summary block has no table"

    expected: dict[tuple[str, str], list[int]] = {}
    totals = [0] * (1 + len(STATUSES))
    for name, (main, _) in ledgers().items():
        for kind in KIND_KEYS:
            rows = [row for row in main if row["kind"] == kind]
            if rows:
                statuses = Counter(row["status"] for row in rows)
                counts = [len(rows), *(statuses[s] for s in STATUSES)]
                expected[(name, kind)] = counts
                totals = [a + b for a, b in zip(totals, counts, strict=True)]
    expected[("all", "all")] = totals
    assert reported == expected, "summary is stale; run scripts/parity_ledger.py generate"


@pytest.mark.parametrize("column", ["subject", "context"])
def test_quoted_lua_stays_short(column: str):
    """The Lua source is proprietary: quote identifiers and short conditions, never code."""
    long_quotes = []
    for _, row in all_rows():
        for match in _CODE_SPAN.finditer(row[column]):
            quote = match.group(1) or match.group(2) or ""
            if len(quote) > QUOTE_LIMIT:
                long_quotes.append(f"{row['id']}: {len(quote)} chars")
    assert not long_quotes, "\n".join(long_quotes[:20])
