"""Every error message BalatroBot's vendored endpoints can send.

Scans ``third_party/balatrobot/v1.5.2/src/lua/endpoints/**/*.lua`` for the
``message = <expr>`` fields of error responses and turns each expression into a
pattern: string literals match themselves, anything computed (``.. idx ..``)
matches any text, and ``string.format`` turns its format string's specifiers
into wildcards.  A message held in a variable (``message = msg``) stands for
every assignment to that variable before it, one site each (``buy.lua:96#2``).
The error-table gate (:mod:`tests.bot.test_error_table`) checks that a test row
produces every one of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINTS = ROOT / "third_party" / "balatrobot" / "v1.5.2" / "src" / "lua" / "endpoints"

_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"' + r"|'((?:[^'\\]|\\.)*)'")
_NAME = re.compile(r"name\s*=\s*BB_ERROR_NAMES\.(\w+)")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_FORMAT_SPEC = re.compile(r"%[-+ #0]*\d*(?:\.\d+)?[sdifgxXqc]|%%")


@dataclass(frozen=True)
class Site:
    """One message an endpoint's error response can carry."""

    file: str  # path relative to the endpoints directory
    line: int  # the ``message =`` line
    variant: int  # which assignment, for a message held in a variable (0 otherwise)
    expr: str  # the Lua expression, joined onto one line
    name: str  # the BB_ERROR_NAMES entry it is sent with
    pattern: re.Pattern[str]

    @property
    def id(self) -> str:
        return f"{self.file}:{self.line}" + (f"#{self.variant}" if self.variant else "")

    def matches(self, message: str) -> bool:
        return self.pattern.fullmatch(message) is not None


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m[1], m[1]), s)


def _literal(m: re.Match[str]) -> str:
    return _unescape(m[1] if m[1] is not None else m[2])


def _format_pattern(fmt: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _FORMAT_SPEC.finditer(fmt):
        out.append(re.escape(fmt[pos : m.start()]))
        out.append("%" if m[0] == "%%" else ".*")
        pos = m.end()
    out.append(re.escape(fmt[pos:]))
    return "".join(out)


def _pattern(expr: str) -> re.Pattern[str]:
    """Literals of a ``..`` concatenation as themselves, everything else as ``.*``;
    ``string.format(fmt, ...)`` as ``fmt`` with its specifiers as ``.*``."""
    expr = expr.strip().rstrip(",").strip()
    if expr.startswith("string.format("):
        first = _STRING.search(expr)
        assert first is not None, expr
        return re.compile(_format_pattern(_literal(first)), re.DOTALL)
    out: list[str] = []
    pos = 0
    for m in _STRING.finditer(expr):
        if expr[pos : m.start()].strip().strip(".").strip():
            out.append(".*")
        out.append(re.escape(_literal(m)))
        pos = m.end()
    if expr[pos:].strip().strip(".").strip() or not out:
        out.append(".*")
    text = "".join(out)
    while ".*.*" in text:
        text = text.replace(".*.*", ".*")
    return re.compile(text, re.DOTALL)


def _balanced(parts: list[str]) -> bool:
    text = _STRING.sub("", " ".join(parts))
    return text.count("(") <= text.count(")")


def _expression(lines: list[str], start: int, head: str) -> tuple[str, int]:
    """The expression after ``head`` on line ``start`` and the lines that continue it (a
    ``..`` chain, or an open ``string.format(``)."""
    parts = [lines[start].split(head, 1)[1].strip()]
    i = start + 1
    while i < len(lines):
        s = lines[i].strip()
        joined = " ".join(parts)
        continues = s.startswith("..") or joined.endswith("..") or not _balanced(parts)
        if not continues:
            break
        parts.append(s)
        i += 1
    return " ".join(p for p in parts if p).rstrip(","), i


def _assignments(lines: list[str], before: int, var: str) -> list[str]:
    """The expressions assigned to ``var`` in the block that ends at line ``before``."""
    found: list[str] = []
    head = re.compile(rf"^\s*(?:local\s+)?{re.escape(var)}\s*=\s*\S")
    i = before - 1
    while i >= 0 and not re.match(rf"^\s*local\s+{re.escape(var)}\s*$", lines[i]):
        if head.match(lines[i]):
            expr, _ = _expression(lines, i, "=")
            found.append(expr)
        i -= 1
    return list(reversed(found))


@cache
def sites() -> tuple[Site, ...]:
    found: list[Site] = []
    for path in sorted(ENDPOINTS.rglob("*.lua")):
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(ENDPOINTS).as_posix()
        for n, line in enumerate(lines):
            if not re.search(r"\bmessage\s*=", line) or line.strip().startswith("--"):
                continue
            expr, after = _expression(lines, n, "message =")
            name = ""
            for look in lines[after : after + 3]:
                m = _NAME.search(look)
                if m:
                    name = m[1]
                    break
            if _IDENT.fullmatch(expr):
                for k, assigned in enumerate(_assignments(lines, n, expr), 1):
                    found.append(Site(rel, n + 1, k, assigned, name, _pattern(assigned)))
            else:
                found.append(Site(rel, n + 1, 0, expr, name, _pattern(expr)))
    return tuple(found)


def sites_for(file: str, message: str) -> list[Site]:
    """The sites of ``file`` whose pattern matches ``message``."""
    return [s for s in sites() if s.file == file and s.matches(message)]


if __name__ == "__main__":
    for s in sites():
        print(f"{s.id:28} {s.name:15} {s.pattern.pattern}")
