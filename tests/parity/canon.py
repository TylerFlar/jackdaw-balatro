"""Canonical encoding, digest and diff for engine snapshots.

A snapshot (see ``docs/snapshot-schema.md``) is a tree of ``None``, ``bool``,
numbers, ``str``, ``list`` and ``dict`` (string keys).  Both engines must
produce byte-identical canonical encodings for equal game states, so the
encoding is defined here once and mirrored by the Rust implementation:

* ``None`` -> ``N``; ``True``/``False`` -> ``T``/``F``.
* Every number (``int`` or ``float``) is a Lua number, i.e. an IEEE-754
  double: ``D`` + 8 bytes big-endian.  ``-0.0`` encodes as ``+0.0`` and every
  NaN as ``0x7FF8000000000000`` (Lua cannot tell NaNs apart in gameplay).
* ``str`` -> ``S`` + u32 big-endian byte length + UTF-8 bytes.
* ``list`` -> ``[`` + u32 count + items.
* ``dict`` -> ``{`` + u32 count + (key, value) pairs with keys sorted by
  their UTF-8 bytes; keys are encoded as strings.

The digest is the SHA-256 hex digest of that encoding.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

_CANONICAL_NAN = struct.pack(">Q", 0x7FF8000000000000)


def _number(v: float | int) -> bytes:
    if isinstance(v, int):
        v = float(v)  # Lua numbers are doubles; ints above 2**53 round like Lua would
    if math.isnan(v):
        return b"D" + _CANONICAL_NAN
    if v == 0.0:
        v = 0.0  # -0.0 -> +0.0
    return b"D" + struct.pack(">d", v)


def encode(obj: Any) -> bytes:
    """Canonical byte encoding of a snapshot tree."""
    out: list[bytes] = []
    _encode(obj, out)
    return b"".join(out)


def _encode(obj: Any, out: list[bytes]) -> None:
    if obj is None:
        out.append(b"N")
    elif obj is True:
        out.append(b"T")
    elif obj is False:
        out.append(b"F")
    elif isinstance(obj, (int, float)):
        out.append(_number(obj))
    elif isinstance(obj, str):
        raw = obj.encode("utf-8")
        out.append(b"S" + struct.pack(">I", len(raw)) + raw)
    elif isinstance(obj, (list, tuple)):
        out.append(b"[" + struct.pack(">I", len(obj)))
        for item in obj:
            _encode(item, out)
    elif isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: _key_bytes(kv[0]))
        out.append(b"{" + struct.pack(">I", len(items)))
        for k, v in items:
            _encode(k, out)
            _encode(v, out)
    else:
        raise TypeError(f"snapshot: unsupported value {type(obj).__qualname__}: {obj!r}")


def _key_bytes(k: Any) -> bytes:
    if not isinstance(k, str):
        raise TypeError(f"snapshot: dict keys must be str, got {type(k).__qualname__}: {k!r}")
    return k.encode("utf-8")


def digest(obj: Any) -> str:
    """SHA-256 hex digest of the canonical encoding."""
    return hashlib.sha256(encode(obj)).hexdigest()


def chain(prev: str, item: str) -> str:
    """Fold one record digest into a running trajectory digest."""
    return hashlib.sha256((prev + ":" + item).encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Equality and diff (same semantics as the encoding)
# ---------------------------------------------------------------------------


def _num_eq(a: float | int, b: float | int) -> bool:
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return fa == fb


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def diff(a: Any, b: Any, path: str = "", limit: int = 50) -> list[str]:
    """Paths where two snapshot trees differ (under the canonical semantics)."""
    out: list[str] = []
    _diff(a, b, path or "$", out, limit)
    return out


def _short(v: Any) -> str:
    text = json.dumps(v, sort_keys=True, default=str)
    return text if len(text) <= 120 else text[:117] + "..."


def _diff(a: Any, b: Any, path: str, out: list[str], limit: int) -> None:
    if len(out) >= limit:
        return
    if _is_num(a) and _is_num(b):
        if not _num_eq(a, b):
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: <missing> != {_short(b[k])}")
            elif k not in b:
                out.append(f"{path}.{k}: {_short(a[k])} != <missing>")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out, limit)
            if len(out) >= limit:
                return
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _diff(x, y, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                return
        return
    if type(a) is not type(b) or a != b:
        out.append(f"{path}: {_short(a)} != {_short(b)}")


# ---------------------------------------------------------------------------
# JSON text form (non-finite numbers are objects so the text stays valid JSON)
# ---------------------------------------------------------------------------


def to_json_compatible(obj: Any) -> Any:
    """Replace non-finite floats with ``{"$f": "inf"|"-inf"|"nan"}``."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"$f": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {k: to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_compatible(v) for v in obj]
    return obj


def from_json_compatible(obj: Any) -> Any:
    """Inverse of :func:`to_json_compatible`."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "$f" in obj:
            return float(obj["$f"])
        return {k: from_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_json_compatible(v) for v in obj]
    return obj
