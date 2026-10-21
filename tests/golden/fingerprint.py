"""Canonical, order-sensitive fingerprints of engine and env objects.

The golden-trajectory tests hash everything observable about a run (the
whole engine state dict, observations, masks, action tables) after every
step.  Any behavioural change, including one that only moves an RNG draw,
changes a digest.

Shared references to mutable objects are encoded by first-visit order, so
aliasing (the same Card in two areas) is part of the fingerprint and cycles
terminate.  Immutable values are encoded by value.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import types
from typing import Any

import numpy as np


def _is_mutable(obj: Any) -> bool:
    if isinstance(obj, (tuple, frozenset)):
        return False
    if dataclasses.is_dataclass(obj):
        return not type(obj).__dataclass_params__.frozen
    return True


class Fingerprinter:
    """Streams a canonical encoding of Python objects into a sha256."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self._seen: dict[int, int] = {}
        self._keepalive: list[Any] = []

    def hexdigest(self) -> str:
        return self._h.hexdigest()

    def _w(self, s: str) -> None:
        self._h.update(s.encode())

    def add(self, obj: Any) -> None:
        w = self._w
        if obj is None or isinstance(obj, (bool, int, str, bytes)):
            w(f"{type(obj).__name__}:{obj!r};")
            return
        if isinstance(obj, float):
            w(f"f:{obj.hex()};")
            return
        if isinstance(obj, enum.Enum):
            w(f"E:{type(obj).__qualname__}.{obj.name};")
            return
        if isinstance(obj, np.generic):
            w(f"np:{obj.dtype}:{obj.item()!r};")
            return
        if isinstance(obj, np.ndarray):
            w(f"nd:{obj.dtype}:{obj.shape}:")
            self._h.update(np.ascontiguousarray(obj).tobytes())
            w(";")
            return
        if isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
            w(f"T:{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', obj)!r};")
            return

        # Mutable containers / objects: track identity, since aliasing is
        # observable (mutating one alias changes the other) and cycles must
        # terminate.  Immutable values (tuples, frozensets, frozen
        # dataclasses) are encoded by value, so sharing one cached instance
        # is not a behaviour change.
        if _is_mutable(obj):
            oid = id(obj)
            if oid in self._seen:
                w(f"@{self._seen[oid]};")
                return
            self._seen[oid] = len(self._seen)
            self._keepalive.append(obj)

        if isinstance(obj, dict):
            w(f"{{{type(obj).__name__}:{len(obj)}:")
            for k, v in obj.items():
                self.add(k)
                self.add(v)
            w("}")
        elif isinstance(obj, (list, tuple)):
            w(f"[{type(obj).__name__}:{len(obj)}:")
            for v in obj:
                self.add(v)
            w("]")
        elif isinstance(obj, (set, frozenset)):
            # Set iteration order depends on PYTHONHASHSEED; hash members
            # independently and sort so the encoding is process-stable.
            parts = []
            for v in obj:
                sub = Fingerprinter()
                sub.add(v)
                parts.append(sub.hexdigest())
            w(f"<{type(obj).__name__}:{len(obj)}:{','.join(sorted(parts))}>")
        else:
            cls = type(obj)
            w(f"O:{cls.__module__}.{cls.__qualname__}(")
            # Attribute order is a storage detail (``__dict__`` vs
            # ``__slots__``), not behaviour, so attributes are sorted.
            attrs: dict[str, Any] = {}
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    attrs[f.name] = getattr(obj, f.name)
            else:
                for klass in cls.__mro__:
                    slots = getattr(klass, "__slots__", ())
                    for slot in (slots,) if isinstance(slots, str) else slots:
                        if slot not in ("__dict__", "__weakref__") and hasattr(obj, slot):
                            attrs[slot] = getattr(obj, slot)
                if hasattr(obj, "__dict__"):
                    attrs.update(vars(obj))
            for k in sorted(attrs):
                w(k + "=")
                self.add(attrs[k])
            w(")")


def fingerprint(*objs: Any) -> str:
    fp = Fingerprinter()
    for o in objs:
        fp.add(o)
    return fp.hexdigest()
