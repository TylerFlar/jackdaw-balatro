"""Canonical, order-sensitive fingerprints of engine and env objects.

The golden-trajectory tests hash everything observable about a run (the
whole engine state dict, observations, masks, action tables, env info)
after every step.  Any behavioural change, including one that only moves an
RNG draw, changes a digest.

Encoding rules:

* Mutable objects (dicts, lists, sets, ndarrays, plain objects, non-frozen
  dataclasses) are identity-tracked: a second reference encodes as a
  back-reference to its first-visit index, so aliasing (the same Card in
  two areas) is part of the fingerprint and cycles terminate.
* Immutable values (tuples, frozensets, frozen dataclasses) are encoded by
  value, so sharing one cached instance is not a behaviour change.
* Object attributes are the union of dataclass fields, ``__slots__`` and
  the instance ``__dict__`` (runtime attributes such as ``Card.vampired``
  count), sorted by name: storage layout is not behaviour.
* Dict insertion order is kept (iteration order can be behaviour).
* Set members must be atomic values; they are sorted by encoding.
* Anything the encoder cannot see into raises ``TypeError`` rather than
  silently hashing only its type.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import types
from typing import Any

import numpy as np

_ATOMIC = (type(None), bool, int, float, str, bytes)
_CALLABLE_TYPES = (
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.BuiltinMethodType,
)


def _atomic_key(v: Any) -> str:
    """Encoding of a set member; only atomic (hashable, non-graph) values."""
    if isinstance(v, enum.Enum):
        return f"E:{type(v).__qualname__}.{v.name}"
    if isinstance(v, float):
        return f"f:{v.hex()}"
    if isinstance(v, _ATOMIC):
        return f"{type(v).__name__}:{v!r}"
    if isinstance(v, tuple):
        return "(" + ",".join(_atomic_key(x) for x in v) + ")"
    raise TypeError(f"golden fingerprint: unsupported set member {type(v).__qualname__}")


def _attr_names(cls: type) -> tuple[tuple[str, ...], bool]:
    """(declared field/slot names, instances may carry a __dict__)."""
    names: list[str] = []
    if dataclasses.is_dataclass(cls):
        names.extend(f.name for f in dataclasses.fields(cls))
    for klass in cls.__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot not in ("__dict__", "__weakref__"):
                names.append(slot)
    has_dict = any("__dict__" in k.__dict__ for k in cls.__mro__ if k is not object)
    return tuple(dict.fromkeys(names)), has_dict


_ATTR_CACHE: dict[type, tuple[tuple[str, ...], bool]] = {}


class Fingerprinter:
    """Accumulates a canonical encoding of Python objects; ``hexdigest`` hashes it."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._seen: dict[int, int] = {}
        self._keepalive: list[Any] = []

    def hexdigest(self) -> str:
        return hashlib.sha256("".join(self._parts).encode()).hexdigest()

    def _track(self, obj: Any) -> bool:
        """Record first visit of a mutable object; False if already seen."""
        oid = id(obj)
        seen = self._seen.get(oid)
        if seen is not None:
            self._parts.append(f"@{seen};")
            return False
        self._seen[oid] = len(self._seen)
        self._keepalive.append(obj)
        return True

    def add(self, obj: Any) -> None:
        w = self._parts.append
        t = type(obj)
        if t is str or t is int or t is bool or obj is None or t is bytes:
            w(f"{t.__name__}:{obj!r};")
            return
        if t is float:
            w(f"f:{obj.hex()};")
            return
        if isinstance(obj, enum.Enum):
            w(f"E:{t.__qualname__}.{obj.name};")
            return
        if isinstance(obj, np.generic):
            w(f"np:{obj.dtype}:{obj.item()!r};")
            return
        if isinstance(obj, (int, str, float)):  # subclasses (not enums)
            w(f"{t.__qualname__}:{obj!r};")
            return
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise TypeError("golden fingerprint: object-dtype ndarray")
            if not self._track(obj):
                return
            w(f"nd:{obj.dtype}:{obj.shape}:{np.ascontiguousarray(obj).tobytes().hex()};")
            return
        if isinstance(obj, _CALLABLE_TYPES):
            w(f"T:{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '?')};")
            return

        if t is tuple or isinstance(obj, tuple):
            w(f"({t.__qualname__}:{len(obj)}:")
            for v in obj:
                self.add(v)
            w(")")
            return
        if isinstance(obj, frozenset):
            w(f"<frozenset:{','.join(sorted(_atomic_key(v) for v in obj))}>")
            return

        if isinstance(obj, (dict, list, set)):
            if not self._track(obj):
                return
            if isinstance(obj, dict):
                w(f"{{{t.__qualname__}:{len(obj)}:")
                for k, v in obj.items():
                    self.add(k)
                    self.add(v)
                w("}")
            elif isinstance(obj, list):
                w(f"[{t.__qualname__}:{len(obj)}:")
                for v in obj:
                    self.add(v)
                w("]")
            else:
                w(f"<set:{','.join(sorted(_atomic_key(v) for v in obj))}>")
            return

        # Plain objects and dataclasses
        meta = _ATTR_CACHE.get(t)
        if meta is None:
            meta = _ATTR_CACHE[t] = _attr_names(t)
        names, has_dict = meta
        if not names and not has_dict:
            raise TypeError(f"golden fingerprint: opaque object {t.__module__}.{t.__qualname__}")
        frozen = dataclasses.is_dataclass(t) and t.__dataclass_params__.frozen  # type: ignore[attr-defined]
        if not frozen and not self._track(obj):
            return
        attrs: dict[str, Any] = {}
        for name in names:
            if hasattr(obj, name):
                attrs[name] = getattr(obj, name)
        if has_dict:
            attrs.update(vars(obj))
        w(f"O:{t.__module__}.{t.__qualname__}(")
        for k in sorted(attrs):
            w(k + "=")
            self.add(attrs[k])
        w(")")


def fingerprint(*objs: Any) -> str:
    fp = Fingerprinter()
    for o in objs:
        fp.add(o)
    return fp.hexdigest()
