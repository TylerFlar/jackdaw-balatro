"""Balatro-compatible PRNG system.

Replicates the three-layer RNG architecture from the Balatro source:
  1. pseudohash(str) — deterministic string → float hash
  2. PseudoRandom.seed(key) — stateful float-domain LCG per named stream
  3. PseudoRandom.random() — bridges to uniform random output

The original Lua code lives in misc_functions.lua (lines 206-320).
See docs/source-map/rng-system.md for the full specification.
"""

from __future__ import annotations

import math
import random as _random_mod  # aliased to avoid shadowing by method names

# ---------------------------------------------------------------------------
# Module-level pure function — used standalone and by PseudoRandom
# ---------------------------------------------------------------------------

def pseudohash(s: str) -> float:
    """Hash a string to a float in [0, 1).

    Iterates bytes in reverse, accumulating via::

        num = ((1.1239285023 / num) * byte * pi + pi * i) % 1

    The index ``i`` is 1-based (matching Lua's string.byte indexing),
    running from ``len(s)`` down to ``1``.

    For an empty string the loop body never executes and the result is 1.0.
    """
    num = 1.0
    # Lua: for i=#str, 1, -1 do ... string.byte(str, i)
    # i is 1-based, so for Python's 0-based indexing: byte = ord(s[i-1])
    for i in range(len(s), 0, -1):
        byte = ord(s[i - 1])
        num = ((1.1239285023 / num) * byte * math.pi + math.pi * i) % 1
    return num


def _truncate_13(x: float) -> float:
    """Replicate Lua's ``tonumber(string.format("%.13f", x))``.

    Truncates (rounds) to 13 decimal places, then converts back to float.
    This prevents floating-point drift and is essential for cross-platform
    determinism between Lua and Python.
    """
    return float(f"{x:.13f}")


# ---------------------------------------------------------------------------
# Internal RNG used by PseudoRandom.random / .element / .shuffle
# ---------------------------------------------------------------------------
# We use a *dedicated* Random instance so we never disturb (or get disturbed
# by) the module-level ``random`` state that other code might rely on.

_rng = _random_mod.Random()


# ---------------------------------------------------------------------------
# PseudoRandom — the stateful PRNG matching G.GAME.pseudorandom
# ---------------------------------------------------------------------------

class PseudoRandom:
    """Balatro-compatible stateful PRNG.

    Holds the per-stream float state that in the Lua source lives on
    ``G.GAME.pseudorandom``.  Exposes the four operations the game uses:

    * :meth:`seed` — advance a named stream (``pseudoseed``)
    * :meth:`random` — draw a uniform value seeded by a stream (``pseudorandom``)
    * :meth:`element` — pick from a collection (``pseudorandom_element``)
    * :meth:`shuffle` — Fisher-Yates shuffle (``pseudoshuffle``)

    Plus a stateless :meth:`predict_seed` for preview/look-ahead without
    mutating state (used by ``get_first_legendary`` in the source).
    """

    # The two reserved keys that are NOT stream counters
    _RESERVED = frozenset({"seed", "hashed_seed"})

    def __init__(self, seed_str: str) -> None:
        self._state: dict[str, float] = {
            "seed": seed_str,
            "hashed_seed": pseudohash(seed_str),
        }

    # -- accessors ----------------------------------------------------------

    @property
    def seed_str(self) -> str:
        return self._state["seed"]

    @property
    def hashed_seed(self) -> float:
        return self._state["hashed_seed"]

    @property
    def state(self) -> dict[str, float]:
        """Direct access to the underlying state dict (for save/load)."""
        return self._state

    # -- core: pseudoseed ---------------------------------------------------

    def seed(self, key: str) -> float:
        """Advance stream *key* and return a coupled float.

        Equivalent to Lua ``pseudoseed(key)`` (misc_functions.lua:298).

        On the first call for a given *key*, initialises the stream from
        ``pseudohash(key + seed_string)``.  Each call then advances via a
        float-domain LCG::

            state[key] = abs(truncate_13( (2.134453429141 + state[key] * 1.72431234) % 1 ))

        and returns ``(state[key] + hashed_seed) / 2``.
        """
        st = self._state

        # Lazy init
        if key not in st:
            st[key] = pseudohash(key + st["seed"])

        # Advance
        raw = (2.134453429141 + st[key] * 1.72431234) % 1
        st[key] = abs(_truncate_13(raw))

        return (st[key] + st["hashed_seed"]) / 2

    # -- predict_seed (stateless) -------------------------------------------

    def predict_seed(self, key: str, predict_seed_str: str) -> float:
        """Stateless seed preview — does NOT mutate any stream.

        Equivalent to Lua ``pseudoseed(key, predict_seed)``
        (misc_functions.lua:302-307).  Used by ``get_first_legendary``
        to test what a hypothetical seed would produce.
        """
        _pseed = pseudohash(key + predict_seed_str)
        _pseed = abs(_truncate_13((2.134453429141 + _pseed * 1.72431234) % 1))
        return (_pseed + pseudohash(predict_seed_str)) / 2

    # -- pseudorandom -------------------------------------------------------

    def random(
        self,
        key: str | float,
        min_val: int | None = None,
        max_val: int | None = None,
    ) -> float | int:
        """Generate a uniform pseudorandom number.

        Equivalent to Lua ``pseudorandom(seed, min, max)``
        (misc_functions.lua:315).

        If *key* is a string, :meth:`seed` is called first to advance
        that stream and obtain a numeric seed.  The numeric seed is then
        used to initialise a one-shot RNG from which the output is drawn.

        Returns a float in [0, 1) when no range is given, or an int
        in [min_val, max_val] inclusive otherwise.
        """
        if isinstance(key, str):
            numeric_seed = self.seed(key)
        else:
            numeric_seed = key

        _rng.seed(numeric_seed)

        if min_val is not None and max_val is not None:
            return _rng.randint(min_val, max_val)
        return _rng.random()

    # -- pseudorandom_element -----------------------------------------------

    def element(self, table: dict | list, seed_val: float) -> tuple:
        """Select a random element from *table*.

        Equivalent to Lua ``pseudorandom_element(_t, seed)``
        (misc_functions.lua:253).

        The collection is first sorted deterministically — by ``sort_id``
        attribute if the values are dicts/objects with one, otherwise by
        key — to compensate for non-deterministic iteration order.

        Returns ``(selected_value, selected_key)``.
        """
        _rng.seed(seed_val)

        # Build (key, value) pairs
        if isinstance(table, dict):
            entries = [(k, v) for k, v in table.items()]
        else:
            # Lists: key is the 1-based index (matching Lua)
            entries = [(i + 1, v) for i, v in enumerate(table)]

        if not entries:
            raise ValueError("cannot select from empty table")

        # Deterministic sort
        first_val = entries[0][1]
        if isinstance(first_val, dict) and "sort_id" in first_val:
            entries.sort(key=lambda e: e[1]["sort_id"])
        elif hasattr(first_val, "sort_id"):
            entries.sort(key=lambda e: e[1].sort_id)
        else:
            entries.sort(key=lambda e: (isinstance(e[0], str), e[0]))

        idx = _rng.randint(0, len(entries) - 1)
        k, v = entries[idx]
        return v, k

    # -- pseudoshuffle ------------------------------------------------------

    def shuffle(self, lst: list, seed_val: float) -> None:
        """Fisher-Yates shuffle *lst* in place.

        Equivalent to Lua ``pseudoshuffle(list, seed)``
        (misc_functions.lua:206).

        Pre-sorts by ``sort_id`` (if elements have one) to ensure a
        deterministic starting order, then applies the standard backward
        Fisher-Yates.
        """
        _rng.seed(seed_val)

        # Pre-sort by sort_id if available
        if lst and hasattr(lst[0], "sort_id"):
            lst.sort(key=lambda x: getattr(x, "sort_id", 1))
        elif lst and isinstance(lst[0], dict) and "sort_id" in lst[0]:
            lst.sort(key=lambda x: x.get("sort_id", 1))

        # Backward Fisher-Yates (matches Lua: for i = #list, 2, -1)
        for i in range(len(lst) - 1, 0, -1):
            j = _rng.randint(0, i)  # Lua: math.random(i) returns 1..i
            lst[i], lst[j] = lst[j], lst[i]

    # -- state management ---------------------------------------------------

    def get_state(self) -> dict[str, float]:
        """Return a shallow copy of the state dict (for serialisation)."""
        return dict(self._state)

    def load_state(self, state: dict[str, float]) -> None:
        """Replace internal state from a save file.

        Any stream value that is ``0`` is re-hashed (matching the Lua
        save/load convention in ``game.lua:2167``).
        """
        self._state = dict(state)
        for k, v in self._state.items():
            if k not in self._RESERVED and v == 0:
                self._state[k] = pseudohash(k + self._state["seed"])


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (backward compat with existing tests)
# ---------------------------------------------------------------------------

def pseudoseed(key: str, state: dict[str, float]) -> float:
    """Advance stream *key* in *state* dict — functional API.

    Thin wrapper matching the original Lua ``pseudoseed(key)`` signature.
    Mutates *state* in place.
    """
    # Lazy init
    if key not in state:
        state[key] = pseudohash(key + (state.get("seed") or ""))

    raw = (2.134453429141 + state[key] * 1.72431234) % 1
    state[key] = abs(_truncate_13(raw))

    return (state[key] + (state.get("hashed_seed") or 0)) / 2


def pseudorandom(
    seed: float | str,
    state: dict[str, float] | None = None,
    min_val: int | None = None,
    max_val: int | None = None,
) -> float | int:
    """Functional API matching Lua ``pseudorandom(seed, min, max)``."""
    if isinstance(seed, str):
        if state is None:
            raise ValueError("state dict required when seed is a string key")
        numeric = pseudoseed(seed, state)
    else:
        numeric = seed

    _rng.seed(numeric)
    if min_val is not None and max_val is not None:
        return _rng.randint(min_val, max_val)
    return _rng.random()


def pseudorandom_element(table: dict | list, seed: float) -> tuple:
    """Functional API matching Lua ``pseudorandom_element(_t, seed)``."""
    _rng.seed(seed)

    if isinstance(table, dict):
        entries = [(k, v) for k, v in table.items()]
    else:
        entries = [(i + 1, v) for i, v in enumerate(table)]

    if not entries:
        raise ValueError("cannot select from empty table")

    first_val = entries[0][1]
    if isinstance(first_val, dict) and "sort_id" in first_val:
        entries.sort(key=lambda e: e[1]["sort_id"])
    elif hasattr(first_val, "sort_id"):
        entries.sort(key=lambda e: e[1].sort_id)
    else:
        entries.sort(key=lambda e: (isinstance(e[0], str), e[0]))

    idx = _rng.randint(0, len(entries) - 1)
    k, v = entries[idx]
    return v, k


def pseudoshuffle(lst: list, seed: float) -> None:
    """Functional API matching Lua ``pseudoshuffle(list, seed)``."""
    _rng.seed(seed)

    if lst and hasattr(lst[0], "sort_id"):
        lst.sort(key=lambda x: getattr(x, "sort_id", 1))
    elif lst and isinstance(lst[0], dict) and "sort_id" in lst[0]:
        lst.sort(key=lambda x: x.get("sort_id", 1))

    for i in range(len(lst) - 1, 0, -1):
        j = _rng.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]


def generate_starting_seed(entropy: float = 0.0) -> str:
    """Generate an 8-character alphanumeric seed string.

    Characters: digits 1-9, letters A-N, P-Z (no 0 or O).
    Matches Lua ``random_string`` (misc_functions.lua:270).
    """
    _rng.seed(entropy)
    chars: list[str] = []
    for _ in range(8):
        if _rng.random() > 0.7:
            # digit 1-9
            chars.append(chr(_rng.randint(ord("1"), ord("9"))))
        elif _rng.random() > 0.45:
            # letter A-N
            chars.append(chr(_rng.randint(ord("A"), ord("N"))))
        else:
            # letter P-Z
            chars.append(chr(_rng.randint(ord("P"), ord("Z"))))
    return "".join(chars).upper()
