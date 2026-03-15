"""Balatro-compatible PRNG system.

Replicates the three-layer RNG architecture from the Balatro source:
  1. pseudohash(str) — deterministic string → float hash
  2. PseudoRandom.seed(key) — stateful float-domain LCG per named stream
  3. PseudoRandom.random() — bridges to uniform random output via LuaJIT's TW223

The original Lua code lives in misc_functions.lua (lines 206-320).
The secondary PRNG (TW223) replicates LuaJIT's math.randomseed/math.random
from lj_prng.c — a combined Tausworthe generator with period 2^223.

See docs/source-map/rng-system.md for the full specification.
"""

from __future__ import annotations

import math
import struct

_MASK64 = 0xFFFF_FFFF_FFFF_FFFF

# Pre-compiled struct formatters (avoids repeated format string parsing)
_PACK_D = struct.Struct("<d")
_PACK_Q = struct.Struct("<Q")


# ---------------------------------------------------------------------------
# LuaJIT TW223 PRNG — replicates lj_prng.c
# ---------------------------------------------------------------------------
# Combined Tausworthe generator (4 × 64-bit state words).
# Parameters from LuaJIT v2.1 src/lj_prng.c TW223_GEN macro.
# Unrolled for performance (called millions of times during RL training).

# Pre-computed high masks for each generator
_HMASK0 = (_MASK64 << 1) & _MASK64   # 64-63 = 1
_HMASK1 = (_MASK64 << 6) & _MASK64   # 64-58 = 6
_HMASK2 = (_MASK64 << 9) & _MASK64   # 64-55 = 9
_HMASK3 = (_MASK64 << 17) & _MASK64  # 64-47 = 17

# Min-bit thresholds (from 0x11090601 constant)
_MIN0 = 1 << 1    # 2
_MIN1 = 1 << 6    # 64
_MIN2 = 1 << 9    # 512
_MIN3 = 1 << 17   # 131072

# Seeding constants
_PI = 3.14159265358979323846
_E = 2.7182818284590452354


def _tw223_step(state: list[int]) -> int:
    """Advance the TW223 state and return a raw 64-bit result.

    Replicates the ``TW223_STEP`` macro in ``lj_prng.c``.
    Unrolled for ~30% speedup over the loop version.
    """
    M = _MASK64

    z = state[0]
    z = ((((z << 31) & M) ^ z) >> 45) ^ (((z & _HMASK0) << 18) & M)
    r = z
    state[0] = z

    z = state[1]
    z = ((((z << 19) & M) ^ z) >> 30) ^ (((z & _HMASK1) << 28) & M)
    r ^= z
    state[1] = z

    z = state[2]
    z = ((((z << 24) & M) ^ z) >> 48) ^ (((z & _HMASK2) << 7) & M)
    r ^= z
    state[2] = z

    z = state[3]
    z = ((((z << 21) & M) ^ z) >> 39) ^ (((z & _HMASK3) << 8) & M)
    r ^= z
    state[3] = z

    return r & M


def _luajit_seed(seed_double: float) -> list[int]:
    """Initialise a TW223 state from a double, matching ``random_seed()`` in lib_math.c.

    The double is iteratively transformed (``d = d*pi + e``), reinterpreted
    as uint64 via IEEE 754 bit pattern, and stored into 4 state words.
    A minimum-bit threshold is applied per word, then 10 warm-up steps are run.
    """
    pack_d = _PACK_D.pack
    unpack_q = _PACK_Q.unpack

    d = seed_double

    d = d * _PI + _E
    u0 = unpack_q(pack_d(d))[0]
    if u0 < _MIN0:
        u0 += _MIN0

    d = d * _PI + _E
    u1 = unpack_q(pack_d(d))[0]
    if u1 < _MIN1:
        u1 += _MIN1

    d = d * _PI + _E
    u2 = unpack_q(pack_d(d))[0]
    if u2 < _MIN2:
        u2 += _MIN2

    d = d * _PI + _E
    u3 = unpack_q(pack_d(d))[0]
    if u3 < _MIN3:
        u3 += _MIN3

    state = [u0, u1, u2, u3]

    # 10 warm-up iterations (lj_prng_u64, discards output)
    for _ in range(10):
        _tw223_step(state)
    return state


def _luajit_random(state: list[int]) -> float:
    """Draw a float in [0, 1) from TW223 state, matching ``lj_prng_u64d`` + lib_math.c."""
    r = _tw223_step(state)
    # Mask to 52-bit mantissa, set exponent for [1.0, 2.0)
    u64 = (r & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000
    d = _PACK_D.unpack(_PACK_Q.pack(u64))[0]
    return d - 1.0


def _luajit_random_int(state: list[int], a: int, b: int) -> int:
    """Draw an int in [a, b] from TW223 state, matching LuaJIT ``math.random(a, b)``."""
    d = _luajit_random(state)
    return int(math.floor(d * (b - a + 1))) + a


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
    for i in range(len(s), 0, -1):
        byte = ord(s[i - 1])
        num = ((1.1239285023 / num) * byte * math.pi + math.pi * i) % 1
    return num


def _truncate_13(x: float) -> float:
    """Replicate Lua's ``tonumber(string.format("%.13f", x))``.

    Rounds to 13 decimal places then converts back to float.  This prevents
    floating-point drift and is essential for cross-platform determinism.
    """
    return float(f"{x:.13f}")


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
        """
        st = self._state
        if key not in st:
            st[key] = pseudohash(key + st["seed"])
        raw = (2.134453429141 + st[key] * 1.72431234) % 1
        st[key] = abs(_truncate_13(raw))
        return (st[key] + st["hashed_seed"]) / 2

    # -- predict_seed (stateless) -------------------------------------------

    def predict_seed(self, key: str, predict_seed_str: str) -> float:
        """Stateless seed preview — does NOT mutate any stream."""
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
        that stream.  The resulting float seeds LuaJIT's TW223 PRNG,
        from which the output is drawn.

        Returns a float in [0, 1) when no range is given, or an int
        in [min_val, max_val] inclusive otherwise.
        """
        if isinstance(key, str):
            numeric_seed = self.seed(key)
        else:
            numeric_seed = key

        tw_state = _luajit_seed(numeric_seed)
        if min_val is not None and max_val is not None:
            return _luajit_random_int(tw_state, min_val, max_val)
        return _luajit_random(tw_state)

    # -- pseudorandom_element -----------------------------------------------

    def element(self, table: dict | list, seed_val: float) -> tuple:
        """Select a random element from *table*.

        Equivalent to Lua ``pseudorandom_element(_t, seed)``
        (misc_functions.lua:253).

        The collection is first sorted deterministically — by ``sort_id``
        if available, otherwise by key — then a random index is drawn.

        Returns ``(selected_value, selected_key)``.
        """
        tw_state = _luajit_seed(seed_val)

        if isinstance(table, dict):
            entries = list(table.items())
        else:
            entries = [(i + 1, v) for i, v in enumerate(table)]

        if not entries:
            raise ValueError("cannot select from empty table")

        # Deterministic sort (matches Lua pseudorandom_element)
        first_val = entries[0][1]
        if isinstance(first_val, dict) and "sort_id" in first_val:
            entries.sort(key=lambda e: e[1]["sort_id"])
        elif hasattr(first_val, "sort_id"):
            entries.sort(key=lambda e: e[1].sort_id)
        else:
            entries.sort(key=lambda e: (isinstance(e[0], str), e[0]))

        idx = _luajit_random_int(tw_state, 1, len(entries))
        k, v = entries[idx - 1]  # Lua is 1-based
        return v, k

    # -- pseudoshuffle ------------------------------------------------------

    def shuffle(self, lst: list, seed_val: float) -> None:
        """Fisher-Yates shuffle *lst* in place.

        Equivalent to Lua ``pseudoshuffle(list, seed)``
        (misc_functions.lua:206).
        """
        tw_state = _luajit_seed(seed_val)

        # Pre-sort by sort_id if available
        if lst and hasattr(lst[0], "sort_id"):
            lst.sort(key=lambda x: getattr(x, "sort_id", 1))
        elif lst and isinstance(lst[0], dict) and "sort_id" in lst[0]:
            lst.sort(key=lambda x: x.get("sort_id", 1))

        # Backward Fisher-Yates: for i = #list, 2, -1 do j = random(i)
        for i in range(len(lst) - 1, 0, -1):
            j = _luajit_random_int(tw_state, 1, i + 1) - 1  # convert 1-based to 0-based
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

    tw_state = _luajit_seed(numeric)
    if min_val is not None and max_val is not None:
        return _luajit_random_int(tw_state, min_val, max_val)
    return _luajit_random(tw_state)


def pseudorandom_element(table: dict | list, seed: float) -> tuple:
    """Functional API matching Lua ``pseudorandom_element(_t, seed)``."""
    tw_state = _luajit_seed(seed)

    if isinstance(table, dict):
        entries = list(table.items())
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

    idx = _luajit_random_int(tw_state, 1, len(entries))
    k, v = entries[idx - 1]
    return v, k


def pseudoshuffle(lst: list, seed: float) -> None:
    """Functional API matching Lua ``pseudoshuffle(list, seed)``."""
    tw_state = _luajit_seed(seed)

    if lst and hasattr(lst[0], "sort_id"):
        lst.sort(key=lambda x: getattr(x, "sort_id", 1))
    elif lst and isinstance(lst[0], dict) and "sort_id" in lst[0]:
        lst.sort(key=lambda x: x.get("sort_id", 1))

    for i in range(len(lst) - 1, 0, -1):
        j = _luajit_random_int(tw_state, 1, i + 1) - 1
        lst[i], lst[j] = lst[j], lst[i]


def generate_starting_seed(entropy: float = 0.0) -> str:
    """Generate an 8-character alphanumeric seed string.

    Characters: digits 1-9, letters A-N, P-Z (no 0 or O).
    Matches Lua ``random_string`` (misc_functions.lua:270).
    """
    tw_state = _luajit_seed(entropy)

    chars: list[str] = []
    for _ in range(8):
        if _luajit_random(tw_state) > 0.7:
            chars.append(chr(_luajit_random_int(tw_state, ord("1"), ord("9"))))
        elif _luajit_random(tw_state) > 0.45:
            chars.append(chr(_luajit_random_int(tw_state, ord("A"), ord("N"))))
        else:
            chars.append(chr(_luajit_random_int(tw_state, ord("P"), ord("Z"))))
    return "".join(chars).upper()
