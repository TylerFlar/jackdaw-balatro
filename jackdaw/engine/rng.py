"""Balatro-compatible PRNG system — bit-exact with LuaJIT 2.1.

Three-Layer Architecture
========================

Balatro's RNG is a three-layer system.  Each layer feeds into the next:

**Layer 1 — pseudohash (string → float)**
    A custom hash that maps any string to a float in [0, 1).  Iterates the
    string's bytes in reverse with a nonlinear accumulator involving division,
    multiplication by π, and mod 1.  Used only for *initialisation*, never
    during gameplay random draws.  Source: ``misc_functions.lua:279``.

**Layer 2 — pseudoseed (stateful float LCG per named stream)**
    Each gameplay system (shop, scoring, blinds, etc.) has its own *named
    stream* stored as a float in ``G.GAME.pseudorandom[key]``.  On first
    access the stream is lazily initialised from ``pseudohash(key + seed)``.
    Each call advances the stream via a float-domain linear congruential
    generator::

        state[key] = abs(truncate_13( (2.134453429141 + state[key] * 1.72431234) % 1 ))
        return (state[key] + hashed_seed) / 2

    The ``truncate_13`` step (``tonumber(string.format("%.13f", x))`` in Lua)
    rounds to 13 decimal places to prevent floating-point drift.  Streams are
    independent — advancing ``'boss'`` never touches ``'shuffle'``.
    Source: ``misc_functions.lua:298``.

**Layer 3 — pseudorandom (LuaJIT TW223 secondary PRNG)**
    The float from Layer 2 is used to seed LuaJIT's TW223 combined Tausworthe
    generator (``lj_prng.c``), from which the actual uniform output is drawn.
    The game calls ``math.randomseed(seed_float)`` then ``math.random()`` on
    every random operation — effectively using TW223 as a one-shot
    seed-to-output function.  This module replicates TW223 exactly, including
    the IEEE 754 bit-pattern reinterpretation during seeding and the 10-step
    warm-up.  Source: ``misc_functions.lua:315``, ``lj_prng.c``.

Per-Stream Independence
=======================

The ~65 named streams (``'boss'``, ``'shuffle'``, ``'lucky_mult'``,
``'rarity'``, ``'cdt'``, etc.) each maintain their own float counter.
Advancing one stream never affects another.  Many keys include the current
ante as a suffix (e.g. ``'rarity1sho'``) so that the shop at ante 3 uses
different values than the shop at ante 5.

Seed Lifecycle
==============

1. **Run start**: seed string is set (player-entered, ``"TUTORIAL"``, or
   generated from cursor entropy via :func:`generate_starting_seed`).
2. ``hashed_seed = pseudohash(seed_string)`` — computed once, mixed into
   every Layer 2 output.
3. Streams are lazily initialised on first access:
   ``state[key] = pseudohash(key + seed_string)``.
4. The entire ``state`` dict is serialised on save and restored on load.
   Stream values of ``0`` are re-hashed (the save convention).

Known Non-Determinism
=====================

Three ``math.random()`` calls in the Lua source bypass the pseudoseed
system entirely, drawing from whatever state LuaJIT's global RNG happens
to be in.  These are **not** reproducible from the seed alone:

- ``tag.lua:211`` — Charm Tag: picks Mega Arcana pack variant 1 or 2
- ``tag.lua:226`` — Meteor Tag: picks Mega Celestial pack variant 1 or 2
- ``common_events.lua:1947`` — first Buffoon pack: picks variant 1 or 2

In practice these are often "accidentally deterministic" because
``pseudorandom`` re-seeds the global RNG on every call, but they are
**not guaranteed** to match across implementations.  The simulator
treats these as a known deviation and makes a deterministic choice
(always variant 1) or uses the pseudoseed system instead.

Public API
==========

.. autoclass:: PseudoRandom
.. autofunction:: pseudohash
.. autofunction:: generate_starting_seed
"""

from __future__ import annotations

import math
import struct

__all__ = [
    "PseudoRandom",
    "pseudohash",
    "generate_starting_seed",
    # Functional wrappers (for use without a PseudoRandom instance):
    "pseudoseed",
    "pseudorandom",
    "pseudorandom_element",
    "pseudoshuffle",
]

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
_HMASK0 = (_MASK64 << 1) & _MASK64  # 64-63 = 1
_HMASK1 = (_MASK64 << 6) & _MASK64  # 64-58 = 6
_HMASK2 = (_MASK64 << 9) & _MASK64  # 64-55 = 9
_HMASK3 = (_MASK64 << 17) & _MASK64  # 64-47 = 17

# Min-bit thresholds (from 0x11090601 constant)
_MIN0 = 1 << 1  # 2
_MIN1 = 1 << 6  # 64
_MIN2 = 1 << 9  # 512
_MIN3 = 1 << 17  # 131072

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


# Seeding runs 10 warm-up steps whose output is discarded, so only each
# word's own update matters.  Each word's update (shifts, masks, xors) is
# linear over GF(2), hence so is its 10-fold composition W: for any z,
# W(z) is the xor of W applied to z's individual bytes.  _WARMUP[w][k][v]
# holds W_w(v << 8k), turning the 40 big-int updates of the warm-up into
# 32 table lookups with bit-identical results.
_WARMUP_STEPS = 10


def _build_warmup_tables() -> tuple[tuple[tuple[int, ...], ...], ...]:
    tables = []
    for word in range(4):
        # W of each single bit, then every byte value by linearity.
        basis = []
        for bit in range(64):
            state = [0, 0, 0, 0]
            state[word] = 1 << bit
            for _ in range(_WARMUP_STEPS):
                _tw223_step(state)
            basis.append(state[word])
        per_byte = []
        for k in range(8):
            row = [0] * 256
            for v in range(1, 256):
                low = v & -v
                row[v] = row[v ^ low] ^ basis[8 * k + low.bit_length() - 1]
            per_byte.append(tuple(row))
        tables.append(tuple(per_byte))
    return tuple(tables)


_WARMUP = _build_warmup_tables()


def _luajit_seed(seed_double: float) -> list[int]:
    """Initialise a TW223 state from a double, matching ``random_seed()`` in lib_math.c.

    The double is iteratively transformed (``d = d*pi + e``), reinterpreted
    as uint64 via IEEE 754 bit pattern, and stored into 4 state words.
    A minimum-bit threshold is applied per word, then the 10 warm-up steps are
    applied through precomputed lookup tables (see ``_WARMUP``).
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

    # 10 warm-up iterations (lj_prng_u64, discards output); see _WARMUP.
    # Each u is below 2**64 (a <Q unpack, plus at most a small minimum), so
    # u >> 56 is already a byte.
    t0, t1, t2, t3 = _WARMUP
    return [
        t0[0][u0 & 255] ^ t0[1][(u0 >> 8) & 255] ^ t0[2][(u0 >> 16) & 255]
        ^ t0[3][(u0 >> 24) & 255] ^ t0[4][(u0 >> 32) & 255] ^ t0[5][(u0 >> 40) & 255]
        ^ t0[6][(u0 >> 48) & 255] ^ t0[7][u0 >> 56],
        t1[0][u1 & 255] ^ t1[1][(u1 >> 8) & 255] ^ t1[2][(u1 >> 16) & 255]
        ^ t1[3][(u1 >> 24) & 255] ^ t1[4][(u1 >> 32) & 255] ^ t1[5][(u1 >> 40) & 255]
        ^ t1[6][(u1 >> 48) & 255] ^ t1[7][u1 >> 56],
        t2[0][u2 & 255] ^ t2[1][(u2 >> 8) & 255] ^ t2[2][(u2 >> 16) & 255]
        ^ t2[3][(u2 >> 24) & 255] ^ t2[4][(u2 >> 32) & 255] ^ t2[5][(u2 >> 40) & 255]
        ^ t2[6][(u2 >> 48) & 255] ^ t2[7][u2 >> 56],
        t3[0][u3 & 255] ^ t3[1][(u3 >> 8) & 255] ^ t3[2][(u3 >> 16) & 255]
        ^ t3[3][(u3 >> 24) & 255] ^ t3[4][(u3 >> 32) & 255] ^ t3[5][(u3 >> 40) & 255]
        ^ t3[6][(u3 >> 48) & 255] ^ t3[7][u3 >> 56],
    ]  # fmt: skip


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
    raw = s.encode("utf-8")  # Lua strings are bytes; string.byte reads bytes
    for i in range(len(raw), 0, -1):
        byte = raw[i - 1]
        # Lua float semantics: x/0 = inf, and inf/nan propagate through
        # `% 1` as nan instead of raising. Certain "bugged" seeds (e.g.
        # erratic 7LB2WVPK) hit num == 0 mid-hash and depend on the nan
        # collapse pinning the downstream PRNG to a constant.
        try:
            q = 1.1239285023 / num
        except ZeroDivisionError:
            q = math.copysign(math.inf, num)
        num = _lua_mod1(q * byte * math.pi + math.pi * i)
    return num


# LuaJIT on x86-64 makes invalid operations (inf - inf, 0/0) the SSE
# "default NaN", whose sign bit is set; tonumber("nan") reads back as it too.
# The sign matters: TW223 seeds from the double's bits.
_DEFAULT_NAN = struct.unpack("<d", struct.pack("<Q", 0xFFF8_0000_0000_0000))[0]


def _lua_mod1(a: float) -> float:
    """Lua's ``a % 1`` (``a - floor(a)``), with LuaJIT's NaN results."""
    if a != a:
        return a  # a NaN operand propagates unchanged
    if a == math.inf or a == -math.inf:
        return _DEFAULT_NAN  # inf - floor(inf) = inf - inf
    return a % 1  # exact for finite a: equals a - floor(a)


def _truncate_13(x: float) -> float:
    """Replicate Lua's ``tonumber(string.format("%.13f", x))``.

    The game's LuaJIT 2.0.5 formats through the C runtime, which rounds the
    exact binary value to 13 decimals and breaks exact ties to even (G181;
    LuaJIT 2.1's own formatter breaks them away from zero); reading the
    decimal back gives the nearest double, which exact integer division
    reproduces.  ``"nan"`` reads back as LuaJIT's default NaN.
    """
    if x != x:
        return _DEFAULT_NAN
    if x == 0.0 or x == math.inf or x == -math.inf:
        return x
    num, den = x.as_integer_ratio()
    q, r = divmod(abs(num) * 10_000_000_000_000, den)
    if 2 * r > den or (2 * r == den and q % 2 == 1):
        q += 1
    return (q if num > 0 else -q) / 10_000_000_000_000


# ---------------------------------------------------------------------------
# PseudoRandom — the stateful PRNG matching G.GAME.pseudorandom
# ---------------------------------------------------------------------------


def _select_element(table: dict | list, tw_state: list[int]) -> tuple:
    """Shared body of ``pseudorandom_element``: deterministic sort, then draw.

    Returns ``(selected_value, selected_key)``.
    """
    # An exact list/tuple enumerates as it indexes, so its (i + 1, v)
    # entries need not be materialised unless they have to be sorted.
    plain_seq = type(table) is list or type(table) is tuple
    if plain_seq:
        if not table:
            raise ValueError("cannot select from empty table")
        first_val = table[0]
    else:
        if isinstance(table, dict):
            entries = list(table.items())
        else:
            entries = [(i + 1, v) for i, v in enumerate(table)]
        if not entries:
            raise ValueError("cannot select from empty table")
        first_val = entries[0][1]

    # Deterministic sort (matches Lua pseudorandom_element)
    if isinstance(first_val, dict) and "sort_id" in first_val:
        if plain_seq:
            entries = [(i + 1, v) for i, v in enumerate(table)]
        entries.sort(key=lambda e: e[1]["sort_id"])
    elif hasattr(first_val, "sort_id"):
        if plain_seq:
            entries = [(i + 1, v) for i, v in enumerate(table)]
        entries.sort(key=lambda e: e[1].sort_id)
    elif plain_seq:
        # Keys 1..n are already in sorted order: index directly.
        idx = _luajit_random_int(tw_state, 1, len(table))
        return table[idx - 1], idx
    else:
        entries.sort(key=lambda e: (isinstance(e[0], str), e[0]))

    idx = _luajit_random_int(tw_state, 1, len(entries))
    k, v = entries[idx - 1]  # Lua is 1-based
    return v, k


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
        return _select_element(table, _luajit_seed(seed_val))

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

        # Backward Fisher-Yates: for i = #list, 2, -1 do j = math.random(i)
        # Lua math.random(i) returns [1, i]; we convert to 0-based [0, i-1]
        for i in range(len(lst) - 1, 0, -1):
            j = _luajit_random_int(tw_state, 1, i + 1) - 1  # random(i+1) gives [1,i+1], -1 → [0,i]
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
    return _select_element(table, _luajit_seed(seed))


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
