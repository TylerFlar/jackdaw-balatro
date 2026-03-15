"""Balatro-compatible PRNG system.

Replicates the three-layer RNG architecture from the Balatro source:
  1. pseudohash(str) — deterministic string → float hash
  2. pseudoseed(key) — stateful float-domain LCG per named stream
  3. pseudorandom()  — bridges to uniform random output

See docs/source-map/rng-system.md for the full specification.
"""

from __future__ import annotations


def pseudohash(s: str) -> float:
    """Hash a string to a float in [0, 1).

    Iterates bytes in reverse, accumulating via:
        num = ((1.1239285023 / num) * byte * pi + pi * i) % 1

    Must match Lua's output exactly for seed determinism.
    """
    raise NotImplementedError


def pseudoseed(key: str, state: dict[str, float]) -> float:
    """Advance the named RNG stream and return a float in [0, 1).

    On first call for a given key, initializes state[key] from
    pseudohash(key + state['seed']).

    Each call advances state[key] via the float LCG:
        next = (2.134453429141 + current * 1.72431234) % 1
    truncated to 13 decimal places.

    Returns (advanced + state['hashed_seed']) / 2.

    Args:
        key: Named stream identifier (e.g. 'boss', 'lucky_mult').
        state: The G.GAME.pseudorandom dict. Mutated in place.

    Returns:
        A float coupled to the seed, suitable for seeding further draws.
    """
    raise NotImplementedError


def pseudorandom(seed: float | str, state: dict[str, float] | None = None,
                 min_val: int | None = None, max_val: int | None = None) -> float | int:
    """Generate a pseudorandom number from a seed value.

    If seed is a string, calls pseudoseed(seed, state) first.
    Then uses the resulting float as a one-shot seed for uniform output.

    Args:
        seed: Either a float (direct seed) or string (stream key).
        state: Required if seed is a string.
        min_val: If provided with max_val, returns int in [min_val, max_val].
        max_val: Upper bound for integer output.

    Returns:
        Float in [0, 1) if no range given, else int in [min_val, max_val].
    """
    raise NotImplementedError


def pseudorandom_element(table: dict | list, seed: float) -> tuple:
    """Select a random element from a collection.

    Sorts deterministically (by sort_id if available, else by key)
    before selecting, to compensate for non-deterministic iteration.

    Args:
        table: Dict or list to select from.
        seed: Float seed for the selection.

    Returns:
        (selected_value, selected_key) tuple.
    """
    raise NotImplementedError


def pseudoshuffle(lst: list, seed: float) -> None:
    """Fisher-Yates shuffle a list in place.

    Pre-sorts by sort_id (if elements have one) to ensure deterministic
    starting order, then applies the standard backward Fisher-Yates.

    Args:
        lst: List to shuffle. Mutated in place.
        seed: Float seed for the shuffle.
    """
    raise NotImplementedError


def generate_starting_seed(entropy: float = 0.0) -> str:
    """Generate an 8-character alphanumeric seed string.

    Characters: digits 1-9, letters A-N, P-Z (no 0 or O).

    Args:
        entropy: A float used to seed the generation (in the real game,
                 derived from cursor position and time).

    Returns:
        8-character uppercase string like "A3K9NZ2B".
    """
    raise NotImplementedError
