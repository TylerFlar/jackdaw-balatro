"""The seeded sampler behind the Gym action table's subsampling.

When a card selection space or the whole table exceeds its budget, the Gym
wrapper keeps a uniform random subset, in order.  The draws come from
xoshiro256** (Blackman and Vigna) seeded through SplitMix64, and a subset of
``k`` out of ``n`` is drawn with Floyd's algorithm in ``min(k, n - k)`` draws.
``jackdaw._core`` (``crates/jackdaw-env/src/gym.rs``) implements the same
generator, so a table built in Rust and one built here agree draw for draw.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_MASK64 = (1 << 64) - 1


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Xoshiro256:
    """xoshiro256**, seeded from one integer (taken modulo 2**64) through SplitMix64."""

    __slots__ = ("_s",)

    def __init__(self, seed: int = 0) -> None:
        x = seed & _MASK64
        s = []
        for _ in range(4):
            x = (x + 0x9E3779B97F4A7C15) & _MASK64
            z = x
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
            s.append(z ^ (z >> 31))
        self._s = s

    @classmethod
    def from_state(cls, state: Sequence[int]) -> Xoshiro256:
        rng = cls.__new__(cls)
        rng._s = [int(w) & _MASK64 for w in state]
        return rng

    @property
    def state(self) -> tuple[int, int, int, int]:
        """The four 64-bit state words."""
        s0, s1, s2, s3 = self._s
        return (s0, s1, s2, s3)

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self._s
        result = (_rotl((s1 * 5) & _MASK64, 7) * 9) & _MASK64
        t = (s1 << 17) & _MASK64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        self._s = [s0, s1, s2, _rotl(s3, 45)]
        return result

    def below(self, bound: int) -> int:
        """A uniform integer in ``[0, bound)`` (Lemire's multiply-and-reject)."""
        threshold = (-bound) % bound
        while True:
            m = self.next_u64() * bound
            if (m & _MASK64) >= threshold:
                return m >> 64

    def __repr__(self) -> str:
        return f"Xoshiro256(state={self.state})"


def subsample_indices(n: int, k: int, rng: Xoshiro256) -> list[int]:
    """``k`` distinct indices of ``range(n)``, ascending, uniformly at random (``k <= n``).

    Floyd's algorithm picks the kept indices, or the dropped ones when fewer
    are dropped, in ``min(k, n - k)`` draws.
    """
    m = min(k, n - k)
    picked: set[int] = set()
    for j in range(n - m, n):
        t = rng.below(j + 1)
        picked.add(j if t in picked else t)
    if m == k:
        return sorted(picked)
    return [i for i in range(n) if i not in picked]


def subsample(items: Sequence[Any], budget: int, rng: Xoshiro256) -> Sequence[Any]:
    """*items* if within budget, else a uniform random subset of *budget* of them, in order."""
    if len(items) <= budget:
        return items
    return [items[i] for i in subsample_indices(len(items), budget, rng)]
