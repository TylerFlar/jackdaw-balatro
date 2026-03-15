"""Tests for the Balatro-compatible PRNG system.

Test strategy:
  - Validate pseudohash against known Lua outputs for specific strings
  - Validate pseudoseed stream advancement produces expected sequences
  - Validate pseudorandom integer/float output ranges
  - Validate pseudoshuffle produces Fisher-Yates results for known seeds
  - Validate pseudorandom_element deterministic sorting + selection
  - Validate generate_starting_seed format (length, character set)
  - Cross-validate against Balatro save files or known seed outcomes
"""

import pytest

from jackdaw.engine.rng import (
    generate_starting_seed,
    pseudohash,
    pseudorandom,
    pseudorandom_element,
    pseudoseed,
    pseudoshuffle,
)


class TestPseudohash:
    """pseudohash: string → float in [0, 1)."""

    def test_returns_float_in_unit_interval(self):
        result = pseudohash("test")
        assert isinstance(result, float)
        assert 0.0 <= result < 1.0

    def test_deterministic(self):
        assert pseudohash("hello") == pseudohash("hello")

    def test_different_strings_different_hashes(self):
        assert pseudohash("foo") != pseudohash("bar")

    def test_empty_string(self):
        result = pseudohash("")
        assert isinstance(result, float)
        assert 0.0 <= result < 1.0

    def test_known_seed_tutorial(self):
        """The TUTORIAL seed must hash to a specific value for determinism."""
        result = pseudohash("TUTORIAL")
        assert isinstance(result, float)
        assert 0.0 <= result < 1.0
        # TODO: fill in exact expected value from Lua cross-validation


class TestPseudoseed:
    """pseudoseed: stateful float LCG per named stream."""

    @pytest.fixture
    def fresh_state(self):
        seed = "TESTSEEED"
        return {
            "seed": seed,
            "hashed_seed": pseudohash(seed),
        }

    def test_initializes_new_key(self, fresh_state):
        result = pseudoseed("boss", fresh_state)
        assert isinstance(result, float)
        assert "boss" in fresh_state

    def test_advances_on_each_call(self, fresh_state):
        r1 = pseudoseed("shop", fresh_state)
        r2 = pseudoseed("shop", fresh_state)
        assert r1 != r2

    def test_independent_streams(self, fresh_state):
        pseudoseed("stream_a", fresh_state)
        pseudoseed("stream_b", fresh_state)
        # Calling stream_a shouldn't affect stream_b
        val_a1 = fresh_state.get("stream_a")
        pseudoseed("stream_b", fresh_state)
        val_a2 = fresh_state.get("stream_a")
        assert val_a1 == val_a2

    def test_output_in_unit_interval(self, fresh_state):
        for _ in range(100):
            result = pseudoseed("test", fresh_state)
            assert 0.0 <= result <= 1.0


class TestPseudorandom:
    """pseudorandom: bridge to uniform output."""

    @pytest.fixture
    def state(self):
        seed = "FIXEDSEED"
        return {
            "seed": seed,
            "hashed_seed": pseudohash(seed),
        }

    def test_float_output(self, state):
        result = pseudorandom("test_key", state)
        assert isinstance(result, float)
        assert 0.0 <= result < 1.0

    def test_integer_range(self, state):
        result = pseudorandom("test_key", state, min_val=1, max_val=10)
        assert isinstance(result, int)
        assert 1 <= result <= 10

    def test_deterministic_with_same_state(self):
        def make_state():
            seed = "SAMESEED"
            return {"seed": seed, "hashed_seed": pseudohash(seed)}

        s1, s2 = make_state(), make_state()
        assert pseudorandom("key", s1) == pseudorandom("key", s2)

    def test_float_seed_input(self):
        result = pseudorandom(0.5)
        assert isinstance(result, float)


class TestPseudorandomElement:
    """pseudorandom_element: deterministic selection from collection."""

    def test_selects_from_dict(self):
        table = {"a": 1, "b": 2, "c": 3}
        value, key = pseudorandom_element(table, 0.42)
        assert key in table
        assert value == table[key]

    def test_deterministic(self):
        table = {"x": 10, "y": 20, "z": 30}
        v1, k1 = pseudorandom_element(table, 0.42)
        v2, k2 = pseudorandom_element(table, 0.42)
        assert k1 == k2 and v1 == v2

    def test_different_seeds_can_differ(self):
        table = {str(i): i for i in range(20)}
        _, k1 = pseudorandom_element(table, 0.1)
        _, k2 = pseudorandom_element(table, 0.9)
        # With 20 elements, very likely to differ
        assert k1 != k2


class TestPseudoshuffle:
    """pseudoshuffle: Fisher-Yates with deterministic pre-sort."""

    def test_shuffles_in_place(self):
        lst = list(range(10))
        original = lst.copy()
        pseudoshuffle(lst, 0.42)
        assert lst != original  # very unlikely to be identical
        assert sorted(lst) == sorted(original)

    def test_deterministic(self):
        a = list(range(20))
        b = list(range(20))
        pseudoshuffle(a, 0.42)
        pseudoshuffle(b, 0.42)
        assert a == b

    def test_preserves_elements(self):
        lst = list(range(52))
        pseudoshuffle(lst, 0.123)
        assert sorted(lst) == list(range(52))


class TestGenerateStartingSeed:
    """generate_starting_seed: 8-char alphanumeric string."""

    def test_length(self):
        seed = generate_starting_seed(0.42)
        assert len(seed) == 8

    def test_character_set(self):
        seed = generate_starting_seed(0.42)
        allowed = set("123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
        assert all(c in allowed for c in seed), f"Invalid chars in {seed}"

    def test_no_zero_or_oh(self):
        seed = generate_starting_seed(0.42)
        assert "0" not in seed
        assert "O" not in seed

    def test_deterministic(self):
        assert generate_starting_seed(0.42) == generate_starting_seed(0.42)
