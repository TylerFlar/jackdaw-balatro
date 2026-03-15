"""Tests for the Balatro-compatible PRNG system.

Test strategy:
  - Validate pseudohash against known Lua outputs for specific strings
  - Validate PseudoRandom.seed() stream advancement via Lua ground truth
  - Validate PseudoRandom class API: random, element, shuffle, predict_seed
  - Validate functional wrappers (pseudoseed, pseudorandom, etc.)
  - Validate generate_starting_seed format

Ground-truth values were obtained by running the original Lua functions
(misc_functions.lua:279-320) via lupa, matching to 15 decimal places.
"""

import pytest

from jackdaw.engine.rng import (
    PseudoRandom,
    generate_starting_seed,
    pseudohash,
    pseudorandom,
    pseudorandom_element,
    pseudoseed,
    pseudoshuffle,
)

# Tolerance: Lua uses IEEE 754 doubles, same as Python. pseudohash has no
# truncation so we require near-exact equality. pseudoseed truncates to 13
# decimal places via string.format("%.13f", ...).
HASH_TOL = 1e-15
SEED_TOL = 1e-13


# ============================================================================
# Ground-truth from Lua (via lupa)
# ============================================================================

PSEUDOHASH_TRUTH = {
    "TESTSEED": 0.319272078222355,
    "TUTORIAL": 0.417952113690717,
    "A3K9NZ2B": 0.946384286395556,
    "A": 0.651751842670649,
    "a": 0.641368674218143,
    "AB": 0.916420555346463,
    "bossTUTORIAL": 0.174561306470309,
    "shuffleTUTORIAL": 0.663191794746467,
    "lucky_multTUTORIAL": 0.853518645904103,
    "Joker1shoA3K9NZ2B": 0.854421527927570,
}

# pseudoseed ground truth: {seed_str: {stream_key: [(result, stored), ...]}}
PSEUDOSEED_TRUTH = {
    "TESTSEED": {
        "boss": [
            (0.655250828936977, 0.991229579651600),
            (0.581457451664827, 0.843642825107300),
            (0.454214620624177, 0.589157163026000),
            (0.234808236884227, 0.150344395546100),
            (0.356483101926677, 0.393694125631000),
        ],
        "shuffle": [
            (0.259181259155277, 0.199090440088200),
            (0.398509804991727, 0.477747531761100),
            (0.638755735891777, 0.958239393561200),
            (0.553014759177527, 0.786757440132700),
            (0.405170534985477, 0.491068991748600),
        ],
        "lucky_mult": [
            (0.384863452913327, 0.450454827604300),
            (0.615225162607027, 0.911178246991700),
            (0.512440701295377, 0.705609324368400),
            (0.335208186295427, 0.351144294368500),
            (0.529603973631777, 0.739935869041200),
        ],
        "rarity1": [
            (0.358788231947277, 0.398304385672200),
            (0.570263337327027, 0.821254596431700),
            (0.434912471136127, 0.550552864049900),
            (0.201525302333477, 0.083778526444600),
            (0.299092927169377, 0.278913776116400),
        ],
    },
    "A3K9NZ2B": {
        "boss": [
            (0.894786055642878, 0.843187824890200),
            (0.767378443466228, 0.588372600536900),
            (0.547687925580128, 0.148991564764700),
            (0.668872854608128, 0.391361422820700),
            (0.877833523153128, 0.809282759910700),
        ],
        "shuffle": [
            (0.738665515415678, 0.530946744435800),
            (0.498177869425028, 0.049971452454500),
            (0.583502053825778, 0.220619821256000),
            (0.730627597888428, 0.514870909381300),
            (0.484317989044878, 0.022251691694200),
        ],
        "lucky_mult": [
            (0.713836136097628, 0.481287985799700),
            (0.955364264272378, 0.964344242149200),
            (0.871834196141178, 0.797284105886800),
            (0.727802268901528, 0.509220251407500),
            (0.479446239408228, 0.012508192420900),
        ],
        "rarity1": [
            (0.879998709605878, 0.813613132816200),
            (0.741880440218778, 0.537376594042000),
            (0.503721403935178, 0.061058521474800),
            (0.593060838788878, 0.239737391182200),
            (0.747109928755728, 0.547835571115900),
        ],
    },
}

# predict_seed ground truth
PREDICT_SEED_TRUTH = [
    ("Joker4", "TESTSEED", 0.236212407946077),
    ("Joker4", "TUTORIAL", 0.585922460919508),
    ("boss", "A3K9NZ2B", 0.894786055642878),
]


# ============================================================================
# pseudohash tests
# ============================================================================

class TestPseudohash:
    """pseudohash: string -> float."""

    @pytest.mark.parametrize("s,expected", PSEUDOHASH_TRUTH.items())
    def test_matches_lua(self, s: str, expected: float):
        assert abs(pseudohash(s) - expected) < HASH_TOL, (
            f"pseudohash({s!r}): {pseudohash(s):.15f} != {expected:.15f}"
        )

    def test_empty_string(self):
        assert pseudohash("") == 1.0

    def test_deterministic(self):
        assert pseudohash("hello") == pseudohash("hello")

    def test_different_strings_differ(self):
        assert pseudohash("foo") != pseudohash("bar")

    def test_case_sensitive(self):
        assert pseudohash("A") != pseudohash("a")

    def test_long_string_in_range(self):
        result = pseudohash("x" * 500)
        assert 0.0 <= result < 1.0


# ============================================================================
# PseudoRandom class tests
# ============================================================================

class TestPseudoRandomSeed:
    """PseudoRandom.seed(): stateful LCG per stream."""

    @pytest.mark.parametrize("seed_str", ["TESTSEED", "A3K9NZ2B"])
    @pytest.mark.parametrize("stream_key", ["boss", "shuffle", "lucky_mult", "rarity1"])
    def test_matches_lua_sequence(self, seed_str: str, stream_key: str):
        """5 sequential calls per stream must match Lua ground truth."""
        prng = PseudoRandom(seed_str)
        expected_seq = PSEUDOSEED_TRUTH[seed_str][stream_key]

        for call_idx, (expected_result, expected_stored) in enumerate(expected_seq):
            result = prng.seed(stream_key)
            stored = prng.state[stream_key]

            assert abs(result - expected_result) < SEED_TOL, (
                f"seed={seed_str!r} stream={stream_key!r} call {call_idx + 1}: "
                f"result {result:.15f} != {expected_result:.15f}"
            )
            assert abs(stored - expected_stored) < SEED_TOL, (
                f"seed={seed_str!r} stream={stream_key!r} call {call_idx + 1}: "
                f"stored {stored:.15f} != {expected_stored:.15f}"
            )

    def test_streams_are_independent(self):
        """Advancing one stream must not affect another."""
        prng = PseudoRandom("TESTSEED")

        # Prime both streams
        prng.seed("alpha")
        prng.seed("beta")
        alpha_val = prng.state["alpha"]

        # Advance beta many times
        for _ in range(10):
            prng.seed("beta")

        assert prng.state["alpha"] == alpha_val

    def test_never_repeats_in_50_calls(self):
        prng = PseudoRandom("TESTSEED")
        values = [prng.seed("test") for _ in range(50)]
        assert len(set(values)) == 50

    def test_output_range(self):
        prng = PseudoRandom("TESTSEED")
        for _ in range(500):
            v = prng.seed("range_check")
            assert 0.0 <= v <= 1.0

    def test_hashed_seed_matches_pseudohash(self):
        prng = PseudoRandom("TUTORIAL")
        assert abs(prng.hashed_seed - pseudohash("TUTORIAL")) < HASH_TOL

    def test_init_stores_seed_string(self):
        prng = PseudoRandom("MYSEED")
        assert prng.seed_str == "MYSEED"

    def test_init_only_has_reserved_keys(self):
        """Fresh PseudoRandom has only 'seed' and 'hashed_seed', no streams."""
        prng = PseudoRandom("A3K9NZ2B")
        assert set(prng.state.keys()) == {"seed", "hashed_seed"}

    # LuaJIT ground truth: lazy init values = pseudohash(key + seed)
    LAZY_INIT_TRUTH_A3K9NZ2B = {
        "boss": 0.990965706218390,
        "shuffle": 0.809884197253268,
        "lucky_mult": 0.781085030487418,
        "rarity1": 0.393872785063536,
        "front1": 0.978286517849483,
    }

    @pytest.mark.parametrize("key,expected_init", LAZY_INIT_TRUTH_A3K9NZ2B.items())
    def test_lazy_init_matches_pseudohash(self, key: str, expected_init: float):
        """First access to a stream initializes it from pseudohash(key+seed)."""
        prng = PseudoRandom("A3K9NZ2B")
        assert key not in prng.state
        prng.seed(key)  # triggers lazy init then advances
        # After the first call, the stored value has been ADVANCED past init.
        # Verify the init value itself:
        init_val = pseudohash(key + "A3K9NZ2B")
        assert abs(init_val - expected_init) < HASH_TOL, (
            f"pseudohash({key!r}+'A3K9NZ2B') = {init_val:.15f}, expected {expected_init:.15f}"
        )

    def test_lazy_init_then_advance_matches_a3k9nz2b(self):
        """Full sequence on A3K9NZ2B: lazy init → 5 advances, all match oracle."""
        prng = PseudoRandom("A3K9NZ2B")
        for stream_key in ["boss", "shuffle", "lucky_mult", "rarity1"]:
            expected_seq = PSEUDOSEED_TRUTH["A3K9NZ2B"][stream_key]
            for call_idx, (expected_result, _) in enumerate(expected_seq):
                result = prng.seed(stream_key)
                assert abs(result - expected_result) < SEED_TOL, (
                    f"A3K9NZ2B {stream_key}[{call_idx + 1}]: "
                    f"{result:.15f} != {expected_result:.15f}"
                )


class TestPseudoRandomPredictSeed:
    """PseudoRandom.predict_seed(): stateless preview."""

    @pytest.mark.parametrize("key,pseed,expected", PREDICT_SEED_TRUTH)
    def test_matches_lua(self, key: str, pseed: str, expected: float):
        prng = PseudoRandom("IRRELEVANT")  # state doesn't matter
        result = prng.predict_seed(key, pseed)
        assert abs(result - expected) < SEED_TOL, (
            f"predict_seed({key!r}, {pseed!r}): {result:.15f} != {expected:.15f}"
        )

    def test_does_not_mutate_state(self):
        prng = PseudoRandom("TESTSEED")
        state_before = prng.get_state()
        prng.predict_seed("Joker4", "TESTSEED")
        state_after = prng.get_state()
        assert state_before == state_after

    def test_predict_matches_first_seed_call(self):
        """predict_seed(key, seed) should equal the first seed(key) call."""
        prng = PseudoRandom("A3K9NZ2B")
        predicted = prng.predict_seed("boss", "A3K9NZ2B")
        actual = prng.seed("boss")
        assert abs(predicted - actual) < SEED_TOL


class TestPseudoRandomRandom:
    """PseudoRandom.random(): uniform output from stream."""

    def test_float_in_unit_interval(self):
        prng = PseudoRandom("TESTSEED")
        result = prng.random("test")
        assert isinstance(result, float)
        assert 0.0 <= result < 1.0

    def test_integer_range(self):
        prng = PseudoRandom("TESTSEED")
        result = prng.random("test", min_val=1, max_val=6)
        assert isinstance(result, int)
        assert 1 <= result <= 6

    def test_integer_range_coverage(self):
        """Over many calls, all values in range should appear."""
        prng = PseudoRandom("TESTSEED")
        values = {prng.random("roll", min_val=1, max_val=6) for _ in range(200)}
        assert values == {1, 2, 3, 4, 5, 6}

    def test_deterministic_with_same_seed(self):
        p1 = PseudoRandom("SAME")
        p2 = PseudoRandom("SAME")
        assert p1.random("key") == p2.random("key")

    def test_float_key(self):
        """Passing a numeric seed directly (bypassing stream)."""
        prng = PseudoRandom("TESTSEED")
        result = prng.random(0.5)
        assert isinstance(result, float)

    def test_advances_stream(self):
        """Calling random(str_key) should advance that stream's state."""
        prng = PseudoRandom("TESTSEED")
        prng.random("mystream")
        assert "mystream" in prng.state


class TestPseudoRandomElement:
    """PseudoRandom.element(): deterministic selection.

    Ground truth from LuaJIT 2.1 pseudorandom_element() with seed 'TESTSEED'.
    Each test uses sequential pseudoseed advances on a named stream, then
    passes the resulting float to element() — exactly matching the game's
    usage pattern.
    """

    # LuaJIT ground truth: 5 calls on 'elem_str' stream with string-key dict
    EXPECTED_STR = [
        ("alpha", 1), ("alpha", 1), ("gamma", 3), ("alpha", 1), ("gamma", 3),
    ]

    def test_string_key_dict_matches_luajit(self):
        """String keys are sorted lexicographically before selection."""
        prng = PseudoRandom("TESTSEED")
        table = {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "epsilon": 5}
        for i, (exp_key, exp_val) in enumerate(self.EXPECTED_STR):
            sv = prng.seed("elem_str")
            val, key = prng.element(table, sv)
            assert key == exp_key and val == exp_val, (
                f"str[{i + 1}]: got ({key!r}, {val}) expected ({exp_key!r}, {exp_val})"
            )

    # LuaJIT ground truth: sort_id dicts
    EXPECTED_SORTID = [
        ("b_second", "second"), ("y_fourth", "fourth"), ("b_second", "second"),
        ("a_first", "first"), ("y_fourth", "fourth"),
    ]

    def test_sort_id_dict_matches_luajit(self):
        """Dicts with sort_id values are sorted by sort_id, not key."""
        prng = PseudoRandom("TESTSEED")
        # Keys are deliberately mis-ordered relative to sort_id
        table = {
            "z_last": {"sort_id": 5, "name": "last"},
            "a_first": {"sort_id": 1, "name": "first"},
            "m_mid": {"sort_id": 3, "name": "mid"},
            "b_second": {"sort_id": 2, "name": "second"},
            "y_fourth": {"sort_id": 4, "name": "fourth"},
        }
        for i, (exp_key, exp_name) in enumerate(self.EXPECTED_SORTID):
            sv = prng.seed("elem_sortid")
            val, key = prng.element(table, sv)
            assert key == exp_key and val["name"] == exp_name, (
                f"sortid[{i + 1}]: got ({key!r}, {val['name']!r}) "
                f"expected ({exp_key!r}, {exp_name!r})"
            )

    # LuaJIT ground truth: large dict (20 items)
    EXPECTED_LARGE = [
        ("key_08", 8), ("key_15", 15), ("key_13", 13), ("key_12", 12), ("key_16", 16),
    ]

    def test_large_dict_matches_luajit(self):
        """20-item dict — verifies sort and selection over a larger space."""
        prng = PseudoRandom("TESTSEED")
        table = {f"key_{i:02d}": i for i in range(1, 21)}
        for i, (exp_key, exp_val) in enumerate(self.EXPECTED_LARGE):
            sv = prng.seed("elem_large")
            val, key = prng.element(table, sv)
            assert key == exp_key and val == exp_val, (
                f"large[{i + 1}]: got ({key!r}, {val}) expected ({exp_key!r}, {exp_val})"
            )

    def test_single_element_dict(self):
        """Single-element dict always returns that element."""
        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("elem_single")
        val, key = prng.element({"only": 42}, sv)
        assert key == "only" and val == 42

    # LuaJIT ground truth: list input (Lua array with integer keys 1..N)
    EXPECTED_LIST = [
        (4, "date"), (5, "elderberry"), (1, "apple"), (5, "elderberry"), (3, "cherry"),
    ]

    def test_list_input_matches_luajit(self):
        """Lists use 1-based integer keys matching Lua arrays."""
        prng = PseudoRandom("TESTSEED")
        items = ["apple", "banana", "cherry", "date", "elderberry"]
        for i, (exp_key, exp_val) in enumerate(self.EXPECTED_LIST):
            sv = prng.seed("elem_list")
            val, key = prng.element(items, sv)
            assert key == exp_key and val == exp_val, (
                f"list[{i + 1}]: got (key={key}, val={val!r}) "
                f"expected (key={exp_key}, val={exp_val!r})"
            )

    def test_deterministic(self):
        prng = PseudoRandom("TESTSEED")
        table = {"x": 10, "y": 20, "z": 30}
        v1, k1 = prng.element(table, 0.42)
        v2, k2 = prng.element(table, 0.42)
        assert k1 == k2 and v1 == v2

    def test_empty_raises(self):
        prng = PseudoRandom("TESTSEED")
        with pytest.raises(ValueError, match="empty"):
            prng.element({}, 0.5)


class TestPseudoRandomShuffle:
    """PseudoRandom.shuffle(): Fisher-Yates with sort_id pre-sort.

    Ground truth from LuaJIT 2.1 pseudoshuffle() with seed 'TESTSEED'.
    Each test uses sequential pseudoseed advances on a named stream, then
    passes the resulting float to shuffle() — exactly matching the game's
    deck shuffle pattern.
    """

    # LuaJIT ground truth: plain [1..10], 3 trials on 'shuf_plain' stream
    EXPECTED_PLAIN = [
        [7, 1, 5, 3, 8, 10, 6, 9, 2, 4],
        [4, 10, 1, 2, 6, 7, 5, 8, 9, 3],
        [4, 10, 8, 6, 7, 3, 2, 5, 1, 9],
    ]

    def test_plain_list_matches_luajit(self):
        """[1..10] shuffled 3 times must match LuaJIT exactly."""
        prng = PseudoRandom("TESTSEED")
        for trial, expected in enumerate(self.EXPECTED_PLAIN):
            sv = prng.seed("shuf_plain")
            lst = list(range(1, 11))
            prng.shuffle(lst, sv)
            assert lst == expected, (
                f"plain[{trial + 1}]: {lst} != {expected}"
            )

    # LuaJIT ground truth: 52-item deck shuffle
    EXPECTED_DECK52 = [
        43, 19, 4, 45, 14, 3, 13, 52, 50, 7, 24, 33, 48, 20, 29,
        41, 21, 26, 31, 37, 12, 25, 28, 40, 16, 1, 47, 5, 22, 42,
        10, 46, 18, 23, 30, 15, 36, 8, 11, 39, 6, 35, 2, 38, 27,
        34, 51, 17, 44, 32, 49, 9,
    ]

    def test_52_card_deck_matches_luajit(self):
        """Full 52-card deck shuffle — the actual game use case."""
        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("shuf_deck")
        deck = list(range(1, 53))
        prng.shuffle(deck, sv)
        assert deck == self.EXPECTED_DECK52

    # LuaJIT ground truth: sort_id pre-sort
    EXPECTED_SORTID = [8, 2, 5, 6, 4, 7, 1, 3, 10, 9]

    def test_sort_id_presort_reversed_input(self):
        """Items with sort_id in reverse order are pre-sorted then shuffled."""

        class Item:
            def __init__(self, sort_id: int):
                self.sort_id = sort_id
                self.val = sort_id

        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("shuf_sortid")
        items = [Item(i) for i in range(10, 0, -1)]  # reversed
        prng.shuffle(items, sv)
        result = [x.val for x in items]
        assert result == self.EXPECTED_SORTID, (
            f"sortid_rev: {result} != {self.EXPECTED_SORTID}"
        )

    def test_sort_id_presort_forward_input(self):
        """Items with sort_id in forward order produce same result as reversed."""

        class Item:
            def __init__(self, sort_id: int):
                self.sort_id = sort_id
                self.val = sort_id

        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("shuf_sortid")
        items = [Item(i) for i in range(1, 11)]  # forward
        prng.shuffle(items, sv)
        result = [x.val for x in items]
        assert result == self.EXPECTED_SORTID, (
            f"sortid_fwd: {result} != {self.EXPECTED_SORTID}"
        )

    def test_sort_id_presort_random_input(self):
        """Items in arbitrary order still produce same result after pre-sort."""

        class Item:
            def __init__(self, sort_id: int):
                self.sort_id = sort_id
                self.val = sort_id

        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("shuf_sortid")
        items = [Item(i) for i in [5, 2, 8, 1, 9, 3, 7, 10, 6, 4]]  # scrambled
        prng.shuffle(items, sv)
        result = [x.val for x in items]
        assert result == self.EXPECTED_SORTID

    def test_dict_sort_id_presort(self):
        """sort_id on dicts works the same as on objects."""
        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("shuf_sortid")
        items = [{"sort_id": i, "val": i} for i in range(10, 0, -1)]
        prng.shuffle(items, sv)
        result = [x["val"] for x in items]
        assert result == self.EXPECTED_SORTID

    def test_two_elements(self):
        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("shuf_two")
        lst = [1, 2]
        prng.shuffle(lst, sv)
        assert lst == [2, 1]  # LuaJIT ground truth

    def test_single_element(self):
        prng = PseudoRandom("TESTSEED")
        sv = prng.seed("shuf_one")
        lst = [42]
        prng.shuffle(lst, sv)
        assert lst == [42]

    def test_empty_list(self):
        prng = PseudoRandom("TESTSEED")
        lst: list = []
        prng.shuffle(lst, 0.5)
        assert lst == []

    def test_preserves_elements(self):
        prng = PseudoRandom("TESTSEED")
        lst = list(range(52))
        prng.shuffle(lst, 0.42)
        assert sorted(lst) == list(range(52))


class TestPseudoRandomStateManagement:
    """State save/load for persistence."""

    def test_get_state_is_copy(self):
        prng = PseudoRandom("TESTSEED")
        prng.seed("boss")
        state = prng.get_state()
        prng.seed("boss")  # advance
        assert state["boss"] != prng.state["boss"]

    def test_load_state_restores(self):
        prng1 = PseudoRandom("TESTSEED")
        for _ in range(5):
            prng1.seed("boss")
        saved = prng1.get_state()

        prng2 = PseudoRandom("IGNORED")
        prng2.load_state(saved)

        # Both should produce identical next values
        assert prng1.seed("boss") == prng2.seed("boss")

    def test_load_state_rehashes_zeroes(self):
        """Save convention: stream values of 0 are re-hashed on load."""
        prng = PseudoRandom("TESTSEED")
        prng.load_state({"seed": "TESTSEED", "hashed_seed": pseudohash("TESTSEED"), "boss": 0})
        # boss should have been re-hashed, not left at 0
        assert prng.state["boss"] != 0
        assert abs(prng.state["boss"] - pseudohash("bossTESTSEED")) < HASH_TOL


# ============================================================================
# Functional API wrappers (backward compat)
# ============================================================================

class TestFunctionalPseudoseed:
    """pseudoseed(): functional wrapper matching old API."""

    def test_matches_class_api(self):
        # Class API
        prng = PseudoRandom("TESTSEED")
        class_results = [prng.seed("boss") for _ in range(5)]

        # Functional API
        state = {"seed": "TESTSEED", "hashed_seed": pseudohash("TESTSEED")}
        func_results = [pseudoseed("boss", state) for _ in range(5)]

        for i, (c, f) in enumerate(zip(class_results, func_results)):
            assert abs(c - f) < SEED_TOL, f"Call {i}: class={c:.15f} func={f:.15f}"


class TestFunctionalPseudorandom:
    """pseudorandom(): functional wrapper."""

    def test_float_output(self):
        state = {"seed": "TESTSEED", "hashed_seed": pseudohash("TESTSEED")}
        result = pseudorandom("boss", state)
        assert isinstance(result, float)
        assert 0.0 <= result < 1.0

    def test_integer_output(self):
        state = {"seed": "TESTSEED", "hashed_seed": pseudohash("TESTSEED")}
        result = pseudorandom("boss", state, min_val=1, max_val=10)
        assert isinstance(result, int)
        assert 1 <= result <= 10

    def test_float_seed_direct(self):
        result = pseudorandom(0.5)
        assert isinstance(result, float)

    def test_string_key_requires_state(self):
        with pytest.raises(ValueError, match="state dict required"):
            pseudorandom("boss")


class TestFunctionalElement:
    """pseudorandom_element(): functional wrapper."""

    def test_basic_selection(self):
        table = {"a": 1, "b": 2, "c": 3}
        val, key = pseudorandom_element(table, 0.42)
        assert key in table and val == table[key]

    def test_deterministic(self):
        table = {"a": 1, "b": 2, "c": 3}
        assert pseudorandom_element(table, 0.42) == pseudorandom_element(table, 0.42)


class TestFunctionalShuffle:
    """pseudoshuffle(): functional wrapper."""

    def test_preserves_and_shuffles(self):
        lst = list(range(20))
        pseudoshuffle(lst, 0.42)
        assert sorted(lst) == list(range(20))

    def test_deterministic(self):
        a, b = list(range(20)), list(range(20))
        pseudoshuffle(a, 0.42)
        pseudoshuffle(b, 0.42)
        assert a == b


class TestGenerateStartingSeed:
    """generate_starting_seed: 8-char alphanumeric.

    Ground truth from LuaJIT 2.1 random_string() (misc_functions.lua:270).
    """

    ALLOWED = set("123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")

    # LuaJIT ground truth: entropy → seed string
    LUAJIT_TRUTH = {
        0.0: "7BXBJDJG",
        0.42: "A9B2M1XG",
        0.999: "XVXSVYEJ",
        1.5: "5SVPRIP2",
        3.14159: "84F18YHB",
    }

    @pytest.mark.parametrize("entropy,expected", LUAJIT_TRUTH.items())
    def test_matches_luajit(self, entropy: float, expected: str):
        result = generate_starting_seed(entropy)
        assert result == expected, f"entropy={entropy}: {result!r} != {expected!r}"

    def test_length(self):
        assert len(generate_starting_seed(0.42)) == 8

    def test_character_set(self):
        seed = generate_starting_seed(0.42)
        assert all(c in self.ALLOWED for c in seed), f"Bad chars in {seed!r}"

    def test_no_zero_or_oh(self):
        seed = generate_starting_seed(0.42)
        assert "0" not in seed
        assert "O" not in seed

    def test_deterministic(self):
        assert generate_starting_seed(0.42) == generate_starting_seed(0.42)

    def test_different_entropy_different_seeds(self):
        assert generate_starting_seed(0.1) != generate_starting_seed(0.9)
