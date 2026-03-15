"""Cross-validation tests: Python PseudoRandom vs Lua ground truth.

Two modes:
  1. FIXTURE MODE (always runs): loads pre-generated Lua output from
     tests/fixtures/rng_oracle_TESTSEED.json and asserts Python matches.
  2. LIVE LUA MODE (optional): runs scripts/lua_rng_oracle.lua via subprocess,
     parses the JSON output, and validates Python against it.  Skips gracefully
     if no lua/luajit interpreter is found.

The fixture file contains pseudohash and pseudoseed values which are
Lua-version-independent (pure IEEE 754 float math, identical across
Lua 5.1, 5.4, and LuaJIT).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from jackdaw.engine.rng import PseudoRandom, pseudohash

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "rng_oracle_TESTSEED.json"
ORACLE_SCRIPT = PROJECT_ROOT / "scripts" / "lua_rng_oracle.lua"

# Tolerances
HASH_TOL = 1e-15  # pseudohash: exact IEEE 754 match
SEED_TOL = 1e-13  # pseudoseed: 13 decimal places (Lua truncation)


# ============================================================================
# Fixture loading
# ============================================================================

def load_fixture() -> dict:
    """Load the pre-generated Lua oracle fixture."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def find_lua() -> str | None:
    """Find a Lua interpreter (luajit preferred, then lua, lua5.1, etc.)."""
    for name in ["luajit", "lua", "lua5.1", "lua5.4", "lua54", "lua51"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def run_lua_oracle(seed: str, lua_path: str) -> dict:
    """Run the Lua oracle script and parse its JSON output."""
    result = subprocess.run(
        [lua_path, str(ORACLE_SCRIPT), seed],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"Lua oracle failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")
    # The output is a wrapper with a "seeds" array
    data = json.loads(result.stdout)
    if "seeds" in data:
        return data["seeds"][0]
    return data


# ============================================================================
# MODE 1: Fixture-based tests (always run)
# ============================================================================

class TestFixtureOracle:
    """Validate Python RNG against pre-generated Lua fixture (TESTSEED)."""

    @pytest.fixture(scope="class")
    def fixture(self) -> dict:
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
        return load_fixture()

    @pytest.fixture(scope="class")
    def prng(self) -> PseudoRandom:
        return PseudoRandom("TESTSEED")

    def test_hashed_seed(self, fixture):
        expected = fixture["hashed_seed"]
        actual = pseudohash("TESTSEED")
        assert abs(actual - expected) < HASH_TOL, (
            f"hashed_seed: {actual:.15f} != {expected:.15f}"
        )

    def test_pseudohash_stream_keys(self, fixture):
        """Verify pseudohash(key + seed) for all stream keys."""
        for key, expected in fixture["pseudohash"].items():
            actual = pseudohash(key + "TESTSEED")
            assert abs(actual - expected) < HASH_TOL, (
                f"pseudohash({key}+TESTSEED): {actual:.15f} != {expected:.15f}"
            )

    @pytest.mark.parametrize("stream_key", [
        "boss", "shuffle", "lucky_mult", "rarity1",
        "stdset1", "cdt1", "front1", "edition_generic",
    ])
    def test_pseudoseed_sequence(self, fixture, stream_key):
        """10 consecutive pseudoseed advances must match Lua for each stream."""
        prng = PseudoRandom("TESTSEED")
        expected_seq = fixture["pseudoseed"][stream_key]

        for entry in expected_seq:
            call_num = entry["call"]
            expected_result = entry["result"]
            expected_stored = entry["stored"]

            result = prng.seed(stream_key)
            stored = prng.state[stream_key]

            assert abs(result - expected_result) < SEED_TOL, (
                f"stream={stream_key!r} call {call_num}: "
                f"result {result:.15f} != {expected_result:.15f}"
            )
            assert abs(stored - expected_stored) < SEED_TOL, (
                f"stream={stream_key!r} call {call_num}: "
                f"stored {stored:.15f} != {expected_stored:.15f}"
            )

    def test_predict_seed(self, fixture):
        """Stateless predict_seed must match Lua."""
        prng = PseudoRandom("TESTSEED")
        for entry in fixture["predict_seed"]:
            key = entry["key"]
            predict_with = entry["predict_with"]
            expected = entry["result"]
            actual = prng.predict_seed(key, predict_with)
            assert abs(actual - expected) < SEED_TOL, (
                f"predict_seed({key!r}, {predict_with!r}): "
                f"{actual:.15f} != {expected:.15f}"
            )

    def test_all_streams_independent(self, fixture):
        """Advancing all 8 streams interleaved must still match fixture.

        This catches bugs where advancing one stream corrupts another.
        """
        prng = PseudoRandom("TESTSEED")
        streams = list(fixture["pseudoseed"].keys())

        # Advance all streams call-by-call (interleaved)
        for call_idx in range(10):
            for stream_key in streams:
                entry = fixture["pseudoseed"][stream_key][call_idx]
                result = prng.seed(stream_key)
                assert abs(result - entry["result"]) < SEED_TOL, (
                    f"interleaved: stream={stream_key} call {call_idx + 1}: "
                    f"{result:.15f} != {entry['result']:.15f}"
                )

    def test_fixture_has_expected_structure(self, fixture):
        """Sanity check: fixture has the data we expect."""
        assert fixture["seed"] == "TESTSEED"
        assert "hashed_seed" in fixture
        assert len(fixture["pseudoseed"]) == 8
        assert len(fixture["pseudoseed"]["boss"]) == 10
        assert len(fixture["predict_seed"]) == 3


# ============================================================================
# MODE 2: Live Lua oracle (optional, skips if no Lua available)
# ============================================================================

class TestLiveLuaOracle:
    """Run the Lua oracle script live and validate against Python."""

    @pytest.fixture(scope="class")
    def lua_path(self):
        path = find_lua()
        if not path:
            pytest.skip(
                "No Lua interpreter found. Install lua, lua5.1, or luajit "
                "to enable live Lua cross-validation."
            )
        return path

    @pytest.fixture(scope="class")
    def oracle_data(self, lua_path) -> dict:
        """Run oracle for TESTSEED and parse output."""
        return run_lua_oracle("TESTSEED", lua_path)

    def test_hashed_seed(self, oracle_data):
        expected = oracle_data["hashed_seed"]
        actual = pseudohash("TESTSEED")
        assert abs(actual - expected) < HASH_TOL

    @pytest.mark.parametrize("stream_key", [
        "boss", "shuffle", "lucky_mult", "rarity1",
        "stdset1", "cdt1", "front1", "edition_generic",
    ])
    def test_pseudoseed_sequence(self, oracle_data, stream_key):
        """10 advances must match live Lua output."""
        prng = PseudoRandom("TESTSEED")
        expected_seq = oracle_data["pseudoseed"][stream_key]

        for entry in expected_seq:
            call_num = entry["call"]
            expected_result = entry["result"]
            result = prng.seed(stream_key)
            assert abs(result - expected_result) < SEED_TOL, (
                f"live lua: stream={stream_key!r} call {call_num}: "
                f"{result:.15f} != {expected_result:.15f}"
            )

    def test_predict_seed(self, oracle_data):
        prng = PseudoRandom("TESTSEED")
        for entry in oracle_data["predict_seed"]:
            expected = entry["result"]
            actual = prng.predict_seed(entry["key"], entry["predict_with"])
            assert abs(actual - expected) < SEED_TOL

    @pytest.mark.parametrize("seed_str", ["A3K9NZ2B", "TUTORIAL"])
    def test_other_seeds(self, lua_path, seed_str):
        """Cross-validate additional seeds beyond TESTSEED."""
        oracle = run_lua_oracle(seed_str, lua_path)
        prng = PseudoRandom(seed_str)

        # hashed_seed
        assert abs(pseudohash(seed_str) - oracle["hashed_seed"]) < HASH_TOL

        # All pseudoseed streams
        for stream_key, expected_seq in oracle["pseudoseed"].items():
            prng_stream = PseudoRandom(seed_str)  # fresh per stream
            for entry in expected_seq:
                result = prng_stream.seed(stream_key)
                assert abs(result - entry["result"]) < SEED_TOL, (
                    f"seed={seed_str!r} stream={stream_key!r} call {entry['call']}"
                )
