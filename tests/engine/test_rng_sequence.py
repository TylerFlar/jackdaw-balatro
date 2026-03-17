"""Integration test: simulates the RNG calls of a real Balatro run.

Uses seed "TESTSEED" and replays the exact sequence of pseudoseed/pseudorandom
calls that the game makes during run start (deck shuffle, boss selection, tag
generation, shop population, rerolls).  Every value is cross-validated against
LuaJIT 2.1 ground truth.

This catches:
  - Stream independence bugs (advancing 'boss' must not affect 'shuffle')
  - Ante-suffix key patterns ('cdt1', 'rarity1sho', 'Joker1sho1')
  - Consecutive calls to the same stream producing the correct sequence
  - Full pipeline correctness (pseudoseed → TW223 seed → TW223 output)
"""

from __future__ import annotations

import time

import pytest

from jackdaw.engine.rng import PseudoRandom

SEED_TOL = 1e-13
FLOAT_TOL = 1e-14


# ============================================================================
# Pools (simplified versions of the real game pools, matching the Lua oracle)
# ============================================================================

BOSS_POOL = {
    "bl_hook": "bl_hook",
    "bl_ox": "bl_ox",
    "bl_wall": "bl_wall",
    "bl_wheel": "bl_wheel",
    "bl_arm": "bl_arm",
    "bl_club": "bl_club",
    "bl_fish": "bl_fish",
    "bl_psychic": "bl_psychic",
    "bl_goad": "bl_goad",
    "bl_water": "bl_water",
}

TAG_POOL = {
    "tag_uncommon": "tag_uncommon",
    "tag_rare": "tag_rare",
    "tag_negative": "tag_negative",
    "tag_foil": "tag_foil",
    "tag_holo": "tag_holo",
    "tag_polychrome": "tag_polychrome",
    "tag_investment": "tag_investment",
    "tag_voucher": "tag_voucher",
    "tag_boss": "tag_boss",
    "tag_standard": "tag_standard",
    "tag_charm": "tag_charm",
    "tag_meteor": "tag_meteor",
    "tag_buffoon": "tag_buffoon",
    "tag_handy": "tag_handy",
    "tag_garbage": "tag_garbage",
    "tag_ethereal": "tag_ethereal",
    "tag_coupon": "tag_coupon",
    "tag_double": "tag_double",
    "tag_juggle": "tag_juggle",
    "tag_d6": "tag_d6",
}

JOKER_POOL = {
    "j_joker": "j_joker",
    "j_greedy_joker": "j_greedy_joker",
    "j_lusty_joker": "j_lusty_joker",
    "j_wrathful_joker": "j_wrathful_joker",
    "j_jolly": "j_jolly",
    "j_zany": "j_zany",
    "j_mad": "j_mad",
    "j_crazy": "j_crazy",
    "j_half": "j_half",
    "j_stencil": "j_stencil",
}


# ============================================================================
# LuaJIT 2.1 ground truth for seed "TESTSEED"
# ============================================================================

# Deck shuffle: pseudoseed('shuffle') → pseudoshuffle([1..52])
EXPECTED_DECK = [
    17,
    21,
    14,
    36,
    29,
    31,
    35,
    41,
    43,
    34,
    44,
    4,
    49,
    50,
    32,
    26,
    10,
    42,
    12,
    27,
    6,
    19,
    48,
    52,
    51,
    38,
    8,
    20,
    11,
    1,
    30,
    37,
    13,
    25,
    47,
    22,
    18,
    24,
    16,
    2,
    40,
    39,
    15,
    28,
    5,
    45,
    7,
    3,
    33,
    9,
    23,
    46,
]

# Boss blind: pseudorandom_element(BOSS_POOL, pseudoseed('boss'))
EXPECTED_BOSS = "bl_goad"

# Tags: 2 consecutive calls to pseudoseed('Tag1')
EXPECTED_TAGS = ["tag_voucher", "tag_investment"]

# Shop slots: 3 slots, each does cdt + rarity + pool select
EXPECTED_SHOP = [
    {"cdt": 0.323673317736492, "rarity": 0.127372906355584, "joker": "j_joker"},
    {"cdt": 0.537613412513260, "rarity": 0.777377281373973, "joker": "j_mad"},
    {"cdt": 0.134618964202504, "rarity": 0.309657059223484, "joker": "j_joker"},
]

# Rerolls: 3 more iterations of the same streams (continue advancing)
EXPECTED_REROLLS = [
    {"cdt": 0.605297975170146, "rarity": 0.863258824107398, "joker": "j_greedy_joker"},
    {"cdt": 0.572515377396789, "rarity": 0.854217398275141, "joker": "j_lusty_joker"},
    {"cdt": 0.371821413620441, "rarity": 0.298228735213785, "joker": "j_crazy"},
]

# Boss stream only advanced twice total (once at step 2, once at end)
EXPECTED_BOSS_CALL2 = 0.58145745166482732


# ============================================================================
# Game sequence simulation
# ============================================================================


class TestGameSequence:
    """Simulate the first few RNG calls of a real Balatro run."""

    @pytest.fixture
    def prng(self) -> PseudoRandom:
        return PseudoRandom("TESTSEED")

    def test_step1_deck_shuffle(self, prng: PseudoRandom):
        """Deck shuffle uses 'shuffle' stream, produces known order."""
        sv = prng.seed("shuffle")
        deck = list(range(1, 53))
        prng.shuffle(deck, sv)
        assert deck == EXPECTED_DECK

    def test_step2_boss_selection(self, prng: PseudoRandom):
        """Boss selection uses 'boss' stream, picks from sorted pool."""
        # Advance shuffle first (step 1 happened before us)
        prng.seed("shuffle")

        sv = prng.seed("boss")
        _, boss_key = prng.element(BOSS_POOL, sv)
        assert boss_key == EXPECTED_BOSS

    def test_step3_tag_generation(self, prng: PseudoRandom):
        """Two tags from 'Tag1' stream — consecutive calls, same stream."""
        prng.seed("shuffle")
        prng.seed("boss")

        for expected_tag in EXPECTED_TAGS:
            sv = prng.seed("Tag1")
            _, tag_key = prng.element(TAG_POOL, sv)
            assert tag_key == expected_tag

    def test_step4_shop_population(self, prng: PseudoRandom):
        """3 shop slots, each advancing cdt/rarity/pool streams."""
        # Replay prior steps
        prng.seed("shuffle")
        prng.seed("boss")
        prng.seed("Tag1")
        prng.seed("Tag1")

        ante = 1
        for slot_idx, expected in enumerate(EXPECTED_SHOP):
            sv_cdt = prng.seed(f"cdt{ante}")
            cdt_float = prng.random(sv_cdt)

            sv_rar = prng.seed(f"rarity{ante}sho")
            rarity_float = prng.random(sv_rar)

            sv_pool = prng.seed(f"Joker1sho{ante}")
            _, joker_key = prng.element(JOKER_POOL, sv_pool)

            assert abs(cdt_float - expected["cdt"]) < FLOAT_TOL, (
                f"shop[{slot_idx + 1}] cdt: {cdt_float:.15f} != {expected['cdt']:.15f}"
            )
            assert abs(rarity_float - expected["rarity"]) < FLOAT_TOL, (
                f"shop[{slot_idx + 1}] rarity: {rarity_float:.15f} != {expected['rarity']:.15f}"
            )
            assert joker_key == expected["joker"], (
                f"shop[{slot_idx + 1}] joker: {joker_key!r} != {expected['joker']!r}"
            )

    def test_step5_rerolls(self, prng: PseudoRandom):
        """3 rerolls continue advancing the same shop streams."""
        # Replay ALL prior steps
        prng.seed("shuffle")
        prng.seed("boss")
        prng.seed("Tag1")
        prng.seed("Tag1")

        ante = 1
        # Replay 3 shop slots
        for _ in range(3):
            prng.seed(f"cdt{ante}")
            prng.seed(f"rarity{ante}sho")
            prng.seed(f"Joker1sho{ante}")

        # Now 3 rerolls
        for reroll_idx, expected in enumerate(EXPECTED_REROLLS):
            sv_cdt = prng.seed(f"cdt{ante}")
            cdt_float = prng.random(sv_cdt)

            sv_rar = prng.seed(f"rarity{ante}sho")
            rarity_float = prng.random(sv_rar)

            sv_pool = prng.seed(f"Joker1sho{ante}")
            _, joker_key = prng.element(JOKER_POOL, sv_pool)

            assert abs(cdt_float - expected["cdt"]) < FLOAT_TOL, (
                f"reroll[{reroll_idx + 1}] cdt: {cdt_float:.15f} != {expected['cdt']:.15f}"
            )
            assert abs(rarity_float - expected["rarity"]) < FLOAT_TOL, (
                f"reroll[{reroll_idx + 1}] rarity"
            )
            assert joker_key == expected["joker"], (
                f"reroll[{reroll_idx + 1}] joker: {joker_key!r} != {expected['joker']!r}"
            )

    def test_stream_independence(self, prng: PseudoRandom):
        """Advancing shop streams must not affect the boss stream.

        Boss is advanced once at step 2.  After 6 shop iterations (3 slots +
        3 rerolls) each advancing 3 different streams, the boss stream's
        second call must still produce the expected value.
        """
        # Replay full sequence
        prng.seed("shuffle")  # shuffle: call 1
        prng.seed("boss")  # boss: call 1
        prng.seed("Tag1")  # Tag1: call 1
        prng.seed("Tag1")  # Tag1: call 2

        ante = 1
        for _ in range(6):  # 3 shop + 3 reroll
            prng.seed(f"cdt{ante}")
            prng.seed(f"rarity{ante}sho")
            prng.seed(f"Joker1sho{ante}")

        # Boss call 2 — should match oracle regardless of interleaved streams
        sv_boss2 = prng.seed("boss")
        assert abs(sv_boss2 - EXPECTED_BOSS_CALL2) < SEED_TOL, (
            f"boss call 2: {sv_boss2:.17g} != {EXPECTED_BOSS_CALL2:.17g}"
        )

    def test_full_sequence_in_one_pass(self, prng: PseudoRandom):
        """Run all steps sequentially in a single pass, verifying each.

        This is the most realistic test — a single PseudoRandom instance
        processes the full game-start sequence exactly as the engine would.
        """
        ante = 1

        # 1. Deck shuffle
        sv = prng.seed("shuffle")
        deck = list(range(1, 53))
        prng.shuffle(deck, sv)
        assert deck == EXPECTED_DECK

        # 2. Boss selection
        sv = prng.seed("boss")
        _, boss = prng.element(BOSS_POOL, sv)
        assert boss == EXPECTED_BOSS

        # 3. Tags
        for exp_tag in EXPECTED_TAGS:
            sv = prng.seed("Tag1")
            _, tag = prng.element(TAG_POOL, sv)
            assert tag == exp_tag

        # 4. Shop + 5. Rerolls (6 iterations total)
        for expected in EXPECTED_SHOP + EXPECTED_REROLLS:
            sv_cdt = prng.seed(f"cdt{ante}")
            cdt = prng.random(sv_cdt)
            assert abs(cdt - expected["cdt"]) < FLOAT_TOL

            sv_rar = prng.seed(f"rarity{ante}sho")
            rar = prng.random(sv_rar)
            assert abs(rar - expected["rarity"]) < FLOAT_TOL

            sv_pool = prng.seed(f"Joker1sho{ante}")
            _, jk = prng.element(JOKER_POOL, sv_pool)
            assert jk == expected["joker"]

        # 6. Boss call 2
        sv = prng.seed("boss")
        assert abs(sv - EXPECTED_BOSS_CALL2) < SEED_TOL


# ============================================================================
# Performance benchmark
# ============================================================================


class TestPerformance:
    """Benchmark the RNG system for RL training viability."""

    def test_pseudoseed_throughput(self):
        """1M pseudoseed advances must complete in reasonable time.

        Target: < 2 seconds for 1M calls (> 500K calls/sec).
        The RNG is called ~100-500 times per game action, and RL training
        may process millions of actions.
        """
        prng = PseudoRandom("BENCHMARK")
        n = 1_000_000

        start = time.perf_counter()
        for _ in range(n):
            prng.seed("bench")
        elapsed = time.perf_counter() - start

        rate = n / elapsed
        print(f"\n  pseudoseed: {n:,} calls in {elapsed:.3f}s = {rate:,.0f} calls/sec")
        # Fail if unreasonably slow (> 5 seconds)
        assert elapsed < 5.0, f"Too slow: {elapsed:.1f}s for {n:,} calls"

    def test_pseudorandom_throughput(self):
        """1M pseudorandom (full pipeline: seed → TW223 → output) calls."""
        prng = PseudoRandom("BENCHMARK")
        n = 1_000_000

        start = time.perf_counter()
        for _ in range(n):
            prng.random("bench")
        elapsed = time.perf_counter() - start

        rate = n / elapsed
        print(f"\n  pseudorandom: {n:,} calls in {elapsed:.3f}s = {rate:,.0f} calls/sec")
        assert elapsed < 10.0, f"Too slow: {elapsed:.1f}s for {n:,} calls"

    def test_pseudoshuffle_throughput(self):
        """10K deck shuffles (52 cards each)."""
        prng = PseudoRandom("BENCHMARK")
        n = 10_000

        start = time.perf_counter()
        for i in range(n):
            deck = list(range(52))
            prng.shuffle(deck, float(i) / n)
        elapsed = time.perf_counter() - start

        rate = n / elapsed
        print(f"\n  pseudoshuffle(52): {n:,} calls in {elapsed:.3f}s = {rate:,.0f} shuffles/sec")
        assert elapsed < 10.0, f"Too slow: {elapsed:.1f}s for {n:,} shuffles"

    def test_truncate_13_is_not_bottleneck(self):
        """Profile whether float formatting is the hot path.

        Compare pseudoseed (which truncates) vs raw float math (which doesn't).
        """
        prng = PseudoRandom("BENCHMARK")
        n = 500_000

        # With truncation (actual pseudoseed)
        start = time.perf_counter()
        for _ in range(n):
            prng.seed("trunc_test")
        t_with = time.perf_counter() - start

        # Raw float math only (no truncation, no string formatting)
        val = 0.5
        start = time.perf_counter()
        for _ in range(n):
            val = abs((2.134453429141 + val * 1.72431234) % 1)
        t_raw = time.perf_counter() - start

        overhead = t_with / max(t_raw, 1e-9)
        print(f"\n  pseudoseed({n:,}): {t_with:.3f}s")
        print(f"  raw float math({n:,}): {t_raw:.3f}s")
        print(f"  overhead factor: {overhead:.1f}x")
        # The overhead is expected (dict lookups, function calls, string
        # formatting).  Just documenting it, not asserting a threshold.
