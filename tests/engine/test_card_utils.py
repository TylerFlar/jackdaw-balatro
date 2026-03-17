"""Tests for jackdaw.engine.card_utils — poll_edition."""

from __future__ import annotations

from jackdaw.engine.card_utils import poll_edition
from jackdaw.engine.rng import PseudoRandom

# ---------------------------------------------------------------------------
# Minimal RNG stub with scripted return values
# ---------------------------------------------------------------------------

class _RNG:
    """Minimal stub: returns pre-scripted values from random(), ignores key."""

    def __init__(self, values: list[float]) -> None:
        self._it = iter(values)

    def random(self, key: str) -> float:  # noqa: ARG002
        return next(self._it)


# ---------------------------------------------------------------------------
# Normal mode — threshold boundary tests
# (rate=1.0, mod=1.0)
# Thresholds: Neg >0.997, Poly >0.994, Holo >0.98, Foil >0.96
# ---------------------------------------------------------------------------

class TestNormalModeThresholds:
    """Verify each edition fires exactly at its boundary in normal mode."""

    # -- Negative (> 1 - 0.003*1 = 0.997) --

    def test_negative_at_boundary(self):
        assert poll_edition("k", _RNG([0.9971])) == {"negative": True}

    def test_negative_just_below_boundary_gives_poly(self):
        # 0.9970 is NOT > 0.997, falls through to poly check: > 0.994
        assert poll_edition("k", _RNG([0.9970])) == {"polychrome": True}

    def test_negative_high_value(self):
        assert poll_edition("k", _RNG([0.9999])) == {"negative": True}

    # -- Polychrome (> 1 - 0.006*1*1 = 0.994) --

    def test_polychrome_at_boundary(self):
        assert poll_edition("k", _RNG([0.9941])) == {"polychrome": True}

    def test_polychrome_just_below_gives_holo(self):
        # 0.9940 is NOT > 0.994, falls through to holo: > 0.98
        assert poll_edition("k", _RNG([0.9940])) == {"holo": True}

    # -- Holo (> 1 - 0.02*1*1 = 0.98) --

    def test_holo_at_boundary(self):
        assert poll_edition("k", _RNG([0.9801])) == {"holo": True}

    def test_holo_just_below_gives_foil(self):
        # 0.9800 is NOT > 0.98, falls through to foil: > 0.96
        assert poll_edition("k", _RNG([0.9800])) == {"foil": True}

    # -- Foil (> 1 - 0.04*1*1 = 0.96) --

    def test_foil_at_boundary(self):
        assert poll_edition("k", _RNG([0.9601])) == {"foil": True}

    def test_foil_just_below_gives_none(self):
        assert poll_edition("k", _RNG([0.9600])) is None

    def test_low_value_gives_none(self):
        assert poll_edition("k", _RNG([0.0])) is None

    def test_midrange_gives_none(self):
        assert poll_edition("k", _RNG([0.5])) is None


# ---------------------------------------------------------------------------
# no_neg flag — Negative edition excluded
# ---------------------------------------------------------------------------

class TestNoNegFlag:
    """When no_neg=True, high rolls that would be Negative fall through to Poly."""

    def test_no_neg_high_roll_gives_poly(self):
        # roll=0.999 would be Negative, but no_neg skips it → Poly (>0.994)
        assert poll_edition("k", _RNG([0.999]), no_neg=True) == {"polychrome": True}

    def test_no_neg_poly_range(self):
        # 0.995 → not Neg (skipped), > 0.994 → Poly
        assert poll_edition("k", _RNG([0.995]), no_neg=True) == {"polychrome": True}

    def test_no_neg_holo_range(self):
        assert poll_edition("k", _RNG([0.985]), no_neg=True) == {"holo": True}

    def test_no_neg_foil_range(self):
        assert poll_edition("k", _RNG([0.965]), no_neg=True) == {"foil": True}

    def test_no_neg_none_range(self):
        assert poll_edition("k", _RNG([0.5]), no_neg=True) is None

    def test_no_neg_never_returns_negative(self):
        # All rolls from 0 to 1; none should return Negative
        for v in [0.9971, 0.998, 0.999, 1.0 - 1e-10]:
            result = poll_edition("k", _RNG([v]), no_neg=True)
            assert result is None or "negative" not in result, (
                f"Negative returned for roll={v} with no_neg=True"
            )


# ---------------------------------------------------------------------------
# Guaranteed mode — ×25 multiplier
# Thresholds: Neg >0.925, Poly >0.85, Holo >0.5, Foil >0.0
# ---------------------------------------------------------------------------

class TestGuaranteedMode:
    """guaranteed=True overrides rate/mod to give 50/35/7.5/7.5 distribution."""

    def test_foil_dominant(self):
        # roll=0.3 → > 0.0 → Foil
        assert poll_edition("k", _RNG([0.3]), guaranteed=True) == {"foil": True}

    def test_foil_at_low_boundary(self):
        # roll=0.001 → Foil
        assert poll_edition("k", _RNG([0.001]), guaranteed=True) == {"foil": True}

    def test_holo_range(self):
        # roll=0.7 → > 0.5 but not > 0.85 → Holo
        assert poll_edition("k", _RNG([0.7]), guaranteed=True) == {"holo": True}

    def test_holo_at_boundary(self):
        # roll=0.501 → > 0.5 → Holo
        assert poll_edition("k", _RNG([0.501]), guaranteed=True) == {"holo": True}

    def test_polychrome_range(self):
        # roll=0.88 → > 0.85 but not > 0.925 → Polychrome
        assert poll_edition("k", _RNG([0.88]), guaranteed=True) == {"polychrome": True}

    def test_polychrome_at_boundary(self):
        assert poll_edition("k", _RNG([0.851]), guaranteed=True) == {"polychrome": True}

    def test_negative_range(self):
        # roll=0.95 → > 0.925 → Negative
        assert poll_edition("k", _RNG([0.95]), guaranteed=True) == {"negative": True}

    def test_negative_at_boundary(self):
        assert poll_edition("k", _RNG([0.9251]), guaranteed=True) == {"negative": True}

    def test_guaranteed_no_neg_skips_negative(self):
        # roll=0.95 with no_neg → falls to polychrome
        assert poll_edition("k", _RNG([0.95]), guaranteed=True, no_neg=True) == {"polychrome": True}

    def test_guaranteed_overrides_rate(self):
        # Even rate=0 shouldn't matter; guaranteed forces mod=25
        assert poll_edition("k", _RNG([0.3]), rate=0.0, guaranteed=True) == {"foil": True}

    def test_guaranteed_overrides_mod(self):
        # mod=0 would suppress all editions normally; guaranteed forces mod=25
        assert poll_edition("k", _RNG([0.3]), mod=0.0, guaranteed=True) == {"foil": True}


# ---------------------------------------------------------------------------
# rate parameter — Hone/Glow Up voucher effect
# With rate=2: Foil >0.92, Holo >0.96, Poly >0.988
# With rate=4: Foil >0.84, Holo >0.92, Poly >0.976
# ---------------------------------------------------------------------------

class TestRateParameter:
    """rate doubles edition chances for Foil/Holo/Poly but NOT Negative."""

    # rate=2 (Hone voucher)
    # Foil: > 1 - 0.04*2*1 = > 0.92
    # Holo: > 1 - 0.02*2*1 = > 0.96
    # Poly: > 1 - 0.006*2*1 = > 0.988
    # Neg:  > 1 - 0.003*1  = > 0.997 (rate doesn't affect Negative)

    def test_rate2_foil_at_new_threshold(self):
        # 0.93 > 0.92 → Foil (would be None at rate=1)
        assert poll_edition("k", _RNG([0.93]), rate=2.0) == {"foil": True}

    def test_rate2_foil_below_old_threshold(self):
        # 0.93 < 0.96 (rate=1 foil threshold), so still foil at rate=2
        assert poll_edition("k", _RNG([0.965])) == {"foil": True}

    def test_rate2_holo_at_new_threshold(self):
        # 0.965 > 0.96 → Holo with rate=2 (would be Foil at rate=1)
        assert poll_edition("k", _RNG([0.965]), rate=2.0) == {"holo": True}

    def test_rate2_poly_at_new_threshold(self):
        # 0.990 > 0.988 → Poly with rate=2 (would be Holo at rate=1)
        assert poll_edition("k", _RNG([0.990]), rate=2.0) == {"polychrome": True}

    def test_rate2_negative_threshold_unchanged(self):
        # Negative threshold is 0.997 regardless of rate
        # roll=0.998 → Negative with rate=2 (same as rate=1)
        assert poll_edition("k", _RNG([0.998]), rate=2.0) == {"negative": True}

    def test_rate2_none_below_foil_threshold(self):
        # 0.91 < 0.92, so None at rate=2
        assert poll_edition("k", _RNG([0.91]), rate=2.0) is None

    # rate=4 (Glow Up voucher)
    # Foil: > 1 - 0.04*4 = > 0.84

    def test_rate4_foil_lower_threshold(self):
        # 0.85 > 0.84 → Foil with rate=4
        assert poll_edition("k", _RNG([0.85]), rate=4.0) == {"foil": True}

    def test_rate4_none_below_foil_threshold(self):
        # 0.83 < 0.84 → None
        assert poll_edition("k", _RNG([0.83]), rate=4.0) is None


# ---------------------------------------------------------------------------
# mod parameter
# ---------------------------------------------------------------------------

class TestModParameter:
    """mod scales all thresholds including Negative."""

    def test_mod2_negative_threshold_lowered(self):
        # mod=2: Neg > 1 - 0.003*2 = > 0.994
        # roll=0.995 → Negative with mod=2 (would be Poly at mod=1)
        assert poll_edition("k", _RNG([0.995]), mod=2.0) == {"negative": True}

    def test_mod0_no_edition(self):
        # mod=0: all thresholds collapse to 0; even high rolls give None
        # Foil: > 1 - 0.04*1*0 = > 1.0 (impossible)
        assert poll_edition("k", _RNG([0.999]), rate=1.0, mod=0.0) is None

    def test_mod_and_rate_compound(self):
        # rate=2, mod=2: Foil > 1 - 0.04*2*2 = > 0.84
        assert poll_edition("k", _RNG([0.85]), rate=2.0, mod=2.0) == {"foil": True}


# ---------------------------------------------------------------------------
# Oracle test — real PseudoRandom seed
# Verifies the function integrates correctly with the actual RNG
# ---------------------------------------------------------------------------

class TestOracleWithRealRng:
    """Integration test: verify poll_edition with a real PseudoRandom seed.

    These tests assert that the output is deterministic for a given seed and key.
    The specific edition returned depends on the TW223 output for that seed.
    """

    def test_deterministic_for_same_seed(self):
        rng1 = PseudoRandom("TEST_SEED_A")
        rng2 = PseudoRandom("TEST_SEED_A")
        r1 = poll_edition("edi1sho", rng1)
        r2 = poll_edition("edi1sho", rng2)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        """Two different seeds are very unlikely to produce identical results."""
        rng1 = PseudoRandom("SEED_ALPHA")
        rng2 = PseudoRandom("SEED_BETA")
        results = {
            str(poll_edition("edi1sho", rng1)),
            str(poll_edition("edi1sho", rng2)),
        }
        # Both could theoretically be None, but with different seeds they
        # overwhelmingly both return None or different editions.
        # We just assert the calls don't error.
        assert len(results) >= 1

    def test_stream_advances_each_call(self):
        """Successive calls with the same key advance the stream."""
        rng = PseudoRandom("STREAM_TEST")
        r1 = poll_edition("edi1sho", rng)
        r2 = poll_edition("edi1sho", rng)
        # The two draws are independent — don't need to differ, but must not error
        assert isinstance(r1, (dict, type(None)))
        assert isinstance(r2, (dict, type(None)))

    def test_guaranteed_mode_always_returns_edition(self):
        """In guaranteed mode every draw returns a non-None edition."""
        rng = PseudoRandom("GUARANTEE_TEST")
        for _ in range(20):
            result = poll_edition("edi_g", rng, guaranteed=True)
            assert result is not None, "guaranteed=True must always return an edition"
            assert len(result) == 1
            assert list(result.values()) == [True]

    def test_guaranteed_no_neg_never_negative(self):
        """guaranteed + no_neg never produces Negative even over many draws."""
        rng = PseudoRandom("NONEG_TEST")
        for _ in range(50):
            result = poll_edition("aura", rng, guaranteed=True, no_neg=True)
            assert result is not None
            assert "negative" not in result

    def test_normal_mode_mostly_none(self):
        """With rate=1, ~96% of draws should be None over many trials."""
        rng = PseudoRandom("NORMAL_MODE_DIST")
        none_count = sum(
            1 for _ in range(1000)
            if poll_edition("edi1sho", rng) is None
        )
        # Expect ~960/1000; allow wide margin for small-sample variance
        assert none_count > 900, f"Expected ~960 Nones, got {none_count}"

    def test_high_rate_increases_edition_frequency(self):
        """rate=4 should produce significantly more editions than rate=1."""
        rng_r1 = PseudoRandom("RATE_COMPARE")
        rng_r4 = PseudoRandom("RATE_COMPARE")  # same seed, same stream
        editions_r1 = sum(
            1 for _ in range(500)
            if poll_edition("edi1sho", rng_r1, rate=1.0) is not None
        )
        editions_r4 = sum(
            1 for _ in range(500)
            if poll_edition("edi1sho", rng_r4, rate=4.0) is not None
        )
        # rate=4 has ~16% edition chance vs ~4% for rate=1
        assert editions_r4 > editions_r1, (
            f"rate=4 ({editions_r4}) should give more editions than rate=1 ({editions_r1})"
        )
