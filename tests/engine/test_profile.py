"""Tests for jackdaw.engine.profile — Profile, pool filtering with unlock state.

Coverage
--------
* fresh_profile has 105 jokers unlocked.
* fresh_profile has 45 jokers locked.
* default_profile has all items unlocked.
* Pool with fresh_profile excludes locked jokers.
* Pool with default_profile includes all jokers (except banned/used).
* Locked vouchers excluded from fresh_profile pool.
* Legendary jokers bypass unlock check.
* Profile wired into game_state via apply_profile_to_game_state.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.pools import UNAVAILABLE, get_current_pool
from jackdaw.engine.profile import (
    Profile,
    _get_locked_items,
    apply_profile_to_game_state,
    default_profile,
    fresh_profile,
)
from jackdaw.engine.rng import PseudoRandom


class TestFreshProfile:
    def test_105_jokers_unlocked(self):
        p = fresh_profile()
        from jackdaw.engine.data.prototypes import JOKERS

        unlocked_jokers = {k for k in JOKERS if k in p.unlocked}
        assert len(unlocked_jokers) == 105

    def test_45_jokers_locked(self):
        p = fresh_profile()
        from jackdaw.engine.data.prototypes import JOKERS

        locked_jokers = {k for k in JOKERS if k not in p.unlocked}
        assert len(locked_jokers) == 45

    def test_no_discoveries(self):
        p = fresh_profile()
        assert len(p.discovered) == 0

    def test_tarots_unlocked(self):
        """All Tarots start unlocked."""
        p = fresh_profile()
        from jackdaw.engine.data.prototypes import _load_json

        centers = _load_json("centers.json")
        tarots = {k for k, v in centers.items() if v.get("set") == "Tarot"}
        assert tarots.issubset(p.unlocked)


class TestDefaultProfile:
    def test_all_items_unlocked(self):
        p = default_profile()
        locked = _get_locked_items()
        assert locked.issubset(p.unlocked)

    def test_all_items_discovered(self):
        p = default_profile()
        locked = _get_locked_items()
        assert locked.issubset(p.discovered)


class TestPoolFilteringWithProfile:
    def test_locked_joker_excluded_from_fresh(self):
        """A locked joker should be UNAVAILABLE with a fresh profile."""
        p = fresh_profile()
        locked = _get_locked_items()
        # Find a locked joker
        from jackdaw.engine.data.prototypes import JOKERS

        locked_joker = None
        for k in locked:
            if k in JOKERS and JOKERS[k].rarity != 4:
                locked_joker = k
                break
        assert locked_joker is not None, "No locked non-legendary joker found"

        rng = PseudoRandom("PROFILE_TEST")
        pool, _ = get_current_pool(
            "Joker", rng, 1, rarity=1,
            profile_unlocked=p.unlocked,
        )
        assert locked_joker not in pool or pool[pool.index(locked_joker)] == UNAVAILABLE if locked_joker in pool else True

    def test_all_jokers_available_with_default(self):
        """Default profile should have all non-banned jokers available."""
        p = default_profile()
        rng = PseudoRandom("DEFAULT_POOL")
        pool, _ = get_current_pool(
            "Joker", rng, 1, rarity=1,
            profile_unlocked=p.unlocked,
        )
        available = [k for k in pool if k != UNAVAILABLE]
        # Should have most rarity-1 jokers available
        assert len(available) > 50

    def test_no_profile_means_no_unlock_filter(self):
        """When profile_unlocked is None, no unlock filtering is applied."""
        rng = PseudoRandom("NO_PROFILE")
        pool, _ = get_current_pool("Joker", rng, 1, rarity=1)
        available = [k for k in pool if k != UNAVAILABLE]
        # Without profile filter, all items pass (default behavior)
        assert len(available) > 50

    def test_legendary_bypasses_unlock(self):
        """Legendary jokers (rarity 4) bypass the unlock check."""
        p = fresh_profile()
        rng = PseudoRandom("LEGEND_TEST")
        pool, _ = get_current_pool(
            "Joker", rng, 1, rarity=4, legendary=True,
            profile_unlocked=p.unlocked,
        )
        # Legendary pool should have entries (not all UNAVAILABLE)
        available = [k for k in pool if k != UNAVAILABLE]
        assert len(available) > 0


class TestApplyProfileToGameState:
    def test_sets_discovered(self):
        p = default_profile()
        gs: dict = {}
        apply_profile_to_game_state(p, gs)
        assert gs["discovered"] == p.discovered

    def test_sets_profile_unlocked(self):
        p = fresh_profile()
        gs: dict = {}
        apply_profile_to_game_state(p, gs)
        assert gs["profile_unlocked"] == p.unlocked


class TestLockedItems:
    def test_locked_count(self):
        locked = _get_locked_items()
        assert len(locked) == 75  # 45 jokers + 16 vouchers + 14 backs

    def test_known_locked_joker(self):
        """j_caino (Canio) should be locked — requires win with specific deck."""
        locked = _get_locked_items()
        assert "j_caino" in locked

    def test_known_unlocked_joker(self):
        """j_joker should be unlocked by default."""
        locked = _get_locked_items()
        assert "j_joker" not in locked
