"""Tests for jackdaw.engine.stakes — apply_stake_modifiers.

Coverage
--------
* Stake 1: no modifiers applied at all.
* Stake 2: Small Blind reward removed.
* Stake 3: scaling = 2.
* Stake 4: enable_eternals_in_shop = True, scaling still 2.
* Stake 5: discards decremented, all stake-4 modifiers present.
* Stake 6: scaling = 3 (overrides stake-3's 2), all lower mods present.
* Stake 7: enable_perishables_in_shop = True, scaling = 3.
* Stake 8: all modifiers present — eternals + perishables + rentals.
* Back + stake integration: Red Deck at stake 5, Black Deck at stake 8.
* game_state isolation: modifiers sub-dict created lazily.
* Idempotency: calling twice gives same result as calling once.
"""

from __future__ import annotations

import copy
from typing import Any

from jackdaw.engine.back import Back
from jackdaw.engine.stakes import DEFAULT_STARTING_PARAMS, apply_stake_modifiers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gs(extra_params: dict | None = None) -> dict[str, Any]:
    """Return a fresh game_state with default starting_params."""
    params = dict(DEFAULT_STARTING_PARAMS)
    if extra_params:
        params.update(extra_params)
    return {"starting_params": params}


def _apply(stake: int, extra_params: dict | None = None) -> dict[str, Any]:
    """Apply stake modifiers and return the game_state for inspection."""
    gs = _gs(extra_params)
    apply_stake_modifiers(stake, gs)
    return gs


# ---------------------------------------------------------------------------
# Stake 1 (White): no modifiers
# ---------------------------------------------------------------------------


class TestStake1:
    def test_no_modifiers_key(self):
        gs = _apply(1)
        assert gs.get("modifiers", {}) == {}

    def test_discards_unchanged(self):
        gs = _apply(1)
        assert gs["starting_params"]["discards"] == 3

    def test_hands_unchanged(self):
        gs = _apply(1)
        assert gs["starting_params"]["hands"] == 4

    def test_modifiers_not_created(self):
        """Stake 1 must not add an empty 'modifiers' dict."""
        gs = {"starting_params": dict(DEFAULT_STARTING_PARAMS)}
        apply_stake_modifiers(1, gs)
        assert "modifiers" not in gs


# ---------------------------------------------------------------------------
# Stake 2 (Red): Small Blind no reward
# ---------------------------------------------------------------------------


class TestStake2:
    def test_no_blind_reward_small(self):
        gs = _apply(2)
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True

    def test_scaling_not_set(self):
        gs = _apply(2)
        assert "scaling" not in gs["modifiers"]

    def test_eternals_not_set(self):
        gs = _apply(2)
        assert "enable_eternals_in_shop" not in gs["modifiers"]

    def test_discards_unchanged(self):
        gs = _apply(2)
        assert gs["starting_params"]["discards"] == 3


# ---------------------------------------------------------------------------
# Stake 3 (Green): scaling = 2
# ---------------------------------------------------------------------------


class TestStake3:
    def test_scaling_is_2(self):
        gs = _apply(3)
        assert gs["modifiers"]["scaling"] == 2

    def test_no_blind_reward_small_still_set(self):
        gs = _apply(3)
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True

    def test_eternals_not_set(self):
        gs = _apply(3)
        assert "enable_eternals_in_shop" not in gs["modifiers"]


# ---------------------------------------------------------------------------
# Stake 4 (Black): eternals in shop
# ---------------------------------------------------------------------------


class TestStake4:
    def test_enable_eternals(self):
        gs = _apply(4)
        assert gs["modifiers"]["enable_eternals_in_shop"] is True

    def test_scaling_still_2(self):
        gs = _apply(4)
        assert gs["modifiers"]["scaling"] == 2

    def test_perishables_not_set(self):
        gs = _apply(4)
        assert "enable_perishables_in_shop" not in gs["modifiers"]

    def test_discards_unchanged(self):
        gs = _apply(4)
        assert gs["starting_params"]["discards"] == 3


# ---------------------------------------------------------------------------
# Stake 5 (Blue): -1 discard, all stake-4 effects
# ---------------------------------------------------------------------------


class TestStake5:
    def test_discards_decremented(self):
        gs = _apply(5)
        assert gs["starting_params"]["discards"] == 2  # 3 - 1

    def test_enable_eternals_still_set(self):
        gs = _apply(5)
        assert gs["modifiers"]["enable_eternals_in_shop"] is True

    def test_scaling_still_2(self):
        gs = _apply(5)
        assert gs["modifiers"]["scaling"] == 2

    def test_no_blind_reward_small_still_set(self):
        gs = _apply(5)
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True

    def test_perishables_not_set(self):
        gs = _apply(5)
        assert "enable_perishables_in_shop" not in gs["modifiers"]

    def test_hands_unchanged(self):
        gs = _apply(5)
        assert gs["starting_params"]["hands"] == 4


# ---------------------------------------------------------------------------
# Stake 6 (Purple): scaling = 3, overrides Green's 2
# ---------------------------------------------------------------------------


class TestStake6:
    def test_scaling_is_3(self):
        gs = _apply(6)
        assert gs["modifiers"]["scaling"] == 3

    def test_scaling_is_not_2(self):
        """Purple must replace Green's 2 — not leave it at 2."""
        gs = _apply(6)
        assert gs["modifiers"]["scaling"] != 2

    def test_eternals_still_set(self):
        gs = _apply(6)
        assert gs["modifiers"]["enable_eternals_in_shop"] is True

    def test_discards_decremented(self):
        gs = _apply(6)
        assert gs["starting_params"]["discards"] == 2

    def test_perishables_not_set(self):
        gs = _apply(6)
        assert "enable_perishables_in_shop" not in gs["modifiers"]


# ---------------------------------------------------------------------------
# Stake 7 (Orange): perishables in shop
# ---------------------------------------------------------------------------


class TestStake7:
    def test_enable_perishables(self):
        gs = _apply(7)
        assert gs["modifiers"]["enable_perishables_in_shop"] is True

    def test_scaling_is_3(self):
        gs = _apply(7)
        assert gs["modifiers"]["scaling"] == 3

    def test_eternals_still_set(self):
        gs = _apply(7)
        assert gs["modifiers"]["enable_eternals_in_shop"] is True

    def test_rentals_not_set(self):
        gs = _apply(7)
        assert "enable_rentals_in_shop" not in gs["modifiers"]


# ---------------------------------------------------------------------------
# Stake 8 (Gold): all modifiers
# ---------------------------------------------------------------------------


class TestStake8:
    def test_enable_rentals(self):
        gs = _apply(8)
        assert gs["modifiers"]["enable_rentals_in_shop"] is True

    def test_enable_eternals(self):
        gs = _apply(8)
        assert gs["modifiers"]["enable_eternals_in_shop"] is True

    def test_enable_perishables(self):
        gs = _apply(8)
        assert gs["modifiers"]["enable_perishables_in_shop"] is True

    def test_scaling_is_3(self):
        gs = _apply(8)
        assert gs["modifiers"]["scaling"] == 3

    def test_no_blind_reward_small(self):
        gs = _apply(8)
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True

    def test_discards_decremented_once(self):
        """Blue Stake fires only once regardless of stake level."""
        gs = _apply(8)
        assert gs["starting_params"]["discards"] == 2  # 3 - 1

    def test_all_three_sticker_flags_set(self):
        gs = _apply(8)
        mods = gs["modifiers"]
        assert mods["enable_eternals_in_shop"] is True
        assert mods["enable_perishables_in_shop"] is True
        assert mods["enable_rentals_in_shop"] is True


# ---------------------------------------------------------------------------
# Back + stake integration
# ---------------------------------------------------------------------------


class TestBackAndStakeIntegration:
    """Combine Back.apply_to_run mutations with apply_stake_modifiers."""

    def _build_params_from_back(self, back_key: str) -> dict[str, int]:
        """Apply back mutations on top of default starting params."""
        params = dict(DEFAULT_STARTING_PARAMS)
        mutations = Back(back_key).apply_to_run({})
        if "hands_delta" in mutations:
            params["hands"] += mutations["hands_delta"]
        if "discards_delta" in mutations:
            params["discards"] += mutations["discards_delta"]
        if "hand_size_delta" in mutations:
            params["hand_size"] += mutations["hand_size_delta"]
        if "joker_slots_delta" in mutations:
            params["joker_slots"] += mutations["joker_slots_delta"]
        if "consumable_slots_delta" in mutations:
            params["consumable_slots"] += mutations["consumable_slots_delta"]
        return params

    def test_red_deck_stake5_discards(self):
        """Red Deck (+1 discard) at stake 5 (−1 discard) = 3 discards total."""
        # Red Deck: discards_delta = +1 → 3 + 1 = 4
        # Blue Stake (5): discards -= 1 → 4 - 1 = 3
        params = self._build_params_from_back("b_red")
        assert params["discards"] == 4  # base 3 + red deck +1
        gs = {"starting_params": params}
        apply_stake_modifiers(5, gs)
        assert gs["starting_params"]["discards"] == 3

    def test_black_deck_stake8_hands(self):
        """Black Deck (−1 hand) at stake 8 = 4 - 1 = 3 hands."""
        params = self._build_params_from_back("b_black")
        assert params["hands"] == 3  # base 4 + black deck -1
        gs = {"starting_params": params}
        apply_stake_modifiers(8, gs)
        assert gs["starting_params"]["hands"] == 3  # stake 8 doesn't touch hands

    def test_black_deck_stake8_joker_slots(self):
        """Black Deck (+1 joker slot) at stake 8 = 5 + 1 = 6."""
        params = self._build_params_from_back("b_black")
        assert params["joker_slots"] == 6  # base 5 + black deck +1
        gs = {"starting_params": params}
        apply_stake_modifiers(8, gs)
        assert gs["starting_params"]["joker_slots"] == 6  # stake 8 doesn't touch slots

    def test_black_deck_stake8_all_sticker_flags(self):
        """Black Deck at stake 8 gets all three sticker-enable flags."""
        params = self._build_params_from_back("b_black")
        gs = {"starting_params": params}
        apply_stake_modifiers(8, gs)
        mods = gs["modifiers"]
        assert mods["enable_eternals_in_shop"] is True
        assert mods["enable_perishables_in_shop"] is True
        assert mods["enable_rentals_in_shop"] is True

    def test_black_deck_stake8_discards(self):
        """Black Deck (no discard delta) at stake 8: 3 - 1 = 2 discards."""
        params = self._build_params_from_back("b_black")
        assert params["discards"] == 3  # Black Deck doesn't change discards
        gs = {"starting_params": params}
        apply_stake_modifiers(8, gs)
        assert gs["starting_params"]["discards"] == 2  # blue stake -1

    def test_blue_deck_stake1_hands_unchanged(self):
        """Blue Deck (+1 hand) at stake 1 = 4 + 1 = 5 hands, no stake mods."""
        params = self._build_params_from_back("b_blue")
        assert params["hands"] == 5
        gs = {"starting_params": params}
        apply_stake_modifiers(1, gs)
        assert gs["starting_params"]["hands"] == 5
        assert "modifiers" not in gs

    def test_painted_deck_stake5_joker_slots(self):
        """Painted Deck (−1 joker slot) at stake 5 = 5 - 1 = 4 joker slots."""
        params = self._build_params_from_back("b_painted")
        assert params["joker_slots"] == 4  # base 5 + painted -1
        gs = {"starting_params": params}
        apply_stake_modifiers(5, gs)
        assert gs["starting_params"]["joker_slots"] == 4  # stake 5 doesn't touch joker slots


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_modifiers_created_lazily_at_stake_2(self):
        """modifiers sub-dict is created by setdefault at stake >= 2."""
        gs = {"starting_params": dict(DEFAULT_STARTING_PARAMS)}
        assert "modifiers" not in gs
        apply_stake_modifiers(2, gs)
        assert "modifiers" in gs

    def test_existing_modifiers_dict_not_replaced(self):
        """Pre-existing modifiers dict entries are preserved."""
        gs = _gs()
        gs["modifiers"] = {"custom_flag": True}
        apply_stake_modifiers(4, gs)
        assert gs["modifiers"]["custom_flag"] is True
        assert gs["modifiers"]["enable_eternals_in_shop"] is True

    def test_no_blind_reward_accumulates_existing_entries(self):
        """Pre-existing no_blind_reward entries are preserved."""
        gs = _gs()
        gs["modifiers"] = {"no_blind_reward": {"Big": True}}
        apply_stake_modifiers(2, gs)
        assert gs["modifiers"]["no_blind_reward"]["Big"] is True
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True

    def test_calling_twice_same_result(self):
        """apply_stake_modifiers is idempotent for flags; discards decrements twice."""
        gs1 = _gs()
        apply_stake_modifiers(4, gs1)
        gs2 = copy.deepcopy(gs1)
        apply_stake_modifiers(4, gs2)
        # Flags are still True (idempotent for booleans)
        assert gs2["modifiers"]["enable_eternals_in_shop"] is True
        # But discards is NOT decremented at stake 4 — only at stake 5+
        assert gs2["starting_params"]["discards"] == 3

    def test_stake_above_8_treated_as_8(self):
        """Stake values > 8 don't add new effects but don't crash."""
        gs = _gs()
        apply_stake_modifiers(10, gs)
        mods = gs["modifiers"]
        assert mods["enable_rentals_in_shop"] is True
        assert mods["scaling"] == 3

    def test_default_starting_params_has_expected_keys(self):
        assert DEFAULT_STARTING_PARAMS["hands"] == 4
        assert DEFAULT_STARTING_PARAMS["discards"] == 3
        assert DEFAULT_STARTING_PARAMS["hand_size"] == 8
        assert DEFAULT_STARTING_PARAMS["joker_slots"] == 5
        assert DEFAULT_STARTING_PARAMS["consumable_slots"] == 2
