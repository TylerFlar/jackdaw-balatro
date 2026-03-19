"""Tests for the factored action space module.

Covers:
- get_action_mask in each phase (blind_select, selecting_hand, shop, pack_opening, round_eval)
- factored_to_engine roundtrip for every action type
- Swap operations produce correct permutations
- Consumable targeting masks are correct
- PlayHand card_target with 1-5 cards
- Empty entity lists produce all-False masks
- type_mask consistency with engine's get_legal_actions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from jackdaw.engine.actions import (
    BuyCard as EngineBuyCard,
)
from jackdaw.engine.actions import (
    CashOut as EngineCashOut,
)
from jackdaw.engine.actions import (
    Discard as EngineDiscard,
)
from jackdaw.engine.actions import (
    GamePhase,
    get_legal_actions,
)
from jackdaw.engine.actions import (
    NextRound as EngineNextRound,
)
from jackdaw.engine.actions import (
    OpenBooster as EngineOpenBooster,
)
from jackdaw.engine.actions import (
    PickPackCard as EnginePickPackCard,
)
from jackdaw.engine.actions import (
    PlayHand as EnginePlayHand,
)
from jackdaw.engine.actions import (
    RedeemVoucher as EngineRedeemVoucher,
)
from jackdaw.engine.actions import (
    ReorderHand as EngineReorderHand,
)
from jackdaw.engine.actions import (
    ReorderJokers as EngineReorderJokers,
)
from jackdaw.engine.actions import (
    Reroll as EngineReroll,
)
from jackdaw.engine.actions import (
    SelectBlind as EngineSelectBlind,
)
from jackdaw.engine.actions import (
    SellCard as EngineSellCard,
)
from jackdaw.engine.actions import (
    SkipBlind as EngineSkipBlind,
)
from jackdaw.engine.actions import (
    SkipPack as EngineSkipPack,
)
from jackdaw.engine.actions import (
    SortHand as EngineSortHand,
)
from jackdaw.engine.actions import (
    UseConsumable as EngineUseConsumable,
)
from jackdaw.env.action_space import (
    NUM_ACTION_TYPES,
    ActionType,
    FactoredAction,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
    get_consumable_target_info,
)

# ---------------------------------------------------------------------------
# Lightweight mock Card
# ---------------------------------------------------------------------------


@dataclass
class MockCard:
    """Minimal mock card for testing action space logic."""

    center_key: str = "c_base"
    ability: dict[str, Any] = field(default_factory=dict)
    edition: dict[str, bool] | None = None
    debuff: bool = False
    eternal: bool = False
    cost: int = 0
    sell_cost: int = 0
    base: Any = None
    seal: str | None = None
    sort_id: int = 0


def _make_hand(n: int = 5) -> list[MockCard]:
    """Create n hand cards."""
    return [MockCard(center_key="c_base", sort_id=i) for i in range(n)]


def _make_jokers(n: int = 3, eternal_indices: set[int] | None = None) -> list[MockCard]:
    """Create n joker cards, optionally with some eternal."""
    eternal_indices = eternal_indices or set()
    return [
        MockCard(
            center_key=f"j_joker_{i}",
            ability={"set": "Joker", "name": f"Joker {i}"},
            eternal=(i in eternal_indices),
            sell_cost=3,
        )
        for i in range(n)
    ]


def _make_consumable(key: str, cfg: dict | None = None) -> MockCard:
    """Create a consumable with optional config."""
    ability: dict[str, Any] = {"set": "Tarot"}
    if cfg is not None:
        ability["consumeable"] = cfg
    return MockCard(center_key=key, ability=ability)


def _make_planet(key: str = "c_mercury") -> MockCard:
    """Create a planet consumable (always usable, no targets)."""
    return MockCard(
        center_key=key,
        ability={"set": "Planet", "consumeable": {"hand_type": "Pair"}},
    )


def _make_shop_card(cost: int = 3, card_set: str = "Joker") -> MockCard:
    return MockCard(
        center_key="j_test",
        ability={"set": card_set},
        cost=cost,
    )


def _make_voucher(cost: int = 10) -> MockCard:
    return MockCard(center_key="v_test", cost=cost)


def _make_booster(cost: int = 4) -> MockCard:
    return MockCard(center_key="p_test", cost=cost)


# ---------------------------------------------------------------------------
# Base game state builders
# ---------------------------------------------------------------------------


def _blind_select_state(**overrides: Any) -> dict[str, Any]:
    gs: dict[str, Any] = {
        "phase": GamePhase.BLIND_SELECT,
        "blind_on_deck": "Small",
        "hand": [],
        "jokers": [],
        "consumables": [],
        "dollars": 4,
        "joker_slots": 5,
        "consumable_slots": 2,
        "current_round": {},
        "shop_cards": [],
        "shop_vouchers": [],
        "shop_boosters": [],
        "pack_cards": [],
        "pack_choices_remaining": 0,
    }
    gs.update(overrides)
    return gs


def _selecting_hand_state(**overrides: Any) -> dict[str, Any]:
    gs = _blind_select_state(
        phase=GamePhase.SELECTING_HAND,
        hand=_make_hand(8),
        current_round={"hands_left": 4, "discards_left": 3},
    )
    gs.update(overrides)
    return gs


def _shop_state(**overrides: Any) -> dict[str, Any]:
    gs = _blind_select_state(
        phase=GamePhase.SHOP,
        dollars=20,
        current_round={"reroll_cost": 5, "free_rerolls": 0},
    )
    gs.update(overrides)
    return gs


def _pack_opening_state(**overrides: Any) -> dict[str, Any]:
    gs = _blind_select_state(
        phase=GamePhase.PACK_OPENING,
        pack_cards=[MockCard() for _ in range(3)],
        pack_choices_remaining=1,
    )
    gs.update(overrides)
    return gs


def _round_eval_state(**overrides: Any) -> dict[str, Any]:
    gs = _blind_select_state(phase=GamePhase.ROUND_EVAL)
    gs.update(overrides)
    return gs


# =========================================================================
# Tests: ActionType enum
# =========================================================================


class TestActionType:
    def test_count(self):
        assert len(ActionType) == NUM_ACTION_TYPES == 21

    def test_values_sequential(self):
        for i, at in enumerate(ActionType):
            assert at.value == i


# =========================================================================
# Tests: get_action_mask — per phase
# =========================================================================


class TestMaskBlindSelect:
    def test_select_blind_always_available(self):
        mask = get_action_mask(_blind_select_state())
        assert mask.type_mask[ActionType.SelectBlind]

    def test_skip_blind_small_big(self):
        for blind in ("Small", "Big"):
            mask = get_action_mask(_blind_select_state(blind_on_deck=blind))
            assert mask.type_mask[ActionType.SkipBlind]

    def test_skip_blind_boss_not_available(self):
        mask = get_action_mask(_blind_select_state(blind_on_deck="Boss"))
        assert not mask.type_mask[ActionType.SkipBlind]

    def test_no_play_discard_in_blind_select(self):
        mask = get_action_mask(_blind_select_state())
        assert not mask.type_mask[ActionType.PlayHand]
        assert not mask.type_mask[ActionType.Discard]


class TestMaskSelectingHand:
    def test_play_hand_available(self):
        mask = get_action_mask(_selecting_hand_state())
        assert mask.type_mask[ActionType.PlayHand]

    def test_discard_available(self):
        mask = get_action_mask(_selecting_hand_state())
        assert mask.type_mask[ActionType.Discard]

    def test_no_play_when_no_hands_left(self):
        gs = _selecting_hand_state(current_round={"hands_left": 0, "discards_left": 3})
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.PlayHand]

    def test_no_discard_when_no_discards_left(self):
        gs = _selecting_hand_state(current_round={"hands_left": 4, "discards_left": 0})
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.Discard]

    def test_no_play_when_empty_hand(self):
        gs = _selecting_hand_state(hand=[])
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.PlayHand]
        assert not mask.type_mask[ActionType.Discard]

    def test_sort_available_with_multiple_cards(self):
        mask = get_action_mask(_selecting_hand_state())
        assert mask.type_mask[ActionType.SortHandRank]
        assert mask.type_mask[ActionType.SortHandSuit]

    def test_sort_unavailable_with_one_card(self):
        gs = _selecting_hand_state(hand=_make_hand(1))
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.SortHandRank]
        assert not mask.type_mask[ActionType.SortHandSuit]

    def test_hand_swap_masks(self):
        gs = _selecting_hand_state(hand=_make_hand(5))
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.SwapHandLeft]
        assert mask.type_mask[ActionType.SwapHandRight]
        # Left: can't swap index 0
        left = mask.entity_masks[ActionType.SwapHandLeft]
        assert not left[0]
        assert all(left[1:])
        # Right: can't swap last index
        right = mask.entity_masks[ActionType.SwapHandRight]
        assert not right[4]
        assert all(right[:4])

    def test_joker_swap_masks(self):
        gs = _selecting_hand_state(jokers=_make_jokers(3))
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.SwapJokersLeft]
        left = mask.entity_masks[ActionType.SwapJokersLeft]
        assert not left[0]
        assert left[1] and left[2]

    def test_no_joker_swap_with_single_joker(self):
        gs = _selecting_hand_state(jokers=_make_jokers(1))
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.SwapJokersLeft]
        assert not mask.type_mask[ActionType.SwapJokersRight]

    def test_card_mask_shape(self):
        gs = _selecting_hand_state(hand=_make_hand(7))
        mask = get_action_mask(gs)
        assert mask.card_mask.shape == (7,)
        assert mask.card_mask.all()


class TestMaskShop:
    def test_next_round_always_available(self):
        mask = get_action_mask(_shop_state())
        assert mask.type_mask[ActionType.NextRound]

    def test_reroll_available_with_money(self):
        mask = get_action_mask(_shop_state(dollars=10))
        assert mask.type_mask[ActionType.Reroll]

    def test_reroll_unavailable_without_money(self):
        gs = _shop_state(
            dollars=2,
            current_round={"reroll_cost": 5, "free_rerolls": 0},
        )
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.Reroll]

    def test_reroll_with_free_rerolls(self):
        gs = _shop_state(
            dollars=0,
            current_round={"reroll_cost": 5, "free_rerolls": 1},
        )
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.Reroll]

    def test_buy_card_affordable(self):
        gs = _shop_state(
            shop_cards=[_make_shop_card(cost=5)],
            dollars=10,
        )
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.BuyCard]
        assert mask.entity_masks[ActionType.BuyCard][0]

    def test_buy_card_too_expensive(self):
        gs = _shop_state(
            shop_cards=[_make_shop_card(cost=50)],
            dollars=10,
        )
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.BuyCard]

    def test_buy_card_no_joker_slots(self):
        gs = _shop_state(
            shop_cards=[_make_shop_card(cost=3, card_set="Joker")],
            jokers=_make_jokers(5),
            joker_slots=5,
            dollars=10,
        )
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.BuyCard]

    def test_sell_joker(self):
        gs = _shop_state(jokers=_make_jokers(2))
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.SellJoker]
        assert mask.entity_masks[ActionType.SellJoker].all()

    def test_sell_joker_eternal_excluded(self):
        gs = _shop_state(jokers=_make_jokers(3, eternal_indices={1}))
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.SellJoker]
        em = mask.entity_masks[ActionType.SellJoker]
        assert em[0] and not em[1] and em[2]

    def test_sell_consumable(self):
        gs = _shop_state(consumables=[_make_planet()])
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.SellConsumable]

    def test_redeem_voucher(self):
        gs = _shop_state(
            shop_vouchers=[_make_voucher(cost=10)],
            dollars=15,
        )
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.RedeemVoucher]
        assert mask.entity_masks[ActionType.RedeemVoucher][0]

    def test_redeem_voucher_too_expensive(self):
        gs = _shop_state(
            shop_vouchers=[_make_voucher(cost=10)],
            dollars=5,
        )
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.RedeemVoucher]

    def test_open_booster(self):
        gs = _shop_state(
            shop_boosters=[_make_booster(cost=4)],
            dollars=10,
        )
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.OpenBooster]

    def test_open_booster_too_expensive(self):
        gs = _shop_state(
            shop_boosters=[_make_booster(cost=20)],
            dollars=5,
        )
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.OpenBooster]


class TestMaskPackOpening:
    def test_pick_pack_card(self):
        mask = get_action_mask(_pack_opening_state())
        assert mask.type_mask[ActionType.PickPackCard]
        assert mask.entity_masks[ActionType.PickPackCard].shape == (3,)
        assert mask.entity_masks[ActionType.PickPackCard].all()

    def test_skip_pack_always(self):
        mask = get_action_mask(_pack_opening_state())
        assert mask.type_mask[ActionType.SkipPack]

    def test_no_pick_when_no_remaining(self):
        gs = _pack_opening_state(pack_choices_remaining=0)
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.PickPackCard]


class TestMaskRoundEval:
    def test_cashout_available(self):
        mask = get_action_mask(_round_eval_state())
        assert mask.type_mask[ActionType.CashOut]

    def test_no_play_in_round_eval(self):
        mask = get_action_mask(_round_eval_state())
        assert not mask.type_mask[ActionType.PlayHand]


class TestMaskGameOver:
    def test_all_false(self):
        gs = _blind_select_state(phase=GamePhase.GAME_OVER)
        mask = get_action_mask(gs)
        assert not mask.type_mask.any()


class TestMaskEmptyEntities:
    def test_empty_jokers_no_sell(self):
        gs = _shop_state(jokers=[])
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.SellJoker]

    def test_empty_consumables_no_sell_or_use(self):
        gs = _shop_state(consumables=[])
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.SellConsumable]
        assert not mask.type_mask[ActionType.UseConsumable]

    def test_empty_shop(self):
        gs = _shop_state(shop_cards=[], shop_vouchers=[], shop_boosters=[])
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.BuyCard]
        assert not mask.type_mask[ActionType.RedeemVoucher]
        assert not mask.type_mask[ActionType.OpenBooster]

    def test_empty_hand_no_swaps(self):
        gs = _selecting_hand_state(hand=[])
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.SwapHandLeft]
        assert not mask.type_mask[ActionType.SwapHandRight]
        assert mask.card_mask.shape == (0,)


# =========================================================================
# Tests: Consumable targeting
# =========================================================================


class TestConsumableTargeting:
    def test_planet_no_targets(self):
        """Planets need 0 card targets."""
        card = _make_planet("c_mercury")
        min_c, max_c, needs = get_consumable_target_info(card)
        assert not needs
        assert min_c == 0
        assert max_c == 0

    def test_magician_targets(self):
        """The Magician targets up to 2 cards (max_highlighted=2)."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        min_c, max_c, needs = get_consumable_target_info(card)
        assert needs
        assert min_c == 1
        assert max_c == 2

    def test_death_exactly_two(self):
        """Death requires exactly 2 cards."""
        card = _make_consumable("c_death", {"max_highlighted": 2, "min_highlighted": 2})
        min_c, max_c, needs = get_consumable_target_info(card)
        assert needs
        assert min_c == 2
        assert max_c == 2

    def test_star_up_to_three(self):
        """The Star targets up to 3 cards (suit change)."""
        card = _make_consumable("c_star", {"max_highlighted": 3})
        min_c, max_c, needs = get_consumable_target_info(card)
        assert needs
        assert min_c == 1
        assert max_c == 3

    def test_single_target_chariot(self):
        """The Chariot targets exactly 1 card."""
        card = _make_consumable("c_chariot", {"max_highlighted": 1})
        min_c, max_c, needs = get_consumable_target_info(card)
        assert needs
        assert min_c == 1
        assert max_c == 1

    def test_no_consumeable_config(self):
        """Consumable with no config needs no targets."""
        card = MockCard(center_key="c_hermit", ability={"set": "Tarot"})
        min_c, max_c, needs = get_consumable_target_info(card)
        assert not needs

    def test_use_consumable_mask_with_planet(self):
        """Planet consumable should be marked usable."""
        gs = _selecting_hand_state(consumables=[_make_planet()])
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.UseConsumable]
        assert mask.entity_masks[ActionType.UseConsumable][0]


# =========================================================================
# Tests: factored_to_engine_action
# =========================================================================


class TestFactoredToEngine:
    def test_play_hand(self):
        fa = FactoredAction(ActionType.PlayHand, card_target=(0, 2, 4))
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EnginePlayHand)
        assert action.card_indices == (0, 2, 4)

    def test_play_hand_single_card(self):
        fa = FactoredAction(ActionType.PlayHand, card_target=(3,))
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EnginePlayHand)
        assert action.card_indices == (3,)

    def test_play_hand_five_cards(self):
        fa = FactoredAction(ActionType.PlayHand, card_target=(0, 1, 2, 3, 4))
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EnginePlayHand)
        assert action.card_indices == (0, 1, 2, 3, 4)

    def test_play_hand_no_target_raises(self):
        fa = FactoredAction(ActionType.PlayHand, card_target=None)
        with pytest.raises(ValueError, match="card_target"):
            factored_to_engine_action(fa, {})

    def test_discard(self):
        fa = FactoredAction(ActionType.Discard, card_target=(1, 3))
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineDiscard)
        assert action.card_indices == (1, 3)

    def test_select_blind(self):
        fa = FactoredAction(ActionType.SelectBlind)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineSelectBlind)

    def test_skip_blind(self):
        fa = FactoredAction(ActionType.SkipBlind)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineSkipBlind)

    def test_cashout(self):
        fa = FactoredAction(ActionType.CashOut)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineCashOut)

    def test_reroll(self):
        fa = FactoredAction(ActionType.Reroll)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineReroll)

    def test_next_round(self):
        fa = FactoredAction(ActionType.NextRound)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineNextRound)

    def test_skip_pack(self):
        fa = FactoredAction(ActionType.SkipPack)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineSkipPack)

    def test_buy_card(self):
        fa = FactoredAction(ActionType.BuyCard, entity_target=2)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineBuyCard)
        assert action.shop_index == 2

    def test_sell_joker(self):
        fa = FactoredAction(ActionType.SellJoker, entity_target=1)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineSellCard)
        assert action.area == "jokers"
        assert action.card_index == 1

    def test_sell_consumable(self):
        fa = FactoredAction(ActionType.SellConsumable, entity_target=0)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineSellCard)
        assert action.area == "consumables"
        assert action.card_index == 0

    def test_use_consumable_no_targets(self):
        fa = FactoredAction(ActionType.UseConsumable, entity_target=0)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineUseConsumable)
        assert action.card_index == 0
        assert action.target_indices is None

    def test_use_consumable_with_targets(self):
        fa = FactoredAction(
            ActionType.UseConsumable,
            entity_target=0,
            card_target=(1, 3),
        )
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineUseConsumable)
        assert action.target_indices == (1, 3)

    def test_redeem_voucher(self):
        fa = FactoredAction(ActionType.RedeemVoucher, entity_target=0)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineRedeemVoucher)
        assert action.card_index == 0

    def test_open_booster(self):
        fa = FactoredAction(ActionType.OpenBooster, entity_target=1)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineOpenBooster)
        assert action.card_index == 1

    def test_pick_pack_card(self):
        fa = FactoredAction(ActionType.PickPackCard, entity_target=2)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EnginePickPackCard)
        assert action.card_index == 2

    def test_sort_hand_rank(self):
        fa = FactoredAction(ActionType.SortHandRank)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineSortHand)
        assert action.mode == "rank"

    def test_sort_hand_suit(self):
        fa = FactoredAction(ActionType.SortHandSuit)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EngineSortHand)
        assert action.mode == "suit"

    def test_entity_target_required(self):
        """Entity-targeted actions raise ValueError if entity_target is None."""
        for at in (
            ActionType.BuyCard,
            ActionType.SellJoker,
            ActionType.SellConsumable,
            ActionType.UseConsumable,
            ActionType.RedeemVoucher,
            ActionType.OpenBooster,
            ActionType.PickPackCard,
        ):
            fa = FactoredAction(at, entity_target=None)
            with pytest.raises(ValueError):
                factored_to_engine_action(fa, {})


# =========================================================================
# Tests: Swap operations → permutations
# =========================================================================


class TestSwapPermutations:
    def test_swap_jokers_left(self):
        """SwapJokersLeft(idx=2) swaps jokers[2] with jokers[1]."""
        gs = {"jokers": _make_jokers(4)}
        fa = FactoredAction(ActionType.SwapJokersLeft, entity_target=2)
        action = factored_to_engine_action(fa, gs)
        assert isinstance(action, EngineReorderJokers)
        assert action.new_order == (0, 2, 1, 3)

    def test_swap_jokers_right(self):
        """SwapJokersRight(idx=1) swaps jokers[1] with jokers[2]."""
        gs = {"jokers": _make_jokers(4)}
        fa = FactoredAction(ActionType.SwapJokersRight, entity_target=1)
        action = factored_to_engine_action(fa, gs)
        assert isinstance(action, EngineReorderJokers)
        assert action.new_order == (0, 2, 1, 3)

    def test_swap_jokers_left_first_raises(self):
        """SwapJokersLeft(idx=0) is invalid (nothing to the left)."""
        gs = {"jokers": _make_jokers(3)}
        fa = FactoredAction(ActionType.SwapJokersLeft, entity_target=0)
        with pytest.raises(ValueError):
            factored_to_engine_action(fa, gs)

    def test_swap_jokers_right_last_raises(self):
        """SwapJokersRight(idx=last) is invalid (nothing to the right)."""
        gs = {"jokers": _make_jokers(3)}
        fa = FactoredAction(ActionType.SwapJokersRight, entity_target=2)
        with pytest.raises(ValueError):
            factored_to_engine_action(fa, gs)

    def test_swap_hand_left(self):
        gs = {"hand": _make_hand(5)}
        fa = FactoredAction(ActionType.SwapHandLeft, entity_target=3)
        action = factored_to_engine_action(fa, gs)
        assert isinstance(action, EngineReorderHand)
        assert action.new_order == (0, 1, 3, 2, 4)

    def test_swap_hand_right(self):
        gs = {"hand": _make_hand(5)}
        fa = FactoredAction(ActionType.SwapHandRight, entity_target=0)
        action = factored_to_engine_action(fa, gs)
        assert isinstance(action, EngineReorderHand)
        assert action.new_order == (1, 0, 2, 3, 4)

    def test_swap_at_boundary(self):
        """Swap the last-but-one left and first right to edge positions."""
        gs = {"jokers": _make_jokers(2)}
        # Swap left: move index 1 to index 0
        fa = FactoredAction(ActionType.SwapJokersLeft, entity_target=1)
        action = factored_to_engine_action(fa, gs)
        assert action.new_order == (1, 0)

        # Swap right: move index 0 to index 1
        fa = FactoredAction(ActionType.SwapJokersRight, entity_target=0)
        action = factored_to_engine_action(fa, gs)
        assert action.new_order == (1, 0)

    def test_swap_requires_entity_target(self):
        gs = {"jokers": _make_jokers(3)}
        fa = FactoredAction(ActionType.SwapJokersLeft, entity_target=None)
        with pytest.raises(ValueError, match="entity_target"):
            factored_to_engine_action(fa, gs)


# =========================================================================
# Tests: engine_action_to_factored roundtrip
# =========================================================================


class TestEngineToFactored:
    def test_play_hand_roundtrip(self):
        action = EnginePlayHand(card_indices=(0, 1, 2))
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.PlayHand
        assert fa.card_target == (0, 1, 2)

    def test_discard_roundtrip(self):
        action = EngineDiscard(card_indices=(4,))
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.Discard
        assert fa.card_target == (4,)

    def test_simple_actions_roundtrip(self):
        pairs = [
            (EngineSelectBlind(), ActionType.SelectBlind),
            (EngineSkipBlind(), ActionType.SkipBlind),
            (EngineCashOut(), ActionType.CashOut),
            (EngineReroll(), ActionType.Reroll),
            (EngineNextRound(), ActionType.NextRound),
            (EngineSkipPack(), ActionType.SkipPack),
        ]
        for action, expected_type in pairs:
            fa = engine_action_to_factored(action, {})
            assert fa.action_type == expected_type
            assert fa.card_target is None
            assert fa.entity_target is None

    def test_buy_card_roundtrip(self):
        action = EngineBuyCard(shop_index=1)
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.BuyCard
        assert fa.entity_target == 1

    def test_sell_joker_roundtrip(self):
        action = EngineSellCard(area="jokers", card_index=2)
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.SellJoker
        assert fa.entity_target == 2

    def test_sell_consumable_roundtrip(self):
        action = EngineSellCard(area="consumables", card_index=0)
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.SellConsumable
        assert fa.entity_target == 0

    def test_use_consumable_roundtrip(self):
        action = EngineUseConsumable(card_index=1, target_indices=(2, 3))
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.UseConsumable
        assert fa.entity_target == 1
        assert fa.card_target == (2, 3)

    def test_use_consumable_no_targets_roundtrip(self):
        action = EngineUseConsumable(card_index=0, target_indices=None)
        fa = engine_action_to_factored(action, {})
        assert fa.card_target is None

    def test_redeem_voucher_roundtrip(self):
        action = EngineRedeemVoucher(card_index=0)
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.RedeemVoucher
        assert fa.entity_target == 0

    def test_open_booster_roundtrip(self):
        action = EngineOpenBooster(card_index=1)
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.OpenBooster
        assert fa.entity_target == 1

    def test_pick_pack_card_roundtrip(self):
        action = EnginePickPackCard(card_index=2)
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.PickPackCard
        assert fa.entity_target == 2

    def test_sort_hand_rank_roundtrip(self):
        action = EngineSortHand(mode="rank")
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.SortHandRank

    def test_sort_hand_suit_roundtrip(self):
        action = EngineSortHand(mode="suit")
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.SortHandSuit

    def test_reorder_jokers_adjacent_swap(self):
        """ReorderJokers with adjacent swap → SwapJokersLeft."""
        action = EngineReorderJokers(new_order=(0, 2, 1, 3))
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.SwapJokersLeft
        assert fa.entity_target == 2  # element at index 2 moved left

    def test_reorder_hand_adjacent_swap(self):
        action = EngineReorderHand(new_order=(1, 0, 2))
        fa = engine_action_to_factored(action, {})
        assert fa.action_type == ActionType.SwapHandLeft
        assert fa.entity_target == 1  # element at index 1 moved left

    def test_reorder_non_adjacent_raises(self):
        """Non-adjacent swaps can't be represented as single swap."""
        action = EngineReorderJokers(new_order=(2, 1, 0))
        with pytest.raises(ValueError, match="not an adjacent swap"):
            engine_action_to_factored(action, {})

    def test_reorder_complex_permutation_raises(self):
        """3-cycle permutation can't be represented."""
        action = EngineReorderJokers(new_order=(1, 2, 0))
        with pytest.raises(ValueError):
            engine_action_to_factored(action, {})

    def test_sell_unknown_area_raises(self):
        action = EngineSellCard(area="unknown", card_index=0)
        with pytest.raises(ValueError, match="Unknown SellCard area"):
            engine_action_to_factored(action, {})


# =========================================================================
# Tests: Full roundtrip (factored → engine → factored)
# =========================================================================


class TestFullRoundtrip:
    """Verify that factored→engine→factored is identity for all types."""

    def _roundtrip(self, fa: FactoredAction, gs: dict[str, Any]) -> FactoredAction:
        engine = factored_to_engine_action(fa, gs)
        return engine_action_to_factored(engine, gs)

    def test_play_hand_roundtrip(self):
        fa = FactoredAction(ActionType.PlayHand, card_target=(0, 3, 4))
        result = self._roundtrip(fa, {})
        assert result == fa

    def test_discard_roundtrip(self):
        fa = FactoredAction(ActionType.Discard, card_target=(1, 2))
        result = self._roundtrip(fa, {})
        assert result == fa

    def test_simple_roundtrips(self):
        for at in (
            ActionType.SelectBlind,
            ActionType.SkipBlind,
            ActionType.CashOut,
            ActionType.Reroll,
            ActionType.NextRound,
            ActionType.SkipPack,
            ActionType.SortHandRank,
            ActionType.SortHandSuit,
        ):
            fa = FactoredAction(at)
            assert self._roundtrip(fa, {}) == fa

    def test_entity_target_roundtrips(self):
        for at in (
            ActionType.BuyCard,
            ActionType.SellJoker,
            ActionType.SellConsumable,
            ActionType.RedeemVoucher,
            ActionType.OpenBooster,
            ActionType.PickPackCard,
        ):
            fa = FactoredAction(at, entity_target=1)
            assert self._roundtrip(fa, {}) == fa

    def test_use_consumable_roundtrip(self):
        fa = FactoredAction(ActionType.UseConsumable, entity_target=0, card_target=(2, 4))
        assert self._roundtrip(fa, {}) == fa

    def test_use_consumable_no_target_roundtrip(self):
        fa = FactoredAction(ActionType.UseConsumable, entity_target=0)
        assert self._roundtrip(fa, {}) == fa

    def test_swap_jokers_left_roundtrip(self):
        gs = {"jokers": _make_jokers(5)}
        fa = FactoredAction(ActionType.SwapJokersLeft, entity_target=3)
        result = self._roundtrip(fa, gs)
        assert result == fa

    def test_swap_jokers_right_roundtrip(self):
        gs = {"jokers": _make_jokers(5)}
        fa = FactoredAction(ActionType.SwapJokersRight, entity_target=2)
        # SwapRight(2) → permutation (0,1,3,2,4) → detected as SwapLeft(3)
        engine = factored_to_engine_action(fa, gs)
        assert engine.new_order == (0, 1, 3, 2, 4)
        result = engine_action_to_factored(engine, gs)
        # engine_action_to_factored always returns SwapLeft for an adjacent swap
        assert result.action_type == ActionType.SwapJokersLeft
        assert result.entity_target == 3

    def test_swap_hand_left_roundtrip(self):
        gs = {"hand": _make_hand(4)}
        fa = FactoredAction(ActionType.SwapHandLeft, entity_target=2)
        result = self._roundtrip(fa, gs)
        assert result == fa

    def test_swap_hand_right_roundtrip(self):
        gs = {"hand": _make_hand(4)}
        fa = FactoredAction(ActionType.SwapHandRight, entity_target=1)
        engine = factored_to_engine_action(fa, gs)
        assert engine.new_order == (0, 2, 1, 3)
        result = engine_action_to_factored(engine, gs)
        assert result.action_type == ActionType.SwapHandLeft
        assert result.entity_target == 2


# =========================================================================
# Tests: type_mask consistency with engine get_legal_actions
# =========================================================================


class TestMaskConsistencyWithEngine:
    """Verify type_mask is consistent with engine's get_legal_actions.

    Uses MockCard which passes the same attribute checks that
    get_legal_actions uses internally.
    """

    def test_blind_select_consistency(self):
        gs = _blind_select_state()
        mask = get_action_mask(gs)
        legal = get_legal_actions(gs)
        legal_types = {type(a) for a in legal}

        if EngineSelectBlind in legal_types:
            assert mask.type_mask[ActionType.SelectBlind]
        if EngineSkipBlind in legal_types:
            assert mask.type_mask[ActionType.SkipBlind]

    def test_selecting_hand_consistency(self):
        gs = _selecting_hand_state()
        mask = get_action_mask(gs)
        legal = get_legal_actions(gs)
        {type(a) for a in legal}

        # PlayHand/Discard markers
        has_play = any(isinstance(a, EnginePlayHand) for a in legal)
        has_discard = any(isinstance(a, EngineDiscard) for a in legal)
        assert mask.type_mask[ActionType.PlayHand] == has_play
        assert mask.type_mask[ActionType.Discard] == has_discard

        # Sort
        has_sort_rank = EngineSortHand(mode="rank") in legal
        has_sort_suit = EngineSortHand(mode="suit") in legal
        assert mask.type_mask[ActionType.SortHandRank] == has_sort_rank
        assert mask.type_mask[ActionType.SortHandSuit] == has_sort_suit

    def test_round_eval_consistency(self):
        gs = _round_eval_state()
        mask = get_action_mask(gs)
        legal = get_legal_actions(gs)
        has_cashout = any(isinstance(a, EngineCashOut) for a in legal)
        assert mask.type_mask[ActionType.CashOut] == has_cashout

    def test_pack_opening_consistency(self):
        gs = _pack_opening_state()
        mask = get_action_mask(gs)
        legal = get_legal_actions(gs)
        has_pick = any(isinstance(a, EnginePickPackCard) for a in legal)
        has_skip = any(isinstance(a, EngineSkipPack) for a in legal)
        assert mask.type_mask[ActionType.PickPackCard] == has_pick
        assert mask.type_mask[ActionType.SkipPack] == has_skip

    def test_shop_consistency(self):
        gs = _shop_state(
            shop_cards=[_make_shop_card(cost=3)],
            shop_vouchers=[_make_voucher(cost=10)],
            shop_boosters=[_make_booster(cost=4)],
            jokers=_make_jokers(2),
            consumables=[_make_planet()],
        )
        mask = get_action_mask(gs)
        legal = get_legal_actions(gs)

        has_buy = any(isinstance(a, EngineBuyCard) for a in legal)
        has_sell_j = any(isinstance(a, EngineSellCard) and a.area == "jokers" for a in legal)
        has_sell_c = any(isinstance(a, EngineSellCard) and a.area == "consumables" for a in legal)
        has_voucher = any(isinstance(a, EngineRedeemVoucher) for a in legal)
        has_booster = any(isinstance(a, EngineOpenBooster) for a in legal)
        has_reroll = any(isinstance(a, EngineReroll) for a in legal)
        has_next = any(isinstance(a, EngineNextRound) for a in legal)

        assert mask.type_mask[ActionType.BuyCard] == has_buy
        assert mask.type_mask[ActionType.SellJoker] == has_sell_j
        assert mask.type_mask[ActionType.SellConsumable] == has_sell_c
        assert mask.type_mask[ActionType.RedeemVoucher] == has_voucher
        assert mask.type_mask[ActionType.OpenBooster] == has_booster
        assert mask.type_mask[ActionType.Reroll] == has_reroll
        assert mask.type_mask[ActionType.NextRound] == has_next


# =========================================================================
# Tests: PlayHand card_target with 1-5 cards
# =========================================================================


class TestPlayHandCardTargets:
    @pytest.mark.parametrize("n_cards", [1, 2, 3, 4, 5])
    def test_play_n_cards(self, n_cards: int):
        indices = tuple(range(n_cards))
        fa = FactoredAction(ActionType.PlayHand, card_target=indices)
        action = factored_to_engine_action(fa, {})
        assert isinstance(action, EnginePlayHand)
        assert action.card_indices == indices
        assert len(action.card_indices) == n_cards

    def test_play_hand_empty_raises(self):
        fa = FactoredAction(ActionType.PlayHand, card_target=())
        with pytest.raises(ValueError):
            factored_to_engine_action(fa, {})
