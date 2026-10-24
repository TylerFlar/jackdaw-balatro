"""The factored action space: masks, conversions and default selections, by rule.

The masks, the conversions and the default card selections are the Rust engine's
(:mod:`jackdaw.env.rust_env`); ``test_golden_env.py`` and ``test_rust_action_space.py``
check them against the Python env layer's frozen outputs at thousands of states. These
tests pin the rules by name on states built with the Rust API.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from jackdaw.core import Game
from jackdaw.engine.actions import (
    BuyAndUse,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    Reroll,
    RerollBoss,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
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
# States
# ---------------------------------------------------------------------------


def _blind_select(seed: str = "ASPACE") -> Game:
    return Game.new("b_red", 1, seed)


def _selecting_hand(seed: str = "ASPACE") -> Game:
    game = _blind_select(seed)
    game.step(SelectBlind())
    return game


def _round_eval(seed: str = "ASPACE") -> Game:
    game = _selecting_hand(seed)
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    assert game.phase == GamePhase.ROUND_EVAL
    return game


def _shop(seed: str = "ASPACE") -> Game:
    game = _round_eval(seed)
    game.step(CashOut())
    return game


def _pack(key: str = "p_arcana_normal_1", seed: str = "ASPACE") -> Game:
    """In the shop, a booster of kind ``key`` added and opened."""
    game = _shop(seed)
    game.debug_add(key)
    game.debug_set(money=50)
    game.step(OpenBooster(card_index=len(game.shop_boosters) - 1))
    assert game.phase == GamePhase.PACK_OPENING
    return game


def _edit(game: Game, change) -> Game:
    """The run from an edited snapshot."""
    snap = game.to_dict()
    change(snap)
    return Game.from_dict(snap)


def _with_jokers(game: Game, *keys: str) -> Game:
    for key in keys:
        game.debug_add(key)
    return game


def _phases() -> dict[str, Game]:
    return {
        "blind_select": _blind_select(),
        "selecting_hand": _selecting_hand(),
        "round_eval": _round_eval(),
        "shop": _shop(),
        "pack_opening": _pack(),
    }


# =========================================================================
# ActionType
# =========================================================================


class TestActionType:
    def test_count(self):
        assert len(ActionType) == NUM_ACTION_TYPES == 23

    def test_values_sequential(self):
        for i, at in enumerate(ActionType):
            assert at.value == i


# =========================================================================
# Masks
# =========================================================================


class TestMaskBlindSelect:
    def test_select_blind_always_available(self):
        assert get_action_mask(_blind_select()).type_mask[ActionType.SelectBlind]

    def test_skip_blind_small_big(self):
        game = _blind_select()
        assert get_action_mask(game).type_mask[ActionType.SkipBlind]
        game.step(SkipBlind())
        assert get_action_mask(game).type_mask[ActionType.SkipBlind]

    def test_skip_blind_boss_not_available(self):
        game = _blind_select()
        game.step(SkipBlind())
        game.step(SkipBlind())
        mask = get_action_mask(game)
        assert mask.type_mask[ActionType.SelectBlind]
        assert not mask.type_mask[ActionType.SkipBlind]

    def test_no_play_discard_in_blind_select(self):
        mask = get_action_mask(_blind_select())
        assert not mask.type_mask[ActionType.PlayHand]
        assert not mask.type_mask[ActionType.Discard]
        assert mask.card_mask.shape == (0,)


class TestMaskSelectingHand:
    def test_play_and_discard_available(self):
        mask = get_action_mask(_selecting_hand())
        assert mask.type_mask[ActionType.PlayHand]
        assert mask.type_mask[ActionType.Discard]
        assert (mask.min_card_select, mask.max_card_select) == (1, 5)

    def test_no_discard_when_no_discards_left(self):
        game = _selecting_hand()
        game.debug_set(discards=0)
        mask = get_action_mask(game)
        assert mask.type_mask[ActionType.PlayHand]
        assert not mask.type_mask[ActionType.Discard]

    def test_card_mask_covers_the_hand(self):
        game = _selecting_hand()
        mask = get_action_mask(game)
        assert mask.card_mask.shape == (len(game.hand),)
        assert mask.card_mask.all()

    def test_no_forced_card_by_default(self):
        assert get_action_mask(_selecting_hand()).forced_card is None

    def test_cerulean_bell_forced_card_reported(self):
        game = _blind_select()
        game.apply_setup({"op": "force_boss", "key": "bl_final_bell"})
        game.step(SelectBlind())
        forced = get_action_mask(game).forced_card
        assert forced is not None
        assert game.hand[forced].ability.get("forced_selection")

    def test_sort_available_with_multiple_cards(self):
        mask = get_action_mask(_selecting_hand())
        assert mask.type_mask[ActionType.SortHandRank]
        assert mask.type_mask[ActionType.SortHandSuit]

    def test_hand_swap_masks(self):
        game = _selecting_hand()
        n = len(game.hand)
        mask = get_action_mask(game)
        assert mask.entity_masks[ActionType.SwapHandLeft].tolist() == [False] + [True] * (n - 1)
        assert mask.entity_masks[ActionType.SwapHandRight].tolist() == [True] * (n - 1) + [False]

    def test_joker_swap_masks(self):
        mask = get_action_mask(_with_jokers(_selecting_hand(), "j_joker", "j_egg", "j_misprint"))
        assert mask.entity_masks[ActionType.SwapJokersLeft].tolist() == [False, True, True]
        assert mask.entity_masks[ActionType.SwapJokersRight].tolist() == [True, True, False]

    def test_no_joker_swap_with_single_joker(self):
        mask = get_action_mask(_with_jokers(_selecting_hand(), "j_joker"))
        assert not mask.type_mask[ActionType.SwapJokersLeft]
        assert not mask.type_mask[ActionType.SwapJokersRight]


class TestMaskShop:
    def test_next_round_always_available(self):
        assert get_action_mask(_shop()).type_mask[ActionType.NextRound]

    def test_reroll_follows_money(self):
        game = _shop()
        game.debug_set(money=10)
        assert get_action_mask(game).type_mask[ActionType.Reroll]
        game.debug_set(money=0)
        assert not get_action_mask(game).type_mask[ActionType.Reroll]

    def test_reroll_with_free_rerolls(self):
        game = _shop()
        game.debug_set(money=0)

        def free(snap):
            snap["current_round"]["free_rerolls"] = 1

        assert get_action_mask(_edit(game, free)).type_mask[ActionType.Reroll]

    def test_buy_card_follows_money(self):
        game = _shop()
        game.debug_set(money=50)
        mask = get_action_mask(game)
        assert mask.entity_masks[ActionType.BuyCard].all()
        game.debug_set(money=0)
        assert not get_action_mask(game).type_mask[ActionType.BuyCard]

    def test_everything_affordable_on_credit(self):
        """Credit Card lets the money go to -$20 (bankrupt_at)."""
        game = _with_jokers(_shop(), "j_credit_card")
        game.debug_set(money=0)
        mask = get_action_mask(game)
        assert mask.type_mask[ActionType.BuyCard]
        assert mask.type_mask[ActionType.Reroll]

    def test_buy_card_no_joker_slots(self):
        game = _with_jokers(_shop(), *["j_joker"] * 5)
        game.debug_set(money=50)
        mask = get_action_mask(game)
        jokers = [i for i, c in enumerate(game.shop_jokers) if c.key.startswith("j_")]
        for i in jokers:
            assert mask.entity_masks.get(ActionType.BuyCard, np.zeros(9, bool))[i] == bool(
                game.shop_jokers[i].edition == "negative"
            )

    def test_sell_joker_eternal_excluded(self):
        game = _shop()
        game.debug_add("j_joker")
        game.debug_add("j_egg", eternal=True)
        mask = get_action_mask(game)
        assert mask.entity_masks[ActionType.SellJoker].tolist() == [True, False]

    def test_sell_consumable(self):
        game = _shop()
        game.debug_add("c_pluto")
        mask = get_action_mask(game)
        assert mask.entity_masks[ActionType.SellConsumable].tolist() == [True]

    def test_redeem_voucher_follows_money(self):
        game = _shop()
        game.debug_set(money=50)
        assert get_action_mask(game).type_mask[ActionType.RedeemVoucher]
        game.debug_set(money=0)
        assert not get_action_mask(game).type_mask[ActionType.RedeemVoucher]

    def test_open_booster_follows_money(self):
        game = _shop()
        game.debug_set(money=50)
        assert get_action_mask(game).type_mask[ActionType.OpenBooster]
        game.debug_set(money=0)
        assert not get_action_mask(game).type_mask[ActionType.OpenBooster]


class TestMaskPackOpening:
    def test_pick_pack_card(self):
        game = _pack("p_buffoon_normal_1")
        mask = get_action_mask(game)
        assert mask.type_mask[ActionType.PickPackCard]
        assert mask.entity_masks[ActionType.PickPackCard].shape == (len(game.pack_cards),)

    def test_skip_pack_follows_can_skip_booster(self):
        """G202: can_skip_booster (button_callbacks.lua:2132-2141) lets a
        Celestial, Standard or Buffoon pack go at once, an Arcana or Spectral
        pack only once its hand is dealt (or the hand can hold no cards)."""
        assert get_action_mask(_pack("p_buffoon_normal_1")).type_mask[ActionType.SkipPack]
        arcana = _pack("p_arcana_normal_1")
        assert arcana.hand and get_action_mask(arcana).type_mask[ActionType.SkipPack]

        def undeal(snap):
            for card in snap["areas"]["hand"]:
                card["facing"] = "front"
            snap["areas"]["deck"] += snap["areas"]["hand"]
            snap["areas"]["hand"] = []

        assert not get_action_mask(_edit(arcana, undeal)).type_mask[ActionType.SkipPack]

        def no_room(snap):
            undeal(snap)
            snap["game"]["hand_size"] = 0

        spectral = _pack("p_spectral_normal_1")
        assert get_action_mask(_edit(spectral, no_room)).type_mask[ActionType.SkipPack]

    def test_no_pick_when_no_remaining(self):
        def spent(snap):
            snap["pack"]["choices_remaining"] = 0

        mask = get_action_mask(_edit(_pack("p_buffoon_normal_1"), spent))
        assert not mask.type_mask[ActionType.PickPackCard]


class TestMaskRoundEval:
    def test_cashout_available(self):
        assert get_action_mask(_round_eval()).type_mask[ActionType.CashOut]

    def test_no_play_in_round_eval(self):
        assert not get_action_mask(_round_eval()).type_mask[ActionType.PlayHand]


class TestMaskGameOver:
    def test_all_false(self):
        game = _selecting_hand()
        game.debug_set(hands=1)
        game.step(PlayHand(card_indices=(0,)))
        assert game.phase == GamePhase.GAME_OVER
        assert not get_action_mask(game).type_mask.any()


class TestMaskReordering:
    """G206: a drag reorders the jokers in every phase and the hand whenever it
    holds cards (cardarea.lua:214-227, 436-465, 509-529)."""

    @pytest.mark.parametrize(
        "phase", ["blind_select", "selecting_hand", "round_eval", "shop", "pack_opening"]
    )
    def test_jokers_reorder_in_every_phase(self, phase):
        game = _with_jokers(_phases()[phase], "j_joker", "j_egg", "j_misprint")
        mask = get_action_mask(game)
        assert mask.entity_masks[ActionType.SwapJokersLeft].tolist() == [False, True, True]
        assert mask.entity_masks[ActionType.SwapJokersRight].tolist() == [True, True, False]

    def test_a_packs_dealt_hand_reorders_but_is_not_sorted(self):
        game = _pack("p_arcana_normal_1")
        n = len(game.hand)
        mask = get_action_mask(game)
        assert mask.entity_masks[ActionType.SwapHandLeft].tolist() == [False] + [True] * (n - 1)
        assert mask.entity_masks[ActionType.SwapHandRight].tolist() == [True] * (n - 1) + [False]
        assert not mask.type_mask[ActionType.SortHandRank]
        assert not mask.type_mask[ActionType.SortHandSuit]

    def test_a_pack_without_a_hand_has_no_hand_swap(self):
        game = _with_jokers(_pack("p_buffoon_normal_1"), "j_joker", "j_egg")
        mask = get_action_mask(game)
        assert mask.type_mask[ActionType.SwapJokersLeft]
        assert not mask.type_mask[ActionType.SwapHandLeft]
        assert not mask.type_mask[ActionType.SwapHandRight]

    def test_entity_masks_follow_the_legal_list_order(self):
        """The swaps come last in these phases, as in the engine's legal list."""
        game = _with_jokers(_pack("p_arcana_normal_1"), "j_joker", "j_egg")
        assert list(get_action_mask(game).entity_masks)[-4:] == [
            ActionType.SwapHandLeft,
            ActionType.SwapHandRight,
            ActionType.SwapJokersLeft,
            ActionType.SwapJokersRight,
        ]


class TestMaskConsistencyWithEngine:
    """The mask offers exactly what the engine's legal list offers."""

    def test_selling_is_not_shop_only(self):
        """Bug #74: selling was gated to SHOP in mask AND executor.

        Card:can_sell_card (card.lua:1640) has no state gate -- its
        blockers are cards mid-scoring, a locked controller and STOP_USE
        -- and vanilla explicitly COMMENTED OUT the blind-select
        restriction. Consumables qualify too: can_sell_card wants
        area.config.type == 'joker' and game.lua:2239 gives the
        consumables area that type. G203: the cash-out screen sells too.
        """
        for phase in ("selecting_hand", "blind_select", "shop", "round_eval"):
            game = _with_jokers(_phases()[phase], "j_joker", "j_greedy_joker")
            mask = get_action_mask(game)
            assert mask.type_mask[ActionType.SellJoker], phase
            game.step(SellCard(area="jokers", card_index=0))
            assert len(game.jokers) == 1, phase

    @pytest.mark.parametrize("seed", ["MASK_LEGAL_1", "MASK_LEGAL_2", "MASK_LEGAL_3"])
    def test_mask_and_legal_actions_agree_along_a_run(self, seed: str):
        """At every state of a random run: an action type is masked in exactly when some
        legal action has it, with the same entities."""
        game = Game.new("b_red", 1, seed)
        for _ in range(3):
            game.debug_add("j_joker")
        rng = random.Random(seed)
        for _ in range(300):
            if game.phase == GamePhase.GAME_OVER:
                break
            legal = game.legal_actions()
            mask = get_action_mask(game)
            entities: dict[int, set[int]] = {}
            for action in legal:
                fa = engine_action_to_factored(action)
                entities.setdefault(int(fa.action_type), set())
                if fa.entity_target is not None:
                    entities[int(fa.action_type)].add(fa.entity_target)
            assert set(np.flatnonzero(mask.type_mask)) == set(entities), game
            for t, targets in entities.items():
                if targets:
                    assert set(np.flatnonzero(mask.entity_masks[t])) == targets, (game, t)
            action = rng.choice(legal)
            if isinstance(action, (PlayHand, Discard)) and not action.card_indices:
                n = rng.randint(1, min(5, len(game.hand)))
                cards = sorted(rng.sample(range(len(game.hand)), n))
                forced = mask.forced_card
                if forced is not None and forced not in cards:
                    cards[0] = forced
                action = type(action)(card_indices=tuple(sorted(cards)))
            game.step(action)


# =========================================================================
# Consumable targeting info
# =========================================================================


class TestConsumableTargeting:
    @pytest.mark.parametrize(
        ("key", "info"),
        [
            ("c_pluto", (0, 0, False)),
            ("c_magician", (1, 2, True)),
            ("c_death", (2, 2, True)),
            ("c_star", (1, 3, True)),
            ("c_chariot", (1, 1, True)),
            ("c_hermit", (0, 0, False)),
        ],
    )
    def test_target_info(self, key: str, info: tuple[int, int, bool]):
        game = _selecting_hand()
        game.debug_add(key)
        assert get_consumable_target_info(game, "consumables", 0) == info

    def test_past_the_end_raises(self):
        with pytest.raises(ValueError):
            get_consumable_target_info(_selecting_hand(), "consumables", 0)

    def test_use_consumable_mask_with_planet(self):
        game = _shop()
        game.debug_add("c_pluto")
        mask = get_action_mask(game)
        assert mask.type_mask[ActionType.UseConsumable]
        assert mask.entity_masks[ActionType.UseConsumable].tolist() == [True]

    def test_targeting_tarot_not_offered_without_a_hand(self):
        game = _blind_select()
        game.debug_add("c_magician")
        assert not get_action_mask(game).type_mask[ActionType.UseConsumable]


# =========================================================================
# factored_to_engine_action / engine_action_to_factored
# =========================================================================

_SIMPLE = [
    (ActionType.SelectBlind, SelectBlind()),
    (ActionType.SkipBlind, SkipBlind()),
    (ActionType.CashOut, CashOut()),
    (ActionType.Reroll, Reroll()),
    (ActionType.NextRound, NextRound()),
    (ActionType.SkipPack, SkipPack()),
    (ActionType.SortHandRank, SortHand(mode="rank")),
    (ActionType.SortHandSuit, SortHand(mode="suit")),
    (ActionType.RerollBoss, RerollBoss()),
]

_ENTITY = [
    (ActionType.BuyCard, 2, BuyCard(shop_index=2)),
    (ActionType.SellJoker, 1, SellCard(area="jokers", card_index=1)),
    (ActionType.SellConsumable, 0, SellCard(area="consumables", card_index=0)),
    (ActionType.RedeemVoucher, 0, RedeemVoucher(card_index=0)),
    (ActionType.OpenBooster, 1, OpenBooster(card_index=1)),
    (ActionType.SwapJokersLeft, 2, SwapJokersLeft(idx=2)),
    (ActionType.SwapJokersRight, 0, SwapJokersRight(idx=0)),
    (ActionType.SwapHandLeft, 3, SwapHandLeft(idx=3)),
    (ActionType.SwapHandRight, 4, SwapHandRight(idx=4)),
]


class TestFactoredToEngine:
    @pytest.mark.parametrize(("t", "action"), _SIMPLE)
    def test_simple(self, t, action):
        assert factored_to_engine_action(FactoredAction(t), _blind_select()) == action

    @pytest.mark.parametrize(("t", "entity", "action"), _ENTITY)
    def test_entity(self, t, entity, action):
        fa = FactoredAction(t, entity_target=entity)
        assert factored_to_engine_action(fa, _blind_select()) == action

    def test_play_and_discard(self):
        game = _selecting_hand()
        fa = FactoredAction(ActionType.PlayHand, card_target=(0, 2, 4))
        assert factored_to_engine_action(fa, game) == PlayHand(card_indices=(0, 2, 4))
        fa = FactoredAction(ActionType.Discard, card_target=(1,))
        assert factored_to_engine_action(fa, game) == Discard(card_indices=(1,))

    @pytest.mark.parametrize(
        "t",
        [ActionType.PlayHand, ActionType.Discard],
    )
    def test_a_selection_is_required(self, t):
        with pytest.raises(ValueError, match="card_target"):
            factored_to_engine_action(FactoredAction(t), _selecting_hand())

    @pytest.mark.parametrize("t", [t for t, _, _ in _ENTITY] + [ActionType.UseConsumable])
    def test_an_entity_is_required(self, t):
        with pytest.raises(ValueError, match="entity_target"):
            factored_to_engine_action(FactoredAction(t), _selecting_hand())

    def test_use_and_pick_keep_explicit_targets(self):
        game = _selecting_hand()
        game.debug_add("c_magician")
        fa = FactoredAction(ActionType.UseConsumable, card_target=(3, 5), entity_target=0)
        assert factored_to_engine_action(fa, game) == UseConsumable(0, (3, 5))


class TestUseConsumableDefaultTargets:
    """A targeting consumable used without card targets gets the default selection the
    mask judged legal: the lowest-index legal cards."""

    def test_targeting_tarot_gets_lowest_indices(self):
        game = _selecting_hand()
        game.debug_add("c_death")
        fa = FactoredAction(ActionType.UseConsumable, entity_target=0)
        assert factored_to_engine_action(fa, game) == UseConsumable(0, (0, 1))

    def test_aura_skips_an_editioned_card(self):
        def foil_first(snap):
            snap["areas"]["hand"][0]["edition"] = {"foil": True, "chips": 50, "type": "foil"}

        game = _selecting_hand()
        game.debug_add("c_aura")
        game = _edit(game, foil_first)
        fa = FactoredAction(ActionType.UseConsumable, entity_target=0)
        assert factored_to_engine_action(fa, game) == UseConsumable(0, (1,))

    def test_untargeted_consumable_stays_targetless(self):
        game = _selecting_hand()
        game.debug_add("c_pluto")
        fa = FactoredAction(ActionType.UseConsumable, entity_target=0)
        assert factored_to_engine_action(fa, game) == UseConsumable(0, None)

    def test_pack_pick_default_targets(self):
        game = _pack("p_arcana_normal_1")
        for i, card in enumerate(game.pack_cards):
            lo, hi, needs = get_consumable_target_info(game, "pack_cards", i)
            action = factored_to_engine_action(
                FactoredAction(ActionType.PickPackCard, entity_target=i), game
            )
            assert isinstance(action, PickPackCard) and action.card_index == i
            if needs:
                assert action.target_indices == tuple(range(lo)), card.key
            else:
                assert action.target_indices is None, card.key

    def test_buy_and_use_default_targets(self):
        game = _shop()
        fa = FactoredAction(ActionType.BuyAndUse, entity_target=0)
        assert factored_to_engine_action(fa, game) == BuyAndUse(0, None)


class TestEngineToFactored:
    @pytest.mark.parametrize(("t", "action"), _SIMPLE)
    def test_simple(self, t, action):
        assert engine_action_to_factored(action) == FactoredAction(t)

    @pytest.mark.parametrize(("t", "entity", "action"), _ENTITY)
    def test_entity(self, t, entity, action):
        assert engine_action_to_factored(action) == FactoredAction(t, entity_target=entity)

    def test_cards(self):
        assert engine_action_to_factored(PlayHand(card_indices=(0, 1))) == FactoredAction(
            ActionType.PlayHand, card_target=(0, 1)
        )
        assert engine_action_to_factored(UseConsumable(1, (2,))) == FactoredAction(
            ActionType.UseConsumable, card_target=(2,), entity_target=1
        )
        assert engine_action_to_factored(PickPackCard(0, None)) == FactoredAction(
            ActionType.PickPackCard, entity_target=0
        )
        assert engine_action_to_factored(BuyAndUse(3, None)) == FactoredAction(
            ActionType.BuyAndUse, entity_target=3
        )

    @pytest.mark.parametrize(
        "action",
        [a for _, a in _SIMPLE] + [a for _, _, a in _ENTITY] + [PlayHand(card_indices=(1, 3))],
    )
    def test_roundtrip(self, action):
        game = _selecting_hand()
        assert factored_to_engine_action(engine_action_to_factored(action), game) == action
