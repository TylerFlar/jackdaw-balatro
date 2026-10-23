//! The action mask of `jackdaw/env/action_space.py` (`get_action_mask`): which action types are
//! legal, which entities each entity-targeted type may pick, the hand cards a selection may use,
//! the selection size bounds and Cerulean Bell's forced card.
//!
//! The rules are the engine's own (its legal actions, `jackdaw_core::legal`), read per entity:
//! a consumable is usable when [`Game::consumable_use`] finds a selection, a pack card can be
//! picked when the engine's legal actions would offer it. The one rule the mask states itself is
//! the shop's buy space, which follows `check_for_buy_space` (a Negative card raises either
//! limit by one) as the Python mask does.

use jackdaw_core::action::Action;
use jackdaw_core::area::AreaId;
use jackdaw_core::card::{Card, EditionKind};
use jackdaw_core::game::Game;
use jackdaw_core::state::{BlindSlot, HandSort, Phase};
use jackdaw_data::CenterSet;

use crate::action::{ActionType, NUM_ACTION_TYPES, default_targets};

/// Which actions count as legal (`Legality` in Python).
#[derive(Clone, Copy, Default)]
pub enum Legality<'a> {
    /// The game's own rules (Balatro 1.0.1o).
    #[default]
    Vanilla,
    /// Vanilla, narrowed by a filter. BalatroBot legality (vanilla and the endpoint's validate
    /// step as a dry run) plugs in here once `jackdaw-bot` provides its filter.
    Filtered(&'a dyn ActionFilter),
}

impl std::fmt::Debug for Legality<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Legality::Vanilla => f.write_str("Vanilla"),
            Legality::Filtered(_) => f.write_str("Filtered"),
        }
    }
}

/// Narrows the vanilla mask: asked once per action the vanilla mask allows.
pub trait ActionFilter: Sync {
    /// Whether `action` may be taken. Card-selecting actions come in marker form (PlayHand and
    /// Discard with an empty selection, as the engine's legal actions list them); a consumable
    /// use, pick or buy-and-use carries its default selection.
    fn allows(&self, game: &Game, action: &Action) -> bool;
}

/// The action mask (`ActionMask`), with buffers reused across [`ActionMask::compute`] calls.
#[derive(Clone, Debug)]
pub struct ActionMask {
    /// Which action types are legal.
    pub type_mask: [bool; NUM_ACTION_TYPES],
    /// Which hand cards a selection may use (every card of the hand).
    pub card_mask: Vec<bool>,
    /// Per type, which entities it may target (present only for types that have legal targets).
    entity: [Vec<bool>; NUM_ACTION_TYPES],
    has_entity: [bool; NUM_ACTION_TYPES],
    /// The types with an entity mask, in the order the Python mask adds them.
    order: Vec<ActionType>,
    pub min_card_select: usize,
    pub max_card_select: usize,
    /// Hand index of Cerulean Bell's forced card, which every PlayHand and Discard selection must
    /// include.
    pub forced_card: Option<usize>,
}

impl Default for ActionMask {
    fn default() -> Self {
        ActionMask {
            type_mask: [false; NUM_ACTION_TYPES],
            card_mask: Vec::new(),
            entity: Default::default(),
            has_entity: [false; NUM_ACTION_TYPES],
            order: Vec::new(),
            min_card_select: 1,
            max_card_select: 5,
            forced_card: None,
        }
    }
}

impl ActionMask {
    pub fn new() -> ActionMask {
        ActionMask::default()
    }

    /// The mask of `game` (`get_action_mask(game_state, legality)`).
    pub fn of(game: &Game, legality: Legality<'_>) -> ActionMask {
        let mut m = ActionMask::new();
        m.compute(game, legality);
        m
    }

    /// The entity mask of `t`, if it has one.
    pub fn entity_mask(&self, t: ActionType) -> Option<&[bool]> {
        self.has_entity[t.index()].then(|| self.entity[t.index()].as_slice())
    }

    /// The types with an entity mask, in insertion order (a Python dict's key order).
    pub fn entity_types(&self) -> &[ActionType] {
        &self.order
    }

    pub fn allows(&self, t: ActionType) -> bool {
        self.type_mask[t.index()]
    }

    fn clear(&mut self, n_hand: usize) {
        self.type_mask = [false; NUM_ACTION_TYPES];
        self.card_mask.clear();
        self.card_mask.resize(n_hand, true);
        self.has_entity = [false; NUM_ACTION_TYPES];
        self.order.clear();
        self.min_card_select = 1;
        self.max_card_select = 5;
        self.forced_card = None;
    }

    fn set(&mut self, t: ActionType) {
        self.type_mask[t.index()] = true;
    }

    /// Fill `t`'s entity mask with `f(i)` for `n` entities; keep it (and allow `t`) when any
    /// entity passes, as the Python helpers do with `if mask.any()`.
    fn entities(&mut self, t: ActionType, n: usize, mut f: impl FnMut(usize) -> bool) {
        if n == 0 {
            return;
        }
        let buf = &mut self.entity[t.index()];
        buf.clear();
        buf.extend((0..n).map(&mut f));
        if buf.iter().any(|&b| b) {
            self.type_mask[t.index()] = true;
            self.has_entity[t.index()] = true;
            self.order.push(t);
        }
    }

    /// Recompute the mask for `game`.
    pub fn compute(&mut self, game: &Game, legality: Legality<'_>) {
        let gs = &game.gs;
        let areas = &gs.areas;
        self.clear(areas.hand.len());
        match gs.phase {
            Phase::GameOver => return,
            Phase::BlindSelect => {
                self.set(ActionType::SelectBlind);
                if matches!(gs.blind_on_deck, Some(BlindSlot::Small | BlindSlot::Big)) {
                    self.set(ActionType::SkipBlind);
                }
                if game.boss_reroll_allowed() {
                    self.set(ActionType::RerollBoss);
                }
                self.sell_jokers(game);
                self.sell_consumables(game);
                self.consumables(game);
            }
            Phase::SelectingHand => {
                let n = areas.hand.len();
                let cr = &gs.current_round;
                if n > 0 && cr.hands_left > 0.0 {
                    self.set(ActionType::PlayHand);
                }
                if n > 0 && cr.discards_left > 0.0 {
                    self.set(ActionType::Discard);
                }
                self.forced_card = areas.hand.iter().position(|c| c.ability.forced_selection);
                if n > 1 {
                    self.set(ActionType::SortHandRank);
                    self.set(ActionType::SortHandSuit);
                    self.entities(ActionType::SwapHandLeft, n, |i| i != 0);
                    self.entities(ActionType::SwapHandRight, n, |i| i != n - 1);
                }
                self.joker_swaps(game);
                self.sell_jokers(game);
                self.sell_consumables(game);
                self.consumables(game);
            }
            Phase::RoundEval => {
                self.set(ActionType::CashOut);
                // Card:can_sell_card has no state gate (card.lua:1640-1653; G203).
                self.sell_jokers(game);
                self.sell_consumables(game);
                self.consumables(game);
            }
            Phase::Shop => {
                self.shop_buy(game);
                self.buy_and_use(game);
                self.sell_jokers(game);
                self.sell_consumables(game);
                self.consumables(game);
                let spendable = gs.dollars - gs.bankrupt_at;
                self.entities(ActionType::RedeemVoucher, areas.shop_vouchers.len(), |i| {
                    areas.shop_vouchers[i].cost <= spendable
                });
                self.entities(ActionType::OpenBooster, areas.shop_booster.len(), |i| {
                    can_afford(game, areas.shop_booster[i].cost)
                });
                let cr = &gs.current_round;
                if cr.free_rerolls > 0.0 || can_afford(game, cr.reroll_cost) {
                    self.set(ActionType::Reroll);
                }
                self.set(ActionType::NextRound);
                self.joker_swaps(game);
            }
            Phase::PackOpening => {
                let remaining = gs.pack.as_ref().map_or(0.0, |p| p.choices_remaining);
                let pack = &areas.pack_cards;
                if remaining > 0.0 {
                    self.entities(ActionType::PickPackCard, pack.len(), |i| {
                        pack_pick_allowed(game, &pack[i])
                    });
                }
                // can_skip_booster: an Arcana or Spectral pack waits for its hand (G202).
                if game.pack_skip_allowed() {
                    self.set(ActionType::SkipPack);
                }
                self.sell_jokers(game);
                self.sell_consumables(game);
                self.consumables(game);
            }
        }
        if let Legality::Filtered(filter) = legality {
            self.restrict(game, filter);
        }
    }

    fn sell_jokers(&mut self, game: &Game) {
        let jokers = &game.gs.areas.jokers;
        self.entities(ActionType::SellJoker, jokers.len(), |i| {
            !jokers[i].ability.eternal
        });
    }

    fn sell_consumables(&mut self, game: &Game) {
        self.entities(
            ActionType::SellConsumable,
            game.gs.areas.consumables.len(),
            |_| true,
        );
    }

    /// UseConsumable: the consumables usable now with some selection.
    fn consumables(&mut self, game: &Game) {
        let cons = &game.gs.areas.consumables;
        self.entities(ActionType::UseConsumable, cons.len(), |i| {
            game.consumable_use(cons[i].sort_id, true).is_some()
        });
    }

    /// Neighbour swaps that move no pinned joker.
    fn joker_swaps(&mut self, game: &Game) {
        let jokers = &game.gs.areas.jokers;
        let n = jokers.len();
        if n <= 1 {
            return;
        }
        let free = |i: usize, o: usize| !jokers[i].pinned && !jokers[o].pinned;
        self.entities(ActionType::SwapJokersLeft, n, |i| i > 0 && free(i, i - 1));
        self.entities(ActionType::SwapJokersRight, n, |i| {
            i + 1 < n && free(i, i + 1)
        });
    }

    /// BuyCard: affordable shop cards with room (`check_for_buy_space`: a Negative card raises
    /// the joker or consumable limit by one).
    fn shop_buy(&mut self, game: &Game) {
        let gs = &game.gs;
        let shop = &gs.areas.shop_jokers;
        let jokers = gs.areas.jokers.len() as f64;
        let consumables = gs.areas.consumables.len() as f64;
        self.entities(ActionType::BuyCard, shop.len(), |i| {
            let card = &shop[i];
            if !can_afford(game, card.cost) {
                return false;
            }
            let bonus = if is_negative(card) { 1.0 } else { 0.0 };
            match card.set() {
                CenterSet::Joker => jokers < gs.joker_slots + bonus,
                CenterSet::Tarot | CenterSet::Planet | CenterSet::Spectral => {
                    consumables < gs.consumable_slots + bonus
                }
                _ => true,
            }
        });
    }

    /// BuyAndUse: affordable shop consumables usable now (`can_use_consumeable` only, G138).
    fn buy_and_use(&mut self, game: &Game) {
        let shop = &game.gs.areas.shop_jokers;
        self.entities(ActionType::BuyAndUse, shop.len(), |i| {
            let card = &shop[i];
            card.center().consumeable
                && can_afford(game, card.cost)
                && game.consumable_use(card.sort_id, false).is_some()
        });
    }

    /// Narrow the mask with `filter`, as the bridge's `restrict_mask` does: an entity type keeps
    /// the entities whose action passes, any other type stays if its action passes.
    fn restrict(&mut self, game: &Game, filter: &dyn ActionFilter) {
        for t in ActionType::ALL {
            if !self.type_mask[t.index()] {
                continue;
            }
            let i = t.index();
            if self.has_entity[i] {
                let mut any = false;
                for e in 0..self.entity[i].len() {
                    if self.entity[i][e] {
                        let ok = filter.allows(game, &candidate(game, t, Some(e)));
                        self.entity[i][e] = ok;
                        any |= ok;
                    }
                }
                if !any {
                    self.type_mask[i] = false;
                    self.has_entity[i] = false;
                    self.order.retain(|&o| o != t);
                }
            } else if !filter.allows(game, &candidate(game, t, None)) {
                self.type_mask[i] = false;
            }
        }
    }
}

/// The engine action a mask entry stands for (see [`ActionFilter::allows`]).
fn candidate(game: &Game, t: ActionType, entity: Option<usize>) -> Action {
    use ActionType::*;
    let e = entity.unwrap_or(0);
    match t {
        PlayHand => Action::PlayHand(Vec::new()),
        Discard => Action::Discard(Vec::new()),
        SelectBlind => Action::SelectBlind,
        SkipBlind => Action::SkipBlind,
        CashOut => Action::CashOut,
        Reroll => Action::Reroll,
        NextRound => Action::NextRound,
        SkipPack => Action::SkipPack,
        BuyCard => Action::BuyCard(e),
        SellJoker => Action::SellCard {
            area: AreaId::Jokers,
            index: e,
        },
        SellConsumable => Action::SellCard {
            area: AreaId::Consumables,
            index: e,
        },
        UseConsumable => Action::UseConsumable {
            index: e,
            targets: default_targets(game, AreaId::Consumables, e),
        },
        RedeemVoucher => Action::RedeemVoucher(e),
        OpenBooster => Action::OpenBooster(e),
        PickPackCard => Action::PickPackCard {
            index: e,
            targets: default_targets(game, AreaId::PackCards, e),
        },
        SwapJokersLeft => Action::SwapJokersLeft(e),
        SwapJokersRight => Action::SwapJokersRight(e),
        SwapHandLeft => Action::SwapHandLeft(e),
        SwapHandRight => Action::SwapHandRight(e),
        SortHandRank => Action::SortHand(HandSort::Desc),
        SortHandSuit => Action::SortHand(HandSort::SuitDesc),
        RerollBoss => Action::RerollBoss,
        BuyAndUse => Action::BuyAndUse {
            index: e,
            targets: default_targets(game, AreaId::ShopJokers, e),
        },
    }
}

/// `can_afford`: free, or within `dollars - bankrupt_at`.
fn can_afford(game: &Game, cost: f64) -> bool {
    cost <= 0.0 || cost <= game.gs.dollars - game.gs.bankrupt_at
}

fn is_negative(card: &Card) -> bool {
    card.edition
        .is_some_and(|e| e.kind == EditionKind::Negative)
}

/// `pack_pick_block_reason(card) is None`: a Joker needs room unless it is Negative, a
/// consumable must be usable, anything else (playing cards) can always be taken.
fn pack_pick_allowed(game: &Game, card: &Card) -> bool {
    match card.set() {
        CenterSet::Joker => {
            is_negative(card) || (game.gs.areas.jokers.len() as f64) < game.gs.joker_slots
        }
        CenterSet::Tarot | CenterSet::Planet | CenterSet::Spectral => {
            game.consumable_use(card.sort_id, true).is_some()
        }
        _ => true,
    }
}
