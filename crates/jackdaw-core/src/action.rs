//! Player actions and the step entry point.
//!
//! The actions and their JSON form follow the Python engine (`jackdaw.engine.actions`) and the
//! recorded tapes: `["PlayHand", [0, 2, 3]]`, `["SellCard", "jokers", 1]`, ...

use serde_json::Value;

use crate::area::AreaId;
use crate::card::Card;
use crate::event::Event;
use crate::flow::HAND_HIGHLIGHT_LIMIT;
use crate::game::{Game, LuaState, StepError};
use crate::state::{BlindSlot, HandSort};

/// A player decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    SelectBlind,
    SkipBlind,
    PlayHand(Vec<usize>),
    Discard(Vec<usize>),
    BuyCard(usize),
    SellCard {
        area: AreaId,
        index: usize,
    },
    UseConsumable {
        index: usize,
        targets: Option<Vec<usize>>,
    },
    RedeemVoucher(usize),
    OpenBooster(usize),
    Reroll,
    NextRound,
    PickPackCard {
        index: usize,
        targets: Option<Vec<usize>>,
    },
    SkipPack,
    CashOut,
    SortHand(HandSort),
    SwapHandLeft(usize),
    SwapHandRight(usize),
    SwapJokersLeft(usize),
    SwapJokersRight(usize),
    RerollBoss,
    BuyAndUse {
        index: usize,
        targets: Option<Vec<usize>>,
    },
}

fn index(v: Option<&Value>) -> Result<usize, String> {
    v.and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("expected an index, got {v:?}"))
}

fn indices(v: Option<&Value>) -> Result<Vec<usize>, String> {
    v.and_then(Value::as_array)
        .ok_or_else(|| format!("expected a list of indices, got {v:?}"))?
        .iter()
        .map(|x| index(Some(x)))
        .collect()
}

fn opt_indices(v: Option<&Value>) -> Result<Option<Vec<usize>>, String> {
    match v {
        None | Some(Value::Null) => Ok(None),
        some => indices(some).map(Some),
    }
}

impl Action {
    /// An action from its tape form (`encode_action`).
    pub fn from_json(v: &Value) -> Result<Action, String> {
        let parts = v.as_array().ok_or("an action is a JSON array")?;
        let name = parts
            .first()
            .and_then(Value::as_str)
            .ok_or("an action starts with its name")?;
        let arg = |i: usize| parts.get(i);
        Ok(match name {
            "SelectBlind" => Action::SelectBlind,
            "SkipBlind" => Action::SkipBlind,
            "PlayHand" => Action::PlayHand(indices(arg(1))?),
            "Discard" => Action::Discard(indices(arg(1))?),
            "BuyCard" => Action::BuyCard(index(arg(1))?),
            "SellCard" => {
                let area = match arg(1).and_then(Value::as_str) {
                    Some("jokers") => AreaId::Jokers,
                    Some("consumables") => AreaId::Consumables,
                    other => return Err(format!("SellCard area {other:?}")),
                };
                Action::SellCard {
                    area,
                    index: index(arg(2))?,
                }
            }
            "UseConsumable" => Action::UseConsumable {
                index: index(arg(1))?,
                targets: opt_indices(arg(2))?,
            },
            "RedeemVoucher" => Action::RedeemVoucher(index(arg(1))?),
            "OpenBooster" => Action::OpenBooster(index(arg(1))?),
            "Reroll" => Action::Reroll,
            "NextRound" => Action::NextRound,
            "PickPackCard" => Action::PickPackCard {
                index: index(arg(1))?,
                targets: opt_indices(arg(2))?,
            },
            "SkipPack" => Action::SkipPack,
            "CashOut" => Action::CashOut,
            "SortHand" => Action::SortHand(match arg(1).and_then(Value::as_str) {
                Some("rank") => HandSort::Desc,
                Some("suit") => HandSort::SuitDesc,
                other => return Err(format!("SortHand mode {other:?}")),
            }),
            "SwapHandLeft" => Action::SwapHandLeft(index(arg(1))?),
            "SwapHandRight" => Action::SwapHandRight(index(arg(1))?),
            "SwapJokersLeft" => Action::SwapJokersLeft(index(arg(1))?),
            "SwapJokersRight" => Action::SwapJokersRight(index(arg(1))?),
            "RerollBoss" => Action::RerollBoss,
            "BuyAndUse" => Action::BuyAndUse {
                index: index(arg(1))?,
                targets: opt_indices(arg(2))?,
            },
            other => return Err(format!("unknown action {other:?}")),
        })
    }
}

impl Action {
    /// The action's tape form (the inverse of [`Action::from_json`]).
    pub fn to_json(&self) -> Value {
        use serde_json::json;
        let targets = |t: &Option<Vec<usize>>| t.as_ref().map_or(Value::Null, |v| json!(v));
        match self {
            Action::SelectBlind => json!(["SelectBlind"]),
            Action::SkipBlind => json!(["SkipBlind"]),
            Action::PlayHand(c) => json!(["PlayHand", c]),
            Action::Discard(c) => json!(["Discard", c]),
            Action::BuyCard(i) => json!(["BuyCard", i]),
            Action::SellCard { area, index } => {
                let area = match area {
                    AreaId::Jokers => "jokers",
                    _ => "consumables",
                };
                json!(["SellCard", area, index])
            }
            Action::UseConsumable { index, targets: t } => {
                json!(["UseConsumable", index, targets(t)])
            }
            Action::RedeemVoucher(i) => json!(["RedeemVoucher", i]),
            Action::OpenBooster(i) => json!(["OpenBooster", i]),
            Action::Reroll => json!(["Reroll"]),
            Action::NextRound => json!(["NextRound"]),
            Action::PickPackCard { index, targets: t } => {
                json!(["PickPackCard", index, targets(t)])
            }
            Action::SkipPack => json!(["SkipPack"]),
            Action::CashOut => json!(["CashOut"]),
            Action::SortHand(mode) => json!([
                "SortHand",
                match mode {
                    HandSort::Desc => "rank",
                    HandSort::SuitDesc => "suit",
                }
            ]),
            Action::SwapHandLeft(i) => json!(["SwapHandLeft", i]),
            Action::SwapHandRight(i) => json!(["SwapHandRight", i]),
            Action::SwapJokersLeft(i) => json!(["SwapJokersLeft", i]),
            Action::SwapJokersRight(i) => json!(["SwapJokersRight", i]),
            Action::RerollBoss => json!(["RerollBoss"]),
            Action::BuyAndUse { index, targets: t } => json!(["BuyAndUse", index, targets(t)]),
        }
    }
}

fn illegal<T>(msg: impl Into<String>) -> Result<T, StepError> {
    Err(StepError::Illegal(msg.into()))
}

fn unported<T>(msg: impl Into<String>) -> Result<T, StepError> {
    Err(StepError::Unported(msg.into()))
}

impl Game {
    /// Note that the step needs logic the port doesn't have yet; the step is refused.
    pub(crate) fn unported(&mut self, what: &str) {
        if self.rt.unported.is_none() {
            self.rt.unported = Some(what.to_owned());
        }
    }

    /// Apply an action and run the game until it waits for input again. On an error the
    /// game is unchanged.
    pub fn step(&mut self, action: &Action) -> Result<(), StepError> {
        let mut next = self.clone();
        next.apply(action)?;
        next.settle();
        if next.poll_hand_size() {
            next.settle();
        }
        if let Some(what) = next.rt.unported.take() {
            return unported(what);
        }
        *self = next;
        Ok(())
    }

    /// The game rests in a pack.
    fn require_pack(&self) -> Result<(), StepError> {
        match self.rt.state {
            LuaState::TarotPack
            | LuaState::PlanetPack
            | LuaState::SpectralPack
            | LuaState::StandardPack
            | LuaState::BuffoonPack => Ok(()),
            other => illegal(format!("no pack is open (now {other:?})")),
        }
    }

    /// Highlight the hand cards a consumable targets (`G.hand.highlighted`), in the given
    /// order; the forced card (Cerulean Bell) stays highlighted.
    fn highlight_targets(&mut self, targets: Option<&[usize]>) -> Result<(), StepError> {
        let hand = self.gs.areas.ids(AreaId::Hand);
        let mut ids: Vec<u32> = self
            .gs
            .areas
            .hand
            .iter()
            .filter(|c| c.ability.forced_selection)
            .map(|c| c.sort_id)
            .collect();
        for &i in targets.unwrap_or(&[]) {
            let Some(&id) = hand.get(i) else {
                return illegal(format!("hand index {i} out of range"));
            };
            if ids.contains(&id) {
                continue;
            }
            ids.push(id);
        }
        if ids.len() > HAND_HIGHLIGHT_LIMIT {
            return illegal("select at most five cards");
        }
        self.rt.highlighted = ids;
        Ok(())
    }

    /// Cards can be sold whenever the game waits for input with nothing being played: in a
    /// round, the shop, the blind-select screen, round evaluation and packs.
    fn require_sell_state(&self) -> Result<(), StepError> {
        match self.rt.state {
            LuaState::GameOver => illegal("the run is over"),
            _ => Ok(()),
        }
    }

    fn require(&self, state: LuaState) -> Result<(), StepError> {
        if self.rt.state != state {
            return illegal(format!("{state:?} only (now {:?})", self.rt.state));
        }
        Ok(())
    }

    /// A card in a shop area: its sort id and price.
    fn shop_card(&self, area: AreaId, i: usize) -> Result<&Card, StepError> {
        match self.gs.areas.get(area).get(i) {
            Some(c) => Ok(c),
            None => illegal(format!("{area:?} index {i} out of range")),
        }
    }

    /// `G.GAME.dollars - G.GAME.bankrupt_at`: what the player can spend.
    fn spendable(&self) -> f64 {
        self.gs.dollars - self.gs.bankrupt_at
    }

    /// The hand's sort ids for a play or discard selection (BalatroBot highlights them).
    fn select(&mut self, cards: &[usize]) -> Result<Vec<u32>, StepError> {
        let hand = self.gs.areas.ids(AreaId::Hand);
        if cards.is_empty() {
            return illegal("select at least one card");
        }
        if cards.len() > HAND_HIGHLIGHT_LIMIT {
            return illegal("select at most five cards");
        }
        let mut ids = Vec::with_capacity(cards.len());
        for &i in cards {
            let Some(&id) = hand.get(i) else {
                return illegal(format!("hand index {i} out of range"));
            };
            if ids.contains(&id) {
                return illegal(format!("hand index {i} selected twice"));
            }
            ids.push(id);
        }
        // Cerulean Bell's card can't be unselected.
        if let Some(forced) = self
            .gs
            .areas
            .hand
            .iter()
            .find(|c| c.ability.forced_selection)
            && !ids.contains(&forced.sort_id)
        {
            return illegal("the forced card (Cerulean Bell) must be selected");
        }
        Ok(ids)
    }

    fn apply(&mut self, action: &Action) -> Result<(), StepError> {
        match action {
            Action::SelectBlind => {
                self.require(LuaState::BlindSelect)?;
                self.select_blind();
            }
            Action::PlayHand(cards) => {
                self.require(LuaState::SelectingHand)?;
                if self.gs.current_round.hands_left <= 0.0 {
                    return illegal("no hands left");
                }
                let ids = self.select(cards)?;
                self.rt.highlighted = ids;
                self.play_cards_from_highlighted();
            }
            Action::Discard(cards) => {
                self.require(LuaState::SelectingHand)?;
                if self.gs.current_round.discards_left <= 0.0 {
                    return illegal("no discards left");
                }
                let ids = self.select(cards)?;
                self.rt.highlighted = ids;
                self.parse_highlighted();
                self.discard_cards_from_highlighted(false);
            }
            Action::SortHand(mode) => {
                self.require(LuaState::SelectingHand)?;
                self.gs.hand_sort = *mode;
                self.sort_hand();
            }
            Action::SwapHandLeft(i) | Action::SwapHandRight(i) => {
                self.require(LuaState::SelectingHand)?;
                let other = match action {
                    Action::SwapHandLeft(_) => i.checked_sub(1),
                    _ => Some(i + 1),
                };
                let hand = &mut self.gs.areas.hand;
                match other {
                    Some(o) if *i < hand.len() && o < hand.len() => hand.swap(*i, o),
                    _ => return illegal("swap index out of range"),
                }
            }
            Action::SwapJokersLeft(i) | Action::SwapJokersRight(i) => {
                if !matches!(self.rt.state, LuaState::SelectingHand | LuaState::Shop) {
                    return illegal("jokers move in the shop or during a round");
                }
                let other = match action {
                    Action::SwapJokersLeft(_) => i.checked_sub(1),
                    _ => Some(i + 1),
                };
                let jokers = &mut self.gs.areas.jokers;
                match other {
                    Some(o) if *i < jokers.len() && o < jokers.len() => {
                        if jokers[*i].pinned || jokers[o].pinned {
                            return illegal("a pinned joker can't move");
                        }
                        jokers.swap(*i, o);
                    }
                    _ => return illegal("swap index out of range"),
                }
            }
            Action::SkipBlind => {
                self.require(LuaState::BlindSelect)?;
                // Only the Small and Big blinds have a tag to skip for (UI_definitions.lua:1519-1523).
                let slot = match self.gs.blind_on_deck {
                    Some(s @ (BlindSlot::Small | BlindSlot::Big)) => s,
                    _ => return illegal("only the Small and Big blinds can be skipped"),
                };
                self.skip_blind(slot);
            }
            Action::CashOut => {
                self.require(LuaState::RoundEval)?;
                self.cash_out();
            }
            Action::NextRound => {
                self.require(LuaState::Shop)?;
                self.toggle_shop();
            }
            Action::Reroll => {
                self.require(LuaState::Shop)?;
                // G.FUNCS.can_reroll (button_callbacks.lua:2076-2090).
                let cost = self.gs.current_round.reroll_cost;
                if self.spendable() - cost < 0.0 && cost != 0.0 {
                    return illegal("can't afford the reroll");
                }
                self.reroll_shop();
            }
            Action::BuyCard(i) => {
                self.require(LuaState::Shop)?;
                let card = self.shop_card(AreaId::ShopJokers, *i)?;
                // G.FUNCS.can_buy and check_for_buy_space (button_callbacks.lua:55-60, 2392).
                if card.cost > self.spendable() && card.cost > 0.0 {
                    return illegal("can't afford the card");
                }
                if !self.check_for_buy_space(card) {
                    return illegal("no room for the card");
                }
                let id = card.sort_id;
                self.queue(Event::BuyFromShop(id));
            }
            Action::SellCard { area, index } => {
                self.require_sell_state()?;
                let card = match self.gs.areas.get(*area).get(*index) {
                    Some(c) => c,
                    None => return illegal(format!("{area:?} index {index} out of range")),
                };
                // Card:can_sell_card (card.lua:1653-1665).
                if !self.gs.areas.play.is_empty() {
                    return illegal("cards are being played");
                }
                if card.ability.eternal {
                    return illegal("an eternal card can't be sold");
                }
                let id = card.sort_id;
                self.sell_card(id);
            }
            Action::RerollBoss => {
                self.require(LuaState::BlindSelect)?;
                if !self.can_reroll_boss() {
                    return illegal("the boss can't be rerolled");
                }
                self.reroll_boss(false);
            }
            Action::BuyAndUse { index, targets } => {
                self.require(LuaState::Shop)?;
                let card = self.shop_card(AreaId::ShopJokers, *index)?;
                let id = card.sort_id;
                if !card.center().consumeable {
                    return illegal("only a consumable can be bought and used");
                }
                // G.FUNCS.can_buy_and_use (button_callbacks.lua:77-89).
                let cost = card.cost;
                self.highlight_targets(targets.as_deref())?;
                if (cost > self.spendable() && cost > 0.0) || !self.can_use_consumeable(id) {
                    return illegal("the card can't be bought and used now");
                }
                self.queue(Event::BuyAndUse(id));
            }
            Action::RedeemVoucher(i) => {
                self.require(LuaState::Shop)?;
                let card = self.shop_card(AreaId::ShopVouchers, *i)?;
                // G.FUNCS.can_redeem: no free-item exemption (button_callbacks.lua:96-104).
                if card.cost > self.spendable() {
                    return illegal("can't afford the voucher");
                }
                let id = card.sort_id;
                self.redeem_voucher(id);
            }
            Action::OpenBooster(i) => {
                self.require(LuaState::Shop)?;
                let card = self.shop_card(AreaId::ShopBooster, *i)?;
                // G.FUNCS.can_open (button_callbacks.lua:111-119).
                if card.cost > 0.0 && card.cost > self.spendable() {
                    return illegal("can't afford the pack");
                }
                let id = card.sort_id;
                self.open_booster(id);
            }
            Action::PickPackCard { index, targets } => {
                self.require_pack()?;
                let card = self.shop_card(AreaId::PackCards, *index)?;
                let id = card.sort_id;
                if card.center().consumeable {
                    self.highlight_targets(targets.as_deref())?;
                    if !self.can_use_consumeable(id) || self.check_use(id) {
                        return illegal("the card can't be used now");
                    }
                } else if card.set() == jackdaw_data::CenterSet::Joker {
                    // G.FUNCS.can_select_card (button_callbacks.lua:2112-2120).
                    let negative = card
                        .edition
                        .is_some_and(|e| e.kind == crate::card::EditionKind::Negative);
                    if !negative && (self.gs.areas.jokers.len() as f64) >= self.gs.joker_slots {
                        return illegal("no room for the joker");
                    }
                }
                self.pick_pack_card(id);
            }
            Action::SkipPack => {
                self.require_pack()?;
                // G.FUNCS.can_skip_booster (button_callbacks.lua:2132-2141).
                let dealt_or_none = matches!(
                    self.rt.state,
                    LuaState::PlanetPack | LuaState::StandardPack | LuaState::BuffoonPack
                ) || !self.gs.areas.hand.is_empty()
                    || self.hand_limit() <= 0.0;
                if self.gs.areas.pack_cards.is_empty() || !dealt_or_none {
                    return illegal("the pack can't be skipped now");
                }
                self.skip_booster();
            }
            Action::UseConsumable { index, targets } => {
                let card = match self.gs.areas.consumables.get(*index) {
                    Some(c) => c,
                    None => return illegal(format!("consumable index {index} out of range")),
                };
                let id = card.sort_id;
                self.highlight_targets(targets.as_deref())?;
                if !self.can_use_consumeable(id) || self.check_use(id) {
                    return illegal("the consumable can't be used now");
                }
                self.use_consumable_card(id, AreaId::Consumables);
            }
        }
        Ok(())
    }
}
