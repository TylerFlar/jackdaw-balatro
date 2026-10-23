//! Player actions and the step entry point.
//!
//! The actions and their JSON form follow the Python engine (`jackdaw.engine.actions`) and the
//! recorded tapes: `["PlayHand", [0, 2, 3]]`, `["SellCard", "jokers", 1]`, ...

use serde_json::Value;

use crate::area::AreaId;
use crate::flow::HAND_HIGHLIGHT_LIMIT;
use crate::game::{Game, LuaState, StepError};
use crate::joker::PORTED_JOKERS;
use crate::state::HandSort;

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
        if let Some(what) = next.rt.unported.take() {
            return unported(what);
        }
        *self = next;
        Ok(())
    }

    fn require(&self, state: LuaState) -> Result<(), StepError> {
        if self.rt.state != state {
            return illegal(format!("{state:?} only (now {:?})", self.rt.state));
        }
        Ok(())
    }

    /// Refuse a step that would run game logic not ported yet.
    fn check_ported(&self) -> Result<(), StepError> {
        for c in &self.gs.areas.jokers {
            if !PORTED_JOKERS.contains(&c.center.key()) {
                return unported(format!("joker {}", c.center.key()));
            }
        }
        if self.gs.modifier("minus_hand_size_per_X_dollar").is_some() {
            return unported("the minus_hand_size_per_X_dollar challenge rule");
        }
        for t in &self.gs.tags {
            let ty = t
                .get()
                .config("type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !matches!(ty, "round_start_bonus" | "eval" | "immediate") {
                return unported(format!("tag {} ({ty})", t.key()));
            }
        }
        Ok(())
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
                self.check_ported()?;
                self.select_blind();
            }
            Action::PlayHand(cards) => {
                self.require(LuaState::SelectingHand)?;
                if self.gs.current_round.hands_left <= 0.0 {
                    return illegal("no hands left");
                }
                let ids = self.select(cards)?;
                self.check_ported()?;
                self.rt.highlighted = ids;
                self.play_cards_from_highlighted();
            }
            Action::Discard(cards) => {
                self.require(LuaState::SelectingHand)?;
                if self.gs.current_round.discards_left <= 0.0 {
                    return illegal("no discards left");
                }
                let ids = self.select(cards)?;
                self.check_ported()?;
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
            other => return unported(format!("action {other:?}")),
        }
        Ok(())
    }
}
