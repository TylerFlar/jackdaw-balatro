//! The factored action space of `jackdaw/env/action_space.py`: 23 action types, each with an
//! optional entity target and an optional hand-card selection, and the conversions to and from
//! engine actions (`factored_to_engine_action`, `engine_action_to_factored`).
//!
//! RerollBoss (21) and BuyAndUse (22) were appended to the original 21 types, so indices 0-20
//! keep their meaning. Default hand selections come from the engine's own consumable gate
//! ([`Game::consumable_use`]), the same one its legal actions use.

use jackdaw_core::action::Action;
use jackdaw_core::area::AreaId;
use jackdaw_core::card::Card;
use jackdaw_core::game::Game;
use jackdaw_core::state::HandSort;
use jackdaw_data::{CenterSet, Value};

/// The number of factored action types.
pub const NUM_ACTION_TYPES: usize = 23;

/// The most cards a factored selection holds (an engine action takes at most five).
pub const MAX_SELECTION: usize = 16;

/// A factored action type (`ActionType` in Python).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ActionType {
    PlayHand = 0,
    Discard = 1,
    SelectBlind = 2,
    SkipBlind = 3,
    CashOut = 4,
    Reroll = 5,
    NextRound = 6,
    SkipPack = 7,
    BuyCard = 8,
    SellJoker = 9,
    SellConsumable = 10,
    UseConsumable = 11,
    RedeemVoucher = 12,
    OpenBooster = 13,
    PickPackCard = 14,
    SwapJokersLeft = 15,
    SwapJokersRight = 16,
    SwapHandLeft = 17,
    SwapHandRight = 18,
    SortHandRank = 19,
    SortHandSuit = 20,
    RerollBoss = 21,
    BuyAndUse = 22,
}

/// What an action type takes (`ActionTypeSpec`): its Gym enumeration group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Targets {
    /// No target.
    None,
    /// An entity index.
    Entity,
    /// A hand-card selection (PlayHand, Discard).
    Cards,
    /// An entity, plus a hand selection when the entity is a targeting Tarot or Spectral
    /// (UseConsumable over the consumables, PickPackCard over the pack).
    EntityCards,
}

impl ActionType {
    pub const ALL: [ActionType; NUM_ACTION_TYPES] = [
        ActionType::PlayHand,
        ActionType::Discard,
        ActionType::SelectBlind,
        ActionType::SkipBlind,
        ActionType::CashOut,
        ActionType::Reroll,
        ActionType::NextRound,
        ActionType::SkipPack,
        ActionType::BuyCard,
        ActionType::SellJoker,
        ActionType::SellConsumable,
        ActionType::UseConsumable,
        ActionType::RedeemVoucher,
        ActionType::OpenBooster,
        ActionType::PickPackCard,
        ActionType::SwapJokersLeft,
        ActionType::SwapJokersRight,
        ActionType::SwapHandLeft,
        ActionType::SwapHandRight,
        ActionType::SortHandRank,
        ActionType::SortHandSuit,
        ActionType::RerollBoss,
        ActionType::BuyAndUse,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Option<ActionType> {
        ActionType::ALL.get(i).copied()
    }

    /// The Python enum name.
    pub fn name(self) -> &'static str {
        match self {
            ActionType::PlayHand => "PlayHand",
            ActionType::Discard => "Discard",
            ActionType::SelectBlind => "SelectBlind",
            ActionType::SkipBlind => "SkipBlind",
            ActionType::CashOut => "CashOut",
            ActionType::Reroll => "Reroll",
            ActionType::NextRound => "NextRound",
            ActionType::SkipPack => "SkipPack",
            ActionType::BuyCard => "BuyCard",
            ActionType::SellJoker => "SellJoker",
            ActionType::SellConsumable => "SellConsumable",
            ActionType::UseConsumable => "UseConsumable",
            ActionType::RedeemVoucher => "RedeemVoucher",
            ActionType::OpenBooster => "OpenBooster",
            ActionType::PickPackCard => "PickPackCard",
            ActionType::SwapJokersLeft => "SwapJokersLeft",
            ActionType::SwapJokersRight => "SwapJokersRight",
            ActionType::SwapHandLeft => "SwapHandLeft",
            ActionType::SwapHandRight => "SwapHandRight",
            ActionType::SortHandRank => "SortHandRank",
            ActionType::SortHandSuit => "SortHandSuit",
            ActionType::RerollBoss => "RerollBoss",
            ActionType::BuyAndUse => "BuyAndUse",
        }
    }

    /// The targets the type takes (`balatro_game_spec()`'s `needs_entity_target` and
    /// `needs_card_select`).
    pub fn targets(self) -> Targets {
        use ActionType::*;
        match self {
            PlayHand | Discard => Targets::Cards,
            UseConsumable | PickPackCard => Targets::EntityCards,
            BuyCard | SellJoker | SellConsumable | RedeemVoucher | OpenBooster | SwapJokersLeft
            | SwapJokersRight | SwapHandLeft | SwapHandRight | BuyAndUse => Targets::Entity,
            SelectBlind | SkipBlind | CashOut | Reroll | NextRound | SkipPack | SortHandRank
            | SortHandSuit | RerollBoss => Targets::None,
        }
    }
}

/// A hand-card selection: up to [`MAX_SELECTION`] indices, in the given order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Selection {
    idx: [u8; MAX_SELECTION],
    len: u8,
}

impl Selection {
    /// A selection from indices; `None` when there are too many or one is out of `u8` range.
    pub fn new(indices: &[usize]) -> Option<Selection> {
        if indices.len() > MAX_SELECTION {
            return None;
        }
        let mut s = Selection::default();
        for &i in indices {
            s.idx[s.len as usize] = u8::try_from(i).ok()?;
            s.len += 1;
        }
        Some(s)
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.idx[..self.len as usize]
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.as_slice().iter().map(|&i| usize::from(i)).collect()
    }

    pub(crate) fn push(&mut self, i: usize) {
        self.idx[self.len as usize] = i as u8;
        self.len += 1;
    }
}

impl std::fmt::Debug for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

/// A factored action (`FactoredAction`): a type, an entity index and a hand selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FactoredAction {
    pub action_type: ActionType,
    pub entity_target: Option<u32>,
    pub card_target: Option<Selection>,
}

impl FactoredAction {
    pub fn new(action_type: ActionType) -> FactoredAction {
        FactoredAction {
            action_type,
            entity_target: None,
            card_target: None,
        }
    }

    pub fn with_entity(action_type: ActionType, entity: usize) -> FactoredAction {
        FactoredAction {
            action_type,
            entity_target: Some(entity as u32),
            card_target: None,
        }
    }
}

/// Why a factored action has no engine form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvertError(pub String);

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConvertError {}

/// The default hand selection for using the card at `index` of `area` (a consumable in the
/// tray, a pack card, or a shop card bought and used): `default_use_targets`, `None` for a card
/// that takes no selection or has no legal one. A pack pick gets one only for a Tarot or
/// Spectral (`_default_pick_targets`).
pub fn default_targets(game: &Game, area: AreaId, index: usize) -> Option<Vec<usize>> {
    let card = game.gs.areas.get(area).get(index)?;
    if area == AreaId::PackCards && !matches!(card.set(), CenterSet::Tarot | CenterSet::Spectral) {
        return None;
    }
    game.consumable_use(card.sort_id, true).flatten()
}

/// `get_consumable_target_info(card)`: `(min_cards, max_cards, needs_targets)` from the
/// consumable's `max_highlighted` / `min_highlighted` / `mod_num` (`(0, 0, false)` for any
/// card without `max_highlighted`, Aura included).
pub fn consumable_target_info(card: &Card) -> (usize, usize, bool) {
    let center = card.center();
    if !center.consumeable {
        return (0, 0, false);
    }
    let num = |key: &str| match center.config(key) {
        Some(Value::Num(x)) => Some(*x),
        _ => None,
    };
    let Some(max_h) = num("max_highlighted").filter(|&x| x != 0.0) else {
        return (0, 0, false);
    };
    let min_h = num("min_highlighted").unwrap_or(1.0);
    let mod_num = num("mod_num").unwrap_or(max_h);
    (min_h as usize, mod_num as usize, true)
}

fn require_entity(fa: &FactoredAction, what: &str) -> Result<usize, ConvertError> {
    fa.entity_target
        .map(|e| e as usize)
        .ok_or_else(|| ConvertError(format!("{what} requires entity_target")))
}

/// `factored_to_engine_action(fa, game_state)`.
pub fn to_engine(fa: &FactoredAction, game: &Game) -> Result<Action, ConvertError> {
    use ActionType::*;
    // Python's `if not fa.card_target`: no selection, or an empty one.
    let cards = fa.card_target.filter(|s| !s.is_empty()).map(|s| s.to_vec());
    Ok(match fa.action_type {
        PlayHand => Action::PlayHand(
            cards.ok_or_else(|| ConvertError("PlayHand requires card_target".into()))?,
        ),
        Discard => Action::Discard(
            cards.ok_or_else(|| ConvertError("Discard requires card_target".into()))?,
        ),
        SelectBlind => Action::SelectBlind,
        SkipBlind => Action::SkipBlind,
        CashOut => Action::CashOut,
        Reroll => Action::Reroll,
        NextRound => Action::NextRound,
        SkipPack => Action::SkipPack,
        BuyCard => Action::BuyCard(require_entity(fa, "BuyCard")?),
        SellJoker => Action::SellCard {
            area: AreaId::Jokers,
            index: require_entity(fa, "SellJoker")?,
        },
        SellConsumable => Action::SellCard {
            area: AreaId::Consumables,
            index: require_entity(fa, "SellConsumable")?,
        },
        UseConsumable => {
            let index = require_entity(fa, "UseConsumable")?;
            let targets = cards.or_else(|| default_targets(game, AreaId::Consumables, index));
            Action::UseConsumable { index, targets }
        }
        RedeemVoucher => Action::RedeemVoucher(require_entity(fa, "RedeemVoucher")?),
        OpenBooster => Action::OpenBooster(require_entity(fa, "OpenBooster")?),
        PickPackCard => {
            let index = require_entity(fa, "PickPackCard")?;
            // `if targets is None`: an empty selection is passed on as it is.
            let targets = match fa.card_target {
                Some(s) => Some(s.to_vec()),
                None => default_targets(game, AreaId::PackCards, index),
            };
            Action::PickPackCard { index, targets }
        }
        SwapJokersLeft => Action::SwapJokersLeft(require_entity(fa, "SwapJokersLeft")?),
        SwapJokersRight => Action::SwapJokersRight(require_entity(fa, "SwapJokersRight")?),
        SwapHandLeft => Action::SwapHandLeft(require_entity(fa, "SwapHandLeft")?),
        SwapHandRight => Action::SwapHandRight(require_entity(fa, "SwapHandRight")?),
        SortHandRank => Action::SortHand(HandSort::Desc),
        SortHandSuit => Action::SortHand(HandSort::SuitDesc),
        RerollBoss => Action::RerollBoss,
        BuyAndUse => {
            let index = require_entity(fa, "BuyAndUse")?;
            let targets = cards.or_else(|| default_targets(game, AreaId::ShopJokers, index));
            Action::BuyAndUse { index, targets }
        }
    })
}

fn selection(cards: &[usize]) -> Result<Selection, ConvertError> {
    Selection::new(cards).ok_or_else(|| ConvertError(format!("card selection {cards:?}")))
}

fn opt_selection(cards: &Option<Vec<usize>>) -> Result<Option<Selection>, ConvertError> {
    cards.as_deref().map(selection).transpose()
}

/// `engine_action_to_factored(action, game_state)`.
pub fn from_engine(action: &Action) -> Result<FactoredAction, ConvertError> {
    use ActionType as T;
    let entity = |t: T, i: usize| FactoredAction::with_entity(t, i);
    Ok(match action {
        Action::PlayHand(c) | Action::Discard(c) => FactoredAction {
            action_type: if matches!(action, Action::PlayHand(_)) {
                T::PlayHand
            } else {
                T::Discard
            },
            entity_target: None,
            // `card_indices if card_indices else None`.
            card_target: if c.is_empty() {
                None
            } else {
                Some(selection(c)?)
            },
        },
        Action::SelectBlind => FactoredAction::new(T::SelectBlind),
        Action::SkipBlind => FactoredAction::new(T::SkipBlind),
        Action::CashOut => FactoredAction::new(T::CashOut),
        Action::Reroll => FactoredAction::new(T::Reroll),
        Action::NextRound => FactoredAction::new(T::NextRound),
        Action::SkipPack => FactoredAction::new(T::SkipPack),
        Action::BuyCard(i) => entity(T::BuyCard, *i),
        Action::SellCard { area, index } => match area {
            AreaId::Jokers => entity(T::SellJoker, *index),
            AreaId::Consumables => entity(T::SellConsumable, *index),
            other => return Err(ConvertError(format!("Unknown SellCard area: {other:?}"))),
        },
        Action::UseConsumable { index, targets } => FactoredAction {
            action_type: T::UseConsumable,
            entity_target: Some(*index as u32),
            card_target: opt_selection(targets)?,
        },
        Action::RedeemVoucher(i) => entity(T::RedeemVoucher, *i),
        Action::OpenBooster(i) => entity(T::OpenBooster, *i),
        Action::PickPackCard { index, targets } => FactoredAction {
            action_type: T::PickPackCard,
            entity_target: Some(*index as u32),
            card_target: opt_selection(targets)?,
        },
        Action::SortHand(HandSort::Desc) => FactoredAction::new(T::SortHandRank),
        Action::SortHand(HandSort::SuitDesc) => FactoredAction::new(T::SortHandSuit),
        Action::SwapJokersLeft(i) => entity(T::SwapJokersLeft, *i),
        Action::SwapJokersRight(i) => entity(T::SwapJokersRight, *i),
        Action::SwapHandLeft(i) => entity(T::SwapHandLeft, *i),
        Action::SwapHandRight(i) => entity(T::SwapHandRight, *i),
        Action::RerollBoss => FactoredAction::new(T::RerollBoss),
        Action::BuyAndUse { index, targets } => FactoredAction {
            action_type: T::BuyAndUse,
            entity_target: Some(*index as u32),
            card_target: opt_selection(targets)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_keep_their_indices() {
        for (i, t) in ActionType::ALL.iter().enumerate() {
            assert_eq!(t.index(), i);
            assert_eq!(ActionType::from_index(i), Some(*t));
        }
        assert_eq!(ActionType::SortHandSuit.index(), 20);
        assert_eq!(ActionType::RerollBoss.index(), 21);
        assert_eq!(ActionType::BuyAndUse.index(), 22);
        assert_eq!(ActionType::from_index(23), None);
    }

    #[test]
    fn selections_round_trip() {
        let s = Selection::new(&[4, 0, 2]).unwrap();
        assert_eq!(s.to_vec(), vec![4, 0, 2]);
        assert!(Selection::new(&[0; MAX_SELECTION + 1]).is_none());
        assert!(Selection::new(&[300]).is_none());
    }
}
