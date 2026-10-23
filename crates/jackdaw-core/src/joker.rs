//! `Card:calculate_joker(context)` (card.lua:2291-4062): the context a call carries, and the
//! dispatch.
//!
//! Lua passes a table with optional keys and branches on the first context flag that is set
//! (`open_booster`, `buying_card`, ..., then the `G.jokers` contexts `before`, `after` and the
//! main step). [`Kind`] is that flag; the other fields are the table's other keys.
//!
//! Joker effects are ported in batches. [`PORTED_JOKERS`] lists the jokers whose every context
//! is in; a step that would consult any other joker is refused as unported.

use jackdaw_data::{BlindId, CenterSet};

use crate::area::AreaId;
use crate::game::Game;
use crate::hand_eval::{HandType, PokerHands};
use crate::scoring::Effect;

/// The context flag `calculate_joker` branches on, in Lua's branch order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    OpenBooster,
    BuyingCard,
    SellingSelf,
    SellingCard,
    RerollShop,
    EndingShop,
    SkipBlind,
    SkippingBooster,
    PlayingCardAdded,
    FirstHandDrawn,
    SettingBlind,
    DestroyingCard,
    CardsDestroyed,
    RemovePlayingCards,
    UsingConsumeable,
    DebuffedHand,
    PreDiscard,
    Discard,
    EndOfRound,
    Individual,
    Repetition,
    OtherJoker,
    /// `cardarea == G.jokers` with `before`.
    Before,
    /// `cardarea == G.jokers` with `after`.
    After,
    /// `cardarea == G.jokers` with neither (`joker_main`).
    JokerMain,
}

/// A `calculate_joker` context table.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub kind: Kind,
    pub cardarea: Option<AreaId>,
    pub full_hand: Vec<u32>,
    pub scoring_hand: Vec<u32>,
    pub scoring_name: Option<HandType>,
    pub poker_hands: Option<PokerHands>,
    pub other_card: Option<u32>,
    pub other_joker: Option<u32>,
    pub destroying_card: Option<u32>,
    /// `end_of_round` with `individual`.
    pub individual: bool,
    /// `end_of_round` with `repetition`.
    pub repetition: bool,
    pub game_over: bool,
    /// `pre_discard` from The Hook.
    pub hook: bool,
    /// `remove_playing_cards`'s `removed`.
    pub removed: Vec<u32>,
    /// `card_effects`: whether each collected effect table is non-empty.
    pub card_effects: Vec<bool>,
    /// `setting_blind`'s `blind`.
    pub blind: Option<BlindId>,
    /// `card` (buying, selling, the card added, ...).
    pub card: Option<u32>,
    /// `using_consumeable`'s `consumeable`.
    pub consumeable: Option<u32>,
    /// Blueprint / Brainstorm depth (`context.blueprint`), 0 when not copying.
    pub blueprint: u32,
    pub blueprint_card: Option<u32>,
}

impl Ctx {
    pub fn new(kind: Kind) -> Ctx {
        Ctx {
            kind,
            cardarea: None,
            full_hand: Vec::new(),
            scoring_hand: Vec::new(),
            scoring_name: None,
            poker_hands: None,
            other_card: None,
            other_joker: None,
            destroying_card: None,
            individual: false,
            repetition: false,
            game_over: false,
            hook: false,
            removed: Vec::new(),
            card_effects: Vec::new(),
            blind: None,
            card: None,
            consumeable: None,
            blueprint: 0,
            blueprint_card: None,
        }
    }
}

/// The jokers whose effects are ported.
pub const PORTED_JOKERS: &[&str] = &[];

impl Game {
    /// `card:calculate_joker(context)` for any card.
    pub(crate) fn calculate_joker(&mut self, id: u32, ctx: &mut Ctx) -> Option<Effect> {
        let card = self.gs.areas.card(id)?;
        if card.debuff {
            return None;
        }
        match card.set() {
            CenterSet::Planet => {
                // Observatory: a held Planet multiplies its own hand (card.lua:2293-2302).
                if ctx.kind == Kind::JokerMain
                    && self.gs.sets.used_vouchers.contains("v_observatory")
                    && card
                        .center()
                        .config("hand_type")
                        .and_then(|v| v.as_str())
                        .and_then(HandType::from_name)
                        == ctx.scoring_name
                {
                    let x = jackdaw_data::centers::V_OBSERVATORY
                        .get()
                        .config_num("extra")
                        .expect("Observatory's multiplier");
                    return Some(Effect {
                        xmult_mod: Some(x),
                        message: true,
                        ..Effect::default()
                    });
                }
                None
            }
            CenterSet::Joker => {
                debug_assert!(
                    PORTED_JOKERS.contains(&card.center.key()),
                    "unported joker {} reached calculate_joker",
                    card.center.key()
                );
                None
            }
            _ => None,
        }
    }
}
