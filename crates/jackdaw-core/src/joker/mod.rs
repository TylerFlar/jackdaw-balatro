//! `Card:calculate_joker(context)` (card.lua:2291-4062): the context a call carries and the
//! dispatch.
//!
//! Lua passes a table with optional keys and branches on the first context flag that is set
//! (`open_booster`, `buying_card`, ..., then the `G.jokers` contexts `before`, `after` and the
//! main step). [`Kind`] is that flag; the other fields are the table's other keys.
//!
//! The shared parts of the function live here: the debuff check, Blueprint and Brainstorm
//! (card.lua:2304-2334), the `getting_sliced` and `blueprint` guards of some branches, and the
//! main step's value-driven rules (Loyalty Card, then any joker's `x_mult`, `t_mult` and
//! `t_chips`, card.lua:3632-3671). Each joker's own branches are one function, found through
//! the batch registries ([`lookup`]); within a call only one context branch can match, so a
//! joker's function matches on [`Ctx::kind`].
//!
//! A joker counts as ported once it has a function (or is listed in [`PASSIVE`]); a step that
//! would consult any other joker is refused as unported.

pub mod api;
mod batch_a;
mod batch_b;
mod batch_c;
mod batch_d;
mod batch_e;

use jackdaw_data::{BlindId, CenterId, CenterSet};

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

/// A `calculate_joker` context table. The card lists borrow from the caller, so building one
/// allocates nothing.
#[derive(Clone, Debug)]
pub struct Ctx<'a> {
    pub kind: Kind,
    pub cardarea: Option<AreaId>,
    pub full_hand: &'a [u32],
    pub scoring_hand: &'a [u32],
    pub scoring_name: Option<HandType>,
    pub poker_hands: Option<&'a PokerHands>,
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
    /// `remove_playing_cards`'s `removed` and `cards_destroyed`'s `glass_shattered`.
    pub removed: &'a [u32],
    /// `card_effects`: whether each collected effect table is non-empty.
    pub card_effects: &'a [bool],
    /// `setting_blind`'s `blind`.
    pub blind: Option<BlindId>,
    /// `card` (the card bought, sold or added, ...).
    pub card: Option<u32>,
    /// `playing_card_added`'s `cards`.
    pub cards: &'a [u32],
    /// `using_consumeable`'s `consumeable`.
    pub consumeable: Option<u32>,
    /// Blueprint / Brainstorm depth (`context.blueprint`), 0 when not copying.
    pub blueprint: u32,
    /// The first copier (`context.blueprint_card`).
    pub blueprint_card: Option<u32>,
}

impl<'a> Ctx<'a> {
    pub fn new(kind: Kind) -> Ctx<'a> {
        Ctx {
            kind,
            cardarea: None,
            full_hand: &[],
            scoring_hand: &[],
            scoring_name: None,
            poker_hands: None,
            other_card: None,
            other_joker: None,
            destroying_card: None,
            individual: false,
            repetition: false,
            game_over: false,
            hook: false,
            removed: &[],
            card_effects: &[],
            blind: None,
            card: None,
            cards: &[],
            consumeable: None,
            blueprint: 0,
            blueprint_card: None,
        }
    }

    /// `next(context.poker_hands[hand])`.
    pub fn has_hand(&self, hand: HandType) -> bool {
        self.poker_hands.is_some_and(|p| p.contains(hand))
    }

    /// `context.blueprint_card or self`.
    pub fn source(&self, id: u32) -> u32 {
        self.blueprint_card.unwrap_or(id)
    }
}

/// One joker's branches of `calculate_joker`, for the card with this sort id.
pub type CalcFn = fn(&mut Game, u32, &mut Ctx<'_>) -> Option<Effect>;

/// The ported joker with this center key.
fn lookup(key: &str) -> Option<CalcFn> {
    batch_a::lookup(key)
        .or_else(|| batch_b::lookup(key))
        .or_else(|| batch_c::lookup(key))
        .or_else(|| batch_d::lookup(key))
        .or_else(|| batch_e::lookup(key))
}

/// What `calculate_joker` does for a center, decided once per center.
#[derive(Clone, Copy)]
enum Dispatch {
    Blueprint,
    Brainstorm,
    /// The shared rules only.
    Passive,
    Calc(CalcFn),
    Unported,
}

/// [`Dispatch`] by center index.
static DISPATCH: std::sync::LazyLock<Vec<Dispatch>> = std::sync::LazyLock::new(|| {
    jackdaw_data::CENTERS
        .iter()
        .map(|c| match c.key {
            "j_blueprint" => Dispatch::Blueprint,
            "j_brainstorm" => Dispatch::Brainstorm,
            key if PASSIVE.contains(&key) => Dispatch::Passive,
            key => lookup(key).map_or(Dispatch::Unported, Dispatch::Calc),
        })
        .collect()
});

fn dispatch(center: CenterId) -> Dispatch {
    DISPATCH[usize::from(center.0)]
}

/// Jokers with no branch of their own in `calculate_joker`: their effects are passive
/// (`add_to_deck`, hand evaluation flags, prices, rerolls, ...) or the shared rules here.
pub const PASSIVE: &[&str] = &[
    // The main step's shared rules: Loyalty Card, then `t_mult`, `t_chips`, or `x_mult` with a
    // hand type.
    "j_loyalty_card",
    "j_jolly",
    "j_zany",
    "j_mad",
    "j_crazy",
    "j_droll",
    "j_sly",
    "j_wily",
    "j_clever",
    "j_devious",
    "j_crafty",
    "j_duo",
    "j_trio",
    "j_family",
    "j_order",
    "j_tribe",
    // `add_to_deck` / `remove_from_deck` (card.lua:564-700).
    "j_juggler",
    "j_drunkard",
    "j_merry_andy",
    "j_troubadour",
    "j_credit_card",
    "j_chaos",
    "j_oops",
    "j_to_the_moon",
    "j_astronomer",
    // `calculate_dollar_bonus` (card.lua:1655-1679).
    "j_golden",
    "j_cloud_9",
    "j_satellite",
    "j_delayed_grat",
    // Hand evaluation and pools.
    "j_four_fingers",
    "j_shortcut",
    "j_smeared",
    "j_pareidolia",
    "j_splash",
    "j_ring_master",
];

/// Whether every effect of the joker with this center key is ported.
pub fn is_ported(key: &str) -> bool {
    jackdaw_data::center_by_key(key).is_some_and(|c| !matches!(dispatch(c), Dispatch::Unported))
}

/// The center keys of the ported jokers, in center order.
pub fn ported() -> Vec<&'static str> {
    jackdaw_data::CENTERS
        .iter()
        .filter(|c| c.set == CenterSet::Joker && is_ported(c.key))
        .map(|c| c.key)
        .collect()
}

impl Game {
    /// `playing_card_joker_effects(cards)` (misc_functions.lua:1580-1584): every joker sees the
    /// playing cards just added to the deck.
    pub fn playing_card_joker_effects(&mut self, cards: &[u32]) {
        for j in self.joker_ids() {
            let mut ctx = Ctx::new(Kind::PlayingCardAdded);
            ctx.cards = cards;
            self.calculate_joker(j, &mut ctx);
        }
    }

    /// `card:calculate_joker(context)` for any card.
    pub(crate) fn calculate_joker(&mut self, id: u32, ctx: &mut Ctx<'_>) -> Option<Effect> {
        let card = self.gs.areas.joker_card(id)?;
        if card.debuff {
            return None;
        }
        match card.set() {
            CenterSet::Planet => self.observatory(id, ctx),
            CenterSet::Joker => self.calculate_joker_card(id, ctx),
            _ => None,
        }
    }

    /// Observatory: a held Planet multiplies its own hand (card.lua:2293-2302).
    fn observatory(&self, id: u32, ctx: &Ctx<'_>) -> Option<Effect> {
        let card = self.gs.areas.joker_card(id)?;
        let hand = card
            .center()
            .config("hand_type")
            .and_then(|v| v.as_str())
            .and_then(HandType::from_name);
        if ctx.kind == Kind::JokerMain
            && self.gs.sets.used_vouchers.contains(jackdaw_data::centers::V_OBSERVATORY)
            && hand == ctx.scoring_name
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

    /// The joker part of `calculate_joker` (card.lua:2303-4060).
    fn calculate_joker_card(&mut self, id: u32, ctx: &mut Ctx<'_>) -> Option<Effect> {
        let card = self.gs.areas.joker_card(id)?;
        let key = card.center.key();
        let sliced = card.getting_sliced;
        let what = dispatch(card.center);
        // Blueprint copies the joker to its right, Brainstorm the leftmost one.
        let copied = match what {
            Dispatch::Blueprint => {
                let jokers = &self.gs.areas.jokers;
                jokers
                    .iter()
                    .position(|c| c.sort_id == id)
                    .and_then(|i| jokers.get(i + 1))
                    .map(|c| c.sort_id)
            }
            Dispatch::Brainstorm => self.gs.areas.jokers.first().map(|c| c.sort_id),
            _ => None,
        };
        if let Some(other) = copied.filter(|&o| o != id) {
            ctx.blueprint += 1;
            ctx.blueprint_card = Some(ctx.blueprint_card.unwrap_or(id));
            if ctx.blueprint as usize > self.gs.areas.jokers.len() + 1 {
                return None;
            }
            if let Some(mut ret) = self.calculate_joker(other, ctx) {
                ret.card = ctx.blueprint_card;
                return Some(ret);
            }
        }
        match ctx.kind {
            Kind::SettingBlind | Kind::PlayingCardAdded if sliced => return None,
            Kind::EndOfRound if !ctx.individual && !ctx.repetition && ctx.blueprint > 0 => {
                return None;
            }
            Kind::JokerMain => {
                if let Some(e) = self.joker_main_shared(id, ctx) {
                    return Some(e);
                }
            }
            _ => {}
        }
        match what {
            Dispatch::Calc(f) => f(self, id, ctx),
            Dispatch::Unported => {
                self.unported(&format!("joker {key}"));
                None
            }
            _ => None,
        }
    }

    /// The start of the main step (card.lua:3632-3671): Loyalty Card, then the rules every
    /// joker passes through by its ability values.
    fn joker_main_shared(&mut self, id: u32, ctx: &Ctx<'_>) -> Option<Effect> {
        let card = self.gs.areas.joker_card(id)?;
        let name = card.name();
        if name == "Loyalty Card" {
            return self.loyalty_card(id, ctx);
        }
        let a = &card.ability;
        let hand_type = card
            .center()
            .config("type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let type_ok = |any: bool| match HandType::from_name(hand_type) {
            Some(h) => ctx.has_hand(h),
            None => any && hand_type.is_empty(),
        };
        if name != "Seeing Double" && a.x_mult > 1.0 && type_ok(true) {
            return Some(Effect {
                xmult_mod: Some(a.x_mult),
                message: true,
                ..Effect::default()
            });
        }
        if a.t_mult > 0.0 && type_ok(false) {
            return Some(Effect {
                mult_mod: Some(a.t_mult),
                message: true,
                ..Effect::default()
            });
        }
        if a.t_chips > 0.0 && type_ok(false) {
            return Some(Effect {
                chip_mod: Some(a.t_chips),
                message: true,
                ..Effect::default()
            });
        }
        None
    }

    /// Loyalty Card (card.lua:3632-3652): every `every + 1` hands it gives `Xmult`. Without a
    /// Blueprint the shared rules still run after it when it returns nothing.
    fn loyalty_card(&mut self, id: u32, ctx: &Ctx<'_>) -> Option<Effect> {
        let hands_played = self.gs.hands_played;
        let card = self.gs.areas.card_mut(id)?;
        let t = card.ability.extra.table()?;
        let every = t.num("every")?;
        let xmult = t.num("Xmult")?;
        let remaining = crate::lua::modulo(
            every - 1.0 - (hands_played - card.ability.hands_played_at_create),
            every + 1.0,
        );
        card.ability.loyalty_remaining = Some(remaining);
        if remaining == every {
            return Some(Effect {
                xmult_mod: Some(xmult),
                message: true,
                ..Effect::default()
            });
        }
        let _ = ctx;
        None
    }
}
