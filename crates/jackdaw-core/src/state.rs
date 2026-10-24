//! The run state: Lua's `G.GAME` plus the card areas, in typed form.
//!
//! Field names follow `G.GAME` (and the canonical snapshot schema, `docs/snapshot-schema.md`).
//! Lua numbers are `f64` throughout, so arithmetic and overflow behave as in the game.

use std::collections::BTreeMap;

use jackdaw_data::{BlindId, CenterId, TagId};

use crate::card::{Card, Rank, Suit};
use crate::hand_eval::HandType;
use crate::rng::Rng;

/// `Clone` for a struct, field by field, with a `clone_from` that clones each field into the
/// old one: vectors and tables keep their allocations, so copying a run into a game that held
/// one allocates little. A field missing from the list fails to build `clone`.
macro_rules! clone_fields {
    ($ty:ident { $($field:ident),* $(,)? }) => {
        impl Clone for $ty {
            fn clone(&self) -> Self {
                $ty { $($field: self.$field.clone()),* }
            }

            fn clone_from(&mut self, source: &Self) {
                $(self.$field.clone_from(&source.$field);)*
            }
        }
    };
}

pub(crate) use clone_fields;

/// The game phase (the engine's view of `G.STATE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    BlindSelect,
    SelectingHand,
    RoundEval,
    Shop,
    PackOpening,
    GameOver,
}

impl Phase {
    pub const ALL: [Phase; 6] = [
        Phase::BlindSelect,
        Phase::SelectingHand,
        Phase::RoundEval,
        Phase::Shop,
        Phase::PackOpening,
        Phase::GameOver,
    ];

    /// The snapshot name (`"blind_select"`, ...).
    pub fn name(self) -> &'static str {
        match self {
            Phase::BlindSelect => "blind_select",
            Phase::SelectingHand => "selecting_hand",
            Phase::RoundEval => "round_eval",
            Phase::Shop => "shop",
            Phase::PackOpening => "pack_opening",
            Phase::GameOver => "game_over",
        }
    }

    pub fn from_name(name: &str) -> Option<Phase> {
        Phase::ALL.into_iter().find(|p| p.name() == name)
    }
}

/// One of the ante's three blinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlindSlot {
    Small,
    Big,
    Boss,
}

impl BlindSlot {
    pub const ALL: [BlindSlot; 3] = [BlindSlot::Small, BlindSlot::Big, BlindSlot::Boss];

    pub fn name(self) -> &'static str {
        match self {
            BlindSlot::Small => "Small",
            BlindSlot::Big => "Big",
            BlindSlot::Boss => "Boss",
        }
    }

    pub fn from_name(name: &str) -> Option<BlindSlot> {
        BlindSlot::ALL.into_iter().find(|b| b.name() == name)
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// `G.hand.config.sort`: the order every automatic hand sort uses (cardarea.lua:577-590). The
/// sort buttons set it (button_callbacks.lua:36-48).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HandSort {
    /// By `get_nominal()`, descending (the default).
    #[default]
    Desc,
    /// By `get_nominal('suit')`, descending.
    SuitDesc,
}

impl HandSort {
    pub fn name(self) -> &'static str {
        match self {
            HandSort::Desc => "desc",
            HandSort::SuitDesc => "suit desc",
        }
    }

    pub fn from_name(name: &str) -> Option<HandSort> {
        [HandSort::Desc, HandSort::SuitDesc]
            .into_iter()
            .find(|s| s.name() == name)
    }
}

/// A value from a dynamic Lua table (`G.GAME.modifiers`, `G.GAME.pool_flags`).
#[derive(Clone, Debug, PartialEq)]
pub enum LuaValue {
    Bool(bool),
    Num(f64),
    Str(String),
    Table(BTreeMap<String, LuaValue>),
}

impl LuaValue {
    pub fn truthy(&self) -> bool {
        !matches!(self, LuaValue::Bool(false))
    }

    pub fn as_num(&self) -> Option<f64> {
        match self {
            LuaValue::Num(x) => Some(*x),
            _ => None,
        }
    }
}

/// `G.GAME.starting_params`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct StartingParams {
    pub dollars: f64,
    pub hand_size: f64,
    pub discards: f64,
    pub hands: f64,
    pub reroll_cost: f64,
    pub joker_slots: f64,
    pub consumable_slots: f64,
    pub ante_scaling: f64,
    pub no_faces: bool,
    pub erratic_suits_and_ranks: bool,
}

/// `G.GAME.round_resets`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RoundResets {
    pub hands: f64,
    pub discards: f64,
    pub reroll_cost: f64,
    pub temp_reroll_cost: Option<f64>,
    pub temp_handsize: Option<f64>,
    pub ante: f64,
    pub blind_ante: f64,
    /// `blind_states[slot]` (`Select`, `Current`, `Upcoming`, `Skipped`, `Defeated`).
    pub blind_states: [&'static str; 3],
    /// `blind_choices[slot]`; the boss is unset before the first ante is rolled.
    pub blind_choices: [Option<BlindId>; 3],
    /// `blind_tags` for Small and Big; `None` when not rolled yet (`blind_tags` absent).
    pub blind_tags: Option<[Option<TagId>; 2]>,
    pub boss_rerolled: bool,
}

impl RoundResets {
    /// The values `blind_states` takes.
    pub const STATES: [&'static str; 6] = [
        "Select", "Current", "Upcoming", "Skipped", "Defeated", "Hide",
    ];

    pub fn blind_state(&self, slot: BlindSlot) -> &'static str {
        self.blind_states[slot.index()]
    }

    pub fn set_blind_state(&mut self, slot: BlindSlot, state: &'static str) {
        self.blind_states[slot.index()] = state;
    }

    pub fn blind_choice(&self, slot: BlindSlot) -> Option<BlindId> {
        self.blind_choices[slot.index()]
    }
}

/// `G.GAME.current_round`.
#[derive(Clone, Debug, PartialEq)]
pub struct CurrentRound {
    pub hands_left: f64,
    pub hands_played: f64,
    pub discards_left: f64,
    pub discards_used: f64,
    pub reroll_cost: f64,
    pub reroll_cost_increase: f64,
    pub free_rerolls: f64,
    pub jokers_purchased: f64,
    /// The ante's shop voucher.
    pub voucher: Option<CenterId>,
    pub most_played_poker_hand: HandType,
    /// The Idol's card this round.
    pub idol_card: (Suit, Rank),
    /// Mail-In Rebate's rank.
    pub mail_card: Rank,
    /// Ancient Joker's suit (unset before the first roll).
    pub ancient_card: Option<Suit>,
    /// Castle's suit.
    pub castle_card: Suit,
}

/// `G.GAME.round_bonus`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RoundBonus {
    pub next_hands: f64,
    pub discards: f64,
}

/// The current blind (`G.GAME.blind`).
#[derive(Clone, Debug, PartialEq)]
pub struct BlindState {
    /// `None` for the empty blind between rounds.
    pub key: Option<BlindId>,
    pub chips: f64,
    pub mult: f64,
    pub dollars: f64,
    pub boss: bool,
    pub disabled: bool,
    pub triggered: bool,
    /// The Fish / Crimson Heart one-shot flag.
    pub prepped: bool,
    /// The Eye: hands already played this round.
    pub eye_hands: Vec<HandType>,
    /// The Mouth: the locked hand.
    pub mouth_hand: Option<HandType>,
    /// The Water: discards it removed.
    pub discards_sub: Option<f64>,
    /// The Needle: hands it removed.
    pub hands_sub: Option<f64>,
}

/// A poker hand's row in `G.GAME.hands`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandState {
    pub level: f64,
    pub chips: f64,
    pub mult: f64,
    pub played: f64,
    pub played_this_round: f64,
    pub visible: bool,
}

const CENTER_WORDS: usize = jackdaw_data::CENTERS.len().div_ceil(64);

/// A set of centers (`G.GAME.used_jokers`, `G.GAME.used_vouchers`), one bit per center.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct CenterBits([u64; CENTER_WORDS]);

impl CenterBits {
    pub fn contains(&self, c: CenterId) -> bool {
        let i = usize::from(c.0);
        self.0[i / 64] & (1 << (i % 64)) != 0
    }

    pub fn insert(&mut self, c: CenterId) {
        let i = usize::from(c.0);
        self.0[i / 64] |= 1 << (i % 64);
    }

    pub fn remove(&mut self, c: CenterId) {
        let i = usize::from(c.0);
        self.0[i / 64] &= !(1 << (i % 64));
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// The members, in center order.
    pub fn iter(&self) -> impl Iterator<Item = CenterId> + '_ {
        (0..jackdaw_data::CENTERS.len())
            .map(|i| CenterId(i as u16))
            .filter(|&c| self.contains(c))
    }

    /// The members' keys in key order, as Lua's `pairs` order would never give but the snapshot
    /// lists them.
    pub fn sorted_keys(&self) -> Vec<&'static str> {
        let mut keys: Vec<&'static str> = self.iter().map(CenterId::key).collect();
        keys.sort_unstable();
        keys
    }
}

impl std::fmt::Debug for CenterBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.sorted_keys()).finish()
    }
}

impl FromIterator<CenterId> for CenterBits {
    fn from_iter<I: IntoIterator<Item = CenterId>>(iter: I) -> CenterBits {
        let mut bits = CenterBits::default();
        for c in iter {
            bits.insert(c);
        }
        bits
    }
}

/// `G.GAME.bosses_used`: how often each boss appeared, by blind index (`None`: no entry).
#[derive(Clone, Copy, PartialEq)]
pub struct BossCounts([Option<f64>; jackdaw_data::BLINDS.len()]);

impl BossCounts {
    pub fn empty() -> BossCounts {
        BossCounts([None; jackdaw_data::BLINDS.len()])
    }

    pub fn get(&self, b: BlindId) -> Option<f64> {
        self.0[usize::from(b.0)]
    }

    pub fn set(&mut self, b: BlindId, n: f64) {
        self.0[usize::from(b.0)] = Some(n);
    }

    /// The entries, in blind order.
    pub fn iter(&self) -> impl Iterator<Item = (BlindId, f64)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.map(|n| (BlindId(i as u8), n)))
    }
}

impl Default for BossCounts {
    fn default() -> BossCounts {
        BossCounts::empty()
    }
}

impl std::fmt::Debug for BossCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.iter().map(|(b, n)| (b.key(), n)))
            .finish()
    }
}

/// `G.GAME.used_jokers` and friends.
#[derive(Debug, PartialEq, Default)]
pub struct Sets {
    /// Every center created this run (`used_jokers` covers all sets).
    pub used_jokers: CenterBits,
    pub used_vouchers: CenterBits,
    /// Challenge bans: centers, tags or blinds.
    pub banned_keys: std::collections::BTreeSet<String>,
    pub pool_flags: BTreeMap<String, LuaValue>,
    /// Times each boss appeared.
    pub bosses_used: BossCounts,
}

/// `G.GAME.consumeable_usage` / `consumeable_usage_total`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ConsumeableUsage {
    pub by_key: BTreeMap<&'static str, f64>,
    /// `tarot`, `planet`, `spectral`, `tarot_planet`, `all` (empty until the first use).
    pub totals: BTreeMap<String, f64>,
}

/// The card areas, in area order.
#[derive(Debug, PartialEq, Default)]
pub struct Areas {
    /// The draw pile; index 0 is the bottom, cards are drawn from the end.
    pub deck: Vec<Card>,
    pub hand: Vec<Card>,
    pub discard: Vec<Card>,
    pub play: Vec<Card>,
    pub jokers: Vec<Card>,
    pub consumables: Vec<Card>,
    pub shop_jokers: Vec<Card>,
    pub shop_vouchers: Vec<Card>,
    pub shop_booster: Vec<Card>,
    pub pack_cards: Vec<Card>,
}

impl Areas {
    pub const NAMES: [&'static str; 10] = [
        "deck",
        "hand",
        "discard",
        "play",
        "jokers",
        "consumables",
        "shop_jokers",
        "shop_vouchers",
        "shop_booster",
        "pack_cards",
    ];

    pub fn by_name(&self, name: &str) -> Option<&Vec<Card>> {
        Some(match name {
            "deck" => &self.deck,
            "hand" => &self.hand,
            "discard" => &self.discard,
            "play" => &self.play,
            "jokers" => &self.jokers,
            "consumables" => &self.consumables,
            "shop_jokers" => &self.shop_jokers,
            "shop_vouchers" => &self.shop_vouchers,
            "shop_booster" => &self.shop_booster,
            "pack_cards" => &self.pack_cards,
            _ => return None,
        })
    }

    pub fn by_name_mut(&mut self, name: &str) -> Option<&mut Vec<Card>> {
        Some(match name {
            "deck" => &mut self.deck,
            "hand" => &mut self.hand,
            "discard" => &mut self.discard,
            "play" => &mut self.play,
            "jokers" => &mut self.jokers,
            "consumables" => &mut self.consumables,
            "shop_jokers" => &mut self.shop_jokers,
            "shop_vouchers" => &mut self.shop_vouchers,
            "shop_booster" => &mut self.shop_booster,
            "pack_cards" => &mut self.pack_cards,
            _ => return None,
        })
    }
}

/// An open booster pack.
#[derive(Clone, Debug, PartialEq)]
pub struct PackState {
    /// `Arcana`, `Celestial`, `Spectral`, `Standard`, `Buffoon`.
    pub kind: String,
    pub choices_remaining: f64,
    /// Sort ids of the cards an Arcana/Spectral pack dealt into the hand, still held.
    pub dealt: Vec<u32>,
    /// The phase the pack returns to.
    pub return_phase: Phase,
}

/// A tag in `G.GAME.tags` (`Tag`, tag.lua).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldTag {
    pub id: TagId,
    /// `self.triggered`: the tag has fired and leaves `G.GAME.tags` with its `yep` / `nope`
    /// events. Every step ends after those events, so a snapshot never holds one.
    pub triggered: bool,
    /// The Orbital Tag's hand (`self.ability.orbital_hand`, tag.lua:101-112).
    pub orbital_hand: Option<HandType>,
}

impl HeldTag {
    pub fn new(id: TagId) -> HeldTag {
        HeldTag {
            id,
            triggered: false,
            orbital_hand: None,
        }
    }
}

/// The last played hand's result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LastScore {
    pub hand_type: HandType,
    pub chips: f64,
    pub mult: f64,
    pub total: f64,
    pub dollars: f64,
    pub debuffed: bool,
}

/// Simulator state with no single `G.GAME` field.
#[derive(Debug, PartialEq, Default)]
pub struct SimState {
    /// The sort id the next card gets.
    pub next_sort_id: u32,
    /// Booster keys queued by tags (Double Tag), opened when the current pack closes.
    pub pending_tag_packs: Vec<CenterId>,
    /// Marble Joker stones joining the bottom of the deck after the next round-start shuffle.
    pub pending_deck_bottom: Vec<Card>,
    pub shop_d6ed: bool,
    /// A Coupon Tag already applied this shop (`G.GAME.shop_free`, tag.lua:448).
    pub shop_free: bool,
    /// Luxury Tax: `G.hand.config.last_poll_size` (cardarea.lua:245).
    pub hand_last_poll_size: f64,
    pub last_blind_was_boss: bool,
    /// What `cash_out` pays (only meaningful in round_eval).
    pub pending_cashout: f64,
    pub last_score: Option<LastScore>,
    /// Every tag awarded this run, minus consumed Double Tags (observation v1 encodes it).
    pub tag_history: Vec<TagId>,
}

/// The whole run.
#[derive(Debug, PartialEq)]
pub struct GameState {
    pub phase: Phase,
    // run
    pub back: CenterId,
    pub stake: f64,
    pub challenge: Option<String>,
    pub win_ante: f64,
    pub won: bool,
    // game scalars
    pub dollars: f64,
    pub chips: f64,
    pub round: f64,
    pub skips: f64,
    pub hands_played: f64,
    pub unused_discards: f64,
    pub interest_cap: f64,
    pub interest_amount: f64,
    pub discount_percent: f64,
    pub inflation: f64,
    pub rental_rate: f64,
    pub bankrupt_at: f64,
    pub base_reroll_cost: f64,
    pub edition_rate: f64,
    pub joker_rate: f64,
    pub tarot_rate: f64,
    pub planet_rate: f64,
    pub spectral_rate: f64,
    pub playing_card_rate: f64,
    pub probabilities_normal: f64,
    /// `G.hand.config.real_card_limit` (the hand size before clamping at 0).
    pub hand_size: f64,
    pub hand_sort: HandSort,
    pub joker_slots: f64,
    pub consumable_slots: f64,
    pub starting_deck_size: f64,
    pub last_tarot_planet: Option<CenterId>,
    pub last_hand_played: Option<HandType>,
    pub pack_size: f64,
    pub ecto_minus: f64,
    pub perishable_rounds: f64,
    pub blind_on_deck: Option<BlindSlot>,
    pub shop_joker_max: f64,
    pub first_shop_buffoon: bool,
    /// `G.GAME.orbital_choices`: ante (as its string key) -> the Orbital Tag hand for the
    /// Small, Big and Boss blind (UI_definitions.lua:1506-1516).
    pub orbital_choices: BTreeMap<String, [HandType; 3]>,
    // tables
    pub starting_params: StartingParams,
    pub modifiers: BTreeMap<String, LuaValue>,
    pub round_resets: RoundResets,
    pub current_round: CurrentRound,
    pub round_bonus: RoundBonus,
    pub blind: Option<BlindState>,
    pub hands: [HandState; 12],
    /// `G.GAME.tags`, in order.
    pub tags: Vec<HeldTag>,
    pub sets: Sets,
    pub consumeable_usage: ConsumeableUsage,
    pub areas: Areas,
    pub pack: Option<PackState>,
    pub rng: Rng,
    pub sim: SimState,
}

impl GameState {
    pub fn hand(&self, hand: HandType) -> &HandState {
        &self.hands[hand.index()]
    }

    pub fn hand_mut(&mut self, hand: HandType) -> &mut HandState {
        &mut self.hands[hand.index()]
    }

    /// `G.GAME.modifiers[key]` when set and truthy.
    pub fn modifier(&self, key: &str) -> Option<&LuaValue> {
        self.modifiers.get(key).filter(|v| v.truthy())
    }

    /// Allocate the next card sort id (`G.sort_id = G.sort_id + 1`).
    pub fn next_sort_id(&mut self) -> u32 {
        let id = self.sim.next_sort_id;
        self.sim.next_sort_id += 1;
        id
    }
}

clone_fields!(Sets {
    used_jokers,
    used_vouchers,
    banned_keys,
    pool_flags,
    bosses_used,
});

clone_fields!(Areas {
    deck,
    hand,
    discard,
    play,
    jokers,
    consumables,
    shop_jokers,
    shop_vouchers,
    shop_booster,
    pack_cards,
});

clone_fields!(SimState {
    next_sort_id,
    pending_tag_packs,
    pending_deck_bottom,
    shop_d6ed,
    shop_free,
    hand_last_poll_size,
    last_blind_was_boss,
    pending_cashout,
    last_score,
    tag_history,
});

clone_fields!(GameState {
    phase,
    back,
    stake,
    challenge,
    win_ante,
    won,
    dollars,
    chips,
    round,
    skips,
    hands_played,
    unused_discards,
    interest_cap,
    interest_amount,
    discount_percent,
    inflation,
    rental_rate,
    bankrupt_at,
    base_reroll_cost,
    edition_rate,
    joker_rate,
    tarot_rate,
    planet_rate,
    spectral_rate,
    playing_card_rate,
    probabilities_normal,
    hand_size,
    hand_sort,
    joker_slots,
    consumable_slots,
    starting_deck_size,
    last_tarot_planet,
    last_hand_played,
    pack_size,
    ecto_minus,
    perishable_rounds,
    blind_on_deck,
    shop_joker_max,
    first_shop_buffoon,
    orbital_choices,
    starting_params,
    modifiers,
    round_resets,
    current_round,
    round_bonus,
    blind,
    hands,
    tags,
    sets,
    consumeable_usage,
    areas,
    pack,
    rng,
    sim,
});
