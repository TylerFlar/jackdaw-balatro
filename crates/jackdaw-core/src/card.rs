//! The card object (`card.lua` `Card`): playing cards, jokers, consumables, vouchers, boosters.
//!
//! One struct for every kind of card, as in Lua. The center (`self.config.center`) carries the
//! static data; `ability` holds the per-card state `Card:set_ability` builds and effects mutate.
//! Methods port the `Card:*` functions that only touch the card itself; they take the few game
//! values they read (`G.GAME.inflation`, the Smeared Joker flag, ...) as arguments, so the game
//! can call them on a card it holds mutably.

use jackdaw_data::{CardId, CenterId, CenterSet, Value, centers};

use crate::hand_eval::HandType;
use crate::rng::Rng;

// ---------------------------------------------------------------------------
// Suits and ranks
// ---------------------------------------------------------------------------

/// A playing-card suit (`card.base.suit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Suit {
    Hearts,
    Diamonds,
    Clubs,
    Spades,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Hearts, Suit::Diamonds, Suit::Clubs, Suit::Spades];

    pub fn name(self) -> &'static str {
        match self {
            Suit::Hearts => "Hearts",
            Suit::Diamonds => "Diamonds",
            Suit::Clubs => "Clubs",
            Suit::Spades => "Spades",
        }
    }

    pub fn from_name(name: &str) -> Option<Suit> {
        Suit::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The `P_CARDS` key letter (`H`, `D`, `C`, `S`).
    pub fn letter(self) -> char {
        match self {
            Suit::Hearts => 'H',
            Suit::Diamonds => 'D',
            Suit::Clubs => 'C',
            Suit::Spades => 'S',
        }
    }

    /// Hearts and Diamonds (Smeared Joker merges suits of one colour).
    pub fn is_red(self) -> bool {
        matches!(self, Suit::Hearts | Suit::Diamonds)
    }

    /// `base.suit_nominal` (`Card:set_base`).
    pub fn nominal(self) -> f64 {
        match self {
            Suit::Diamonds => 0.01,
            Suit::Clubs => 0.02,
            Suit::Hearts => 0.03,
            Suit::Spades => 0.04,
        }
    }

    /// `base.suit_nominal_original` when first set (`Card:set_base`).
    pub fn nominal_original(self) -> f64 {
        match self {
            Suit::Diamonds => 0.001,
            Suit::Clubs => 0.002,
            Suit::Hearts => 0.003,
            Suit::Spades => 0.004,
        }
    }

    /// Bit in [`crate::hand_eval::EvalCard::flush_suits`] (Spades, Hearts, Clubs, Diamonds).
    pub fn flush_bit(self) -> u8 {
        match self {
            Suit::Spades => 1,
            Suit::Hearts => 2,
            Suit::Clubs => 4,
            Suit::Diamonds => 8,
        }
    }
}

/// A playing-card rank (`card.base.value`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rank {
    Two = 2,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    /// `base.id` (2-14).
    pub fn id(self) -> u8 {
        self as u8
    }

    pub fn from_id(id: u8) -> Option<Rank> {
        Rank::ALL.into_iter().find(|r| r.id() == id)
    }

    /// The Lua value string (`"2"` .. `"10"`, `"Jack"`, ..., `"Ace"`).
    pub fn name(self) -> &'static str {
        match self {
            Rank::Two => "2",
            Rank::Three => "3",
            Rank::Four => "4",
            Rank::Five => "5",
            Rank::Six => "6",
            Rank::Seven => "7",
            Rank::Eight => "8",
            Rank::Nine => "9",
            Rank::Ten => "10",
            Rank::Jack => "Jack",
            Rank::Queen => "Queen",
            Rank::King => "King",
            Rank::Ace => "Ace",
        }
    }

    pub fn from_name(name: &str) -> Option<Rank> {
        Rank::ALL.into_iter().find(|r| r.name() == name)
    }

    /// The `P_CARDS` key letter (`2`..`9`, `T`, `J`, `Q`, `K`, `A`).
    pub fn letter(self) -> char {
        match self {
            Rank::Ten => 'T',
            Rank::Jack => 'J',
            Rank::Queen => 'Q',
            Rank::King => 'K',
            Rank::Ace => 'A',
            r => char::from(b'0' + r.id()),
        }
    }

    /// `base.nominal` (chip value).
    pub fn nominal(self) -> f64 {
        match self {
            Rank::Jack | Rank::Queen | Rank::King => 10.0,
            Rank::Ace => 11.0,
            r => f64::from(r.id()),
        }
    }

    /// `base.face_nominal`.
    pub fn face_nominal(self) -> f64 {
        match self {
            Rank::Jack => 0.1,
            Rank::Queen => 0.2,
            Rank::King => 0.3,
            Rank::Ace => 0.4,
            _ => 0.0,
        }
    }
}

/// `card.base`: a playing card's identity (`Card:set_base`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Base {
    /// The `P_CARDS` front (`self.config.card_key`).
    pub front: CardId,
    pub suit: Suit,
    pub value: Rank,
    /// The suit nominal of the card's first suit; kept through suit changes.
    pub suit_nominal_original: f64,
    /// The rank the card was created with; kept through rank changes.
    pub original_value: Rank,
    pub times_played: f64,
}

impl Base {
    pub fn id(&self) -> u8 {
        self.value.id()
    }

    pub fn nominal(&self) -> f64 {
        self.value.nominal()
    }

    pub fn suit_nominal(&self) -> f64 {
        self.suit.nominal()
    }

    pub fn face_nominal(&self) -> f64 {
        self.value.face_nominal()
    }

    /// The `P_CARDS` key for a suit and rank (`"H_A"`).
    pub fn front_for(suit: Suit, value: Rank) -> CardId {
        let key = format!("{}_{}", suit.letter(), value.letter());
        jackdaw_data::card_by_key(&key).expect("every suit and rank has a front")
    }
}

/// `Card:set_base(card, initial)` (card.lua:97-145) for a front, given the current base (if any).
pub fn set_base(current: Option<&Base>, front: CardId, initial: bool) -> Base {
    let proto = front.get();
    let suit = Suit::from_name(proto.suit).expect("front suit");
    let value = Rank::from_name(proto.value).expect("front value");
    // The original suit nominal survives later set_base calls (suit changes).
    let suit_nominal_original = current
        .map(|b| b.suit_nominal_original)
        .unwrap_or_else(|| suit.nominal_original());
    Base {
        front,
        suit,
        value,
        suit_nominal_original,
        original_value: if initial {
            value
        } else {
            current.map_or(value, |b| b.original_value)
        },
        times_played: 0.0,
    }
}

/// `Card:get_nominal(mod)` for a base and `unique_val`.
pub fn nominal(base: &Base, stone: bool, unique_val: f64, suit_sort: bool) -> f64 {
    let mut mult = if suit_sort { 1000.0 } else { 1.0 };
    if stone {
        mult = -1000.0;
    }
    base.nominal()
        + base.suit_nominal() * mult
        + base.suit_nominal_original * 0.0001 * mult
        + base.face_nominal()
        + 0.000_001 * unique_val
}

// ---------------------------------------------------------------------------
// Editions and seals
// ---------------------------------------------------------------------------

/// The edition kinds (`card.edition.type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EditionKind {
    Foil,
    Holo,
    Polychrome,
    Negative,
}

impl EditionKind {
    pub fn name(self) -> &'static str {
        match self {
            EditionKind::Foil => "foil",
            EditionKind::Holo => "holo",
            EditionKind::Polychrome => "polychrome",
            EditionKind::Negative => "negative",
        }
    }

    pub fn from_name(name: &str) -> Option<EditionKind> {
        [Self::Foil, Self::Holo, Self::Polychrome, Self::Negative]
            .into_iter()
            .find(|e| e.name() == name)
    }

    /// The `e_*` center.
    pub fn center(self) -> CenterId {
        match self {
            EditionKind::Foil => centers::E_FOIL,
            EditionKind::Holo => centers::E_HOLO,
            EditionKind::Polychrome => centers::E_POLYCHROME,
            EditionKind::Negative => centers::E_NEGATIVE,
        }
    }

    /// The edition's extra cost (`Card:set_cost`).
    fn extra_cost(self) -> f64 {
        match self {
            EditionKind::Holo => 3.0,
            EditionKind::Foil => 2.0,
            EditionKind::Polychrome | EditionKind::Negative => 5.0,
        }
    }
}

/// `card.edition` as `Card:set_edition` builds it (card.lua:387-462).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edition {
    pub kind: EditionKind,
    /// Foil: `edition.chips`.
    pub chips: Option<f64>,
    /// Holo: `edition.mult`.
    pub mult: Option<f64>,
    /// Polychrome: `edition.x_mult`.
    pub x_mult: Option<f64>,
}

impl Edition {
    /// The edition `set_edition` gives a card, with values from `G.P_CENTERS.e_*.config.extra`.
    pub fn new(kind: EditionKind) -> Edition {
        let extra = kind.center().get().config_num("extra");
        let mut e = Edition {
            kind,
            chips: None,
            mult: None,
            x_mult: None,
        };
        match kind {
            EditionKind::Holo => e.mult = extra,
            EditionKind::Foil => e.chips = extra,
            EditionKind::Polychrome => e.x_mult = extra,
            EditionKind::Negative => {}
        }
        e
    }
}

/// A playing card's seal (`card.seal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Seal {
    Gold,
    Red,
    Blue,
    Purple,
}

impl Seal {
    pub fn name(self) -> &'static str {
        match self {
            Seal::Gold => "Gold",
            Seal::Red => "Red",
            Seal::Blue => "Blue",
            Seal::Purple => "Purple",
        }
    }

    pub fn from_name(name: &str) -> Option<Seal> {
        [Seal::Gold, Seal::Red, Seal::Blue, Seal::Purple]
            .into_iter()
            .find(|s| s.name() == name)
    }
}

// ---------------------------------------------------------------------------
// Ability
// ---------------------------------------------------------------------------

/// A string in `ability.extra`: static data, or text built at runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum LuaStr {
    Static(&'static str),
    Owned(Box<str>),
}

impl LuaStr {
    pub fn as_str(&self) -> &str {
        match self {
            LuaStr::Static(s) => s,
            LuaStr::Owned(s) => s,
        }
    }
}

/// A value in an `ability.extra` table.
#[derive(Clone, Debug, PartialEq)]
pub enum ExtraVal {
    Num(f64),
    Str(LuaStr),
    Bool(bool),
}

const EXTRA_SLOTS: usize = 3;

/// `ability.extra`: nil, a number, a string, or a small table copied from the center config
/// (`copy_table(center.config.extra)`). Table keys are static and kept in key order.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Extra {
    #[default]
    Nil,
    Num(f64),
    Str(&'static str),
    Table(ExtraTable),
}

/// The entries of an `ability.extra` table.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ExtraTable {
    entries: Vec<(&'static str, ExtraVal)>,
}

impl ExtraTable {
    pub fn get(&self, key: &str) -> Option<&ExtraVal> {
        self.entries.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn num(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(ExtraVal::Num(x)) => Some(*x),
            _ => None,
        }
    }

    /// `extra[key] = value` for a key the table already has (the game never adds keys).
    pub fn set(&mut self, key: &str, value: ExtraVal) {
        let slot = self
            .entries
            .iter_mut()
            .find(|(k, _)| *k == key)
            .unwrap_or_else(|| panic!("ability.extra has no key {key:?}"));
        slot.1 = value;
    }

    pub fn entries(&self) -> &[(&'static str, ExtraVal)] {
        &self.entries
    }
}

impl Extra {
    /// `copy_table(center.config.extra)`.
    pub fn from_config(v: Option<&'static Value>) -> Extra {
        match v {
            None => Extra::Nil,
            Some(Value::Num(x)) => Extra::Num(*x),
            Some(Value::Str(s)) => Extra::Str(s),
            Some(Value::Table(entries)) => {
                assert!(entries.len() <= EXTRA_SLOTS, "extra table too large");
                Extra::Table(ExtraTable {
                    entries: entries
                        .iter()
                        .map(|(k, v)| {
                            let val = match v {
                                Value::Num(x) => ExtraVal::Num(*x),
                                Value::Str(s) => ExtraVal::Str(LuaStr::Static(s)),
                                Value::Bool(b) => ExtraVal::Bool(*b),
                                other => panic!("unexpected extra value {other:?}"),
                            };
                            (*k, val)
                        })
                        .collect(),
                })
            }
            Some(Value::List([])) => Extra::Table(ExtraTable::default()),
            Some(other) => panic!("unexpected extra {other:?}"),
        }
    }

    pub fn num(&self) -> Option<f64> {
        match self {
            Extra::Num(x) => Some(*x),
            _ => None,
        }
    }

    pub fn table(&self) -> Option<&ExtraTable> {
        match self {
            Extra::Table(t) => Some(t),
            _ => None,
        }
    }

    pub fn table_mut(&mut self) -> Option<&mut ExtraTable> {
        match self {
            Extra::Table(t) => Some(t),
            _ => None,
        }
    }
}

/// `card.ability` (card.lua:223-342 `Card:set_ability`). `name`, `effect`, `set`, `order`,
/// `type` and `consumeable` come from the center and are read through [`Card::center`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Ability {
    pub mult: f64,
    pub h_mult: f64,
    pub h_x_mult: f64,
    pub h_dollars: f64,
    pub p_dollars: f64,
    pub t_mult: f64,
    pub t_chips: f64,
    pub x_mult: f64,
    pub h_size: f64,
    pub d_size: f64,
    pub extra: Extra,
    pub extra_value: f64,
    /// `forced_selection` (Cerulean Bell); kept through `set_ability`.
    pub forced_selection: bool,
    pub perma_bonus: f64,
    pub bonus: f64,
    pub hands_played_at_create: f64,
    // Set by set_ability for specific jokers.
    pub invis_rounds: Option<f64>,
    pub to_do_poker_hand: Option<HandType>,
    pub caino_xmult: Option<f64>,
    pub yorick_discards: Option<f64>,
    pub burnt_hand: Option<f64>,
    pub loyalty_remaining: Option<f64>,
    // Set during play.
    pub couponed: bool,
    pub played_this_ante: bool,
    // Stickers (`ability.eternal`, `.perishable`, `.perish_tally`, `.rental`).
    pub eternal: bool,
    pub perishable: bool,
    pub perish_tally: Option<f64>,
    pub rental: bool,
}

// ---------------------------------------------------------------------------
// Card
// ---------------------------------------------------------------------------

/// `card.facing`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Facing {
    #[default]
    Front,
    Back,
}

/// A card (`Card`).
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    /// `G.sort_id` at creation; the card's identity (per run).
    pub sort_id: u32,
    pub center: CenterId,
    /// Playing cards only.
    pub base: Option<Base>,
    pub ability: Ability,
    pub edition: Option<Edition>,
    pub seal: Option<Seal>,
    pub debuff: bool,
    pub facing: Facing,
    /// `self.playing_card` (the creation index in `G.playing_cards`).
    pub playing_card: Option<u32>,
    pub base_cost: f64,
    pub extra_cost: f64,
    pub cost: f64,
    pub sell_cost: f64,
    /// `self.added_to_deck`: the passive effects of `add_to_deck` are applied.
    pub added_to_deck: bool,
    /// Challenge sticker: the card stays leftmost and cannot be moved.
    pub pinned: bool,
    /// Within one scoring pass: a Lucky Card roll succeeded.
    pub lucky_trigger: bool,
    /// Within one scoring pass: Vampire took this card's enhancement.
    pub vampired: bool,
    /// Within one step: the card is being destroyed (glass shatter / joker destroy).
    pub shattered: bool,
    /// Within setting_blind: Ceremonial Dagger or Madness is destroying this joker.
    pub getting_sliced: bool,
}

/// The game values `Card:set_cost` reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CostContext {
    /// `G.GAME.inflation`.
    pub inflation: f64,
    /// `G.GAME.discount_percent`.
    pub discount_percent: f64,
    /// `G.GAME.modifiers.booster_ante_scaling`.
    pub booster_ante_scaling: bool,
    /// `G.GAME.round_resets.ante`.
    pub ante: f64,
    /// A (non-debuffed) Astronomer is held.
    pub astronomer: bool,
    /// The card sits in the shop's joker or booster area (couponed cards there are free).
    pub in_shop: bool,
}

impl Card {
    /// `Card:init` minus the UI: `set_ability(center, true)`, `set_base(front, true)`, then
    /// `set_cost()`. `hands_played` is `G.GAME.hands_played`.
    pub fn new(
        sort_id: u32,
        center: CenterId,
        front: Option<CardId>,
        hands_played: f64,
        cost: &CostContext,
    ) -> Card {
        let mut card = Card {
            sort_id,
            center,
            base: None,
            ability: Ability::default(),
            edition: None,
            seal: None,
            debuff: false,
            facing: Facing::Front,
            playing_card: None,
            base_cost: 0.0,
            extra_cost: 0.0,
            cost: 0.0,
            sell_cost: 0.0,
            added_to_deck: false,
            pinned: false,
            lucky_trigger: false,
            vampired: false,
            shattered: false,
            getting_sliced: false,
        };
        card.set_ability(center, hands_played);
        if let Some(front) = front {
            card.base = Some(set_base(None, front, true));
        }
        card.set_cost(cost);
        card
    }

    pub fn center(&self) -> &'static jackdaw_data::Center {
        self.center.get()
    }

    /// `ability.name`.
    pub fn name(&self) -> &'static str {
        self.center().name
    }

    /// `ability.set`.
    pub fn set(&self) -> CenterSet {
        self.center().set
    }

    /// `ability.effect` (`None` when the center has none).
    pub fn effect(&self) -> Option<&'static str> {
        self.center().effect
    }

    pub fn is_stone(&self) -> bool {
        self.effect() == Some("Stone Card")
    }

    pub fn is_joker(&self) -> bool {
        self.set() == CenterSet::Joker
    }

    /// `Card:set_ability(center)` for the per-card state. The caller handles the game-level
    /// side effects: marking `G.GAME.used_jokers` for every center of this name, the To Do List
    /// hand roll ([`Card::roll_to_do_hand`]) and the boss re-debuff on a non-initial call.
    pub fn set_ability(&mut self, center: CenterId, hands_played: f64) {
        let c = center.get();
        let num = |k: &str| c.config_num(k);
        let old = std::mem::take(&mut self.ability);
        let mut ability = Ability {
            mult: num("mult").unwrap_or(0.0),
            h_mult: num("h_mult").unwrap_or(0.0),
            h_x_mult: num("h_x_mult").unwrap_or(0.0),
            h_dollars: num("h_dollars").unwrap_or(0.0),
            p_dollars: num("p_dollars").unwrap_or(0.0),
            t_mult: num("t_mult").unwrap_or(0.0),
            t_chips: num("t_chips").unwrap_or(0.0),
            x_mult: num("Xmult").unwrap_or(1.0),
            h_size: num("h_size").unwrap_or(0.0),
            d_size: num("d_size").unwrap_or(0.0),
            extra: Extra::from_config(c.config("extra")),
            extra_value: 0.0,
            forced_selection: old.forced_selection,
            perma_bonus: old.perma_bonus,
            // The old center's bonus is subtracted from the old table, which set_ability then
            // discards, so the new bonus is just the new center's.
            bonus: num("bonus").unwrap_or(0.0),
            hands_played_at_create: hands_played,
            ..Ability::default()
        };
        match c.name {
            "Invisible Joker" => ability.invis_rounds = Some(0.0),
            "Caino" => ability.caino_xmult = Some(1.0),
            "Yorick" => {
                ability.yorick_discards = ability.extra.table().and_then(|t| t.num("discards"))
            }
            "Loyalty Card" => {
                ability.burnt_hand = Some(0.0);
                ability.loyalty_remaining = ability.extra.table().and_then(|t| t.num("every"));
            }
            _ => {}
        }
        self.center = center;
        self.ability = ability;
        self.base_cost = c.cost.unwrap_or(1.0);
    }

    /// The To Do List hand roll inside `Card:set_ability` (card.lua:311-321): a random visible
    /// hand, never the previous one. `visible_hands` is in `pairs(G.GAME.hands)` order.
    pub fn roll_to_do_hand(&mut self, visible_hands: &[HandType], rng: &mut Rng) {
        let old = self.ability.to_do_poker_hand;
        self.ability.to_do_poker_hand = None;
        while self.ability.to_do_poker_hand.is_none() {
            let i = crate::rng::element_index(visible_hands.len(), rng.seed("to_do"));
            let pick = visible_hands[i];
            if Some(pick) != old {
                self.ability.to_do_poker_hand = Some(pick);
            }
        }
    }

    /// `Card:set_cost()` (card.lua:369-385).
    pub fn set_cost(&mut self, ctx: &CostContext) {
        self.extra_cost = ctx.inflation;
        if let Some(e) = self.edition {
            self.extra_cost += e.kind.extra_cost();
        }
        self.cost = ((self.base_cost + self.extra_cost + 0.5) * (100.0 - ctx.discount_percent)
            / 100.0)
            .floor()
            .max(1.0);
        let set = self.set();
        if set == CenterSet::Booster && ctx.booster_ante_scaling {
            self.cost += ctx.ante - 1.0;
        }
        if (set == CenterSet::Planet
            || (set == CenterSet::Booster && self.name().contains("Celestial")))
            && ctx.astronomer
        {
            self.cost = 0.0;
        }
        if self.ability.rental {
            self.cost = 1.0;
        }
        self.sell_cost = (self.cost / 2.0).floor().max(1.0) + self.ability.extra_value;
        if ctx.in_shop && self.ability.couponed {
            self.cost = 0.0;
        }
    }

    /// `Card:set_edition(edition)` (card.lua:387-462), then `set_cost()`. Returns true when a
    /// Negative edition lands on a card already added to the deck: its area's card limit grows
    /// by one, which the caller applies.
    pub fn set_edition(&mut self, kind: Option<EditionKind>, cost: &CostContext) -> bool {
        let had_edition = self.edition.is_some();
        self.edition = kind.map(Edition::new);
        let negative_slot =
            kind == Some(EditionKind::Negative) && !had_edition && self.added_to_deck;
        self.set_cost(cost);
        negative_slot
    }

    /// `Card:set_seal(seal)`, then `set_cost()`.
    pub fn set_seal(&mut self, seal: Option<Seal>, cost: &CostContext) {
        self.seal = seal;
        self.set_cost(cost);
    }

    /// `Card:set_eternal(eternal)`: only on compatible centers, never with Perishable.
    pub fn set_eternal(&mut self, eternal: bool) {
        self.ability.eternal = false;
        if self.center().eternal_compat && !self.ability.perishable {
            self.ability.eternal = eternal;
        }
    }

    /// `Card:set_perishable()`: only on compatible centers, never with Eternal.
    pub fn set_perishable(&mut self, perishable_rounds: f64) {
        self.ability.perishable = false;
        if self.center().perishable_compat && !self.ability.eternal {
            self.ability.perishable = true;
            self.ability.perish_tally = Some(perishable_rounds);
        }
    }

    /// `Card:set_rental(rental)`, then `set_cost()`.
    pub fn set_rental(&mut self, rental: bool, cost: &CostContext) {
        self.ability.rental = rental;
        self.set_cost(cost);
    }

    /// `Card:get_id()`: the rank id, or `None` for a Stone card that Vampire hasn't drained.
    pub fn get_id(&self) -> Option<u8> {
        if self.is_stone() && !self.vampired {
            return None;
        }
        self.base.map(|b| b.id())
    }

    /// `Card:get_nominal(mod)` (card.lua:950): `mod_mult` is 1 for rank sorting, 1000 for suit
    /// sorting. Lua's `unique_val` (`1 - node_id/1603301`) comes from UI node ids, which follow
    /// creation order; the sort id stands in for the node id.
    pub fn get_nominal(&self, suit_sort: bool) -> f64 {
        let base = self.base.as_ref().expect("get_nominal on a playing card");
        let unique_val = 1.0 - f64::from(self.sort_id) / 1_603_301.0;
        nominal(base, self.is_stone(), unique_val, suit_sort)
    }

    /// `Card:is_suit(suit, bypass_debuff, flush_calc)` (card.lua:4064). `smeared` is
    /// `next(find_joker('Smeared Joker'))`. `None` stands for Lua's bare `return` (nil): a
    /// debuffed card outside flush checks.
    pub fn is_suit(
        &self,
        suit: Suit,
        bypass_debuff: bool,
        flush_calc: bool,
        smeared: bool,
    ) -> Option<bool> {
        let base = self.base.as_ref()?;
        if !flush_calc && self.debuff && !bypass_debuff {
            return None;
        }
        if self.is_stone() {
            return Some(false);
        }
        let wild = self.name() == "Wild Card";
        if wild && (!flush_calc || !self.debuff) {
            return Some(true);
        }
        if smeared && base.suit.is_red() == suit.is_red() {
            return Some(true);
        }
        Some(base.suit == suit)
    }

    /// `card:is_suit(suit, nil, true)` for each flush suit, as hand-evaluation bits.
    pub fn flush_suits(&self, smeared: bool) -> u8 {
        Suit::ALL
            .into_iter()
            .filter(|&s| self.is_suit(s, false, true, smeared) == Some(true))
            .fold(0, |acc, s| acc | s.flush_bit())
    }

    /// `Card:is_face(from_boss)` (card.lua:964): `pareidolia` is `next(find_joker("Pareidolia"))`.
    pub fn is_face(&self, from_boss: bool, pareidolia: bool) -> bool {
        if self.debuff && !from_boss {
            return false;
        }
        matches!(self.get_id(), Some(11..=13)) || pareidolia
    }

    /// `Card:get_chip_bonus()`.
    pub fn get_chip_bonus(&self) -> f64 {
        if self.debuff {
            return 0.0;
        }
        if self.is_stone() {
            return self.ability.bonus + self.ability.perma_bonus;
        }
        self.base.map_or(0.0, |b| b.nominal()) + self.ability.bonus + self.ability.perma_bonus
    }

    /// `Card:get_chip_mult()`: a Lucky Card rolls `lucky_mult` (1 in 5, scaled by
    /// `G.GAME.probabilities.normal`).
    pub fn get_chip_mult(&mut self, rng: &mut Rng, probabilities_normal: f64) -> f64 {
        if self.debuff || self.is_joker() {
            return 0.0;
        }
        if self.effect() == Some("Lucky Card") {
            if rng.random("lucky_mult") < probabilities_normal / 5.0 {
                self.lucky_trigger = true;
                return self.ability.mult;
            }
            return 0.0;
        }
        self.ability.mult
    }

    /// `Card:get_chip_x_mult()`.
    pub fn get_chip_x_mult(&self) -> f64 {
        if self.debuff || self.is_joker() || self.ability.x_mult <= 1.0 {
            return 0.0;
        }
        self.ability.x_mult
    }

    /// `Card:get_chip_h_mult()`.
    pub fn get_chip_h_mult(&self) -> f64 {
        if self.debuff {
            0.0
        } else {
            self.ability.h_mult
        }
    }

    /// `Card:get_chip_h_x_mult()`.
    pub fn get_chip_h_x_mult(&self) -> f64 {
        if self.debuff {
            0.0
        } else {
            self.ability.h_x_mult
        }
    }

    /// `Card:get_edition()`: the edition's scoring bonuses, unless debuffed.
    pub fn get_edition(&self) -> Option<Edition> {
        if self.debuff { None } else { self.edition }
    }

    /// `Card:get_p_dollars()` without the dollar buffer: Gold Seal $3, plus `p_dollars` (a Lucky
    /// Card rolls `lucky_money`, 1 in 15). The caller adds a positive result to
    /// `G.GAME.dollar_buffer`.
    pub fn get_p_dollars(&mut self, rng: &mut Rng, probabilities_normal: f64) -> f64 {
        if self.debuff {
            return 0.0;
        }
        let mut ret = 0.0;
        if self.seal == Some(Seal::Gold) {
            ret += 3.0;
        }
        if self.ability.p_dollars > 0.0 {
            if self.effect() == Some("Lucky Card") {
                if rng.random("lucky_money") < probabilities_normal / 15.0 {
                    self.lucky_trigger = true;
                    ret += self.ability.p_dollars;
                }
            } else {
                ret += self.ability.p_dollars;
            }
        }
        ret
    }

    /// `Card:get_seal(bypass_debuff)`.
    pub fn get_seal(&self, bypass_debuff: bool) -> Option<Seal> {
        if self.debuff && !bypass_debuff {
            None
        } else {
            self.seal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jackdaw_data::{card_by_key, center_by_key};

    fn playing(key: &str, sort_id: u32) -> Card {
        Card::new(
            sort_id,
            centers::C_BASE,
            card_by_key(key),
            0.0,
            &CostContext::default(),
        )
    }

    #[test]
    fn new_playing_card() {
        let c = playing("H_A", 1);
        let base = c.base.unwrap();
        assert_eq!(
            (base.suit, base.value, base.id()),
            (Suit::Hearts, Rank::Ace, 14)
        );
        assert_eq!(c.get_chip_bonus(), 11.0);
        assert_eq!(c.name(), "Default Base");
        // c_base has no cost: base_cost 1 -> cost max(1, floor(1.5)) = 1, sell 1.
        assert_eq!((c.cost, c.sell_cost), (1.0, 1.0));
    }

    #[test]
    fn suit_change_keeps_the_original_nominal() {
        let mut c = playing("H_7", 1);
        let new = set_base(
            c.base.as_ref(),
            Base::front_for(Suit::Spades, Rank::Seven),
            false,
        );
        c.base = Some(new);
        assert_eq!(new.suit, Suit::Spades);
        assert_eq!(new.suit_nominal_original, Suit::Hearts.nominal_original());
        assert_eq!(new.original_value, Rank::Seven);
    }

    #[test]
    fn set_cost_rules() {
        let joker = center_by_key("j_joker").unwrap();
        let ctx = CostContext {
            discount_percent: 25.0,
            ..CostContext::default()
        };
        let mut c = Card::new(1, joker, None, 0.0, &ctx);
        // base 2: floor(2.5 * 0.75) = 1
        assert_eq!((c.cost, c.sell_cost), (1.0, 1.0));
        c.set_edition(Some(EditionKind::Polychrome), &CostContext::default());
        // 2 + 5: floor(7.5) = 7, sell floor(3.5) = 3
        assert_eq!((c.cost, c.sell_cost), (7.0, 3.0));
        assert_eq!(c.edition.unwrap().x_mult, Some(1.5));
        c.ability.couponed = true;
        c.set_cost(&CostContext {
            in_shop: true,
            ..CostContext::default()
        });
        assert_eq!((c.cost, c.sell_cost), (0.0, 3.0));
    }

    #[test]
    fn stickers_respect_compat() {
        let gros = center_by_key("j_gros_michel").unwrap();
        let mut c = Card::new(1, gros, None, 0.0, &CostContext::default());
        c.set_eternal(true);
        assert!(!c.ability.eternal, "Gros Michel is not eternal-compatible");
        c.set_perishable(5.0);
        assert!(c.ability.perishable);
        let joker = center_by_key("j_joker").unwrap();
        let mut j = Card::new(2, joker, None, 0.0, &CostContext::default());
        j.set_perishable(5.0);
        j.set_eternal(true);
        assert!(!j.ability.eternal, "Eternal never joins Perishable");
    }

    #[test]
    fn ability_specials() {
        let loyalty = center_by_key("j_loyalty_card").unwrap();
        let c = Card::new(1, loyalty, None, 3.0, &CostContext::default());
        assert_eq!(c.ability.loyalty_remaining, Some(5.0));
        assert_eq!(c.ability.burnt_hand, Some(0.0));
        assert_eq!(c.ability.hands_played_at_create, 3.0);
        let glass = center_by_key("m_glass").unwrap();
        let g = Card::new(2, glass, card_by_key("S_2"), 0.0, &CostContext::default());
        assert_eq!(g.ability.x_mult, 2.0);
        assert_eq!(g.get_chip_x_mult(), 2.0);
    }
}
