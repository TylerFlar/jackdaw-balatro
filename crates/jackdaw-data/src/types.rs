//! The types the generated tables are made of.

/// A static Lua value from the game's data tables (configs, challenge rules, blind debuffs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Num(f64),
    Str(&'static str),
    /// A sequence (`{a, b, c}`); an empty Lua table is an empty list.
    List(&'static [Value]),
    /// A table with string keys, sorted by key.
    Table(&'static [(&'static str, Value)]),
}

impl Value {
    pub fn as_num(&self) -> Option<f64> {
        match *self {
            Value::Num(x) => Some(x),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&'static str> {
        match *self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_list(&self) -> &'static [Value] {
        match *self {
            Value::List(items) => items,
            _ => &[],
        }
    }

    /// The entries of a table; an empty Lua table (`{}`, written as an empty list) has none.
    pub fn as_table(&self) -> &'static [(&'static str, Value)] {
        match *self {
            Value::Table(entries) => entries,
            _ => &[],
        }
    }

    /// `t[key]` for a table value.
    pub fn get(&self, key: &str) -> Option<&'static Value> {
        lookup(self.as_table(), key)
    }
}

/// `t[key]` in a sorted table.
pub fn lookup(entries: &'static [(&'static str, Value)], key: &str) -> Option<&'static Value> {
    entries
        .binary_search_by(|(k, _)| (*k).cmp(key))
        .ok()
        .map(|i| &entries[i].1)
}

/// Index into [`crate::CENTERS`]: a `G.P_CENTERS` entry.
///
/// Ids are ordered by set (Joker, Tarot, Planet, Spectral, Voucher, Booster, Enhanced, Edition,
/// Back, Default, then the set-less sprite centers) and by Lua `order` within a set, so each set
/// is a contiguous range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CenterId(pub u16);

impl CenterId {
    pub fn get(self) -> &'static Center {
        &crate::CENTERS[self.0 as usize]
    }

    pub fn key(self) -> &'static str {
        self.get().key
    }
}

/// `center.set`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CenterSet {
    Joker,
    Tarot,
    Planet,
    Spectral,
    Voucher,
    Booster,
    Enhanced,
    Edition,
    Back,
    Default,
    /// Sprite-only centers without a set (`soul`, `undiscovered_*`).
    None,
}

impl CenterSet {
    /// The Lua name (`"Joker"`, ...); empty for [`CenterSet::None`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Joker => "Joker",
            Self::Tarot => "Tarot",
            Self::Planet => "Planet",
            Self::Spectral => "Spectral",
            Self::Voucher => "Voucher",
            Self::Booster => "Booster",
            Self::Enhanced => "Enhanced",
            Self::Edition => "Edition",
            Self::Back => "Back",
            Self::Default => "Default",
            Self::None => "",
        }
    }
}

/// A `G.P_CENTERS` prototype. Sprite and unlock-condition fields are left out.
#[derive(Debug)]
pub struct Center {
    pub key: &'static str,
    /// `center.name` (empty for the three set-less sprite centers).
    pub name: &'static str,
    pub set: CenterSet,
    pub order: Option<u16>,
    pub cost: Option<f64>,
    pub cost_mult: Option<f64>,
    pub rarity: Option<u8>,
    /// `center.effect`; `None` and `Some("")` differ (BalatroBot reports them differently).
    pub effect: Option<&'static str>,
    pub label: Option<&'static str>,
    pub consumeable: bool,
    pub blueprint_compat: bool,
    pub eternal_compat: bool,
    pub perishable_compat: bool,
    pub unlocked: Option<bool>,
    pub discovered: Option<bool>,
    pub hidden: bool,
    pub omit: bool,
    pub available: Option<bool>,
    /// Booster kind (`"Arcana"`, `"Celestial"`, ...).
    pub kind: Option<&'static str>,
    /// Booster weight.
    pub weight: Option<f64>,
    pub freq: Option<f64>,
    pub max: Option<f64>,
    /// Back unlock stake.
    pub stake: Option<u8>,
    /// Voucher prerequisites (`center.requires`).
    pub requires: &'static [&'static str],
    pub enhancement_gate: Option<&'static str>,
    pub yes_pool_flag: Option<&'static str>,
    pub no_pool_flag: Option<&'static str>,
    pub soul_pos: bool,
    /// `center.config`, sorted by key.
    pub config: &'static [(&'static str, Value)],
}

impl Center {
    /// `center.config[key]`.
    pub fn config(&self, key: &str) -> Option<&'static Value> {
        lookup(self.config, key)
    }

    /// `center.config[key]` as a number.
    pub fn config_num(&self, key: &str) -> Option<f64> {
        self.config(key).and_then(Value::as_num)
    }
}

/// Index into [`crate::BLINDS`] (`G.P_BLINDS`), in Lua `order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlindId(pub u8);

impl BlindId {
    pub fn get(self) -> &'static Blind {
        &crate::BLINDS[self.0 as usize]
    }

    pub fn key(self) -> &'static str {
        self.get().key
    }
}

/// `blind.boss`: the ante range a boss can appear in.
#[derive(Debug)]
pub struct BossRange {
    pub min: u8,
    pub max: u8,
    pub showdown: bool,
}

/// A `G.P_BLINDS` prototype.
#[derive(Debug)]
pub struct Blind {
    pub key: &'static str,
    pub name: &'static str,
    pub order: u8,
    pub mult: f64,
    pub dollars: f64,
    pub boss: Option<BossRange>,
    /// `blind.debuff`, sorted by key (empty when the blind debuffs nothing).
    pub debuff: &'static [(&'static str, Value)],
}

/// Index into [`crate::TAGS`] (`G.P_TAGS`), in Lua `order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TagId(pub u8);

impl TagId {
    pub fn get(self) -> &'static Tag {
        &crate::TAGS[self.0 as usize]
    }

    pub fn key(self) -> &'static str {
        self.get().key
    }
}

/// A `G.P_TAGS` prototype.
#[derive(Debug)]
pub struct Tag {
    pub key: &'static str,
    pub name: &'static str,
    pub order: u8,
    pub min_ante: Option<u8>,
    /// The center that must be discovered for the tag to appear.
    pub requires: Option<&'static str>,
    /// `tag.config`, sorted by key (`type` is the context the tag fires in).
    pub config: &'static [(&'static str, Value)],
}

impl Tag {
    pub fn config(&self, key: &str) -> Option<&'static Value> {
        lookup(self.config, key)
    }
}

/// A `G.P_STAKES` prototype, in Lua `order` ([`crate::STAKES`]).
#[derive(Debug)]
pub struct Stake {
    pub key: &'static str,
    pub name: &'static str,
    pub order: u8,
    pub stake_level: u8,
}

/// A `G.P_SEALS` entry, in Lua `order` ([`crate::SEALS`]).
#[derive(Debug)]
pub struct SealProto {
    pub key: &'static str,
    pub order: u8,
}

/// Index into [`crate::P_CARDS`]: playing-card fronts in bytewise key order (`C_2` .. `S_T`),
/// the order `pseudorandom_element` sorts `G.P_CARDS` into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CardId(pub u8);

impl CardId {
    pub fn get(self) -> &'static PlayingCard {
        &crate::P_CARDS[self.0 as usize]
    }
}

/// A `G.P_CARDS` entry.
#[derive(Debug)]
pub struct PlayingCard {
    pub key: &'static str,
    pub name: &'static str,
    /// `"Hearts"`, `"Diamonds"`, `"Clubs"`, `"Spades"`.
    pub suit: &'static str,
    /// `"2"` .. `"10"`, `"Jack"`, `"Queen"`, `"King"`, `"Ace"`.
    pub value: &'static str,
}

/// A poker hand's base row in `G.GAME.hands` (`Game:init_game_object`).
#[derive(Debug)]
pub struct HandBase {
    pub name: &'static str,
    pub order: u8,
    pub visible: bool,
    pub s_chips: f64,
    pub s_mult: f64,
    pub l_chips: f64,
    pub l_mult: f64,
    /// Example cards (`{card key, scores}`), reported by BalatroBot's gamestate.
    pub example: &'static [(&'static str, bool)],
}

/// A `G.CHALLENGES` entry; `data` is the whole table (rules, restrictions, deck, jokers...).
#[derive(Debug)]
pub struct Challenge {
    pub id: &'static str,
    pub name: &'static str,
    pub data: Value,
}
