//! Observation encoder v1: a bit-for-bit port of the Python encoder `jackdaw/env/observation.py`
//! had (its `encode_observation` now calls this one).
//!
//! The output equals the Python encoder's on the same state (`tests/env/test_rust_observation.py`
//! checks it against that encoder's frozen outputs): the same arrays and shapes, the same
//! `float32` values, quirks included. Every feature is computed in `f64` the way Python computes
//! it (Python's `min`/`max` tie rules, float floor division, the platform libm's `log2` and
//! `log1p`, which CPython calls too) and rounded to `f32` once, as a numpy assignment does.
//!
//! Quirks kept on purpose (encoder v2 is where they get fixed):
//! - The flag features are dead. `global[29]` packs `gs["four_fingers"]`, `["shortcut"]`,
//!   `["smeared"]` and `["splash"]`, keys the Python engine never set, so it is always 0; for the
//!   same reason a playing card's `is_scoring` (row 9) is just "not debuffed".
//! - `global[26]` looks the blind key up among the center keys, where no blind is: always 0.
//! - The best-hand analysis runs `get_best_hand` on the whole face-up hand, so flushes and
//!   straights only show with five or fewer face-up cards, Python's Two Pair takes the best two
//!   of three or more pairs, and six or more cards of one rank make no group.
//! - A joker's `ability_extra` (row 13) sums a table's numbers in key order, a boolean counting
//!   as 0 or 1.
//! - Joker Stencil and Throwback encode `x_mult` 1 and Swashbuckler its config `mult`: the Python
//!   engine never ran `Card:update`, so its live jokers kept their `set_ability` values there
//!   (the Rust engine and the snapshot hold Lua's per-frame values).

mod analysis;

use jackdaw_core::area::Cards;
use jackdaw_core::card::{Card, EditionKind, Extra, ExtraVal, Facing, Rank, Seal, Suit};
use jackdaw_core::state::{BlindSlot, GameState, Phase};
use jackdaw_data::{CenterId, CenterSet, Value, centers};

use analysis::{Analyzer, Flags, HandAnalysis, py_max, py_min};

/// Features per playing card (hand and pack rows).
pub const D_PLAYING_CARD: usize = 15;
/// Features per joker.
pub const D_JOKER: usize = 15;
/// Features per consumable.
pub const D_CONSUMABLE: usize = 7;
/// Features per shop item.
pub const D_SHOP: usize = 9;
/// The global context's length.
pub const D_GLOBAL: usize = 235;
/// Distinct center keys (`centers.json`), the scale of the center-id features.
pub const NUM_CENTER_KEYS: usize = jackdaw_data::CENTER_KEYS.len();

/// The entity arrays, in buffer order.
pub const ENTITIES: [&str; 5] = [
    "hand_cards",
    "jokers",
    "consumables",
    "shop_cards",
    "pack_cards",
];
/// Each entity array's row width, in [`ENTITIES`] order.
pub const ENTITY_WIDTHS: [usize; 5] = [
    D_PLAYING_CARD,
    D_JOKER,
    D_CONSUMABLE,
    D_SHOP,
    D_PLAYING_CARD,
];

const NUM_HAND_TYPES: usize = 12;
const NUM_RANKS: usize = 13;

// Global layout offsets.
const G_HAND_LEVELS: usize = 30;
const G_VOUCHERS: usize = G_HAND_LEVELS + NUM_HAND_TYPES * 5; // 90
const G_BLIND: usize = G_VOUCHERS + VOUCHER_KEYS.len(); // 122
const G_ROUND_POSITION: usize = G_BLIND + 8; // 130
const G_ROUND_PROGRESS: usize = G_ROUND_POSITION + 3; // 133
const G_TAGS: usize = G_ROUND_PROGRESS + 2; // 135
const G_DISCARDS: usize = G_TAGS + TAG_KEYS.len(); // 159
const G_HAND_TYPES: usize = G_DISCARDS + 4 * NUM_RANKS; // 211
const G_BASE_SCORE: usize = G_HAND_TYPES + NUM_HAND_TYPES; // 223
const G_RATIO: usize = G_BASE_SCORE + 3; // 226
const G_DRAWS: usize = G_RATIO + 1; // 227
const G_INTEREST: usize = G_DRAWS + 2; // 229
const G_OUTS: usize = G_INTEREST + 1; // 230
const G_DEBUFFED: usize = G_OUTS + 2; // 232
const G_SPENDABLE: usize = G_DEBUFFED + 1; // 233
const G_URGENCY: usize = G_SPENDABLE + 1; // 234
const _: () = assert!(G_URGENCY + 1 == D_GLOBAL);

/// The voucher keys of the owned-vouchers block, in order (sorted).
const VOUCHER_KEYS: [&str; 32] = [
    "v_antimatter",
    "v_blank",
    "v_clearance_sale",
    "v_crystal_ball",
    "v_directors_cut",
    "v_glow_up",
    "v_grabber",
    "v_hieroglyph",
    "v_hone",
    "v_illusion",
    "v_liquidation",
    "v_magic_trick",
    "v_money_tree",
    "v_nacho_tong",
    "v_observatory",
    "v_omen_globe",
    "v_overstock_norm",
    "v_overstock_plus",
    "v_paint_brush",
    "v_palette",
    "v_petroglyph",
    "v_planet_merchant",
    "v_planet_tycoon",
    "v_recyclomancy",
    "v_reroll_glut",
    "v_reroll_surplus",
    "v_retcon",
    "v_seed_money",
    "v_tarot_merchant",
    "v_tarot_tycoon",
    "v_telescope",
    "v_wasteful",
];

/// The tag keys of the awarded-tags block, in order (sorted).
const TAG_KEYS: [&str; 24] = [
    "tag_boss",
    "tag_buffoon",
    "tag_charm",
    "tag_coupon",
    "tag_d_six",
    "tag_double",
    "tag_economy",
    "tag_ethereal",
    "tag_foil",
    "tag_garbage",
    "tag_handy",
    "tag_holo",
    "tag_investment",
    "tag_juggle",
    "tag_meteor",
    "tag_negative",
    "tag_orbital",
    "tag_polychrome",
    "tag_rare",
    "tag_skip",
    "tag_standard",
    "tag_top_up",
    "tag_uncommon",
    "tag_voucher",
];

/// `center_key_id(key)` by [`CenterId`]: the 1-based position of the key among the sorted
/// center keys.
static CENTER_KEY_IDS: [u16; NUM_CENTER_KEYS] = {
    let mut ids = [0u16; NUM_CENTER_KEYS];
    let mut i = 0;
    while i < NUM_CENTER_KEYS {
        ids[jackdaw_data::CENTER_KEYS[i].1.0 as usize] = (i + 1) as u16;
        i += 1;
    }
    ids
};

/// The position of each voucher's key in [`VOUCHER_KEYS`] by [`CenterId`] (`u8::MAX`: none), so
/// an observation finds a voucher's slot without a string search.
static VOUCHER_SLOTS: std::sync::LazyLock<[u8; NUM_CENTER_KEYS]> = std::sync::LazyLock::new(|| {
    let mut slots = [u8::MAX; NUM_CENTER_KEYS];
    for (id, c) in jackdaw_data::CENTERS.iter().enumerate() {
        if let Ok(i) = VOUCHER_KEYS.binary_search(&c.key) {
            slots[id] = i as u8;
        }
    }
    slots
});

/// The position of each tag's key in [`TAG_KEYS`] by `TagId` (`u8::MAX`: none).
static TAG_SLOTS: std::sync::LazyLock<[u8; NUM_TAGS]> = std::sync::LazyLock::new(|| {
    let mut slots = [u8::MAX; NUM_TAGS];
    for (id, t) in jackdaw_data::TAGS.iter().enumerate() {
        if let Ok(i) = TAG_KEYS.binary_search(&t.key) {
            slots[id] = i as u8;
        }
    }
    slots
});

const NUM_TAGS: usize = jackdaw_data::TAGS.len();

/// The blind-effect features of a blind's `debuff` config.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BlindDebuff {
    /// `debuff.suit`, when it names a suit.
    suit: Option<Suit>,
    /// `debuff.is_face == "face"`.
    face: bool,
}

impl BlindDebuff {
    fn of(debuff: &'static [(&'static str, Value)]) -> BlindDebuff {
        let suit = match jackdaw_data::lookup(debuff, "suit") {
            Some(Value::Str(s)) => Suit::from_name(s),
            _ => None,
        };
        let face = matches!(
            jackdaw_data::lookup(debuff, "is_face"),
            Some(Value::Str("face"))
        );
        BlindDebuff { suit, face }
    }
}

const NUM_BLINDS: usize = jackdaw_data::BLINDS.len();

/// [`BlindDebuff`] by `BlindId`, so an observation reads its blind's effect without key searches.
static BLIND_DEBUFFS: std::sync::LazyLock<[BlindDebuff; NUM_BLINDS]> =
    std::sync::LazyLock::new(|| {
        std::array::from_fn(|i| BlindDebuff::of(jackdaw_data::BLINDS[i].debuff))
    });

/// What the consumable features read from `card.ability["consumeable"]` (the center's config
/// for a consumable, else empty), by center.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ConsumableConfig {
    /// Python truthiness of `cfg.get("max_highlighted")`.
    max_highlighted_truthy: bool,
    /// Python truthiness of `cfg.get("hand_type")`.
    hand_type_truthy: bool,
    /// The numbers under `max_highlighted`, `min_highlighted` and `mod_num`.
    max_highlighted: Option<f64>,
    min_highlighted: Option<f64>,
    mod_num: Option<f64>,
}

impl ConsumableConfig {
    fn of(center: &jackdaw_data::Center) -> ConsumableConfig {
        let cfg: &'static [(&'static str, Value)] = if center.consumeable {
            center.config
        } else {
            &[]
        };
        ConsumableConfig {
            max_highlighted_truthy: config_truthy(cfg, "max_highlighted"),
            hand_type_truthy: config_truthy(cfg, "hand_type"),
            max_highlighted: config_num(cfg, "max_highlighted"),
            min_highlighted: config_num(cfg, "min_highlighted"),
            mod_num: config_num(cfg, "mod_num"),
        }
    }
}

/// [`ConsumableConfig`] by [`CenterId`].
static CONSUMABLE_CONFIGS: std::sync::LazyLock<Vec<ConsumableConfig>> =
    std::sync::LazyLock::new(|| {
        jackdaw_data::CENTERS
            .iter()
            .map(ConsumableConfig::of)
            .collect()
    });

/// The [`ConsumableConfig`] of `c`'s center.
fn consumable_config(c: &Card) -> &'static ConsumableConfig {
    &CONSUMABLE_CONFIGS[usize::from(c.center.0)]
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// An encoded observation: the global context and five entity arrays, row-major `f32`, back to
/// back in one buffer (global, hand, jokers, consumables, shop, pack).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Observation {
    data: Vec<f32>,
    rows: [usize; 5],
}

impl Observation {
    /// Every array back to back, in [`ENTITIES`] order after the global context.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The row count of each entity array, in [`ENTITIES`] order.
    pub fn rows(&self) -> [usize; 5] {
        self.rows
    }

    /// The global context (`D_GLOBAL` values).
    pub fn global(&self) -> &[f32] {
        &self.data[..D_GLOBAL]
    }

    /// Entity array `k` (an index into [`ENTITIES`]), `rows()[k] * ENTITY_WIDTHS[k]` values.
    pub fn entity(&self, k: usize) -> &[f32] {
        let start = D_GLOBAL
            + (0..k)
                .map(|j| self.rows[j] * ENTITY_WIDTHS[j])
                .sum::<usize>();
        &self.data[start..start + self.rows[k] * ENTITY_WIDTHS[k]]
    }

    pub fn hand_cards(&self) -> &[f32] {
        self.entity(0)
    }

    pub fn jokers(&self) -> &[f32] {
        self.entity(1)
    }

    pub fn consumables(&self) -> &[f32] {
        self.entity(2)
    }

    pub fn shop_cards(&self) -> &[f32] {
        self.entity(3)
    }

    pub fn pack_cards(&self) -> &[f32] {
        self.entity(4)
    }
}

/// Encodes game states into one reusable [`Observation`]; after the first few calls it no
/// longer allocates.
#[derive(Clone, Debug, Default)]
pub struct Encoder {
    obs: Observation,
    analyzer: Analyzer,
    memo: LogMemo,
}

impl Encoder {
    pub fn new() -> Encoder {
        Encoder::default()
    }

    /// `encode_observation(gs)`.
    pub fn encode(&mut self, gs: &GameState) -> &Observation {
        let areas = &gs.areas;
        let rows = [
            areas.hand().len(),
            areas.jokers().len(),
            areas.consumables().len(),
            areas.shop_jokers().len() + areas.shop_vouchers().len() + areas.shop_booster().len(),
            areas.pack_cards().len(),
        ];
        let total = D_GLOBAL
            + rows
                .iter()
                .zip(ENTITY_WIDTHS)
                .map(|(n, w)| n * w)
                .sum::<usize>();
        self.obs.rows = rows;
        self.obs.data.clear();
        self.obs.data.resize(total, 0.0);

        // The best-hand analysis of the live hand serves the global context and the hand rows.
        let flags = Flags::from_jokers(areas.jokers());
        let hand_analysis = self.analyzer.analyze(areas.hand(), flags);

        let log = LogScale::new();
        let (global, rest) = self.obs.data.split_at_mut(D_GLOBAL);
        encode_global(gs, &hand_analysis, log, &mut self.memo, global);
        let (hand, rest) = rest.split_at_mut(rows[0] * D_PLAYING_CARD);
        encode_playing_cards(areas.hand(), &self.analyzer.scoring, log, hand);
        let (jokers, rest) = rest.split_at_mut(rows[1] * D_JOKER);
        encode_jokers(areas.jokers(), log, jokers);
        let (consumables, rest) = rest.split_at_mut(rows[2] * D_CONSUMABLE);
        encode_consumables(gs, log, consumables);
        let (shop, pack) = rest.split_at_mut(rows[3] * D_SHOP);
        encode_shop(gs, log, shop);
        // Pack cards get their own analysis (only their scoring flags are encoded).
        if !areas.pack_cards().is_empty() {
            self.analyzer.analyze(areas.pack_cards(), flags);
            encode_playing_cards(areas.pack_cards(), &self.analyzer.scoring, log, pack);
        }
        &self.obs
    }

    /// [`Encoder::encode`] straight into a padded layout: the global context, then each entity
    /// array as its first `max_rows[k]` rows (zeros past the area's cards), then the five row
    /// counts (each capped at its `max_rows`). The values are the ones `encode` writes: every
    /// row depends on its card, its index and the game only, and both analyses still see every
    /// card. `out` holds exactly that many values.
    pub fn encode_padded(&mut self, gs: &GameState, max_rows: [usize; 5], out: &mut [f32]) {
        let areas = &gs.areas;
        let rows = [
            areas.hand().len(),
            areas.jokers().len(),
            areas.consumables().len(),
            areas.shop_jokers().len() + areas.shop_vouchers().len() + areas.shop_booster().len(),
            areas.pack_cards().len(),
        ];
        out.fill(0.0);
        let flags = Flags::from_jokers(areas.jokers());
        let hand_analysis = self.analyzer.analyze(areas.hand(), flags);

        let log = LogScale::new();
        let (global, rest) = out.split_at_mut(D_GLOBAL);
        encode_global(gs, &hand_analysis, log, &mut self.memo, global);
        let (hand, rest) = rest.split_at_mut(max_rows[0] * D_PLAYING_CARD);
        encode_playing_cards(areas.hand(), &self.analyzer.scoring, log, hand);
        let (jokers, rest) = rest.split_at_mut(max_rows[1] * D_JOKER);
        encode_jokers(areas.jokers(), log, jokers);
        let (consumables, rest) = rest.split_at_mut(max_rows[2] * D_CONSUMABLE);
        encode_consumables(gs, log, consumables);
        let (shop, rest) = rest.split_at_mut(max_rows[3] * D_SHOP);
        encode_shop(gs, log, shop);
        let (pack, counts) = rest.split_at_mut(max_rows[4] * D_PLAYING_CARD);
        if !areas.pack_cards().is_empty() {
            self.analyzer.analyze(areas.pack_cards(), flags);
            encode_playing_cards(areas.pack_cards(), &self.analyzer.scoring, log, pack);
        }
        for (k, slot) in counts.iter_mut().enumerate() {
            *slot = rows[k].min(max_rows[k]) as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// Python numerics
// ---------------------------------------------------------------------------

/// CPython's `math.log2` (`m_log2`): NaN and +inf pass through, finite inputs go to libm.
fn py_log2(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        x
    } else {
        x.log2()
    }
}

/// CPython's `math.log1p` (`m_log1p`): a zero passes through with its sign.
fn py_log1p(x: f64) -> f64 {
    if x == 0.0 { x } else { x.ln_1p() }
}

/// [`log_scale`] of the whole numbers below its length: most of an observation's scaled values
/// (hand levels, card counts, costs, times played) are small whole numbers.
static LOG_SCALE_INTS: std::sync::LazyLock<[f64; 1024]> =
    std::sync::LazyLock::new(|| std::array::from_fn(|i| py_log2(1.0 + i as f64)));

/// 2^52: adding it to a double in `[0, 1024)` rounds that to a whole number, exactly when it is
/// one, and leaves the whole number in the low mantissa bits.
const TWO_52: f64 = 4_503_599_627_370_496.0;

/// `_log_scale(x)` ([`LogScale::get`]).
#[cfg(test)]
fn log_scale(x: f64) -> f64 {
    LogScale::new().get(x)
}

/// `_log_scale`, with [`LOG_SCALE_INTS`] resolved once per observation.
#[derive(Clone, Copy)]
struct LogScale(&'static [f64; 1024]);

impl LogScale {
    fn new() -> LogScale {
        LogScale(&LOG_SCALE_INTS)
    }

    /// `_log_scale(x)`: `sign(x) * log2(1 + |x|)`, as Python branches it. A whole number in the
    /// table's range reads the value the same `py_log2` call computed (`-0.0` too: `1 + -0.0`
    /// is `1.0`, as for `0.0`); anything else, NaN included, is computed.
    #[inline]
    fn get(self, x: f64) -> f64 {
        if (0.0..1024.0).contains(&x) {
            // `x + 2^52` is exact iff `x` is whole (else it rounds to a whole number, which then
            // differs from `x`), and then its low mantissa bits are `x`: no float-int round trip.
            let y = x + TWO_52;
            if y - TWO_52 == x {
                return self.0[(y.to_bits() & 1023) as usize];
            }
        }
        log_scale_computed(x)
    }

    /// [`LogScale::get`] remembering, per slot, the last value computed off the table: a slot
    /// sees the same chip count, blind target or dollars for many steps in a row.
    #[inline]
    fn memo(self, memo: &mut LogMemo, slot: usize, x: f64) -> f64 {
        if (0.0..1024.0).contains(&x) {
            let y = x + TWO_52;
            if y - TWO_52 == x {
                return self.0[(y.to_bits() & 1023) as usize];
            }
        }
        let entry = &mut memo.0[slot];
        if entry.0 != x.to_bits() {
            *entry = (x.to_bits(), log_scale_computed(x));
        }
        debug_assert_eq!(entry.1.to_bits(), log_scale_computed(x).to_bits());
        entry.1
    }
}

/// `_log_scale(x)` computed, the value [`LOG_SCALE_INTS`] holds for a whole number.
fn log_scale_computed(x: f64) -> f64 {
    if x >= 0.0 {
        py_log2(1.0 + x)
    } else {
        -py_log2(1.0 - x)
    }
}

/// The memo slots of [`LogScale::memo`]: the global context's values that can leave the
/// whole-number table (chip counts, the best hand's base score, money).
const MEMO_DOLLARS: usize = 0;
const MEMO_BLIND_CHIPS: usize = 1;
const MEMO_CHIPS: usize = 2;
const MEMO_BASE_CHIPS: usize = 3;
const MEMO_BASE_MULT: usize = 4;
const MEMO_BASE_SCORE: usize = 5;
const MEMO_SPENDABLE: usize = 6;
/// Two per hand type: its chips, its mult.
const MEMO_HAND_LEVELS: usize = 7;
const NUM_MEMO: usize = MEMO_HAND_LEVELS + 2 * NUM_HAND_TYPES;

/// Per slot, the bits of the last input [`LogScale::memo`] computed and its `log_scale`. Every
/// entry starts as `(+0.0, log_scale(+0.0) = 0.0)`, a true pair.
#[derive(Clone, Debug)]
struct LogMemo([(u64, f64); NUM_MEMO]);

impl Default for LogMemo {
    fn default() -> LogMemo {
        LogMemo([(0, 0.0); NUM_MEMO])
    }
}

/// `_finite(x)`: an overflowed (or NaN) chip count becomes the largest double.
fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else if x < 0.0 {
        -f64::MAX
    } else {
        f64::MAX
    }
}

/// Python's `a // b` on numbers (`float_floor_div`); integers give the same result.
fn py_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let mut floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv += 1.0;
        }
        floordiv
    } else {
        0.0f64.copysign(a / b)
    }
}

fn flag(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Categorical tables
// ---------------------------------------------------------------------------

/// `center_key_id(key) / max(NUM_CENTER_KEYS, 1)`.
fn center_feature(center: CenterId) -> f64 {
    f64::from(CENTER_KEY_IDS[center.0 as usize]) / NUM_CENTER_KEYS as f64
}

/// `_SUIT_IDX`.
fn suit_index(s: Suit) -> usize {
    match s {
        Suit::Hearts => 0,
        Suit::Diamonds => 1,
        Suit::Clubs => 2,
        Suit::Spades => 3,
    }
}

/// `_RANK_IDX`.
fn rank_index(r: Rank) -> usize {
    r.id() as usize - 2
}

/// `_RANK_CHIPS`.
fn rank_chips(r: Rank) -> f64 {
    match r {
        Rank::Jack | Rank::Queen | Rank::King => 10.0,
        Rank::Ace => 11.0,
        other => f64::from(other.id()),
    }
}

/// `_ENHANCEMENT_IDX` by center (anything else is 0).
fn enhancement_index(center: CenterId) -> f64 {
    match center {
        centers::M_BONUS => 1.0,
        centers::M_MULT => 2.0,
        centers::M_WILD => 3.0,
        centers::M_GLASS => 4.0,
        centers::M_STEEL => 5.0,
        centers::M_STONE => 6.0,
        centers::M_GOLD => 7.0,
        centers::M_LUCKY => 8.0,
        _ => 0.0,
    }
}

/// `_edition_idx(card.edition)`.
fn edition_index(c: &Card) -> f64 {
    match c.edition.map(|e| e.kind) {
        None => 0.0,
        Some(EditionKind::Foil) => 1.0,
        Some(EditionKind::Holo) => 2.0,
        Some(EditionKind::Polychrome) => 3.0,
        Some(EditionKind::Negative) => 4.0,
    }
}

/// `_SEAL_IDX`.
fn seal_index(seal: Option<Seal>) -> f64 {
    match seal {
        None => 0.0,
        Some(Seal::Gold) => 1.0,
        Some(Seal::Red) => 2.0,
        Some(Seal::Blue) => 3.0,
        Some(Seal::Purple) => 4.0,
    }
}

/// `_CARD_SET_IDX` by `ability.set`.
fn set_index(set: CenterSet) -> f64 {
    match set {
        CenterSet::Default | CenterSet::None => 0.0,
        CenterSet::Enhanced => 1.0,
        CenterSet::Joker => 2.0,
        CenterSet::Tarot => 3.0,
        CenterSet::Planet => 4.0,
        CenterSet::Spectral => 5.0,
        CenterSet::Voucher => 6.0,
        CenterSet::Booster => 7.0,
        CenterSet::Back => 8.0,
        CenterSet::Edition => 9.0,
    }
}

fn phase_index(p: Phase) -> usize {
    match p {
        Phase::BlindSelect => 0,
        Phase::SelectingHand => 1,
        Phase::RoundEval => 2,
        Phase::Shop => 3,
        Phase::PackOpening => 4,
        Phase::GameOver => 5,
    }
}

// ---------------------------------------------------------------------------
// Global context
// ---------------------------------------------------------------------------

/// `encode_global_context(gs, hand_analysis)`.
fn encode_global(
    gs: &GameState,
    ha: &HandAnalysis,
    log: LogScale,
    memo: &mut LogMemo,
    v: &mut [f32],
) {
    let v: &mut [f32; D_GLOBAL] = v.try_into().expect("the global context");
    let mut set = |i: usize, x: f64| v[i] = x as f32;
    let cr = &gs.current_round;
    let areas = &gs.areas;

    set(phase_index(gs.phase), 1.0);
    set(6 + gs.blind_on_deck.map_or(0, |b| b.index() + 1), 1.0);

    // Scalars [10:30].
    let blind = gs.blind.as_ref();
    let blind_chips = finite(blind.map_or(0.0, |b| b.chips));
    let chips = finite(gs.chips);
    set(10, gs.round_resets.ante / 8.0);
    set(11, gs.round / 30.0);
    set(12, log.memo(memo, MEMO_DOLLARS, gs.dollars));
    set(13, cr.hands_left / 10.0);
    set(14, cr.discards_left / 10.0);
    set(15, gs.hand_size / 15.0);
    set(16, gs.joker_slots / 10.0);
    set(17, gs.consumable_slots / 5.0);
    set(18, log.memo(memo, MEMO_BLIND_CHIPS, blind_chips));
    set(19, log.memo(memo, MEMO_CHIPS, chips));
    set(20, py_min(chips / py_max(blind_chips, 1.0), 10.0) / 10.0);
    set(21, cr.reroll_cost / 10.0);
    set(22, py_min(cr.free_rerolls, 5.0) / 5.0);
    set(23, gs.interest_cap / 100.0);
    set(24, gs.discount_percent / 50.0);
    set(25, gs.skips / 10.0);
    // center_key_id(blind key): blinds aren't centers, so this is 0 for every blind (checked
    // by `center_key_ids_follow_sorted_keys`), without a key search per observation.
    set(26, 0.0);
    set(27, log.get(areas.deck().len() as f64));
    set(28, log.get(areas.discard().len() as f64));
    // [29]: the four_fingers/shortcut/smeared/splash game-state flags, never set: 0.

    // Hand levels [30:90].
    for (j, hs) in gs.hands.iter().enumerate() {
        let o = G_HAND_LEVELS + j * 5;
        set(o, hs.level / 20.0);
        set(o + 1, log.memo(memo, MEMO_HAND_LEVELS + 2 * j, hs.chips));
        set(o + 2, log.memo(memo, MEMO_HAND_LEVELS + 2 * j + 1, hs.mult));
        set(o + 3, log.get(hs.played));
        set(o + 4, if hs.visible { 1.0 } else { 0.0 });
    }

    // Vouchers owned [90:122].
    let voucher_slots = &*VOUCHER_SLOTS;
    for voucher in gs.sets.used_vouchers.iter() {
        let i = voucher_slots[voucher.0 as usize];
        if i != u8::MAX {
            set(G_VOUCHERS + usize::from(i), 1.0);
        }
    }

    // Blind effect [122:130].
    if let Some(b) = blind {
        set(G_BLIND, if b.boss { 1.0 } else { 0.0 });
        set(G_BLIND + 1, if b.disabled { 1.0 } else { 0.0 });
        set(G_BLIND + 2, b.mult / 4.0);
        let debuff = b
            .key
            .map_or(BlindDebuff::default(), |k| BLIND_DEBUFFS[usize::from(k.0)]);
        if let Some(suit) = debuff.suit {
            set(G_BLIND + 3 + suit_index(suit), 1.0);
        }
        if debuff.face {
            set(G_BLIND + 7, 1.0);
        }
    }

    // Round position [130:133]: the first blind that is current or up for selection.
    if let Some(slot) = BlindSlot::ALL
        .into_iter()
        .find(|&s| matches!(gs.round_resets.blind_state(s), "Current" | "Select"))
    {
        set(G_ROUND_POSITION + slot.index(), 1.0);
    }

    // Round progress [133:135].
    set(G_ROUND_PROGRESS, cr.hands_played / 10.0);
    set(G_ROUND_PROGRESS + 1, cr.discards_used / 10.0);

    // Awarded tags [135:159]: every tag awarded this run.
    let tag_slots = &*TAG_SLOTS;
    for tag in &gs.sim.tag_history {
        let i = tag_slots[tag.0 as usize];
        if i != u8::MAX {
            set(G_TAGS + usize::from(i), 1.0);
        }
    }

    // Discard pile [159:211]: which suit/rank pairs it holds.
    for c in areas.discard() {
        if let Some(b) = c.base {
            set(
                G_DISCARDS + suit_index(b.suit) * NUM_RANKS + rank_index(b.value),
                1.0,
            );
        }
    }

    // Hand types present among the face-up hand cards [211:223].
    for (j, &present) in ha.types.iter().enumerate() {
        if present {
            set(G_HAND_TYPES + j, 1.0);
        }
    }

    // Base score of the best hand at its level [223:226].
    let mut base_score = 0.0;
    if let Some(best) = ha.best {
        let base = &jackdaw_data::HANDS[best.index()];
        let level = gs.hand(best).level;
        let base_chips = py_max(0.0, base.s_chips + base.l_chips * (level - 1.0));
        let base_mult = py_max(1.0, base.s_mult + base.l_mult * (level - 1.0));
        base_score = base_chips * base_mult;
        set(G_BASE_SCORE, log.memo(memo, MEMO_BASE_CHIPS, base_chips));
        set(G_BASE_SCORE + 1, log.memo(memo, MEMO_BASE_MULT, base_mult));
        set(
            G_BASE_SCORE + 2,
            log.memo(memo, MEMO_BASE_SCORE, base_score),
        );
    }

    // Score-to-blind ratio [226].
    if base_score > 0.0 && blind_chips > 0.0 {
        set(
            G_RATIO,
            py_min(py_log1p(base_score) / py_log1p(blind_chips), 2.0) / 2.0,
        );
    }

    // Draw proximity [227:229].
    let draws = analysis::analyze_draws(areas.hand());
    set(G_DRAWS, draws.flush_proximity);
    set(G_DRAWS + 1, draws.straight_proximity);

    // Interest bracket [229].
    let dollars = gs.dollars;
    let max_interest = py_floordiv(gs.interest_cap, 5.0);
    let interest = py_min(py_floordiv(py_max(dollars, 0.0), 5.0), max_interest);
    set(G_INTEREST, interest / py_max(max_interest, 1.0));

    // Draw outs [230:232]: the deck's share of cards completing the flush or straight draw.
    let deck = areas.deck();
    if !deck.is_empty() {
        let n = deck.len() as f64;
        if let Some(suit) = draws.best_flush_suit
            && draws.best_flush_count < 5
        {
            let suited = deck
                .iter()
                .filter(|c| c.base.is_some_and(|b| b.suit == suit))
                .count();
            set(G_OUTS, suited as f64 / n);
        }
        if draws.straight_gap_ranks != 0 {
            let gaps = draws.straight_gap_ranks;
            let matching = deck
                .iter()
                .filter(|c| c.base.is_some_and(|b| gaps & (1 << b.id()) != 0))
                .count();
            set(G_OUTS + 1, matching as f64 / n);
        }
    }

    // Debuffed share of the hand [232]: face-up debuffed cards over all hand cards.
    let hand = areas.hand();
    if !hand.is_empty() {
        let debuffed = hand
            .iter()
            .filter(|c| c.facing != Facing::Back && c.debuff)
            .count();
        set(G_DEBUFFED, debuffed as f64 / hand.len() as f64);
    }

    // Money above the interest floor [233].
    let spendable = py_max(0.0, dollars - interest * 5.0);
    set(G_SPENDABLE, log.memo(memo, MEMO_SPENDABLE, spendable));

    // Urgency [234]: chips still needed over what the best hand scores in the hands left.
    let hands_left = cr.hands_left;
    let blind_remaining = py_max(blind_chips - chips, 0.0);
    if base_score > 0.0 && hands_left > 0.0 {
        let urgency = blind_remaining / (base_score * hands_left);
        set(G_URGENCY, py_min(urgency, 5.0) / 5.0);
    }
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

/// `encode_playing_cards_batch(cards, gs, analysis)`; `scoring` marks the best hand's cards.
fn encode_playing_cards(cards: Cards<'_>, scoring: &[bool], log: LogScale, out: &mut [f32]) {
    let rows = out.as_chunks_mut::<D_PLAYING_CARD>().0;
    for (i, (c, row)) in cards.iter().zip(rows).enumerate() {
        let position = (i as f64 / 20.0) as f32;
        let base = match c.base {
            Some(b) if c.facing != Facing::Back => b,
            // Face down (or no base): only that, and the position.
            _ => {
                row[7] = 1.0;
                row[12] = position;
                continue;
            }
        };
        row[0] = (f64::from(base.id()) / 14.0) as f32;
        row[1] = (suit_index(base.suit) as f64 / 3.0) as f32;
        row[2] = (rank_chips(base.value) / 11.0) as f32;
        row[3] = (enhancement_index(c.center) / 8.0) as f32;
        row[4] = (edition_index(c) / 4.0) as f32;
        row[5] = (seal_index(c.seal) / 4.0) as f32;
        row[6] = flag(c.debuff);
        row[8] = flag(matches!(base.id(), 11..=13));
        // `splash or not debuff`, and the splash flag is never set.
        row[9] = flag(!c.debuff);
        row[10] = log.get(c.ability.bonus + c.ability.perma_bonus) as f32;
        row[11] = log.get(base.times_played) as f32;
        row[12] = position;
        row[13] = flag(scoring[i]);
    }
}

/// `ability.mult` and `ability.x_mult` as the Python engine held them: it never ran
/// `Card:update`, so Joker Stencil's and Throwback's `x_mult` and Swashbuckler's `mult` kept their
/// `set_ability` values, while the Rust engine holds Lua's per-frame values.
fn python_mult_x_mult(c: &Card) -> (f64, f64) {
    let config = |key: &str, default: f64| c.center().config_num(key).unwrap_or(default);
    match c.center {
        centers::J_STENCIL | centers::J_THROWBACK => (c.ability.mult, config("Xmult", 1.0)),
        centers::J_SWASHBUCKLER => (config("mult", 0.0), c.ability.x_mult),
        _ => (c.ability.mult, c.ability.x_mult),
    }
}

/// Row 13 of a joker: `ability.extra` flattened to one number, if it is a number or a table.
fn extra_feature(extra: &Extra, log: LogScale) -> Option<f64> {
    match extra {
        Extra::Num(x) => Some(log.get(*x)),
        Extra::Table(t) => {
            // Python sums the dict's int, float and bool values in insertion order, which is
            // key order here (the center configs list their keys sorted).
            let mut total = 0.0;
            for (_, v) in t.entries() {
                match v {
                    ExtraVal::Num(x) => total += x,
                    ExtraVal::Bool(b) => total += if *b { 1.0 } else { 0.0 },
                    ExtraVal::Str(_) => {}
                }
            }
            Some(log.get(total))
        }
        Extra::Nil | Extra::Str(_) => None,
    }
}

/// `encode_jokers_batch(jokers)`.
fn encode_jokers(jokers: Cards<'_>, log: LogScale, out: &mut [f32]) {
    let rows = out.as_chunks_mut::<D_JOKER>().0;
    for (i, (c, row)) in jokers.iter().zip(rows).enumerate() {
        let position = (i as f64 / 20.0) as f32;
        if c.facing == Facing::Back {
            // Face down (The Amber Acorn): the position only.
            row[9] = position;
            continue;
        }
        let a = &c.ability;
        row[0] = center_feature(c.center) as f32;
        row[1] = if c.base_cost != 0.0 {
            py_min(c.base_cost / 20.0, 1.0) as f32
        } else {
            0.0
        };
        row[2] = (edition_index(c) / 4.0) as f32;
        row[3] = log.get(c.sell_cost) as f32;
        row[4] = flag(a.eternal);
        row[5] = flag(a.perishable);
        row[6] = if a.perishable {
            (a.perish_tally.unwrap_or(5.0) / 5.0) as f32
        } else {
            1.0
        };
        row[7] = flag(a.rental);
        row[8] = flag(c.debuff);
        row[9] = position;
        let (mult, x_mult) = python_mult_x_mult(c);
        row[10] = log.get(mult + a.t_mult) as f32;
        row[11] = x_mult as f32;
        row[12] = log.get(a.t_chips + a.bonus) as f32;
        if let Some(x) = extra_feature(&a.extra, log) {
            row[13] = x as f32;
        }
        row[14] = flag(!c.debuff);
    }
}

fn config_num(cfg: &'static [(&'static str, Value)], key: &str) -> Option<f64> {
    jackdaw_data::lookup(cfg, key).and_then(Value::as_num)
}

/// Python truthiness of `cfg.get(key)`.
fn config_truthy(cfg: &'static [(&'static str, Value)], key: &str) -> bool {
    match jackdaw_data::lookup(cfg, key) {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Num(x)) => *x != 0.0,
        Some(Value::Str(s)) => !s.is_empty(),
        Some(Value::List(items)) => !items.is_empty(),
        Some(Value::Table(entries)) => !entries.is_empty(),
    }
}

/// `consumable_block_reason(card, gs) is None`, for a card in the consumable area: usable now
/// with the hand selection `default_use_targets` would pick.
fn consumable_usable(card: &Card, gs: &GameState) -> bool {
    let cfg = consumable_config(card);
    // _selection_size: how many hand cards the use is made against.
    let size = if card.center == centers::C_AURA {
        Some(1.0)
    } else if cfg.max_highlighted_truthy {
        Some(cfg.min_highlighted.unwrap_or(1.0))
    } else {
        None
    };
    let Some(k) = size else {
        return can_use_consumable(card, gs, cfg, 0.0, false);
    };
    let hand = gs.areas.hand();
    if (hand.len() as f64) < k {
        return false;
    }
    // default_use_targets tries selections of k hand cards until one passes; only Aura's test
    // looks at which cards are picked (one card without an edition). Under Cerulean Bell the
    // forced card is always highlighted first (G137), so one card means that card.
    if k == 1.0 {
        // Only Aura's answer depends on the card: every other one is the same for each card,
        // and the hand isn't empty, so one call answers for all of them.
        if card.center != centers::C_AURA {
            return can_use_consumable(card, gs, cfg, 1.0, false);
        }
        if let Some(forced) = hand.iter().find(|c| c.ability.forced_selection) {
            return can_use_consumable(card, gs, cfg, 1.0, forced.edition.is_none());
        }
        hand.iter()
            .any(|c| can_use_consumable(card, gs, cfg, 1.0, c.edition.is_none()))
    } else {
        can_use_consumable(card, gs, cfg, k, false)
    }
}

/// `can_use_consumable(card, highlighted=...)` for a consumable in the consumable area, with
/// `n_selected` highlighted hand cards; `editionless` is whether the (single) highlighted card has
/// no edition.
fn can_use_consumable(
    card: &Card,
    gs: &GameState,
    cfg: &ConsumableConfig,
    n_selected: f64,
    editionless: bool,
) -> bool {
    use centers::*;
    let jokers = gs.areas.jokers();
    let key = card.center;
    if matches!(
        key,
        C_MERCURY
            | C_VENUS
            | C_EARTH
            | C_MARS
            | C_JUPITER
            | C_SATURN
            | C_URANUS
            | C_NEPTUNE
            | C_PLUTO
            | C_PLANET_X
            | C_CERES
            | C_ERIS
            | C_HERMIT
            | C_TEMPERANCE
            | C_BLACK_HOLE
    ) {
        return true;
    }
    if cfg.hand_type_truthy {
        return true;
    }
    if matches!(key, C_FOOL | C_EMPEROR | C_HIGH_PRIESTESS) {
        // `#G.consumeables.cards < card_limit or self.area == G.consumeables`
        // (card.lua:1550-1555): from the tray always, even over the limit (G142).
        if key == C_FOOL {
            return gs.last_tarot_planet.is_some_and(|k| k != C_FOOL);
        }
        return true;
    }
    if matches!(key, C_JUDGEMENT | C_SOUL | C_WRAITH) {
        return (jokers.len() as f64) < gs.joker_slots;
    }
    if matches!(key, C_WHEEL_OF_FORTUNE | C_ECTOPLASM | C_HEX) {
        return jokers
            .iter()
            .any(|j| j.set() == CenterSet::Joker && j.edition.is_none());
    }
    if key == C_ANKH {
        let n = jokers.len() as f64;
        return n > 0.0 && gs.joker_slots > 1.0 && n < gs.joker_slots;
    }
    if key == C_AURA {
        return n_selected == 1.0 && editionless;
    }
    if matches!(
        key,
        C_FAMILIAR | C_GRIM | C_INCANTATION | C_IMMOLATE | C_SIGIL | C_OUIJA
    ) {
        return gs.areas.hand().len() > 1;
    }
    if cfg.max_highlighted_truthy {
        let max_h = cfg.max_highlighted.unwrap_or(0.0);
        let min_h = cfg.min_highlighted.unwrap_or(1.0);
        let mod_num = cfg.mod_num.unwrap_or(max_h);
        return min_h <= n_selected && n_selected <= mod_num;
    }
    false
}

/// `encode_consumable(card, gs)` for each consumable.
fn encode_consumables(gs: &GameState, log: LogScale, out: &mut [f32]) {
    let rows = out.as_chunks_mut::<D_CONSUMABLE>().0;
    for (c, row) in gs.areas.consumables().iter().zip(rows) {
        row[0] = center_feature(c.center) as f32;
        row[1] = (set_index(c.set()) / 9.0) as f32;
        row[2] = log.get(c.sell_cost) as f32;
        row[3] = flag(consumable_usable(c, gs));
        let cfg = consumable_config(c);
        let max_h = cfg.max_highlighted.unwrap_or(0.0);
        let min_h = cfg.min_highlighted.unwrap_or(0.0);
        row[4] = flag(max_h > 0.0);
        row[5] = (py_min(max_h, 5.0) / 5.0) as f32;
        row[6] = (py_min(min_h, 5.0) / 5.0) as f32;
    }
}

/// `encode_shop_item(card, gs)` for the shop's jokers, then vouchers, then boosters.
fn encode_shop(gs: &GameState, log: LogScale, out: &mut [f32]) {
    let areas = &gs.areas;
    let items = areas
        .shop_jokers()
        .iter()
        .chain(areas.shop_vouchers())
        .chain(areas.shop_booster());
    let rows = out.as_chunks_mut::<D_SHOP>().0;
    for (c, row) in items.zip(rows) {
        let set = c.set();
        row[0] = center_feature(c.center) as f32;
        row[1] = (set_index(set) / 9.0) as f32;
        row[2] = log.get(c.cost) as f32;
        // can_afford: free, or within dollars - bankrupt_at.
        row[3] = flag(c.cost <= 0.0 || c.cost <= gs.dollars - gs.bankrupt_at);
        // check_for_buy_space: a Negative card raises the limit by one (G264).
        let bonus = if c.edition.is_some_and(|e| e.kind == EditionKind::Negative) {
            1.0
        } else {
            0.0
        };
        row[4] = flag(match set {
            CenterSet::Joker => (areas.jokers().len() as f64) < gs.joker_slots + bonus,
            CenterSet::Tarot | CenterSet::Planet | CenterSet::Spectral => {
                (areas.consumables().len() as f64) < gs.consumable_slots + bonus
            }
            _ => true,
        });
        row[5] = (edition_index(c) / 4.0) as f32;
        row[6] = flag(c.ability.eternal);
        row[7] = flag(c.ability.perishable);
        row[8] = flag(c.ability.rental);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_tables_are_sorted() {
        assert!(VOUCHER_KEYS.windows(2).all(|w| w[0] < w[1]));
        assert!(TAG_KEYS.windows(2).all(|w| w[0] < w[1]));
        for key in VOUCHER_KEYS {
            assert!(jackdaw_data::center_by_key(key).is_some(), "{key}");
        }
        for key in TAG_KEYS {
            assert!(jackdaw_data::tag_by_key(key).is_some(), "{key}");
        }
    }

    #[test]
    fn center_key_ids_follow_sorted_keys() {
        assert_eq!(NUM_CENTER_KEYS, 299);
        let mut keys: Vec<&str> = jackdaw_data::CENTERS.iter().map(|c| c.key).collect();
        keys.sort_unstable();
        for (i, key) in keys.iter().enumerate() {
            let id = jackdaw_data::center_by_key(key).unwrap();
            assert_eq!(CENTER_KEY_IDS[id.0 as usize] as usize, i + 1, "{key}");
        }
        // No blind is a center, so the blind feature is always 0.
        for b in jackdaw_data::BLINDS.iter() {
            assert!(jackdaw_data::center_by_key(b.key).is_none());
        }
    }

    #[test]
    fn extra_tables_list_keys_sorted() {
        // Python sums `extra` in insertion order; the Rust table iterates in key order. They
        // agree because every center config lists its extra keys sorted.
        for c in jackdaw_data::CENTERS.iter() {
            if let Some(Value::Table(entries)) = c.config("extra") {
                assert!(entries.windows(2).all(|w| w[0].0 < w[1].0), "{}", c.key);
            }
        }
    }

    #[test]
    fn log_scale_reads_the_table_exactly_for_whole_numbers() {
        let log = LogScale::new();
        let mut xs = vec![
            -0.0,
            0.5,
            0.499_999_999_999_999_94,
            1023.0,
            1023.5,
            1_023.999_999_999_999_9,
            1024.0,
            1024.5,
            TWO_52,
            TWO_52 + 1.0,
            1e300,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            -0.5,
            -1023.0,
            -5e-324,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
        ];
        xs.extend((0..1100).map(f64::from));
        xs.extend((0..4400).map(|i| f64::from(i) * 0.25 - 50.0));
        for x in xs {
            let want = log_scale_computed(x);
            assert_eq!(log.get(x).to_bits(), want.to_bits(), "{x}");
            if (0.0..1024.0).contains(&x) && x.fract() == 0.0 {
                assert_eq!(LOG_SCALE_INTS[x as usize].to_bits(), want.to_bits(), "{x}");
            }
        }
        // The memo's starting entries are true pairs.
        assert_eq!(log_scale_computed(0.0).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn log_memo_returns_the_computed_value() {
        let log = LogScale::new();
        let mut memo = LogMemo::default();
        let xs = [
            0.0,
            1500.0,
            1500.0,
            0.5,
            1e9,
            1e9,
            -3.25,
            1500.0,
            f64::NAN,
            f64::NAN,
            -0.0,
            7.0,
        ];
        for (k, &x) in xs.iter().enumerate() {
            for slot in [k % NUM_MEMO, NUM_MEMO - 1, 0] {
                let got = log.memo(&mut memo, slot, x);
                assert_eq!(got.to_bits(), log_scale_computed(x).to_bits(), "{x}");
            }
        }
    }

    /// The game states along random rollouts, one per step.
    fn rollout_states(steps: usize) -> Vec<GameState> {
        use crate::env::{EnvConfig, GymEnv};
        use crate::gym::Xoshiro256;
        let spec = |i: u64| jackdaw_core::run::RunSpec {
            back: jackdaw_core::card::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: format!("OBSCACHE{i}"),
            challenge: None,
        };
        let mut env = GymEnv::new(&spec(0), EnvConfig::default(), 1).unwrap();
        let mut rng = Xoshiro256::seed_from_u64(11);
        let mut episodes = 0;
        let mut states = Vec::with_capacity(steps);
        while states.len() < steps {
            if env.ended() {
                episodes += 1;
                env.reset(&spec(episodes), None).unwrap();
            }
            let n = env.table().len() as u64;
            env.step_or_cut(rng.below(n) as usize).unwrap();
            states.push(env.game().gs.clone());
        }
        states
    }

    /// `gs` with its chip counts, money and hand levels off the whole-number table.
    fn with_big_values(gs: &GameState) -> GameState {
        let mut big = gs.clone();
        big.chips = gs.chips * 1234.5 + 2048.0;
        big.dollars = gs.dollars * 97.0 + 1500.25;
        if let Some(b) = big.blind.as_mut() {
            b.chips = b.chips * 7.0 + 0.5;
        }
        for h in &mut big.hands {
            h.chips += 2000.5;
            h.mult *= 1100.0;
        }
        big
    }

    /// The bits of `gs`'s padded observation from `encoder`.
    fn padded_bits(encoder: &mut Encoder, gs: &GameState) -> Vec<u32> {
        use crate::env::{GYM_OBS_LEN, MAX_COUNTS};
        let mut out = vec![0f32; GYM_OBS_LEN];
        encoder.encode_padded(gs, MAX_COUNTS, &mut out);
        out.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn a_reused_encoder_matches_a_fresh_one() {
        // The encoder's memos carry from one observation to the next; each observation must
        // still equal a fresh encoder's, bit for bit.
        let mut warm = Encoder::new();
        let mut warm_full = Encoder::new();
        for gs in rollout_states(3000) {
            let big = with_big_values(&gs);
            for s in [&gs, &big, &gs, &big] {
                assert_eq!(
                    padded_bits(&mut warm, s),
                    padded_bits(&mut Encoder::new(), s)
                );
                let fresh = Encoder::new().encode(s).clone();
                let got = warm_full.encode(s);
                assert_eq!(got.rows(), fresh.rows());
                let bits =
                    |o: &Observation| o.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(got), bits(&fresh));
            }
        }
    }

    #[test]
    fn analysis_matches_the_reference_on_rollout_states() {
        let mut fast = Analyzer::default();
        for gs in rollout_states(3000) {
            let flags = Flags::from_jokers(gs.areas.jokers());
            for cards in [gs.areas.hand(), gs.areas.pack_cards(), gs.areas.deck()] {
                let mut slow = analysis::reference::RefAnalyzer::default();
                assert_eq!(fast.analyze(cards, flags), slow.analyze(cards, flags));
                assert_eq!(fast.scoring, slow.scoring);
                assert_eq!(
                    analysis::analyze_draws(cards),
                    analysis::reference::analyze_draws(cards)
                );
            }
        }
    }

    #[test]
    fn blind_and_consumable_tables_hold_the_configs() {
        let debuff = |key: &str| {
            let id = jackdaw_data::BLINDS
                .iter()
                .position(|b| b.key == key)
                .unwrap();
            BLIND_DEBUFFS[id]
        };
        let suit = |s| BlindDebuff {
            suit: Some(s),
            face: false,
        };
        assert_eq!(debuff("bl_club"), suit(Suit::Clubs));
        assert_eq!(debuff("bl_goad"), suit(Suit::Spades));
        assert_eq!(debuff("bl_head"), suit(Suit::Hearts));
        assert_eq!(debuff("bl_window"), suit(Suit::Diamonds));
        assert!(debuff("bl_plant").face);
        let marked = BLIND_DEBUFFS
            .iter()
            .filter(|d| **d != BlindDebuff::default())
            .count();
        assert_eq!(marked, 5);

        let cfg =
            |key: &str| CONSUMABLE_CONFIGS[jackdaw_data::center_by_key(key).unwrap().0 as usize];
        let magician = cfg("c_magician");
        assert!(magician.max_highlighted_truthy && !magician.hand_type_truthy);
        assert_eq!(magician.max_highlighted, Some(2.0));
        assert!(cfg("c_mercury").hand_type_truthy);
        // Not a consumable: its config is not read.
        assert_eq!(cfg("j_joker"), ConsumableConfig::default());
        assert_eq!(cfg("m_wild"), ConsumableConfig::default());
    }

    #[test]
    fn python_numerics() {
        assert_eq!(py_floordiv(23.0, 5.0), 4.0);
        assert_eq!(py_floordiv(-1.0, 5.0), -1.0);
        assert_eq!(py_floordiv(0.0, 5.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(py_floordiv(12.5, 5.0), 2.0);
        assert_eq!(log_scale(0.0), 0.0);
        assert_eq!(log_scale(3.0), 2.0);
        assert_eq!(log_scale(-3.0), -2.0);
        assert!(log_scale(f64::NAN).is_nan());
        assert_eq!(finite(f64::NAN), f64::MAX);
        assert_eq!(finite(f64::NEG_INFINITY), -f64::MAX);
        assert_eq!(py_min(f64::NAN, 1.0).to_bits(), f64::NAN.to_bits());
    }
}
