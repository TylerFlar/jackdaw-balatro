//! The hand analysis encoder v1 runs over an area's face-up cards, with the Python encoder's
//! semantics: `_compute_hand_analysis` (the Python engine's `get_hand_eval_flags` and
//! `get_best_hand`) and `_analyze_draws`.
//!
//! This is not [`jackdaw_core::hand_eval`]: that one follows Lua for played hands of at most five
//! cards, while the encoder evaluates a whole hand, where the two disagree. Python's Two Pair
//! takes `len(_2) >= 2` pairs (Lua: exactly two), and its downgrades copy every group instead of
//! the first few. Groups are sets of cards here: a rank's group is every card of that rank, a
//! flush is every card of its suit, a straight every card whose rank is in the run, so the scoring
//! cards are kept as rank and suit masks.

use jackdaw_core::area::Cards;
use jackdaw_core::card::{Card, Facing, Suit};
use jackdaw_core::hand_eval::HandType;
use jackdaw_data::centers;

/// The detection flags `get_hand_eval_flags` reads from the jokers (non-debuffed only). Splash
/// and Pareidolia don't change detection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Flags {
    pub four_fingers: bool,
    pub shortcut: bool,
    pub smeared: bool,
}

impl Flags {
    pub fn from_jokers(jokers: Cards<'_>) -> Flags {
        let mut f = Flags::default();
        for j in jokers.iter().filter(|j| !j.debuff) {
            match j.center {
                centers::J_FOUR_FINGERS => f.four_fingers = true,
                centers::J_SHORTCUT => f.shortcut = true,
                centers::J_SMEARED => f.smeared = true,
                _ => {}
            }
        }
        f
    }
}

/// What the analysis reads from one face-up card.
#[derive(Clone, Copy, Debug)]
struct Item {
    /// Position in the area (face-down cards included).
    pos: usize,
    /// Python's `card.get_id()` where it can join a group or straight (2-14); 0 for a Stone card
    /// (Python -1) or a card without a base (Python 0), which never do.
    id: u8,
    /// `card.is_suit(s, flush_calc=True, smeared)` for Spades, Hearts, Clubs, Diamonds, as the
    /// bits of [`Suit::flush_bit`].
    flush_bits: u8,
}

/// [`Card::flush_suits`] with the Wild Card test on the center id (`m_wild` is the only center
/// named "Wild Card"; `flush_bits_are_flush_suits` checks both).
fn flush_bits(c: &Card, smeared: bool) -> u8 {
    let Some(base) = c.base.as_ref() else {
        return 0;
    };
    if c.is_stone() {
        return 0;
    }
    if !c.debuff && c.center == centers::M_WILD {
        return 0b1111;
    }
    let table = if smeared { &SMEARED_BITS } else { &SUIT_BITS };
    table[base.suit as usize]
}

/// [`Suit::flush_bit`] by `suit as usize` (Hearts, Diamonds, Clubs, Spades), a load instead
/// of a jump per card.
const SUIT_BITS: [u8; 4] = [2, 8, 4, 1];
/// The flush bits of a suit's colour (Smeared Joker), by `suit as usize`.
const SMEARED_BITS: [u8; 4] = [2 | 8, 2 | 8, 4 | 1, 4 | 1];

/// The highest rank id among `mask`'s bits (0 for none).
fn highest(mask: u16) -> u8 {
    if mask == 0 {
        0
    } else {
        15 - mask.leading_zeros() as u8
    }
}

/// The hand types present and the best one (`None` for no face-up cards, Python's `""`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HandAnalysis {
    /// `full_results[hand]` is non-empty, by [`HandType::index`].
    pub types: [bool; 12],
    pub best: Option<HandType>,
}

/// `_analyze_draws(hand)` over the face-up cards.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Draws {
    pub flush_proximity: f64,
    pub straight_proximity: f64,
    /// The suit with the most cards (first seen on a tie), `None` for Python's `""`.
    pub best_flush_suit: Option<Suit>,
    pub best_flush_count: usize,
    /// The ranks that would extend the best run, as bits by rank id.
    pub straight_gap_ranks: u16,
}

/// Flush suits in `get_flush` order (Spades, Hearts, Clubs, Diamonds), as flush bits.
const FLUSH_BITS: [u8; 4] = [1, 2, 4, 8];

/// Scratch space for the analysis, reused across calls.
#[derive(Clone, Debug, Default)]
pub(crate) struct Analyzer {
    items: Vec<Item>,
    /// After [`Analyzer::analyze`]: whether each card of the area (by position) is in the best
    /// hand (`id(card) in scoring_ids`).
    pub scoring: Vec<bool>,
}

impl Analyzer {
    /// `_compute_hand_analysis(_face_up(cards), jokers, ...)`: the hand types present among the
    /// face-up cards and the best one; marks the best hand's cards in [`Analyzer::scoring`].
    #[cfg(test)]
    pub fn analyze(&mut self, cards: Cards<'_>, flags: Flags) -> HandAnalysis {
        self.analyze_with(cards, flags, false)
    }

    /// [`Analyzer::analyze`]; with `hide_stone_bases` (the player's view) a Stone card's
    /// nominal is 0 when High Card picks its highest card, so among Stones alone the first one
    /// is marked, whatever their hidden bases. (Any other card still outranks a Stone.)
    pub fn analyze_with(
        &mut self,
        cards: Cards<'_>,
        flags: Flags,
        hide_stone_bases: bool,
    ) -> HandAnalysis {
        self.items.clear();
        let mut count = [0usize; 15];
        for (pos, c) in cards.iter().enumerate() {
            if c.facing == Facing::Back {
                continue;
            }
            let id = match c.base {
                Some(b) if !c.is_stone() => b.id(),
                _ => 0,
            };
            count[usize::from(id)] += 1;
            self.items.push(Item {
                pos,
                id,
                flush_bits: flush_bits(c, flags.smeared),
            });
        }
        self.scoring.clear();
        self.scoring.resize(cards.len(), false);
        if self.items.is_empty() {
            return HandAnalysis::default();
        }
        let items = &self.items;
        let n = items.len();

        // get_X_same: every card of a rank, for the ranks with exactly `num` cards, highest
        // first. As rank masks: the count of ranks with `num` cards, and the highest of them
        // (the two highest for pairs). Id 0 (no rank) never groups.
        let (mut m5, mut m4, mut m3, mut m2) = (0u16, 0u16, 0u16, 0u16);
        for (id, &k) in count.iter().enumerate().skip(1) {
            let bit = 1u16 << id;
            m5 |= if k == 5 { bit } else { 0 };
            m4 |= if k == 4 { bit } else { 0 };
            m3 |= if k == 3 { bit } else { 0 };
            m2 |= if k == 2 { bit } else { 0 };
        }
        let (n5, n4, n3, n2) = (
            m5.count_ones(),
            m4.count_ones(),
            m3.count_ones(),
            m2.count_ones(),
        );
        let (r5, r4, r3, r2a) = (highest(m5), highest(m4), highest(m3), highest(m2));
        let r2b = highest(m2 & !(1u16 << r2a));

        // get_flush and get_straight take 4 (Four Fingers) or 5 cards, never more than five.
        let need = if flags.four_fingers { 4 } else { 5 };
        let sized = n >= need && n <= 5;
        let flush_bit = if sized {
            FLUSH_BITS
                .into_iter()
                .find(|&bit| items.iter().filter(|it| it.flush_bits & bit != 0).count() >= need)
                .unwrap_or(0)
        } else {
            0
        };
        let straight_ranks = if sized {
            let present = items
                .iter()
                .filter(|it| it.id > 1)
                .fold(0u16, |acc, it| acc | (1 << it.id));
            get_straight(present, need, flags.shortcut)
        } else {
            0
        };
        let flush = flush_bit != 0;
        let straight = straight_ranks != 0;
        let mut t = [false; 12];
        use HandType::*;
        t[FlushFive.index()] = n5 > 0 && flush;
        t[FlushHouse.index()] = n3 > 0 && n2 > 0 && flush;
        t[FiveOfAKind.index()] = n5 > 0;
        t[StraightFlush.index()] = flush && straight;
        // The downgrades: Five of a Kind fills Four, Four fills Three, Three fills Pair.
        t[FourOfAKind.index()] = n4 > 0 || n5 > 0;
        t[FullHouse.index()] = n3 > 0 && n2 > 0;
        t[Flush.index()] = flush;
        t[Straight.index()] = straight;
        t[ThreeOfAKind.index()] = n3 > 0 || t[FourOfAKind.index()];
        t[TwoPair.index()] = n2 >= 2 || (n3 == 1 && n2 == 1);
        t[Pair.index()] = n2 > 0 || t[ThreeOfAKind.index()];
        t[HighCard.index()] = true;
        let best = HandType::ALL.into_iter().find(|h| t[h.index()]);

        // The best hand's first group, as rank and suit masks (or the highest card).
        let rank = |r: u8| 1u16 << r;
        let (ranks, suit, high) = match best.expect("a face-up card makes a High Card") {
            FlushFive | FiveOfAKind => (rank(r5), 0, None),
            FlushHouse | FullHouse => (rank(r3) | rank(r2a), 0, None),
            StraightFlush => (straight_ranks, flush_bit, None),
            // Five of a Kind outranks it, so its group is the four of a kind.
            FourOfAKind => (rank(r4), 0, None),
            Flush => (0, flush_bit, None),
            Straight => (straight_ranks, 0, None),
            ThreeOfAKind => (rank(r3), 0, None),
            TwoPair if n2 >= 2 => (rank(r2a) | rank(r2b), 0, None),
            TwoPair => (rank(r2a) | rank(r3), 0, None),
            Pair => (rank(r2a), 0, None),
            // get_highest: the first card with the highest nominal (`_card_nominal(card)`:
            // `card.get_nominal()`, or 0.0 without a base).
            HighCard => {
                let nominal = |it: &Item| {
                    let c = &cards[it.pos];
                    if hide_stone_bases && c.is_stone() {
                        0.0
                    } else if c.base.is_some() {
                        c.get_nominal(false)
                    } else {
                        0.0
                    }
                };
                let mut highest = 0;
                let mut top = nominal(&items[0]);
                for (i, it) in items.iter().enumerate().skip(1) {
                    let x = nominal(it);
                    if x > top {
                        highest = i;
                        top = x;
                    }
                }
                (0, 0, Some(highest))
            }
        };
        for (i, it) in items.iter().enumerate() {
            let in_ranks = it.id != 0 && ranks & (1 << it.id) != 0;
            if in_ranks || it.flush_bits & suit != 0 || high == Some(i) {
                self.scoring[it.pos] = true;
            }
        }
        HandAnalysis { types: t, best }
    }
}

/// `get_straight`: the ranks of the cards it collects (bits by rank id), or 0 for no straight.
/// `present` has a bit for each rank id 2-14 among the cards; an Ace counts low (`j == 1`) and
/// high.
fn get_straight(present: u16, need: usize, shortcut: bool) -> u16 {
    let mut t = 0u16;
    let mut length = 0;
    let mut straight = false;
    let mut skipped_rank = false;
    for j in 1..=14u16 {
        let id = if j == 1 { 14 } else { j };
        if present & (1 << id) != 0 {
            length += 1;
            skipped_rank = false;
            t |= 1 << id;
        } else if shortcut && !skipped_rank && j != 14 {
            skipped_rank = true;
        } else {
            length = 0;
            skipped_rank = false;
            if !straight {
                t = 0;
            }
            if straight {
                break;
            }
        }
        if length >= need {
            straight = true;
        }
    }
    if straight { t } else { 0 }
}

/// `_analyze_draws(_face_up(cards))`: raw base suits and ranks, Stone cards included unless
/// `skip_stones` (the player's view: a Stone shows no rank or suit).
pub(crate) fn analyze_draws(cards: Cards<'_>, skip_stones: bool) -> Draws {
    let mut seen: [Suit; 4] = [Suit::Hearts; 4];
    let mut n_seen = 0;
    let mut counts = [0usize; 4];
    let mut ranks = 0u16;
    let mut any = false;
    for c in cards.iter().filter(|c| c.facing != Facing::Back) {
        any = true;
        let Some(b) = c.base else { continue };
        if skip_stones && c.is_stone() {
            continue;
        }
        let s = b.suit as usize;
        if counts[s] == 0 {
            seen[n_seen] = b.suit;
            n_seen += 1;
        }
        counts[s] += 1;
        ranks |= 1 << b.id();
    }
    if !any {
        return Draws::default();
    }
    // max(suit_counts, key=suit_counts.get): the first suit seen with the most cards.
    let mut best_suit = None;
    let mut best_count = 0;
    for &s in &seen[..n_seen] {
        if best_suit.is_none() || counts[s as usize] > best_count {
            best_suit = Some(s);
            best_count = counts[s as usize];
        }
    }
    let flush_proximity = py_min(best_count as f64 / 5.0, 1.0);

    if ranks & (1 << 14) != 0 {
        ranks |= 1 << 1; // Ace low
    }
    // The longest run of consecutive ranks; the lowest one on a tie. Each run of set bits,
    // lowest first, replaces the best only when strictly longer.
    let (mut best_run, mut best_start) = (0i32, 0i32);
    let mut rest = u32::from(ranks);
    while rest != 0 {
        let start = rest.trailing_zeros();
        let len = (!(rest >> start)).trailing_zeros();
        if len as i32 > best_run {
            best_run = len as i32;
            best_start = start as i32;
        }
        rest &= !(((1u32 << len) - 1) << start);
    }
    let straight_proximity = py_min(f64::from(best_run) / 5.0, 1.0);

    let mut gaps = 0u16;
    if best_run >= 3 {
        let run_end = best_start + best_run - 1;
        let mut push = |r: i32| {
            if (2..=14).contains(&r) && ranks & (1 << r) == 0 {
                gaps |= 1 << r;
            }
        };
        // range(best_start - 1, max(best_start - 3, 0), -1)
        let mut r = best_start - 1;
        while r > (best_start - 3).max(0) {
            push(r);
            r -= 1;
        }
        // range(run_end + 1, min(run_end + 3, 15))
        for r in run_end + 1..(run_end + 3).min(15) {
            push(r);
        }
    }
    Draws {
        flush_proximity,
        straight_proximity,
        best_flush_suit: best_suit,
        best_flush_count: best_count,
        straight_gap_ranks: gaps,
    }
}

/// Python's `min(a, b)`: `a` unless `b < a`.
pub(crate) fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Python's `max(a, b)`: `a` unless `b > a`.
pub(crate) fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// The analysis as first written (one pass per feature, every nominal computed, the rank and
/// run scans as Python loops them): the reference the faster one is checked against.
#[cfg(test)]
pub(crate) mod reference {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct RefItem {
        pos: usize,
        id: u8,
        flush_bits: u8,
        nominal: f64,
    }

    #[derive(Default)]
    pub(crate) struct RefAnalyzer {
        items: Vec<RefItem>,
        pub scoring: Vec<bool>,
    }

    impl RefAnalyzer {
        pub fn analyze(&mut self, cards: Cards<'_>, flags: Flags) -> HandAnalysis {
            self.items.clear();
            for (pos, c) in cards.iter().enumerate() {
                if c.facing == Facing::Back {
                    continue;
                }
                let id = match c.base {
                    Some(b) if !c.is_stone() => b.id(),
                    _ => 0,
                };
                self.items.push(RefItem {
                    pos,
                    id,
                    flush_bits: c.flush_suits(flags.smeared),
                    nominal: if c.base.is_some() {
                        c.get_nominal(false)
                    } else {
                        0.0
                    },
                });
            }
            self.scoring.clear();
            self.scoring.resize(cards.len(), false);
            if self.items.is_empty() {
                return HandAnalysis::default();
            }
            let items = &self.items;
            let n = items.len();

            // get_X_same: every card of a rank, for the ranks with exactly `num` cards, highest first.
            let mut count = [0usize; 15];
            for it in items {
                if it.id != 0 {
                    count[it.id as usize] += 1;
                }
            }
            let (mut n5, mut r5, mut n4, mut r4, mut n3, mut r3) = (0, 0u8, 0, 0u8, 0, 0u8);
            let (mut n2, mut r2a, mut r2b) = (0, 0u8, 0u8);
            for id in (1..=14u8).rev() {
                match count[id as usize] {
                    5 => {
                        if n5 == 0 {
                            r5 = id;
                        }
                        n5 += 1;
                    }
                    4 => {
                        if n4 == 0 {
                            r4 = id;
                        }
                        n4 += 1;
                    }
                    3 => {
                        if n3 == 0 {
                            r3 = id;
                        }
                        n3 += 1;
                    }
                    2 => {
                        match n2 {
                            0 => r2a = id,
                            1 => r2b = id,
                            _ => {}
                        }
                        n2 += 1;
                    }
                    _ => {}
                }
            }

            // get_flush and get_straight take 4 (Four Fingers) or 5 cards, never more than five.
            let need = if flags.four_fingers { 4 } else { 5 };
            let sized = n >= need && n <= 5;
            let flush_bit = if sized {
                FLUSH_BITS
                    .into_iter()
                    .find(|&bit| items.iter().filter(|it| it.flush_bits & bit != 0).count() >= need)
                    .unwrap_or(0)
            } else {
                0
            };
            let straight_ranks = if sized {
                let present = items
                    .iter()
                    .filter(|it| it.id > 1)
                    .fold(0u16, |acc, it| acc | (1 << it.id));
                get_straight(present, need, flags.shortcut)
            } else {
                0
            };
            // get_highest: the first card with the highest nominal.
            let mut highest = 0;
            for (i, it) in items.iter().enumerate() {
                if it.nominal > items[highest].nominal {
                    highest = i;
                }
            }

            let flush = flush_bit != 0;
            let straight = straight_ranks != 0;
            let mut t = [false; 12];
            use HandType::*;
            t[FlushFive.index()] = n5 > 0 && flush;
            t[FlushHouse.index()] = n3 > 0 && n2 > 0 && flush;
            t[FiveOfAKind.index()] = n5 > 0;
            t[StraightFlush.index()] = flush && straight;
            // The downgrades: Five of a Kind fills Four, Four fills Three, Three fills Pair.
            t[FourOfAKind.index()] = n4 > 0 || n5 > 0;
            t[FullHouse.index()] = n3 > 0 && n2 > 0;
            t[Flush.index()] = flush;
            t[Straight.index()] = straight;
            t[ThreeOfAKind.index()] = n3 > 0 || t[FourOfAKind.index()];
            t[TwoPair.index()] = n2 >= 2 || (n3 == 1 && n2 == 1);
            t[Pair.index()] = n2 > 0 || t[ThreeOfAKind.index()];
            t[HighCard.index()] = true;
            let best = HandType::ALL.into_iter().find(|h| t[h.index()]);

            // The best hand's first group, as rank and suit masks (or the highest card).
            let rank = |r: u8| 1u16 << r;
            let (ranks, suit, high) = match best.expect("a face-up card makes a High Card") {
                FlushFive | FiveOfAKind => (rank(r5), 0, None),
                FlushHouse | FullHouse => (rank(r3) | rank(r2a), 0, None),
                StraightFlush => (straight_ranks, flush_bit, None),
                // Five of a Kind outranks it, so its group is the four of a kind.
                FourOfAKind => (rank(r4), 0, None),
                Flush => (0, flush_bit, None),
                Straight => (straight_ranks, 0, None),
                ThreeOfAKind => (rank(r3), 0, None),
                TwoPair if n2 >= 2 => (rank(r2a) | rank(r2b), 0, None),
                TwoPair => (rank(r2a) | rank(r3), 0, None),
                Pair => (rank(r2a), 0, None),
                HighCard => (0, 0, Some(highest)),
            };
            for (i, it) in items.iter().enumerate() {
                let in_ranks = it.id != 0 && ranks & (1 << it.id) != 0;
                if in_ranks || it.flush_bits & suit != 0 || high == Some(i) {
                    self.scoring[it.pos] = true;
                }
            }
            HandAnalysis { types: t, best }
        }
    }

    pub(crate) fn analyze_draws(cards: Cards<'_>) -> Draws {
        let mut seen: [Suit; 4] = [Suit::Hearts; 4];
        let mut n_seen = 0;
        let mut counts = [0usize; 4];
        let mut ranks = 0u16;
        let mut any = false;
        for c in cards.iter().filter(|c| c.facing != Facing::Back) {
            any = true;
            let Some(b) = c.base else { continue };
            let s = b.suit as usize;
            if counts[s] == 0 {
                seen[n_seen] = b.suit;
                n_seen += 1;
            }
            counts[s] += 1;
            ranks |= 1 << b.id();
        }
        if !any {
            return Draws::default();
        }
        // max(suit_counts, key=suit_counts.get): the first suit seen with the most cards.
        let mut best_suit = None;
        let mut best_count = 0;
        for &s in &seen[..n_seen] {
            if best_suit.is_none() || counts[s as usize] > best_count {
                best_suit = Some(s);
                best_count = counts[s as usize];
            }
        }
        let flush_proximity = py_min(best_count as f64 / 5.0, 1.0);

        if ranks & (1 << 14) != 0 {
            ranks |= 1 << 1; // Ace low
        }
        // The longest run of consecutive ranks; the lowest one on a tie.
        let (mut best_run, mut best_start) = (0i32, 0i32);
        let (mut run, mut start) = (0i32, 0i32);
        let mut prev: Option<i32> = None;
        for r in 1..=14i32 {
            if ranks & (1 << r) == 0 {
                continue;
            }
            if prev != Some(r - 1) {
                run = 1;
                start = r;
            } else {
                run += 1;
            }
            if run > best_run {
                best_run = run;
                best_start = start;
            }
            prev = Some(r);
        }
        let straight_proximity = py_min(f64::from(best_run) / 5.0, 1.0);

        let mut gaps = 0u16;
        if best_run >= 3 {
            let run_end = best_start + best_run - 1;
            let mut push = |r: i32| {
                if (2..=14).contains(&r) && ranks & (1 << r) == 0 {
                    gaps |= 1 << r;
                }
            };
            // range(best_start - 1, max(best_start - 3, 0), -1)
            let mut r = best_start - 1;
            while r > (best_start - 3).max(0) {
                push(r);
                r -= 1;
            }
            // range(run_end + 1, min(run_end + 3, 15))
            for r in run_end + 1..(run_end + 3).min(15) {
                push(r);
            }
        }
        Draws {
            flush_proximity,
            straight_proximity,
            best_flush_suit: best_suit,
            best_flush_count: best_count,
            straight_gap_ranks: gaps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jackdaw_core::card::{Card, CostContext};
    use jackdaw_data::card_by_key;

    fn card(key: &str, sort_id: u32) -> Card {
        Card::new(
            sort_id,
            centers::C_BASE,
            card_by_key(key),
            0.0,
            &CostContext::default(),
        )
    }

    fn hand(keys: &[&str]) -> Vec<Card> {
        keys.iter()
            .enumerate()
            .map(|(i, k)| card(k, i as u32 + 1))
            .collect()
    }

    fn analyze(cards: &[Card], flags: Flags) -> (HandAnalysis, Vec<bool>) {
        let mut a = Analyzer::default();
        let r = a.analyze(Cards::from_slice(cards), flags);
        (r, a.scoring)
    }

    #[test]
    fn suit_bit_tables_follow_the_suits() {
        for s in Suit::ALL {
            assert_eq!(SUIT_BITS[s as usize], s.flush_bit());
            let colour = Suit::ALL
                .into_iter()
                .filter(|t| t.is_red() == s.is_red())
                .fold(0, |acc, t| acc | t.flush_bit());
            assert_eq!(SMEARED_BITS[s as usize], colour);
        }
    }

    #[test]
    fn flush_bits_are_flush_suits() {
        let wild: Vec<_> = jackdaw_data::CENTERS
            .iter()
            .enumerate()
            .filter(|(_, c)| c.name == "Wild Card")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(wild, [usize::from(centers::M_WILD.0)]);
        for center in 0..jackdaw_data::CENTERS.len() {
            for front in 0..52u8 {
                for debuff in [false, true] {
                    for smeared in [false, true] {
                        let mut c = card("S_A", 1);
                        c.base = Some(jackdaw_core::card::set_base(
                            None,
                            jackdaw_data::CardId(front),
                            true,
                        ));
                        c.center = jackdaw_data::CenterId(center as u16);
                        c.debuff = debuff;
                        assert_eq!(flush_bits(&c, smeared), c.flush_suits(smeared));
                        c.base = None;
                        assert_eq!(flush_bits(&c, smeared), c.flush_suits(smeared));
                    }
                }
            }
        }
    }

    /// A random area: up to 10 cards, often sharing ranks or suits or running in sequence, with
    /// random enhancements (Wild, Stone, ...), debuffs, face-down cards, changed suits and ranks
    /// (the original suit nominal kept), jokers without a base and repeated sort ids.
    fn random_cards(rng: &mut crate::gym::Xoshiro256) -> Vec<Card> {
        use jackdaw_core::card::{Base, Rank, set_base};
        let n = rng.below(11) as usize;
        let mode = rng.below(4);
        let start = rng.below(13) as usize;
        let pool: Vec<u64> = (0..1 + rng.below(4)).map(|_| rng.below(52)).collect();
        let suits = [rng.below(4) as usize, rng.below(4) as usize];
        let enhancements = [
            centers::C_BASE,
            centers::M_WILD,
            centers::M_STONE,
            centers::M_GLASS,
            centers::M_BONUS,
            centers::J_JOKER,
        ];
        let mut cards = Vec::new();
        for i in 0..n {
            let (suit, rank) = match mode {
                0 => (rng.below(4) as usize, rng.below(13) as usize),
                1 => {
                    let f = pool[rng.below(pool.len() as u64) as usize] as usize;
                    (f / 13, f % 13)
                }
                2 => (suits[rng.below(2) as usize], (start + i) % 13),
                _ => (suits[0], rng.below(13) as usize),
            };
            let front = Base::front_for(Suit::ALL[suit], Rank::ALL[rank]);
            let ids = if rng.below(3) == 0 { 4 } else { 5000 };
            let sort_id = 1 + rng.below(ids) as u32;
            let mut c = card("S_A", sort_id);
            c.base = Some(set_base(None, front, true));
            let kinds = if rng.below(2) == 0 { 1 } else { 6 };
            let center = enhancements[rng.below(kinds) as usize];
            c.center = center;
            if center == centers::J_JOKER {
                c.base = None;
            }
            if let Some(b) = c.base.as_mut() {
                if rng.below(8) == 0 {
                    b.suit = Suit::ALL[rng.below(4) as usize];
                }
                if rng.below(8) == 0 {
                    b.value = Rank::ALL[rng.below(13) as usize];
                }
            }
            c.debuff = rng.below(5) == 0;
            if rng.below(7) == 0 {
                c.facing = Facing::Back;
            }
            cards.push(c);
        }
        cards
    }

    #[test]
    fn matches_the_reference_on_random_areas() {
        let mut rng = crate::gym::Xoshiro256::seed_from_u64(3);
        let mut fast = Analyzer::default();
        for _ in 0..200_000 {
            let cards = random_cards(&mut rng);
            let bits = rng.below(8);
            let flags = Flags {
                four_fingers: bits & 1 != 0,
                shortcut: bits & 2 != 0,
                smeared: bits & 4 != 0,
            };
            let area = Cards::from_slice(&cards);
            let mut slow = reference::RefAnalyzer::default();
            let want = slow.analyze(area, flags);
            assert_eq!(fast.analyze(area, flags), want, "{cards:?} {flags:?}");
            assert_eq!(fast.scoring, slow.scoring, "{cards:?} {flags:?}");
            assert_eq!(
                analyze_draws(area, false),
                reference::analyze_draws(area),
                "{cards:?}"
            );
        }
    }

    #[test]
    fn three_pairs_make_two_pair_from_the_highest_two() {
        // Python takes len(_2) >= 2; Lua (and jackdaw_core::hand_eval) needs exactly two.
        let cards = hand(&["H_2", "S_2", "H_9", "C_9", "D_K", "S_K", "C_5", "D_7"]);
        let (r, scoring) = analyze(&cards, Flags::default());
        assert_eq!(r.best, Some(HandType::TwoPair));
        assert!(r.types[HandType::Pair.index()]);
        assert!(!r.types[HandType::Flush.index()], "eight cards never flush");
        assert_eq!(
            scoring,
            [false, false, true, true, true, true, false, false],
            "the kings and nines"
        );
    }

    #[test]
    fn six_of_a_rank_is_no_group() {
        let cards = hand(&["H_4", "S_4", "C_4", "D_4", "H_4", "S_4", "C_2"]);
        let (r, scoring) = analyze(&cards, Flags::default());
        assert_eq!(r.best, Some(HandType::HighCard));
        assert!(!r.types[HandType::Pair.index()]);
        // The highest nominal: the fours tie on rank, Spades has the highest suit nominal and
        // the older card wins a full tie.
        assert_eq!(scoring.iter().filter(|&&s| s).count(), 1);
        assert!(scoring[1]);
    }

    #[test]
    fn straights_need_at_most_five_cards() {
        let five = hand(&["H_A", "S_2", "C_3", "D_4", "H_5"]);
        let (r, scoring) = analyze(&five, Flags::default());
        assert_eq!(r.best, Some(HandType::Straight));
        assert!(scoring.iter().all(|&s| s));
        let mut six = five.clone();
        six.push(card("S_9", 9));
        let (r, _) = analyze(&six, Flags::default());
        assert_eq!(r.best, Some(HandType::HighCard));
    }

    #[test]
    fn face_down_cards_are_left_out() {
        let mut cards = hand(&["H_9", "S_9", "C_3"]);
        cards[1].facing = Facing::Back;
        let (r, scoring) = analyze(&cards, Flags::default());
        assert_eq!(r.best, Some(HandType::HighCard));
        assert_eq!(scoring, [true, false, false]);
        cards.iter_mut().for_each(|c| c.facing = Facing::Back);
        let (r, scoring) = analyze(&cards, Flags::default());
        assert_eq!(r, HandAnalysis::default());
        assert_eq!(scoring, [false, false, false]);
    }

    #[test]
    fn draws_find_runs_and_gaps() {
        let cards = hand(&["H_5", "S_6", "C_7", "D_K", "H_2"]);
        let d = analyze_draws(Cards::from_slice(&cards), false);
        assert_eq!(d.best_flush_suit, Some(Suit::Hearts));
        assert_eq!(d.best_flush_count, 2);
        assert_eq!(d.straight_proximity, 3.0 / 5.0);
        // Run 5-6-7: below 4 and 3, above 8 and 9.
        let want = (1 << 3) | (1 << 4) | (1 << 8) | (1 << 9);
        assert_eq!(d.straight_gap_ranks, want);
    }

    #[test]
    fn ace_low_run_has_no_gap_below() {
        let cards = hand(&["H_A", "S_2", "C_3"]);
        let d = analyze_draws(Cards::from_slice(&cards), false);
        assert_eq!(d.straight_gap_ranks, (1 << 4) | (1 << 5));
    }
}
