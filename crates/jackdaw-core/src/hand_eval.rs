//! Poker-hand evaluation: `evaluate_poker_hand` and its helpers (`functions/misc_functions.lua`
//! 376-621) and `G.FUNCS.get_poker_hand_info` (`functions/state_events.lua` 540-569).
//!
//! A literal port: the same passes in the same order, so each poker hand's groups hold the same
//! cards in the same order as in Lua. Jokers' scoring conditions (`next(poker_hands["Pair"])`)
//! and the scoring hand depend on those groups, including Lua's quirks: the downgrades at the
//! end make a full house's "Pair" the three-of-a-kind cards, and exactly two pairs (not three)
//! make Two Pair. Groups are fixed-capacity, so evaluation never allocates.

/// The most cards one evaluation takes.
pub const MAX_CARDS: usize = 16;
const MAX_GROUPS: usize = MAX_CARDS / 2;

/// The twelve poker hands, in Lua `order` (`G.GAME.hands`), best first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandType {
    FlushFive,
    FlushHouse,
    FiveOfAKind,
    StraightFlush,
    FourOfAKind,
    FullHouse,
    Flush,
    Straight,
    ThreeOfAKind,
    TwoPair,
    Pair,
    HighCard,
}

impl HandType {
    pub const ALL: [HandType; 12] = [
        Self::FlushFive,
        Self::FlushHouse,
        Self::FiveOfAKind,
        Self::StraightFlush,
        Self::FourOfAKind,
        Self::FullHouse,
        Self::Flush,
        Self::Straight,
        Self::ThreeOfAKind,
        Self::TwoPair,
        Self::Pair,
        Self::HighCard,
    ];

    /// Index into `G.GAME.hands` order (0 = Flush Five).
    pub fn index(self) -> usize {
        self as usize
    }

    /// The Lua name (`"Flush Five"`, ...).
    pub fn name(self) -> &'static str {
        jackdaw_data::HANDS[self.index()].name
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|h| h.name() == name)
    }
}

/// The order `pairs(G.GAME.hands)` visits the hands in: the game's own LuaJIT 2.0.5 on
/// `init_game_object`'s `hands` constructor (G182, `scripts/luajit205_probe.py hands-order`;
/// LuaJIT 2.0 hashes strings without a seed, and Steamodded only rewrites the existing keys).
/// Confirmed live: To Do List's hand matches the game's on every seed tried. The most played
/// hand's tie-break, To Do List and the Orbital Tag depend on it.
pub const PAIRS_ORDER: [HandType; 12] = [
    HandType::FlushHouse,
    HandType::FullHouse,
    HandType::Flush,
    HandType::Pair,
    HandType::HighCard,
    HandType::StraightFlush,
    HandType::Straight,
    HandType::TwoPair,
    HandType::FlushFive,
    HandType::FiveOfAKind,
    HandType::ThreeOfAKind,
    HandType::FourOfAKind,
];

/// Lua's flush-suit order in `get_flush`.
pub const FLUSH_SUITS: [&str; 4] = ["Spades", "Hearts", "Clubs", "Diamonds"];

/// What evaluation needs from a card.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalCard {
    /// `card:get_id()`: the rank id 2-14, or `None` for a Stone card. Lua returns a fresh random
    /// negative id on every call for a Stone card (`card.lua` `Card:get_id`), so it never groups
    /// with another card or joins a straight. (A vampired Stone card reports its rank.)
    pub id: Option<u8>,
    /// `card:is_suit(suit, nil, true)` (the flush test) for each of [`FLUSH_SUITS`], as bits.
    pub flush_suits: u8,
    /// `card:get_nominal()`, compared by `get_highest`.
    pub nominal: f64,
}

/// The most cards a group holds: an x-of-a-kind holds x <= 5, flushes and straights only form
/// in hands of five cards or fewer, and the hands built from two groups (Two Pair, Full House,
/// Flush House, Straight Flush) take at most five positions.
const GROUP_CAP: usize = 5;

/// The evaluation views of a hand, on the stack.
#[derive(Clone, Copy)]
pub struct EvalCards {
    len: usize,
    cards: [EvalCard; MAX_CARDS],
}

impl FromIterator<EvalCard> for EvalCards {
    fn from_iter<I: IntoIterator<Item = EvalCard>>(iter: I) -> EvalCards {
        let blank = EvalCard {
            id: None,
            flush_suits: 0,
            nominal: 0.0,
        };
        let mut out = EvalCards {
            len: 0,
            cards: [blank; MAX_CARDS],
        };
        for c in iter {
            assert!(
                out.len < MAX_CARDS,
                "evaluating more than {MAX_CARDS} cards"
            );
            out.cards[out.len] = c;
            out.len += 1;
        }
        out
    }
}

impl std::ops::Deref for EvalCards {
    type Target = [EvalCard];

    fn deref(&self) -> &[EvalCard] {
        &self.cards[..self.len]
    }
}

/// Positions in the evaluated hand (0-based).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Group {
    idx: [u8; GROUP_CAP],
    len: u8,
}

impl Default for Group {
    fn default() -> Self {
        Self {
            idx: [0; GROUP_CAP],
            len: 0,
        }
    }
}

impl Group {
    fn push(&mut self, i: usize) {
        self.idx[self.len as usize] = i as u8;
        self.len += 1;
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

    pub fn contains(&self, i: usize) -> bool {
        self.as_slice().contains(&(i as u8))
    }
}

impl std::fmt::Debug for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

/// A poker hand's entry in `evaluate_poker_hand`'s result: a list of groups.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Groups {
    g: [Group; MAX_GROUPS],
    len: u8,
}

impl Groups {
    fn push(&mut self, group: Group) {
        self.g[self.len as usize] = group;
        self.len += 1;
    }

    fn single(group: Group) -> Self {
        let mut gs = Self::default();
        gs.push(group);
        gs
    }

    /// The first `n` groups (Lua's `{t[1], t[2], ..., t[n]}` with trailing nils dropped).
    fn first(&self, n: usize) -> Self {
        let mut gs = *self;
        gs.len = gs.len.min(n as u8);
        gs
    }

    pub fn as_slice(&self) -> &[Group] {
        &self.g[..self.len as usize]
    }

    /// `next(t) ~= nil`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }
}

impl std::fmt::Debug for Groups {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

/// `evaluate_poker_hand(hand)`: every poker hand's groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PokerHands {
    by_type: [Groups; 12],
}

impl PokerHands {
    pub fn get(&self, hand: HandType) -> &Groups {
        &self.by_type[hand.index()]
    }

    /// `next(poker_hands[hand])`: the hand contains this poker hand.
    pub fn contains(&self, hand: HandType) -> bool {
        !self.get(hand).is_empty()
    }

    fn set(&mut self, hand: HandType, groups: Groups) {
        self.by_type[hand.index()] = groups;
    }
}

/// The joker flags that change how hands are formed (`find_joker` on non-debuffed jokers).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalFlags {
    /// Four Fingers: flushes and straights need 4 cards.
    pub four_fingers: bool,
    /// Shortcut: straights may skip single ranks.
    pub shortcut: bool,
}

/// `get_X_same(num, hand)` for num 2, 3, 4 and 5 (in that order), from one pass over the hand.
fn get_x_same_all(hand: &[EvalCard]) -> [Groups; 4] {
    let mut count = [0usize; 15];
    let mut at = [[0u8; MAX_CARDS]; 15];
    for (i, c) in hand.iter().enumerate() {
        if let Some(id) = c.id {
            let id = id as usize;
            at[id][count[id]] = i as u8;
            count[id] += 1;
        }
    }
    // Lua keeps, per id, the group built from the first card of that id: its positions in order.
    let mut ret = [Groups::default(); 4];
    for (slot, num) in ret.iter_mut().zip(2..=5) {
        for id in (1..=14).rev() {
            if count[id] == num {
                let mut g = Group::default();
                for &i in &at[id][..num] {
                    g.push(usize::from(i));
                }
                slot.push(g);
            }
        }
    }
    ret
}

fn get_flush(hand: &[EvalCard], flags: EvalFlags) -> Groups {
    let need = if flags.four_fingers { 4 } else { 5 };
    if hand.len() > 5 || hand.len() < need {
        return Groups::default();
    }
    for bit in 0..4 {
        let mut t = Group::default();
        for (i, c) in hand.iter().enumerate() {
            if c.flush_suits & (1 << bit) != 0 {
                t.push(i);
            }
        }
        if t.len() >= need {
            return Groups::single(t);
        }
    }
    Groups::default()
}

fn get_straight(hand: &[EvalCard], flags: EvalFlags) -> Groups {
    let need = if flags.four_fingers { 4 } else { 5 };
    if hand.len() > 5 || hand.len() < need {
        return Groups::default();
    }
    let mut ids = [Group::default(); 15];
    for (i, c) in hand.iter().enumerate() {
        if let Some(id) = c.id
            && id > 1
            && id < 15
        {
            ids[id as usize].push(i);
        }
    }
    let mut t = Group::default();
    let mut straight_length = 0;
    let mut straight = false;
    let mut skipped_rank = false;
    for j in 1..=14usize {
        let rank = if j == 1 { 14 } else { j };
        if !ids[rank].is_empty() {
            straight_length += 1;
            skipped_rank = false;
            for &i in ids[rank].as_slice() {
                t.push(i as usize);
            }
        } else if flags.shortcut && !skipped_rank && j != 14 {
            skipped_rank = true;
        } else {
            straight_length = 0;
            skipped_rank = false;
            if !straight {
                t = Group::default();
            }
            if straight {
                break;
            }
        }
        if straight_length >= need {
            straight = true;
        }
    }
    if straight {
        Groups::single(t)
    } else {
        Groups::default()
    }
}

fn get_highest(hand: &[EvalCard]) -> Groups {
    let mut highest: Option<usize> = None;
    for (i, c) in hand.iter().enumerate() {
        if highest.is_none_or(|h| c.nominal > hand[h].nominal) {
            highest = Some(i);
        }
    }
    match highest {
        Some(h) => {
            let mut g = Group::default();
            g.push(h);
            Groups::single(g)
        }
        None => Groups::default(),
    }
}

fn concat(a: &Group, b: &Group) -> Group {
    let mut g = *a;
    for &i in b.as_slice() {
        g.push(i as usize);
    }
    g
}

/// `evaluate_poker_hand(hand)`.
pub fn evaluate_poker_hand(hand: &[EvalCard], flags: EvalFlags) -> PokerHands {
    assert!(hand.len() <= MAX_CARDS, "evaluating {} cards", hand.len());
    use HandType::*;
    let [p2, p3, p4, p5] = get_x_same_all(hand);
    let flush = get_flush(hand, flags);
    let straight = get_straight(hand, flags);
    let highest = get_highest(hand);

    let mut r = PokerHands::default();
    if !p5.is_empty() && !flush.is_empty() {
        r.set(FlushFive, p5);
    }
    if !p3.is_empty() && !p2.is_empty() && !flush.is_empty() {
        r.set(FlushHouse, Groups::single(concat(&p3.g[0], &p2.g[0])));
    }
    if !p5.is_empty() {
        r.set(FiveOfAKind, p5);
    }
    if !flush.is_empty() && !straight.is_empty() {
        let f = flush.g[0];
        let mut ret = f;
        for &i in straight.g[0].as_slice() {
            if !f.contains(i as usize) {
                ret.push(i as usize);
            }
        }
        r.set(StraightFlush, Groups::single(ret));
    }
    if !p4.is_empty() {
        r.set(FourOfAKind, p4);
    }
    if !p3.is_empty() && !p2.is_empty() {
        r.set(FullHouse, Groups::single(concat(&p3.g[0], &p2.g[0])));
    }
    if !flush.is_empty() {
        r.set(Flush, flush);
    }
    if !straight.is_empty() {
        r.set(Straight, straight);
    }
    if !p3.is_empty() {
        r.set(ThreeOfAKind, p3);
    }
    if p2.len() == 2 || (p3.len() == 1 && p2.len() == 1) {
        let a = p2.g[0];
        let b = if p2.len() >= 2 { p2.g[1] } else { p3.g[0] };
        r.set(TwoPair, Groups::single(concat(&a, &b)));
    }
    if !p2.is_empty() {
        r.set(Pair, p2);
    }
    if !highest.is_empty() {
        r.set(HighCard, highest);
    }
    // Lower hands are rebuilt from the higher ones (misc_functions.lua 512-522).
    if !r.get(FiveOfAKind).is_empty() {
        let g = r.get(FiveOfAKind).first(4);
        r.set(FourOfAKind, g);
    }
    if !r.get(FourOfAKind).is_empty() {
        let g = r.get(FourOfAKind).first(3);
        r.set(ThreeOfAKind, g);
    }
    if !r.get(ThreeOfAKind).is_empty() {
        let g = r.get(ThreeOfAKind).first(2);
        r.set(Pair, g);
    }
    r
}

/// `G.FUNCS.get_poker_hand_info(cards)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandInfo {
    /// The best poker hand; `None` for no cards (Lua's `'NULL'`).
    pub hand: Option<HandType>,
    pub poker_hands: PokerHands,
    /// `scoring_hand`: the best hand's first group.
    pub scoring: Group,
    /// A straight flush whose lowest id is 10 or more ("Royal Flush").
    pub royal: bool,
}

/// `G.FUNCS.get_poker_hand_info(cards)`.
pub fn get_poker_hand_info(hand: &[EvalCard], flags: EvalFlags) -> HandInfo {
    let poker_hands = evaluate_poker_hand(hand, flags);
    let best = HandType::ALL.into_iter().find(|&h| poker_hands.contains(h));
    let scoring = best.map(|h| poker_hands.get(h).g[0]).unwrap_or_default();
    let royal = best == Some(HandType::StraightFlush)
        && scoring
            .as_slice()
            .iter()
            .all(|&i| hand[i as usize].id.is_some_and(|id| id >= 10));
    HandInfo {
        hand: best,
        poker_hands,
        scoring,
        royal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(id: u8, suit: usize) -> EvalCard {
        EvalCard {
            id: Some(id),
            flush_suits: 1 << suit,
            nominal: f64::from(id),
        }
    }

    #[test]
    fn hand_type_names_follow_lua_order() {
        assert_eq!(HandType::FlushFive.name(), "Flush Five");
        assert_eq!(HandType::HighCard.name(), "High Card");
        assert_eq!(HandType::from_name("Two Pair"), Some(HandType::TwoPair));
    }

    #[test]
    fn full_house_pair_is_the_three_cards() {
        let hand = [card(5, 0), card(5, 1), card(9, 2), card(5, 3), card(9, 0)];
        let info = get_poker_hand_info(&hand, EvalFlags::default());
        assert_eq!(info.hand, Some(HandType::FullHouse));
        assert_eq!(info.scoring.as_slice(), &[0, 1, 3, 2, 4]);
        // Lua rebuilds "Pair" from "Three of a Kind": the three fives, not the nines.
        assert_eq!(
            info.poker_hands.get(HandType::Pair).as_slice()[0].as_slice(),
            &[0, 1, 3]
        );
    }

    #[test]
    fn empty_hand_is_null() {
        let info = get_poker_hand_info(&[], EvalFlags::default());
        assert_eq!(info.hand, None);
        assert!(info.scoring.is_empty());
    }
}
