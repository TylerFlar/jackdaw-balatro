//! Card areas (`CardArea`, cardarea.lua): the ten areas of [`Areas`] addressed by id, with the
//! gameplay part of the Lua methods.
//!
//! Card identity is the sort id: Lua code holds card references across events, and the port
//! holds sort ids and looks the card up again ([`Areas::locate`]).
//!
//! Facing follows Lua: `CardArea:align_cards` turns every card in the deck and the discard pile
//! face down (cardarea.lua:415, 427), and `emplace` turns a face-down card face up when it lands
//! anywhere else unless it stays flipped (cardarea.lua:38). The snapshot writes deck and
//! discard cards as face up, since their facing carries no information there.

use crate::card::{Card, Facing};
use crate::rng::{Rng, shuffle};
use crate::state::Areas;

/// Reorder `items` so that position `k` holds the item that was at `order[k]` (a permutation
/// of the positions). Each cycle of the permutation moves its items once, plus one copy.
pub(crate) fn permute<T: Copy>(items: &mut [T], order: &[usize]) {
    let n = items.len();
    let mut seen_inline = [false; 256];
    let mut seen_spill = Vec::new();
    let seen: &mut [bool] = if n <= seen_inline.len() {
        &mut seen_inline[..n]
    } else {
        seen_spill.resize(n, false);
        &mut seen_spill
    };
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let first = items[start];
        let mut k = start;
        loop {
            seen[k] = true;
            let from = order[k];
            if from == start {
                items[k] = first;
                break;
            }
            items[k] = items[from];
            k = from;
        }
    }
}

/// The sort ids of an area's cards, kept on the stack for areas of up to [`Ids::INLINE`] cards
/// (the loops over the jokers or the hand run for every scored card).
#[derive(Clone)]
pub struct Ids {
    len: usize,
    inline: [u32; Ids::INLINE],
    spill: Vec<u32>,
}

impl Ids {
    pub const INLINE: usize = 16;

    pub fn of(cards: &[Card]) -> Ids {
        let mut ids = Ids {
            len: cards.len(),
            inline: [0; Ids::INLINE],
            spill: Vec::new(),
        };
        if cards.len() <= Ids::INLINE {
            for (slot, c) in ids.inline.iter_mut().zip(cards) {
                *slot = c.sort_id;
            }
        } else {
            ids.spill = cards.iter().map(|c| c.sort_id).collect();
        }
        ids
    }

    pub fn push(&mut self, id: u32) {
        if self.len < Ids::INLINE {
            self.inline[self.len] = id;
        } else {
            if self.len == Ids::INLINE {
                self.spill = self.inline.to_vec();
            }
            self.spill.push(id);
        }
        self.len += 1;
    }

    pub fn from_slice(ids: &[u32]) -> Ids {
        let mut out = Ids {
            len: ids.len(),
            inline: [0; Ids::INLINE],
            spill: Vec::new(),
        };
        if ids.len() <= Ids::INLINE {
            out.inline[..ids.len()].copy_from_slice(ids);
        } else {
            out.spill = ids.to_vec();
        }
        out
    }
}

impl std::ops::Deref for Ids {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        if self.len <= Ids::INLINE {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }
}

impl std::ops::DerefMut for Ids {
    fn deref_mut(&mut self) -> &mut [u32] {
        if self.len <= Ids::INLINE {
            &mut self.inline[..self.len]
        } else {
            &mut self.spill
        }
    }
}

impl std::fmt::Debug for Ids {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl PartialEq for Ids {
    fn eq(&self, other: &Ids) -> bool {
        **self == **other
    }
}

impl<'a> IntoIterator for &'a Ids {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for Ids {
    type Item = u32;
    type IntoIter = IdsIter;

    fn into_iter(self) -> IdsIter {
        let end = self.len;
        IdsIter {
            ids: self,
            at: 0,
            end,
        }
    }
}

/// [`Ids`] by value.
pub struct IdsIter {
    ids: Ids,
    at: usize,
    end: usize,
}

impl Iterator for IdsIter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.at == self.end {
            return None;
        }
        self.at += 1;
        Some(self.ids[self.at - 1])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.at;
        (n, Some(n))
    }
}

impl DoubleEndedIterator for IdsIter {
    fn next_back(&mut self) -> Option<u32> {
        if self.at == self.end {
            return None;
        }
        self.end -= 1;
        Some(self.ids[self.end])
    }
}

impl ExactSizeIterator for IdsIter {}

/// The cards [`Areas::draw_cards`] moves.
#[derive(Clone, Copy, Debug)]
pub enum Draw<'a> {
    /// These cards, in this order; a card no longer in the area (or listed twice) is skipped.
    Cards(&'a [u32]),
    /// The area's next cards, one after the other (`CardArea:draw_card_from`): the last card of
    /// a deck or discard pile, else the first.
    Next(usize),
}

/// One of the run's card areas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AreaId {
    Deck,
    Hand,
    Discard,
    Play,
    Jokers,
    Consumables,
    ShopJokers,
    ShopVouchers,
    ShopBooster,
    PackCards,
}

/// `CardArea.config.type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaType {
    Deck,
    Discard,
    Hand,
    Play,
    /// `G.jokers` and `G.consumeables` (game.lua:2235-2245).
    Joker,
    /// The shop's three areas (UI_definitions.lua:638-655).
    Shop,
    /// `G.pack_cards` (UI_definitions.lua:1631).
    Consumeable,
}

impl AreaId {
    pub const ALL: [AreaId; 10] = [
        AreaId::Deck,
        AreaId::Hand,
        AreaId::Discard,
        AreaId::Play,
        AreaId::Jokers,
        AreaId::Consumables,
        AreaId::ShopJokers,
        AreaId::ShopVouchers,
        AreaId::ShopBooster,
        AreaId::PackCards,
    ];

    /// The snapshot name (`Areas::NAMES`).
    pub fn name(self) -> &'static str {
        Areas::NAMES[self as usize]
    }

    pub fn kind(self) -> AreaType {
        match self {
            AreaId::Deck => AreaType::Deck,
            AreaId::Discard => AreaType::Discard,
            AreaId::Hand => AreaType::Hand,
            AreaId::Play => AreaType::Play,
            AreaId::Jokers | AreaId::Consumables => AreaType::Joker,
            AreaId::ShopJokers | AreaId::ShopVouchers | AreaId::ShopBooster => AreaType::Shop,
            AreaId::PackCards => AreaType::Consumeable,
        }
    }

    /// Deck and discard keep their cards face down.
    pub fn face_down(self) -> bool {
        matches!(self.kind(), AreaType::Deck | AreaType::Discard)
    }

    /// Areas that hold playing cards between steps (`G.playing_cards` lives in these).
    pub const PLAYING: [AreaId; 4] = [AreaId::Deck, AreaId::Hand, AreaId::Discard, AreaId::Play];

    /// Every other area: the jokers, consumables, shop and pack.
    pub const NOT_PLAYING: [AreaId; 6] = [
        AreaId::Jokers,
        AreaId::Consumables,
        AreaId::ShopJokers,
        AreaId::ShopVouchers,
        AreaId::ShopBooster,
        AreaId::PackCards,
    ];
}

impl Areas {
    pub fn get(&self, id: AreaId) -> &Vec<Card> {
        match id {
            AreaId::Deck => &self.deck,
            AreaId::Hand => &self.hand,
            AreaId::Discard => &self.discard,
            AreaId::Play => &self.play,
            AreaId::Jokers => &self.jokers,
            AreaId::Consumables => &self.consumables,
            AreaId::ShopJokers => &self.shop_jokers,
            AreaId::ShopVouchers => &self.shop_vouchers,
            AreaId::ShopBooster => &self.shop_booster,
            AreaId::PackCards => &self.pack_cards,
        }
    }

    pub fn get_mut(&mut self, id: AreaId) -> &mut Vec<Card> {
        match id {
            AreaId::Deck => &mut self.deck,
            AreaId::Hand => &mut self.hand,
            AreaId::Discard => &mut self.discard,
            AreaId::Play => &mut self.play,
            AreaId::Jokers => &mut self.jokers,
            AreaId::Consumables => &mut self.consumables,
            AreaId::ShopJokers => &mut self.shop_jokers,
            AreaId::ShopVouchers => &mut self.shop_vouchers,
            AreaId::ShopBooster => &mut self.shop_booster,
            AreaId::PackCards => &mut self.pack_cards,
        }
    }

    /// The area and index of the card with this sort id. A card is in one area only, so the
    /// search order is free: the small areas most lookups are for come first, the deck last.
    pub fn locate(&self, sort_id: u32) -> Option<(AreaId, usize)> {
        const ORDER: [AreaId; 10] = [
            AreaId::Hand,
            AreaId::Play,
            AreaId::Jokers,
            AreaId::Consumables,
            AreaId::PackCards,
            AreaId::ShopJokers,
            AreaId::ShopVouchers,
            AreaId::ShopBooster,
            AreaId::Discard,
            AreaId::Deck,
        ];
        ORDER.into_iter().find_map(|a| {
            self.get(a)
                .iter()
                .position(|c| c.sort_id == sort_id)
                .map(|i| (a, i))
        })
    }

    pub fn card(&self, sort_id: u32) -> Option<&Card> {
        self.locate(sort_id).map(|(a, i)| &self.get(a)[i])
    }

    pub fn card_mut(&mut self, sort_id: u32) -> Option<&mut Card> {
        let (a, i) = self.locate(sort_id)?;
        Some(&mut self.get_mut(a)[i])
    }

    /// [`Areas::card`], looking among the jokers and consumables first: `calculate_joker`'s
    /// cards are there.
    pub fn joker_card(&self, sort_id: u32) -> Option<&Card> {
        self.jokers
            .iter()
            .chain(&self.consumables)
            .find(|c| c.sort_id == sort_id)
            .or_else(|| self.card(sort_id))
    }

    /// The sort ids of an area's cards, in order.
    pub fn ids(&self, id: AreaId) -> Ids {
        Ids::of(self.get(id))
    }

    /// `CardArea:emplace(card, location, stay_flipped)` (cardarea.lua:32-64), minus UI.
    pub fn emplace(&mut self, id: AreaId, mut card: Card, front: bool, stay_flipped: bool) {
        if id.face_down() {
            card.facing = Facing::Back;
        } else if card.facing == Facing::Back && !stay_flipped {
            card.facing = Facing::Front;
        }
        let cards = self.get_mut(id);
        // A copied game's areas have no spare room: grow by at least a hand's worth at once.
        if cards.len() == cards.capacity() {
            cards.reserve(cards.len().max(8));
        }
        if front || id.kind() == AreaType::Deck {
            cards.insert(0, card);
        } else {
            cards.push(card);
        }
        self.align(id);
    }

    /// `emplace` each card in turn, in one move (a deck takes each new card in front).
    pub fn emplace_all(&mut self, id: AreaId, mut new: Vec<Card>) {
        for card in &mut new {
            if id.face_down() {
                card.facing = Facing::Back;
            } else if card.facing == Facing::Back {
                card.facing = Facing::Front;
            }
        }
        let cards = self.get_mut(id);
        if id.kind() == AreaType::Deck {
            new.reverse();
            cards.splice(0..0, new);
        } else {
            cards.extend(new);
        }
        self.align(id);
    }

    /// `draw_card`'s moves from `from` to `to`, made at once: every card lands as
    /// [`Areas::emplace`] puts it (none stays flipped) and the result is that of moving them one
    /// at a time, but each card moves once. One at a time, taking a card out shifts every card
    /// behind it, and the deck takes each new card in front. Returns the sort ids moved, in the
    /// order they moved.
    pub fn draw_cards(&mut self, from: AreaId, to: AreaId, draw: Draw<'_>) -> Ids {
        debug_assert_ne!(from, to);
        let src = std::mem::take(self.get_mut(from));
        let n = src.len();
        // The positions in `from` of the cards that move, in the order they move.
        let mut order = Ids::from_slice(&[]);
        let mut taken_inline = [false; 256];
        let mut taken_spill = Vec::new();
        let taken: &mut [bool] = if n <= taken_inline.len() {
            &mut taken_inline[..n]
        } else {
            taken_spill.resize(n, false);
            &mut taken_spill
        };
        match draw {
            Draw::Cards(ids) => {
                for &id in ids {
                    // `remove_card(card)` finds nothing for a card that already moved.
                    if let Some(p) = src.iter().position(|c| c.sort_id == id)
                        && !taken[p]
                    {
                        taken[p] = true;
                        order.push(p as u32);
                    }
                }
            }
            Draw::Next(count) => {
                let count = count.min(n);
                for k in 0..count {
                    // `remove_card()`: the last card of a deck or discard pile, else the first.
                    let p = if matches!(from.kind(), AreaType::Deck | AreaType::Discard) {
                        n - 1 - k
                    } else {
                        k
                    };
                    taken[p] = true;
                    order.push(p as u32);
                }
            }
        }
        let land = |p: &u32| {
            let mut c = src[*p as usize];
            if to.face_down() {
                c.facing = Facing::Back;
            } else if c.facing == Facing::Back {
                c.facing = Facing::Front;
            }
            c
        };
        let mut moved = Ids::from_slice(&[]);
        for &p in &order {
            moved.push(src[p as usize].sort_id);
        }
        let dst = self.get_mut(to);
        if to.kind() == AreaType::Deck {
            // Each card goes in front of the ones before it.
            dst.splice(0..0, order.iter().rev().map(land));
        } else {
            dst.extend(order.iter().map(land));
        }
        let mut src = src;
        let mut at = 0;
        src.retain(|_| {
            at += 1;
            !taken[at - 1]
        });
        *self.get_mut(from) = src;
        self.align(to);
        moved
    }

    /// `CardArea:remove_card(card)` (cardarea.lua:66-92): the given card, or the area's next
    /// card (the last one of a deck or discard pile, else the first).
    pub fn remove_card(&mut self, id: AreaId, sort_id: Option<u32>) -> Option<Card> {
        let cards = self.get_mut(id);
        let index = match sort_id {
            Some(s) => cards.iter().position(|c| c.sort_id == s)?,
            None if matches!(id.kind(), AreaType::Deck | AreaType::Discard) => {
                cards.len().checked_sub(1)?
            }
            None if cards.is_empty() => return None,
            None => 0,
        };
        Some(cards.remove(index))
    }

    /// `CardArea:align_cards()` (cardarea.lua:410-552) where it changes order: positions follow
    /// the index everywhere except that pinned jokers sit leftmost, newest first (line 528).
    pub fn align(&mut self, id: AreaId) {
        if id.kind() == AreaType::Joker && self.get(id).iter().any(|c| c.pinned) {
            let cards = self.get_mut(id);
            let key = |(k, c): &(usize, &Card)| -> (bool, i64, usize) {
                if c.pinned {
                    (false, -i64::from(c.sort_id), *k)
                } else {
                    (true, 0, *k)
                }
            };
            let mut order: Vec<(usize, &Card)> = cards.iter().enumerate().collect();
            order.sort_by_key(key);
            let perm: Vec<usize> = order.into_iter().map(|(k, _)| k).collect();
            let old = std::mem::take(cards);
            let mut slots: Vec<Option<Card>> = old.into_iter().map(Some).collect();
            *cards = perm
                .into_iter()
                .map(|k| slots[k].take().expect("permutation"))
                .collect();
        }
    }

    /// `CardArea:shuffle(seed)` (cardarea.lua:572): `pseudoshuffle(self.cards, pseudoseed(seed))`.
    /// The next frame's `align_cards` puts pinned jokers back in front (cardarea.lua:528; G200).
    pub fn shuffle(&mut self, id: AreaId, rng: &mut Rng, key: &str) {
        let seed = rng.seed(key);
        let cards = self.get_mut(id);
        // pseudoshuffle sorts by sort_id first when the entries carry one (misc_functions.lua),
        // then swaps. Both run on positions, and the result moves each card at most once.
        let mut order: Vec<usize> = (0..cards.len()).collect();
        if !cards.is_sorted_by_key(|c| c.sort_id) {
            // Sort ids are unique, so any sort gives the stable sort's order. The sort runs on
            // (sort id, position) words, away from the cards.
            let mut keyed: Vec<u64> = cards
                .iter()
                .enumerate()
                .map(|(i, c)| (u64::from(c.sort_id) << 32) | i as u64)
                .collect();
            keyed.sort_unstable();
            for (slot, k) in order.iter_mut().zip(&keyed) {
                *slot = (k & 0xFFFF_FFFF) as usize;
            }
        }
        shuffle(&mut order, seed);
        permute(cards, &order);
        self.align(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::CostContext;
    use jackdaw_data::{card_by_key, centers};

    fn card(sort_id: u32, facing: Facing) -> Card {
        const FRONTS: [&str; 8] = ["S_A", "H_K", "D_2", "C_T", "S_5", "H_9", "D_J", "C_3"];
        let front = card_by_key(FRONTS[sort_id as usize % FRONTS.len()]);
        let mut c = Card::new(
            sort_id,
            centers::C_BASE,
            front,
            0.0,
            &CostContext::default(),
        );
        c.facing = facing;
        c
    }

    /// The moves one `draw_card` event at a time, as the event handler makes them.
    fn one_at_a_time(areas: &mut Areas, from: AreaId, to: AreaId, draw: Draw<'_>) -> Vec<u32> {
        let mut moved = Vec::new();
        let mut step = |areas: &mut Areas, id: Option<u32>| {
            if let Some(c) = areas.remove_card(from, id) {
                moved.push(c.sort_id);
                areas.emplace(to, c, false, false);
            }
        };
        match draw {
            Draw::Cards(ids) => ids.iter().for_each(|&id| step(areas, Some(id))),
            Draw::Next(n) => (0..n).for_each(|_| step(areas, None)),
        }
        moved
    }

    #[test]
    fn batched_draws_move_the_cards_as_single_draws_do() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut below = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n.max(1)
        };
        for _ in 0..2_000 {
            let mut areas = Areas::default();
            let mut next_id = 1;
            for a in AreaId::PLAYING {
                for _ in 0..below(14) {
                    let facing = if below(2) == 0 {
                        Facing::Front
                    } else {
                        Facing::Back
                    };
                    areas.get_mut(a).push(card(next_id, facing));
                    next_id += 1;
                }
            }
            let from = AreaId::PLAYING[below(4) as usize];
            let to = loop {
                let to = AreaId::PLAYING[below(4) as usize];
                if to != from {
                    break to;
                }
            };
            // Cards of `from`, others, and repeats, in any order.
            let ids: Vec<u32> = (0..below(9))
                .map(|_| 1 + below(next_id as u64) as u32)
                .collect();
            let count = below(16) as usize;
            for draw in [Draw::Cards(&ids), Draw::Next(count)] {
                let mut single = areas.clone();
                let expected = one_at_a_time(&mut single, from, to, draw);
                let mut batched = areas.clone();
                let moved = batched.draw_cards(from, to, draw);
                assert_eq!(*moved, expected[..], "{from:?} -> {to:?} {draw:?}");
                assert_eq!(batched, single, "{from:?} -> {to:?} {draw:?}");
            }
        }
    }
}
