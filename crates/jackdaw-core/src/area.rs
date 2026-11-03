//! Card areas (`CardArea`, cardarea.lua): the ten areas of [`Areas`] addressed by id, with the
//! gameplay part of the Lua methods.
//!
//! Card identity is the sort id: Lua code holds card references across events, and the port
//! holds sort ids and looks the card up again ([`Areas::card`], [`Areas::locate`]). The cards
//! live in an arena and the areas list their slots, so moving a card between or within areas
//! moves a slot number, never the card.
//!
//! Facing follows Lua: `CardArea:align_cards` turns every card in the deck and the discard pile
//! face down (cardarea.lua:415, 427), and `emplace` turns a face-down card face up when it lands
//! anywhere else unless it stays flipped (cardarea.lua:38). The snapshot writes deck and
//! discard cards as face up, since their facing carries no information there.

use crate::card::{Card, Facing};
use crate::rng::{Rng, shuffle};

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

/// No slot: a sort id no card in an area carries.
const NO_SLOT: u32 = u32::MAX;

/// The area of a free slot.
const FREE: u8 = u8::MAX;

/// The card areas, in area order.
///
/// The cards live in an arena and never move while they stay in an area: each area is a list
/// of arena slots, so drawing, shuffling, sorting and reordering move 4-byte slots instead of
/// whole cards, and a card is found by its sort id in one lookup. A card taken out of its area
/// ([`Areas::remove_card`]) leaves the arena and its slot is reused.
///
/// Equality and `Debug` see the areas' card sequences, not the arena's layout.
#[derive(Default)]
pub struct Areas {
    /// The cards, by slot. A free slot holds a stale card no area lists.
    cards: Vec<Card>,
    /// Each slot's area (`AreaId as u8`), or `FREE`.
    home: Vec<u8>,
    /// Free slots, the last freed last.
    free: Vec<u32>,
    /// Each area's cards in order, as slots (`AreaId` order).
    lists: [Vec<u32>; 10],
    /// By sort id: the slot of the area card with it, or `NO_SLOT`.
    by_sort: Vec<u32>,
}

impl Clone for Areas {
    fn clone(&self) -> Self {
        Areas {
            cards: self.cards.clone(),
            home: self.home.clone(),
            free: self.free.clone(),
            lists: self.lists.clone(),
            by_sort: self.by_sort.clone(),
        }
    }

    /// Clones into the old allocations: copying a run into a game that held one allocates
    /// little.
    fn clone_from(&mut self, source: &Self) {
        self.cards.clone_from(&source.cards);
        self.home.clone_from(&source.home);
        self.free.clone_from(&source.free);
        for (a, b) in self.lists.iter_mut().zip(&source.lists) {
            a.clone_from(b);
        }
        self.by_sort.clone_from(&source.by_sort);
    }
}

impl PartialEq for Areas {
    fn eq(&self, other: &Areas) -> bool {
        AreaId::ALL.iter().all(|&a| self.get(a) == other.get(a))
    }
}

impl std::fmt::Debug for Areas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Areas");
        for a in AreaId::ALL {
            s.field(a.name(), &self.get(a));
        }
        s.finish()
    }
}

/// `0, 1, 2, ...`: the slots of [`Cards::from_slice`].
static IDENTITY: [u32; 1024] = {
    let mut ids = [0u32; 1024];
    let mut i = 0;
    while i < ids.len() {
        ids[i] = i as u32;
        i += 1;
    }
    ids
};

/// A card [`Areas::detach`] took out of its area, to [`Areas::attach`] to one. Dropping it
/// instead leaks its arena slot until the areas are rebuilt.
#[derive(Debug)]
pub struct Detached(u32);

/// An area's cards, in order (or any slice of cards, [`Cards::from_slice`]).
#[derive(Clone, Copy)]
pub struct Cards<'a> {
    slots: &'a [u32],
    arena: &'a [Card],
}

impl<'a> Cards<'a> {
    /// A slice of up to 1024 cards seen as an area's.
    pub fn from_slice(cards: &'a [Card]) -> Cards<'a> {
        Cards {
            slots: &IDENTITY[..cards.len()],
            arena: cards,
        }
    }

    pub fn len(self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(self) -> bool {
        self.slots.is_empty()
    }

    pub fn get(self, i: usize) -> Option<&'a Card> {
        self.slots.get(i).map(|&s| &self.arena[s as usize])
    }

    /// Card `i` (panics past the end).
    pub fn at(self, i: usize) -> &'a Card {
        &self.arena[self.slots[i] as usize]
    }

    pub fn first(self) -> Option<&'a Card> {
        self.get(0)
    }

    pub fn last(self) -> Option<&'a Card> {
        self.slots.last().map(|&s| &self.arena[s as usize])
    }

    pub fn iter(self) -> CardsIter<'a> {
        CardsIter {
            slots: self.slots.iter(),
            arena: self.arena,
        }
    }

    pub fn to_vec(self) -> Vec<Card> {
        self.iter().copied().collect()
    }

    /// The first `n` cards (all of them when there are fewer).
    pub fn take(self, n: usize) -> Cards<'a> {
        Cards {
            slots: &self.slots[..n.min(self.slots.len())],
            arena: self.arena,
        }
    }

    /// The cards' sort ids, in order.
    pub fn ids(self) -> Ids {
        let mut ids = Ids::from_slice(&[]);
        for c in self {
            ids.push(c.sort_id);
        }
        ids
    }
}

impl std::ops::Index<usize> for Cards<'_> {
    type Output = Card;

    fn index(&self, i: usize) -> &Card {
        self.at(i)
    }
}

impl<'a> IntoIterator for Cards<'a> {
    type Item = &'a Card;
    type IntoIter = CardsIter<'a>;

    fn into_iter(self) -> CardsIter<'a> {
        self.iter()
    }
}

impl std::fmt::Debug for Cards<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for Cards<'_> {
    fn eq(&self, other: &Cards<'_>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

/// [`Cards::iter`].
#[derive(Clone)]
pub struct CardsIter<'a> {
    slots: std::slice::Iter<'a, u32>,
    arena: &'a [Card],
}

impl<'a> Iterator for CardsIter<'a> {
    type Item = &'a Card;

    #[inline]
    fn next(&mut self) -> Option<&'a Card> {
        self.slots.next().map(|&s| &self.arena[s as usize])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.slots.size_hint()
    }
}

impl DoubleEndedIterator for CardsIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.slots.next_back().map(|&s| &self.arena[s as usize])
    }
}

impl ExactSizeIterator for CardsIter<'_> {}

impl std::iter::FusedIterator for CardsIter<'_> {}

/// An area's cards, in order, to change in place.
pub struct CardsMut<'a> {
    slots: &'a [u32],
    arena: &'a mut [Card],
}

impl<'a> CardsMut<'a> {
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&Card> {
        self.slots.get(i).map(|&s| &self.arena[s as usize])
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut Card> {
        self.slots.get(i).map(|&s| &mut self.arena[s as usize])
    }

    /// Card `i`, for as long as the area is borrowed (panics past the end).
    pub fn into_at(self, i: usize) -> &'a mut Card {
        &mut self.arena[self.slots[i] as usize]
    }

    /// `f` on each card, in order.
    pub fn for_each(self, mut f: impl FnMut(&mut Card)) {
        for &s in self.slots {
            f(&mut self.arena[s as usize]);
        }
    }
}

impl std::ops::Index<usize> for CardsMut<'_> {
    type Output = Card;

    fn index(&self, i: usize) -> &Card {
        &self.arena[self.slots[i] as usize]
    }
}

impl std::ops::IndexMut<usize> for CardsMut<'_> {
    fn index_mut(&mut self, i: usize) -> &mut Card {
        &mut self.arena[self.slots[i] as usize]
    }
}

/// `Areas::deck()`, `Areas::hand()`, ...: an area's cards by name.
macro_rules! area_accessors {
    ($($name:ident $name_mut:ident $id:ident),* $(,)?) => {
        impl Areas {
            $(
                pub fn $name(&self) -> Cards<'_> {
                    self.get(AreaId::$id)
                }

                pub fn $name_mut(&mut self) -> CardsMut<'_> {
                    self.get_mut(AreaId::$id)
                }
            )*
        }
    };
}

area_accessors! {
    deck deck_mut Deck,
    hand hand_mut Hand,
    discard discard_mut Discard,
    play play_mut Play,
    jokers jokers_mut Jokers,
    consumables consumables_mut Consumables,
    shop_jokers shop_jokers_mut ShopJokers,
    shop_vouchers shop_vouchers_mut ShopVouchers,
    shop_booster shop_booster_mut ShopBooster,
    pack_cards pack_cards_mut PackCards,
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

    pub fn get(&self, id: AreaId) -> Cards<'_> {
        Cards {
            slots: &self.lists[id as usize],
            arena: &self.cards,
        }
    }

    pub fn get_mut(&mut self, id: AreaId) -> CardsMut<'_> {
        CardsMut {
            slots: &self.lists[id as usize],
            arena: &mut self.cards,
        }
    }

    pub fn by_name(&self, name: &str) -> Option<Cards<'_>> {
        let k = Areas::NAMES.iter().position(|&n| n == name)?;
        Some(self.get(AreaId::ALL[k]))
    }

    /// The cards of the areas `which`, in no particular order (the arena's): for passes whose
    /// order doesn't matter. One sweep over the arena, however the cards are spread.
    pub fn unordered(&self, which: &[AreaId]) -> impl Iterator<Item = &Card> {
        let mask = which.iter().fold(0u16, |m, &a| m | 1 << a as u16);
        self.cards
            .iter()
            .zip(&self.home)
            .filter(move |&(_, &h)| h != FREE && mask & (1 << h) != 0)
            .map(|(c, _)| c)
    }

    /// Every card in an area, in no particular order (the arena's), to change in place: for
    /// changes that don't depend on the order.
    pub fn for_each_card_mut(&mut self, mut f: impl FnMut(AreaId, &mut Card)) {
        for (c, &h) in self.cards.iter_mut().zip(&self.home) {
            if h != FREE {
                f(AreaId::ALL[h as usize], c);
            }
        }
    }

    /// Put `card` in a slot of area `id` (not yet in its list).
    fn alloc(&mut self, id: AreaId, card: Card) -> u32 {
        let sort = card.sort_id as usize;
        let slot = match self.free.pop() {
            Some(s) => {
                self.cards[s as usize] = card;
                self.home[s as usize] = id as u8;
                s
            }
            None => {
                self.cards.push(card);
                self.home.push(id as u8);
                (self.cards.len() - 1) as u32
            }
        };
        if sort >= self.by_sort.len() {
            self.by_sort.resize(sort + 1, NO_SLOT);
        }
        debug_assert_eq!(
            self.by_sort[sort], NO_SLOT,
            "two cards in the areas with sort id {sort}"
        );
        self.by_sort[sort] = slot;
        slot
    }

    /// Free `slot` (already out of its list) and return its card.
    fn release(&mut self, slot: u32) -> Card {
        let card = self.cards[slot as usize];
        self.home[slot as usize] = FREE;
        self.free.push(slot);
        if let Some(s) = self.by_sort.get_mut(card.sort_id as usize)
            && *s == slot
        {
            *s = NO_SLOT;
        }
        card
    }

    /// Append `card` to area `id` as it is (no facing change, no alignment): for building
    /// areas.
    pub fn push(&mut self, id: AreaId, card: Card) {
        let slot = self.alloc(id, card);
        self.lists[id as usize].push(slot);
    }

    /// Insert `card` at position `i` of area `id` as it is.
    pub fn insert(&mut self, id: AreaId, i: usize, card: Card) {
        let slot = self.alloc(id, card);
        self.lists[id as usize].insert(i, slot);
    }

    /// Take card `i` out of area `id`.
    pub fn remove(&mut self, id: AreaId, i: usize) -> Card {
        let slot = self.lists[id as usize].remove(i);
        self.release(slot)
    }

    /// Replace area `id`'s cards with `cards`, as they are. Into empty areas (a new run's
    /// deck) the vector becomes the arena, so the cards don't move.
    pub fn set(&mut self, id: AreaId, cards: Vec<Card>) {
        if self.home.iter().all(|&h| h == FREE) {
            let n = cards.len();
            self.cards = cards;
            self.home.clear();
            self.home.resize(n, id as u8);
            self.free.clear();
            for l in &mut self.lists {
                l.clear();
            }
            self.lists[id as usize].extend(0..n as u32);
            self.by_sort.clear();
            let top = self.cards.iter().map(|c| c.sort_id as usize + 1).max();
            // Room for the sort ids of the cards a run makes later.
            self.by_sort.reserve(top.unwrap_or(0) + Ids::INLINE * 4);
            self.by_sort.resize(top.unwrap_or(0), NO_SLOT);
            for (slot, c) in self.cards.iter().enumerate() {
                let s = &mut self.by_sort[c.sort_id as usize];
                debug_assert_eq!(
                    *s, NO_SLOT,
                    "two cards in the areas with sort id {}",
                    c.sort_id
                );
                *s = slot as u32;
            }
            return;
        }
        self.clear(id);
        for c in cards {
            self.push(id, c);
        }
    }

    /// Take every card out of area `id`.
    pub fn clear(&mut self, id: AreaId) {
        let list = std::mem::take(&mut self.lists[id as usize]);
        for &slot in &list {
            self.release(slot);
        }
        let mut list = list;
        list.clear();
        self.lists[id as usize] = list;
    }

    /// Take every card out of every area.
    pub fn clear_all(&mut self) {
        self.cards.clear();
        self.home.clear();
        self.free.clear();
        for l in &mut self.lists {
            l.clear();
        }
        self.by_sort.clear();
    }

    /// Swap cards `i` and `j` of area `id`.
    pub fn swap(&mut self, id: AreaId, i: usize, j: usize) {
        self.lists[id as usize].swap(i, j);
    }

    /// Keep the cards of area `id` for which `keep` holds, in order; take the others out.
    pub fn retain(&mut self, id: AreaId, mut keep: impl FnMut(&Card) -> bool) {
        let mut list = std::mem::take(&mut self.lists[id as usize]);
        let mut dropped = Ids::from_slice(&[]);
        list.retain(|&s| {
            let k = keep(&self.cards[s as usize]);
            if !k {
                dropped.push(s);
            }
            k
        });
        for s in dropped {
            self.release(s);
        }
        self.lists[id as usize] = list;
    }

    /// Reorder area `id` so that position `k` holds the card that was at `order[k]`.
    pub fn reorder(&mut self, id: AreaId, order: &[usize]) {
        permute(&mut self.lists[id as usize], order);
    }

    /// Sort area `id` stably by `cmp`.
    pub fn sort_by(&mut self, id: AreaId, mut cmp: impl FnMut(&Card, &Card) -> std::cmp::Ordering) {
        let cards = &self.cards;
        self.lists[id as usize].sort_by(|&a, &b| cmp(&cards[a as usize], &cards[b as usize]));
    }

    /// Move the card at position `from` of area `id` to position `to`, shifting the cards
    /// between.
    pub fn move_card(&mut self, id: AreaId, from: usize, to: usize) {
        let list = &mut self.lists[id as usize];
        let slot = list.remove(from);
        list.insert(to, slot);
    }

    /// The area and index of the card with this sort id.
    pub fn locate(&self, sort_id: u32) -> Option<(AreaId, usize)> {
        let slot = *self.by_sort.get(sort_id as usize)?;
        if slot == NO_SLOT {
            return None;
        }
        let a = self.home[slot as usize] as usize;
        let i = self.lists[a].iter().position(|&s| s == slot)?;
        Some((AreaId::ALL[a], i))
    }

    fn slot_of(&self, sort_id: u32) -> Option<usize> {
        match self.by_sort.get(sort_id as usize) {
            Some(&s) if s != NO_SLOT => Some(s as usize),
            _ => None,
        }
    }

    pub fn card(&self, sort_id: u32) -> Option<&Card> {
        self.slot_of(sort_id).map(|s| &self.cards[s])
    }

    pub fn card_mut(&mut self, sort_id: u32) -> Option<&mut Card> {
        self.slot_of(sort_id).map(|s| &mut self.cards[s])
    }

    /// The area of the card with this sort id.
    pub fn area_of(&self, sort_id: u32) -> Option<AreaId> {
        self.slot_of(sort_id)
            .map(|s| AreaId::ALL[self.home[s] as usize])
    }

    /// [`Areas::card`] (`calculate_joker`'s cards are among the jokers and consumables).
    pub fn joker_card(&self, sort_id: u32) -> Option<&Card> {
        self.card(sort_id)
    }

    /// The sort ids of an area's cards, in order.
    pub fn ids(&self, id: AreaId) -> Ids {
        self.get(id).ids()
    }

    /// [`Areas::remove_card`], the card staying where it is in the arena until
    /// [`Areas::attach`] puts it in an area (meanwhile no area holds it, and only
    /// [`Areas::detached`] finds it).
    pub fn detach(&mut self, id: AreaId, sort_id: Option<u32>) -> Option<Detached> {
        let list = &self.lists[id as usize];
        let index = match sort_id {
            Some(s) => {
                let slot = self.slot_of(s)?;
                list.iter().position(|&x| x as usize == slot)?
            }
            None if matches!(id.kind(), AreaType::Deck | AreaType::Discard) => {
                list.len().checked_sub(1)?
            }
            None if list.is_empty() => return None,
            None => 0,
        };
        let slot = self.lists[id as usize].remove(index);
        self.home[slot as usize] = FREE;
        self.by_sort[self.cards[slot as usize].sort_id as usize] = NO_SLOT;
        Some(Detached(slot))
    }

    /// A detached card.
    pub fn detached(&self, d: &Detached) -> &Card {
        &self.cards[d.0 as usize]
    }

    /// [`Areas::emplace`] for a detached card.
    pub fn attach(&mut self, id: AreaId, d: Detached, front: bool, stay_flipped: bool) {
        let slot = d.0;
        let card = &mut self.cards[slot as usize];
        if id.face_down() {
            card.facing = Facing::Back;
        } else if card.facing == Facing::Back && !stay_flipped {
            card.facing = Facing::Front;
        }
        let sort = card.sort_id as usize;
        self.home[slot as usize] = id as u8;
        debug_assert_eq!(self.by_sort[sort], NO_SLOT);
        self.by_sort[sort] = slot;
        let list = &mut self.lists[id as usize];
        if front || id.kind() == AreaType::Deck {
            list.insert(0, slot);
        } else {
            list.push(slot);
        }
        self.align(id);
    }

    /// `CardArea:emplace(card, location, stay_flipped)` (cardarea.lua:32-64), minus UI.
    pub fn emplace(&mut self, id: AreaId, mut card: Card, front: bool, stay_flipped: bool) {
        if id.face_down() {
            card.facing = Facing::Back;
        } else if card.facing == Facing::Back && !stay_flipped {
            card.facing = Facing::Front;
        }
        let slot = self.alloc(id, card);
        let list = &mut self.lists[id as usize];
        if front || id.kind() == AreaType::Deck {
            list.insert(0, slot);
        } else {
            list.push(slot);
        }
        self.align(id);
    }

    /// `emplace` each card in turn, in one move (a deck takes each new card in front).
    pub fn emplace_all(&mut self, id: AreaId, new: Vec<Card>) {
        let mut slots = Vec::with_capacity(new.len());
        for mut card in new {
            if id.face_down() {
                card.facing = Facing::Back;
            } else if card.facing == Facing::Back {
                card.facing = Facing::Front;
            }
            slots.push(self.alloc(id, card));
        }
        let list = &mut self.lists[id as usize];
        if id.kind() == AreaType::Deck {
            list.splice(0..0, slots.into_iter().rev());
        } else {
            list.extend(slots);
        }
        self.align(id);
    }

    /// `draw_card`'s moves from `from` to `to`, made at once: every card lands as
    /// [`Areas::emplace`] puts it (none stays flipped) and the result is that of moving them one
    /// at a time. Only slots move. Returns the sort ids moved, in the order they moved.
    pub fn draw_cards(&mut self, from: AreaId, to: AreaId, draw: Draw<'_>) -> Ids {
        debug_assert_ne!(from, to);
        let mut src = std::mem::take(&mut self.lists[from as usize]);
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
                    if let Some(slot) = self.slot_of(id)
                        && let Some(p) = src.iter().position(|&s| s as usize == slot)
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
        let mut moved = Ids::from_slice(&[]);
        for &p in &order {
            let slot = src[p as usize];
            let c = &mut self.cards[slot as usize];
            moved.push(c.sort_id);
            if to.face_down() {
                c.facing = Facing::Back;
            } else if c.facing == Facing::Back {
                c.facing = Facing::Front;
            }
            self.home[slot as usize] = to as u8;
        }
        let dst = &mut self.lists[to as usize];
        if to.kind() == AreaType::Deck {
            // Each card goes in front of the ones before it.
            dst.splice(0..0, order.iter().rev().map(|&p| src[p as usize]));
        } else {
            dst.extend(order.iter().map(|&p| src[p as usize]));
        }
        let mut at = 0;
        src.retain(|_| {
            at += 1;
            !taken[at - 1]
        });
        self.lists[from as usize] = src;
        self.align(to);
        moved
    }

    /// `CardArea:remove_card(card)` (cardarea.lua:66-92): the given card, or the area's next
    /// card (the last one of a deck or discard pile, else the first).
    pub fn remove_card(&mut self, id: AreaId, sort_id: Option<u32>) -> Option<Card> {
        let list = &self.lists[id as usize];
        let index = match sort_id {
            Some(s) => {
                let slot = self.slot_of(s)?;
                list.iter().position(|&x| x as usize == slot)?
            }
            None if matches!(id.kind(), AreaType::Deck | AreaType::Discard) => {
                list.len().checked_sub(1)?
            }
            None if list.is_empty() => return None,
            None => 0,
        };
        Some(self.remove(id, index))
    }

    /// `CardArea:align_cards()` (cardarea.lua:410-552) where it changes order: positions follow
    /// the index everywhere except that pinned jokers sit leftmost, newest first (line 528).
    pub fn align(&mut self, id: AreaId) {
        if id.kind() == AreaType::Joker && self.get(id).iter().any(|c| c.pinned) {
            let cards = &self.cards;
            let list = &mut self.lists[id as usize];
            let mut keyed: Vec<((bool, i64, usize), u32)> = list
                .iter()
                .enumerate()
                .map(|(k, &s)| {
                    let c = &cards[s as usize];
                    let key = if c.pinned {
                        (false, -i64::from(c.sort_id), k)
                    } else {
                        (true, 0, k)
                    };
                    (key, s)
                })
                .collect();
            keyed.sort_unstable_by_key(|&(key, _)| key);
            for (slot, (_, s)) in list.iter_mut().zip(keyed) {
                *slot = s;
            }
        }
    }

    /// `CardArea:shuffle(seed)` (cardarea.lua:572): `pseudoshuffle(self.cards, pseudoseed(seed))`.
    /// The next frame's `align_cards` puts pinned jokers back in front (cardarea.lua:528; G200).
    pub fn shuffle(&mut self, id: AreaId, rng: &mut Rng, key: &str) {
        let seed = rng.seed(key);
        let cards = &self.cards;
        let list = &mut self.lists[id as usize];
        // pseudoshuffle sorts by sort_id first when the entries carry one (misc_functions.lua),
        // then swaps.
        if !list.is_sorted_by_key(|&s| cards[s as usize].sort_id) {
            // Sort ids are unique, so any sort gives the stable sort's order. The sort runs on
            // (sort id, slot) words, away from the cards.
            let mut keyed_inline = [0u64; 256];
            let mut keyed_spill = Vec::new();
            let keyed: &mut [u64] = if list.len() <= keyed_inline.len() {
                &mut keyed_inline[..list.len()]
            } else {
                keyed_spill.resize(list.len(), 0);
                &mut keyed_spill
            };
            for (k, &s) in keyed.iter_mut().zip(list.iter()) {
                *k = (u64::from(cards[s as usize].sort_id) << 32) | u64::from(s);
            }
            keyed.sort_unstable();
            for (s, &k) in list.iter_mut().zip(keyed.iter()) {
                *s = k as u32;
            }
        }
        shuffle(list, seed);
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
                    areas.push(a, card(next_id, facing));
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
                batched.check();
                single.check();
            }
        }
    }

    impl Areas {
        /// The arena's bookkeeping agrees with the lists: every listed slot is used once, by
        /// its area, and found by its sort id; every other slot is free.
        fn check(&self) {
            let mut listed = vec![false; self.cards.len()];
            for (a, list) in self.lists.iter().enumerate() {
                for &s in list {
                    assert!(
                        !std::mem::replace(&mut listed[s as usize], true),
                        "slot {s} twice"
                    );
                    assert_eq!(self.home[s as usize] as usize, a);
                    assert_eq!(self.by_sort[self.cards[s as usize].sort_id as usize], s);
                }
            }
            for (s, &l) in listed.iter().enumerate() {
                assert_eq!(l, self.home[s] != FREE, "slot {s}");
                assert_eq!(l, !self.free.contains(&(s as u32)), "slot {s}");
            }
            let mapped = self.by_sort.iter().filter(|&&s| s != NO_SLOT).count();
            assert_eq!(mapped, listed.iter().filter(|&&l| l).count());
        }
    }

    /// The areas as plain vectors of cards, moved the way they were before the arena.
    #[derive(Clone, Debug, PartialEq)]
    struct Model([Vec<Card>; 10]);

    impl Model {
        fn of(areas: &Areas) -> Model {
            Model(AreaId::ALL.map(|a| areas.get(a).to_vec()))
        }

        fn locate(&self, sort_id: u32) -> Option<(AreaId, usize)> {
            AreaId::ALL.into_iter().find_map(|a| {
                self.0[a as usize]
                    .iter()
                    .position(|c| c.sort_id == sort_id)
                    .map(|i| (a, i))
            })
        }

        fn remove_card(&mut self, id: AreaId, sort_id: Option<u32>) -> Option<Card> {
            let cards = &mut self.0[id as usize];
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

        fn emplace(&mut self, id: AreaId, mut card: Card, front: bool, stay_flipped: bool) {
            if id.face_down() {
                card.facing = Facing::Back;
            } else if card.facing == Facing::Back && !stay_flipped {
                card.facing = Facing::Front;
            }
            if front || id.kind() == AreaType::Deck {
                self.0[id as usize].insert(0, card);
            } else {
                self.0[id as usize].push(card);
            }
            self.align(id);
        }

        fn align(&mut self, id: AreaId) {
            let cards = &mut self.0[id as usize];
            if id.kind() == AreaType::Joker && cards.iter().any(|c| c.pinned) {
                let mut keyed: Vec<(usize, Card)> = cards.iter().copied().enumerate().collect();
                keyed.sort_by_key(|(k, c)| {
                    if c.pinned {
                        (false, -i64::from(c.sort_id), *k)
                    } else {
                        (true, 0, *k)
                    }
                });
                *cards = keyed.into_iter().map(|(_, c)| c).collect();
            }
        }

        fn shuffle(&mut self, id: AreaId, seed: f64) {
            let cards = &mut self.0[id as usize];
            let mut order: Vec<usize> = (0..cards.len()).collect();
            order.sort_by_key(|&i| cards[i].sort_id);
            crate::rng::shuffle(&mut order, seed);
            let old = cards.clone();
            for (slot, &k) in cards.iter_mut().zip(&order) {
                *slot = old[k];
            }
            self.align(id);
        }
    }

    /// Random moves on the arena match the same moves on vectors of cards, and the arena's
    /// bookkeeping holds throughout.
    #[test]
    fn the_arena_moves_cards_as_vectors_do() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut below = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n.max(1)
        };
        let mut rng = Rng::new("ARENA");
        for round in 0..300 {
            let mut areas = Areas::default();
            let mut next_id = 1;
            let deck: Vec<Card> = (0..below(60))
                .map(|_| {
                    next_id += 1;
                    card(next_id - 1, Facing::Back)
                })
                .collect();
            areas.set(AreaId::Deck, deck);
            let mut model = Model::of(&areas);
            for _ in 0..60 {
                let a = AreaId::ALL[below(10) as usize];
                let b = AreaId::ALL[below(10) as usize];
                let pick = |model: &Model, below: &mut dyn FnMut(u64) -> u64| {
                    let cards = &model.0[a as usize];
                    (!cards.is_empty() && below(4) != 0)
                        .then(|| cards[below(cards.len() as u64) as usize].sort_id)
                };
                let op = below(9);
                match op {
                    0 => {
                        let mut c = card(next_id, Facing::Back);
                        c.pinned = below(3) == 0;
                        next_id += 1;
                        let front = below(2) == 0;
                        let flip = below(2) == 0;
                        areas.emplace(a, c, front, flip);
                        model.emplace(a, c, front, flip);
                    }
                    1 => {
                        let id = pick(&model, &mut below);
                        assert_eq!(areas.remove_card(a, id), model.remove_card(a, id));
                    }
                    2 if a != b => {
                        let id = pick(&model, &mut below);
                        let flip = below(2) == 0;
                        let d = areas.detach(a, id);
                        let m = model.remove_card(a, id);
                        assert_eq!(d.as_ref().map(|d| *areas.detached(d)), m);
                        if let (Some(d), Some(m)) = (d, m) {
                            areas.attach(b, d, false, flip);
                            model.emplace(b, m, false, flip);
                        }
                    }
                    3 => {
                        let mut same = rng.clone();
                        areas.shuffle(a, &mut rng, "arena");
                        model.shuffle(a, same.seed("arena"));
                    }
                    4 if !model.0[a as usize].is_empty() => {
                        let n = model.0[a as usize].len();
                        let (i, j) = (below(n as u64) as usize, below(n as u64) as usize);
                        areas.swap(a, i, j);
                        model.0[a as usize].swap(i, j);
                    }
                    5 => {
                        let k = below(4) as u32;
                        areas.retain(a, |c| c.sort_id % 4 != k);
                        model.0[a as usize].retain(|c| c.sort_id % 4 != k);
                    }
                    6 if a != b => {
                        let n = below(6) as usize;
                        let moved = areas.draw_cards(a, b, Draw::Next(n));
                        for _ in 0..moved.len() {
                            let c = model.remove_card(a, None).expect("moved");
                            model.emplace(b, c, false, false);
                        }
                        // `draw_cards` aligns the target even when nothing moved.
                        model.align(b);
                    }
                    7 if !model.0[a as usize].is_empty() => {
                        let n = model.0[a as usize].len();
                        let (i, j) = (below(n as u64) as usize, below(n as u64) as usize);
                        areas.move_card(a, i, j);
                        let c = model.0[a as usize].remove(i);
                        model.0[a as usize].insert(j, c);
                    }
                    _ => {
                        let id = 1 + below(next_id as u64) as u32;
                        assert_eq!(areas.locate(id), model.locate(id));
                        assert_eq!(
                            areas.card(id),
                            model.locate(id).map(|(a, i)| &model.0[a as usize][i])
                        );
                    }
                }
                assert!(Model::of(&areas) == model, "round {round}, move {op}");
                areas.check();
            }
            let copy = areas.clone();
            assert_eq!(copy, areas);
            let mut into = Areas::default();
            into.clone_from(&areas);
            into.check();
            assert_eq!(into, areas);
        }
    }
}
