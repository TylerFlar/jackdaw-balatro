//! The Gym wrapper's flat action table (`BalatroGymnasiumEnv._enumerate_actions`): every legal
//! factored action in type order, card selections enumerated by size then lexicographically,
//! and subsampled to budgets with a seeded xoshiro256** generator. `jackdaw/env/sampler.py` is
//! the same generator in Python, so both tables agree draw for draw.

use jackdaw_core::game::Game;

use crate::action::{
    ActionType, FactoredAction, MAX_SELECTION, Selection, Targets, consumable_target_info,
};
use crate::mask::ActionMask;

/// The size of the Gym `Discrete` action space: the table holds at most this many actions.
pub const MAX_ACTIONS: usize = 500;
/// Card selections kept per action type (and per entity).
pub const CARD_COMBO_BUDGET: usize = 200;

/// The xoshiro256** generator (Blackman and Vigna), seeded from one integer through SplitMix64.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    pub fn seed_from_u64(seed: u64) -> Xoshiro256 {
        let mut x = seed;
        let mut s = [0u64; 4];
        for word in &mut s {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *word = z ^ (z >> 31);
        }
        Xoshiro256 { s }
    }

    pub fn from_state(s: [u64; 4]) -> Xoshiro256 {
        Xoshiro256 { s }
    }

    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// A uniform integer in `[0, bound)` (Lemire's multiply-and-reject), `bound >= 1`.
    ///
    /// The rejection threshold `2^64 mod bound` is below `bound`, so a draw whose low word is
    /// at least `bound` is accepted without computing it: the division runs only on the rare
    /// draws that could be rejected, and the accepted values are the same as always dividing.
    #[inline]
    pub fn below(&mut self, bound: u64) -> u64 {
        let mut m = u128::from(self.next_u64()) * u128::from(bound);
        if (m as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(bound);
            }
        }
        (m >> 64) as u64
    }
}

/// Floyd's draws for a uniform `k`-subset of `0..n` (`k <= n`), left in `picked` as a bitset of
/// the kept indices (one bit per index, none past `n`). It draws `min(k, n - k)` times: it picks
/// the kept indices, or the dropped ones when fewer are dropped, then flips the bits.
fn subsample_bits(n: usize, k: usize, rng: &mut Xoshiro256, picked: &mut Vec<u64>) {
    let m = k.min(n - k);
    picked.clear();
    picked.resize(n.div_ceil(64), 0);
    for j in n - m..n {
        let t = rng.below(j as u64 + 1) as usize;
        let slot = if picked[t / 64] & (1 << (t % 64)) != 0 {
            j
        } else {
            t
        };
        picked[slot / 64] |= 1 << (slot % 64);
    }
    if m != k {
        for (w, bits) in picked.iter_mut().enumerate() {
            *bits = !*bits;
            let base = w * 64;
            if base + 64 > n {
                *bits &= (1u64 << (n - base)) - 1;
            }
        }
    }
}

/// The set bits of `words`, lowest first, as indices.
fn set_bits(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(w, &word)| {
        let mut bits = word;
        std::iter::from_fn(move || {
            (bits != 0).then(|| {
                let i = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                i
            })
        })
    })
}

/// The index of the `r`-th set bit (from 0) of `words`; it must exist.
fn nth_set_bit(words: &[u64], mut r: usize) -> usize {
    for (w, &word) in words.iter().enumerate() {
        let ones = word.count_ones() as usize;
        if r < ones {
            let mut bits = word;
            for _ in 0..r {
                bits &= bits - 1;
            }
            return w * 64 + bits.trailing_zeros() as usize;
        }
        r -= ones;
    }
    unreachable!("fewer set bits than asked for")
}

/// `k` distinct indices of `0..n`, ascending, uniformly at random (`k <= n`), into `out`. Floyd's
/// algorithm draws `min(k, n - k)` times: it picks the kept indices, or the dropped ones when
/// fewer are dropped. `picked` is scratch space (a bitset, one bit per index).
pub fn subsample_indices(
    n: usize,
    k: usize,
    rng: &mut Xoshiro256,
    picked: &mut Vec<u64>,
    out: &mut Vec<usize>,
) {
    subsample_bits(n, k, rng, picked);
    out.clear();
    out.reserve(k);
    out.extend(set_bits(picked));
}

/// Cerulean Bell's forced card for one enumeration, `(card, below)`: a selection without the
/// card is kept only when it holds fewer than `below` cards. PlayHand and Discard pass `0` (every
/// selection holds the card); a consumable passes its highlight limit, since its use highlights
/// the forced card as well as the selection.
type Forced = Option<(usize, usize)>;

/// The card selections of one enumeration: which cards, how many, which card each must hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComboKey {
    /// The legal cards, as bits by hand index.
    legal: u128,
    lower: usize,
    upper: usize,
    forced: Forced,
}

/// One cache slot: its key (`None` for a hand too large to key, 128 cards or more, which no
/// lookup matches), its selections, and the last enumeration that used it.
#[derive(Clone, Debug, Default)]
struct ComboSlot {
    key: Option<ComboKey>,
    selections: Vec<Selection>,
    used: u64,
}

/// Selections already enumerated (a table asks for the same ones for PlayHand and Discard, and
/// the hand rarely changes size between steps). The action table refers to its slots, so a slot
/// the current enumeration used is never replaced: the cache grows instead. A slot's selections
/// depend only on its key, so which slot is replaced never changes a table.
#[derive(Clone, Debug, Default)]
struct ComboCache {
    slots: Vec<ComboSlot>,
    next: usize,
    /// The current enumeration (slots it used carry this stamp).
    generation: u64,
}

const COMBO_CACHE_SIZE: usize = 8;

impl ComboCache {
    /// The slot holding the selections of `lower..=upper` of `legal` (by size, then
    /// lexicographically) that `forced` keeps, enumerated once per key.
    fn slot(&mut self, legal: &[usize], lower: usize, upper: usize, forced: Forced) -> usize {
        let key = legal
            .iter()
            .try_fold(0u128, |acc, &i| (i < 128).then(|| acc | (1u128 << i)))
            .map(|bits| ComboKey {
                legal: bits,
                lower,
                upper,
                forced,
            });
        if key.is_some()
            && let Some(i) = self.slots.iter().position(|s| s.key == key)
        {
            self.slots[i].used = self.generation;
            return i;
        }
        let free = (0..self.slots.len())
            .map(|j| (self.next + j) % self.slots.len())
            .find(|&i| self.slots[i].used != self.generation);
        let slot = match free {
            Some(i) if self.slots.len() >= COMBO_CACHE_SIZE => {
                self.next = (i + 1) % self.slots.len();
                i
            }
            _ => {
                self.slots.push(ComboSlot::default());
                self.slots.len() - 1
            }
        };
        let s = &mut self.slots[slot];
        s.key = key;
        s.used = self.generation;
        s.selections.clear();
        combinations_into(legal, lower, upper, forced, &mut s.selections);
        slot
    }
}

/// The `lower..=upper`-card selections of `legal`, by size then lexicographically, those
/// `forced` keeps only, appended to `out`.
fn combinations_into(
    legal: &[usize],
    lower: usize,
    upper: usize,
    forced: Forced,
    out: &mut Vec<Selection>,
) {
    let n = legal.len();
    let mut idx = [0usize; MAX_SELECTION];
    for k in lower..=upper.min(MAX_SELECTION) {
        for (j, slot) in idx.iter_mut().take(k).enumerate() {
            *slot = j;
        }
        loop {
            let mut sel = Selection::default();
            for &p in &idx[..k] {
                sel.push(legal[p]);
            }
            if forced.is_none_or(|(r, below)| k < below || sel.as_slice().contains(&(r as u8))) {
                out.push(sel);
            }
            let mut i = k;
            while i > 0 && idx[i - 1] == i - 1 + n - k {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            idx[i - 1] += 1;
            for j in i..k {
                idx[j] = idx[j - 1] + 1;
            }
        }
    }
}

/// A run of the table's actions.
#[derive(Clone, Copy, Debug)]
enum Group {
    One(FactoredAction),
    /// Card actions of one type (and entity): every selection of a cache slot, or (`words`, a
    /// range of the table's bit arena) the subsample whose indices are its set bits.
    Cards {
        action_type: ActionType,
        entity_target: Option<u32>,
        slot: usize,
        words: Option<(usize, usize)>,
    },
}

/// The flat action table and its scratch buffers, reused across enumerations. Card actions are
/// kept as references to the cached selections (and subsample bitsets) and built when read: a
/// step reads one action of the hundreds a table holds.
#[derive(Clone, Debug, Default)]
pub struct ActionTable {
    /// Each group with the table index of its first action.
    groups: Vec<(usize, Group)>,
    len: usize,
    bits: Vec<u64>,
    legal_cards: Vec<usize>,
    combos: ComboCache,
    picked: Vec<u64>,
    chosen: Vec<usize>,
    spill: Vec<FactoredAction>,
}

impl ActionTable {
    pub fn new() -> ActionTable {
        ActionTable::default()
    }

    /// The actions, in table order.
    pub fn actions(&self) -> Vec<FactoredAction> {
        let mut out = Vec::with_capacity(self.len);
        for (_, g) in &self.groups {
            match *g {
                Group::One(a) => out.push(a),
                Group::Cards {
                    action_type,
                    entity_target,
                    slot,
                    words,
                } => {
                    let sels = &self.combos.slots[slot].selections;
                    let make = |sel: Selection| FactoredAction {
                        action_type,
                        entity_target,
                        card_target: Some(sel),
                    };
                    match words {
                        None => out.extend(sels.iter().map(|&s| make(s))),
                        Some((a, b)) => {
                            out.extend(set_bits(&self.bits[a..b]).map(|i| make(sels[i])));
                        }
                    }
                }
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Action `i` of the table.
    pub fn get(&self, i: usize) -> Option<FactoredAction> {
        if i >= self.len {
            return None;
        }
        let g = self.groups.partition_point(|&(start, _)| start <= i) - 1;
        let (start, group) = self.groups[g];
        Some(match group {
            Group::One(a) => a,
            Group::Cards {
                action_type,
                entity_target,
                slot,
                words,
            } => {
                let sels = &self.combos.slots[slot].selections;
                let k = match words {
                    None => i - start,
                    Some((a, b)) => nth_set_bit(&self.bits[a..b], i - start),
                };
                FactoredAction {
                    action_type,
                    entity_target,
                    card_target: Some(sels[k]),
                }
            }
        })
    }

    /// Empty the table (an episode ended).
    pub fn clear(&mut self) {
        self.groups.clear();
        self.len = 0;
    }

    fn push(&mut self, group: Group, count: usize) {
        if count > 0 {
            self.groups.push((self.len, group));
            self.len += count;
        }
    }

    /// `_enumerate_actions(mask, info)` for `game`: the legal actions of the mask, the
    /// entity-and-card types targeting the consumables (UseConsumable) and the pack cards
    /// (PickPackCard).
    pub fn enumerate(&mut self, mask: &ActionMask, game: &Game, rng: &mut Xoshiro256) {
        let areas = &game.gs.areas;
        self.enumerate_with(
            &mask.type_mask,
            &mask.card_mask,
            |t| mask.entity_mask(t),
            (mask.min_card_select, mask.max_card_select),
            mask.forced_card,
            |t, e| {
                let source = match t {
                    ActionType::UseConsumable => areas.consumables(),
                    _ => areas.pack_cards(),
                };
                source.get(e).map(consumable_target_info)
            },
            rng,
        );
    }

    /// The enumeration for any mask: `entity(t)` is `t`'s entity mask, `target_info(t, e)` the
    /// `(min, max, needs)` targeting of entity `e` of an entity-and-card type (`None` past the
    /// end of its source).
    #[allow(clippy::too_many_arguments)]
    pub fn enumerate_with<'m>(
        &mut self,
        type_mask: &[bool],
        card_mask: &[bool],
        entity: impl Fn(ActionType) -> Option<&'m [bool]>,
        (min_select, max_select): (usize, usize),
        forced: Option<usize>,
        target_info: impl Fn(ActionType, usize) -> Option<(usize, usize, bool)>,
        rng: &mut Xoshiro256,
    ) {
        self.clear();
        self.bits.clear();
        self.combos.generation += 1;
        self.legal_cards.clear();
        self.legal_cards.extend(
            card_mask
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| i),
        );
        for (i, &allowed) in type_mask.iter().enumerate() {
            let Some(t) = ActionType::from_index(i) else {
                continue;
            };
            if !allowed {
                continue;
            }
            match t.targets() {
                Targets::None => self.push(Group::One(FactoredAction::new(t)), 1),
                Targets::Entity => {
                    if let Some(em) = entity(t) {
                        for (e, &ok) in em.iter().enumerate() {
                            if ok {
                                self.push(Group::One(FactoredAction::with_entity(t, e)), 1);
                            }
                        }
                    }
                }
                Targets::Cards => {
                    let forced = forced.map(|r| (r, 0));
                    self.card_actions(t, None, min_select, max_select, forced, rng);
                }
                Targets::EntityCards => {
                    if let Some(em) = entity(t) {
                        for (e, &ok) in em.iter().enumerate() {
                            if !ok {
                                continue;
                            }
                            match target_info(t, e) {
                                Some((lo, hi, true)) => {
                                    // The use highlights the forced card first, then the
                                    // selection: one card more when it leaves the card out
                                    // (`consumable_highlight`, #49).
                                    let forced = forced.map(|r| (r, hi.min(max_select)));
                                    self.card_actions(t, Some(e), lo, hi, forced, rng);
                                }
                                Some((_, _, false)) => {
                                    self.push(Group::One(FactoredAction::with_entity(t, e)), 1);
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        }
        if self.len > MAX_ACTIONS {
            // Rare: build the actions, keep a subsample, and hold those one by one.
            self.spill = self.actions();
            subsample_indices(
                self.spill.len(),
                MAX_ACTIONS,
                rng,
                &mut self.picked,
                &mut self.chosen,
            );
            self.clear();
            for k in 0..self.chosen.len() {
                let a = self.spill[self.chosen[k]];
                self.push(Group::One(a), 1);
            }
        }
    }

    /// `_card_actions`: the selections of `min_select..=max_select` legal cards (capped at the
    /// number of legal cards), by size then lexicographically, those `forced` keeps only,
    /// subsampled to [`CARD_COMBO_BUDGET`].
    fn card_actions(
        &mut self,
        t: ActionType,
        entity: Option<usize>,
        min_select: usize,
        max_select: usize,
        forced: Forced,
        rng: &mut Xoshiro256,
    ) {
        let n = self.legal_cards.len();
        let upper = n.min(max_select);
        let lower = min_select.min(upper);
        let slot = self.combos.slot(&self.legal_cards, lower, upper, forced);
        let count = self.combos.slots[slot].selections.len();
        let entity_target = entity.map(|e| e as u32);
        if count <= CARD_COMBO_BUDGET {
            let group = Group::Cards {
                action_type: t,
                entity_target,
                slot,
                words: None,
            };
            self.push(group, count);
        } else {
            subsample_bits(count, CARD_COMBO_BUDGET, rng, &mut self.picked);
            let a = self.bits.len();
            self.bits.extend_from_slice(&self.picked);
            let group = Group::Cards {
                action_type: t,
                entity_target,
                slot,
                words: Some((a, self.bits.len())),
            };
            self.push(group, CARD_COMBO_BUDGET);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xoshiro_matches_the_reference() {
        // xoshiro256** from state [1, 2, 3, 4] (the reference implementation's output).
        let mut r = Xoshiro256::from_state([1, 2, 3, 4]);
        let got: Vec<u64> = (0..4).map(|_| r.next_u64()).collect();
        assert_eq!(got, [11520, 0, 1509978240, 1215971899390074240]);
    }

    #[test]
    fn subsample_is_sorted_and_sized() {
        let mut rng = Xoshiro256::seed_from_u64(7);
        let (mut picked, mut out) = (Vec::new(), Vec::new());
        for (n, k) in [(218, 200), (650, 500), (10, 3), (5, 5), (5, 0)] {
            subsample_indices(n, k, &mut rng, &mut picked, &mut out);
            assert_eq!(out.len(), k);
            assert!(out.windows(2).all(|w| w[0] < w[1]));
            assert!(out.iter().all(|&i| i < n));
        }
    }

    /// Enumerate with every type allowed, `cards` legal hand cards and `entities` entities per
    /// entity type, each entity-and-card type taking `lo..=hi` cards.
    fn table_of(cards: usize, entities: usize, (lo, hi): (usize, usize), seed: u64) -> ActionTable {
        let mut table = ActionTable::new();
        let mut rng = Xoshiro256::seed_from_u64(seed);
        let em = vec![true; entities];
        table.enumerate_with(
            &[true; crate::action::NUM_ACTION_TYPES],
            &vec![true; cards],
            |_| Some(&em[..]),
            (1, 5),
            None,
            |_, _| Some((lo, hi, true)),
            &mut rng,
        );
        table
    }

    #[test]
    fn get_reads_the_built_actions() {
        // Whole groups, subsampled groups (more than CARD_COMBO_BUDGET selections), a table cut
        // to MAX_ACTIONS, and enough entity-and-card groups to outgrow the selection cache.
        for (cards, entities, targets) in [
            (3, 1, (1, 1)),
            (8, 2, (1, 2)),
            (10, 5, (1, 3)),
            (12, 0, (1, 1)),
        ] {
            for seed in 0..4 {
                let table = table_of(cards, entities, targets, seed);
                let actions = table.actions();
                assert_eq!(actions.len(), table.len());
                assert!(table.len() <= MAX_ACTIONS);
                for (i, a) in actions.iter().enumerate() {
                    assert_eq!(table.get(i), Some(*a), "{cards} cards, action {i}");
                }
                assert_eq!(table.get(table.len()), None);
            }
        }
    }

    #[test]
    fn many_card_groups_keep_their_selections() {
        // Ten entities with ten different selection sizes, plus PlayHand and Discard: more
        // distinct selection sets in one table than the cache has slots. Each group must still
        // read its own selections.
        let range = |e: usize| (1 + e % 2, (1 + e % 2).max(1 + e % 5));
        let mut table = ActionTable::new();
        let mut rng = Xoshiro256::seed_from_u64(5);
        let em = [true; 10];
        let mut types = [false; crate::action::NUM_ACTION_TYPES];
        types[ActionType::PlayHand.index()] = true;
        types[ActionType::UseConsumable.index()] = true;
        table.enumerate_with(
            &types,
            &[true; 6],
            |_| Some(&em[..]),
            (1, 5),
            None,
            |_, e| {
                let (lo, hi) = range(e);
                Some((lo, hi, true))
            },
            &mut rng,
        );
        let actions = table.actions();
        let mut entities = std::collections::BTreeSet::new();
        for a in &actions {
            let sel = a.card_target.expect("card actions only");
            let (lo, hi) = match a.entity_target {
                Some(e) => range(e as usize),
                None => (1, 5),
            };
            assert!((lo..=hi).contains(&sel.len()), "{a:?}");
            assert!(sel.as_slice().windows(2).all(|w| w[0] < w[1]));
            entities.insert(a.entity_target);
        }
        assert_eq!(entities.len(), 11);
        for (i, a) in actions.iter().enumerate() {
            assert_eq!(table.get(i), Some(*a));
        }
    }

    #[test]
    fn a_forced_card_joins_a_consumable_highlight() {
        // Cerulean Bell's card (hand index 2 of 6) is highlighted first when a consumable is
        // used, then the selection: a selection without it holds one card more, and the
        // highlight takes at most `max_highlighted` cards (#49). Plays and discards must hold it.
        let mut types = [false; crate::action::NUM_ACTION_TYPES];
        types[ActionType::PlayHand.index()] = true;
        types[ActionType::UseConsumable.index()] = true;
        let targets = [(1, 1), (1, 2), (2, 2)];
        let mut table = ActionTable::new();
        let mut rng = Xoshiro256::seed_from_u64(9);
        table.enumerate_with(
            &types,
            &[true; 6],
            |_| Some(&[true; 3][..]),
            (1, 5),
            Some(2),
            |_, e| Some((targets[e].0, targets[e].1, true)),
            &mut rng,
        );
        let actions = table.actions();
        let sels = |t: ActionType, e: Option<u32>| -> Vec<Vec<u8>> {
            actions
                .iter()
                .filter(|a| a.action_type == t && a.entity_target == e)
                .map(|a| a.card_target.unwrap().as_slice().to_vec())
                .collect()
        };
        // The Tower (1, 1): the forced card only.
        assert_eq!(sels(ActionType::UseConsumable, Some(0)), [vec![2]]);
        // Strength (1, 2): any one card, or the forced card and one other.
        let strength = sels(ActionType::UseConsumable, Some(1));
        let mut want: Vec<Vec<u8>> = (0..6).map(|i| vec![i]).collect();
        want.extend([vec![0, 2], vec![1, 2], vec![2, 3], vec![2, 4], vec![2, 5]]);
        assert_eq!(strength, want);
        // Death (2, 2): the forced card and one other.
        let death = sels(ActionType::UseConsumable, Some(2));
        assert_eq!(death, [[0, 2], [1, 2], [2, 3], [2, 4], [2, 5]]);
        // PlayHand: every selection holds the forced card, as before.
        let plays = sels(ActionType::PlayHand, None);
        assert_eq!(plays.len(), 1 + 5 + 10 + 10 + 5);
        assert!(plays.iter().all(|s| s.contains(&2)));
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = Xoshiro256::seed_from_u64(1);
        for bound in [1u64, 2, 3, 7, 1000, u64::MAX] {
            for _ in 0..100 {
                assert!(rng.below(bound) < bound);
            }
        }
    }
}
