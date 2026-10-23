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
    pub fn below(&mut self, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let m = u128::from(self.next_u64()) * u128::from(bound);
            if (m as u64) >= threshold {
                return (m >> 64) as u64;
            }
        }
    }
}

/// `k` distinct indices of `0..n`, ascending, uniformly at random (`k <= n`), into `out`. Floyd's
/// algorithm draws `min(k, n - k)` times: it picks the kept indices, or the dropped ones when
/// fewer are dropped. `picked` is scratch space.
pub fn subsample_indices(
    n: usize,
    k: usize,
    rng: &mut Xoshiro256,
    picked: &mut Vec<bool>,
    out: &mut Vec<usize>,
) {
    let m = k.min(n - k);
    picked.clear();
    picked.resize(n, false);
    for j in n - m..n {
        let t = rng.below(j as u64 + 1) as usize;
        if picked[t] {
            picked[j] = true;
        } else {
            picked[t] = true;
        }
    }
    out.clear();
    let keep_picked = m == k;
    out.extend((0..n).filter(|&i| picked[i] == keep_picked));
}

/// The flat action table and its scratch buffers, reused across enumerations.
#[derive(Clone, Debug, Default)]
pub struct ActionTable {
    actions: Vec<FactoredAction>,
    legal_cards: Vec<usize>,
    combos: Vec<Selection>,
    picked: Vec<bool>,
    chosen: Vec<usize>,
}

impl ActionTable {
    pub fn new() -> ActionTable {
        ActionTable::default()
    }

    pub fn actions(&self) -> &[FactoredAction] {
        &self.actions
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&FactoredAction> {
        self.actions.get(i)
    }

    /// Empty the table (an episode ended).
    pub fn clear(&mut self) {
        self.actions.clear();
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
                    ActionType::UseConsumable => &areas.consumables,
                    _ => &areas.pack_cards,
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
        self.actions.clear();
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
                Targets::None => self.actions.push(FactoredAction::new(t)),
                Targets::Entity => {
                    if let Some(em) = entity(t) {
                        for (e, &ok) in em.iter().enumerate() {
                            if ok {
                                self.actions.push(FactoredAction::with_entity(t, e));
                            }
                        }
                    }
                }
                Targets::Cards => self.card_actions(t, None, min_select, max_select, forced, rng),
                Targets::EntityCards => {
                    if let Some(em) = entity(t) {
                        for (e, &ok) in em.iter().enumerate() {
                            if !ok {
                                continue;
                            }
                            match target_info(t, e) {
                                Some((lo, hi, true)) => {
                                    self.card_actions(t, Some(e), lo, hi, None, rng);
                                }
                                Some((_, _, false)) => {
                                    self.actions.push(FactoredAction::with_entity(t, e));
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        }
        if self.actions.len() > MAX_ACTIONS {
            subsample_indices(
                self.actions.len(),
                MAX_ACTIONS,
                rng,
                &mut self.picked,
                &mut self.chosen,
            );
            for (dst, &src) in self.chosen.iter().enumerate() {
                self.actions[dst] = self.actions[src];
            }
            self.actions.truncate(MAX_ACTIONS);
        }
    }

    /// `_card_actions`: the selections of `min_select..=max_select` legal cards (capped at the
    /// number of legal cards), by size then lexicographically, those containing `required`
    /// only, subsampled to [`CARD_COMBO_BUDGET`].
    fn card_actions(
        &mut self,
        t: ActionType,
        entity: Option<usize>,
        min_select: usize,
        max_select: usize,
        required: Option<usize>,
        rng: &mut Xoshiro256,
    ) {
        let n = self.legal_cards.len();
        let upper = n.min(max_select);
        let lower = min_select.min(upper);
        self.combos.clear();
        let mut idx = [0usize; MAX_SELECTION];
        for k in lower..=upper.min(MAX_SELECTION) {
            // Lexicographic k-combinations of positions 0..n.
            for (j, slot) in idx.iter_mut().take(k).enumerate() {
                *slot = j;
            }
            loop {
                let mut sel = Selection::default();
                for &p in &idx[..k] {
                    sel.push(self.legal_cards[p]);
                }
                if required.is_none_or(|r| sel.as_slice().contains(&(r as u8))) {
                    self.combos.push(sel);
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
        let entity_target = entity.map(|e| e as u32);
        let make = |sel: Selection| FactoredAction {
            action_type: t,
            entity_target,
            card_target: Some(sel),
        };
        if self.combos.len() <= CARD_COMBO_BUDGET {
            self.actions.extend(self.combos.iter().map(|&s| make(s)));
        } else {
            subsample_indices(
                self.combos.len(),
                CARD_COMBO_BUDGET,
                rng,
                &mut self.picked,
                &mut self.chosen,
            );
            self.actions
                .extend(self.chosen.iter().map(|&i| make(self.combos[i])));
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
