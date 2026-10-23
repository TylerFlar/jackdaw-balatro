//! The engine's actions as BalatroBot requests, and which of them BalatroBot executes.
//!
//! **Translation** ([`translate`]): every action BalatroBot can express becomes its request.
//! `SortHand` and the `Swap*` actions become `rearrange` permutations of the whole area (a sort
//! is the permutation that sorts the hand; the sort *mode*, `G.hand.config.sort`, has no
//! BalatroBot counterpart, so later draws in the live game keep sorting by rank). The marker
//! `PlayHand([])` / `Discard([])` of the legal-action list stands for any selection; it is
//! checked with one card (Cerulean Bell's forced card, else the first).
//!
//! **Sim-only actions** have no request: `RerollBoss` (Director's Cut / Retcon) and
//! `BuyAndUse`.
//!
//! **Legality** ([`Legality`]): `VANILLA` is the engine's own list; `BALATROBOT` keeps the
//! actions whose request passes the dispatcher's checks and the endpoint's own validation, run
//! as a dry run ([`Session::check`]). That masks what vanilla allows but the bot refuses:
//! sim-only actions, selling or using outside `SELECTING_HAND` / `SHOP`, using a card-selecting
//! consumable in the shop, Aura (`use` never highlights its target), and buying or picking a
//! Negative joker at full slots.

use jackdaw_core::action::Action;
use jackdaw_core::area::AreaId;
use jackdaw_core::game::Game;
use jackdaw_core::state::HandSort;

use crate::lua::{Table, Value};
use crate::session::{Session, SessionConfig};

/// Which actions count as legal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Legality {
    /// Vanilla 1.0.1o's rules (the engine's `legal_actions`).
    Vanilla,
    /// Vanilla's rules, and the live BalatroBot v1.5.2 accepts the request.
    Balatrobot,
}

impl Legality {
    pub fn name(self) -> &'static str {
        match self {
            Legality::Vanilla => "vanilla",
            Legality::Balatrobot => "balatrobot",
        }
    }

    pub fn from_name(name: &str) -> Option<Legality> {
        match name.to_ascii_lowercase().as_str() {
            "vanilla" => Some(Legality::Vanilla),
            "balatrobot" => Some(Legality::Balatrobot),
            _ => None,
        }
    }
}

fn nums(items: impl IntoIterator<Item = usize>) -> Value {
    Value::Table(Table::sequence(
        items.into_iter().map(|i| Value::Num(i as f64)),
    ))
}

fn num(i: usize) -> Value {
    Value::Num(i as f64)
}

/// The permutation of hand positions that sorts the hand by `mode` (`CardArea:sort`,
/// cardarea.lua:577-590): position `k` gets the card that was at `perm[k]`.
pub fn sort_permutation(game: &Game, mode: HandSort) -> Vec<usize> {
    let suit = mode == HandSort::SuitDesc;
    let hand = &game.gs.areas.hand;
    let mut perm: Vec<usize> = (0..hand.len()).collect();
    perm.sort_by(|&a, &b| {
        hand[b]
            .get_nominal(suit)
            .partial_cmp(&hand[a].get_nominal(suit))
            .expect("nominals are finite")
    });
    perm
}

/// The identity permutation of `n` positions with `i` and `j` swapped.
fn swap_permutation(n: usize, i: usize, j: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    if i < n && j < n {
        p.swap(i, j);
    }
    p
}

/// The BalatroBot request that performs `action` (method and params), or `None` for a sim-only
/// action.
pub fn translate(game: &Game, action: &Action) -> Option<(&'static str, Table)> {
    let hand_n = game.gs.areas.hand.len();
    let joker_n = game.gs.areas.jokers.len();
    let selection = |cards: &Vec<usize>| -> Vec<usize> {
        if !cards.is_empty() {
            return cards.clone();
        }
        // The marker stands for any selection: check one card (the forced one if any).
        let forced = game
            .gs
            .areas
            .hand
            .iter()
            .position(|c| c.ability.forced_selection);
        vec![forced.unwrap_or(0)]
    };
    Some(match action {
        Action::SelectBlind => ("select", Table::new()),
        Action::SkipBlind => ("skip", Table::new()),
        Action::PlayHand(cards) => ("play", Table::object([("cards", nums(selection(cards)))])),
        Action::Discard(cards) => (
            "discard",
            Table::object([("cards", nums(selection(cards)))]),
        ),
        Action::BuyCard(i) => ("buy", Table::object([("card", num(*i))])),
        Action::RedeemVoucher(i) => ("buy", Table::object([("voucher", num(*i))])),
        Action::OpenBooster(i) => ("buy", Table::object([("pack", num(*i))])),
        Action::SellCard { area, index } => {
            let key = if *area == AreaId::Jokers {
                "joker"
            } else {
                "consumable"
            };
            ("sell", Table::object([(key, num(*index))]))
        }
        Action::UseConsumable { index, targets } => {
            let mut t = Table::object([("consumable", num(*index))]);
            if let Some(ts) = targets
                && !ts.is_empty()
            {
                t.set("cards", Some(nums(ts.iter().copied())));
            }
            ("use", t)
        }
        Action::PickPackCard { index, targets } => {
            let mut t = Table::object([("card", num(*index))]);
            if let Some(ts) = targets
                && !ts.is_empty()
            {
                t.set("targets", Some(nums(ts.iter().copied())));
            }
            ("pack", t)
        }
        Action::SkipPack => ("pack", Table::object([("skip", Value::Bool(true))])),
        Action::Reroll => ("reroll", Table::new()),
        Action::NextRound => ("next_round", Table::new()),
        Action::CashOut => ("cash_out", Table::new()),
        Action::SortHand(mode) => (
            "rearrange",
            Table::object([("hand", nums(sort_permutation(game, *mode)))]),
        ),
        Action::SwapHandLeft(i) => (
            "rearrange",
            Table::object([(
                "hand",
                nums(swap_permutation(hand_n, *i, i.wrapping_sub(1))),
            )]),
        ),
        Action::SwapHandRight(i) => (
            "rearrange",
            Table::object([("hand", nums(swap_permutation(hand_n, *i, i + 1)))]),
        ),
        Action::SwapJokersLeft(i) => (
            "rearrange",
            Table::object([(
                "jokers",
                nums(swap_permutation(joker_n, *i, i.wrapping_sub(1))),
            )]),
        ),
        Action::SwapJokersRight(i) => (
            "rearrange",
            Table::object([("jokers", nums(swap_permutation(joker_n, *i, i + 1)))]),
        ),
        Action::RerollBoss | Action::BuyAndUse { .. } => return None,
    })
}

/// Whether the live BalatroBot executes `action` (its request passes every check), judged in a
/// session resting in a copy of `game`. The action itself is assumed vanilla-legal.
pub fn bot_accepts(game: &Game, action: &Action) -> bool {
    let mut session = Session::with_game(game.clone(), SessionConfig::default());
    accepts_in(&mut session, action)
}

fn accepts_in(session: &mut Session, action: &Action) -> bool {
    let Some(game) = session.game() else {
        return false;
    };
    let Some((method, params)) = translate(game, action) else {
        return false;
    };
    session.check(method, Some(&Value::Table(params))).is_ok()
}

/// The legal actions under `legality`, in the engine's order.
pub fn legal_actions(game: &Game, legality: Legality) -> Vec<Action> {
    let vanilla = game.legal_actions();
    match legality {
        Legality::Vanilla => vanilla,
        Legality::Balatrobot => {
            let mut session = Session::with_game(game.clone(), SessionConfig::default());
            vanilla
                .into_iter()
                .filter(|a| accepts_in(&mut session, a))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jackdaw_core::run::RunSpec;

    fn run(seed: &str) -> Game {
        Game::new_run(&RunSpec {
            back: jackdaw_data::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: seed.into(),
            challenge: None,
        })
        .unwrap()
    }

    #[test]
    fn sim_only_actions_are_masked() {
        let g = run("LEGAL1");
        assert!(translate(&g, &Action::RerollBoss).is_none());
        let vanilla = legal_actions(&g, Legality::Vanilla);
        let bot = legal_actions(&g, Legality::Balatrobot);
        assert!(vanilla.contains(&Action::SelectBlind) && bot.contains(&Action::SelectBlind));
        assert!(bot.iter().all(|a| vanilla.contains(a)));
    }

    #[test]
    fn swaps_become_permutations() {
        let mut g = run("LEGAL2");
        g.step(&Action::SelectBlind).unwrap();
        let (m, t) = translate(&g, &Action::SwapHandRight(0)).unwrap();
        assert_eq!(m, "rearrange");
        let perm: Vec<f64> = t
            .get("hand")
            .unwrap()
            .as_table()
            .unwrap()
            .ipairs()
            .map(|v| v.as_num().unwrap())
            .collect();
        assert_eq!(&perm[..3], &[1.0, 0.0, 2.0]);
        // The hand is dealt sorted by rank: the rank sort is the identity.
        let sorted = sort_permutation(&g, HandSort::Desc);
        assert_eq!(sorted, (0..g.gs.areas.hand.len()).collect::<Vec<_>>());
    }
}
