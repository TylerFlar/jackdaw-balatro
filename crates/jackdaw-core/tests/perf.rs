//! Rough timings of the engine's hot operations against the port's performance targets
//! (`cargo test --release -p jackdaw-core --test perf -- --ignored --nocapture`). Not a
//! benchmark suite. It allocates with mimalloc, as the Python extension does.
//!
//! A step is timed as an environment runs it: on a game that has stepped before, copied back
//! to the same position each time with `clone_from` (which keeps the game's allocations), minus
//! the time of that copy.

use std::hint::black_box;
use std::time::Instant;

use jackdaw_core::action::Action;
use jackdaw_core::debug::{AddKey, AddSpec, SetField};
use jackdaw_core::game::Game;
use jackdaw_core::run::RunSpec;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The mean time of `f` in microseconds, the best of five runs of `n` calls.
fn mean_us<T>(n: u32, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..n.min(200) {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..n {
            black_box(f());
        }
        best = best.min(start.elapsed().as_secs_f64() / f64::from(n));
    }
    best * 1e6
}

fn report(label: &str, us: f64, target_us: f64) {
    let verdict = if us <= target_us { "ok" } else { "OVER" };
    println!("{label:<34} {us:>8.3} us   target {target_us:>5.2} us   {verdict}");
}

/// The time of `action` from `game`'s position, without the copy back to it.
fn step_us(game: &Game, action: &Action, n: u32) -> f64 {
    let mut g = game.clone();
    let with_step = mean_us(n, || {
        g.clone_from(game);
        g.step_in_place(action).unwrap();
    });
    with_step - mean_us(n, || g.clone_from(game))
}

fn joker(game: &mut Game, key: &str) {
    let center = jackdaw_core::card::center_by_key(key).unwrap();
    game.debug_add(&AddSpec::new(AddKey::Joker(center)))
        .unwrap();
}

#[test]
#[ignore = "timing; run with --release --ignored --nocapture"]
fn timings() {
    let spec = RunSpec {
        back: jackdaw_core::card::center_by_key("b_red").unwrap(),
        stake: 1,
        seed: "PERFRUN1".to_owned(),
        challenge: None,
    };
    report(
        "new run",
        mean_us(5_000, || Game::new_run(&spec).unwrap()),
        4.0,
    );
    let fresh = Game::new_run(&spec).unwrap();
    report("clone", mean_us(20_000, || fresh.clone()), 3.0);
    let mut into = fresh.clone();
    let us = mean_us(20_000, || into.clone_from(&fresh));
    println!(
        "{:<34} {us:>8.3} us   (no target)",
        "clone_from into a used game"
    );

    let mut round = fresh.clone();
    round.step(&Action::SelectBlind).unwrap();
    let play = Action::PlayHand(vec![0, 1, 2, 3, 4]);
    // So far behind that no hand wins the blind.
    let mut behind = round.clone();
    behind.debug_set(SetField::Chips(-1e12));
    report(
        "play 5 cards, no jokers",
        step_us(&behind, &play, 20_000),
        1.0,
    );

    let mut jokers = fresh.clone();
    for key in [
        "j_joker",
        "j_greedy_joker",
        "j_lusty_joker",
        "j_wrathful_joker",
        "j_gluttenous_joker",
    ] {
        joker(&mut jokers, key);
    }
    jokers.step(&Action::SelectBlind).unwrap();
    jokers.debug_set(SetField::Chips(-1e12));
    report(
        "play 5 cards, 5 jokers",
        step_us(&jokers, &play, 20_000),
        3.0,
    );

    // Win the blind with the next hand, then cash out into the shop.
    let mut winning = round.clone();
    winning.debug_set(SetField::Chips(1e12));
    let us = step_us(&winning, &play, 20_000);
    println!(
        "{:<34} {us:>8.3} us   (the round's end, no target)",
        "play 5 cards, wins the blind"
    );
    let mut won = winning.clone();
    won.step(&play).unwrap();
    report(
        "cash out + shop",
        step_us(&won, &Action::CashOut, 20_000),
        3.0,
    );
    let mut shop = won.clone();
    shop.step(&Action::CashOut).unwrap();
    shop.debug_set(SetField::Money(1e12));
    report(
        "open booster",
        step_us(&shop, &Action::OpenBooster(0), 20_000),
        1.5,
    );
    let mut rerolls = shop.clone();
    report(
        "reroll",
        mean_us(20_000, || rerolls.step_in_place(&Action::Reroll).unwrap()),
        1.5,
    );

    let mut rng = fresh.clone().gs.rng;
    rng.random("perf");
    report(
        "rng draw (pseudorandom)",
        mean_us(1_000_000, || rng.random("perf")),
        0.07,
    );
}
