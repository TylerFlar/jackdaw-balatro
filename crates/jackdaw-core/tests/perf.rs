//! Rough timings of the engine's hot operations (`cargo test --release --test perf -- --ignored
//! --nocapture`). Not a benchmark suite: a quick look against the port's performance targets.

use std::hint::black_box;
use std::time::Instant;

use jackdaw_core::action::Action;
use jackdaw_core::game::Game;
use jackdaw_core::run::RunSpec;

fn time<T>(label: &str, n: u32, mut f: impl FnMut() -> T) {
    for _ in 0..n.min(100) {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..n {
        black_box(f());
    }
    let per = start.elapsed().as_secs_f64() / f64::from(n);
    println!("{label:<32} {:>10.2} us", per * 1e6);
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
    time("new_run", 2_000, || Game::new_run(&spec).unwrap());
    let fresh = Game::new_run(&spec).unwrap();
    time("clone", 20_000, || fresh.clone());
    time("select blind (clone + step)", 5_000, || {
        let mut g = fresh.clone();
        g.step(&Action::SelectBlind).unwrap();
        g
    });
    let mut in_round = fresh.clone();
    in_round.step(&Action::SelectBlind).unwrap();
    time("play 5 cards (clone + step)", 5_000, || {
        let mut g = in_round.clone();
        g.step(&Action::PlayHand(vec![0, 1, 2, 3, 4])).unwrap();
        g
    });
    time("play 5 cards (clone + step_in_place)", 5_000, || {
        let mut g = in_round.clone();
        g.step_in_place(&Action::PlayHand(vec![0, 1, 2, 3, 4]))
            .unwrap();
        g
    });
    time("discard 5 cards (clone + step)", 5_000, || {
        let mut g = in_round.clone();
        g.step(&Action::Discard(vec![0, 1, 2, 3, 4])).unwrap();
        g
    });
    time("snapshot export", 2_000, || in_round.snapshot());
    let snap = in_round.snapshot();
    time("snapshot import", 2_000, || {
        Game::from_snapshot(&snap).unwrap()
    });
}
