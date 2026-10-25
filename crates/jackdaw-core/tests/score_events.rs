//! Scoring events (`jackdaw_core::trace`): recording a run changes nothing in it, and each
//! play's events account for its score.
//!
//! Random runs over backs, stakes and boards seeded with jokers, editions, enhancements and
//! seals: a twin that doesn't record must reach the same state after every step, and every
//! recorded play must replay, event by event, to the chips, mult, total and money it scored.

use jackdaw_core::action::Action;
use jackdaw_core::card::{EditionKind, Seal, center_by_key};
use jackdaw_core::debug::{AddKey, AddSpec};
use jackdaw_core::game::{Game, StepError};
use jackdaw_core::lua;
use jackdaw_core::run::RunSpec;
use jackdaw_core::trace::{PlayTrace, ScoreKind};

/// splitmix64: a small seeded generator, so the runs are the same every time.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

const BACKS: [&str; 8] = [
    "b_red",
    "b_blue",
    "b_yellow",
    "b_green",
    "b_black",
    "b_plasma",
    "b_checkered",
    "b_abandoned",
];
const ENHANCEMENTS: [&str; 8] = [
    "m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky",
];

fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// A new run with a seeded board: jokers (some with editions or stickers) and special cards on
/// top of the deck.
fn seeded_run(rng: &mut Rng, n: usize) -> Game {
    let spec = RunSpec {
        back: center_by_key(BACKS[n % BACKS.len()]).unwrap(),
        stake: 1 + (n % 8) as u8,
        seed: format!("TRACE{n}"),
        challenge: None,
    };
    let mut game = Game::new_run(&spec).unwrap();
    let jokers = jackdaw_core::joker::ported();
    for _ in 0..2 + rng.below(4) {
        let mut add = AddSpec::new(AddKey::Joker(center_by_key(rng.pick(&jokers)).unwrap()));
        if rng.chance(0.3) {
            add.edition = Some(*rng.pick(&[
                EditionKind::Foil,
                EditionKind::Holo,
                EditionKind::Polychrome,
                EditionKind::Negative,
            ]));
        }
        add.eternal = rng.chance(0.1);
        game.debug_add(&add).unwrap();
    }
    for _ in 0..rng.below(10) {
        let suit = rng.pick(&["H", "C", "D", "S"]);
        let rank = rng.pick(&["2", "5", "9", "T", "J", "Q", "K", "A"]);
        let key = format!("{suit}_{rank}");
        let mut add = AddSpec::new(AddKey::parse(&key).unwrap());
        if rng.chance(0.7) {
            add.enhancement = Some(center_by_key(rng.pick(&ENHANCEMENTS)).unwrap());
        }
        if rng.chance(0.4) {
            add.seal = Some(*rng.pick(&[Seal::Red, Seal::Blue, Seal::Gold, Seal::Purple]));
        }
        if rng.chance(0.2) {
            add.edition = Some(*rng.pick(&[
                EditionKind::Foil,
                EditionKind::Holo,
                EditionKind::Polychrome,
            ]));
        }
        game.debug_add(&add).unwrap();
    }
    game
}

/// A random action, choosing cards for a play or a discard (the forced card included).
fn choose(rng: &mut Rng, game: &Game) -> Option<Action> {
    let legal = game.legal_actions();
    if legal.is_empty() {
        return None;
    }
    let play = legal.iter().any(|a| matches!(a, Action::PlayHand(_)));
    let discard = legal.iter().any(|a| matches!(a, Action::Discard(_)));
    let next_round = legal.iter().any(|a| matches!(a, Action::NextRound));
    let action = if play && rng.chance(0.65) {
        Action::PlayHand(Vec::new())
    } else if discard && rng.chance(0.6) {
        Action::Discard(Vec::new())
    } else if next_round && rng.chance(0.5) {
        Action::NextRound
    } else {
        rng.pick(&legal).clone()
    };
    Some(match action {
        Action::PlayHand(_) => Action::PlayHand(selection(rng, game)),
        Action::Discard(_) => Action::Discard(selection(rng, game)),
        other => other,
    })
}

fn selection(rng: &mut Rng, game: &Game) -> Vec<usize> {
    let hand = &game.gs.areas.hand;
    let mut picked: Vec<usize> = hand
        .iter()
        .position(|c| c.ability.forced_selection)
        .into_iter()
        .collect();
    let want = 1 + rng.below(hand.len().min(5));
    while picked.len() < want {
        let i = rng.below(hand.len());
        if !picked.contains(&i) {
            picked.push(i);
        }
    }
    picked.sort_unstable();
    picked
}

/// The play's events account for what it scored.
fn check_play(game: &Game, play: &PlayTrace, selected: &[u32], at: &str) {
    let score = game
        .gs
        .sim
        .last_score
        .as_ref()
        .expect("a play records its score");
    assert_eq!(play.hand, score.hand_type, "{at}");
    let mut played = play.played.clone();
    played.sort_unstable();
    let mut chosen = selected.to_vec();
    chosen.sort_unstable();
    assert_eq!(played, chosen, "{at}: the played cards");
    assert!(
        play.scoring.iter().all(|id| play.played.contains(id)),
        "{at}"
    );

    let mut pair: Option<(f64, f64)> = None;
    let mut dollars = 0.0;
    let mut hands = 0;
    for (n, ev) in play.events.iter().enumerate() {
        let at = format!("{at} event {n} {:?}", ev.kind);
        dollars += ev.dollars.unwrap_or(0.0);
        if ev.kind == ScoreKind::LevelUp {
            assert!(pair.is_none(), "{at}: a level-up after scoring started");
            continue;
        }
        if ev.kind.sets_score() {
            hands += usize::from(ev.kind == ScoreKind::Hand);
            pair = Some((ev.hand_chips, ev.hand_mult));
            continue;
        }
        let (mut c, mut m) = pair.unwrap_or_else(|| panic!("{at}: before the hand's base"));
        if let Some(x) = ev.chips {
            c = lua::add(c, x);
        }
        if let Some(x) = ev.mult {
            m = lua::add(m, x);
        }
        if let Some(x) = ev.x_mult {
            m = lua::mul(m, x);
        }
        if ev.swap {
            (c, m) = (m, c);
        }
        assert!(
            same(c, ev.hand_chips) && same(m, ev.hand_mult),
            "{at}: {c} x {m} replayed, {} x {} recorded",
            ev.hand_chips,
            ev.hand_mult
        );
        pair = Some((c, m));
    }
    assert_eq!(hands, 1, "{at}: one hand event");
    let (c, m) = pair.unwrap();
    assert!(
        same(c, score.chips) && same(m, score.mult),
        "{at}: the final chips and mult"
    );
    assert!(
        same(lua::floor(lua::mul(c, m)), score.total),
        "{at}: the total"
    );
    assert!(
        (dollars - score.dollars).abs() < 1e-9,
        "{at}: {dollars} in events, {} paid",
        score.dollars
    );
}

#[test]
fn recorded_plays_account_for_their_score_and_change_nothing() {
    let mut rng = Rng(0x5C0E_E7E5);
    let mut plays = 0;
    let mut kinds = std::collections::BTreeSet::new();
    for n in 0..200 {
        let mut game = seeded_run(&mut rng, n);
        let mut twin = game.clone();
        game.set_record_scoring(true);
        for step in 0..400 {
            if game.gs.phase == jackdaw_core::state::Phase::GameOver {
                break;
            }
            let mut taken = None;
            for _ in 0..20 {
                let Some(action) = choose(&mut rng, &game) else {
                    break;
                };
                let selected: Vec<u32> = match &action {
                    Action::PlayHand(cards) => cards
                        .iter()
                        .map(|&i| game.gs.areas.hand[i].sort_id)
                        .collect(),
                    _ => Vec::new(),
                };
                match game.step(&action) {
                    Ok(()) => {
                        taken = Some((action, selected));
                        break;
                    }
                    Err(StepError::Illegal(_)) => continue,
                    Err(e) => panic!("run {n} step {step}: {e}"),
                }
            }
            let Some((action, selected)) = taken else {
                break;
            };
            twin.step(&action).unwrap();
            assert_eq!(game.snapshot(), twin.snapshot(), "run {n} step {step}");
            assert!(twin.last_play().is_none(), "the twin doesn't record");
            let at = format!("run {n} step {step} {action:?}");
            match (&action, game.last_play()) {
                (Action::PlayHand(_), Some(play)) => {
                    check_play(&game, play, &selected, &at);
                    kinds.extend(play.events.iter().map(|e| e.kind.name()));
                    plays += 1;
                }
                (Action::PlayHand(_), None) => panic!("{at}: no trace"),
                (_, Some(_)) => panic!("{at}: a trace without a play"),
                (_, None) => {}
            }
        }
    }
    assert!(plays > 800, "only {plays} plays");
    for kind in [
        "hand",
        "card",
        "joker",
        "edition",
        "held",
        "retrigger",
        "destroyed",
        "plasma",
    ] {
        assert!(
            kinds.contains(kind),
            "no {kind} event in {plays} plays: {kinds:?}"
        );
    }
}

#[test]
fn recording_off_keeps_no_trace() {
    let mut rng = Rng(7);
    let mut game = seeded_run(&mut rng, 3);
    game.set_record_scoring(true);
    game.step(&Action::SelectBlind).unwrap();
    game.step(&Action::PlayHand(vec![0, 1])).unwrap();
    assert!(game.last_play().is_some());
    game.set_record_scoring(false);
    assert!(game.last_play().is_none());
    game.step(&Action::PlayHand(vec![0])).unwrap();
    assert!(game.last_play().is_none());
}
