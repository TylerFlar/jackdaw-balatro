//! A game built from the player's view (`jackdaw_core::rebuild`): it shows what the view shows,
//! and lets the player do what the run lets them do.
//!
//! Random runs over backs, stakes, boards seeded with jokers and special cards, and bosses that
//! deal face down, shuffle the jokers or force a card: at every decision the run's view is built
//! into a game on the last one (stepped with the action since), and that game's own view must
//! be the run's, card ids aside, with the same legal actions.

use jackdaw_core::action::Action;
use jackdaw_core::area::AreaId;
use jackdaw_core::card::{Card, EditionKind, Seal, center_by_key};
use jackdaw_core::debug::{AddKey, AddSpec};
use jackdaw_core::game::Game;
use jackdaw_core::rebuild::{self, Shadow, ViewIds, from_player_view, same_view};
use jackdaw_core::run::RunSpec;
use jackdaw_core::state::Phase;
use serde_json::{Value, json};

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

const BACKS: [&str; 10] = [
    "b_red",
    "b_blue",
    "b_yellow",
    "b_green",
    "b_black",
    "b_plasma",
    "b_checkered",
    "b_abandoned",
    "b_erratic",
    "b_zodiac",
];
const ENHANCEMENTS: [&str; 8] = [
    "m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky",
];
/// Bosses that hide cards, shuffle the jokers, force a card, or keep their own count.
const BOSSES: [&str; 12] = [
    "bl_wheel",
    "bl_house",
    "bl_mark",
    "bl_fish",
    "bl_final_acorn",
    "bl_final_bell",
    "bl_pillar",
    "bl_water",
    "bl_needle",
    "bl_final_heart",
    "bl_psychic",
    "bl_eye",
];
/// Jokers whose card has its own size, plus some that shake or change the hidden counters.
const JOKERS: [&str; 12] = [
    "j_half",
    "j_photograph",
    "j_square",
    "j_wee",
    "j_dna",
    "j_trading",
    "j_drunkard",
    "j_troubadour",
    "j_satellite",
    "j_gros_michel",
    "j_joker",
    "j_burglar",
];

/// A new run with a seeded board: jokers (some eternal) and special cards, and sometimes a
/// boss on deck.
fn seeded_run(rng: &mut Rng, n: usize) -> Game {
    let spec = RunSpec {
        back: center_by_key(BACKS[n % BACKS.len()]).unwrap(),
        stake: 1 + (n % 8) as u8,
        seed: format!("REBUILD{n}"),
        challenge: None,
    };
    let mut game = Game::new_run(&spec).unwrap();
    if n.is_multiple_of(3) {
        return game;
    }
    for _ in 0..1 + rng.below(4) {
        let mut add = AddSpec::new(AddKey::Joker(center_by_key(rng.pick(&JOKERS)).unwrap()));
        if rng.chance(0.2) {
            add.edition = Some(*rng.pick(&[EditionKind::Foil, EditionKind::Negative]));
        }
        add.eternal = rng.chance(0.3);
        game.debug_add(&add).unwrap();
    }
    for _ in 0..rng.below(8) {
        let suit = rng.pick(&["H", "C", "D", "S"]);
        let rank = rng.pick(&["2", "5", "9", "T", "J", "Q", "K", "A"]);
        let mut add = AddSpec::new(AddKey::parse(&format!("{suit}_{rank}")).unwrap());
        if rng.chance(0.7) {
            add.enhancement = Some(center_by_key(rng.pick(&ENHANCEMENTS)).unwrap());
        }
        if rng.chance(0.3) {
            add.seal = Some(*rng.pick(&[Seal::Red, Seal::Blue, Seal::Gold, Seal::Purple]));
        }
        game.debug_add(&add).unwrap();
    }
    if n % 3 == 1 {
        let boss = rng.pick(&BOSSES);
        game.apply_setup_op(&json!({"op": "force_boss", "key": boss}))
            .unwrap();
    }
    game
}

/// A random action, choosing cards for a play or a discard (the forced card included).
fn choose(rng: &mut Rng, game: &Game) -> Option<Action> {
    let legal = game.legal_actions();
    if legal.is_empty() {
        return None;
    }
    let has = |f: fn(&Action) -> bool| legal.iter().any(f);
    let action = if has(|a| matches!(a, Action::PlayHand(_))) && rng.chance(0.65) {
        Action::PlayHand(Vec::new())
    } else if has(|a| matches!(a, Action::Discard(_))) && rng.chance(0.5) {
        Action::Discard(Vec::new())
    } else if has(|a| matches!(a, Action::NextRound)) && rng.chance(0.5) {
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
    let hand = game.gs.areas.hand();
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

/// The last game built from a view, and the shadow: it, stepped with the action since, with the
/// playing cards that step destroyed (`jackdaw.core.ViewTracker` in Python).
#[derive(Default)]
struct Tracker {
    game: Option<Game>,
    ids: ViewIds,
    shadow: Option<Game>,
    gone: Vec<Card>,
}

impl Tracker {
    fn update(&mut self, view: &Value) -> Game {
        let base = self.shadow.as_ref().or(self.game.as_ref());
        let rebuilt = from_player_view(
            view,
            base.map(|game| Shadow {
                game,
                ids: &self.ids,
                gone: &self.gone,
            }),
            None,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.ids = rebuilt.ids;
        self.shadow = None;
        self.gone.clear();
        self.game.insert(rebuilt.game).clone()
    }

    fn step(&mut self, action: &Action) {
        let mut shadow = self.game.clone().expect("a game built");
        let before: Vec<Card> = shadow.playing_cards().copied().collect();
        if shadow.step(action).is_ok() {
            let alive: Vec<u32> = shadow.playing_cards().map(|c| c.sort_id).collect();
            self.gone = before
                .into_iter()
                .filter(|c| !alive.contains(&c.sort_id))
                .collect();
            self.shadow = Some(shadow);
        }
    }
}

const KEY: u64 = 0x5EED;

#[test]
fn every_view_of_a_run_builds_into_a_game_that_shows_it() {
    let mut rng = Rng(0x0BAD_5EED);
    let mut decisions = 0;
    let mut face_down = 0;
    for n in 0..200 {
        let mut game = seeded_run(&mut rng, n);
        let mut tracker = Tracker::default();
        for step in 0..250 {
            if game.gs.phase == Phase::GameOver {
                break;
            }
            let at = format!("run {n} step {step} ({})", game.gs.phase.name());
            let view = game.player_view(KEY);
            face_down += view["hand"]
                .as_array()
                .unwrap()
                .iter()
                .chain(view["jokers"].as_array().unwrap())
                .filter(|c| c["face_down"] == json!(true))
                .count();
            let rebuilt = tracker.update(&view);
            let again = rebuilt.player_view(KEY ^ 1);
            assert!(
                same_view(&again, &view),
                "{at}: the view differs\n{}\n{}",
                rebuild::without_ids(&again),
                rebuild::without_ids(&view)
            );
            assert_eq!(rebuilt.legal_actions(), game.legal_actions(), "{at}");
            assert_eq!(rebuilt.gs.phase, game.gs.phase, "{at}");
            // Every card the view names has the sort id the ids say.
            for (area, key) in [(AreaId::Hand, "hand"), (AreaId::Jokers, "jokers")] {
                for (c, vc) in rebuilt
                    .gs
                    .areas
                    .get(area)
                    .iter()
                    .zip(view[key].as_array().unwrap())
                {
                    assert_eq!(tracker.ids[&vc["id"].as_u64().unwrap()], c.sort_id, "{at}");
                }
            }
            decisions += 1;
            let Some(action) = choose(&mut rng, &game) else {
                break;
            };
            if game.step(&action).is_err() {
                break;
            }
            tracker.step(&action);
        }
    }
    assert!(decisions > 2_000, "{decisions} decisions");
    println!("{decisions} decisions, {face_down} face-down cards");
    assert!(face_down > 100, "only {face_down} face-down cards seen");
}

/// Amber Acorn turns the jokers face down and shuffles them; each one the rebuild places where
/// the run's is has its size and sell button, and the ones that show neither differ only in
/// where they went.
#[test]
fn amber_acorn_s_jokers_keep_what_shows() {
    let spec = RunSpec {
        back: center_by_key("b_red").unwrap(),
        stake: 1,
        seed: "ACORN".to_owned(),
        challenge: None,
    };
    let mut game = Game::new_run(&spec).unwrap();
    for (key, eternal) in [
        ("j_wee", false),
        ("j_joker", true),
        ("j_half", false),
        ("j_greedy_joker", false),
    ] {
        let mut add = AddSpec::new(AddKey::Joker(center_by_key(key).unwrap()));
        add.eternal = eternal;
        game.debug_add(&add).unwrap();
    }
    game.apply_setup_op(&json!({"op": "force_boss", "key": "bl_final_acorn"}))
        .unwrap();
    let mut tracker = Tracker::default();
    tracker.update(&game.player_view(KEY));
    game.step(&Action::SelectBlind).unwrap();
    tracker.step(&Action::SelectBlind);
    let view = game.player_view(KEY);
    let rebuilt = tracker.update(&view);
    assert!(same_view(&rebuilt.player_view(1), &view));
    let real = game.gs.areas.jokers();
    let built = rebuilt.gs.areas.jokers();
    assert!(
        built
            .iter()
            .all(|c| c.facing == jackdaw_core::card::Facing::Back)
    );
    for (r, b) in real.iter().zip(built.iter()) {
        assert_eq!(
            jackdaw_core::player_view::shape(r),
            jackdaw_core::player_view::shape(b)
        );
        assert_eq!(r.ability.eternal, b.ability.eternal);
    }
    let keys = |cs: jackdaw_core::area::Cards<'_>| {
        let mut k: Vec<_> = cs.iter().map(|c| c.center.key()).collect();
        k.sort_unstable();
        k
    };
    assert_eq!(
        keys(real),
        keys(built),
        "the same jokers, wherever they went"
    );
}

/// A seed gives other stand-ins: the same view, another draw pile order.
#[test]
fn a_seed_gives_other_stand_ins() {
    let spec = RunSpec {
        back: center_by_key("b_red").unwrap(),
        stake: 1,
        seed: "STANDINS".to_owned(),
        challenge: None,
    };
    let mut game = Game::new_run(&spec).unwrap();
    game.step(&Action::SelectBlind).unwrap();
    let view = game.player_view(KEY);
    let a = from_player_view(&view, None, Some("AAAA")).unwrap().game;
    let b = from_player_view(&view, None, Some("BBBB")).unwrap().game;
    assert!(same_view(&a.player_view(1), &view) && same_view(&b.player_view(2), &view));
    let order = |g: &Game| -> Vec<u32> {
        g.gs.areas
            .deck()
            .iter()
            .map(|c| c.base.unwrap().front.0.into())
            .collect()
    };
    assert_ne!(order(&a), order(&b));
    assert_eq!(a.gs.rng.seed_str(), "AAAA");
}

#[test]
fn a_view_that_doesnt_add_up_is_refused() {
    let spec = RunSpec {
        back: center_by_key("b_red").unwrap(),
        stake: 1,
        seed: "REFUSED".to_owned(),
        challenge: None,
    };
    let mut game = Game::new_run(&spec).unwrap();
    game.step(&Action::SelectBlind).unwrap();
    let view = game.player_view(KEY);
    let mut v = view.clone();
    v["format"] = json!("jackdaw.player_view/0");
    assert!(from_player_view(&v, None, None).is_err());
    let mut v = view.clone();
    v["deck_size"] = json!(43);
    let e = from_player_view(&v, None, None).unwrap_err();
    assert!(e.0.contains("remaining cards"), "{e}");
    // A shadow of another run.
    let other = Game::new_run(&RunSpec {
        back: center_by_key("b_blue").unwrap(),
        ..spec
    })
    .unwrap();
    let e = from_player_view(
        &view,
        Some(Shadow {
            game: &other,
            ids: &ViewIds::new(),
            gone: &[],
        }),
        None,
    )
    .unwrap_err();
    assert!(e.0.contains("another run"), "{e}");
}
