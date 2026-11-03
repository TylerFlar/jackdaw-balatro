//! End-to-end steps per second of one Gym env on a single thread: random legal actions, each
//! step followed by the observation and the action mask, as `BalatroGymnasiumEnv` runs them.
//!
//! `cargo run --release -p jackdaw-env --example rollout [steps] [--verify] [--deep]`
//!
//! `--deep` swaps the random policy for a cheap heuristic that plays its best-looking hand,
//! discards weak ones and takes shop and pack actions, so episodes reach shops, packs, jokers
//! and later antes (a random policy mostly loses the first blind).
//!
//! The checksum covers the rewards. `--verify` (slower; not a timing) also hashes every
//! observation, action mask and action table, and the canonical snapshot every 16 steps: an
//! optimisation of the engine or the env must leave it unchanged.
//!
//! It allocates with mimalloc, as the Python extension does. Under a profiler (callgrind,
//! `perf`, samply) build it with `--profile profiling`.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

use jackdaw_core::run::RunSpec;
use jackdaw_env::action::ActionType;
use jackdaw_env::env::{EnvConfig, GYM_OBS_LEN, GymEnv};
use jackdaw_env::gym::{MAX_ACTIONS, Xoshiro256};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn spec(i: u64) -> RunSpec {
    RunSpec {
        back: jackdaw_core::card::center_by_key("b_red").unwrap(),
        stake: 1,
        seed: format!("ROLL{i:04}"),
        challenge: None,
    }
}

/// The `--deep` policy: the play scoring best on rank and suit counts (unless a discard is
/// drawn instead when it scores low), else a random progress action (select a blind, cash
/// out, buy, open or pick from a pack, move on), else any action.
fn heuristic(env: &GymEnv, pick: &mut Xoshiro256) -> usize {
    let table = env.table();
    let hand = &env.game().gs.areas.hand();
    let mut best: Option<(i64, usize)> = None;
    let mut progress: Vec<usize> = Vec::new();
    let mut discards: Vec<usize> = Vec::new();
    for (i, a) in table.actions().into_iter().enumerate() {
        match a.action_type {
            ActionType::PlayHand => {
                let sel = a.card_target.unwrap();
                let mut ranks = [0u8; 15];
                let mut suits = [0u8; 4];
                for &k in sel.as_slice() {
                    if let Some(b) = hand.get(k as usize).and_then(|c| c.base) {
                        ranks[b.value.id() as usize] += 1;
                        suits[b.suit as usize] += 1;
                    }
                }
                let kind = i64::from(*ranks.iter().max().unwrap());
                let pairs = ranks.iter().filter(|&&r| r >= 2).count() as i64;
                let flush = if suits.iter().any(|&s| s >= 5) { 4 } else { 0 };
                let score = kind * 3 + pairs + flush + sel.len() as i64;
                if best.is_none_or(|(s, _)| score > s) {
                    best = Some((score, i));
                }
            }
            ActionType::Discard => discards.push(i),
            ActionType::SelectBlind
            | ActionType::CashOut
            | ActionType::BuyCard
            | ActionType::OpenBooster
            | ActionType::PickPackCard
            | ActionType::NextRound
            | ActionType::SkipPack
            | ActionType::UseConsumable => progress.push(i),
            _ => {}
        }
    }
    if let Some((score, i)) = best {
        if score < 10 && !discards.is_empty() && pick.below(3) != 0 {
            return discards[pick.below(discards.len() as u64) as usize];
        }
        return i;
    }
    if !progress.is_empty() {
        return progress[pick.below(progress.len() as u64) as usize];
    }
    pick.below(table.len() as u64) as usize
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verify = args.iter().any(|a| a == "--verify");
    let deep = args.iter().any(|a| a == "--deep");
    let steps: u64 = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map_or(2_000_000, |s| s.parse().expect("steps"));
    let mut hasher = DefaultHasher::new();
    let mut env = GymEnv::new(&spec(0), EnvConfig::default(), 1).unwrap();
    let mut pick = Xoshiro256::seed_from_u64(7);
    let mut obs = vec![0f32; GYM_OBS_LEN];
    let mut mask = vec![0u8; MAX_ACTIONS];
    let (mut episodes, mut checksum, mut best_ante) = (0u64, 0f64, 0f64);
    let start = Instant::now();
    for step in 0..steps {
        if env.ended() {
            episodes += 1;
            best_ante = best_ante.max(env.max_ante);
            env.reset(&spec(episodes), None).unwrap();
        }
        let i = if deep {
            heuristic(&env, &mut pick)
        } else {
            pick.below(env.table().len() as u64) as usize
        };
        let (r, _) = env.step_or_cut(i).unwrap();
        checksum += r.reward;
        env.write_obs(&mut obs);
        env.write_action_mask(&mut mask);
        checksum += f64::from(obs[0]) + f64::from(mask[0]);
        if verify {
            r.reward.to_bits().hash(&mut hasher);
            for x in &obs {
                x.to_bits().hash(&mut hasher);
            }
            mask.hash(&mut hasher);
            env.table().actions().hash(&mut hasher);
            if step % 16 == 0 && !env.ended() {
                jackdaw_core::canon::digest(&env.game().snapshot()).hash(&mut hasher);
            }
        }
    }
    if verify {
        println!("verify hash {:016x}", hasher.finish());
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{steps} steps, {episodes} episodes (best ante {best_ante}) in {secs:.3} s: {:.0} steps/s ({:.3} us/step), checksum {checksum}",
        steps as f64 / secs,
        secs * 1e6 / steps as f64
    );
}
