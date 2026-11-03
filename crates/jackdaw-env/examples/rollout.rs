//! End-to-end steps per second of one Gym env on a single thread: random legal actions, each
//! step followed by the observation and the action mask, as `BalatroGymnasiumEnv` runs them.
//!
//! `cargo run --release -p jackdaw-env --example rollout [steps] [--verify]`
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verify = args.iter().any(|a| a == "--verify");
    let steps: u64 = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map_or(2_000_000, |s| s.parse().expect("steps"));
    let mut hasher = DefaultHasher::new();
    let mut env = GymEnv::new(&spec(0), EnvConfig::default(), 1).unwrap();
    let mut pick = Xoshiro256::seed_from_u64(7);
    let mut obs = vec![0f32; GYM_OBS_LEN];
    let mut mask = vec![0u8; MAX_ACTIONS];
    let (mut episodes, mut checksum) = (0u64, 0f64);
    let start = Instant::now();
    for step in 0..steps {
        if env.ended() {
            episodes += 1;
            env.reset(&spec(episodes), None).unwrap();
        }
        let n = env.table().len() as u64;
        let i = pick.below(n) as usize;
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
        "{steps} steps, {episodes} episodes in {secs:.3} s: {:.0} steps/s ({:.3} us/step), checksum {checksum}",
        steps as f64 / secs,
        secs * 1e6 / steps as f64
    );
}
