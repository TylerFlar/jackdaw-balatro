//! Env-steps per second of the vector env: `n` envs stepped together on `threads` threads with
//! a random legal policy and next-step autoreset, as `BalatroVectorEnv` runs them.
//!
//! `cargo run --release -p jackdaw-env --example vector [env-steps per config] [n:threads ...]`
//!
//! Without configurations it runs n = 16, 64 and 256 on 1 and 4 threads. Each configuration
//! steps about the same number of env-steps (default 400 000). The checksum covers the
//! rewards and does not depend on the thread count.

use std::time::Instant;

use jackdaw_core::run::RunSpec;
use jackdaw_env::env::{EnvConfig, GYM_OBS_LEN};
use jackdaw_env::gym::{MAX_ACTIONS, Xoshiro256};
use jackdaw_env::vector::{EnvOut, ResetSpec, VectorEnv};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn spec(i: u64) -> RunSpec {
    RunSpec {
        back: jackdaw_core::card::center_by_key("b_red").unwrap(),
        stake: 1,
        seed: format!("VEC{i:06}"),
        challenge: None,
    }
}

fn run(n: usize, threads: usize, env_steps: usize) {
    let steps = (env_steps / n).max(100);
    let resets: Vec<ResetSpec> = (0..n)
        .map(|i| ResetSpec {
            spec: spec(i as u64),
            rng_seed: Some(i as u64),
        })
        .collect();
    let mut v = VectorEnv::new(&resets, EnvConfig::default(), threads).unwrap();
    let mut obs = vec![0f32; n * GYM_OBS_LEN];
    let mut masks = vec![0u8; n * MAX_ACTIONS];
    let mut out = vec![EnvOut::default(); n];
    v.reset(&resets, &mut obs, &mut masks).unwrap();
    let mut policy = Xoshiro256::seed_from_u64(99);
    let mut episodes = vec![0u64; n];
    let mut actions = vec![0usize; n];
    let mut next: Vec<Option<ResetSpec>> = vec![None; n];
    let mut checksum = 0f64;
    let start = Instant::now();
    for _ in 0..steps {
        for i in 0..n {
            if next[i].is_none() {
                actions[i] = policy.below(v.envs()[i].table().len() as u64) as usize;
            }
        }
        v.step(&actions, &next, &mut obs, &mut masks, &mut out)
            .unwrap();
        for i in 0..n {
            checksum += out[i].reward;
            next[i] = out[i].episode_end.then(|| {
                episodes[i] += 1;
                ResetSpec {
                    spec: spec(1_000_000 * (1 + i as u64) + episodes[i]),
                    rng_seed: None,
                }
            });
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let total = (steps * n) as f64;
    println!(
        "n={n:<4} threads={threads}: {:8.1} us/batch {:6.3} us/env-step {:>9.0} env-steps/s, checksum {checksum}",
        secs * 1e6 / steps as f64,
        secs * 1e6 / total,
        total / secs
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env_steps: usize = args
        .iter()
        .find(|a| !a.contains(':'))
        .map_or(400_000, |s| s.parse().expect("env-steps"));
    let mut configs: Vec<(usize, usize)> = args
        .iter()
        .filter_map(|a| {
            let (n, t) = a.split_once(':')?;
            Some((n.parse().expect("n"), t.parse().expect("threads")))
        })
        .collect();
    if configs.is_empty() {
        configs = vec![(16, 1), (64, 1), (256, 1), (16, 4), (64, 4), (256, 4)];
    }
    for (n, threads) in configs {
        run(n, threads, env_steps);
    }
}
