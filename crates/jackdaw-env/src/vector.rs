//! A batch of Gym environments stepped in parallel (`BalatroVectorEnv`), with next-step
//! autoreset: an env whose episode ended is reset by the following step instead of stepped.
//!
//! Each env owns its run, buffers and sampler, and writes only its own slice of the outputs, so
//! the results don't depend on how many threads step them.

use jackdaw_core::run::RunSpec;
use rayon::prelude::*;

use crate::env::{EnvConfig, GYM_OBS_LEN, GymEnv, StepError};
use crate::gym::MAX_ACTIONS;

/// A reset: the run to start and, optionally, a new sampler seed.
#[derive(Clone, Debug)]
pub struct ResetSpec {
    pub spec: RunSpec,
    pub rng_seed: Option<u64>,
}

/// What one env did in a vector step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnvOut {
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    /// The episode ended on this step.
    pub episode_end: bool,
    /// At an episode end: the best ante, blinds beaten, and whether the run was won.
    pub ante_reached: f64,
    pub rounds_beaten: u64,
    pub won: bool,
    /// The engine can't take the step yet; the episode was cut (truncated).
    pub not_ported: bool,
    /// The engine refused the table's action; the episode was cut (truncated).
    pub refused: bool,
}

/// Envs stepped together on a thread pool.
pub struct VectorEnv {
    envs: Vec<GymEnv>,
    pool: rayon::ThreadPool,
}

impl VectorEnv {
    /// One env per reset spec (each must carry a sampler seed), on `threads` worker threads
    /// (0: one per core).
    pub fn new(resets: &[ResetSpec], config: EnvConfig, threads: usize) -> Result<Self, String> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| e.to_string())?;
        let envs = pool.install(|| {
            resets
                .par_iter()
                .map(|r| GymEnv::new(&r.spec, config, r.rng_seed.unwrap_or(0)))
                .collect::<Result<Vec<_>, _>>()
        })?;
        Ok(VectorEnv { envs, pool })
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    pub fn envs(&self) -> &[GymEnv] {
        &self.envs
    }

    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Reset every env, writing the observations (`len * GYM_OBS_LEN`) and action masks
    /// (`len * MAX_ACTIONS`).
    pub fn reset(
        &mut self,
        resets: &[ResetSpec],
        obs: &mut [f32],
        masks: &mut [u8],
    ) -> Result<(), String> {
        assert_eq!(resets.len(), self.envs.len());
        self.pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(resets.par_iter())
                .zip(obs.par_chunks_mut(GYM_OBS_LEN))
                .zip(masks.par_chunks_mut(MAX_ACTIONS))
                .try_for_each(|(((env, reset), o), m)| {
                    env.reset(&reset.spec, reset.rng_seed)?;
                    env.write_obs(o);
                    env.write_action_mask(m);
                    Ok(())
                })
        })
    }

    /// Step every env: one with a reset spec starts its next run (its output is all zeros),
    /// every other takes its action (an index into its table). An env the engine refuses ends
    /// its episode as truncated. Fails without stepping anything when an action is out of its
    /// table.
    pub fn step(
        &mut self,
        actions: &[usize],
        resets: &[Option<ResetSpec>],
        obs: &mut [f32],
        masks: &mut [u8],
        out: &mut [EnvOut],
    ) -> Result<(), String> {
        assert_eq!(actions.len(), self.envs.len());
        assert_eq!(resets.len(), self.envs.len());
        for (i, ((env, &a), reset)) in self.envs.iter().zip(actions).zip(resets).enumerate() {
            if reset.is_none() && a >= env.table().len() {
                return Err(format!(
                    "env {i}: action {a} is not in its action table ({} actions)",
                    env.table().len()
                ));
            }
        }
        self.pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(actions.par_iter())
                .zip(resets.par_iter())
                .zip(obs.par_chunks_mut(GYM_OBS_LEN))
                .zip(masks.par_chunks_mut(MAX_ACTIONS))
                .zip(out.par_iter_mut())
                .try_for_each(|(((((env, &a), reset), o), m), r)| {
                    *r = match reset {
                        Some(reset) => {
                            env.reset(&reset.spec, reset.rng_seed)?;
                            EnvOut::default()
                        }
                        None => step_one(env, a),
                    };
                    env.write_obs(o);
                    env.write_action_mask(m);
                    Ok(())
                })
        })
    }
}

/// One env's step, refusals cut into a truncated episode end.
fn step_one(env: &mut GymEnv, action: usize) -> EnvOut {
    let (result, refused) = match env.step(action) {
        Ok(r) => (r, false),
        Err(StepError::BadIndex(_)) => unreachable!("actions are checked before stepping"),
        Err(StepError::Convert(_) | StepError::Illegal(_)) => {
            let fa = *env.table().get(action).expect("checked index");
            (env.cut(fa), true)
        }
    };
    let ended = result.terminated() || result.truncated();
    EnvOut {
        reward: result.reward,
        terminated: result.terminated(),
        truncated: result.truncated(),
        episode_end: ended,
        ante_reached: if ended { env.max_ante } else { 0.0 },
        rounds_beaten: if ended { env.blinds_beaten } else { 0 },
        won: ended && env.episode_won,
        not_ported: result.not_ported.is_some() && !refused,
        refused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gym::Xoshiro256;

    fn reset(seed: &str, rng_seed: u64) -> ResetSpec {
        ResetSpec {
            spec: RunSpec {
                back: jackdaw_core::card::center_by_key("b_red").unwrap(),
                stake: 1,
                seed: seed.to_owned(),
                challenge: None,
            },
            rng_seed: Some(rng_seed),
        }
    }

    /// Run `n` envs for `steps` steps on `threads` threads with a seeded random policy; return
    /// every observation, mask and output.
    fn run(n: usize, threads: usize, steps: usize) -> (Vec<f32>, Vec<u8>, Vec<EnvOut>) {
        let resets: Vec<ResetSpec> = (0..n)
            .map(|i| reset(&format!("VEC{i}"), i as u64))
            .collect();
        let mut v = VectorEnv::new(&resets, EnvConfig::default(), threads).unwrap();
        let mut obs = vec![0f32; n * GYM_OBS_LEN];
        let mut masks = vec![0u8; n * MAX_ACTIONS];
        let mut out = vec![EnvOut::default(); n];
        v.reset(&resets, &mut obs, &mut masks).unwrap();
        let mut policy = Xoshiro256::seed_from_u64(99);
        let mut episodes = vec![0u64; n];
        let (mut all_obs, mut all_masks, mut all_out) = (Vec::new(), Vec::new(), Vec::new());
        let mut needs_reset = vec![false; n];
        for _ in 0..steps {
            let mut actions = vec![0usize; n];
            let mut next: Vec<Option<ResetSpec>> = vec![None; n];
            for i in 0..n {
                if needs_reset[i] {
                    episodes[i] += 1;
                    next[i] = Some(ResetSpec {
                        rng_seed: None,
                        ..reset(&format!("VEC{i}_{}", episodes[i]), 0)
                    });
                } else {
                    let len = v.envs()[i].table().len() as u64;
                    actions[i] = policy.below(len) as usize;
                }
            }
            v.step(&actions, &next, &mut obs, &mut masks, &mut out)
                .unwrap();
            for i in 0..n {
                needs_reset[i] = out[i].episode_end;
            }
            all_obs.extend_from_slice(&obs);
            all_masks.extend_from_slice(&masks);
            all_out.extend_from_slice(&out);
        }
        (all_obs, all_masks, all_out)
    }

    #[test]
    fn results_do_not_depend_on_threads() {
        let one = run(6, 1, 300);
        let four = run(6, 4, 300);
        assert!(
            one.2.iter().any(|o| o.episode_end),
            "some episode should end"
        );
        assert_eq!(one.1, four.1);
        assert_eq!(one.2, four.2);
        assert!(
            one.0
                .iter()
                .zip(&four.0)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn bad_actions_step_nothing() {
        let resets = vec![reset("BAD0", 0), reset("BAD1", 1)];
        let mut v = VectorEnv::new(&resets, EnvConfig::default(), 2).unwrap();
        let mut obs = vec![0f32; 2 * GYM_OBS_LEN];
        let mut masks = vec![0u8; 2 * MAX_ACTIONS];
        let mut out = vec![EnvOut::default(); 2];
        let before: Vec<_> = v.envs().iter().map(|e| e.game().snapshot()).collect();
        let err = v.step(&[0, 10_000], &[None, None], &mut obs, &mut masks, &mut out);
        assert!(err.is_err());
        let after: Vec<_> = v.envs().iter().map(|e| e.game().snapshot()).collect();
        assert_eq!(before, after);
    }
}
