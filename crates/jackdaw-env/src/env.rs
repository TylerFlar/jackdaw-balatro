//! One Gym environment on the Rust engine: `BalatroGymnasiumEnv` over `BalatroEnvironment`, with
//! a built-in reward. The Python wrapper and the vector env both drive it, so a vector env of one
//! steps exactly as the Gym wrapper does.
//!
//! The rules follow the Python env step for step: an action is an index into the action table;
//! the run is converted, stepped and summarised; `terminated` is game over or a won run (with
//! `end_on_win`), `truncated` is the step limit; the episode's best ante and blinds beaten are
//! tracked for the episode-end info; the next table is enumerated unless the episode ended.
//!
//! A step the engine can't take yet (not ported) leaves the run unchanged and ends the episode
//! as truncated, as the step limit does.

use jackdaw_core::game::{Game, StepError as CoreStepError};
use jackdaw_core::run::RunSpec;

use crate::action::{FactoredAction, to_engine};
use crate::gym::{ActionTable, MAX_ACTIONS, Xoshiro256};
use crate::mask::{ActionMask, Legality};
use crate::observation::{D_GLOBAL, ENTITY_WIDTHS, Encoder};
use crate::reward::{Reward, StateSummary, Transition, run_won};

/// The most rows of each entity array the Gym observation holds (`balatro_game_spec()`'s
/// `max_count`: hand cards, jokers, consumables, shop items, pack cards).
pub const MAX_COUNTS: [usize; 5] = [8, 5, 2, 10, 5];

/// The Gym observation, flattened: the global context, each entity array padded to its
/// `MAX_COUNTS` rows, then the five entity counts.
pub const GYM_OBS_LEN: usize = D_GLOBAL
    + MAX_COUNTS[0] * ENTITY_WIDTHS[0]
    + MAX_COUNTS[1] * ENTITY_WIDTHS[1]
    + MAX_COUNTS[2] * ENTITY_WIDTHS[2]
    + MAX_COUNTS[3] * ENTITY_WIDTHS[3]
    + MAX_COUNTS[4] * ENTITY_WIDTHS[4]
    + 5;

/// The env's settings (`BalatroEnvironment`'s `max_steps` and `end_on_win`, and the reward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvConfig {
    pub max_steps: u64,
    pub end_on_win: bool,
    pub reward: Reward,
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig {
            max_steps: 10_000,
            end_on_win: true,
            reward: Reward::Sparse,
        }
    }
}

/// Why a step was refused; the run is unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepError {
    /// The action index is past the end of the table.
    BadIndex(usize),
    /// The table's action has no engine form here.
    Convert(String),
    /// The engine refused the action.
    Illegal(String),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::BadIndex(i) => write!(f, "action {i} is not in the action table"),
            StepError::Convert(m) | StepError::Illegal(m) => f.write_str(m),
        }
    }
}

/// What a step did.
#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    pub action: FactoredAction,
    pub transition: Transition,
    pub reward: f64,
    /// The engine can't take the step yet: the run is unchanged and the episode was cut.
    pub not_ported: Option<String>,
}

impl StepResult {
    pub fn terminated(&self) -> bool {
        self.transition.terminated
    }

    pub fn truncated(&self) -> bool {
        self.transition.truncated
    }
}

/// One Gym environment.
#[derive(Clone, Debug)]
pub struct GymEnv {
    game: Game,
    encoder: Encoder,
    mask: ActionMask,
    table: ActionTable,
    rng: Xoshiro256,
    pub config: EnvConfig,
    /// `BalatroEnvironment._step_count` (steps this episode, for the step limit).
    step_count: u64,
    /// `BalatroEnvironment.episode_length`.
    pub episode_length: u64,
    /// `BalatroEnvironment.episode_ante` (the ante after the last step).
    pub episode_ante: f64,
    /// `BalatroEnvironment.episode_won`.
    pub episode_won: bool,
    /// The best ante this episode (`balatro/ante_reached`).
    pub max_ante: f64,
    /// Blinds beaten this episode (`balatro/rounds_beaten`).
    pub blinds_beaten: u64,
    state: StateSummary,
    ended: bool,
}

impl GymEnv {
    /// An env at the start of `spec`'s run, its sampler seeded with `rng_seed`.
    pub fn new(spec: &RunSpec, config: EnvConfig, rng_seed: u64) -> Result<GymEnv, String> {
        let game = Game::new_run(spec)?;
        let state = StateSummary::of(&game);
        let mut env = GymEnv {
            game,
            encoder: Encoder::new(),
            mask: ActionMask::new(),
            table: ActionTable::new(),
            rng: Xoshiro256::seed_from_u64(rng_seed),
            config,
            step_count: 0,
            episode_length: 0,
            episode_ante: 1.0,
            episode_won: false,
            max_ante: 1.0,
            blinds_beaten: 0,
            state,
            ended: false,
        };
        env.refresh();
        Ok(env)
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn state(&self) -> &StateSummary {
        &self.state
    }

    pub fn table(&self) -> &ActionTable {
        &self.table
    }

    pub fn rng_state(&self) -> [u64; 4] {
        self.rng.state()
    }

    /// The episode is over: the table is empty until the next reset.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// Start a new run (`reset`); reseed the sampler when `rng_seed` is given.
    pub fn reset(&mut self, spec: &RunSpec, rng_seed: Option<u64>) -> Result<(), String> {
        if let Some(seed) = rng_seed {
            self.rng = Xoshiro256::seed_from_u64(seed);
        }
        self.game = Game::new_run(spec)?;
        self.step_count = 0;
        self.episode_length = 0;
        self.episode_ante = 1.0;
        self.episode_won = false;
        self.max_ante = 1.0;
        self.blinds_beaten = 0;
        self.state = StateSummary::of(&self.game);
        self.ended = false;
        self.refresh();
        Ok(())
    }

    /// The mask and the action table of the current state.
    fn refresh(&mut self) {
        self.mask.compute(&self.game, Legality::Vanilla);
        self.table.enumerate(&self.mask, &self.game, &mut self.rng);
    }

    /// Take the table's action `index`.
    pub fn step(&mut self, index: usize) -> Result<StepResult, StepError> {
        let action = *self.table.get(index).ok_or(StepError::BadIndex(index))?;
        let engine_action = to_engine(&action, &self.game).map_err(|e| StepError::Convert(e.0))?;
        let not_ported = match self.game.step(&engine_action) {
            Ok(()) => None,
            Err(CoreStepError::Illegal(m)) => return Err(StepError::Illegal(m)),
            Err(CoreStepError::Unported(m)) => Some(m),
        };
        Ok(self.finish(action, not_ported))
    }

    /// End the episode here as truncated without a step (the run is unchanged): the vector env's
    /// answer to an action the engine refused.
    pub fn cut(&mut self, action: FactoredAction) -> StepResult {
        self.finish(action, Some(String::new()))
    }

    fn finish(&mut self, action: FactoredAction, not_ported: Option<String>) -> StepResult {
        self.episode_length += 1;
        self.step_count += 1;
        let prev = self.state;
        let curr = StateSummary::of(&self.game);
        let won = run_won(&self.game);
        let terminated = curr.done || (self.config.end_on_win && won);
        let truncated = self.step_count >= self.config.max_steps || not_ported.is_some();
        self.episode_ante = self.game.gs.round_resets.ante;
        self.episode_won = won;
        let transition = Transition {
            prev,
            curr,
            terminated,
            truncated,
        };
        let reward = self.config.reward.of(&transition);
        // Python's max(best, ante): the first unless the second is larger.
        if curr.ante > self.max_ante {
            self.max_ante = curr.ante;
        }
        if transition.blind_beaten() {
            self.blinds_beaten += 1;
        }
        self.state = curr;
        if terminated || truncated {
            self.ended = true;
            self.table.clear();
        } else {
            self.refresh();
        }
        StepResult {
            action,
            transition,
            reward,
            not_ported,
        }
    }

    /// The Gym observation of the current state, flattened ([`GYM_OBS_LEN`] values): the global
    /// context, each entity array's first rows padded with zeros, the entity counts.
    pub fn write_obs(&mut self, out: &mut [f32]) {
        let obs = self.encoder.encode(&self.game.gs);
        let (global, mut rest) = out.split_at_mut(D_GLOBAL);
        global.copy_from_slice(obs.global());
        let rows = obs.rows();
        let mut counts = [0f32; 5];
        for k in 0..5 {
            let width = ENTITY_WIDTHS[k];
            let (slot, tail) = rest.split_at_mut(MAX_COUNTS[k] * width);
            let n = rows[k].min(MAX_COUNTS[k]);
            slot[..n * width].copy_from_slice(&obs.entity(k)[..n * width]);
            slot[n * width..].fill(0.0);
            counts[k] = n as f32;
            rest = tail;
        }
        rest.copy_from_slice(&counts);
    }

    /// The Gym action mask: the table's actions are the first legal slots.
    pub fn write_action_mask(&self, out: &mut [u8]) {
        let n = self.table.len().min(MAX_ACTIONS);
        out[..n].fill(1);
        out[n..].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(seed: &str) -> RunSpec {
        RunSpec {
            back: jackdaw_core::card::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: seed.to_owned(),
            challenge: None,
        }
    }

    #[test]
    fn layout_is_614_values() {
        assert_eq!(GYM_OBS_LEN, 614);
    }

    #[test]
    fn steps_until_the_episode_ends() {
        let mut env = GymEnv::new(&spec("ENVTEST"), EnvConfig::default(), 7).unwrap();
        let mut obs = vec![0f32; GYM_OBS_LEN];
        let mut steps = 0;
        let mut rng = Xoshiro256::seed_from_u64(1);
        while !env.ended() && steps < 2000 {
            env.write_obs(&mut obs);
            assert_eq!(obs[GYM_OBS_LEN - 5..].len(), 5);
            let n = env.table().len() as u64;
            assert!(n > 0);
            let r = env.step(rng.below(n) as usize).unwrap();
            assert_eq!(r.terminated() || r.truncated(), env.ended());
            steps += 1;
        }
        assert!(env.ended());
        assert!(env.table().is_empty());
    }
}
