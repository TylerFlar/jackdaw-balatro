//! The Gym environments for Python: one env (`GymEnv`, behind `BalatroGymnasiumEnv`) and a batch
//! stepped on worker threads with the GIL released (`VectorEnv`, behind `BalatroVectorEnv`).
//! Both run `jackdaw_env::env::GymEnv`, so a vector env of one steps exactly as the Gym env.

use jackdaw_core::run::RunSpec;
use jackdaw_env::env::{EnvConfig, GYM_OBS_LEN, GymEnv as CoreGymEnv, StepError, StepResult};
use jackdaw_env::gym::MAX_ACTIONS;
use jackdaw_env::reward::{Reward, ShapedWeights, StateSummary};
use jackdaw_env::vector::{EnvOut, ResetSpec, VectorEnv as CoreVectorEnv};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use crate::env::{FactoredParts, factored_to_py};
use crate::{IllegalActionError, value_error};

/// A run's start: back key, stake, seed and challenge id.
pub fn run_spec(back: &str, stake: u8, seed: &str, challenge: Option<&str>) -> PyResult<RunSpec> {
    let back = jackdaw_core::card::center_by_key(back)
        .ok_or_else(|| value_error(format!("unknown back {back:?}")))?;
    let challenge = match challenge {
        Some(id) => Some(
            jackdaw_core::run::challenge_id(id)
                .ok_or_else(|| value_error(format!("unknown challenge {id:?}")))?,
        ),
        None => None,
    };
    Ok(RunSpec {
        back,
        stake,
        seed: seed.to_owned(),
        challenge,
    })
}

/// A built-in reward by name (`"sparse"`, `"blind_progress"`, `"shaped"` with its eight
/// weights, or `"none"`).
fn reward(kind: &str, weights: Option<[f64; 8]>) -> PyResult<Reward> {
    Ok(match kind {
        "sparse" => Reward::Sparse,
        "blind_progress" => Reward::BlindProgress,
        "none" => Reward::None,
        "shaped" => {
            let [
                step_cost,
                shop_step_cost,
                blind_beaten,
                boss_beaten,
                hands_left,
                score_progress,
                win,
                loss,
            ] = weights.ok_or_else(|| value_error("a shaped reward needs its weights"))?;
            Reward::Shaped(ShapedWeights {
                step_cost,
                shop_step_cost,
                blind_beaten,
                boss_beaten,
                hands_left,
                score_progress,
                win,
                loss,
            })
        }
        other => return Err(value_error(format!("unknown reward {other:?}"))),
    })
}

fn config(
    max_steps: u64,
    end_on_win: bool,
    reward_kind: &str,
    weights: Option<[f64; 8]>,
) -> PyResult<EnvConfig> {
    Ok(EnvConfig {
        max_steps,
        end_on_win,
        reward: reward(reward_kind, weights)?,
    })
}

/// A number as Python sees the engine's: an `int` when it is a whole number, else a `float`.
#[derive(IntoPyObject)]
pub enum Num {
    I(i64),
    F(f64),
}

pub fn num(x: f64) -> Num {
    if x.is_finite() && x.fract() == 0.0 && x.abs() < 9.0e15 {
        Num::I(x as i64)
    } else {
        Num::F(x)
    }
}

/// A `GameState` for Python: `(phase, blind_on_deck, won, done, [ante, round, dollars,
/// hands_left, discards_left, hand_size, joker_slots, consumable_slots, blind_chips], chips)`.
pub type StateParts = (
    &'static str,
    Option<&'static str>,
    bool,
    bool,
    [Num; 9],
    f64,
);

pub fn state_to_py(s: &StateSummary) -> StateParts {
    (
        s.phase.name(),
        s.blind_on_deck.map(|b| b.name()),
        s.won,
        s.done,
        [
            num(s.ante),
            num(s.round),
            num(s.dollars),
            num(s.hands_left),
            num(s.discards_left),
            num(s.hand_size),
            num(s.joker_slots),
            num(s.consumable_slots),
            num(s.blind_chips),
        ],
        s.chips,
    )
}

fn obs_bytes<'py>(py: Python<'py>, env: &mut CoreGymEnv) -> PyResult<Bound<'py, PyByteArray>> {
    let mut obs = [0f32; GYM_OBS_LEN];
    env.write_obs(&mut obs);
    PyByteArray::new_with(py, GYM_OBS_LEN * 4, |buf| {
        for (dst, x) in buf.as_chunks_mut::<4>().0.iter_mut().zip(&obs) {
            *dst = x.to_ne_bytes();
        }
        Ok(())
    })
}

/// A run's start for a reset: `(back, stake, seed, challenge, rng_seed)`.
type ResetTuple = (String, u8, String, Option<String>, Option<u64>);

fn reset_spec(t: &ResetTuple) -> PyResult<ResetSpec> {
    Ok(ResetSpec {
        spec: run_spec(&t.0, t.1, &t.2, t.3.as_deref())?,
        rng_seed: t.4,
    })
}

/// What a Gym step returns: `(obs, reward, terminated, truncated, n_actions, episode,
/// not_ported)`, `episode` being `(ante_reached, rounds_beaten, won)` at an episode end.
type StepParts<'py> = (
    Bound<'py, PyByteArray>,
    f64,
    bool,
    bool,
    usize,
    Option<(Num, u64, bool)>,
    Option<String>,
);

/// One Gym environment on the Rust engine (`BalatroGymnasiumEnv`'s engine).
#[pyclass(name = "GymEnv", module = "jackdaw._core")]
pub struct GymEnv {
    inner: CoreGymEnv,
    last: Option<StepResult>,
}

#[pymethods]
impl GymEnv {
    /// An env at the start of a run. ``reward`` is ``"sparse"``, ``"blind_progress"``,
    /// ``"shaped"`` (with ``weights``: step_cost, shop_step_cost, blind_beaten, boss_beaten,
    /// hands_left, score_progress, win, loss) or ``"none"``.
    #[new]
    #[pyo3(signature = (back, stake, seed, challenge, rng_seed, max_steps, end_on_win, reward, weights=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        back: &str,
        stake: u8,
        seed: &str,
        challenge: Option<&str>,
        rng_seed: u64,
        max_steps: u64,
        end_on_win: bool,
        reward: &str,
        weights: Option<[f64; 8]>,
    ) -> PyResult<Self> {
        let spec = run_spec(back, stake, seed, challenge)?;
        let cfg = config(max_steps, end_on_win, reward, weights)?;
        let inner = CoreGymEnv::new(&spec, cfg, rng_seed).map_err(value_error)?;
        Ok(GymEnv { inner, last: None })
    }

    /// Start a new run; reseed the sampler when ``rng_seed`` is given. Returns the observation
    /// (``bytearray`` of float32) and the number of actions.
    #[pyo3(signature = (back, stake, seed, challenge=None, rng_seed=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        back: &str,
        stake: u8,
        seed: &str,
        challenge: Option<&str>,
        rng_seed: Option<u64>,
    ) -> PyResult<(Bound<'py, PyByteArray>, usize)> {
        let spec = run_spec(back, stake, seed, challenge)?;
        self.inner.reset(&spec, rng_seed).map_err(value_error)?;
        self.last = None;
        Ok((obs_bytes(py, &mut self.inner)?, self.inner.table().len()))
    }

    /// Take action ``index`` of the table. Raises ``IndexError`` for an index past the table and
    /// ``IllegalActionError`` when the engine refuses the action (the run is unchanged).
    fn step<'py>(&mut self, py: Python<'py>, index: usize) -> PyResult<StepParts<'py>> {
        let r = self.inner.step(index).map_err(|e| match e {
            StepError::BadIndex(_) => PyIndexError::new_err(e.to_string()),
            StepError::Convert(m) => PyValueError::new_err(m),
            StepError::Illegal(m) => IllegalActionError::new_err(m),
        })?;
        let ended = r.terminated() || r.truncated();
        let episode = ended.then(|| {
            (
                num(self.inner.max_ante),
                self.inner.blinds_beaten,
                self.inner.episode_won,
            )
        });
        let out = (
            obs_bytes(py, &mut self.inner)?,
            r.reward,
            r.terminated(),
            r.truncated(),
            self.inner.table().len(),
            episode,
            r.not_ported.clone(),
        );
        self.last = Some(r);
        Ok(out)
    }

    /// The last step as a reward sees it: ``(prev, curr, action)``, the states as
    /// ``state()`` gives them and the action as ``(action_type, entity_target, card_target)``.
    fn last_transition(&self) -> Option<(StateParts, StateParts, FactoredParts)> {
        self.last.as_ref().map(|r| {
            (
                state_to_py(&r.transition.prev),
                state_to_py(&r.transition.curr),
                factored_to_py(&r.action),
            )
        })
    }

    /// The current state's summary (``GameState``).
    fn state(&self) -> StateParts {
        state_to_py(self.inner.state())
    }

    /// The action table: ``(action_type, entity_target, card_target)`` per action.
    fn action_table(&self) -> Vec<FactoredParts> {
        self.inner
            .table()
            .actions()
            .iter()
            .map(factored_to_py)
            .collect()
    }

    /// Action ``index`` of the table.
    fn action(&self, index: usize) -> PyResult<FactoredParts> {
        self.inner
            .table()
            .get(index)
            .map(factored_to_py)
            .ok_or_else(|| PyIndexError::new_err(format!("no action {index}")))
    }

    /// A copy of the run.
    fn game(&self) -> crate::Game {
        crate::Game::wrap(self.inner.game().clone())
    }

    #[getter]
    fn n_actions(&self) -> usize {
        self.inner.table().len()
    }

    #[getter]
    fn rng_state(&self) -> [u64; 4] {
        self.inner.rng_state()
    }

    #[getter]
    fn episode_won(&self) -> bool {
        self.inner.episode_won
    }

    #[getter]
    fn episode_ante(&self) -> Num {
        num(self.inner.episode_ante)
    }

    #[getter]
    fn episode_length(&self) -> u64 {
        self.inner.episode_length
    }

    #[getter]
    fn ended(&self) -> bool {
        self.inner.ended()
    }
}

/// Envs stepped together on worker threads, the GIL released (`BalatroVectorEnv`'s engine).
///
/// Outputs go to caller-owned buffers: observations (``num_envs x 614`` float32), action masks
/// (``num_envs x 500`` bytes) and per-env stats (``num_envs x 9`` float64: reward, terminated,
/// truncated, episode_end, ante_reached, rounds_beaten, won, not_ported, refused).
#[pyclass(name = "VectorEnv", module = "jackdaw._core")]
pub struct VectorEnv {
    inner: CoreVectorEnv,
    obs: Vec<f32>,
    masks: Vec<u8>,
    out: Vec<EnvOut>,
    stats: Vec<f64>,
}

/// Columns of the stats buffer.
pub const STATS: usize = 9;

#[pymethods]
impl VectorEnv {
    /// One env per reset tuple ``(back, stake, seed, challenge, rng_seed)``, on ``threads``
    /// worker threads (0: one per core).
    #[new]
    #[pyo3(signature = (resets, max_steps, end_on_win, reward, weights=None, threads=0))]
    fn new(
        py: Python<'_>,
        resets: Vec<ResetTuple>,
        max_steps: u64,
        end_on_win: bool,
        reward: &str,
        weights: Option<[f64; 8]>,
        threads: usize,
    ) -> PyResult<Self> {
        let specs = resets
            .iter()
            .map(reset_spec)
            .collect::<PyResult<Vec<_>>>()?;
        let cfg = config(max_steps, end_on_win, reward, weights)?;
        let inner = py
            .detach(|| CoreVectorEnv::new(&specs, cfg, threads))
            .map_err(value_error)?;
        let n = inner.len();
        Ok(VectorEnv {
            inner,
            obs: vec![0.0; n * GYM_OBS_LEN],
            masks: vec![0; n * MAX_ACTIONS],
            out: vec![EnvOut::default(); n],
            stats: vec![0.0; n * STATS],
        })
    }

    /// Reset every env; write the observations and action masks.
    fn reset(
        &mut self,
        py: Python<'_>,
        resets: Vec<ResetTuple>,
        obs: PyBuffer<f32>,
        masks: PyBuffer<u8>,
    ) -> PyResult<()> {
        let specs = resets
            .iter()
            .map(reset_spec)
            .collect::<PyResult<Vec<_>>>()?;
        if specs.len() != self.inner.len() {
            return Err(value_error("one reset per env"));
        }
        let VectorEnv {
            inner,
            obs: o,
            masks: m,
            ..
        } = self;
        py.detach(|| inner.reset(&specs, o, m))
            .map_err(value_error)?;
        obs.copy_from_slice(py, &self.obs)?;
        masks.copy_from_slice(py, &self.masks)?;
        Ok(())
    }

    /// Step every env: one with a reset tuple in ``resets`` starts its next run instead (its
    /// stats are zero), every other takes its action from ``actions`` (int64, an index into its
    /// table). Writes the observations, action masks and stats. Raises ``IndexError`` without
    /// stepping anything when an action is past its env's table.
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyBuffer<i64>,
        resets: Vec<Option<ResetTuple>>,
        obs: PyBuffer<f32>,
        masks: PyBuffer<u8>,
        stats: PyBuffer<f64>,
    ) -> PyResult<()> {
        let n = self.inner.len();
        let actions: Vec<usize> = actions
            .to_vec(py)?
            .into_iter()
            .map(|a| usize::try_from(a).map_err(|_| PyIndexError::new_err(format!("action {a}"))))
            .collect::<PyResult<_>>()?;
        if actions.len() != n || resets.len() != n {
            return Err(value_error("one action and one reset slot per env"));
        }
        let specs = resets
            .iter()
            .map(|r| r.as_ref().map(reset_spec).transpose())
            .collect::<PyResult<Vec<_>>>()?;
        let VectorEnv {
            inner,
            obs: o,
            masks: m,
            out,
            stats: s,
        } = self;
        py.detach(|| inner.step(&actions, &specs, o, m, out))
            .map_err(PyIndexError::new_err)?;
        for (row, e) in s.as_chunks_mut::<STATS>().0.iter_mut().zip(out.iter()) {
            *row = [
                e.reward,
                f64::from(u8::from(e.terminated)),
                f64::from(u8::from(e.truncated)),
                f64::from(u8::from(e.episode_end)),
                e.ante_reached,
                e.rounds_beaten as f64,
                f64::from(u8::from(e.won)),
                f64::from(u8::from(e.not_ported)),
                f64::from(u8::from(e.refused)),
            ];
        }
        obs.copy_from_slice(py, &self.obs)?;
        masks.copy_from_slice(py, &self.masks)?;
        stats.copy_from_slice(py, &self.stats)?;
        Ok(())
    }

    /// A copy of env ``i``'s run.
    fn game(&self, i: usize) -> PyResult<crate::Game> {
        let env = self
            .inner
            .envs()
            .get(i)
            .ok_or_else(|| PyIndexError::new_err(format!("no env {i}")))?;
        Ok(crate::Game::wrap(env.game().clone()))
    }

    /// Env ``i``'s action table.
    fn action_table(&self, i: usize) -> PyResult<Vec<FactoredParts>> {
        let env = self
            .inner
            .envs()
            .get(i)
            .ok_or_else(|| PyIndexError::new_err(format!("no env {i}")))?;
        Ok(env.table().actions().iter().map(factored_to_py).collect())
    }

    /// Env ``i``'s current state summary (``GameState``).
    fn state(&self, i: usize) -> PyResult<StateParts> {
        let env = self
            .inner
            .envs()
            .get(i)
            .ok_or_else(|| PyIndexError::new_err(format!("no env {i}")))?;
        Ok(state_to_py(env.state()))
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }
}
