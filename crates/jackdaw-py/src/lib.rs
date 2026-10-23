//! PyO3 bindings for jackdaw: the `jackdaw._core` extension module.
//!
//! This is the only crate that depends on PyO3; the engine crates stay Python-free. The
//! workspace's `default-members` leave it out, so plain `cargo build` / `cargo test` never link
//! libpython. maturin builds it (`uv sync`, `uv build`) against the stable ABI (`abi3-py312`), so
//! one wheel per platform serves CPython 3.12 and newer. Release builds keep `panic = "unwind"`,
//! so a Rust panic reaches Python as an exception instead of aborting the interpreter.

use jackdaw_env::observation::{Encoder, Observation};
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

mod bot;

pyo3::create_exception!(
    _core,
    IllegalActionError,
    pyo3::exceptions::PyValueError,
    "The action is not legal in this state; the game is unchanged."
);
pyo3::create_exception!(
    _core,
    NotPortedError,
    pyo3::exceptions::PyNotImplementedError,
    "The step needs game logic the Rust engine doesn't have yet; the game is unchanged."
);

/// A run in the Rust engine.
///
/// Snapshots and actions cross the boundary as JSON text: a snapshot is the canonical
/// `jackdaw.snapshot/1` tree (non-finite numbers as `{"$f": ...}`), an action its tape form
/// (`["PlayHand", [0, 2, 3]]`). `jackdaw.core.Game` wraps this with Python objects.
#[pyclass(name = "Game", module = "jackdaw._core")]
#[derive(Clone)]
struct Game {
    inner: jackdaw_core::game::Game,
    /// The observation encoder's buffers, reused by every `observation()` call.
    encoder: Encoder,
}

impl Game {
    fn wrap(inner: jackdaw_core::game::Game) -> Self {
        Game {
            inner,
            encoder: Encoder::new(),
        }
    }
}

fn value_error(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Row counts of the five entity arrays (hand, jokers, consumables, shop, pack).
type ObservationRows = (usize, usize, usize, usize, usize);

/// An encoded observation for Python: its `f32` values in native byte order, in a fresh
/// `bytearray` (so `np.frombuffer` gives writable arrays without another copy), and the row
/// counts.
fn observation_to_py<'py>(
    py: Python<'py>,
    obs: &Observation,
) -> PyResult<(Bound<'py, PyByteArray>, ObservationRows)> {
    let data = obs.data();
    let buf = PyByteArray::new_with(py, std::mem::size_of_val(data), |buf| {
        for (dst, x) in buf.as_chunks_mut::<4>().0.iter_mut().zip(data) {
            *dst = x.to_ne_bytes();
        }
        Ok(())
    })?;
    let [a, b, c, d, e] = obs.rows();
    Ok((buf, (a, b, c, d, e)))
}

#[pymethods]
impl Game {
    /// A new run at the blind-select screen (``back`` is ignored for a challenge).
    #[staticmethod]
    #[pyo3(signature = (back, stake, seed, challenge=None))]
    fn new(back: &str, stake: u8, seed: &str, challenge: Option<&str>) -> PyResult<Self> {
        use jackdaw_core::run::RunSpec;
        let back = jackdaw_core::card::center_by_key(back)
            .ok_or_else(|| value_error(format!("unknown back {back:?}")))?;
        let challenge = match challenge {
            Some(id) => Some(
                jackdaw_core::run::challenge_id(id)
                    .ok_or_else(|| value_error(format!("unknown challenge {id:?}")))?,
            ),
            None => None,
        };
        let spec = RunSpec {
            back,
            stake,
            seed: seed.to_owned(),
            challenge,
        };
        let inner = jackdaw_core::game::Game::new_run(&spec).map_err(value_error)?;
        Ok(Game::wrap(inner))
    }

    /// A run resting at a snapshot (JSON text).
    #[staticmethod]
    fn from_snapshot(text: &str) -> PyResult<Self> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(value_error)?;
        let inner = jackdaw_core::game::Game::from_snapshot(&value).map_err(value_error)?;
        Ok(Game::wrap(inner))
    }

    /// The canonical snapshot (JSON text).
    fn snapshot(&self) -> String {
        self.inner.snapshot().to_string()
    }

    /// The observation (encoder v1, `jackdaw.env.observation`) of the current state.
    ///
    /// Returns ``(data, rows)``: ``data`` is a ``bytearray`` of native-endian float32 values,
    /// the global context followed by the hand, joker, consumable, shop and pack rows, and
    /// ``rows`` their row counts. ``jackdaw.env.rust_observation`` wraps it as an
    /// ``Observation``.
    fn observation<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyByteArray>, ObservationRows)> {
        let obs = self.encoder.encode(&self.inner.gs);
        observation_to_py(py, obs)
    }

    /// Take one action (its tape form, JSON text) and play until the game waits again.
    /// On an error the game is unchanged.
    fn step(&mut self, action: &str) -> PyResult<()> {
        use jackdaw_core::action::Action;
        use jackdaw_core::game::StepError;
        let value: serde_json::Value = serde_json::from_str(action).map_err(value_error)?;
        let action = Action::from_json(&value).map_err(value_error)?;
        self.inner.step(&action).map_err(|e| match e {
            StepError::Illegal(m) => IllegalActionError::new_err(m),
            StepError::Unported(m) => NotPortedError::new_err(m),
        })
    }

    /// The legal actions here, as a JSON list of tape-form actions, in the Python engine's
    /// order (`jackdaw.engine.actions.get_legal_actions`). ``legality`` is ``"vanilla"`` (the
    /// default) or ``"balatrobot"``: only what the live BalatroBot also executes.
    #[pyo3(signature = (legality=None))]
    fn legal_actions(&self, legality: Option<&str>) -> PyResult<String> {
        bot::legal_actions(&self.inner, legality)
    }

    /// BalatroBot's gamestate view of the run (JSON text, as the bot encodes it).
    fn bot_view(&self) -> PyResult<String> {
        bot::bot_view(&self.inner)
    }

    /// The BalatroBot request that performs an action (tape form, JSON text):
    /// ``(method, params_json)``, or ``None`` for a sim-only action.
    fn bot_translate(&self, action: &str) -> PyResult<Option<(String, String)>> {
        bot::translate(&self.inner, action)
    }

    /// Apply one parity setup operation (JSON text, `tests/parity/harness.py`): ``force_boss``,
    /// ``action``, ``add`` (BalatroBot's ``add`` with the oracle's names) or ``set``.
    fn apply_setup(&mut self, op: &str) -> PyResult<()> {
        let value: serde_json::Value = serde_json::from_str(op).map_err(value_error)?;
        self.inner.apply_setup_op(&value).map_err(value_error)
    }

    /// An independent copy of the run.
    #[pyo3(name = "clone")]
    fn py_clone(&self) -> Self {
        self.clone()
    }

    /// The SHA-256 hex digest of the canonical snapshot's encoding.
    fn state_hash(&self) -> String {
        jackdaw_core::canon::digest(&self.inner.snapshot())
    }

    /// The decision point: ``blind_select``, ``selecting_hand``, ``round_eval``, ``shop``,
    /// ``pack_opening`` or ``game_over``.
    #[getter]
    fn phase(&self) -> &'static str {
        self.inner.gs.phase.name()
    }

    #[getter]
    fn ante(&self) -> f64 {
        self.inner.gs.round_resets.ante
    }

    #[getter]
    fn round(&self) -> f64 {
        self.inner.gs.round
    }

    #[getter]
    fn dollars(&self) -> f64 {
        self.inner.gs.dollars
    }

    #[getter]
    fn chips(&self) -> f64 {
        self.inner.gs.chips
    }

    #[getter]
    fn hands_left(&self) -> f64 {
        self.inner.gs.current_round.hands_left
    }

    #[getter]
    fn discards_left(&self) -> f64 {
        self.inner.gs.current_round.discards_left
    }

    #[getter]
    fn won(&self) -> bool {
        self.inner.gs.won
    }

    #[getter]
    fn seed(&self) -> String {
        self.inner.gs.rng.seed_str().to_owned()
    }
}

// Written by build.rs.
const BUILD_PROFILE: &str = env!("JACKDAW_BUILD_PROFILE");
const BUILD_OPT_LEVEL: &str = env!("JACKDAW_BUILD_OPT_LEVEL");
const BUILD_TARGET: &str = env!("JACKDAW_BUILD_TARGET");
const BUILD_RUSTC: &str = env!("JACKDAW_BUILD_RUSTC");

/// Native core of jackdaw, written in Rust.
#[pymodule]
mod _core {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    #[pymodule_export]
    use super::{Game, IllegalActionError, NotPortedError};

    #[pymodule_export]
    use super::bot::{BotSession, serve};

    /// The BalatroBot release the bot layer ports.
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const BALATROBOT_VERSION: &str = jackdaw_bot::BALATROBOT_VERSION;

    /// Version of the Rust crates; matches the `jackdaw` distribution version.
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    /// Describe how this extension was built.
    ///
    /// Keys: ``version``, ``profile`` (``"release"`` or ``"debug"``), ``opt_level``,
    /// ``debug_assertions``, ``target`` (the Rust target triple) and ``rustc``.
    #[pyfunction]
    fn _build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let info = PyDict::new(py);
        info.set_item("version", __version__)?;
        info.set_item("profile", super::BUILD_PROFILE)?;
        info.set_item("opt_level", super::BUILD_OPT_LEVEL)?;
        info.set_item("debug_assertions", cfg!(debug_assertions))?;
        info.set_item("target", super::BUILD_TARGET)?;
        info.set_item("rustc", super::BUILD_RUSTC)?;
        Ok(info)
    }

    /// Import a canonical snapshot (JSON text) into the Rust run state and export it again.
    ///
    /// Raises ``ValueError`` when the Rust state can't hold the snapshot. The parity tests use
    /// it to check that the Rust state keeps everything the snapshot records.
    #[pyfunction]
    fn _snapshot_roundtrip(text: &str) -> PyResult<String> {
        use pyo3::exceptions::PyValueError;
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| PyValueError::new_err(format!("snapshot: bad JSON: {e}")))?;
        let gs = jackdaw_core::snapshot::import(&value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(jackdaw_core::snapshot::export(&gs).to_string())
    }

    /// Start a new run in Rust and return its snapshot (JSON text) at the blind-select screen.
    ///
    /// ``back`` is a back key (``"b_red"``); a challenge (``"c_omelette_1"``) plays on the
    /// Challenge Deck. Raises ``ValueError`` for an unknown back, stake or challenge, or a run
    /// whose setup isn't ported yet.
    #[pyfunction]
    #[pyo3(signature = (back, stake, seed, challenge=None))]
    fn _new_run(back: &str, stake: u8, seed: &str, challenge: Option<&str>) -> PyResult<String> {
        use jackdaw_core::game::Game;
        use jackdaw_core::run::RunSpec;
        use pyo3::exceptions::PyValueError;
        let back = jackdaw_core::card::center_by_key(back)
            .ok_or_else(|| PyValueError::new_err(format!("unknown back {back:?}")))?;
        let challenge = match challenge {
            Some(id) => Some(
                jackdaw_core::run::challenge_id(id)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown challenge {id:?}")))?,
            ),
            None => None,
        };
        let spec = RunSpec {
            back,
            stake,
            seed: seed.to_owned(),
            challenge,
        };
        let game = Game::new_run(&spec).map_err(PyValueError::new_err)?;
        Ok(game.snapshot().to_string())
    }

    /// Encode a canonical snapshot (JSON text) with observation encoder v1
    /// (``jackdaw.env.observation.encode_observation``).
    ///
    /// Returns ``(data, rows)`` as ``Game.observation()`` does. Raises ``ValueError`` when the
    /// Rust state can't hold the snapshot.
    #[pyfunction]
    fn _encode_observation<'py>(
        py: Python<'py>,
        text: &str,
    ) -> PyResult<(Bound<'py, pyo3::types::PyByteArray>, super::ObservationRows)> {
        use pyo3::exceptions::PyValueError;
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| PyValueError::new_err(format!("snapshot: bad JSON: {e}")))?;
        let gs = jackdaw_core::snapshot::import(&value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut encoder = jackdaw_env::observation::Encoder::new();
        super::observation_to_py(py, encoder.encode(&gs))
    }

    /// The observation layout: the global context's length and the row width of the hand,
    /// joker, consumable, shop and pack arrays.
    #[pyfunction]
    fn _observation_layout() -> (usize, [usize; 5]) {
        use jackdaw_env::observation::{D_GLOBAL, ENTITY_WIDTHS};
        (D_GLOBAL, ENTITY_WIDTHS)
    }

    /// Center keys of the jokers the Rust engine has ported; a step that consults any other
    /// joker is refused as unported.
    #[pyfunction]
    fn _ported_jokers() -> Vec<&'static str> {
        jackdaw_core::joker::ported()
    }

    /// Import a snapshot, take one action (its tape form, JSON text) and export the result.
    ///
    /// Returns ``(status, snapshot)``: ``"ok"`` with the new snapshot, or ``"illegal: ..."`` /
    /// ``"unported: ..."`` with an empty string. The parity tests use it to step the Rust engine
    /// from any Python oracle state.
    #[pyfunction]
    fn _step_snapshot(text: &str, action: &str) -> PyResult<(String, String)> {
        use jackdaw_core::action::Action;
        use jackdaw_core::game::{Game, StepError};
        use pyo3::exceptions::PyValueError;
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| PyValueError::new_err(format!("snapshot: bad JSON: {e}")))?;
        let mut game =
            Game::from_snapshot(&value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let action: serde_json::Value = serde_json::from_str(action)
            .map_err(|e| PyValueError::new_err(format!("action: bad JSON: {e}")))?;
        let action = Action::from_json(&action).map_err(PyValueError::new_err)?;
        Ok(match game.step(&action) {
            Ok(()) => ("ok".to_owned(), game.snapshot().to_string()),
            Err(StepError::Illegal(m)) => (format!("illegal: {m}"), String::new()),
            Err(StepError::Unported(m)) => (format!("unported: {m}"), String::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    // Not run by plain `cargo test` (this crate is not a default member); `cargo test -p
    // jackdaw-py` runs it where the Python library is loadable.
    #[test]
    fn build_metadata_is_baked_in() {
        assert!(super::BUILD_TARGET.contains('-'));
        assert!(super::BUILD_RUSTC.starts_with("rustc "));
        assert!(["debug", "release"].contains(&super::BUILD_PROFILE));
    }
}
