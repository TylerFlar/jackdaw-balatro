//! PyO3 bindings for jackdaw: the `jackdaw._core` extension module.
//!
//! This is the only crate that depends on PyO3; the engine crates stay Python-free. The
//! workspace's `default-members` leave it out, so plain `cargo build` / `cargo test` never link
//! libpython. maturin builds it (`uv sync`, `uv build`) against the stable ABI (`abi3-py312`), so
//! one wheel per platform serves CPython 3.12 and newer. Release builds keep `panic = "unwind"`,
//! so a Rust panic reaches Python as an exception instead of aborting the interpreter.

use pyo3::prelude::*;

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
