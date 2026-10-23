//! Python bindings for the BalatroBot layer (`jackdaw-bot`): `BotSession`, `serve`, and the
//! helpers behind `Game.bot_view()` / `Game.legal_actions(legality)`.
//!
//! Requests cross the boundary as JSON text and are decoded with BalatroBot's own JSON reader
//! (Steamodded's rxi `json.lua` semantics), so a `BotSession` sees exactly what `jackdaw serve`
//! would. Responses are rxi-encoded JSON text.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use jackdaw_bot::json::{self, Json};
use jackdaw_bot::view::{self, Source};
use jackdaw_bot::{Legality, Session, SessionConfig};
use jackdaw_core::action::Action;

use super::Game;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `json.encode(view)`; a view holding NaN or ±inf can't be encoded (the live encoder raises).
fn encode(v: &Json) -> PyResult<String> {
    json::encode(v).map_err(|e| value_error(format!("Failed to encode response: {e}")))
}

/// The gamestate view of a run, as JSON text.
pub(crate) fn bot_view(game: &jackdaw_core::game::Game) -> PyResult<String> {
    encode(&view::gamestate(Source::Run(game)))
}

pub(crate) fn parse_legality(name: Option<&str>) -> PyResult<Legality> {
    match name {
        None => Ok(Legality::Vanilla),
        Some(n) => Legality::from_name(n)
            .ok_or_else(|| value_error(format!("unknown legality {n:?} (vanilla, balatrobot)"))),
    }
}

/// The legal actions under `legality`, as a JSON list of tape-form actions.
pub(crate) fn legal_actions(
    game: &jackdaw_core::game::Game,
    legality: Option<&str>,
) -> PyResult<String> {
    let list: Vec<serde_json::Value> = jackdaw_bot::legal_actions(game, parse_legality(legality)?)
        .iter()
        .map(Action::to_json)
        .collect();
    Ok(serde_json::Value::Array(list).to_string())
}

/// The BalatroBot request for an action (tape form, JSON text): `(method, params JSON)`, or
/// `None` for a sim-only action.
pub(crate) fn translate(
    game: &jackdaw_core::game::Game,
    action: &str,
) -> PyResult<Option<(String, String)>> {
    let value: serde_json::Value = serde_json::from_str(action).map_err(value_error)?;
    let action = Action::from_json(&value).map_err(value_error)?;
    let Some((method, params)) = jackdaw_bot::translate(game, &action) else {
        return Ok(None);
    };
    let params = json::from_value(&jackdaw_bot::lua::Value::Table(params)).map_err(value_error)?;
    Ok(Some((method.to_owned(), encode(&params)?)))
}

/// A BalatroBot session: the simulator as one game with BalatroBot loaded.
///
/// ``call(method, params_json)`` runs a request through the dispatcher and returns
/// ``(ok, json_text)``: the result, or an error object ``{"code", "name", "message"}``.
/// ``handle_body`` takes a whole JSON-RPC 2.0 request body, as ``jackdaw serve`` does.
#[pyclass(name = "BotSession", module = "jackdaw._core")]
#[derive(Clone)]
pub struct BotSession {
    inner: Session,
}

fn config(fs_root: Option<PathBuf>, debug: bool) -> SessionConfig {
    SessionConfig {
        fs_root,
        debug_endpoints: debug,
    }
}

fn error_text(e: &jackdaw_bot::errors::BotError) -> String {
    json::encode(&Json::obj([
        ("code", Json::Num(e.name.code() as f64)),
        ("name", Json::str(e.name.name())),
        ("message", Json::str(e.message.clone())),
    ]))
    .expect("an error object encodes")
}

#[pymethods]
impl BotSession {
    /// A session at the main menu. ``fs_root`` confines ``save``/``load`` paths; ``debug``
    /// registers BalatroBot's test endpoints.
    #[new]
    #[pyo3(signature = (fs_root=None, debug=false))]
    fn new(fs_root: Option<PathBuf>, debug: bool) -> Self {
        BotSession {
            inner: Session::new(config(fs_root, debug)),
        }
    }

    /// A session resting in (a copy of) ``game``.
    #[staticmethod]
    #[pyo3(signature = (game, fs_root=None, debug=false))]
    fn from_game(game: &Game, fs_root: Option<PathBuf>, debug: bool) -> Self {
        BotSession {
            inner: Session::with_game(game.inner.clone(), config(fs_root, debug)),
        }
    }

    /// Run one request (``params`` is JSON text): ``(True, result)`` or ``(False, error)``.
    #[pyo3(signature = (method, params="{}"))]
    fn call(&mut self, method: &str, params: &str) -> PyResult<(bool, String)> {
        let value = json::decode(params.as_bytes()).map_err(value_error)?;
        Ok(match self.inner.dispatch(method, value.as_ref()) {
            Ok(result) => match json::encode(&result) {
                Ok(text) => (true, text),
                Err(e) => (
                    false,
                    error_text(&jackdaw_bot::errors::BotError::internal(format!(
                        "Failed to encode response: {e}"
                    ))),
                ),
            },
            Err(e) => (false, error_text(&e)),
        })
    }

    /// Validate one request as BALATROBOT legality does (schema, state, the endpoint's own
    /// checks) without running it: ``(True, "")`` or ``(False, error)``.
    #[pyo3(signature = (method, params="{}"))]
    fn check(&mut self, method: &str, params: &str) -> PyResult<(bool, String)> {
        let value = json::decode(params.as_bytes()).map_err(value_error)?;
        Ok(match self.inner.check(method, value.as_ref()) {
            Ok(()) => (true, String::new()),
            Err(e) => (false, error_text(&e)),
        })
    }

    /// A JSON-RPC 2.0 request body in, the response body out.
    fn handle_body(&mut self, body: &[u8]) -> String {
        self.inner.handle_body(body)
    }

    /// The gamestate view (JSON text).
    fn view(&self) -> PyResult<String> {
        encode(&self.inner.view())
    }

    /// The view's ``state`` (``MENU``, ``BLIND_SELECT``, ..., ``UNKNOWN``).
    fn state(&self) -> &'static str {
        view::state_name(self.inner.source())
    }

    /// A copy of the run in progress, if any.
    fn game(&self) -> Option<Game> {
        self.inner.game().cloned().map(Game::wrap)
    }

    /// The run's canonical snapshot (JSON text), if a run is in progress.
    fn snapshot(&self) -> Option<String> {
        self.inner.game().map(|g| g.snapshot().to_string())
    }

    /// The run's legal actions under ``legality`` (``"vanilla"`` or ``"balatrobot"``), as a JSON
    /// list of tape-form actions; empty without a run.
    #[pyo3(signature = (legality=None))]
    fn legal_actions(&self, legality: Option<&str>) -> PyResult<String> {
        match self.inner.game() {
            Some(g) => legal_actions(g, legality),
            None => Ok("[]".to_owned()),
        }
    }

    /// The request for an action (tape form, JSON text) in the current run: ``(method,
    /// params_json)``, or ``None`` for a sim-only action.
    fn translate(&self, action: &str) -> PyResult<Option<(String, String)>> {
        let game = self
            .inner
            .game()
            .ok_or_else(|| value_error("no run in progress"))?;
        translate(game, action)
    }

    #[pyo3(name = "clone")]
    fn py_clone(&self) -> Self {
        self.clone()
    }
}

/// Serve BalatroBot's JSON-RPC API over HTTP on ``host:port`` until interrupted (Ctrl+C).
///
/// A non-loopback ``host`` needs ``fs_root``. ``ready`` (if given) is called with the bound port
/// once the server listens (``port=0`` picks a free one).
#[pyfunction]
#[pyo3(signature = (host="127.0.0.1", port=12346, fs_root=None, debug=false, ready=None))]
pub fn serve(
    py: Python<'_>,
    host: &str,
    port: u16,
    fs_root: Option<PathBuf>,
    debug: bool,
    ready: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let session = Session::new(config(fs_root, debug));
    let mut server = jackdaw_bot::server::Server::bind(host, port, session).map_err(value_error)?;
    if let Some(cb) = ready {
        cb.call1((server.port(),))?;
    }
    let mut interrupted: Option<PyErr> = None;
    let mut ticks = 0u32;
    let result = py.detach(|| {
        server.run(|| {
            ticks += 1;
            if !ticks.is_multiple_of(20) {
                return false;
            }
            Python::attach(|py| match py.check_signals() {
                Ok(()) => false,
                Err(e) => {
                    interrupted = Some(e);
                    true
                }
            })
        })
    });
    if let Some(e) = interrupted {
        return Err(e);
    }
    result.map_err(value_error)
}
