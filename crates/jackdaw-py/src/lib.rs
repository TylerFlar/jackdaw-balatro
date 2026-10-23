//! PyO3 bindings for jackdaw: the `jackdaw._core` extension module.
//!
//! This is the only crate that depends on PyO3; the engine crates stay Python-free. The
//! workspace's `default-members` leave it out, so plain `cargo build` / `cargo test` never link
//! libpython. maturin builds it (`uv sync`, `uv build`) against the stable ABI (`abi3-py312`), so
//! one wheel per platform serves CPython 3.12 and newer. Release builds keep `panic = "unwind"`,
//! so a Rust panic reaches Python as an exception instead of aborting the interpreter.

mod env;
mod gym;

/// mimalloc for every Rust allocation in the extension (Python's own allocations are
/// unaffected): it keeps the vector env's worker threads from contending on the system heap.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use jackdaw_core::area::AreaId;
use jackdaw_env::mask::ActionMask;
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
    /// The action mask's buffers, reused by every `action_mask()` call.
    mask: ActionMask,
}

impl Game {
    fn wrap(inner: jackdaw_core::game::Game) -> Self {
        Game {
            inner,
            encoder: Encoder::new(),
            mask: ActionMask::new(),
        }
    }
}

fn value_error(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// A card area by its snapshot name (`"consumables"`, `"pack_cards"`, `"shop_jokers"`, ...).
fn area_id(name: &str) -> PyResult<AreaId> {
    AreaId::ALL
        .into_iter()
        .find(|a| a.name() == name)
        .ok_or_else(|| value_error(format!("unknown area {name:?}")))
}

/// A snapshot (JSON text) imported into a resting game.
fn game_from_snapshot(text: &str) -> PyResult<jackdaw_core::game::Game> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| value_error(format!("snapshot: bad JSON: {e}")))?;
    jackdaw_core::game::Game::from_snapshot(&value).map_err(value_error)
}

/// A refused step as the Python exception for it.
fn step_error(e: jackdaw_core::game::StepError) -> PyErr {
    use jackdaw_core::game::StepError;
    match e {
        StepError::Illegal(m) => IllegalActionError::new_err(m),
        StepError::Unported(m) => NotPortedError::new_err(m),
    }
}

/// Row counts of the five entity arrays (hand, jokers, consumables, shop, pack).
type ObservationRows = (usize, usize, usize, usize, usize);

/// Everything `BalatroEnvironment` reads from a run: the observation (as `observation()`), the
/// action mask (as `action_mask()`), the state summary and the shop's (cards, vouchers,
/// boosters) counts.
type EnvView<'py> = (
    Bound<'py, PyByteArray>,
    ObservationRows,
    env::MaskParts<'py>,
    gym::StateParts,
    (usize, usize, usize),
);

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
        let spec = gym::run_spec(back, stake, seed, challenge)?;
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

    /// Replace the run's state with a canonical snapshot (JSON text) at the same decision point,
    /// keeping the runtime a snapshot doesn't carry: the card-area memo BalatroBot's view reads
    /// (the deck's limit, the shop's voucher slots, a removed shop's areas) and the hand's
    /// highlights, except that Cerulean Bell's forced cards follow the snapshot. The live
    /// lockstep (`jackdaw.bridge.live`) copies values the live game reports into its shadow run
    /// this way; `from_snapshot` would restart that memo (as `load` does).
    fn replace_state(&mut self, text: &str) -> PyResult<()> {
        let fresh = game_from_snapshot(text)?;
        let old = &self.inner;
        if fresh.gs.phase != old.gs.phase {
            return Err(value_error(format!(
                "replace_state: the snapshot is at {}, the run at {}",
                fresh.gs.phase.name(),
                old.gs.phase.name()
            )));
        }
        let was_forced = |id: u32| {
            old.gs
                .areas
                .hand
                .iter()
                .any(|c| c.sort_id == id && c.ability.forced_selection)
        };
        let hand = &fresh.gs.areas.hand;
        let mut highlighted: Vec<u32> = old
            .rt
            .highlighted
            .iter()
            .copied()
            .filter(|&id| {
                hand.iter()
                    .find(|c| c.sort_id == id)
                    .is_some_and(|c| c.ability.forced_selection || !was_forced(id))
            })
            .collect();
        for c in hand {
            if c.ability.forced_selection && !highlighted.contains(&c.sort_id) {
                highlighted.push(c.sort_id);
            }
        }
        let mut rt = old.rt.clone();
        rt.highlighted = highlighted;
        // `from_state` already ran a frame of `Card:update` on the new cards.
        self.inner = jackdaw_core::game::Game { gs: fresh.gs, rt };
        Ok(())
    }

    /// The poker hand the hand cards at ``positions`` make and the hand positions of the cards
    /// that would score, as ``G.FUNCS.evaluate_play`` builds its scoring hand
    /// (state_events.lua:571-600): the best hand's cards plus every Stone Card, or every card
    /// with Splash. ``None`` for no cards, too many, or a position past the hand.
    fn scoring_cards(&self, positions: Vec<usize>) -> Option<(String, Vec<usize>)> {
        use jackdaw_core::hand_eval::{MAX_CARDS, get_poker_hand_info};
        let hand = &self.inner.gs.areas.hand;
        let mut pos = positions;
        pos.sort_unstable();
        pos.dedup();
        if pos.is_empty() || pos.len() > MAX_CARDS || pos.iter().any(|&p| p >= hand.len()) {
            return None;
        }
        let cards: Vec<jackdaw_core::card::Card> = pos.iter().map(|&p| hand[p].clone()).collect();
        let info = get_poker_hand_info(&self.inner.eval_cards(&cards), self.inner.eval_flags());
        let text = info.hand?;
        let mut scoring: Vec<usize> = info
            .scoring
            .as_slice()
            .iter()
            .map(|&i| pos[i as usize])
            .collect();
        if self.inner.has_joker("Splash") {
            scoring.clone_from(&pos);
        } else {
            for (k, c) in cards.iter().enumerate() {
                if c.is_stone() && !scoring.contains(&pos[k]) {
                    scoring.push(pos[k]);
                }
            }
        }
        scoring.sort_unstable();
        Some((text.name().to_owned(), scoring))
    }

    /// One section of the snapshot (JSON text) without exporting the rest: an area (`"hand"`,
    /// `"jokers"`, `"consumables"`, `"deck"`, `"discard"`, `"play"`, `"shop_jokers"`,
    /// `"shop_vouchers"`, `"shop_booster"`, `"pack_cards"`), `"blind"`, `"hands"`, `"pack"`,
    /// `"tags"` or `"last_score"`.
    fn view(&self, name: &str) -> PyResult<String> {
        jackdaw_core::snapshot::view(&self.inner.gs, name)
            .map(|v| v.to_string())
            .ok_or_else(|| value_error(format!("unknown view {name:?}")))
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

    /// The action mask (`jackdaw.env.action_space.get_action_mask`) of the current state.
    ///
    /// Returns ``(data, n_hand, entities, min_card_select, max_card_select, forced_card)``:
    /// ``data`` holds booleans as bytes (the 23 type flags, the card mask, then each entity
    /// mask) and ``entities`` lists ``(action_type, offset, length)`` in key order.
    /// ``jackdaw.env.rust_env.action_mask`` wraps it as an ``ActionMask``.
    #[pyo3(signature = (legality="vanilla"))]
    fn action_mask<'py>(
        &mut self,
        py: Python<'py>,
        legality: &str,
    ) -> PyResult<env::MaskParts<'py>> {
        self.mask.compute(&self.inner, env::legality(legality)?);
        env::mask_to_py(py, &self.mask)
    }

    /// ``factored_to_engine_action``: the engine action (tape form, JSON text) a factored
    /// action stands for here. Raises ``ValueError`` when it has none.
    #[pyo3(signature = (action_type, entity_target=None, card_target=None))]
    fn factored_to_engine(
        &self,
        action_type: i64,
        entity_target: Option<i64>,
        card_target: Option<Vec<i64>>,
    ) -> PyResult<String> {
        let fa = env::factored_from_py(action_type, entity_target, card_target)?;
        let action = jackdaw_env::action::to_engine(&fa, &self.inner).map_err(value_error)?;
        Ok(action.to_json().to_string())
    }

    /// The Gym wrapper's action table (``BalatroGymnasiumEnv._enumerate_actions``) for the
    /// current state's vanilla mask, subsampling with the xoshiro256** generator in
    /// ``rng_state``. Returns ``(actions, rng_state)``: each action as
    /// ``(action_type, entity_target, card_target)``, and the generator's state after.
    fn action_table(&mut self, rng_state: [u64; 4]) -> (Vec<env::FactoredParts>, [u64; 4]) {
        use jackdaw_env::gym::{ActionTable, Xoshiro256};
        use jackdaw_env::mask::Legality;
        self.mask.compute(&self.inner, Legality::Vanilla);
        let mut rng = Xoshiro256::from_state(rng_state);
        let mut table = ActionTable::new();
        table.enumerate(&self.mask, &self.inner, &mut rng);
        let actions = table.actions().iter().map(env::factored_to_py).collect();
        (actions, rng.state())
    }

    /// The default hand selection for using the card at ``index`` of ``area``
    /// (``"consumables"``, ``"pack_cards"`` or ``"shop_jokers"``), or ``None``.
    fn default_targets(&self, area: &str, index: usize) -> PyResult<Option<Vec<usize>>> {
        Ok(jackdaw_env::action::default_targets(
            &self.inner,
            area_id(area)?,
            index,
        ))
    }

    /// ``get_consumable_target_info`` for the card at ``index`` of ``area``:
    /// ``(min_cards, max_cards, needs_targets)``.
    fn consumable_target_info(&self, area: &str, index: usize) -> PyResult<(usize, usize, bool)> {
        let card = self
            .inner
            .gs
            .areas
            .get(area_id(area)?)
            .get(index)
            .ok_or_else(|| value_error(format!("no card {index} in {area}")))?;
        Ok(jackdaw_env::action::consumable_target_info(card))
    }

    /// Take one action (its tape form, JSON text) and play until the game waits again.
    /// On an error the game is unchanged.
    fn step(&mut self, action: &str) -> PyResult<()> {
        use jackdaw_core::action::Action;
        let value: serde_json::Value = serde_json::from_str(action).map_err(value_error)?;
        let action = Action::from_json(&value).map_err(value_error)?;
        self.inner.step(&action).map_err(step_error)
    }

    /// What ``BalatroEnvironment`` reads from the run: ``(obs, rows, mask, state, shop_splits)``,
    /// the observation and mask as ``observation()`` and ``action_mask()`` give them, the
    /// ``GameState`` summary, and the shop's card, voucher and booster counts.
    #[pyo3(signature = (legality="vanilla"))]
    fn env_view<'py>(&mut self, py: Python<'py>, legality: &str) -> PyResult<EnvView<'py>> {
        let legality = env::legality(legality)?;
        let obs = observation_to_py(py, self.encoder.encode(&self.inner.gs))?;
        self.mask.compute(&self.inner, legality);
        let mask = env::mask_to_py(py, &self.mask)?;
        let state = gym::state_to_py(&jackdaw_env::reward::StateSummary::of(&self.inner));
        let areas = &self.inner.gs.areas;
        let splits = (
            areas.shop_jokers.len(),
            areas.shop_vouchers.len(),
            areas.shop_booster.len(),
        );
        Ok((obs.0, obs.1, mask, state, splits))
    }

    /// One ``BalatroEnvironment`` step: the factored action's engine form taken, then
    /// ``env_view``. Raises ``ValueError`` for a factored action with no engine form,
    /// ``IllegalActionError`` or ``NotPortedError`` when the step is refused (the run is
    /// unchanged).
    #[pyo3(signature = (action_type, entity_target=None, card_target=None, legality="vanilla"))]
    fn env_step<'py>(
        &mut self,
        py: Python<'py>,
        action_type: i64,
        entity_target: Option<i64>,
        card_target: Option<Vec<i64>>,
        legality: &str,
    ) -> PyResult<EnvView<'py>> {
        let fa = env::factored_from_py(action_type, entity_target, card_target)?;
        let action = jackdaw_env::action::to_engine(&fa, &self.inner).map_err(value_error)?;
        // No rollback copy: an illegal action changes nothing, and BalatroEnvironment ends the
        // episode on a NotPorted refusal.
        self.inner.step_in_place(&action).map_err(step_error)?;
        self.env_view(py, legality)
    }

    /// The ``GameState`` summary of the run (``jackdaw.env.game_interface.GameState``).
    fn env_state(&self) -> gym::StateParts {
        gym::state_to_py(&jackdaw_env::reward::StateSummary::of(&self.inner))
    }

    /// BalatroBot's ``add`` (``jackdaw/engine/debug.py`` names): ``key`` is ``j_*``, ``c_*``,
    /// ``v_*``, ``p_*`` or a playing card (``H_A``); ``seal`` ``"Red"``, ``edition``
    /// ``"foil"``, ``enhancement`` ``"m_steel"``, ``perishable`` a round count. Returns the new
    /// card's sort id. Debug only: no player action does this. On an error the game is
    /// unchanged.
    #[pyo3(signature = (
        key, seal=None, edition=None, enhancement=None, eternal=false, perishable=None,
        rental=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn debug_add(
        &mut self,
        key: &str,
        seal: Option<&str>,
        edition: Option<&str>,
        enhancement: Option<&str>,
        eternal: bool,
        perishable: Option<f64>,
        rental: bool,
    ) -> PyResult<u32> {
        use jackdaw_core::card::{EditionKind, Seal};
        use jackdaw_core::debug::{AddKey, AddSpec};
        let parsed =
            AddKey::parse(key).ok_or_else(|| value_error(format!("unknown card key {key:?}")))?;
        let mut spec = AddSpec::new(parsed);
        if let Some(s) = seal {
            spec.seal =
                Some(Seal::from_name(s).ok_or_else(|| value_error(format!("unknown seal {s:?}")))?);
        }
        if let Some(e) = edition {
            spec.edition = Some(
                EditionKind::from_name(e)
                    .ok_or_else(|| value_error(format!("unknown edition {e:?}")))?,
            );
        }
        if let Some(m) = enhancement.filter(|m| *m != "c_base") {
            spec.enhancement = Some(
                jackdaw_core::card::center_by_key(m)
                    .ok_or_else(|| value_error(format!("unknown enhancement {m:?}")))?,
            );
        }
        spec.eternal = eternal;
        spec.perishable = perishable;
        spec.rental = rental;
        self.inner.debug_add(&spec).map_err(step_error)
    }

    /// One field of BalatroBot's ``set``: ``money``, ``chips``, ``ante``, ``round``, ``hands``
    /// or ``discards``, written directly.
    fn debug_set(&mut self, field: &str, value: f64) -> PyResult<()> {
        use jackdaw_core::debug::SetField;
        let field = match field {
            "money" => SetField::Money(value),
            "chips" => SetField::Chips(value),
            "ante" => SetField::Ante(value),
            "round" => SetField::Round(value),
            "hands" => SetField::Hands(value),
            "discards" => SetField::Discards(value),
            other => return Err(value_error(format!("unknown set field {other:?}"))),
        };
        self.inner.debug_set(field);
        Ok(())
    }

    /// ``set {shop: true}``: a new shop, as on arrival (SHOP only).
    fn debug_restock_shop(&mut self) -> PyResult<()> {
        self.inner.debug_restock_shop().map_err(step_error)
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
    use super::gym::{GymEnv, VectorEnv};

    /// The Gym observation's flattened length (global context, padded entity arrays, counts).
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const GYM_OBS_LEN: usize = jackdaw_env::env::GYM_OBS_LEN;

    /// The Gym action space's size.
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const MAX_ACTIONS: usize = jackdaw_env::gym::MAX_ACTIONS;

    /// The Gym observation's rows per entity array (hand cards, jokers, consumables, shop
    /// items, pack cards).
    #[pyfunction]
    fn _gym_max_counts() -> [usize; 5] {
        jackdaw_env::env::MAX_COUNTS
    }

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

    /// The action mask of a canonical snapshot (JSON text), as ``Game.action_mask()`` returns
    /// it. Raises ``ValueError`` when the Rust state can't hold the snapshot.
    #[pyfunction]
    #[pyo3(signature = (text, legality="vanilla"))]
    fn _action_mask<'py>(
        py: Python<'py>,
        text: &str,
        legality: &str,
    ) -> PyResult<super::env::MaskParts<'py>> {
        let game = super::game_from_snapshot(text)?;
        let mask = jackdaw_env::mask::ActionMask::of(&game, super::env::legality(legality)?);
        super::env::mask_to_py(py, &mask)
    }

    /// ``engine_action_to_factored``: an engine action (tape form, JSON text) as
    /// ``(action_type, entity_target, card_target)``.
    #[pyfunction]
    fn engine_to_factored(action: &str) -> PyResult<super::env::FactoredParts> {
        use jackdaw_core::action::Action;
        let value: serde_json::Value = serde_json::from_str(action).map_err(super::value_error)?;
        let action = Action::from_json(&value).map_err(super::value_error)?;
        let fa = jackdaw_env::action::from_engine(&action).map_err(super::value_error)?;
        Ok(super::env::factored_to_py(&fa))
    }

    /// The number of factored action types.
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const NUM_ACTION_TYPES: usize = jackdaw_env::action::NUM_ACTION_TYPES;

    /// The Gym action table for any mask (tests): ``entity_masks`` maps an action type to its
    /// entity mask, ``target_info`` maps ``(action_type, entity)`` of the consumable and pack
    /// types to ``(min_cards, max_cards, needs_targets)`` (absent past the end of the source).
    /// Returns ``(actions, rng_state)`` as ``Game.action_table`` does.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn _enumerate_actions(
        type_mask: Vec<bool>,
        card_mask: Vec<bool>,
        entity_masks: std::collections::HashMap<usize, Vec<bool>>,
        min_card_select: usize,
        max_card_select: usize,
        forced_card: Option<usize>,
        target_info: std::collections::HashMap<(usize, usize), (usize, usize, bool)>,
        rng_state: [u64; 4],
    ) -> (Vec<super::env::FactoredParts>, [u64; 4]) {
        use jackdaw_env::gym::{ActionTable, Xoshiro256};
        let mut rng = Xoshiro256::from_state(rng_state);
        let mut table = ActionTable::new();
        table.enumerate_with(
            &type_mask,
            &card_mask,
            |t| entity_masks.get(&t.index()).map(Vec::as_slice),
            (min_card_select, max_card_select),
            forced_card,
            |t, e| target_info.get(&(t.index(), e)).copied(),
            &mut rng,
        );
        let actions = table
            .actions()
            .iter()
            .map(super::env::factored_to_py)
            .collect();
        (actions, rng.state())
    }

    /// ``k`` of ``range(n)`` subsampled with the xoshiro256** generator in ``rng_state``
    /// (tests): ``(indices, rng_state)``.
    #[pyfunction]
    fn _subsample_indices(n: usize, k: usize, rng_state: [u64; 4]) -> (Vec<usize>, [u64; 4]) {
        use jackdaw_env::gym::{Xoshiro256, subsample_indices};
        let mut rng = Xoshiro256::from_state(rng_state);
        let (mut picked, mut out) = (Vec::new(), Vec::new());
        subsample_indices(n, k.min(n), &mut rng, &mut picked, &mut out);
        (out, rng.state())
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
