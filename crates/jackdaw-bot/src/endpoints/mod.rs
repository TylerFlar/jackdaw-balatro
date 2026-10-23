//! BalatroBot's endpoints, one module per `src/lua/endpoints/*.lua` file.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`).
//! Each endpoint keeps the Lua's checks in the Lua's order, with its exact error names and
//! messages. Where the Lua then drives the game (`G.FUNCS.*`), the port takes the matching step
//! in the engine, which follows vanilla 1.0.1o: a request BalatroBot would execute but vanilla
//! forbids (selling an eternal joker, a greyed-out pack pick) is refused with `NOT_ALLOWED`
//! and a message that says so (`docs/balatrobot.md`). A step that needs game logic the engine
//! has not ported yet is an `INTERNAL_ERROR` (`not ported yet: ...`).
//!
//! Every endpoint takes a dry-run flag: it runs its checks and returns before changing
//! anything ([`crate::legality`]).

use jackdaw_core::action::Action;
use jackdaw_core::game::{Game, StepError};

use crate::errors::BotError;
use crate::json::Json;
use crate::lua::{Table, Value};
use crate::session::Session;
use crate::states::GState;
use crate::validator::Field;

pub mod add;
pub mod buy;
pub mod cash_out;
pub mod discard;
pub mod gamestate;
pub mod health;
pub mod load;
pub mod menu;
pub mod next_round;
pub mod pack;
pub mod play;
pub mod rearrange;
pub mod reroll;
pub mod save;
pub mod screenshot;
pub mod select;
pub mod sell;
pub mod set;
pub mod skip;
pub mod start;
pub mod test_endpoints;
pub mod use_consumable;

/// What an endpoint runs with.
pub struct Ctx<'a> {
    pub session: &'a mut Session,
    pub args: &'a Table,
    /// Check only: return `Ok(Json::Null)` once every check passed, before acting.
    pub dry_run: bool,
}

impl Ctx<'_> {
    /// `args[name]`.
    pub fn arg(&self, name: &str) -> Option<&Value> {
        self.args.get(name)
    }

    /// `args[name]` as a number (the schema checked its type).
    pub fn num(&self, name: &str) -> Option<f64> {
        self.arg(name).and_then(Value::as_num)
    }

    /// `args[name]` as a string.
    pub fn str(&self, name: &str) -> Option<&str> {
        self.arg(name).and_then(Value::as_str)
    }

    /// `args[name]` as a table.
    pub fn table(&self, name: &str) -> Option<&Table> {
        self.arg(name).and_then(Value::as_table)
    }

    /// Lua's `if args[name] then`: present and not `false`.
    pub fn truthy(&self, name: &str) -> bool {
        self.arg(name).is_some_and(Value::truthy)
    }

    /// The run in progress (the state gate guarantees one).
    pub fn game(&self) -> &Game {
        self.session.game().expect("the state gate requires a run")
    }

    pub fn game_mut(&mut self) -> &mut Game {
        self.session
            .game_mut()
            .expect("the state gate requires a run")
    }

    /// Take a step and answer with the gamestate, or explain the refusal.
    pub fn step(&mut self, action: &Action, refused: &str) -> Result<Json, BotError> {
        self.game_mut()
            .step(action)
            .map_err(|e| step_error(refused, e))?;
        Ok(self.session.view())
    }
}

/// A refused engine step as a response: vanilla's refusal (`NOT_ALLOWED`, prefixed with what
/// was refused) or an unported step (`INTERNAL_ERROR`).
pub fn step_error(refused: &str, e: StepError) -> BotError {
    match e {
        StepError::Illegal(m) => BotError::not_allowed(format!("{refused}: {m}")),
        StepError::Unported(m) => BotError::internal(format!("not ported yet: {m}")),
    }
}

/// An endpoint (`{name, schema, requires_state, execute}`).
pub struct Endpoint {
    pub name: &'static str,
    /// The fields in the order `pairs(schema)` visits them in Balatro's LuaJIT 2.0.5.
    pub schema: &'static [Field],
    pub requires_state: Option<&'static [GState]>,
    pub run: fn(&mut Ctx) -> Result<Json, BotError>,
}

/// `BB_ENDPOINTS` (balatrobot.lua:8-44), in registration order.
pub const ENDPOINTS: &[&Endpoint] = &[
    &health::ENDPOINT,
    &gamestate::ENDPOINT,
    &save::ENDPOINT,
    &load::ENDPOINT,
    &screenshot::ENDPOINT,
    &set::ENDPOINT,
    &add::ENDPOINT,
    &menu::ENDPOINT,
    &start::ENDPOINT,
    &skip::ENDPOINT,
    &select::ENDPOINT,
    &play::ENDPOINT,
    &discard::ENDPOINT,
    &cash_out::ENDPOINT,
    &next_round::ENDPOINT,
    &reroll::ENDPOINT,
    &buy::ENDPOINT,
    &pack::ENDPOINT,
    &rearrange::ENDPOINT,
    &sell::ENDPOINT,
    &use_consumable::ENDPOINT,
];

/// The test endpoints `utils/debugger.lua` registers in debug mode (`echo.lua` registers as
/// `test_endpoint`; `endpoint.lua` is never loaded).
pub const TEST_ENDPOINTS: &[&Endpoint] = &[
    &test_endpoints::ECHO,
    &test_endpoints::STATE,
    &test_endpoints::ERROR,
    &test_endpoints::VALIDATION,
];

/// `BB_DISPATCHER.endpoints[name]`.
pub fn find(name: &str, debug: bool) -> Option<&'static Endpoint> {
    let tests: &[&Endpoint] = if debug { TEST_ENDPOINTS } else { &[] };
    ENDPOINTS
        .iter()
        .chain(tests.iter())
        .find(|e| e.name == name)
        .copied()
}

/// The states every run is in at rest (for the `requires_state` lists).
pub mod states {
    use crate::states::GState;

    pub const SELECTING_HAND: &[GState] = &[GState::SelectingHand];
    pub const SHOP: &[GState] = &[GState::Shop];
    pub const BLIND_SELECT: &[GState] = &[GState::BlindSelect];
    pub const ROUND_EVAL: &[GState] = &[GState::RoundEval];
    pub const MENU: &[GState] = &[GState::Menu];
    pub const BOOSTER: &[GState] = &[GState::SmodsBoosterOpened];
}
