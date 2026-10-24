//! The request dispatcher: routes a request to its endpoint through BalatroBot's four tiers.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/core/dispatcher.lua` (`BB_DISPATCHER.dispatch`):
//!
//! 1. **Protocol**: `method` must be a string; `rpc.discover` returns the vendored OpenRPC
//!    document; an unregistered method is refused.
//! 2. **Schema**: the endpoint's arguments go through [`crate::validator`].
//! 3. **Game state**: `G.STATE` must be one of the endpoint's `requires_state`.
//! 4. **Execution**: the endpoint runs; a runtime error becomes `INTERNAL_ERROR`.
//!
//! A dry run ([`dispatch`] with `dry_run`) stops each endpoint before it changes anything: it is
//! the "validate step" BALATROBOT legality runs ([`crate::legality`]).

use std::sync::OnceLock;

use crate::endpoints::{self, Ctx};
use crate::errors::BotError;
use crate::json::{self, Json};
use crate::lua::{Table, Value};
use crate::session::Session;

/// The vendored `src/lua/utils/openrpc.json`, which `rpc.discover` serves.
pub const OPENRPC: &str =
    include_str!("../../../third_party/balatrobot/v1.5.2/src/lua/utils/openrpc.json");

/// `rpc.discover`'s result: the spec as `json.decode` reads it (so `json.encode` writes `{}` as
/// `[]`, drops `null`s and reformats numbers).
pub fn openrpc_spec() -> Result<Json, BotError> {
    static SPEC: OnceLock<Result<Json, String>> = OnceLock::new();
    SPEC.get_or_init(|| {
        let v = json::decode(OPENRPC.as_bytes())
            .map_err(|e| e.0)?
            .ok_or_else(|| "null spec".to_owned())?;
        json::from_value(&v).map_err(|e| e.0)
    })
    .clone()
    .map_err(|_| BotError::internal("Failed to parse OpenRPC spec"))
}

/// `BB_DISPATCHER.dispatch(request)` for a request whose `method` and `params` were decoded.
/// `method` is `None` when the request had no string `method`.
pub fn dispatch(
    session: &mut Session,
    method: Option<&str>,
    params: Option<&Value>,
    dry_run: bool,
) -> Result<Json, BotError> {
    // TIER 1: protocol validation.
    let Some(method) = method else {
        return Err(BotError::bad_request("Request missing 'method' field"));
    };
    if method == "rpc.discover" {
        return openrpc_spec();
    }
    // `request.params or {}`: a missing (or `false`) params is an empty table.
    let empty = Table::new();
    let args: Option<&Table> = match params {
        None | Some(Value::Bool(false)) => Some(&empty),
        Some(Value::Table(t)) => Some(t),
        Some(_) => None,
    };
    let Some(endpoint) = endpoints::find(method, session.config.debug_endpoints) else {
        return Err(BotError::bad_request(format!("Unknown method: {method}")));
    };
    // TIER 2: schema validation.
    crate::validator::validate(args, endpoint.schema)?;
    let args = args.expect("validated as a table");
    // TIER 3: game state validation.
    if let Some(required) = endpoint.requires_state {
        let current = session.state();
        if !required.iter().any(|s| Some(*s) == current) {
            let names: Vec<&str> = required.iter().map(|s| s.name()).collect();
            return Err(BotError::invalid_state(format!(
                "Method '{method}' requires one of these states: {}",
                names.join(", ")
            )));
        }
    }
    // TIER 4: execute.
    let mut ctx = Ctx {
        session,
        args,
        dry_run,
    };
    (endpoint.run)(&mut ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionConfig;

    fn call(s: &mut Session, method: &str, params: &str) -> Result<Json, BotError> {
        let v = json::decode(params.as_bytes()).unwrap();
        dispatch(s, Some(method), v.as_ref(), false)
    }

    #[test]
    fn four_tiers() {
        let mut s = Session::new(SessionConfig::default());
        assert_eq!(
            dispatch(&mut s, None, None, false).unwrap_err().message,
            "Request missing 'method' field"
        );
        assert_eq!(
            call(&mut s, "nope", "{}").unwrap_err().message,
            "Unknown method: nope"
        );
        assert_eq!(
            call(&mut s, "health", "\"x\"").unwrap_err().message,
            "Arguments must be a table"
        );
        assert_eq!(
            call(&mut s, "health", "false").unwrap(),
            Json::obj([("status", Json::str("ok"))])
        );
        let e = call(&mut s, "buy", r#"{"card": 0}"#).unwrap_err();
        assert_eq!(e.message, "Method 'buy' requires one of these states: SHOP");
        assert_eq!(e.name.code(), -32002);
        let e = call(&mut s, "rearrange", "{}").unwrap_err();
        assert_eq!(
            e.message,
            "Method 'rearrange' requires one of these states: SELECTING_HAND, SHOP, SMODS_BOOSTER_OPENED"
        );
        let spec = call(&mut s, "rpc.discover", "{}").unwrap();
        assert_eq!(spec.get("openrpc"), Some(&Json::str("1.3.2")));
    }

    #[test]
    fn test_endpoints_need_debug_mode() {
        let mut s = Session::new(SessionConfig::default());
        assert!(call(&mut s, "test_validation", r#"{"required_field": "x"}"#).is_err());
        let mut d = Session::new(SessionConfig {
            debug_endpoints: true,
            ..SessionConfig::default()
        });
        let ok = call(&mut d, "test_validation", r#"{"required_field": "x"}"#).unwrap();
        assert_eq!(ok.get("success"), Some(&Json::Bool(true)));
        let err = call(
            &mut d,
            "test_error_endpoint",
            r#"{"error_type": "throw_error"}"#,
        )
        .unwrap_err();
        assert_eq!(err.name.name(), "INTERNAL_ERROR");
        assert!(
            err.message
                .ends_with("Intentional test error from endpoint execution")
        );
    }
}
