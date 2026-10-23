//! The BalatroBot layer: a 1:1 port of BalatroBot's command space and gamestate view.
//!
//! Holds the JSON-RPC envelope and request validator, one module per endpoint (with BalatroBot's
//! exact error codes and messages), the view builder ported from `gamestate.lua`, the session,
//! and the JSON-RPC server behind `jackdaw serve`. Anything trained on the simulator can then
//! drive the live game through BalatroBot unchanged.
//!
//! Not ported yet: this is the Phase 0 stub.
#![forbid(unsafe_code)]

/// The BalatroBot release (coder/balatrobot) whose endpoints and view this crate mirrors.
pub const BALATROBOT_VERSION: &str = "1.5.2";

#[cfg(test)]
mod tests {
    #[test]
    fn pins_balatrobot_and_the_vanilla_data() {
        assert_eq!(super::BALATROBOT_VERSION, "1.5.2");
        assert_eq!(jackdaw_core::data::BALATRO_VERSION, "1.0.1o-FULL");
    }
}
