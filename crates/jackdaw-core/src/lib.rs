//! The jackdaw game engine: Balatro 1.0.1o, ported from the game's Lua source.
//!
//! Covers Lua numerics, the RNG, game state, rules, scoring, actions, snapshots, debug operations
//! and score preview. The port follows the Lua structure: one Rust function per Lua function,
//! called in Lua order, with RNG draws in Lua order; queued `G.E_MANAGER` events become typed
//! deferred events drained at documented sync points.
//!
//! Crate rules: no PyO3 dependency, no global mutable state, and every public state type is
//! `Send + Sync + Clone`.
//!
//! Not ported yet: this is the Phase 0 stub.
#![forbid(unsafe_code)]

pub mod hand_eval;
pub mod lua;
pub mod rng;

/// The generated data tables the engine runs on.
pub use jackdaw_data as data;

#[cfg(test)]
mod tests {
    #[test]
    fn reexports_the_data_tables() {
        assert_eq!(crate::data::BALATRO_VERSION, "1.0.1o-FULL");
    }
}
