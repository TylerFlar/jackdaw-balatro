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
//! Ported so far: Lua numerics, the RNG, the card model, poker hand evaluation, the run state
//! and its canonical snapshots.
#![forbid(unsafe_code)]

pub mod action;
pub mod area;
mod blind;
pub mod card;
mod cards;
mod consumables;
pub mod event;
mod flow;
pub mod game;
pub mod hand_eval;
pub mod joker;
pub mod lua;
mod packs;
pub mod pools;
pub mod rng;
mod round;
pub mod rules;
pub mod scoring;
mod select;
mod shop;
pub mod snapshot;
pub mod state;
mod tags;
mod vouchers;

/// The generated data tables the engine runs on.
pub use jackdaw_data as data;

#[cfg(test)]
mod tests {
    #[test]
    fn reexports_the_data_tables() {
        assert_eq!(crate::data::BALATRO_VERSION, "1.0.1o-FULL");
    }
}
