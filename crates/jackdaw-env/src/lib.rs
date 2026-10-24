//! Reinforcement-learning hot paths on top of the engine: the observation encoder, the action
//! masks and the factored action space, the Gym action enumeration, and the batched vector
//! environment.
#![forbid(unsafe_code)]

/// The version of the action space: what the action types mean, which ones a mask allows where,
/// and how the Gym action table is built. A policy is tied to the masks it was trained with, so
/// this goes up whenever they change (and stays the same when only the engine's internals do).
///
/// - 1: jackdaw 0.1.
/// - 2: jackdaw 0.2. Reordering follows the game's drag and drop (jokers in every decision
///   state, the hand in a dealt pack, G206), and the envs cap consecutive free moves.
pub const ACTION_VERSION: u32 = 2;

pub mod action;
pub mod env;
pub mod gym;
pub mod mask;
pub mod observation;
pub mod reward;
pub mod vector;

#[cfg(test)]
mod tests {
    #[test]
    fn builds_on_the_engine() {
        assert!(!jackdaw_core::data::BALATRO_VERSION.is_empty());
    }
}
