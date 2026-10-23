//! Reinforcement-learning hot paths on top of the engine: the observation encoder, the action
//! masks and the factored action space, the Gym action enumeration, and the batched vector
//! environment.
#![forbid(unsafe_code)]

pub mod action;
pub mod gym;
pub mod mask;
pub mod observation;

#[cfg(test)]
mod tests {
    #[test]
    fn builds_on_the_engine() {
        assert!(!jackdaw_core::data::BALATRO_VERSION.is_empty());
    }
}
