//! Reinforcement-learning hot paths on top of the engine.
//!
//! Holds the observation encoder (writing into caller-owned buffers). The action masks for both
//! legality modes, the Gym action enumeration and the batched vector environment come next.
#![forbid(unsafe_code)]

pub mod observation;

#[cfg(test)]
mod tests {
    #[test]
    fn builds_on_the_engine() {
        assert!(!jackdaw_core::data::BALATRO_VERSION.is_empty());
    }
}
