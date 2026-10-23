//! Reinforcement-learning hot paths on top of the engine.
//!
//! Holds the observation encoder (writing into caller-owned buffers), the action masks for both
//! legality modes, the Gym action enumeration, and the batched vector environment that steps many
//! games in parallel.
//!
//! Not ported yet: this is the Phase 0 stub.
#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    #[test]
    fn builds_on_the_engine() {
        assert!(!jackdaw_core::data::BALATRO_VERSION.is_empty());
    }
}
