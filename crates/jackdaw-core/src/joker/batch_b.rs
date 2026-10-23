//! Joker batch B: one function per joker, all of its `calculate_joker` branches.

use super::CalcFn;

/// The function of a ported joker in this batch.
pub(super) fn lookup(key: &str) -> Option<CalcFn> {
    let _ = key;
    None
}
