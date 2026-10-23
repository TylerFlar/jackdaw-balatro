//! Conversions between the env layer (`jackdaw_env`) and Python: action masks as one
//! `bytearray` with offsets, factored actions as `(type, entity, cards)` tuples, and legality
//! names.

use jackdaw_env::action::{ActionType, FactoredAction, NUM_ACTION_TYPES, Selection};
use jackdaw_env::mask::{ActionMask, Legality};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

/// BALATROBOT legality's filter: an action the live BalatroBot v1.5.2 also executes (a dry run
/// of its endpoint's checks, `jackdaw_bot::bot_accepts`).
struct BotFilter;

impl jackdaw_env::mask::ActionFilter for BotFilter {
    fn allows(
        &self,
        game: &jackdaw_core::game::Game,
        action: &jackdaw_core::action::Action,
    ) -> bool {
        jackdaw_bot::bot_accepts(game, action)
    }
}

static BOT_FILTER: BotFilter = BotFilter;

/// `Legality` by its Python value.
pub fn legality(name: &str) -> PyResult<Legality<'static>> {
    match name {
        "vanilla" => Ok(Legality::Vanilla),
        "balatrobot" => Ok(Legality::Filtered(&BOT_FILTER)),
        other => Err(PyValueError::new_err(format!("unknown legality {other:?}"))),
    }
}

/// An action mask for Python: `(data, n_hand, entities, min_card_select, max_card_select,
/// forced_card)`. `data` holds booleans as bytes: the type mask, the card mask, then each entity
/// mask; `entities` lists `(action_type, offset, length)` in the Python mask's key order.
pub type MaskParts<'py> = (
    Bound<'py, PyByteArray>,
    usize,
    Vec<(u8, usize, usize)>,
    usize,
    usize,
    Option<usize>,
);

pub fn mask_to_py<'py>(py: Python<'py>, mask: &ActionMask) -> PyResult<MaskParts<'py>> {
    let n_hand = mask.card_mask.len();
    let types = mask.entity_types();
    let total = NUM_ACTION_TYPES
        + n_hand
        + types
            .iter()
            .map(|&t| mask.entity_mask(t).map_or(0, <[bool]>::len))
            .sum::<usize>();
    let mut entities = Vec::with_capacity(types.len());
    let data = PyByteArray::new_with(py, total, |buf| {
        let (head, rest) = buf.split_at_mut(NUM_ACTION_TYPES);
        for (dst, &b) in head.iter_mut().zip(&mask.type_mask) {
            *dst = u8::from(b);
        }
        let (cards, mut rest) = rest.split_at_mut(n_hand);
        for (dst, &b) in cards.iter_mut().zip(&mask.card_mask) {
            *dst = u8::from(b);
        }
        let mut offset = NUM_ACTION_TYPES + n_hand;
        for &t in types {
            let e = mask.entity_mask(t).unwrap_or(&[]);
            let (dst, tail) = rest.split_at_mut(e.len());
            for (d, &b) in dst.iter_mut().zip(e) {
                *d = u8::from(b);
            }
            entities.push((t as u8, offset, e.len()));
            offset += e.len();
            rest = tail;
        }
        Ok(())
    })?;
    Ok((
        data,
        n_hand,
        entities,
        mask.min_card_select,
        mask.max_card_select,
        mask.forced_card,
    ))
}

/// A factored action from its Python fields.
pub fn factored_from_py(
    action_type: i64,
    entity_target: Option<i64>,
    card_target: Option<Vec<i64>>,
) -> PyResult<FactoredAction> {
    let t = usize::try_from(action_type)
        .ok()
        .and_then(ActionType::from_index)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown action type: {action_type}")))?;
    let entity_target = entity_target
        .map(|e| u32::try_from(e).map_err(|_| PyValueError::new_err(format!("entity_target {e}"))))
        .transpose()?;
    let card_target = card_target
        .map(|cards| {
            let idx: Option<Vec<usize>> = cards.iter().map(|&c| usize::try_from(c).ok()).collect();
            idx.as_deref()
                .and_then(Selection::new)
                .ok_or_else(|| PyValueError::new_err(format!("card_target {cards:?}")))
        })
        .transpose()?;
    Ok(FactoredAction {
        action_type: t,
        entity_target,
        card_target,
    })
}

/// A factored action as `(action_type, entity_target, card_target)` (`Vec<u8>` would cross as
/// `bytes`, so the cards are `usize`).
pub type FactoredParts = (u8, Option<u32>, Option<Vec<usize>>);

pub fn factored_to_py(fa: &FactoredAction) -> FactoredParts {
    (
        fa.action_type as u8,
        fa.entity_target,
        fa.card_target.map(|s| s.to_vec()),
    )
}
