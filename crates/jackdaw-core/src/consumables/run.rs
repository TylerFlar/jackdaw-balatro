//! The `Card:use_consumeable` blocks that make cards, move money or change the jokers: The
//! Fool (card.lua:1373-1383), The Hermit (1385-1391), Temperance (1393-1399), The Emperor and
//! The High Priestess (1401-1413), Judgement and The Soul (1415-1424), Ankh (1426-1452), Wraith
//! (1454-1465) and the edition block of The Wheel of Fortune, Ectoplasm and Hex (1467-1519).

use crate::area::AreaId;
use crate::card::EditionKind;
use crate::event::Event;
use crate::game::Game;
use crate::pools::{CardSpec, PoolType};

impl Game {
    /// The Emperor and The High Priestess (card.lua:1401-1413): one event per card that fits
    /// now, each making its card if there is still room.
    pub(super) fn use_emperor_priestess(&mut self, id: u32) {
        let center = self.card(id).center();
        let n = center
            .config_num("tarots")
            .or_else(|| center.config_num("planets"))
            .expect("the number of cards to make");
        let room = self.gs.consumable_slots - self.gs.areas.consumables.len() as f64;
        let mut i = 1.0;
        while i <= n.min(room) {
            self.queue(Event::Call(emperor_priestess, id, 0.0));
            i += 1.0;
        }
    }

    /// Ankh (card.lua:1426-1452): a random joker is chosen now; one event dissolves every other
    /// non-eternal joker, the next copies the chosen one.
    ///
    /// The first event blocks for 0.75s, longer than a dissolve takes (0.735s), so the jokers
    /// are gone before the copy is made. The port removes them in the event itself rather than
    /// in [`Event::RemoveCard`]s queued after the copy: the order shows when both the copy and a
    /// removed joker change the discards (`ease_discard` stops at 0) or the hand size.
    pub(super) fn use_ankh(&mut self) {
        let deletable: Vec<u32> = self
            .gs
            .areas
            .jokers
            .iter()
            .filter(|j| !j.ability.eternal)
            .map(|j| j.sort_id)
            .collect();
        let mut jokers = self.joker_ids();
        jokers.sort_unstable();
        let seed = self.gs.rng.seed("ankh_choice");
        let chosen = jokers[crate::rng::element_index(jokers.len(), seed)];
        // The dissolving event, one call per joker: they run back to back, as one loop does.
        for v in deletable {
            if v != chosen {
                self.queue(Event::Call(ankh_remove, v, 0.0));
            }
        }
        self.queue(Event::Call(ankh_copy, chosen, 0.0));
    }

    /// The Wheel of Fortune, Ectoplasm and Hex (card.lua:1467-1519). The Wheel rolls its chance
    /// now; on a miss its event only shows "Nope!".
    pub(super) fn use_edition_joker(&mut self, id: u32) {
        let wheel = self.card(id).name() == "The Wheel of Fortune";
        if !wheel || {
            let odds = self.extra(id);
            self.roll("wheel_of_fortune", odds)
        } {
            self.queue(Event::Call(edition_joker, id, 0.0));
        }
    }
}

/// The Fool: a copy of the last Tarot or Planet used, if there is room (card.lua:1374-1382).
/// `G.GAME.last_tarot_planet` still holds the card before The Fool: The Fool's own entry is set
/// two events later.
pub(super) fn fool(g: &mut Game, _: u32, _: f64) {
    if g.gs.consumable_slots > g.gs.areas.consumables.len() as f64 {
        let forced = g.gs.last_tarot_planet;
        let spec = CardSpec::new(PoolType::TarotPlanet, AreaId::Consumables, "fool").forced(forced);
        g.create_and_emplace(spec, AreaId::Consumables);
    }
}

/// The Hermit: money doubles, up to `extra` (card.lua:1386-1390); `ease_dollars(..., true)`.
pub(super) fn hermit(g: &mut Game, id: u32, _: f64) {
    let gain = g.gs.dollars.min(g.extra(id)).max(0.0);
    g.gs.dollars += gain;
}

/// Temperance: the jokers' sell value, capped, as `Card:update` keeps it in `ability.money`
/// (card.lua:1394-1398, 4167-4174); `ease_dollars(..., true)`.
pub(super) fn temperance(g: &mut Game, id: u32, _: f64) {
    let money = g.card(id).ability.money.unwrap_or(0.0);
    g.gs.dollars += money;
}

/// One card of The Emperor ('emp' Tarot) or The High Priestess ('pri' Planet), if there is
/// still room (card.lua:1403-1411).
fn emperor_priestess(g: &mut Game, id: u32, _: f64) {
    if g.gs.consumable_slots > g.gs.areas.consumables.len() as f64 {
        let spec = if g.card(id).name() == "The Emperor" {
            CardSpec::new(PoolType::Tarot, AreaId::Consumables, "emp")
        } else {
            CardSpec::new(PoolType::Planet, AreaId::Consumables, "pri")
        };
        g.create_and_emplace(spec, AreaId::Consumables);
    }
}

/// Judgement ('jud') or The Soul (a Legendary, 'sou'): a joker, with no room check
/// (card.lua:1416-1423).
pub(super) fn judgement_soul(g: &mut Game, id: u32, _: f64) {
    let spec = if g.card(id).name() == "The Soul" {
        CardSpec::new(PoolType::Joker, AreaId::Jokers, "sou").legendary()
    } else {
        CardSpec::new(PoolType::Joker, AreaId::Jokers, "jud")
    };
    g.create_and_emplace(spec, AreaId::Jokers);
}

/// Ankh's first event: a joker other than the chosen one dissolves (card.lua:1435-1443), and
/// `Card:remove` runs before the next event (see [`Game::use_ankh`]).
fn ankh_remove(g: &mut Game, joker: u32, _: f64) {
    g.remove_card(joker);
}

/// Ankh's second event (card.lua:1444-1452): `copy_card` of the chosen joker, without its
/// edition when that is Negative, added to the deck and emplaced. (Lua then drops a Negative
/// edition from the copy, which never has one.)
fn ankh_copy(g: &mut Game, chosen: u32, _: f64) {
    let strip = g
        .card(chosen)
        .edition
        .is_some_and(|e| e.kind == EditionKind::Negative);
    let card = g.copy_card(chosen, strip, AreaId::Jokers);
    let id = card.sort_id;
    g.gs.areas.emplace(AreaId::Jokers, card, false, false);
    g.add_to_deck(id, false);
}

/// Wraith: a Rare joker ('wra'), then the money goes to 0 (card.lua:1455-1464).
pub(super) fn wraith(g: &mut Game, _: u32, _: f64) {
    let spec = CardSpec::new(PoolType::Joker, AreaId::Jokers, "wra").rarity(0.99);
    g.create_and_emplace(spec, AreaId::Jokers);
    if g.gs.dollars != 0.0 {
        // `ease_dollars(-G.GAME.dollars, true)`.
        g.gs.dollars -= g.gs.dollars;
    }
}

/// The edition event (card.lua:1471-1500): an editionless joker, picked from the list
/// `Card:update` keeps, gets Negative (Ectoplasm), Polychrome (Hex) or a guaranteed non-Negative
/// edition (The Wheel of Fortune). Hex dissolves every other non-eternal joker; Ectoplasm takes
/// `ecto_minus` hand size, which then grows by one.
///
/// Hex's jokers are gone 0.735s later, before `use_card`'s next event (it waits 0.6s + 0.2s), so
/// what their removal queues (a hand size given back, ...) runs before the state comes back, in
/// `PLAY_TAROT`, where a bigger hand draws nothing. The port removes them in this event rather
/// than in [`Event::RemoveCard`]s queued after `use_card`'s.
fn edition_joker(g: &mut Game, id: u32, _: f64) {
    let name = g.card(id).name();
    let pool = g.editionless_jokers();
    let key = match name {
        "The Wheel of Fortune" => "wheel_of_fortune",
        "Ectoplasm" => "ectoplasm",
        _ => "hex",
    };
    let seed = g.gs.rng.seed(key);
    let eligible = pool[crate::rng::element_index(pool.len(), seed)];
    let edition = match name {
        "Ectoplasm" => Some(EditionKind::Negative),
        "Hex" => Some(EditionKind::Polychrome),
        _ => g.poll_edition("wheel_of_fortune", 1.0, true, true),
    };
    g.set_card_edition(eligible, edition);
    if name == "Hex" {
        for v in g.joker_ids() {
            if v != eligible && !g.card(v).ability.eternal {
                // `v:start_dissolve()`, then `Card:remove` when the dissolve ends.
                g.remove_card(v);
            }
        }
    }
    if name == "Ectoplasm" {
        let minus = g.gs.ecto_minus;
        g.change_size(AreaId::Hand, -minus);
        g.gs.ecto_minus = minus + 1.0;
    }
}
