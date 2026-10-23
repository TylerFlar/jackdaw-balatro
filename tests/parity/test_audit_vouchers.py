"""Parity cases for the vouchers the recorded tapes never redeem (card.lua:1880-1980).

Each case puts the voucher in the next shop with the oracle
(:class:`tests.parity.pyengine.PyEngine`), cashes out and redeems it, checking every
step with :func:`tests.parity.rust_step.check_step`.
"""

from __future__ import annotations

from typing import Any

from tests.parity.pyengine import PyEngine
from tests.parity.test_audit_flow import at_blind_select, check, in_round

_WIN = 10**12


def redeem(seed: str, voucher: str, jokers: tuple[str, ...] = ()) -> tuple[PyEngine, dict]:
    """Beat the first blind with ``voucher`` as the ante's shop voucher, cash out and
    redeem it. Returns the engine and the snapshot before the redeem."""
    eng = in_round(seed, jokers, dollars=100)
    eng.gs["chips"] = _WIN
    check(eng, ["PlayHand", [0]])
    eng.gs["current_round"]["voucher"] = voucher
    before = check(eng, ["CashOut"])
    assert before["areas"]["shop_vouchers"][0]["center"] == voucher
    check(eng, ["RedeemVoucher", 0])
    return eng, before


def game(snap: dict[str, Any]) -> dict[str, Any]:
    return snap["game"]


class TestVouchers:
    def test_liquidation_halves_prices(self):
        eng, _ = redeem("AU_V_LIQ", "v_liquidation")
        snap = eng.snapshot()
        assert game(snap)["discount_percent"] == 50

    def test_glow_up_quadruples_the_edition_rate(self):
        eng, _ = redeem("AU_V_GLOW", "v_glow_up")
        assert game(eng.snapshot())["edition_rate"] == 4

    def test_nacho_tong_adds_a_hand(self):
        eng, before = redeem("AU_V_NACHO", "v_nacho_tong")
        snap = eng.snapshot()
        assert snap["round_resets"]["hands"] == before["round_resets"]["hands"] + 1
        assert snap["current_round"]["hands_left"] == before["current_round"]["hands_left"] + 1

    def test_recyclomancy_adds_a_discard(self):
        eng, before = redeem("AU_V_RECYCLO", "v_recyclomancy")
        snap = eng.snapshot()
        assert snap["round_resets"]["discards"] == before["round_resets"]["discards"] + 1

    def test_antimatter_adds_a_joker_slot(self):
        eng, before = redeem("AU_V_ANTI", "v_antimatter")
        assert game(eng.snapshot())["joker_slots"] == game(before)["joker_slots"] + 1

    def test_petroglyph_turns_the_ante_back(self):
        eng, before = redeem("AU_V_PETRO", "v_petroglyph")
        snap = eng.snapshot()
        assert snap["round_resets"]["ante"] == before["round_resets"]["ante"] - 1
        assert snap["round_resets"]["discards"] == before["round_resets"]["discards"] - 1

    def test_observatory_planets_multiply_their_hand(self):
        """card.lua:2293-2302: a held Planet of the played hand gives X1.5."""
        eng, _ = redeem("AU_V_OBS", "v_observatory", ("c_pluto",))
        check(eng, ["NextRound"], ["SelectBlind"])
        snap = check(eng, ["PlayHand", [0]])
        last = snap["sim"]["last_score"]
        assert last["hand_type"] == "High Card"
        assert last["mult"] == 1.5


class TestFirstHandDrawnComesFirst:
    def test_certificate_card_is_in_the_hand_when_cerulean_bell_picks(self):
        """game.lua:3226-3241 (G14): first_hand_drawn runs before drawn_to_hand."""
        eng = at_blind_select("AU_CERT_BELL", ("j_certificate",), boss="bl_final_bell")
        snap = check(eng, ["SelectBlind"])
        assert len(snap["areas"]["hand"]) == snap["game"]["hand_size"] + 1
        assert sum(1 for c in snap["areas"]["hand"] if c["ability"].get("forced_selection")) == 1
