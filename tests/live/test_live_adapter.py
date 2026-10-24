"""LiveAdapter against stand-ins for the live game (in process, fast).

The "live game" here is the simulator's own BalatroBot session (``SimBackend``,
or ``FakeLive`` with live-style card ids), so with nothing injected the shadow
must never diverge. Injected differences check the divergence, its bundle and
``replay``; the reconciles and vetoes of the deviation catalog are checked on
the deviations they exist for.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from jackdaw.bridge.backend import LiveBackend, RPCError, SimBackend
from jackdaw.bridge.live import DivergenceError, LiveAdapter, VetoedActionError, replay_bundle
from jackdaw.bridge.lockstep import Job, RandomPolicy, SmartPolicy, play_game
from jackdaw.bridge.reconcile import front_base
from jackdaw.core import Game, IllegalActionError
from jackdaw.engine.actions import (
    OpenBooster,
    PlayHand,
    RerollBoss,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    UseConsumable,
)
from jackdaw.env.game_interface import BridgeAdapter
from tests.live.fakes import FakeLive, reshuffle_round, resuit, score_off_by_one


def _play(adapter: LiveAdapter, steps: int, seed: int = 0) -> None:
    policy = SmartPolicy()
    rng = random.Random(seed)
    for _ in range(steps):
        if adapter.done:
            return
        action = policy.choose(adapter.game, adapter.get_legal_actions(), rng, adapter.shadow_view)
        try:
            adapter.step(action)
        except IllegalActionError:
            continue


def _beat_small_blind(adapter: LiveAdapter) -> None:
    adapter.step(SelectBlind())
    adapter.request("set", {"chips": 100000})
    adapter.request("play", {"cards": [0]})
    adapter.request("cash_out")


def _open_pack(adapter: LiveAdapter, key: str) -> None:
    """In the shop: make room for the booster ``key`` (open and skip the first), add
    it and open it."""
    adapter.request("set", {"money": 100})
    if len(adapter.view["packs"]["cards"]) >= adapter.view["packs"]["limit"]:
        adapter.step(OpenBooster(0))
        adapter.step(SkipPack())
    adapter.request("add", {"key": key})
    adapter.step(OpenBooster(len(adapter.view["packs"]["cards"]) - 1))


# ---------------------------------------------------------------------------
# The sim on both sides: no divergence
# ---------------------------------------------------------------------------


def test_sim_against_sim_never_diverges() -> None:
    kinds: Counter[str] = Counter()
    steps = 0
    combos = [("RED", "WHITE"), ("PLASMA", "GOLD"), ("GHOST", "RED")]
    for policy in (RandomPolicy, SmartPolicy):
        for n, (deck, stake) in enumerate(combos):
            adapter = LiveAdapter(FakeLive())
            result = play_game(adapter, policy(), Job(f"QUICK{n}", deck, stake), max_steps=150)
            assert result.outcome in ("lost", "won", "max-steps"), result.line()
            steps += result.steps
            kinds.update(result.actions)
    assert steps > 150
    assert {"SelectBlind", "PlayHand", "Discard", "CashOut", "BuyCard", "NextRound"} <= set(kinds)


def test_the_live_ids_are_matched_not_compared() -> None:
    adapter = LiveAdapter(FakeLive(ids=lambda i: 9 * i + 12345))
    adapter.start("RED", "WHITE", "IDS")
    _play(adapter, 40)
    assert adapter.steps > 10
    live_ids = {c["id"] for c in adapter.view["cards"]["cards"]}
    sim_ids = {c["id"] for c in adapter.shadow_view["cards"]["cards"]}
    assert live_ids.isdisjoint(sim_ids)


def test_start_reads_the_seed_back() -> None:
    adapter = LiveAdapter(SimBackend())
    view = adapter.start("BLUE", "GREEN")  # no seed: the live game picks one
    assert adapter.game.seed == view["seed"] and len(view["seed"]) == 8
    assert adapter.start_params["back"] == "b_blue" and adapter.start_params["stake_level"] == 3


def test_attach_shadows_a_run_started_elsewhere() -> None:
    live = SimBackend()
    live.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "ATTACHED"})
    adapter = LiveAdapter(live)
    adapter.attach()
    assert adapter.game.seed == "ATTACHED"
    adapter.step(SelectBlind())
    with pytest.raises(ValueError, match="first blind select"):
        LiveAdapter(live).attach()


def test_a_challenge_needs_attach() -> None:
    with pytest.raises(ValueError, match="attach"):
        LiveAdapter(SimBackend()).reset("b_red", 1, "X", challenge="c_omelette_1")


def test_bridge_adapter_is_a_live_adapter() -> None:
    assert isinstance(BridgeAdapter(LiveBackend(url="http://127.0.0.1:1")), LiveAdapter)
    assert isinstance(BridgeAdapter(FakeLive()), LiveAdapter)
    assert isinstance(BridgeAdapter(SimBackend()), LiveAdapter)


# ---------------------------------------------------------------------------
# Injected differences: the divergence, its bundle, replay
# ---------------------------------------------------------------------------


def _money_at(call: int):
    calls = {"n": 0}

    def perturb(method: str, params: dict, view: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == call:
            view = dict(view, money=view["money"] + 1)
        return view

    return perturb


def test_an_injected_difference_raises_and_writes_a_bundle(tmp_path: Path) -> None:
    adapter = LiveAdapter(FakeLive(perturb=_money_at(6)), bundle_dir=tmp_path, label="money")
    adapter.start("RED", "WHITE", "BUNDLE1")
    with pytest.raises(DivergenceError) as info:
        _play(adapter, 50)
    err = info.value
    assert err.kind == "view" and err.unclassified and err.step == 4
    assert [(d.path, d.live - d.sim) for d in err.diffs] == [("money", 1)]
    assert err.bundle_path is not None and err.bundle_path.parent == tmp_path
    files = {p.name for p in err.bundle_path.iterdir()}
    assert {
        "bundle.json",
        "sim_view.json",
        "live_view.json",
        "sim_view_before.json",
        "live_view_before.json",
        "shadow_before.json",
        "summary.txt",
    } <= files
    head = json.loads((err.bundle_path / "bundle.json").read_text(encoding="utf-8"))
    assert head["start"]["seed"] == "BUNDLE1" and head["start"]["back"] == "b_red"
    assert len(head["log"]) == 3 and head["step"]["method"]
    assert head["divergence"]["diff"] == [
        {"path": "money", "sim": err.diffs[0].sim, "live": err.diffs[0].live}
    ]
    live = json.loads((err.bundle_path / "live_view.json").read_text(encoding="utf-8"))
    sim = json.loads((err.bundle_path / "sim_view.json").read_text(encoding="utf-8"))
    assert live["money"] == sim["money"] + 1
    assert "replay: jackdaw lockstep replay" in (err.bundle_path / "summary.txt").read_text(
        encoding="utf-8"
    )
    # The adapter refuses to go on with a diverged shadow.
    with pytest.raises(RuntimeError, match="diverged"):
        adapter.step(SelectBlind())

    report = replay_bundle(err.bundle_path)
    assert report.replayed and report.shadow_matches
    assert report.sim_changed == [] and report.reproduced
    assert [d.path for d in report.diffs] == ["money"]
    assert report.exit_code() == 1


def test_a_card_difference_names_the_card(tmp_path: Path) -> None:
    def perturb(method: str, params: dict, view: dict) -> dict:
        if method == "select":
            view["hand"]["cards"][2]["modifier"] = {"seal": "RED"}
        return view

    adapter = LiveAdapter(FakeLive(perturb=perturb), bundle_dir=tmp_path)
    adapter.start("RED", "WHITE", "CARDDIFF")
    with pytest.raises(DivergenceError) as info:
        adapter.step(SelectBlind())
    (d,) = info.value.diffs
    assert d.path == "hand.cards[2].modifier.seal" and d.live == "RED"
    assert "sim " in d.where and "live " in d.where


def test_a_live_refusal_and_a_live_failure_are_divergences(tmp_path: Path) -> None:
    refuse = FakeLive(
        fail=lambda m, p: (
            RPCError(-32002, "Method 'select' requires ...") if m == "select" else None
        )
    )
    adapter = LiveAdapter(refuse, bundle_dir=tmp_path)
    adapter.start("RED", "WHITE", "REFUSED")
    with pytest.raises(DivergenceError) as info:
        adapter.step(SelectBlind())
    assert info.value.kind == "live-refused" and info.value.unclassified
    timeout = FakeLive(fail=lambda m, p: TimeoutError("timed out") if m == "select" else None)
    adapter = LiveAdapter(timeout, bundle_dir=tmp_path)
    adapter.start("RED", "WHITE", "TIMEOUT")
    with pytest.raises(DivergenceError) as info:
        adapter.step(SelectBlind())
    assert info.value.kind == "live-error"
    report = replay_bundle(info.value.bundle_path)
    assert report.replayed and report.reproduced


def test_settle_compares_a_fresh_gamestate() -> None:
    """A live answer sent before the step's events have all run (here: one dollar
    short) diverges, unless ``settle`` compares the gamestate read a moment later."""

    def early(method: str, params: dict, view: dict) -> dict:
        return view if method == "gamestate" else dict(view, money=view["money"] - 1)

    adapter = LiveAdapter(FakeLive(perturb=early), settle=0.001)
    adapter.start("RED", "WHITE", "SETTLE")
    _play(adapter, 20)
    assert adapter.steps > 5
    adapter = LiveAdapter(FakeLive(perturb=early))
    with pytest.raises(DivergenceError):
        adapter.start("RED", "WHITE", "SETTLE")


def test_recheck_reads_a_differing_answer_again() -> None:
    """An answer sent before the game's events have run (here: the reroll cost a tag
    hasn't lowered yet) is read again until it agrees; a lasting difference is still
    a divergence, with the last reading in the bundle."""

    def early(method: str, params: dict, view: dict) -> dict:
        if method == "cash_out":
            view["round"]["reroll_cost"] += 1
        return view

    adapter = LiveAdapter(FakeLive(perturb=early), recheck=0.5)
    adapter.start("RED", "WHITE", "RECHECK")
    _beat_small_blind(adapter)
    assert adapter.rechecked == Counter({"cash_out": 1})
    assert LiveAdapter(FakeLive()).recheck == 0  # the simulator answers settled

    def lasting(method: str, params: dict, view: dict) -> dict:
        view["round"]["reroll_cost"] += method != "start" and view["state"] == "SHOP"
        return view

    adapter = LiveAdapter(FakeLive(perturb=lasting), recheck=0.3)
    adapter.start("RED", "WHITE", "RECHECK")
    with pytest.raises(DivergenceError) as info:
        _beat_small_blind(adapter)
    assert [d.path for d in info.value.diffs] == ["round.reroll_cost"]


def test_a_refusal_for_the_games_state_is_asked_again() -> None:
    """A live refusal while the game catches up (``use`` under a pack pick's STOP_USE) is
    retried within the recheck window; one that lasts is still a divergence."""
    refusals = {"left": 2}

    def busy(method: str, params: dict) -> Exception | None:
        if method == "select" and refusals["left"] > 0:
            refusals["left"] -= 1
            return RPCError(-32003, "Consumable 'X' cannot be used at this time")
        return None

    adapter = LiveAdapter(FakeLive(fail=busy), recheck=0.5)
    adapter.start("RED", "WHITE", "RETRY")
    adapter.step(SelectBlind())
    assert adapter.rechecked == Counter({"select (refused)": 1})
    refusals["left"] = 100
    adapter = LiveAdapter(FakeLive(fail=busy), recheck=0.2)
    adapter.start("RED", "WHITE", "RETRY")
    with pytest.raises(DivergenceError) as info:
        adapter.step(SelectBlind())
    assert info.value.kind == "live-refused"


def test_s14_the_live_deck_limit_follows_the_playing_cards() -> None:
    def follows(method: str, params: dict, view: dict) -> dict:
        if method == "add":
            view["cards"]["limit"] = len(view["cards"]["cards"]) + len(view["hand"]["cards"])
        return view

    adapter = LiveAdapter(FakeLive(perturb=follows))
    adapter.start("RED", "WHITE", "S14TEST")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "H_A"})  # into the hand: 53 playing cards
    assert adapter.view["cards"]["limit"] == 53
    assert adapter.explained["S14"] == 1


def test_s04_a_lost_rounds_hand_size_is_dropped() -> None:
    def applied(method: str, params: dict, view: dict) -> dict:
        if view.get("state") == "GAME_OVER":
            view["hand"]["limit"] -= 1  # Turtle Bean's decay, applied at once live
        return view

    adapter = LiveAdapter(FakeLive(perturb=applied))
    adapter.start("RED", "WHITE", "S04LOST")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_turtle_bean"})
    adapter.request("set", {"hands": 1})
    adapter.request("play", {"cards": [0]})
    assert adapter.view["state"] == "GAME_OVER"
    assert adapter.explained["S04-GAME-OVER"] == 1


def test_a_refusal_by_the_shadow_never_reaches_the_live_game() -> None:
    live = FakeLive()
    adapter = LiveAdapter(live)
    adapter.start("RED", "WHITE", "REFUSE")
    sent = len(live.calls)
    with pytest.raises(IllegalActionError):
        adapter.step(PlayHand((0,)))  # not in a round
    with pytest.raises(RPCError):
        adapter.request("play", {"cards": [0]})
    assert len(live.calls) == sent


# ---------------------------------------------------------------------------
# Reconciles
# ---------------------------------------------------------------------------


def test_s10_reconciles_a_one_chip_score_and_replays(tmp_path: Path) -> None:
    adapter = LiveAdapter(FakeLive(after=score_off_by_one), bundle_dir=tmp_path)
    adapter.start("RED", "WHITE", "S10TEST")
    adapter.step(SelectBlind())
    adapter.step(PlayHand((0,)))
    assert adapter.reconciled == Counter({"S10": 1})
    assert adapter.game.chips == adapter.view["round"]["chips"]
    _play(adapter, 10)
    # A later divergence replays through the recorded reconciles.
    adapter.backend.perturb = _money_at(1)
    with pytest.raises(DivergenceError) as info:
        _play(adapter, 30)
    head = json.loads((info.value.bundle_path / "bundle.json").read_text(encoding="utf-8"))
    assert any("reconcile" in entry for entry in head["log"])
    report = replay_bundle(info.value.bundle_path)
    assert report.shadow_matches and report.reproduced


def test_s01_takes_the_live_deal_order() -> None:
    adapter = LiveAdapter(FakeLive(after=reshuffle_round()))
    adapter.start("RED", "WHITE", "S01TEST")
    _beat_small_blind(adapter)
    adapter.request("add", {"key": "j_marble"})
    adapter.request("next_round")
    adapter.step(SelectBlind())
    assert adapter.reconciled == Counter({"S01": 1})
    hand = [c["key"] for c in adapter.shadow_view["hand"]["cards"]]
    assert hand == [c["key"] for c in adapter.view["hand"]["cards"]]
    _play(adapter, 30)  # and the round goes on in step


def test_without_marble_a_new_deal_order_is_unclassified() -> None:
    def deal(method: str, params: dict, game: Game, before: dict) -> bool:
        return method == "select" and reshuffle_round()(method, params, _with_marble(game), before)

    adapter = LiveAdapter(FakeLive(after=deal))
    adapter.start("RED", "WHITE", "S01TEST")
    with pytest.raises(DivergenceError) as info:
        adapter.step(SelectBlind())
    assert info.value.unclassified and info.value.diffs


def _with_marble(game: Game) -> Game:
    """Make the fake's hook see Marble Joker (the run itself has none)."""

    class _View:
        def __getattr__(self, name: str):  # noqa: ANN204
            return getattr(game, name)

        def bot_view(self) -> dict:
            view = game.bot_view()
            view["jokers"]["cards"] = [{"key": "j_marble"}]
            return view

    return _View()  # type: ignore[return-value]


@pytest.mark.parametrize(("spectral", "created_only"), [("c_sigil", False), ("c_familiar", True)])
def test_s08_takes_the_live_fronts(spectral: str, created_only: bool) -> None:
    adapter = LiveAdapter(FakeLive(after=resuit(created_only)))
    adapter.start("RED", "WHITE", f"S08{spectral}")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": spectral})
    adapter.step(UseConsumable(0, None))
    assert adapter.reconciled == Counter({"S08": 1})
    _play(adapter, 30)


def test_s03_takes_the_live_seals_of_a_standard_pack() -> None:
    def reseal(method: str, params: dict, game: Game, before: dict) -> bool:
        if method != "buy" or "pack" not in params:
            return False
        snap = game.to_dict()
        if (snap.get("pack") or {}).get("kind") != "Standard":
            return False
        for card in snap["areas"]["pack_cards"]:
            card["seal"] = "Purple" if card["seal"] is None else None
        game.replace_state(snap)
        return True

    adapter = LiveAdapter(FakeLive(after=reseal))
    adapter.start("RED", "WHITE", "S03TEST")
    _beat_small_blind(adapter)
    adapter.request("add", {"key": "j_certificate"})
    adapter.request("next_round")
    _beat_small_blind(adapter)  # the Big Blind, where Certificate fires
    _open_pack(adapter, "p_standard_normal_1")
    assert adapter.reconciled["S03"] >= 1
    assert adapter.shadow_view["pack"]["cards"] == adapter.view["pack"]["cards"] or all(
        a["modifier"] == b["modifier"]
        for a, b in zip(
            adapter.shadow_view["pack"]["cards"], adapter.view["pack"]["cards"], strict=True
        )
    )


def _reprice(copies_only: bool):
    def reprice(method: str, params: dict, game: Game, before: dict) -> bool:
        if method != "use":
            return False
        old = {c["id"] for c in before["jokers"]["cards"]}
        snap = game.to_dict()
        for card in snap["areas"]["jokers"]:
            if card["sort_id"] not in old or not copies_only:
                card["costs"]["sell_cost"] += 2
        game.replace_state(snap)
        return True

    return reprice


def test_s05_takes_the_live_price_of_a_copy() -> None:
    adapter = LiveAdapter(FakeLive(after=_reprice(copies_only=True)))
    adapter.start("RED", "WHITE", "S05TEST")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_egg"})
    adapter.request("add", {"key": "c_ankh"})
    adapter.step(UseConsumable(0, None))
    assert adapter.reconciled == Counter({"S05": 1})
    assert [c["cost"] for c in adapter.shadow_view["jokers"]["cards"]] == [
        c["cost"] for c in adapter.view["jokers"]["cards"]
    ]
    # Only the copy: the original's price is the simulator's business.
    adapter = LiveAdapter(FakeLive(after=_reprice(copies_only=False)))
    adapter.start("RED", "WHITE", "S05TEST")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_egg"})
    adapter.request("add", {"key": "c_ankh"})
    with pytest.raises(DivergenceError) as info:
        adapter.step(UseConsumable(0, None))
    assert info.value.unclassified


def test_s09_takes_the_live_perishable_tally() -> None:
    def no_tick(method: str, params: dict, game: Game, before: dict) -> bool:
        if method != "play" or game.phase.value != "round_eval":
            return False
        snap = game.to_dict()
        for card in snap["areas"]["jokers"]:
            if card["stickers"]["perishable"]:
                card["stickers"]["perish_tally"] += 1  # a debuffed joker doesn't tick live
        game.replace_state(snap)
        return True

    adapter = LiveAdapter(FakeLive(after=no_tick))
    adapter.start("RED", "WHITE", "S09TEST")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_joker", "perishable": 3})
    adapter.request("set", {"chips": 100000})
    adapter.request("play", {"cards": [0]})
    assert adapter.reconciled == Counter({"S09": 1})
    tally = adapter.shadow_view["jokers"]["cards"][0]["modifier"]["perishable"]
    assert tally == adapter.view["jokers"]["cards"][0]["modifier"]["perishable"]


def test_s06_takes_the_live_score_of_seeing_double() -> None:
    def doubled(method: str, params: dict, game: Game, before: dict) -> bool:
        if method != "play" or game.phase.value != "selecting_hand":
            return False
        game.debug_set(chips=game.chips * 2)
        return True

    adapter = LiveAdapter(FakeLive(after=doubled))
    adapter.start("RED", "WHITE", "S06TEST")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_seeing_double"})
    adapter.request("add", {"key": "H_A", "enhancement": "WILD"})
    hand = adapter.view["hand"]["cards"]
    wild = next(i for i, c in enumerate(hand) if (c["modifier"] or {}).get("enhancement") == "WILD")
    adapter.step(PlayHand((wild,)))
    assert adapter.reconciled == Counter({"S06": 1})
    # The same doubling on an ordinary play is a real divergence.
    with pytest.raises(DivergenceError):
        adapter.step(PlayHand((0, 1)))


def test_b03_takes_the_live_debuff_of_an_added_card() -> None:
    def debuff_added(method: str, params: dict, game: Game, before: dict) -> bool:
        if method != "add" or params.get("key") != "C_A":
            return False
        snap = game.to_dict()
        snap["areas"]["hand"][-1]["debuff"] = True
        game.replace_state(snap)
        return True

    adapter = LiveAdapter(FakeLive(after=debuff_added))
    adapter.start("RED", "WHITE", "B03TEST")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "C_A", "enhancement": "BONUS"})
    assert adapter.reconciled == Counter({"B03": 1})
    assert adapter.shadow_view["hand"]["cards"][-1]["state"] == {"debuff": True}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _forced_boss(boss: str):
    def make(back: str, stake: int, seed: str, challenge: str | None) -> Game:
        g = Game.new(back, stake, seed, challenge)
        g.apply_setup({"op": "force_boss", "key": boss})
        return g

    return make


def _defeat_answers_early(extra_suit: str | None = None):
    """BB-BOSS-DEFEAT: the play that beats a boss answers before Blind:defeat's
    set_blind(nil): what was debuffed is still debuffed, the hand size as it was.
    ``extra_suit``: also debuff the deck's cards of that suit (not the boss's doing)."""
    last: dict[str, dict] = {}

    def perturb(method: str, params: dict, view: dict) -> dict:
        before = last.get("view")
        if method == "play" and view["state"] == "ROUND_EVAL" and before is not None:
            debuffed = {
                c["id"]
                for area in ("hand", "cards", "jokers")
                for c in before[area]["cards"]
                if (c["state"] or {}).get("debuff")
            }
            for area in ("cards", "jokers"):
                for c in view[area]["cards"]:
                    suit = (c["value"] or {}).get("suit")
                    if c["id"] in debuffed or (extra_suit and suit == extra_suit):
                        c["state"] = {**(c["state"] or {}), "debuff": True}
            view["hand"]["limit"] = before["hand"]["limit"]
        last["view"] = view
        return view

    return perturb


def test_the_boss_debuffs_outlasting_the_winning_play_are_dropped() -> None:
    make = _forced_boss("bl_club")
    live = FakeLive(game=make("b_red", 1, "DEFEAT1", None), perturb=_defeat_answers_early())
    adapter = LiveAdapter(live, shadow_factory=make)
    adapter.attach()
    adapter.step(SelectBlind())
    adapter.request("set", {"chips": 10**6})
    adapter.request("play", {"cards": [0]})
    assert adapter.explained["BB-BOSS-DEFEAT"] == 13  # every Club
    adapter.request("cash_out")  # the next answer has the debuffs lifted


def test_the_manacles_hand_size_outlasts_the_winning_play() -> None:
    make = _forced_boss("bl_manacle")
    live = FakeLive(game=make("b_red", 1, "DEFEAT4", None), perturb=_defeat_answers_early())
    adapter = LiveAdapter(live, shadow_factory=make)
    adapter.attach()
    adapter.step(SelectBlind())
    assert adapter.view["hand"]["limit"] == 7
    adapter.request("set", {"chips": 10**6})
    adapter.request("play", {"cards": [0]})
    assert adapter.explained["BB-BOSS-DEFEAT"] == 1  # hand.limit: 7 live, 8 in the shadow
    adapter.request("cash_out")


def test_crimson_hearts_joker_and_its_passive_outlast_the_winning_play() -> None:
    make = _forced_boss("bl_final_heart")
    live = FakeLive(game=make("b_red", 1, "DEFEAT2", None), perturb=_defeat_answers_early())
    adapter = LiveAdapter(live, shadow_factory=make)
    adapter.attach()
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_juggler"})
    adapter.request("play", {"cards": [0]})  # the next draw debuffs Juggler: one card fewer
    assert adapter.view["jokers"]["cards"][0]["state"] == {"debuff": True}
    adapter.request("set", {"chips": 10**6})
    adapter.request("play", {"cards": [0]})
    assert adapter.explained["BB-BOSS-DEFEAT"] == 2  # Juggler's debuff and the hand size
    adapter.request("cash_out")


def _stale_cash_out(at: int, extra: int = 0):
    """BB-CASHOUT-RACE: the ``at``-th cash_out pays the previous round's total
    (plus ``extra``: then it isn't the race)."""
    seen = {"n": 0, "last": None}

    def after(method: str, params: dict, game: Game, before: dict) -> bool:
        if method != "cash_out":
            return False
        seen["n"] += 1
        paid = game.dollars - before["money"]
        last, seen["last"] = seen["last"], paid
        if seen["n"] != at or last is None:
            return False
        game.debug_set(money=before["money"] + last + extra)
        return True

    return after


def _two_rounds(adapter: LiveAdapter) -> None:
    for n in range(2):
        if n:
            adapter.request("next_round")
        adapter.step(SelectBlind())
        adapter.request("set", {"chips": 10**6})
        adapter.request("play", {"cards": [0]})
        adapter.request("cash_out")


def test_a_cash_out_paying_the_last_rounds_total_is_reconciled() -> None:
    adapter = LiveAdapter(FakeLive(after=_stale_cash_out(2)))
    adapter.start("RED", "WHITE", "RACE1")
    _two_rounds(adapter)
    assert adapter.reconciled == Counter({"BB-CASHOUT-RACE": 1})
    assert adapter.game.dollars == adapter.view["money"]
    assert adapter.shadow_view["money"] == adapter.view["money"]
    _play(adapter, 20)  # and the run goes on in step


def test_another_wrong_payout_is_unclassified() -> None:
    adapter = LiveAdapter(FakeLive(after=_stale_cash_out(2, extra=1)))
    adapter.start("RED", "WHITE", "RACE2")
    with pytest.raises(DivergenceError) as info:
        _two_rounds(adapter)
    assert [d.path for d in info.value.diffs] == ["money"]


def test_live_backend_waits_before_cash_out_unless_it_is_the_simulator(monkeypatch) -> None:
    import jackdaw.bridge.backend as backend_mod

    class Response:
        def __init__(self, server: str) -> None:
            self.headers = {"server": server} if server else {}

        def json(self) -> dict:
            return {"jsonrpc": "2.0", "result": {}, "id": 1}

    for server, waits in (("", True), ("jackdaw", False)):
        slept: list[float] = []
        monkeypatch.setattr(backend_mod.time, "sleep", slept.append)
        backend = LiveBackend(cash_out_wait=0.5)

        class Client:
            def post(self, url: str, json: dict) -> Response:  # noqa: A002 - httpx's name
                return Response(server)

        backend._client = Client()
        backend.handle("play", {"cards": [0]})
        backend.handle("cash_out", {})
        backend.handle("gamestate", {})
        assert bool(slept) is waits and all(0 < s <= 0.5 for s in slept)


def test_a_debuff_the_boss_did_not_cause_is_unclassified() -> None:
    make = _forced_boss("bl_club")
    live = FakeLive(game=make("b_red", 1, "DEFEAT3", None), perturb=_defeat_answers_early("H"))
    adapter = LiveAdapter(live, shadow_factory=make)
    adapter.attach()
    adapter.step(SelectBlind())
    adapter.request("set", {"chips": 10**6})
    with pytest.raises(DivergenceError) as info:
        adapter.request("play", {"cards": [0]})
    assert len(info.value.diffs) == 13  # the Hearts
    assert all(d.path.endswith(".state.debuff") and d.live is True for d in info.value.diffs)


# ---------------------------------------------------------------------------
# Vetoes
# ---------------------------------------------------------------------------


def test_sim_only_moves_are_vetoed() -> None:
    live = FakeLive()
    adapter = LiveAdapter(live)
    adapter.start("RED", "WHITE", "VETOES")
    sent = len(live.calls)
    with pytest.raises(VetoedActionError) as info:
        adapter.step(RerollBoss())
    assert info.value.deviation == "BB-SIM-ONLY"
    assert RerollBoss() not in adapter.get_legal_actions()
    # Using a consumable with a pack open: vanilla allows it, BalatroBot doesn't.
    _beat_small_blind(adapter)
    adapter.request("add", {"key": "c_pluto"})
    _open_pack(adapter, "p_arcana_normal_1")
    with pytest.raises(VetoedActionError) as info:
        adapter.step(UseConsumable(0, None))
    assert info.value.deviation == "BB-SIM-ONLY"
    assert adapter.vetoed["BB-SIM-ONLY"] == 2
    assert sum(1 for m, _ in live.calls[sent:] if m == "use") == 0


def test_selling_an_eternal_joker_is_vetoed() -> None:
    adapter = LiveAdapter(FakeLive())
    adapter.start("RED", "WHITE", "ETERNAL")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_joker", "eternal": True})
    with pytest.raises(VetoedActionError) as info:
        adapter.step(SellCard("jokers", 0))
    assert info.value.deviation == "BB-ETERNAL"


def test_a_pack_tag_skip_is_left_out_and_vetoed() -> None:
    from tests.live.test_deviations import _seed_with_small_tag

    seed = _seed_with_small_tag({"tag_charm", "tag_meteor", "tag_buffoon", "tag_standard"})
    adapter = LiveAdapter(FakeLive())
    adapter.start("RED", "WHITE", seed)
    assert SkipBlind() in adapter.game.legal_actions("balatrobot")
    assert SkipBlind() not in adapter.get_legal_actions()
    with pytest.raises(VetoedActionError) as info:
        adapter.step(SkipBlind())
    assert info.value.deviation == "BB-SKIP-PACK"


def test_b02_veto_through_a_shadow_factory() -> None:
    def cerulean(back: str, stake: int, seed: str, challenge: str | None) -> Game:
        g = Game.new(back, stake, seed, challenge)
        g.apply_setup({"op": "force_boss", "key": "bl_final_bell"})
        return g

    live = cerulean("b_red", 1, "B02LIVE", None)
    adapter = LiveAdapter(FakeLive(game=live), shadow_factory=cerulean)
    adapter.attach()
    adapter.step(SelectBlind())
    hand = adapter.view["hand"]["cards"]
    forced = next(i for i, c in enumerate(hand) if (c["state"] or {}).get("highlight"))
    others = tuple(i for i in range(len(hand)) if i != forced)[:2]
    with pytest.raises(VetoedActionError) as info:
        adapter.step(PlayHand(others))
    assert info.value.deviation == "B02"
    adapter.step(PlayHand(tuple(sorted((*others, forced)))))


# ---------------------------------------------------------------------------
# The env on a live adapter
# ---------------------------------------------------------------------------


def test_the_env_plays_through_the_live_adapter() -> None:
    from jackdaw.env.agents import RandomAgent
    from jackdaw.env.balatro_env import BalatroEnvironment

    random.seed(5)
    env = BalatroEnvironment(adapter_factory=lambda: LiveAdapter(FakeLive()), max_steps=120)
    obs, mask, info = env.reset(seed="ENVLIVE")
    adapter = env._adapter
    assert info["game"] is adapter.game
    agent = RandomAgent()
    steps = 0
    done = False
    while not done:
        action = agent.act(obs, info["action_mask"], info)
        obs, terminated, truncated, mask, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps > 5 and adapter.steps >= steps - 1
    assert info["game"] is adapter.game


def test_the_mask_follows_balatrobot_legality() -> None:
    from jackdaw.env.action_space import ActionType
    from jackdaw.env.balatro_env import BalatroEnvironment

    env = BalatroEnvironment(adapter_factory=lambda: LiveAdapter(FakeLive()))
    env.reset(seed="MASKLIVE")
    adapter = env._adapter
    _beat_small_blind(adapter)
    adapter.request("add", {"key": "c_pluto"})
    _open_pack(adapter, "p_celestial_normal_1")
    from jackdaw.env.balatro_env import _rust_view

    _, mask, _, _ = _rust_view(adapter.game._g.env_view("balatrobot"))
    vanilla = _rust_view(adapter.game._g.env_view("vanilla"))[1]
    assert vanilla.type_mask[ActionType.UseConsumable]
    assert not mask.type_mask[ActionType.UseConsumable]
    assert mask.type_mask[ActionType.PickPackCard]


# ---------------------------------------------------------------------------
# Classified or not
# ---------------------------------------------------------------------------


def test_a_reconcile_that_does_not_fit_is_unclassified() -> None:
    """S01 is active, but the live deal holds a card the shadow doesn't have: that is
    not Marble Joker's doing."""

    def deal_and_change(method: str, params: dict, game: Game, before: dict) -> bool:
        if not reshuffle_round()(method, params, game, before):
            return False
        snap = game.to_dict()
        card = snap["areas"]["deck"][0]
        front = "S_A" if card["front"] != "S_A" else "H_A"
        card["front"] = front
        card["base"] = front_base(front, card["base"], created=False)
        game.replace_state(snap)
        return True

    adapter = LiveAdapter(FakeLive(after=deal_and_change))
    adapter.start("RED", "WHITE", "S01TEST")
    _beat_small_blind(adapter)
    adapter.request("add", {"key": "j_marble"})
    adapter.request("next_round")
    with pytest.raises(DivergenceError) as info:
        adapter.step(SelectBlind())
    err = info.value
    assert err.kind == "reconcile" and err.unclassified and not err.classified
    assert "S01 doesn't fit" in str(err)


def test_a_known_limit_of_a_reconcile_is_classified() -> None:
    """S09: a perishable joker that expires in the live game only would need its
    passives removed; the divergence is classified under S09."""

    def expire(method: str, params: dict, game: Game, before: dict) -> bool:
        if method != "play" or game.phase.value != "round_eval":
            return False
        snap = game.to_dict()
        joker = snap["areas"]["jokers"][0]
        joker["debuff"] = True
        joker["stickers"]["perish_tally"] = 0
        game.replace_state(snap)
        return True

    adapter = LiveAdapter(FakeLive(after=expire))
    adapter.start("RED", "WHITE", "S09LIMIT")
    adapter.step(SelectBlind())
    adapter.request("add", {"key": "j_joker", "perishable": 3})
    adapter.request("set", {"chips": 100000})
    with pytest.raises(DivergenceError) as info:
        adapter.request("play", {"cards": [0]})
    err = info.value
    assert err.kind == "reconcile" and not err.unclassified
    assert list(err.classified) == ["S09"]
