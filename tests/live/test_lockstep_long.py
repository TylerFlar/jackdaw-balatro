"""Long sim-against-sim lockstep runs (``pytest -m lockstep``).

With the simulator on both sides nothing may diverge, over many organic games
with both policies, and the games must exercise every kind of move: blinds,
plays and discards, the shop (cards, vouchers, packs, rerolls), packs (picks
and skips), consumables, sells, and the rearranges (sorts and swaps).
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pytest

from jackdaw.bridge.backend import LiveBackend
from jackdaw.bridge.live import LiveAdapter
from jackdaw.bridge.lockstep import (
    Job,
    RandomPolicy,
    SmartPolicy,
    Summary,
    make_jobs,
    play_game,
    run_lockstep,
)
from jackdaw.cli.main import main
from jackdaw.cli.validate import run_validate
from tests.bot.server import jackdaw_serve
from tests.live.fakes import FakeLive

pytestmark = pytest.mark.lockstep

#: Every kind of move an organic game can make through BalatroBot.
ALL_KINDS = {
    "SelectBlind",
    "SkipBlind",
    "PlayHand",
    "Discard",
    "CashOut",
    "NextRound",
    "Reroll",
    "BuyCard",
    "RedeemVoucher",
    "OpenBooster",
    "SellCard:jokers",
    "SellCard:consumables",
    "UseConsumable",
    "PickPackCard",
    "SkipPack",
    "SortHand",
    "SwapHandLeft",
    "SwapHandRight",
    "SwapJokersLeft",
    "SwapJokersRight",
}

COMBOS = (("RED", "WHITE"), ("BLUE", "RED"), ("PLASMA", "GOLD"), ("ERRATIC", "BLACK"))


@pytest.fixture(scope="module")
def two_servers() -> list[str]:
    with ExitStack() as stack:
        ports = [stack.enter_context(jackdaw_serve()) for _ in range(2)]
        yield [f"http://127.0.0.1:{p}" for p in ports]


def _report(name: str, summary: Summary) -> dict:
    totals = summary.totals()
    print(f"\n{name}: " + " | ".join(summary.lines()))
    return totals


def test_many_seeds_in_process() -> None:
    summary = Summary()
    for policy in (RandomPolicy, SmartPolicy):
        for job in _jobs(120, rng_seed=11):
            adapter = LiveAdapter(FakeLive())
            summary.add(play_game(adapter, policy(), job, max_steps=400))
    totals = _report("in process", summary)
    assert totals["unclassified_divergences"] == 0 and totals["classified_divergences"] == 0
    assert set(totals["outcomes"]) <= {"lost", "won", "max-steps"}
    assert totals["games"] == 240 and totals["steps"] > 8000
    assert ALL_KINDS <= set(totals["actions"]), ALL_KINDS - set(totals["actions"])


def _jobs(n: int, rng_seed: int) -> list[Job]:
    """``n`` random seeds, cycling through the four back/stake combinations."""
    seeds = [job.seed for job in make_jobs(n, rng_seed=rng_seed)]
    return [Job(seed, *COMBOS[i % len(COMBOS)]) for i, seed in enumerate(seeds)]


def test_many_seeds_over_http(two_servers: list[str], tmp_path: Path) -> None:
    # 200 organic seeds across 4 back/stake combinations, on 2 instances in parallel.
    jobs = _jobs(200, rng_seed=29)
    assert Counter((j.deck, j.stake) for j in jobs) == {c: 50 for c in COMBOS}
    summary = Summary()
    for policy, half in (("smart", jobs[:100]), ("random", jobs[100:])):
        backends = [LiveBackend(url=u, timeout=60) for u in two_servers]
        part = run_lockstep(
            backends, half, policy=policy, max_steps=400, bundle_dir=tmp_path / "bundles"
        )
        for result in part.results:
            summary.add(result)
    totals = _report("over http", summary)
    assert totals["games"] == 200
    assert totals["unclassified_divergences"] == 0 and totals["classified_divergences"] == 0
    assert totals["bundles"] == []
    assert {r.backend for r in summary.results} == set(two_servers)


def test_lockstep_command_with_two_instances(two_servers: list[str], tmp_path: Path) -> None:
    out = tmp_path / "cli"
    argv = ["lockstep", "--instances", "2"]
    for url in two_servers:
        argv += ["--url", url]
    argv += ["--seeds", "24", "--decks", "RED,PLASMA", "--stakes", "WHITE,GOLD"]
    argv += ["--steps", "300", "--out", str(out)]
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["games"] == 24 and summary["unclassified_divergences"] == 0
    assert len({g["backend"] for g in summary["games_detail"]}) == 2


def test_every_validate_scenario_over_http(two_servers: list[str], tmp_path: Path, capsys) -> None:
    code = run_validate(url=two_servers[0], delay=0, bundle_dir=str(tmp_path))
    out = capsys.readouterr().out
    print(out.splitlines()[-1])
    assert code == 0, out[-3000:]
