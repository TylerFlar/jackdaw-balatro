"""``jackdaw lockstep`` and ``jackdaw validate`` against ``jackdaw serve`` (the live stand-in)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jackdaw.bridge.backend import LiveBackend
from jackdaw.bridge.live import DivergenceError, LiveAdapter, replay_bundle
from jackdaw.bridge.lockstep import Job, SmartPolicy, play_game
from jackdaw.cli.main import main
from tests.bot.server import jackdaw_serve
from tests.live.fakes import FakeLive


@pytest.fixture(scope="module")
def serve_url() -> str:
    with jackdaw_serve() as port:
        yield f"http://127.0.0.1:{port}"


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as info:
        main(argv)
    return int(info.value.code or 0)


def test_live_adapter_over_http(serve_url: str) -> None:
    adapter = LiveAdapter(LiveBackend(url=serve_url, timeout=30))
    result = play_game(adapter, SmartPolicy(), Job("HTTPLOCK", "RED", "WHITE"), max_steps=120)
    assert result.outcome in ("lost", "won", "max-steps"), result.line()
    assert result.steps >= 10 and result.methods["play"] > 0


def test_lockstep_command_over_http(serve_url: str, tmp_path: Path, capsys) -> None:
    out = tmp_path / "run"
    code = _run(
        ["lockstep", "--url", serve_url, "--seeds", "2", "--steps", "80", "--out", str(out)]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["games"] == 2 and summary["unclassified_divergences"] == 0
    assert summary["steps"] > 20 and summary["urls"] == [serve_url]
    assert "games: 2" in capsys.readouterr().out


def test_lockstep_command_in_process(tmp_path: Path) -> None:
    out = tmp_path / "sim"
    code = _run(
        [
            "lockstep",
            "--url",
            "sim",
            "--seeds",
            "QUICK1,QUICK2,QUICK3",
            "--decks",
            "RED,BLUE",
            "--stakes",
            "WHITE",
            "--policy",
            "random",
            "--steps",
            "60",
            "--out",
            str(out),
            "--json",
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    seeds = [g["seed"] for g in summary["games_detail"]]
    assert sorted(seeds) == ["QUICK1", "QUICK2", "QUICK3"]
    assert [g["deck"] for g in sorted(summary["games_detail"], key=lambda g: g["seed"])] == [
        "RED",
        "BLUE",
        "RED",
    ]


def test_unreachable_instance(tmp_path: Path) -> None:
    code = _run(
        ["lockstep", "--url", "http://127.0.0.1:9", "--timeout", "2", "--out", str(tmp_path)]
    )
    assert code == 2


def _bundle(tmp_path: Path) -> Path:
    calls = {"n": 0}

    def perturb(method: str, params: dict, view: dict) -> dict:
        calls["n"] += 1
        if method == "play":
            view = dict(view, money=view["money"] + 3)
        return view

    adapter = LiveAdapter(FakeLive(perturb=perturb), bundle_dir=tmp_path)
    adapter.start("RED", "WHITE", "REPLAYCLI")
    adapter.request("select")
    with pytest.raises(DivergenceError) as info:
        adapter.request("play", {"cards": [0, 1]})
    assert info.value.bundle_path is not None
    return info.value.bundle_path


def test_replay_command(tmp_path: Path, capsys) -> None:
    bundle = _bundle(tmp_path)
    assert _run(["lockstep", "replay", str(bundle)]) == 1
    text = capsys.readouterr().out
    assert "REPRODUCED" in text and "money" in text
    assert _run(["lockstep", "replay", str(bundle), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["reproduced"] and report["shadow_matches"]
    assert report["unclassified"] == [
        {
            "path": "money",
            "sim": report["unclassified"][0]["sim"],
            "live": report["unclassified"][0]["sim"] + 3,
        }
    ]


def test_replay_after_the_simulator_changed(tmp_path: Path) -> None:
    """When the simulator is fixed, replay reports the divergence gone (exit 0); when it
    changed in some other way, replay shows how its view changed."""
    bundle = _bundle(tmp_path)
    live = json.loads((bundle / "live_view.json").read_text(encoding="utf-8"))
    sim = json.loads((bundle / "sim_view.json").read_text(encoding="utf-8"))
    # The simulator now gives the money the live game reported: "fixed".
    (bundle / "live_view.json").write_text(json.dumps(dict(live, money=sim["money"])), "utf-8")
    report = replay_bundle(bundle)
    assert report.replayed and not report.reproduced and report.exit_code() == 0
    # The recorded sim view differs from today's: the simulator changed.
    sim["round"]["chips"] += 7
    (bundle / "sim_view.json").write_text(json.dumps(sim), encoding="utf-8")
    (bundle / "live_view.json").write_text(json.dumps(live), encoding="utf-8")
    report = replay_bundle(bundle)
    assert [d.path for d in report.sim_changed] == ["round.chips"]
    assert report.reproduced


def test_replay_of_a_damaged_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    head = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    head["log"][0]["params"] = {"cards": [0]}  # `select` takes no cards: the log no longer replays
    head["log"][0]["method"] = "play"
    (bundle / "bundle.json").write_text(json.dumps(head), encoding="utf-8")
    report = replay_bundle(bundle)
    assert not report.replayed and report.exit_code() == 2
    assert _run(["lockstep", "replay", str(tmp_path / "nope")]) == 2


def test_validate_against_the_simulator(capsys) -> None:
    assert _run(["validate", "--sim", "--delay", "0", "--category", "planets"]) == 0
    out = capsys.readouterr().out
    assert "Total: 13/13 passed, 0 skipped" in out


def test_validate_over_http(serve_url: str, tmp_path: Path, capsys) -> None:
    code = _run(
        [
            "validate",
            "--url",
            serve_url,
            "--delay",
            "0",
            "--scenario",
            "spectral_familiar",
            "--bundles",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert "PASS" in capsys.readouterr().out
