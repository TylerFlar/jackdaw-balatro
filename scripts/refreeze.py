"""Refreeze the parity fixtures from the Rust engine.

The parity fixtures (``tests/parity/frozen.py``) keep the Python oracle's verdicts from the last
lockstep before it was retired. A deliberate behaviour change makes the Rust engine disagree
with them; once the change is reviewed, this rewrites the affected fixtures from the Rust
engine. Every tape is replayed as recorded and its digests recomputed; a step the engine now
refuses ends the tape there. Say which fixtures changed, and why, in the PR.

    .venv/Scripts/python.exe scripts/refreeze.py [fixture ...]

Fixtures (all by default): golden_v2, golden_env, joker_scenarios, fuzz, new_runs, setup_boards,
env_fuzz, env_states, env_runs, cases.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jackdaw.core import Game, IllegalActionError  # noqa: E402
from tests.parity import frozen  # noqa: E402
from tests.parity.harness import RustEngine, load_specs  # noqa: E402

# ---------------------------------------------------------------------------
# Tapes
# ---------------------------------------------------------------------------


def _ended(game: Game) -> str:
    return "game over (won)" if game.won else "game over"


def refreeze_tape(case: dict[str, Any], *, saved: bool = False) -> dict[str, Any]:
    """``case`` (a tape with its digests, :func:`tests.parity.frozen.replay_tape`) with the
    digests the Rust engine gives now. A saved tape (a fuzz regression) is kept whole, its
    last action possibly refused; otherwise ``tape`` becomes the steps taken."""
    game = frozen.new_game(case["start"], case["setup"])
    out = dict(case)
    if "setup_digest" in out:
        out["setup_digest"] = frozen.short(game.state_hash())
    actions = list(case["tape"])
    if not saved and "refused" in case:
        actions.append(case["refused"])
    taken: list[Any] = []
    digests: list[str] = []
    legal: list[str] = []
    refused = None
    stop = "tape end"
    for action in actions:
        if game.phase == "game_over":
            stop = _ended(game)
            break
        before = frozen.legal_digest(json.loads(game._g.legal_actions()))
        try:
            game.step(action)
        except IllegalActionError:
            refused, stop = action, "tape action refused"
            break
        taken.append(action)
        digests.append(frozen.short(game.state_hash()))
        legal.append(before)
    else:
        if game.phase == "game_over":
            stop = _ended(game)
    if not saved:
        out["tape"] = taken
    out["digests"] = digests
    if "legal" in out:
        out["legal"] = legal
    if "stop" in out and (saved or refused is not None):
        out["stop"] = stop
    if refused is None:
        out.pop("refused", None)
    else:
        out["refused"] = refused
    return out


def golden_v2() -> None:
    from tests.parity.golden_v2 import run_all, write_fixture

    results = run_all(load_specs(), lambda s: RustEngine(s.start, s.setup))
    failed = [f"{n}:{k}" for n, r in results.items() for k, ok in r["checks"].items() if not ok]
    if failed:
        raise SystemExit(f"scenario checks failed, golden_v2.json not written: {failed}")
    write_fixture(results)
    print(f"golden_v2.json: {len(results)} tapes")


def joker_scenarios() -> None:
    data = frozen.load("joker_scenarios.json")
    out = {key: [refreeze_tape(case) for case in cases] for key, cases in data.items()}
    frozen.write("joker_scenarios.json", out)
    print(f"joker_scenarios.json: {len(out)} jokers")


def fuzz() -> None:
    cases = frozen.load("fuzz_cases.json")["cases"]
    frozen.write("fuzz_cases.json", {"cases": [refreeze_tape(c) for c in cases]})
    path = frozen.FIXTURES / "fuzz_regressions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    regressions = [refreeze_tape(c, saved=True) for c in data["cases"]]
    path.write_text(
        '{"schema":1,"cases":[\n'
        + ",\n".join(json.dumps(c, separators=(",", ":")) for c in regressions)
        + "\n]}\n",
        encoding="utf-8",
    )
    print(f"fuzz_cases.json: {len(cases)} cases; fuzz_regressions.json: {len(regressions)}")


# ---------------------------------------------------------------------------
# New runs and setups
# ---------------------------------------------------------------------------


def new_runs() -> None:
    out = {}
    for key in frozen.load("new_runs.json"):
        if key.startswith("c_"):
            challenge, seed = key.split("/")
            game = Game.new("b_challenge", 1, seed, challenge)
        else:
            back, stake, seed = key.split("/")
            game = Game.new(back, int(stake), seed)
        out[key] = game.state_hash()
    frozen.write("new_runs.json", out)
    print(f"new_runs.json: {len(out)} runs")


def setup_boards() -> None:
    data = frozen.load("setup_boards.json")
    specs = {
        s.name: frozen.new_game(s.start, s.setup).state_hash() for s in load_specs() if s.setup
    }
    boards = [
        dict(b, digest=frozen.new_game(b["start"], b["setup"]).state_hash()) for b in data["boards"]
    ]
    frozen.write("setup_boards.json", {"specs": specs, "boards": boards})
    print(f"setup_boards.json: {len(specs)} spec setups, {len(boards)} boards")


# ---------------------------------------------------------------------------
# The env layer
# ---------------------------------------------------------------------------


def golden_env() -> None:
    out = {}
    for spec in load_specs():
        game = frozen.new_game(spec.start, spec.setup)
        chain = frozen.Chain()
        for pos in range(len(spec.tape) + 1):
            where = f"{spec.name}@{pos}"
            frozen.push_env_state(game, where, chain, frozen.table_seed(where))
            if game.phase == "game_over" or game.won or pos == len(spec.tape):
                break
            try:
                game.step(spec.tape[pos])
            except IllegalActionError:
                break
        out[spec.name] = chain.as_json()
    frozen.write("golden_env.json", out)
    print(f"golden_env.json: {len(out)} tapes")


def _refreeze_env_run(run: dict[str, Any], every: int | None) -> dict[str, Any]:
    """An env run (:func:`tests.parity.frozen.replay_env_run`) with its chain recomputed."""
    game = frozen.new_game(run["start"], run["setup"])
    chain = frozen.Chain()
    out = dict(run)
    tape = run["tape"]
    states = run["states"]
    refused = run.get("refused")
    for pos in range(run["states"]):
        where = f"{run['name']}@{pos}"
        seed = frozen.table_seed(where) if every and pos % every == 0 else None
        frozen.push_env_state(game, where, chain, seed)
        chain.push("state", frozen.short(game.state_hash()))
        if pos >= len(tape):
            continue
        entry = tape[pos]
        if isinstance(entry, dict):
            game.apply_setup(entry)
            continue
        try:
            game.step(entry)
        except IllegalActionError:
            out["tape"], states, refused = tape[:pos], pos + 1, entry
            break
    else:
        if refused is not None:
            try:
                game.step(refused)
            except IllegalActionError:
                pass
            else:
                refused = None
    out["states"] = states
    out["chain"] = chain.as_json()
    if refused is None:
        out.pop("refused", None)
    else:
        out["refused"] = refused
    return out


def env_fuzz() -> None:
    data = frozen.load("env_fuzz.json")
    out = {
        "action_space": [_refreeze_env_run(r, 4) for r in data["action_space"]],
        "observation": [_refreeze_env_run(r, None) for r in data["observation"]],
        "scenarios": [_refreeze_env_run(r, None) for r in data["scenarios"]],
    }
    frozen.write("env_fuzz.json", out)
    print(f"env_fuzz.json: {sum(len(v) for v in out.values())} runs")


def env_states() -> None:
    from tests.env.test_rust_observation import edited_snapshots, random_states, snapshot_digest

    count = len(frozen.load("env_states.json")["random"])
    edited = {
        phase: [snapshot_digest(snap) for _, snap in edited_snapshots(phase)]
        for phase in ("selecting_hand", "shop", "pack_opening")
    }
    rand = [snapshot_digest(snap) for snap in random_states(count)]
    frozen.write("env_states.json", {"edited": edited, "random": rand})
    print(f"env_states.json: {sum(map(len, edited.values()))} edited, {len(rand)} random")


_STEP_ERRORS = (IllegalActionError, IndexError, ValueError)


def env_runs() -> None:
    from jackdaw.env import BalatroEnvironment, BalatroGymnasiumEnv, DirectAdapter
    from jackdaw.env.game_spec import FactoredAction
    from tests.env.test_rust_envs import _GYM_CASES

    data = frozen.load("env_runs.json")
    environment: dict[str, Any] = {}
    for case, episodes in data["environment"].items():
        back, stake, _ = case.split("-")
        env = BalatroEnvironment(
            adapter_factory=DirectAdapter,
            max_steps=300,
            back_keys=[back],
            stakes=[int(stake)],
            seed_prefix=f"ENVPAR_{back}_{stake}",
        )
        environment[case] = []
        for episode in episodes:
            chain = frozen.Chain()
            chain.push("reset", frozen.env_reset_record(env.reset()[2]))
            tape = []
            for t, entity, cards in episode["tape"]:
                fa = FactoredAction(t, None if cards is None else tuple(cards), entity)
                try:
                    _, terminated, truncated, _, info = env.step(fa)
                except _STEP_ERRORS:
                    break
                chain.push("step", frozen.env_step_record(env, terminated, truncated, info))
                tape.append([t, entity, cards])
            environment[case].append({"tape": tape, "chain": chain.as_json()})
    gym: dict[str, Any] = {}
    for case, episodes in data["gym"].items():
        kwargs = dict(_GYM_CASES[case])
        kwargs.update(back_keys=["b_red"], stakes=[1 + len(case) % 4], seed_prefix=f"GYMPAR_{case}")
        gym_env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, **kwargs)
        gym[case] = []
        for episode in episodes:
            chain = frozen.Chain()
            obs, info = gym_env.reset(seed=episode["seed"])
            chain.push("reset", frozen.gym_record(gym_env, obs, info))
            tape = []
            for action in episode["tape"]:
                try:
                    obs, reward, terminated, truncated, info = gym_env.step(action)
                except _STEP_ERRORS:
                    break
                chain.push(
                    "step", frozen.gym_record(gym_env, obs, info, reward, terminated, truncated)
                )
                tape.append(action)
            gym[case].append({"seed": episode["seed"], "tape": tape, "chain": chain.as_json()})
    frozen.write("env_runs.json", {"environment": environment, "gym": gym})
    print(f"env_runs.json: {len(environment)} environment and {len(gym)} Gym cases")


# ---------------------------------------------------------------------------
# The targeted parity tests
# ---------------------------------------------------------------------------


def cases() -> None:
    """Run the targeted parity tests with ``JACKDAW_REFREEZE_CASES`` (``tests/parity/cases.py``):
    each run's actions and differences stay, its digests are the Rust engine's."""
    files = sorted({test.split("::")[0] for test in frozen.load("parity_cases.json")})
    env = dict(os.environ, JACKDAW_REFREEZE_CASES="1")
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files]
    code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    if code != 0:
        raise SystemExit(f"the targeted parity tests failed ({code}); their runs are refrozen")
    print(f"parity_cases.json: {len(files)} test files")


FIXTURES: dict[str, Callable[[], None]] = {
    "golden_v2": golden_v2,
    "golden_env": golden_env,
    "joker_scenarios": joker_scenarios,
    "fuzz": fuzz,
    "new_runs": new_runs,
    "setup_boards": setup_boards,
    "env_fuzz": env_fuzz,
    "env_states": env_states,
    "env_runs": env_runs,
    "cases": cases,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fixtures", nargs="*", help=f"some of {', '.join(FIXTURES)} (default: all)")
    args = ap.parse_args()
    unknown = sorted(set(args.fixtures) - set(FIXTURES))
    if unknown:
        ap.error(f"unknown fixtures: {unknown}")
    for name in args.fixtures or FIXTURES:
        t0 = time.perf_counter()
        FIXTURES[name]()
        print(f"  {name}: {time.perf_counter() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
