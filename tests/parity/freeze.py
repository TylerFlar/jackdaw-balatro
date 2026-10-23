"""Record the oracle's verdicts into the frozen fixtures (:mod:`tests.parity.frozen`).

Plays the Python oracle and the Rust engine side by side, stops with an error at the first
step where they disagree, and writes each fixture from the oracle's values. It needs the
oracle (``jackdaw.engine``), so it was run once, before the oracle was retired::

    PYTHONPATH=. .venv/Scripts/python.exe -m tests.parity.freeze [fixture ...]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from jackdaw import _core
from jackdaw.core import Game
from jackdaw.engine.actions import GamePhase, get_legal_actions
from jackdaw.engine.challenges import CHALLENGES
from jackdaw.engine.data.prototypes import BACKS
from tests.golden.trajectories import encode_action
from tests.parity import canon, frozen
from tests.parity.harness import load_specs
from tests.parity.pyengine import PyEngine
from tests.parity.rust_step import plain, rust_step


class Disagreement(AssertionError):
    """The engines disagree: nothing is frozen."""


def _digest(engine: PyEngine) -> str:
    return canon.digest(plain(engine.snapshot()))


def _step_both(engine: PyEngine, action: list[Any], where: str) -> bool:
    """Step the oracle and a Rust run imported from its snapshot. True when both took the
    action, False when both refused it; raises when they disagree."""
    status, after = rust_step(engine.snapshot(), action)
    try:
        engine.step(action)
        ok = True
    except Exception:
        ok = False
    if status.startswith("unported"):
        raise Disagreement(f"{where} {action}: {status}")
    if ok != (status == "ok"):
        raise Disagreement(f"{where} {action}: rust {status!r}, oracle ok={ok}")
    if ok and canon.digest(after) != _digest(engine):
        found = canon.diff(plain(engine.snapshot()), after)
        raise Disagreement(f"{where} {action}: {found[:6]}")
    return ok


# ---------------------------------------------------------------------------
# Joker scenarios (tests/parity/joker_scenarios.py)
# ---------------------------------------------------------------------------


def _joker_case(key: str, index: int, steps: int = 40) -> dict[str, Any]:
    from tests.parity.joker_scenarios import _policy

    start = {"back": "b_red", "stake": 1, "seed": f"JK{index}_{key}", "challenge": None}
    setup = [{"op": "action", "action": ["SelectBlind"]}, {"op": "add", "key": key}]
    engine = PyEngine(start, setup)
    case: dict[str, Any] = {"start": start, "setup": setup}
    rust = frozen.new_game(start, setup)
    if rust.state_hash() != _digest(engine):
        raise Disagreement(f"{key}#{index}: setup")
    case["setup_digest"] = frozen.short(rust.state_hash())
    tape: list[Any] = []
    digests: list[str] = []
    rng = random.Random(f"{key}:{index}")
    for pos in range(steps):
        snap = engine.snapshot()
        if snap["phase"] == "game_over" or snap["run"]["won"]:
            break
        with engine._ids():
            legal = get_legal_actions(engine.gs)
            if not legal:
                break
            action = encode_action(_policy(engine.gs, legal, rng))
        if not _step_both(engine, action, f"{key}#{index}@{pos}"):
            case["refused"] = action
            break
        tape.append(action)
        digests.append(frozen.short(_digest(engine)))
    case["tape"] = tape
    case["digests"] = digests
    return case


def joker_scenarios() -> None:
    out = {}
    for key in sorted(_core._ported_jokers()):
        out[key] = [_joker_case(key, index) for index in range(2)]
    frozen.write("joker_scenarios.json", out)
    steps = sum(len(c["tape"]) for v in out.values() for c in v)
    print(f"joker_scenarios.json: {len(out)} jokers, {steps} steps")


# ---------------------------------------------------------------------------
# Fuzz cases and regressions (tests/parity/fuzz.py)
# ---------------------------------------------------------------------------


def _fuzz_run(case: Any, tape: list[list[Any]] | None = None) -> dict[str, Any]:
    """:func:`tests.parity.fuzz.run_case`, recording the tape, each state's legal-action
    digest and each step's state digest."""
    from tests.parity.fuzz import _choose

    py = PyEngine(case.start, case.setup)
    game = frozen.new_game(case.start, case.setup)
    if game.state_hash() != _digest(py):
        raise Disagreement(f"{case.name}: setup")
    out: dict[str, Any] = {
        "name": case.name,
        "start": case.start,
        "setup": case.setup,
        "setup_digest": frozen.short(game.state_hash()),
    }
    taken: list[Any] = []
    digests: list[str] = []
    legal_digests: list[str] = []
    rng = random.Random(case.policy_seed)
    stop = "max steps"
    for pos in range(case.max_steps):
        snap = plain(py.snapshot())
        if snap["phase"] == GamePhase.GAME_OVER:
            stop = "game over (won)" if snap["run"]["won"] else "game over"
            break
        py_legal = py.legal_actions()
        rs_legal = json.loads(game._g.legal_actions())
        if rs_legal != py_legal:
            raise Disagreement(f"{case.name}@{pos}: legal actions")
        if tape is not None:
            if pos == len(tape):
                stop = "tape end"
                break
            action = tape[pos]
        else:
            action = _choose(py, rng, case.smart)
        if action is None:
            stop = "no legal action"
            break
        if not _step_both(py, action, f"{case.name}@{pos}"):
            if tape is None:
                raise Disagreement(f"{case.name}@{pos}: both refused the legal {action}")
            out["refused"] = action
            stop = "tape action refused"
            break
        game.step(action)
        if game.state_hash() != _digest(py):
            raise Disagreement(f"{case.name}@{pos}: the continued Rust run differs")
        taken.append(action)
        digests.append(frozen.short(game.state_hash()))
        legal_digests.append(frozen.legal_digest(py_legal))
    out.update(tape=taken, digests=digests, legal=legal_digests, stop=stop)
    return out


def fuzz() -> None:
    from tests.parity.fuzz import Case, random_case

    cases = [random_case(i, random.Random(1000 + i), max_steps=300) for i in range(8)]
    frozen.write("fuzz_cases.json", {"cases": [_fuzz_run(c) for c in cases]})
    path = frozen.FIXTURES / "fuzz_regressions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for spec in data["cases"]:
        case = Case(spec["name"], spec["start"], spec["setup"], 0, 0.0, len(spec["tape"]) + 1)
        run = _fuzz_run(case, spec["tape"])
        if run["stop"] not in ("tape end", "tape action refused", "game over", "game over (won)"):
            raise Disagreement(f"{spec['name']}: {run['stop']}")
        spec.update(
            setup_digest=run["setup_digest"],
            digests=run["digests"],
            legal=run["legal"],
            stop=run["stop"],
        )
        if "refused" in run:
            spec["refused"] = run["refused"]
    path.write_text(
        '{"schema":1,"cases":[\n'
        + ",\n".join(json.dumps(c, separators=(",", ":")) for c in data["cases"])
        + "\n]}\n",
        encoding="utf-8",
    )
    print(f"fuzz_cases.json: {len(cases)} cases; fuzz_regressions.json: {len(data['cases'])}")


# ---------------------------------------------------------------------------
# New runs and setups (tests/parity/test_rust_new_run.py, test_rust_setup.py)
# ---------------------------------------------------------------------------

NEW_RUN_SEEDS = ("NEWRUN01", "7LB2WVPK")


def new_runs() -> None:
    out: dict[str, str] = {}
    backs = sorted(k for k in BACKS if k != "b_challenge")
    runs = [(b, s, None) for b in backs for s in range(1, 9)]
    runs += [("b_challenge", 1, c) for c in sorted(CHALLENGES)]
    for back, stake, challenge in runs:
        for seed in NEW_RUN_SEEDS:
            start = {"back": back, "stake": stake, "seed": seed, "challenge": challenge}
            digest = _digest(PyEngine(start))
            text = _core._new_run(back, stake, seed, challenge)
            rust = canon.digest(canon.from_json_compatible(json.loads(text)))
            if rust != digest:
                raise Disagreement(f"new run {start}")
            key = f"{challenge}/{seed}" if challenge else f"{back}/{stake}/{seed}"
            out[key] = digest
    frozen.write("new_runs.json", out)
    print(f"new_runs.json: {len(out)} runs")


def setup_boards() -> None:
    from tests.parity.fuzz import random_case

    specs = {}
    for spec in load_specs():
        if spec.setup:
            digest = _digest(PyEngine(spec.start, spec.setup))
            if frozen.new_game(spec.start, spec.setup).state_hash() != digest:
                raise Disagreement(f"{spec.name}: setup")
            specs[spec.name] = digest
    cases = [c for c in (random_case(i, random.Random(i)) for i in range(400)) if c.setup][:150]
    boards = []
    for case in cases:
        digest = _digest(PyEngine(case.start, case.setup))
        if frozen.new_game(case.start, case.setup).state_hash() != digest:
            raise Disagreement(f"{case.name}: setup")
        boards.append(
            {"name": case.name, "start": case.start, "setup": case.setup, "digest": digest}
        )
    frozen.write("setup_boards.json", {"specs": specs, "boards": boards})
    print(f"setup_boards.json: {len(specs)} spec setups, {len(boards)} boards")


# ---------------------------------------------------------------------------
# The env layer (tests/env/test_rust_observation.py, test_rust_action_space.py)
# ---------------------------------------------------------------------------


def _py_conversions(engine: PyEngine, snap: dict[str, Any], table: list[Any]) -> list[Any]:
    """:func:`tests.parity.frozen.conversions` computed by the Python env layer on the
    oracle's state."""
    from jackdaw.env.action_space import (
        _default_buy_targets,
        _default_pick_targets,
        _default_use_targets,
        engine_action_to_factored,
        factored_to_engine_action,
        get_consumable_target_info,
    )
    from jackdaw.env.game_spec import FactoredAction

    with engine._ids():
        gs = engine.gs
        f2e = []
        for t, e, cards in frozen.conversion_samples(table):
            fa = FactoredAction(
                action_type=t, entity_target=e, card_target=None if cards is None else tuple(cards)
            )
            try:
                out = encode_action(factored_to_engine_action(fa, gs))
            except ValueError as exc:
                out = ["ValueError", str(exc)]
            f2e.append([[t, e, cards], out])
        e2f = []
        for a in get_legal_actions(gs):
            fa = engine_action_to_factored(a, gs)
            cards = list(fa.card_target) if fa.card_target is not None else None
            e2f.append([encode_action(a), [int(fa.action_type), fa.entity_target, cards]])
        defaults = {}
        for area, fn in (
            ("consumables", _default_use_targets),
            ("pack_cards", _default_pick_targets),
            ("shop_jokers", _default_buy_targets),
        ):
            got = (fn(gs, i) for i in range(len(snap["areas"][area])))
            defaults[area] = [list(t) if t is not None else None for t in got]
        info = {
            area: [list(get_consumable_target_info(c)) for c in gs.get(area, [])]
            for area in ("consumables", "pack_cards")
        }
    return [f2e, e2f, defaults, info]


class EnvRecorder:
    """The env layer's outputs at oracle states, checked against the Rust env layer and folded
    into a chain: per state ``obs``, ``mask`` and (with a table seed) ``conv``."""

    def __init__(self) -> None:
        from jackdaw.env.game_interface import OracleAdapter
        from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

        self.gym = BalatroGymnasiumEnv(adapter_factory=OracleAdapter)

    def state(self, engine: PyEngine, where: str, chain: frozen.Chain, seed: int | None) -> dict:
        from jackdaw.env.action_space import get_action_mask
        from jackdaw.env.balatro_env import _action_mask_to_game
        from jackdaw.env.observation import encode_observation
        from jackdaw.env.rust_env import action_mask
        from jackdaw.env.rust_observation import encode_game
        from jackdaw.env.sampler import Xoshiro256

        snap = engine.snapshot()
        game = Game.from_dict(plain(snap))
        with engine._ids():
            gs = engine.gs
            obs = frozen.obs_digest(encode_observation(gs))
            mask = get_action_mask(gs)
            mask_d = frozen.mask_digest(mask)
            table = None
            if seed is not None:
                self.gym._rng = Xoshiro256(seed)
                table = self.gym._enumerate_actions(_action_mask_to_game(mask), {"raw_state": gs})
        if frozen.obs_digest(encode_game(game)) != obs:
            raise Disagreement(f"{where}: observation")
        if frozen.mask_digest(action_mask(game)) != mask_d:
            raise Disagreement(f"{where}: mask")
        chain.push("obs", obs)
        chain.push("mask", mask_d)
        if seed is not None:
            assert table is not None
            rs_table, rs_state = game._g.action_table(list(Xoshiro256(seed).state))
            same = frozen.table_parts(table) == frozen.table_parts(rs_table)
            if not same or self.gym._rng.state != tuple(rs_state):
                raise Disagreement(f"{where}: action table")
            # At game over no action is legal; the oracle's consumable gate ignores the phase
            # and names default selections, the engine's names none: no conversions there.
            if snap["phase"] == "game_over":
                chain.push("conv", None)
            else:
                py = canon.to_json_compatible(_py_conversions(engine, snap, table))
                rs = canon.to_json_compatible(frozen.conversions(game, rs_table))
                if canon.digest(py) != canon.digest(rs):
                    raise Disagreement(f"{where}: conversions {canon.diff(py, rs)[:6]}")
                chain.push("conv", frozen.conversions_digest(rs))
        return snap


def golden_env() -> None:
    rec = EnvRecorder()
    out = {}
    for spec in load_specs():
        engine = PyEngine(spec.start, spec.setup)
        chain = frozen.Chain()
        for pos in range(len(spec.tape) + 1):
            where = f"{spec.name}@{pos}"
            snap = rec.state(engine, where, chain, frozen.table_seed(where))
            if snap["phase"] == "game_over" or snap["run"]["won"] or pos == len(spec.tape):
                break
            try:
                engine.step(spec.tape[pos])
            except Exception:
                break
        out[spec.name] = chain.as_json()
    frozen.write("golden_env.json", out)
    print(f"golden_env.json: {len(out)} tapes, {sum(v['records'] for v in out.values())} records")


def _env_run(
    rec: EnvRecorder,
    name: str,
    engine: PyEngine,
    steps: int,
    choose: Callable[[PyEngine, dict[str, Any]], Any],
    full: Callable[[int], bool],
) -> dict[str, Any]:
    """Play ``choose`` on the oracle, recording the env layer and the state at every state
    (conversions where ``full(pos)``) and checking each step in Rust, continued and imported.
    ``choose`` returns a tape action, a setup op (a dict, applied without a step) or None."""
    game = Game.from_dict(plain(engine.snapshot()))
    chain = frozen.Chain()
    tape: list[Any] = []
    out: dict[str, Any] = {}
    states = 0
    for pos in range(steps):
        where = f"{name}@{pos}"
        snap = rec.state(engine, where, chain, frozen.table_seed(where) if full(pos) else None)
        if game.state_hash() != _digest(engine):
            raise Disagreement(f"{where}: the continued Rust run differs")
        chain.push("state", frozen.short(game.state_hash()))
        states += 1
        if snap["phase"] == "game_over" or snap["run"]["won"]:
            break
        entry = choose(engine, snap)
        if entry is None:
            break
        if isinstance(entry, dict):
            engine.apply(entry)
            game.apply_setup(entry)
        elif _step_both(engine, entry, where):
            game.step(entry)
        else:
            out["refused"] = entry
            break
        tape.append(entry)
    out.update(tape=tape, states=states, chain=chain.as_json())
    return out


def env_fuzz() -> None:
    """The env tests' fuzz runs: ``tests/env/test_rust_action_space.py``'s (mask, table and
    conversions on every fourth state) and ``tests/env/test_rust_observation.py``'s, with
    their scenario boards."""
    from jackdaw.engine import actions as A
    from tests.golden.trajectories import _BACKS, _CHALLENGES, _mixed_policy
    from tests.parity.fuzz import _choose, random_case

    rec = EnvRecorder()
    action_space = []
    for chunk in range(16):
        rng = random.Random(f"ENV_FUZZ_{chunk}")
        for i in range(18):
            case = random_case(chunk * 18 + i, rng, max_steps=300)
            policy = random.Random(case.policy_seed)

            def choose(engine: PyEngine, snap: Any, policy=policy, smart=case.smart) -> Any:
                return _choose(engine, policy, smart)

            run = _env_run(
                rec, case.name, PyEngine(case.start, case.setup), case.max_steps, choose,
                lambda pos: pos % 4 == 0,
            )  # fmt: skip
            action_space.append(
                {"name": case.name, "start": case.start, "setup": case.setup, **run}
            )

    def mixed(seed: str, eps: float, skip_rate: float = 0.0, boss: str | None = None) -> Any:
        rng = random.Random(seed)
        pending = [boss]

        def choose(engine: PyEngine, snap: Any) -> Any:
            if pending[0] is not None and snap["phase"] == "blind_select":
                op = {"op": "force_boss", "key": pending[0]}
                pending[0] = None
                return op
            with engine._ids():
                legal = get_legal_actions(engine.gs)
                if not legal:
                    return None
                skips = [a for a in legal if isinstance(a, A.SkipBlind)]
                if skips and rng.random() < skip_rate:
                    return encode_action(skips[0])
                return encode_action(_mixed_policy(engine.gs, legal, rng, eps))

        return choose

    observation = []
    for i in range(24):
        challenge = _CHALLENGES[(i // 4) % len(_CHALLENGES)] if i % 4 == 3 else None
        start = {
            "back": "b_challenge" if challenge else _BACKS[i % len(_BACKS)],
            "stake": 1 + (i * 5) % 8,
            "seed": f"OBS_FUZZ{i}",
            "challenge": challenge,
        }
        choose = mixed(start["seed"], (0.0, 0.05, 0.2)[i % 3], (0.0, 0.2)[i % 2])
        run = _env_run(rec, start["seed"], PyEngine(start), 300, choose, lambda pos: False)
        observation.append({"name": start["seed"], "start": start, "setup": [], **run})

    from tests.env.test_rust_observation import _SCENARIOS

    scenarios = []
    for name in sorted(_SCENARIOS):
        setup, play = _SCENARIOS[name]
        for k, seed in enumerate(("OBSCEN1", "OBSCEN2")):
            start = {"back": "b_red", "stake": 1 + 3 * k, "seed": seed + name, "challenge": None}
            choose = mixed(f"{seed}{name}", **play)
            run = _env_run(
                rec, f"{seed}{name}", PyEngine(start, setup), 150, choose, lambda pos: False
            )
            scenarios.append({"name": f"{seed}{name}", "start": start, "setup": setup, **run})

    frozen.write(
        "env_fuzz.json",
        {"action_space": action_space, "observation": observation, "scenarios": scenarios},
    )
    n = sum(r["states"] for r in action_space + observation + scenarios)
    print(
        f"env_fuzz.json: {len(action_space) + len(observation) + len(scenarios)} runs, {n} states"
    )


def env_states() -> None:
    """The observation of edited and random snapshots (``tests/env/test_rust_observation.py``),
    encoded by the Python encoder from the oracle's state rebuilt from each snapshot."""
    from jackdaw.env.observation import encode_observation
    from tests.env.test_rust_observation import edited_snapshots, random_states, snapshot_digest

    def check(snap: dict[str, Any], where: str) -> str:
        engine = PyEngine.from_snapshot(snap)
        with engine._ids():
            py = frozen.obs_digest(encode_observation(engine.gs))
        if snapshot_digest(snap) != py:
            raise Disagreement(f"{where}: observation")
        return py

    edited = {
        phase: [check(snap, where) for where, snap in edited_snapshots(phase)]
        for phase in ("selecting_hand", "shop", "pack_opening")
    }
    rand = [check(snap, f"random#{n}") for n, snap in enumerate(random_states())]
    frozen.write("env_states.json", {"edited": edited, "random": rand})
    print(f"env_states.json: {sum(map(len, edited.values()))} edited, {len(rand)} random")


ENV_CASES = (("b_red", 1, 0.1), ("b_plasma", 3, 0.3), ("b_ghost", 5, 0.2), ("b_magic", 8, 0.5))


def env_runs() -> None:
    """``tests/env/test_rust_envs.py``'s runs: ``BalatroEnvironment`` and
    ``BalatroGymnasiumEnv`` on the oracle (``OracleAdapter``), stepped side by side with the
    same environments on the Rust engine. The policies read the oracle's state, so the actions
    are kept: factored actions for the environment, table indices for the Gym env."""
    from jackdaw.engine import card as card_module
    from jackdaw.env import BalatroEnvironment, DirectAdapter, OracleAdapter
    from jackdaw.env.action_space import engine_action_to_factored
    from tests.env.test_rust_envs import _gym_pair
    from tests.golden.trajectories import _gym_smart_index, _mixed_policy

    def same(a: list[Any], b: list[Any], where: str) -> list[Any]:
        if canon.digest(a) != canon.digest(b):
            raise Disagreement(f"{where}: {canon.diff(a, b)[:6]}")
        return a

    environment: dict[str, Any] = {}
    for back, stake, eps in ENV_CASES:
        kwargs = {"back_keys": [back], "stakes": [stake], "seed_prefix": f"ENVPAR_{back}_{stake}"}
        rust = BalatroEnvironment(adapter_factory=DirectAdapter, max_steps=300, **kwargs)
        oracle = BalatroEnvironment(adapter_factory=OracleAdapter, max_steps=300, **kwargs)
        policy = random.Random(f"{back}{stake}")
        episodes = []
        for episode in range(5):
            card_module.reset_sort_id_counter()
            chain = frozen.Chain()
            o_info = oracle.reset()[2]
            r_info = rust.reset()[2]
            where = f"{back} ep{episode}"
            rec = same(frozen.env_reset_record(o_info), frozen.env_reset_record(r_info), where)
            chain.push("reset", rec)
            tape = []
            for pos in range(300):
                gs = o_info["raw_state"]
                action = _mixed_policy(gs, oracle._adapter.get_legal_actions(), policy, eps)
                fa = engine_action_to_factored(action, gs)
                _, o_term, o_trunc, _, o_info = oracle.step(fa)
                _, r_term, r_trunc, _, r_info = rust.step(fa)
                rec = same(
                    frozen.env_step_record(oracle, o_term, o_trunc, o_info),
                    frozen.env_step_record(rust, r_term, r_trunc, r_info),
                    f"{where}@{pos}",
                )
                chain.push("step", rec)
                cards = None if fa.card_target is None else list(fa.card_target)
                tape.append([int(fa.action_type), fa.entity_target, cards])
                if o_term or o_trunc:
                    break
            episodes.append({"tape": tape, "chain": chain.as_json()})
        environment[f"{back}-{stake}-{eps}"] = episodes

    from tests.env.test_rust_envs import _GYM_CASES

    gym: dict[str, Any] = {}
    for case in sorted(_GYM_CASES):
        rust_gym, oracle_gym = _gym_pair(case, "b_red", 1 + len(case) % 4)
        policy_np = np.random.default_rng(len(case))
        episodes = []
        for episode in range(6):
            card_module.reset_sort_id_counter()
            seed = 9000 + episode if episode < 5 else None  # the last reset is unseeded
            chain = frozen.Chain()
            o_obs, o_info = oracle_gym.reset(seed=seed)
            r_obs, r_info = rust_gym.reset(seed=seed)
            where = f"{case} ep{episode}"
            rec = same(
                frozen.gym_record(oracle_gym, o_obs, o_info),
                frozen.gym_record(rust_gym, r_obs, r_info),
                where,
            )
            chain.push("reset", rec)
            tape = []
            for pos in range(250):
                if pos % 5:
                    action = _gym_smart_index(oracle_gym, policy_np)
                else:
                    action = int(policy_np.choice(np.flatnonzero(o_info["action_mask"])))
                o_obs, o_r, o_term, o_trunc, o_info = oracle_gym.step(action)
                r_obs, r_r, r_term, r_trunc, r_info = rust_gym.step(action)
                rec = same(
                    frozen.gym_record(oracle_gym, o_obs, o_info, o_r, o_term, o_trunc),
                    frozen.gym_record(rust_gym, r_obs, r_info, r_r, r_term, r_trunc),
                    f"{where}@{pos}",
                )
                chain.push("step", rec)
                tape.append(action)
                if o_term or o_trunc:
                    break
            episodes.append({"seed": seed, "tape": tape, "chain": chain.as_json()})
        gym[case] = episodes

    frozen.write("env_runs.json", {"environment": environment, "gym": gym})
    steps = sum(len(e["tape"]) for v in (*environment.values(), *gym.values()) for e in v)
    print(f"env_runs.json: {len(environment)} environment and {len(gym)} Gym cases, {steps} steps")


CASE_FILES = (
    "tests/parity/test_jokers_batch_b.py",
    "tests/parity/test_jokers_batch_c.py",
    "tests/parity/test_jokers_batch_d.py",
    "tests/parity/test_jokers_batch_e.py",
    "tests/parity/test_consumables_rust.py",
    "tests/parity/test_audit_flow.py",
    "tests/parity/test_audit_sweep.py",
    "tests/parity/test_audit_tags.py",
    "tests/parity/test_audit_vouchers.py",
)


class _CaseRecorder:
    """A pytest plugin recording every run the targeted parity tests build: for each step
    checked against Rust, the oracle's state before it (in full), the action, the state digest
    after it and the legal-action digests; for each legal-action comparison, the state."""

    def __init__(self) -> None:
        self.data: dict[str, list[list[dict[str, Any]]]] = {}
        self.test = ""

    @staticmethod
    def _legal(engine: PyEngine) -> str | None:
        if engine.snapshot()["phase"] == "game_over":
            return None
        with engine._ids():
            return frozen.legal_digest([encode_action(a) for a in get_legal_actions(engine.gs)])

    def _run(self, engine: PyEngine) -> list[dict[str, Any]]:
        return self.data[self.test][engine._case_run]

    def pytest_collection_finish(self, session: Any) -> None:
        recorder = self
        init = PyEngine.__init__

        def engine_init(engine: PyEngine, *args: Any, **kwargs: Any) -> None:
            engine._case_run = len(recorder.data[recorder.test])
            recorder.data[recorder.test].append([])
            init(engine, *args, **kwargs)

        PyEngine.__init__ = engine_init  # type: ignore[method-assign]
        for item in session.items:
            module = item.module
            if hasattr(module, "check_step") and not hasattr(module.check_step, "recording"):
                module.check_step = self._check_step(module.check_step)
            if hasattr(module, "legal_lists") and not hasattr(module.legal_lists, "recording"):
                module.legal_lists = self._legal_lists(module.legal_lists)

    def _check_step(self, check_step: Callable[..., bool]) -> Callable[..., bool]:
        def recording(engine: PyEngine, action: list[Any], where: str, report: Any) -> bool:
            pre = plain(engine.snapshot())
            pre_legal = self._legal(engine)
            ok = check_step(engine, action, where, report)
            if report.mismatches or report.unported:
                raise Disagreement(f"{self.test} {where}: {dict(report.mismatches)}")
            post = canon.digest(plain(engine.snapshot())) if ok else None
            self._run(engine).append(
                {
                    "action": json.loads(json.dumps(action)),
                    "pre_state": canon.to_json_compatible(pre),
                    "pre": frozen.short(canon.digest(pre)),
                    "post": frozen.short(post) if post else None,
                    "legal": [pre_legal, self._legal(engine) if ok else None],
                }
            )
            return ok

        recording.recording = True  # type: ignore[attr-defined]
        return recording

    def _legal_lists(self, legal_lists: Callable[..., Any]) -> Callable[..., Any]:
        def recording(engine: PyEngine) -> Any:
            py, rs = legal_lists(engine)
            snap = plain(engine.snapshot())
            self._run(engine).append(
                {
                    "legal_state": canon.to_json_compatible(snap),
                    "state": frozen.short(canon.digest(snap)),
                    "legal": frozen.legal_digest(py),
                }
            )
            return py, rs

        recording.recording = True  # type: ignore[attr-defined]
        return recording

    def pytest_runtest_setup(self, item: Any) -> None:
        self.test = item.nodeid
        self.data[item.nodeid] = []


def cases() -> None:
    """Record the targeted parity tests' runs (``tests/parity/cases.py``) into the file named
    by ``JACKDAW_CASES_RECORDING``; ``JACKDAW_BUILD_CASES=<that file>`` then builds the
    fixture from the rewritten tests."""
    import os

    import pytest

    out = os.environ["JACKDAW_CASES_RECORDING"]
    recorder = _CaseRecorder()
    code = pytest.main(["-q", "-p", "no:cacheprovider", *CASE_FILES], plugins=[recorder])
    if code != 0:
        raise Disagreement(f"the targeted parity tests failed ({code})")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recorder.data, f, separators=(",", ":"))
    steps = sum(len(run) for runs in recorder.data.values() for run in runs)
    print(f"recorded {len(recorder.data)} tests, {steps} checks into {out}")


FIXTURES: dict[str, Callable[[], None]] = {
    "joker_scenarios": joker_scenarios,
    "fuzz": fuzz,
    "new_runs": new_runs,
    "setup_boards": setup_boards,
    "golden_env": golden_env,
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
