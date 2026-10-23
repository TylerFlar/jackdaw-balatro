"""Engine-agnostic replay of the recorded engine action tapes.

A :class:`ParitySpec` describes one trajectory declaratively: how to start
the run, the setup operations applied before play (the same setups the
golden v1 engine trajectories use), and the recorded action tape.  Any
engine that implements :class:`Engine` can replay it; ``PyEngine`` drives the
Python oracle, and the Rust engine gets an adapter with the same methods.

Setup operations (all fields are Lua-native names):

* ``{"op": "force_boss", "key": "bl_psychic"}`` - the ante's boss blind is
  ``key`` and the next blind to play is the boss.
* ``{"op": "action", "action": ["SelectBlind"]}`` - apply one action.
* ``{"op": "add", "key": "j_joker" | "H_A" | "c_fool", "seal": "Red",
  "enhancement": "m_steel", "edition": "foil"}`` - BalatroBot ``add``
  semantics (``SMODS.add_card``): edition poll, ``add_to_deck``, and playing
  cards join the hand in SELECTING_HAND.
* ``{"op": "set", "dollars": 60, "hands_left": 1, "chips": 123}`` - write
  state directly (BalatroBot ``set`` semantics; ``hands_left`` is the current
  round's hands).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tests.parity import canon

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SPECS_PATH = FIXTURES / "parity_specs.json"


@dataclass
class ParitySpec:
    name: str
    start: dict[str, Any]
    setup: list[dict[str, Any]]
    tape: list[list[Any]]
    max_steps: int
    check: str | None = None
    check_after: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "setup": self.setup,
            "tape": self.tape,
            "max_steps": self.max_steps,
            "check": self.check,
            "check_after": self.check_after,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ParitySpec:
        return cls(**d)


def load_specs(path: Path = SPECS_PATH) -> list[ParitySpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ParitySpec.from_json(d) for d in data["specs"]]


class Engine(Protocol):
    """What the replay needs from an engine."""

    def snapshot(self) -> dict[str, Any]: ...

    def legal_actions(self) -> list[list[Any]]: ...

    def step(self, action: list[Any]) -> None: ...  # raises on an illegal action


# ---------------------------------------------------------------------------
# Scenario checks, evaluated on snapshots (engine-agnostic)
# ---------------------------------------------------------------------------


def _jokers(snap: dict[str, Any]) -> list[dict[str, Any]]:
    return snap["areas"]["jokers"]


def _score_positive(snap: dict[str, Any]) -> bool:
    last = snap["sim"]["last_score"]
    return last is not None and last["total"] > 0


def _bones_saved(snap: dict[str, Any]) -> bool:
    return snap["phase"] != "game_over" and not any(
        j["center"] == "j_mr_bones" for j in _jokers(snap)
    )


def _debuffed_and_saved(snap: dict[str, Any]) -> bool:
    last = snap["sim"]["last_score"]
    return last is not None and last["debuffed"] and _bones_saved(snap)


def _six_destroyed(snap: dict[str, Any]) -> bool:
    from jackdaw.engine.data.prototypes import SPECTRALS

    return any(c["center"] in SPECTRALS for c in snap["areas"]["consumables"])


CHECKS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "score_positive": _score_positive,
    "bones_saved": _bones_saved,
    "debuffed_and_saved": _debuffed_and_saved,
    "six_destroyed": _six_destroyed,
    "two_jokers": lambda snap: len(_jokers(snap)) >= 2,
}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 10


@dataclass
class Replay:
    digest: str = ""
    records: int = 0
    checkpoints: list[str] = field(default_factory=list)
    actions_taken: int = 0
    checks: dict[str, bool] = field(default_factory=dict)
    record_log: list[tuple[str, Any]] = field(default_factory=list)
    keep_log: bool = False

    def push(self, kind: str, payload: Any) -> None:
        self.digest = canon.chain(self.digest, canon.digest([kind, payload]))
        self.records += 1
        if self.records % CHECKPOINT_EVERY == 0:
            self.checkpoints.append(self.digest[:16])
        if self.keep_log:
            self.record_log.append((kind, payload))

    def as_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "records": self.records,
            "actions": self.actions_taken,
            "checkpoints": self.checkpoints,
            "checks": self.checks,
        }


def replay(
    spec: ParitySpec,
    engine: Engine,
    *,
    keep_log: bool = False,
    on_state: Callable[[int, Engine, dict[str, Any]], None] | None = None,
) -> Replay:
    """Replay ``spec`` on an engine that has already run the spec's setup."""
    res = Replay(keep_log=keep_log)
    res.push("setup", engine.snapshot())
    checked = spec.check is None
    pos = 0
    for _ in range(spec.max_steps):
        snap = engine.snapshot()
        if not checked and res.actions_taken >= spec.check_after:
            res.checks["scenario"] = CHECKS[spec.check](snap)  # type: ignore[index]
            checked = True
        legal = engine.legal_actions()
        res.push("state", {"snapshot": snap, "legal": legal})
        if on_state is not None:
            on_state(pos, engine, snap)
        if snap["phase"] == "game_over" or snap["run"]["won"] or not legal:
            break
        if pos >= len(spec.tape):
            res.push("tape-exhausted", None)
            break
        action = spec.tape[pos]
        pos += 1
        try:
            engine.step(action)
        except Exception:  # behaviour includes refusing an action
            res.push("raised", action)
            break
        res.actions_taken += 1
        res.push("action", action)
    final = engine.snapshot()
    if not checked:
        res.checks["scenario"] = CHECKS[spec.check](final)  # type: ignore[index]
    res.push("final", final)
    return res


# ---------------------------------------------------------------------------
# Spec generation from the golden v1 engine trajectories
# ---------------------------------------------------------------------------

# Scenario setups of tests/golden/trajectories.py, as setup operations.
# ``last_hand`` is resolved at generation time (it depends on the blind's
# chip target, which is deterministic for the spec).
_SCENARIO_OPS: dict[str, list[dict[str, Any]]] = {
    "copy_retriggers": [
        *({"op": "add", "key": k} for k in
          ("j_blueprint", "j_hack", "j_brainstorm", "j_mime", "j_baseball")),
        {"op": "add", "key": "H_2", "seal": "Red"},
        {"op": "add", "key": "S_3"},
        {"op": "add", "key": "D_4"},
        {"op": "add", "key": "C_5"},
        {"op": "add", "key": "C_K", "enhancement": "m_steel"},
        {"op": "add", "key": "H_K", "enhancement": "m_steel", "seal": "Red"},
    ],
    "face_retriggers": [
        *({"op": "add", "key": k} for k in
          ("j_sock_and_buskin", "j_hanging_chad", "j_blueprint", "j_selzer", "j_dusk")),
        {"op": "add", "key": "H_K"},
        {"op": "add", "key": "H_Q", "seal": "Red"},
        {"op": "add", "key": "H_J"},
        {"op": "set", "hands_left": 1},
    ],
    "bones_save": [
        {"op": "add", "key": "j_mr_bones"},
        {"op": "add", "key": "j_joker"},
        {"op": "add", "key": "S_2"},
        {"op": "last_hand", "chips_fraction": 0.3},
    ],
    "psychic_bones": [
        *({"op": "add", "key": k} for k in ("j_matador", "j_mr_bones", "j_ice_cream")),
        {"op": "add", "key": "S_2"},
        {"op": "last_hand", "chips_fraction": 0.3},
    ],
    "sixth_sense": [
        *({"op": "add", "key": k} for k in ("j_sixth_sense", "j_caino", "j_glass", "j_hiker")),
        {"op": "add", "key": "H_6"},
    ],
    "space_dna": [
        *({"op": "add", "key": k} for k in
          ("j_oops", "j_space", "j_dna", "j_obelisk", "j_reserved_parking")),
        {"op": "add", "key": "S_9"},
        {"op": "add", "key": "H_K"},
        {"op": "add", "key": "S_Q"},
    ],
    "creation_gates": [
        {"op": "add", "key": "j_riff_raff"},
        {"op": "add", "key": "c_judgement"},
        {"op": "add", "key": "c_wraith"},
        {"op": "add", "key": "H_7", "enhancement": "m_glass"},
        {"op": "add", "key": "S_7", "enhancement": "m_steel"},
        {"op": "add", "key": "D_7", "enhancement": "m_lucky"},
        {"op": "add", "key": "C_7", "enhancement": "m_gold"},
        {"op": "set", "dollars": 80},
    ],
}  # fmt: skip

_SCENARIO_CHECKS: dict[str, str | None] = {
    "copy_retriggers": "score_positive",
    "face_retriggers": None,
    "bones_save": "bones_saved",
    "psychic_bones": "debuffed_and_saved",
    "sixth_sense": "six_destroyed",
    "space_dna": "score_positive",
    "creation_gates": "two_jokers",
}


def specs_from_golden_v1() -> list[ParitySpec]:
    """Parity specs for every golden v1 engine trajectory (resolved setups)."""
    from jackdaw.engine.data.prototypes import JOKERS
    from tests.golden.trajectories import SCENARIOS, engine_specs

    tapes = json.loads((FIXTURES / "golden_tapes.json").read_text(encoding="utf-8"))
    out: list[ParitySpec] = []
    for es in engine_specs():
        scenario = SCENARIOS[es.scenario] if es.scenario else None
        setup: list[dict[str, Any]] = []
        boss = es.boss or (scenario.boss if scenario else None)
        if boss:
            setup.append({"op": "force_boss", "key": boss})
        if es.jokers or scenario:
            setup.append({"op": "action", "action": ["SelectBlind"]})
        if es.jokers:
            setup_rng = random.Random(es.seed + "/setup")
            setup += [{"op": "add", "key": k} for k in setup_rng.sample(sorted(JOKERS), es.jokers)]
        if scenario:
            setup += [dict(op) for op in _SCENARIO_OPS[es.scenario]]  # type: ignore[index]
        if es.dollars is not None:
            setup.append({"op": "set", "dollars": es.dollars})
        out.append(
            ParitySpec(
                name=es.name,
                start={"back": es.back, "stake": es.stake, "seed": es.seed,
                       "challenge": es.challenge},
                setup=setup,
                tape=tapes[es.name],
                max_steps=es.max_steps,
                check=_SCENARIO_CHECKS[es.scenario] if es.scenario else None,
                check_after=len(scenario.script) if scenario else 0,
            )
        )  # fmt: skip
    return out


def resolve_last_hand(specs: list[ParitySpec]) -> list[ParitySpec]:
    """Replace ``last_hand`` ops with literal ``set`` ops, using the Python oracle."""
    from tests.parity.pyengine import PyEngine

    for spec in specs:
        if not any(op["op"] == "last_hand" for op in spec.setup):
            continue
        resolved: list[dict[str, Any]] = []
        eng = PyEngine(spec.start)
        for op in spec.setup:
            if op["op"] == "last_hand":
                chips = int(eng.gs["blind"].chips * op["chips_fraction"])
                op = {"op": "set", "hands_left": 1, "chips": chips}
            eng.apply(op)
            resolved.append(op)
        spec.setup = resolved
    return specs


def write_specs(specs: list[ParitySpec], path: Path = SPECS_PATH) -> None:
    data = {"schema": 1, "specs": [s.to_json() for s in specs]}
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
