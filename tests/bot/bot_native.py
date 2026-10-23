"""Bot path against native path on the engine tapes.

For every recorded tape (``tests/fixtures/parity_specs.json`` and the fuzz
regressions), each action is taken twice from the same state:

* natively, ``Game.step(action)``;
* through BalatroBot's API, the request ``translate`` gives for it, sent to a
  ``BotSession`` resting in the same run.

The two results must have the same canonical snapshot.  An action with no
request (RerollBoss, BuyAndUse) or one the bot refuses (vanilla allows more,
``docs/balatrobot.md``) is taken natively and the session restarts from the
native state; so does ``SortHand``, whose ``rearrange`` permutation sorts the
hand but leaves ``G.hand.config.sort`` alone (the only field allowed to
differ).  Run as a script for the report::

    .venv/Scripts/python.exe -m tests.bot.bot_native
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from jackdaw import _core
from tests.parity import canon
from tests.parity.harness import FIXTURES, load_specs
from tests.parity.rust_setup import rust_setup


def _snap(text: str) -> dict[str, Any]:
    return canon.from_json_compatible(json.loads(text))


@dataclass
class Report:
    tapes: int = 0
    steps: int = 0  # actions taken
    via_bot: int = 0  # ... through the bot's API, matching the native result
    fallback: Counter[str] = field(default_factory=Counter)  # reason -> count
    sort_mode: int = 0  # SortHand steps that differ only in hand_sort
    mismatches: list[str] = field(default_factory=list)
    ended: Counter[str] = field(default_factory=Counter)  # why tapes stopped early


def _reason(error: dict[str, Any]) -> str:
    """A refusal grouped by its message (numbers and quoted names dropped)."""
    message = re.sub(r"'[^']*'", "'_'", error["message"])
    return f"{error['name']}: {re.sub(r'-?\d+', 'N', message)}"


def compare_tape(name: str, spec: dict[str, Any], report: Report) -> None:
    native = rust_setup(spec["start"], spec["setup"])
    session = _core.BotSession.from_game(native)
    report.tapes += 1
    for pos, action in enumerate(spec["tape"]):
        if native.phase == "game_over":
            break
        after = native.clone()
        try:
            after.step(json.dumps(action))
        except (_core.IllegalActionError, _core.NotPortedError) as exc:
            report.ended[type(exc).__name__] += 1
            return
        report.steps += 1
        request = session.translate(json.dumps(action))
        if request is None:
            report.fallback[f"sim-only {action[0]}"] += 1
            session = _core.BotSession.from_game(after)
        else:
            method, params = request
            ok, text = session.check(method, params)
            if not ok:
                report.fallback[f"{action[0]} -> {method}: {_reason(json.loads(text))}"] += 1
                session = _core.BotSession.from_game(after)
            else:
                ok, text = session.call(method, params)
                if not ok:
                    report.mismatches.append(f"{name}@{pos} {action}: check passed, call {text}")
                    return
                bot = session.game()
                if bot.state_hash() != after.state_hash():
                    diff = canon.diff(_snap(after.snapshot()), _snap(bot.snapshot()))
                    if action[0] == "SortHand" and diff and all("hand_sort" in d for d in diff):
                        report.sort_mode += 1
                        session = _core.BotSession.from_game(after)
                    else:
                        where = f"{name}@{pos} {action} ({method} {params})"
                        report.mismatches.append(f"{where}: " + "; ".join(diff[:6]))
                        return
                else:
                    report.via_bot += 1
        native = after


def all_tapes() -> list[tuple[str, dict[str, Any]]]:
    tapes = [(s.name, s.to_json()) for s in load_specs()]
    regressions = json.loads((FIXTURES / "fuzz_regressions.json").read_text(encoding="utf-8"))
    tapes += [(c["name"], c) for c in regressions["cases"]]
    return tapes


def run_all() -> Report:
    report = Report()
    for name, spec in all_tapes():
        compare_tape(name, spec, report)
    return report


def main() -> None:
    r = run_all()
    print(f"tapes {r.tapes}, steps {r.steps}, via bot {r.via_bot}, sort mode {r.sort_mode}")
    print(f"fallbacks {sum(r.fallback.values())}:")
    for reason, n in r.fallback.most_common():
        print(f"  {n:5}  {reason}")
    print("ended early:", dict(r.ended))
    for m in r.mismatches:
        print("MISMATCH", m)


if __name__ == "__main__":
    main()
