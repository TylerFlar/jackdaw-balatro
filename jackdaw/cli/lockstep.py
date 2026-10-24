"""``jackdaw lockstep``: organic games against BalatroBot, every step checked
against a shadow run; ``jackdaw lockstep replay`` reruns a divergence bundle.

Usage::

    jackdaw lockstep --seeds 50                                # one instance on :12346
    jackdaw lockstep --instances 4 --seeds 200 --decks RED,BLUE --stakes WHITE,GOLD
    jackdaw lockstep --url http://127.0.0.1:12346 --url http://127.0.0.1:12350
    jackdaw lockstep --url sim --seeds 20                      # self-check, no game needed
    jackdaw lockstep replay lockstep-out/<run>/bundles/<bundle>

See ``docs/live.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_PORT = 12346


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``lockstep`` command's options, and its ``replay`` subcommand."""
    parser.add_argument(
        "--instances",
        type=int,
        default=None,
        help="BalatroBot instances to play on in parallel (default: one per --url, else 1); "
        "without --url, instance i is http://HOST:(12346+i)",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="A BalatroBot URL (repeat for more instances); 'sim' plays against the "
        "in-process simulator",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host of the default URLs")
    parser.add_argument(
        "--seeds",
        default="10",
        help="A number of random seeds, a comma-separated list, or @FILE (one per line)",
    )
    parser.add_argument("--decks", default="RED", help="Decks to cycle through (RED,BLUE,...)")
    parser.add_argument("--stakes", default="WHITE", help="Stakes to cycle through (WHITE,...)")
    parser.add_argument("--steps", type=int, default=1000, help="Actions per game at most")
    parser.add_argument(
        "--policy",
        choices=("smart", "random"),
        default="smart",
        help="smart (default): plays to get deep, with some random moves; random: a random "
        "kind of move each step",
    )
    parser.add_argument("--rng-seed", type=int, default=0, help="Seed for generated run seeds")
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for the bundles and summary.json (default: lockstep-out/<time>)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Seconds to wait for each live response"
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.0,
        help="Seconds to wait after each live answer and compare a fresh gamestate "
        "instead (for answers sent before the step's events have all run)",
    )
    parser.add_argument(
        "--recheck",
        type=float,
        default=None,
        help="Seconds to keep reading the gamestate again while a live answer differs from "
        "the shadow (default: 2 against a live game, 0 against jackdaw serve)",
    )
    parser.add_argument("--endless", action="store_true", help="Keep playing after a win")
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop at the first unclassified divergence"
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    sub = parser.add_subparsers(dest="lockstep_command")
    replay = sub.add_parser("replay", help="Rerun a divergence bundle on the simulator alone")
    replay.add_argument("bundle", help="A bundle directory (or its bundle.json)")
    replay.add_argument("--json", action="store_true", help="Print the report as JSON")


def urls_for(instances: int | None, urls: list[str] | None, host: str) -> list[str]:
    """One URL per instance: the given ones, else ``http://host:12346+i``."""
    if urls:
        if instances is not None and instances != len(urls):
            raise SystemExit(f"--instances {instances} but {len(urls)} --url given")
        return list(urls)
    return [f"http://{host}:{DEFAULT_PORT + i}" for i in range(instances or 1)]


def backend_for(url: str, timeout: float) -> Any:
    from jackdaw.bridge.backend import LiveBackend, SimBackend

    if url in ("sim", "sim://"):
        return SimBackend()
    return LiveBackend(url=url, timeout=timeout)


def parse_seeds(spec: str) -> list[str] | int:
    """``--seeds``: a count, a comma-separated list, or ``@file``."""
    spec = spec.strip()
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text(encoding="utf-8").splitlines()
        return [s.strip() for s in lines if s.strip() and not s.lstrip().startswith("#")]
    if spec.isdigit():
        return int(spec)
    return [s.strip() for s in spec.split(",") if s.strip()]


def _names(spec: str) -> list[str]:
    return [s.strip().upper() for s in spec.split(",") if s.strip()]


def run(args: argparse.Namespace) -> int:
    from jackdaw.bridge.backend import DECK_FROM_BOT, STAKE_FROM_BOT
    from jackdaw.bridge.lockstep import make_jobs, run_lockstep

    decks, stakes = _names(args.decks), _names(args.stakes)
    for name in decks:
        if name not in DECK_FROM_BOT:
            raise SystemExit(f"unknown deck {name!r} (one of {', '.join(DECK_FROM_BOT)})")
    for name in stakes:
        if name not in STAKE_FROM_BOT:
            raise SystemExit(f"unknown stake {name!r} (one of {', '.join(STAKE_FROM_BOT)})")
    urls = urls_for(args.instances, args.url, args.host)
    backends = [backend_for(u, args.timeout) for u in urls]
    for url, backend in zip(urls, backends, strict=True):
        try:
            backend.handle("health", {})
        except Exception as exc:  # noqa: BLE001 - any reason the server can't be reached
            print(f"jackdaw lockstep: cannot reach BalatroBot at {url}: {exc}", file=sys.stderr)
            return 2
    jobs = make_jobs(parse_seeds(args.seeds), decks, stakes, rng_seed=args.rng_seed)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("lockstep-out") / stamp
    out.mkdir(parents=True, exist_ok=True)
    if not args.json:
        print(
            f"jackdaw lockstep: {len(jobs)} game(s), {args.policy} policy, up to {args.steps} "
            f"steps each, on {len(backends)} instance(s): {', '.join(urls)}"
        )
        print(f"bundles and summary in {out}")
        sys.stdout.flush()

    def on_result(index: int, result: Any) -> None:
        if not args.json:
            print(f"[{index}] {result.line()}")
            sys.stdout.flush()

    summary = run_lockstep(
        backends,
        jobs,
        policy=args.policy,
        max_steps=args.steps,
        bundle_dir=out / "bundles",
        endless=args.endless,
        on_result=on_result,
        stop_on_divergence=args.fail_fast,
        settle=args.settle,
        recheck=args.recheck,
    )
    totals = summary.totals()
    report = {
        "urls": urls,
        "policy": args.policy,
        "steps": args.steps,
        "decks": decks,
        "stakes": stakes,
        **totals,
        "games_detail": [r.to_json() for r in summary.results],
    }
    (out / "summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "games_detail"}, indent=1))
    else:
        print()
        print("\n".join(summary.lines()))
    return 1 if summary.unclassified else 0


def replay(args: argparse.Namespace) -> int:
    from jackdaw.bridge.live import replay_bundle

    try:
        report = replay_bundle(args.bundle)
    except (OSError, ValueError, KeyError) as exc:
        print(f"jackdaw lockstep replay: {exc}", file=sys.stderr)
        return 2
    if args.json:
        c = report.classification
        print(
            json.dumps(
                {
                    "bundle": str(report.bundle),
                    "replayed": report.replayed,
                    "error": report.error,
                    "shadow_matches": report.shadow_matches,
                    "refused": report.refused,
                    "reproduced": report.reproduced,
                    "sim_changed": [d.to_json() for d in report.sim_changed],
                    "unclassified": [] if c is None else [d.to_json() for d in c.unclassified],
                    "explained": {}
                    if c is None
                    else {k: len(v) for k, v in {**c.classified, **c.reconcile}.items()},
                },
                indent=1,
            )
        )
    else:
        print("\n".join(report.lines()))
    return report.exit_code()


def main(args: argparse.Namespace) -> int:
    if getattr(args, "lockstep_command", None) == "replay":
        return replay(args)
    return run(args)
