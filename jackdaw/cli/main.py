"""Jackdaw CLI — top-level entry point.

Usage::

    jackdaw serve --backend sim --port 8080
    jackdaw play --seed ABCD1234 --deck RED --stake WHITE
    jackdaw validate crash --count 200
    jackdaw validate benchmark --count 1000
    jackdaw validate seed --seed TESTSEED --seeds 5
    jackdaw benchmark --count 1000
"""

from __future__ import annotations

import argparse
import secrets
import sys

from jackdaw.bridge.backend import DECK_FROM_BOT, STAKE_FROM_BOT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jackdaw",
        description="Headless Balatro simulator for reinforcement learning research",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- serve ---------------------------------------------------------------
    p_serve = sub.add_parser("serve", help="Start the JSON-RPC server")
    p_serve.add_argument(
        "--backend",
        choices=("sim", "live"),
        default="sim",
        help="Backend type (default: sim)",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    p_serve.add_argument(
        "--live-host",
        default="127.0.0.1",
        help="Balatrobot host for live backend (default: 127.0.0.1)",
    )
    p_serve.add_argument(
        "--live-port",
        type=int,
        default=12346,
        help="Balatrobot port for live backend (default: 12346)",
    )

    # -- play ----------------------------------------------------------------
    p_play = sub.add_parser("play", help="Run a single game with an agent")
    p_play.add_argument(
        "--seed",
        default=None,
        help="RNG seed (default: random 8-char hex)",
    )
    p_play.add_argument(
        "--deck",
        choices=sorted(DECK_FROM_BOT.keys()),
        default="RED",
        help="Deck back (default: RED)",
    )
    p_play.add_argument(
        "--stake",
        choices=sorted(STAKE_FROM_BOT.keys()),
        default="WHITE",
        help="Stake level (default: WHITE)",
    )

    # -- validate ------------------------------------------------------------
    p_validate = sub.add_parser("validate", help="Validation subcommands")
    vsub = p_validate.add_subparsers(dest="validate_command", required=True)

    # validate crash
    p_vcrash = vsub.add_parser("crash", help="Run N games, report crashes")
    p_vcrash.add_argument("--count", type=int, default=200, help="Number of runs (default: 200)")
    p_vcrash.add_argument(
        "--agent",
        choices=("random", "smart"),
        default="random",
        help="Agent type (default: random)",
    )

    # validate benchmark
    p_vbench = vsub.add_parser("benchmark", help="Pure performance measurement")
    p_vbench.add_argument("--count", type=int, default=1000, help="Number of runs (default: 1000)")

    # validate seed
    p_vseed = vsub.add_parser("seed", help="Compare sim vs live balatrobot side-by-side")
    p_vseed.add_argument("--seed", default="TESTSEED", help="Base seed (default: TESTSEED)")
    p_vseed.add_argument("--seeds", type=int, default=5, help="Number of seeds (default: 5)")
    p_vseed.add_argument("--back", default="b_red", help="Deck back key (default: b_red)")
    p_vseed.add_argument("--stake", type=int, default=1, help="Stake level 1-8 (default: 1)")
    p_vseed.add_argument("--host", default="127.0.0.1", help="Balatrobot host (default: 127.0.0.1)")
    p_vseed.add_argument("--port", type=int, default=12346, help="Balatrobot port (default: 12346)")
    p_vseed.add_argument(
        "--agent",
        choices=("default", "smart"),
        default="default",
        help="Agent strategy (default: basic validation agent, smart: economy+scoring aware)",
    )

    # -- benchmark (top-level alias) -----------------------------------------
    p_bench = sub.add_parser("benchmark", help="Alias for 'validate benchmark'")
    p_bench.add_argument("--count", type=int, default=1000, help="Number of runs (default: 1000)")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve default seed for play
    if args.command == "play" and args.seed is None:
        args.seed = secrets.token_hex(4).upper()

    # Dispatch
    if args.command == "serve":
        from jackdaw.cli.serve import run_server

        if args.backend == "live":
            from jackdaw.bridge.backend import LiveBackend

            backend = LiveBackend(host=args.live_host, port=args.live_port)
        else:
            from jackdaw.bridge.backend import SimBackend

            backend = SimBackend()
        run_server(backend, args.host, args.port)
        return

    if args.command == "play":
        from jackdaw.cli.play import run_play

        run_play(args.seed, args.deck, args.stake)
        return

    if args.command == "validate":
        from jackdaw.cli.validate import run_benchmark, run_crash, run_seed

        if args.validate_command == "crash":
            sys.exit(run_crash(args.count, args.agent))
        elif args.validate_command == "benchmark":
            run_benchmark(args.count)
        elif args.validate_command == "seed":
            sys.exit(
                run_seed(
                    args.seed,
                    args.seeds,
                    args.back,
                    args.stake,
                    args.host,
                    args.port,
                    agent=args.agent,
                )
            )
        return

    if args.command == "benchmark":
        from jackdaw.cli.validate import run_benchmark

        run_benchmark(args.count)
        return


if __name__ == "__main__":
    main()
