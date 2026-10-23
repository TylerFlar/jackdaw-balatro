"""Jackdaw CLI — top-level entry point.

Usage::

    jackdaw validate                          # run all scenarios
    jackdaw validate --category jokers        # run only joker scenarios
    jackdaw validate --scenario joker_joker   # run one scenario
    jackdaw validate --host 127.0.0.1 --port 12346
    jackdaw serve                             # BalatroBot's API on the simulator
    jackdaw serve --port 12346 --fs-root runs/
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jackdaw",
        description="Headless Balatro simulator for reinforcement learning research",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- validate ------------------------------------------------------------
    p_validate = sub.add_parser(
        "validate",
        help="Validate engine against live Balatro via scenario-based testing",
    )
    p_validate.add_argument(
        "--category",
        choices=("jokers", "tarots", "planets", "spectrals", "boss_blinds", "modifiers", "tags"),
        default=None,
        help="Run only scenarios in this category",
    )
    p_validate.add_argument(
        "--scenario",
        default=None,
        help="Run a single scenario by name",
    )
    p_validate.add_argument(
        "--host",
        default="127.0.0.1",
        help="Balatrobot host (default: 127.0.0.1)",
    )
    p_validate.add_argument(
        "--port",
        type=int,
        default=12346,
        help="Balatrobot port (default: 12346)",
    )
    p_validate.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between actions for animation watching (default: 0.3)",
    )

    # -- serve ---------------------------------------------------------------
    p_serve = sub.add_parser(
        "serve",
        help="Serve BalatroBot v1.5.2's JSON-RPC API on the simulator",
        description=(
            "Serve BalatroBot v1.5.2's JSON-RPC 2.0 API over HTTP, backed by the "
            "simulator, so any BalatroBot client can drive it. See docs/balatrobot.md."
        ),
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument(
        "--port", type=int, default=12346, help="Bind port (default: 12346; 0 picks one)"
    )
    p_serve.add_argument(
        "--fs-root",
        default=None,
        help="Directory that save/load paths must stay in (required off loopback)",
    )
    p_serve.add_argument(
        "--debug",
        action="store_true",
        help="Also register BalatroBot's test endpoints (test_endpoint, test_error, ...)",
    )

    return parser


def run_serve(host: str, port: int, fs_root: str | None, debug: bool) -> int:
    """Serve until interrupted; 0 on Ctrl+C, 1 if the server can't start."""
    from jackdaw import _core

    def ready(bound: int) -> None:
        print(f"jackdaw serve: BalatroBot {_core.BALATROBOT_VERSION} API on http://{host}:{bound}")
        sys.stdout.flush()

    try:
        _core.serve(host, port, fs_root, debug, ready)
    except KeyboardInterrupt:
        return 0
    except ValueError as exc:
        print(f"jackdaw serve: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        from jackdaw.cli.validate import run_validate

        sys.exit(
            run_validate(
                category=args.category,
                scenario=args.scenario,
                host=args.host,
                port=args.port,
                delay=args.delay,
            )
        )
    elif args.command == "serve":
        sys.exit(run_serve(args.host, args.port, args.fs_root, args.debug))


if __name__ == "__main__":
    main()
