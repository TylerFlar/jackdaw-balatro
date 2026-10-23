"""Scenario-based validation — the simulator against live Balatro, in lockstep.

Usage::

    jackdaw validate                          # run all scenarios
    jackdaw validate --category jokers        # run only joker scenarios
    jackdaw validate --scenario joker_joker   # run one scenario
    jackdaw validate --url http://127.0.0.1:12347 --bundles validate-bundles/
    jackdaw validate --sim                    # self-check against the in-process simulator

Each scenario plays on a :class:`~jackdaw.bridge.live.LiveAdapter`: every
request goes to a shadow run first and then to the live game, and the live
view is compared with the shadow's after each one. A scenario passes when the
two agree throughout (differences the deviation catalog explains aside); a
divergence fails it and writes a repro bundle. See ``docs/live.md``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _run_one(
    scenario: Any, backend: Any, *, delay: float, bundle_dir: Any, settle: float = 0.0
) -> Any:
    from jackdaw.bridge.backend import RPCError
    from jackdaw.bridge.live import DivergenceError, LiveAdapter, VetoedActionError
    from jackdaw.cli.scenarios import ScenarioResult
    from jackdaw.core import IllegalActionError, NotPortedError

    adapter = LiveAdapter(backend, bundle_dir=bundle_dir, label=scenario.name, settle=settle)
    try:
        result = scenario.run(adapter, delay=delay)
    except DivergenceError as exc:
        diffs = [str(d) for d in exc.diffs]
        for dev_id, ds in exc.classified.items():
            diffs += [f"[{dev_id}] {d}" for d in ds]
        result = ScenarioResult(
            passed=False,
            diffs=diffs or [str(exc)],
            details=f"divergence ({exc.kind}) at step {exc.step}",
            bundle="" if exc.bundle_path is None else str(exc.bundle_path),
        )
    except VetoedActionError as exc:
        result = ScenarioResult(
            passed=True, skipped=True, details=f"SKIP (vetoed by {exc.deviation}: {exc.reason})"
        )
    except RPCError as exc:
        result = ScenarioResult(
            passed=False,
            diffs=[f"the simulator refused a request: {exc.message} ({exc.data.get('name')})"],
            details="refused",
        )
    except (IllegalActionError, NotPortedError) as exc:
        result = ScenarioResult(passed=False, diffs=[str(exc)], details=type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - one scenario's failure must not stop the run
        result = ScenarioResult(
            passed=False, diffs=[f"{type(exc).__name__}: {exc}"], details=f"Error: {exc}"
        )
    result.explained = dict(adapter.explained)
    return result


def _make_backend(url: str | None, host: str, port: int, timeout: float, sim: bool) -> Any:
    from jackdaw.bridge.backend import LiveBackend, SimBackend

    if sim:
        return SimBackend()
    if url:
        return LiveBackend(url=url, timeout=timeout)
    return LiveBackend(host=host, port=port, timeout=timeout)


def run_validate(
    category: str | None = None,
    scenario: str | None = None,
    host: str = "127.0.0.1",
    port: int = 12346,
    delay: float = 0.3,
    *,
    url: str | None = None,
    bundle_dir: str | None = None,
    timeout: float = 30.0,
    sim: bool = False,
    settle: float = 0.0,
) -> int:
    """Run validation scenarios against a BalatroBot instance (or, with ``sim``, the
    in-process simulator as a self-check). Returns 0 if no scenario fails."""
    from jackdaw.cli.scenarios import ScenarioResult, get_scenarios

    backend = _make_backend(url, host, port, timeout, sim)
    where = "the in-process simulator" if sim else getattr(backend, "url", "")
    try:
        backend.handle("health", None)
    except Exception as e:  # noqa: BLE001 - any reason the server can't be reached
        print(f"Cannot reach BalatroBot at {where}: {e}")
        print("Start the game with BalatroBot (docs/live.md), or `jackdaw serve` for a dry run.")
        return 1

    # Collect scenarios
    scenarios = get_scenarios(category=category, name=scenario)
    if not scenarios:
        if scenario:
            print(f"No scenario found with name: {scenario}")
        elif category:
            print(f"No scenarios found in category: {category}")
        else:
            print("No scenarios registered")
        return 1

    print(f"Running {len(scenarios)} scenario(s) in lockstep against {where}...")
    if category:
        print(f"  Category: {category}")
    if scenario:
        print(f"  Scenario: {scenario}")
    if bundle_dir:
        print(f"  Bundles: {bundle_dir}")
    print()

    # Run scenarios
    results: list[tuple[str, str, ScenarioResult]] = []
    passed = failed = skipped = 0
    explained: Counter[str] = Counter()

    for s in scenarios:
        print(f"  [{s.category}] {s.name}: {s.description}")
        result = _run_one(s, backend, delay=delay, bundle_dir=bundle_dir, settle=settle)
        results.append((s.name, s.category, result))
        explained.update(result.explained)

        rows = result.sub_results or [("", result)]
        for sub_name, r in rows:
            tag = f"[{sub_name}] " if sub_name else ""
            if r.skipped or (r.passed and "SKIP" in r.details):
                skipped += 1
                print(f"    {tag}SKIP  {r.details}")
            elif r.passed:
                passed += 1
                note = ""
                if result.explained and not sub_name:
                    note = (
                        "  (explained: "
                        + ", ".join(f"{k} x{v}" for k, v in sorted(result.explained.items()))
                        + ")"
                    )
                print(f"    {tag}PASS{note}")
            else:
                failed += 1
                print(f"    {tag}FAIL  {r.details}")
                for d in r.diffs:
                    print(f"      {d}")
                if r.bundle:
                    print(f"      bundle: {r.bundle}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Group by category — expand sub_results for per-item reporting
    categories: dict[str, list[tuple[str, ScenarioResult]]] = {}
    for name, cat, result in results:
        if result.sub_results:
            for sub_name, sub_result in result.sub_results:
                categories.setdefault(cat, []).append((sub_name, sub_result))
        else:
            categories.setdefault(cat, []).append((name, result))

    for cat, cat_results in sorted(categories.items()):
        cat_passed = sum(1 for _, r in cat_results if r.passed)
        cat_total = len(cat_results)
        status = "PASS" if cat_passed == cat_total else "FAIL"
        print(f"\n  {cat}: {status} ({cat_passed}/{cat_total})")
        for name, r in cat_results:
            mark = "-" if r.skipped or "SKIP" in r.details else "+" if r.passed else "X"
            print(f"    [{mark}] {name}")

    if explained:
        print(
            "\nExplained by the deviation catalog: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(explained.items()))
        )
    print(f"\nTotal: {passed}/{passed + failed} passed, {skipped} skipped")
    return 0 if failed == 0 else 1
