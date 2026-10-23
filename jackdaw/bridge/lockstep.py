"""Organic lockstep games against BalatroBot, at scale.

:func:`play_game` plays one run through a :class:`~jackdaw.bridge.live.LiveAdapter`
with a policy that picks among the shadow's legal actions;
:func:`run_lockstep` plays many, one worker per BalatroBot instance, and
collects the divergence bundles. ``jackdaw lockstep`` is the command line for
both (``docs/live.md``).

Two policies:

- :class:`RandomPolicy` picks an action kind at random (weighted so a run
  moves on), then a random instance of it: random card selections, random
  consumable targets. Short runs that touch every kind of move.
- :class:`SmartPolicy` plays to get deep: it tries every play on a copy of the
  shadow and keeps the best, discards when that can't win the round, buys
  jokers, planets, vouchers and packs, and now and then takes a random move
  so rearranges, sells, rerolls and skips still happen.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from jackdaw.bridge.live import DivergenceError, LiveAdapter, describe_backend
from jackdaw.core import Game, IllegalActionError, NotPortedError, random_seed
from jackdaw.engine.actions import (
    Discard,
    GamePhase,
    PickPackCard,
    PlayHand,
    UseConsumable,
)

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

#: How often the random policy picks each kind of move (relative weights).
RANDOM_WEIGHTS: dict[str, float] = {
    "SelectBlind": 4.0,
    "SkipBlind": 1.0,
    "PlayHand": 5.0,
    "Discard": 2.0,
    "CashOut": 5.0,
    "NextRound": 1.5,
    "Reroll": 0.6,
    "BuyCard": 2.5,
    "RedeemVoucher": 1.5,
    "OpenBooster": 2.0,
    "SellCard": 0.6,
    "UseConsumable": 2.5,
    "PickPackCard": 4.0,
    "SkipPack": 1.0,
    "SortHand": 0.4,
    "SwapHandLeft": 0.2,
    "SwapHandRight": 0.2,
    "SwapJokersLeft": 0.3,
    "SwapJokersRight": 0.3,
}


def _cards(view: dict[str, Any], area: str) -> list[dict[str, Any]]:
    a = view.get(area)
    return list(a.get("cards") or []) if isinstance(a, dict) else []


def _forced(view: dict[str, Any]) -> int | None:
    """Cerulean Bell's forced card (the hand card highlighted at rest)."""
    for i, c in enumerate(_cards(view, "hand")):
        state = c.get("state") or {}
        if isinstance(state, dict) and state.get("highlight"):
            return i
    return None


def _group(legal: Iterable[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for a in legal:
        out[type(a).__name__].append(a)
    return out


def _with_forced(cards: list[int], forced: int | None, limit: int = 5) -> tuple[int, ...]:
    if forced is not None and forced not in cards:
        cards = cards[: limit - 1] + [forced]
    return tuple(sorted(set(cards)))


def best_play(
    game: Game, view: dict, rng: random.Random, limit: int = 400
) -> tuple[tuple[int, ...], float, int] | None:
    """``(cards, chips gained, outcome)`` of the best play, found by taking each
    candidate on a copy of the run; outcome 2 wins the round, 1 doesn't, 0 loses."""
    n = len(_cards(view, "hand"))
    if n == 0:
        return None
    forced = _forced(view)
    others = [i for i in range(n) if i != forced]
    cands: list[tuple[int, ...]] = []
    for k in range(1, min(5, n) + 1):
        if forced is None:
            cands.extend(combinations(range(n), k))
        else:
            cands.extend(tuple(sorted((*c, forced))) for c in combinations(others, k - 1))
    if len(cands) > limit:
        cands = rng.sample(cands, limit)
    base = game.chips
    best: tuple[tuple[int, float, int], tuple[int, ...], float, int] | None = None
    for cards in cands:
        g = game.clone()
        try:
            g.step(PlayHand(cards))
        except (IllegalActionError, NotPortedError, ValueError):
            continue
        outcome = {GamePhase.SELECTING_HAND: 1, GamePhase.GAME_OVER: 0}.get(g.phase, 2)
        gained = g.chips - base
        key = (outcome, gained, -len(cards))
        if best is None or key > best[0]:
            best = (key, cards, gained, outcome)
    return None if best is None else (best[1], best[2], best[3])


class RandomPolicy:
    """A random kind of move (:data:`RANDOM_WEIGHTS`), then a random instance of it.

    ``hand_skill`` is the share of hands played as :func:`best_play` finds them,
    so random runs still get past their blinds into shops and packs (0: every
    selection random)."""

    name = "random"

    def __init__(self, hand_skill: float = 0.5) -> None:
        self.hand_skill = hand_skill

    def choose(
        self, game: Game, legal: Sequence[Any], rng: random.Random, view: dict | None = None
    ) -> Any:
        view = view if view is not None else game.bot_view()
        groups = _group(legal)
        kinds = list(groups)
        kind = rng.choices(kinds, [RANDOM_WEIGHTS.get(k, 1.0) for k in kinds])[0]
        if kind == "PlayHand" and rng.random() < self.hand_skill:
            best = best_play(game, view, rng, limit=150)
            if best is not None:
                return PlayHand(best[0])
        return self.concretize(game, rng.choice(groups[kind]), rng, view)

    def concretize(self, game: Game, action: Any, rng: random.Random, view: dict) -> Any:
        """A concrete move for a legal-list entry: a random selection for the PlayHand /
        Discard marker, sometimes random targets for a consumable."""
        hand = _cards(view, "hand")
        forced = _forced(view)
        if isinstance(action, (PlayHand, Discard)) and not action.card_indices and hand:
            k = rng.randint(1, min(5, len(hand)))
            cards = _with_forced(rng.sample(range(len(hand)), k), forced)
            return type(action)(cards)
        if isinstance(action, (UseConsumable, PickPackCard)) and action.target_indices:
            if rng.random() < 0.5 and hand:
                area = "consumables" if isinstance(action, UseConsumable) else "pack_cards"
                try:
                    lo, hi, needs = game._g.consumable_target_info(area, action.card_index)
                except ValueError:
                    return action
                if needs and hi >= 1:
                    k = rng.randint(max(1, lo), max(1, min(hi, len(hand))))
                    targets = _with_forced(rng.sample(range(len(hand)), k), forced, limit=k)
                    return type(action)(action.card_index, targets)
        return action


class SmartPolicy:
    """Plays to get deep, with a little exploration (``explore``: the share of
    random moves)."""

    name = "smart"

    def __init__(self, explore: float = 0.08, max_shop_moves: int = 8) -> None:
        self.explore = explore
        self.max_shop_moves = max_shop_moves
        self._random = RandomPolicy(hand_skill=0.0)
        self._shop_moves = 0

    def choose(
        self, game: Game, legal: Sequence[Any], rng: random.Random, view: dict | None = None
    ) -> Any:
        view = view if view is not None else game.bot_view()
        state = view.get("state")
        if state != "SHOP":
            self._shop_moves = 0
        if rng.random() < self.explore:
            return self._random.choose(game, legal, rng, view)
        by = _group(legal)
        if state == "BLIND_SELECT":
            if by.get("SkipBlind") and rng.random() < 0.12:
                return by["SkipBlind"][0]
            return (by.get("SelectBlind") or list(legal))[0]
        if state == "SELECTING_HAND":
            return self._hand(game, view, by, legal, rng)
        if state == "ROUND_EVAL":
            return (by.get("CashOut") or list(legal))[0]
        if state == "SHOP":
            return self._shop(view, by, legal, rng)
        if state == "SMODS_BOOSTER_OPENED":
            return self._pack(game, view, by, legal, rng)
        return self._random.choose(game, legal, rng, view)

    # -- choosing a hand ------------------------------------------------------------

    def _hand(
        self,
        game: Game,
        view: dict,
        by: dict[str, list[Any]],
        legal: Sequence[Any],
        rng: random.Random,
    ) -> Any:
        consumables = _cards(view, "consumables")
        uses = by.get("UseConsumable", [])
        planets = [
            a
            for a in uses
            if a.card_index < len(consumables) and consumables[a.card_index].get("set") == "PLANET"
        ]
        if planets:
            return planets[0]
        if uses and rng.random() < 0.3:
            return self._random.concretize(game, rng.choice(uses), rng, view)
        moves = [a for k in ("SortHand", "SwapHandLeft", "SwapHandRight") for a in by.get(k, [])]
        if moves and rng.random() < 0.05:
            return rng.choice(moves)
        best = best_play(game, view, rng) if by.get("PlayHand") else None
        if best is None:
            return self._random.choose(game, legal, rng, view)
        cards, gained, outcome = best
        rnd = view.get("round") or {}
        current = [b for b in (view.get("blinds") or {}).values() if b.get("status") == "CURRENT"]
        need = (current[0].get("score", 0) if current else 0) - rnd.get("chips", 0)
        hands_left = rnd.get("hands_left", 1)
        if (
            outcome != 2
            and by.get("Discard")
            and rnd.get("discards_left", 0) > 0
            and gained * hands_left < need
        ):
            hand = _cards(view, "hand")
            forced = _forced(view)
            keep = set(cards)
            spare = [i for i in range(len(hand)) if i not in keep and i != forced]
            spare.sort(key=lambda i: _rank_value(hand[i]))
            if spare:
                return Discard(_with_forced(spare[:5], forced))
        return PlayHand(cards)

    # -- the shop and packs ------------------------------------------------------

    def _shop(
        self, view: dict, by: dict[str, list[Any]], legal: Sequence[Any], rng: random.Random
    ) -> Any:
        self._shop_moves += 1
        leave = (by.get("NextRound") or list(legal))[0]
        if self._shop_moves > self.max_shop_moves:
            return leave
        consumables = _cards(view, "consumables")
        for a in by.get("UseConsumable", []):
            if a.card_index < len(consumables) and consumables[a.card_index].get("set") == "PLANET":
                return a
        shop = _cards(view, "shop")
        jokers = _cards(view, "jokers")
        limit = (view.get("jokers") or {}).get("limit", 5)
        buys = by.get("BuyCard", [])

        def kind(a: Any) -> str:
            return shop[a.shop_index].get("set", "") if a.shop_index < len(shop) else ""

        joker_buys = [a for a in buys if kind(a) == "JOKER"]
        if joker_buys and len(jokers) < limit and rng.random() < 0.8:
            return rng.choice(joker_buys)
        other = [a for a in buys if kind(a) in ("PLANET", "TAROT", "SPECTRAL")]
        if other and rng.random() < 0.5:
            return rng.choice(other)
        if by.get("RedeemVoucher") and rng.random() < 0.6:
            return rng.choice(by["RedeemVoucher"])
        if by.get("OpenBooster") and rng.random() < 0.5:
            return rng.choice(by["OpenBooster"])
        sells = [a for a in by.get("SellCard", []) if a.area == "jokers"]
        if sells and len(jokers) >= limit and rng.random() < 0.15:
            return min(sells, key=lambda a: _sell_value(jokers, a.card_index))
        swaps = by.get("SwapJokersLeft", []) + by.get("SwapJokersRight", [])
        if swaps and rng.random() < 0.05:
            return rng.choice(swaps)
        if by.get("Reroll") and view.get("money", 0) >= 12 and rng.random() < 0.15:
            return by["Reroll"][0]
        return leave

    def _pack(
        self,
        game: Game,
        view: dict,
        by: dict[str, list[Any]],
        legal: Sequence[Any],
        rng: random.Random,
    ) -> Any:
        picks = by.get("PickPackCard", [])
        skip = by.get("SkipPack", [])
        if not picks or (skip and rng.random() < 0.1):
            return (skip or list(legal))[0]
        pack = _cards(view, "pack")
        jokers = _cards(view, "jokers")
        limit = (view.get("jokers") or {}).get("limit", 5)

        def kind(a: Any) -> str:
            return pack[a.card_index].get("set", "") if a.card_index < len(pack) else ""

        for wanted in ("JOKER", "PLANET"):
            if wanted == "JOKER" and len(jokers) >= limit:
                continue
            choice = [a for a in picks if kind(a) == wanted]
            if choice:
                return rng.choice(choice)
        return self._random.concretize(game, rng.choice(picks), rng, view)


_RANK_ORDER = {r: i for i, r in enumerate("23456789TJQKA")}


def _rank_value(card: dict[str, Any]) -> int:
    value = card.get("value") or {}
    rank = value.get("rank") if isinstance(value, dict) else None
    return _RANK_ORDER.get(rank, -1)


def _sell_value(jokers: list[dict[str, Any]], index: int) -> float:
    if index >= len(jokers):
        return 0.0
    cost = jokers[index].get("cost") or {}
    return float(cost.get("sell", 0)) if isinstance(cost, dict) else 0.0


POLICIES: dict[str, Callable[[], Any]] = {"random": RandomPolicy, "smart": SmartPolicy}


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One run to play: seed, deck and stake in BalatroBot's names."""

    seed: str
    deck: str = "RED"
    stake: str = "WHITE"


@dataclass
class GameResult:
    """How one lockstep game went."""

    job: Job
    backend: str
    outcome: str = "error"
    steps: int = 0
    ante: int = 0
    round: int = 0
    seconds: float = 0.0
    refusals: int = 0
    methods: Counter[str] = field(default_factory=Counter)
    actions: Counter[str] = field(default_factory=Counter)
    explained: Counter[str] = field(default_factory=Counter)
    reconciled: Counter[str] = field(default_factory=Counter)
    vetoed: Counter[str] = field(default_factory=Counter)
    #: Live answers read again until they agreed (LiveAdapter.recheck), by method.
    rechecked: Counter[str] = field(default_factory=Counter)
    #: The divergence, if any: kind, message, bundle path, classified ids.
    divergence: dict[str, Any] | None = None
    error: str = ""

    @property
    def unclassified(self) -> bool:
        return self.outcome == "divergence"

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.job.seed,
            "deck": self.job.deck,
            "stake": self.job.stake,
            "backend": self.backend,
            "outcome": self.outcome,
            "steps": self.steps,
            "ante": self.ante,
            "round": self.round,
            "seconds": round(self.seconds, 3),
            "refusals": self.refusals,
            "methods": dict(self.methods),
            "actions": dict(self.actions),
            "explained": dict(self.explained),
            "reconciled": dict(self.reconciled),
            "vetoed": dict(self.vetoed),
            "rechecked": dict(self.rechecked),
            "divergence": self.divergence,
            "error": self.error,
        }

    def line(self) -> str:
        head = (
            f"{self.job.seed} {self.job.deck}/{self.job.stake} on {self.backend}: {self.outcome}, "
            f"{self.steps} steps, ante {self.ante} round {self.round}, {self.seconds:.1f}s"
        )
        if self.divergence:
            head += f"\n    {self.divergence['message']}"
            if self.divergence.get("bundle"):
                head += f"\n    bundle: {self.divergence['bundle']}"
        if self.error:
            head += f"\n    {self.error}"
        return head


def play_game(
    adapter: LiveAdapter,
    policy: Any,
    job: Job,
    *,
    max_steps: int = 1000,
    rng: random.Random | None = None,
    endless: bool = False,
    max_refusals: int = 40,
) -> GameResult:
    """Play one run in lockstep until it ends, diverges or reaches ``max_steps``."""
    rng = rng or random.Random(job.seed)
    result = GameResult(job, describe_backend(adapter.backend))
    fallback = RandomPolicy()
    t0 = time.perf_counter()
    try:
        adapter.start(job.deck, job.stake, job.seed)
        refused: set[Any] = set()
        streak = 0
        while result.steps < max_steps:
            if adapter.done:
                result.outcome = "won" if adapter.won else "lost"
                break
            if adapter.won and not endless:
                result.outcome = "won"
                break
            legal = [a for a in adapter.get_legal_actions() if a not in refused]
            if not legal:
                result.outcome = "stuck"
                break
            # After a refusal, a random move: the policy would pick the same one again.
            chooser = fallback if streak else policy
            action = chooser.choose(adapter.game, legal, rng, adapter.shadow_view)
            try:
                adapter.step(action)
            except IllegalActionError:  # VetoedActionError too
                result.refusals += 1
                streak += 1
                if action in legal:
                    refused.add(action)
                if streak > max_refusals:
                    result.outcome = "stuck"
                    break
                continue
            refused.clear()
            streak = 0
            result.steps += 1
        else:
            result.outcome = "max-steps"
    except DivergenceError as exc:
        result.outcome = "divergence" if exc.unclassified else "classified"
        result.divergence = {
            "kind": exc.kind,
            "step": exc.step,
            "message": str(exc),
            "bundle": None if exc.bundle_path is None else str(exc.bundle_path),
            "classified": sorted(exc.classified),
        }
    except NotPortedError as exc:
        result.outcome = "not-ported"
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - one game's failure must not stop the others
        result.outcome = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.perf_counter() - t0
    result.methods = Counter(adapter.methods)
    result.actions = Counter(adapter.actions)
    result.explained = Counter(adapter.explained)
    result.reconciled = Counter(adapter.reconciled)
    result.vetoed = Counter(adapter.vetoed)
    result.rechecked = Counter(adapter.rechecked)
    if adapter.started:
        result.ante, result.round = adapter.game.ante, adapter.game.round
    return result


# ---------------------------------------------------------------------------
# Many games
# ---------------------------------------------------------------------------


def make_jobs(
    seeds: Sequence[str] | int,
    decks: Sequence[str] = ("RED",),
    stakes: Sequence[str] = ("WHITE",),
    *,
    rng_seed: int = 0,
) -> list[Job]:
    """Jobs for ``seeds`` (a list, or a count of vanilla-style random seeds from
    ``rng_seed``), cycling through the deck/stake combinations."""
    if isinstance(seeds, int):
        r = random.Random(rng_seed)
        seeds = [random_seed(r) for _ in range(seeds)]
    combos = [(d, s) for d in decks for s in stakes]
    return [Job(seed, *combos[i % len(combos)]) for i, seed in enumerate(seeds)]


@dataclass
class Summary:
    """The totals of a lockstep session."""

    results: list[GameResult] = field(default_factory=list)

    def add(self, result: GameResult) -> None:
        self.results.append(result)

    @property
    def unclassified(self) -> list[GameResult]:
        return [r for r in self.results if r.outcome == "divergence"]

    def totals(self) -> dict[str, Any]:
        outcomes = Counter(r.outcome for r in self.results)
        methods: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        explained: Counter[str] = Counter()
        reconciled: Counter[str] = Counter()
        vetoed: Counter[str] = Counter()
        rechecked: Counter[str] = Counter()
        for r in self.results:
            methods.update(r.methods)
            actions.update(r.actions)
            explained.update(r.explained)
            reconciled.update(r.reconciled)
            vetoed.update(r.vetoed)
            rechecked.update(r.rechecked)
        return {
            "games": len(self.results),
            "steps": sum(r.steps for r in self.results),
            "outcomes": dict(outcomes),
            "unclassified_divergences": len(self.unclassified),
            "classified_divergences": outcomes.get("classified", 0),
            "max_ante": max((r.ante for r in self.results), default=0),
            "methods": dict(methods),
            "actions": dict(actions),
            "explained": dict(explained),
            "reconciled": dict(reconciled),
            "vetoed": dict(vetoed),
            "rechecked": dict(rechecked),
            "bundles": [
                r.divergence["bundle"]
                for r in self.results
                if r.divergence and r.divergence.get("bundle")
            ],
        }

    def lines(self) -> list[str]:
        t = self.totals()
        out = [
            f"games: {t['games']}, steps: {t['steps']}, max ante: {t['max_ante']}",
            "outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(t["outcomes"].items())),
            f"unclassified divergences: {t['unclassified_divergences']}, "
            f"classified: {t['classified_divergences']}",
            "requests: " + ", ".join(f"{k} {v}" for k, v in sorted(t["methods"].items())),
            "actions: " + ", ".join(f"{k} {v}" for k, v in sorted(t["actions"].items())),
        ]
        for name in ("explained", "reconciled", "vetoed", "rechecked"):
            if t[name]:
                out.append(f"{name}: " + ", ".join(f"{k} {v}" for k, v in sorted(t[name].items())))
        for b in t["bundles"]:
            out.append(f"bundle: {b}")
        return out


def run_lockstep(
    backends: Sequence[Any],
    jobs: Sequence[Job],
    *,
    policy: str = "smart",
    max_steps: int = 1000,
    bundle_dir: Any = None,
    endless: bool = False,
    on_result: Callable[[int, GameResult], None] | None = None,
    stop_on_divergence: bool = False,
    settle: float = 0.0,
    recheck: float | None = None,
) -> Summary:
    """Play ``jobs`` on ``backends`` in parallel, one worker thread per backend (a
    BalatroBot instance plays one game at a time)."""
    todo: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        todo.put(job)
    summary = Summary()
    lock = threading.Lock()
    stop = threading.Event()

    def worker(index: int, backend: Any) -> None:
        adapter = LiveAdapter(
            backend, bundle_dir=bundle_dir, label=f"i{index}", settle=settle, recheck=recheck
        )
        chooser = POLICIES[policy]()
        while not stop.is_set():
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            result = play_game(
                adapter,
                chooser,
                job,
                max_steps=max_steps,
                rng=random.Random(f"{job.seed}/{job.deck}/{job.stake}"),
                endless=endless,
            )
            with lock:
                summary.add(result)
                if on_result is not None:
                    on_result(index, result)
            if stop_on_divergence and result.unclassified:
                stop.set()

    threads = [
        threading.Thread(target=worker, args=(i, b), name=f"lockstep-{i}", daemon=True)
        for i, b in enumerate(backends)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return summary
