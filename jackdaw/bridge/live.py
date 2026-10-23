"""The live lockstep: a live BalatroBot game played in step with a shadow Rust run.

:class:`LiveAdapter` drives a BalatroBot game (``LiveBackend``, or anything
with ``handle(method, params)``) and keeps a *shadow* run in the Rust engine
(``jackdaw.core.Game``) started from the live run's seed, deck and stake. For
every request:

1. the shadow takes it first, on a copy, through the simulator's port of the
   same BalatroBot endpoint (``jackdaw._core.BotSession``): the bot's argument
   and state checks, the endpoint's own checks, then vanilla's rules. A refusal
   stops the request there, before the live game sees it;
2. the deviation catalog (:mod:`jackdaw.bridge.deviations`) may veto it;
3. it is sent to the live game;
4. the shadow keeps the copy's new state;
5. the live game's view is diffed against the shadow's ``bot_view()``
   (:mod:`jackdaw.bridge.viewdiff`); differences the catalog explains are
   dropped or reconciled, anything else raises :class:`DivergenceError` and
   writes a repro bundle.

Observations, masks and legal actions come from the shadow (BALATROBOT
legality), so an agent sees exactly what it saw in training.
``jackdaw lockstep replay`` reruns a bundle on the simulator alone. See
``docs/live.md``.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jackdaw import _core
from jackdaw.bridge.backend import (
    DECK_FROM_BOT,
    INVALID_STATE,
    NOT_ALLOWED,
    STAKE_FROM_BOT,
    RPCError,
    to_bot_add_params,
)
from jackdaw.bridge.deviations import (
    CATALOG,
    Classification,
    Deviation,
    RunMemory,
    StepContext,
    by_id,
    classify,
    refusal_entry,
    triggered,
    update_memory,
)
from jackdaw.bridge.deviations import (
    veto_reason as catalog_veto,
)
from jackdaw.bridge.reconcile import ReconcileContext, ReconcileError, Unrepairable, apply_edits
from jackdaw.bridge.viewdiff import FieldDiff, IdMap, diff_views
from jackdaw.core import Game, IllegalActionError, Legality, NotPortedError, encode_action
from jackdaw.engine.actions import Discard, GamePhase, PlayHand

if TYPE_CHECKING:
    from jackdaw.env.game_interface import GameState

BUNDLE_FORMAT = "jackdaw.lockstep-bundle/1"

DECK_TO_BOT: dict[str, str] = {v: k for k, v in DECK_FROM_BOT.items()}
STAKE_TO_BOT: dict[int, str] = {v: k for k, v in STAKE_FROM_BOT.items()}

# Requests that start, end or replace the run: not taken in lockstep.
_NOT_IN_LOCKSTEP = frozenset({"start", "menu", "save", "load", "screenshot"})


class DivergenceError(RuntimeError):
    """The live game and the shadow run no longer agree.

    ``kind`` is ``view`` (the views differ), ``reconcile`` (a reconcile left
    differences), ``live-refused`` (the live game refused what the shadow
    took) or ``live-error`` (the live game didn't answer). ``diffs`` are the
    unclassified differences; ``classified`` maps a catalog id to the
    differences it explains but couldn't repair. ``bundle`` is the repro
    bundle, written to ``bundle_path`` when the adapter has a bundle directory.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "view",
        step: int = 0,
        diffs: list[FieldDiff] | None = None,
        classified: dict[str, list[FieldDiff]] | None = None,
        bundle: dict[str, Any] | None = None,
        bundle_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.step = step
        self.diffs = list(diffs or [])
        self.classified = dict(classified or {})
        self.bundle = bundle
        self.bundle_path = bundle_path

    @property
    def unclassified(self) -> bool:
        """Not explained by the catalog: a new divergence to triage."""
        return bool(self.diffs) or not self.classified


class VetoedActionError(IllegalActionError):
    """The deviation catalog keeps this action out of the lockstep."""

    def __init__(self, deviation: str, reason: str) -> None:
        super().__init__(f"vetoed ({deviation}): {reason}")
        self.deviation = deviation
        self.reason = reason


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _jackdaw_version() -> str:
    try:
        from importlib.metadata import version

        return version("jackdaw")
    except Exception:  # noqa: BLE001 - metadata missing in a source checkout
        return _core.__version__


def describe_backend(backend: Any) -> str:
    url = getattr(backend, "url", None)
    return str(url) if url else type(backend).__name__


def _is_marker(action: Any) -> bool:
    return isinstance(action, (PlayHand, Discard)) and not action.card_indices


class LiveAdapter:
    """Plays a live BalatroBot game in lockstep with a shadow Rust run.

    It follows the env's adapter protocol (``reset``, ``step``,
    ``get_legal_actions``, ``done``, ``won``, ``game``), so
    ``BalatroEnvironment(adapter_factory=lambda: LiveAdapter(LiveBackend()))``
    plays the live game with training's observations and masks.
    :meth:`start`, :meth:`attach` and :meth:`request` work in BalatroBot's terms.

    ``bundle_dir``: where a divergence writes its repro bundle (none: the bundle
    is only on the exception). ``catalog``: the deviation catalog. ``label``
    names the bundles. ``shadow_factory(back, stake, seed, challenge)`` makes the
    shadow run (``Game.new``; tests set up special runs with it). ``settle``: seconds
    to wait after each live answer before reading ``gamestate`` again and comparing
    that instead (0: compare the answer itself), for a live game whose endpoints
    answer before all of a step's events have run. ``recheck``: when a live answer
    differs from the shadow, read ``gamestate`` again for up to this many seconds
    until the difference is gone, and compare the last reading (BalatroBot answers
    some requests before the game's queued events have run, ``docs/live.md``).
    The default is 2 seconds against a live game and 0 against the simulator
    (``jackdaw serve``, in-process backends), which answers settled.
    """

    #: Seconds between the readings of a recheck.
    RECHECK_INTERVAL = 0.1

    #: The legality the env computes masks with for this adapter.
    legality = Legality.BALATROBOT.value

    def __init__(
        self,
        backend: Any,
        *,
        bundle_dir: str | Path | None = None,
        catalog: tuple[Deviation, ...] = CATALOG,
        label: str = "",
        shadow_factory: Callable[[str, int, str, str | None], Game] | None = None,
        settle: float = 0.0,
        recheck: float | None = None,
    ) -> None:
        self._backend = backend
        self.bundle_dir = None if bundle_dir is None else Path(bundle_dir)
        self._catalog = catalog
        self._deviations = by_id(catalog)
        self.label = label
        self._shadow_factory = shadow_factory or Game.new
        self.settle = settle
        self._recheck = recheck
        #: This run: answers read again until they agreed (recheck), by method.
        self.rechecked: Counter[str] = Counter()
        self._shadow: Game | None = None
        self._shadow_view: dict[str, Any] = {}
        self._live_view: dict[str, Any] = {}
        self._start: dict[str, Any] = {}
        self._log: list[dict[str, Any]] = []
        self._ids = IdMap()
        self._memory = RunMemory()
        self._broken: DivergenceError | None = None
        self.steps = 0
        #: This run: differences explained by a catalog entry (dropped or reconciled), by id.
        self.explained: Counter[str] = Counter()
        #: This run: reconciles applied, by catalog id.
        self.reconciled: Counter[str] = Counter()
        #: This run: requests the catalog kept out of the lockstep, by id.
        self.vetoed: Counter[str] = Counter()
        #: This run: requests taken in lockstep, by method.
        self.methods: Counter[str] = Counter()
        #: This run: engine actions taken in lockstep, by kind (``PlayHand``, ``SellCard:jokers``).
        self.actions: Counter[str] = Counter()

    def __repr__(self) -> str:
        seed = self._start.get("seed")
        return f"LiveAdapter({describe_backend(self._backend)}, seed={seed!r}, steps={self.steps})"

    # -- the run ----------------------------------------------------------------

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def started(self) -> bool:
        """A run is being shadowed."""
        return self._shadow is not None

    @property
    def game(self) -> Game:
        """The shadow run (``jackdaw.core.Game``): the next step moves it on, like
        :class:`~jackdaw.env.game_interface.DirectAdapter`'s ``game``."""
        if self._shadow is None:
            raise RuntimeError("LiveAdapter: call reset(), start() or attach() first")
        return self._shadow

    @property
    def view(self) -> dict[str, Any]:
        """The live game's last view."""
        return self._live_view

    @property
    def shadow_view(self) -> dict[str, Any]:
        """The shadow run's view (``game.bot_view()``)."""
        return self._shadow_view

    @property
    def start_params(self) -> dict[str, Any]:
        return dict(self._start)

    @property
    def log(self) -> list[dict[str, Any]]:
        """Everything the shadow took since the start: requests and reconciles."""
        return list(self._log)

    @property
    def done(self) -> bool:
        return self.game.phase == GamePhase.GAME_OVER

    @property
    def won(self) -> bool:
        return self.state().won

    def state(self) -> GameState:
        from jackdaw.env.game_interface import game_state

        return game_state(self.game)

    # -- starting -----------------------------------------------------------------

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str | None,
        *,
        challenge: dict[str, Any] | str | None = None,
    ) -> GameState:
        """Start a live run (``menu`` + ``start``) and its shadow. BalatroBot's
        ``start`` can't begin a challenge: start it in the game, then :meth:`attach`."""
        if challenge:
            raise ValueError(
                "BalatroBot's start has no challenge: start the challenge in the game and "
                "call LiveAdapter.attach(challenge=...)"
            )
        if back_key not in DECK_TO_BOT:
            raise ValueError(f"BalatroBot's start can't select {back_key!r}")
        self.start(DECK_TO_BOT[back_key], STAKE_TO_BOT[int(stake)], seed)
        return self.state()

    def start(self, deck: str = "RED", stake: str = "WHITE", seed: str | None = None) -> dict:
        """``menu`` + ``start`` on the live game, then a shadow run from the seed, deck
        and stake the live game reports. Returns the live view."""
        self._backend.handle("menu", {})
        params: dict[str, Any] = {"deck": deck, "stake": stake}
        if seed:
            params["seed"] = seed
        live = self._ask_live("start", params)
        return self._begin(live, params, None)

    def attach(self, challenge: str | None = None) -> dict:
        """Shadow a live run that is already started and untouched (blind select of
        ante 1, round 0): a challenge, which ``start`` can't begin, or a run started
        by hand. Returns the live view."""
        live = self._backend.handle("gamestate", {})
        fresh = (
            live.get("state") == "BLIND_SELECT"
            and live.get("round_num") == 0
            and live.get("ante_num") == 1
        )
        if not fresh:
            raise ValueError(
                "attach needs a run at its first blind select (state "
                f"{live.get('state')}, ante {live.get('ante_num')}, round {live.get('round_num')})"
            )
        return self._begin(live, None, challenge)

    def _begin(
        self, live: dict[str, Any], request: dict[str, Any] | None, challenge: str | None
    ) -> dict:
        seed, deck, stake = live.get("seed"), live.get("deck"), live.get("stake")
        if not seed or stake not in STAKE_FROM_BOT:
            raise ValueError(f"the live run reports no seed or stake (seed {seed!r}, {stake!r})")
        back = DECK_FROM_BOT.get(deck) if deck else None
        if back is None and challenge is None:
            raise ValueError(
                f"the live run's deck is {deck!r}: a challenge? pass attach(challenge=...)"
            )
        self._start = {
            "request": request,
            "seed": seed,
            "deck": deck,
            "stake": stake,
            "back": back or "b_challenge",
            "stake_level": STAKE_FROM_BOT[stake],
            "challenge": challenge,
        }
        self._shadow = self._shadow_factory(back or "b_red", STAKE_FROM_BOT[stake], seed, challenge)
        self._shadow_view = self._shadow.bot_view()
        self._log = []
        self._ids = IdMap()
        self._memory = RunMemory()
        self._broken = None
        self.steps = 0
        for counter in (
            self.explained,
            self.reconciled,
            self.vetoed,
            self.methods,
            self.actions,
            self.rechecked,
        ):
            counter.clear()
        ctx = StepContext(
            "start" if request is not None else "attach",
            request or {},
            post=self._shadow,
            post_view=self._shadow_view,
            memory=self._memory,
        )
        live = self._settled(ctx, live)
        self._live_view = live
        self._compare(ctx, live, None)
        return live

    # -- stepping -----------------------------------------------------------------

    def step(self, action: Any) -> GameState:
        """Take an engine action in lockstep: its BalatroBot request goes to the shadow,
        then to the live game. Raises ``IllegalActionError`` (the shadow refuses; nothing
        is sent), :class:`VetoedActionError`, ``NotPortedError`` or
        :class:`DivergenceError`."""
        request = self.game.bot_request(action)
        if request is None:
            self.vetoed["BB-SIM-ONLY"] += 1
            raise VetoedActionError(
                "BB-SIM-ONLY", f"{type(action).__name__} has no BalatroBot request"
            )
        method, params = request
        self._lockstep(method, params, action)
        kind = type(action).__name__
        self.actions[f"{kind}:{action.area}" if kind == "SellCard" else kind] += 1
        return self.state()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict:
        """Any BalatroBot request in lockstep (``add`` and ``set`` too): the shadow,
        then the live game, then the comparison. Returns the live view; a request the
        shadow refuses raises its ``RPCError`` without reaching the live game.
        ``gamestate`` is :meth:`check`; ``health`` and ``rpc.discover`` only ask the
        live game."""
        if method in ("health", "rpc.discover"):
            return self._backend.handle(method, dict(params or {}))
        if method == "gamestate":
            self.check()
            return self._live_view
        return self._lockstep(method, dict(params or {}), None)

    def check(self, *, raise_on_divergence: bool = True) -> list[FieldDiff]:
        """Compare the live ``gamestate`` with the shadow now. Returns the differences
        the catalog doesn't drop (raising :class:`DivergenceError` for them, unless
        ``raise_on_divergence`` is False). Nothing is reconciled here: every request
        was compared (and reconciled) as it was made, so any difference left is new."""
        self._require_run()
        live = self._backend.handle("gamestate", {})
        ctx = StepContext(
            "gamestate",
            {},
            pre=self.game,
            pre_view=self._shadow_view,
            post=self.game,
            post_view=self._shadow_view,
            memory=self._memory,
        )
        diff = diff_views(self._shadow_view, live, self._ids)
        cls = classify(diff.diffs, ctx, self._catalog)
        left = cls.unclassified + [d for ds in cls.classified.values() for d in ds]
        left += [d for ds in cls.reconcile.values() for d in ds]
        if left and raise_on_divergence:
            self._diverge(
                ctx,
                kind="view",
                message="the live gamestate differs from the shadow",
                live=live,
                diffs=left,
                cls=cls,
            )
        if not left:
            self._ids.commit(diff.pairs)
        self._live_view = live
        return left

    def get_legal_actions(self) -> list[Any]:
        """The shadow's legal actions under BALATROBOT legality, minus the ones the
        catalog vetoes."""
        game = self.game
        shared: dict[str, dict[str, Any]] = {}
        out = []
        for action in game.legal_actions(Legality.BALATROBOT):
            if _is_marker(action):
                out.append(action)
                continue
            request = game.bot_request(action)
            if request is None:
                continue
            ctx = StepContext(
                request[0],
                request[1],
                action,
                pre=game,
                pre_view=self._shadow_view,
                memory=self._memory,
                _snapshots=shared,
            )
            if catalog_veto(ctx, self._catalog) is None:
                out.append(action)
        return out

    def veto_reason(self, action: Any) -> str | None:
        """Why the catalog keeps ``action`` out of the lockstep here, or None."""
        request = self.game.bot_request(action)
        if request is None:
            return f"BB-SIM-ONLY: {type(action).__name__} has no BalatroBot request"
        ctx = StepContext(
            request[0],
            request[1],
            action,
            pre=self.game,
            pre_view=self._shadow_view,
            memory=self._memory,
        )
        out = catalog_veto(ctx, self._catalog)
        return None if out is None else f"{out[0]}: {out[1]}"

    # -- internals ----------------------------------------------------------------

    def _require_run(self) -> None:
        if self._shadow is None:
            raise RuntimeError("LiveAdapter: call reset(), start() or attach() first")
        if self._broken is not None:
            raise RuntimeError(
                f"LiveAdapter: the shadow diverged at step {self._broken.step}; start a new run"
            ) from self._broken

    def _lockstep(self, method: str, params: dict[str, Any], action: Any) -> dict:
        self._require_run()
        if method in _NOT_IN_LOCKSTEP:
            raise ValueError(f"{method!r} isn't taken in lockstep (use start() or attach())")
        if method == "add" and params:
            params = to_bot_add_params(params)
        pre = Game(self.game._g)
        ctx = StepContext(
            method, params, action, pre=pre, pre_view=self._shadow_view, memory=self._memory
        )
        # 1. The shadow takes the request first, on a copy.
        session = _core.BotSession.from_game(pre._g)
        ok, text = session.call(method, json.dumps(params))
        if not ok:
            err = json.loads(text)
            refusal = RPCError(err["code"], err["message"], {"name": err["name"]})
            ctx.refusal = refusal
            if refusal.message.startswith("not ported yet"):
                raise NotPortedError(refusal.message)
            dev = refusal_entry(ctx, self._catalog)
            if dev is not None:
                self.vetoed[dev] += 1
                raise VetoedActionError(dev, refusal.message)
            if action is not None:
                raise IllegalActionError(f"{method} {params}: {refusal.message}")
            raise refusal
        inner = session.game()
        if inner is None:
            raise ValueError(f"{method!r} left the shadow without a run")
        post = Game(inner)
        ctx.post, ctx.post_view = post, json.loads(text)
        # 2. The catalog's vetoes.
        vetoed = catalog_veto(ctx, self._catalog)
        if vetoed is not None:
            self.vetoed[vetoed[0]] += 1
            raise VetoedActionError(*vetoed)
        entry: dict[str, Any] = {
            "kind": "request" if action is None else "action",
            "method": method,
            "params": params,
        }
        if action is not None:
            entry["action"] = encode_action(action)
        # 3. The live game.
        try:
            live = self._ask_live_retrying(method, params)
        except RPCError as exc:
            self._diverge(
                ctx,
                kind="live-refused",
                message=f"the live game refused {method} {params}: {exc.message}",
                live=self._fetch_live(),
                entry=entry,
            )
        except Exception as exc:  # noqa: BLE001 - any failure to talk to the live game
            self._diverge(
                ctx,
                kind="live-error",
                message=f"the live game didn't answer {method} {params}: {exc!r}",
                live=self._fetch_live(),
                entry=entry,
            )
        if not isinstance(live, dict):
            self._diverge(
                ctx,
                kind="live-error",
                message=f"the live game answered {method} with {live!r}",
                live=self._fetch_live(),
                entry=entry,
            )
        # 4. The shadow keeps the step.
        self.game._g = post._g
        self._shadow_view = ctx.post_view
        update_memory(ctx, self._catalog)
        # 5. The comparison.
        live = self._settled(ctx, live)
        self._compare(ctx, live, entry)
        self._log.append(entry)
        self.steps += 1
        self.methods[method] += 1
        self._live_view = live
        return live

    def _ask_live_retrying(self, method: str, params: dict[str, Any]) -> Any:
        """:meth:`_ask_live`, asking again for up to :attr:`recheck` seconds when the live
        game refuses for its state (``INVALID_STATE``, ``NOT_ALLOWED``): it may still be
        running the last request's events (``use`` while ``G.GAME.STOP_USE`` from a pack pick
        holds, for instance). The shadow took the request, and a refused one changes nothing."""
        deadline = time.monotonic() + self.recheck
        retried = False
        while True:
            try:
                return self._ask_live(method, params)
            except RPCError as exc:
                if exc.code not in (INVALID_STATE, NOT_ALLOWED) or time.monotonic() >= deadline:
                    raise
                if not retried:
                    self.rechecked[f"{method} (refused)"] += 1
                    retried = True
                time.sleep(self.RECHECK_INTERVAL)

    def _ask_live(self, method: str, params: dict[str, Any]) -> Any:
        """The live game's answer to a request; with ``settle``, its ``gamestate`` a
        moment later."""
        answer = self._backend.handle(method, params)
        if self.settle > 0:
            time.sleep(self.settle)
            answer = self._backend.handle("gamestate", {})
        return answer

    @property
    def recheck(self) -> float:
        """Seconds a differing answer is read again for (see the class docstring)."""
        if self._recheck is not None:
            return self._recheck
        return 2.0 if getattr(self._backend, "simulator", None) is False else 0.0

    def _settled(self, ctx: StepContext, live: dict[str, Any]) -> dict[str, Any]:
        """The live view to compare: the answer, or, while it differs from the shadow
        in ways the catalog doesn't drop, ``gamestate`` read again until it agrees or
        :attr:`recheck` seconds have passed (the last reading is compared). Differences
        a reconcile could take are read again too: a late answer can look like one."""
        if self.recheck <= 0 or not isinstance(live, dict):
            return live
        deadline = time.monotonic() + self.recheck
        read = 0
        while self._unexplained(ctx, live) and time.monotonic() < deadline:
            time.sleep(self.RECHECK_INTERVAL)
            again = self._fetch_live()
            if again is None:
                break
            live, read = again, read + 1
        if read:
            self.rechecked[ctx.method or "?"] += 1
        return live

    def _unexplained(self, ctx: StepContext, live: dict[str, Any]) -> bool:
        assert ctx.post_view is not None
        diff = diff_views(ctx.post_view, live, self._ids)
        if not diff.diffs:
            return False
        cls = classify(diff.diffs, ctx, self._catalog)
        return bool(cls.unclassified or cls.classified or cls.reconcile)

    def _fetch_live(self) -> dict[str, Any] | None:
        try:
            out = self._backend.handle("gamestate", {})
        except Exception:  # noqa: BLE001 - the bundle records whatever the game still says
            return None
        return out if isinstance(out, dict) else None

    def _compare(
        self,
        ctx: StepContext,
        live: dict[str, Any],
        entry: dict[str, Any] | None,
    ) -> None:
        assert ctx.post_view is not None
        diff = diff_views(ctx.post_view, live, self._ids)
        if not diff.diffs:
            self._ids.commit(diff.pairs)
            return
        cls = classify(diff.diffs, ctx, self._catalog)
        if cls.unclassified or cls.classified:
            self._diverge(
                ctx,
                kind="view",
                message=_divergence_message(ctx.method, cls),
                live=live,
                diffs=cls.unclassified,
                classified=cls.classified,
                entry=entry,
                cls=cls,
            )
        for dev_id, diffs in cls.ignored.items():
            self.explained[dev_id] += len(diffs)
        if not cls.reconcile:
            self._ids.commit(diff.pairs)
            return
        # Reconcile: copy the live values into the shadow.
        snapshot = self.game.to_dict()
        known = set()
        if ctx.pre_view:
            for area in (
                "jokers",
                "consumables",
                "cards",
                "hand",
                "shop",
                "vouchers",
                "packs",
                "pack",
            ):
                a = ctx.pre_view.get(area)
                if isinstance(a, dict):
                    known.update(c.get("id") for c in a.get("cards") or [])
        rctx = ReconcileContext(snapshot, ctx.post_view, live, [], diff.cards, self._ids, known)
        edits: list[dict[str, Any]] = []
        known_limits: dict[str, list[FieldDiff]] = {}  # Unrepairable: classified
        misfits: list[FieldDiff] = []  # the live values don't fit the deviation
        reasons = []
        for dev_id, diffs in cls.reconcile.items():
            dev = self._deviations[dev_id]
            rctx.diffs = diffs
            try:
                assert dev.reconcile is not None
                dev_edits = dev.reconcile(rctx, ctx)
                apply_edits(snapshot, dev_edits)
            except Unrepairable as exc:
                known_limits[dev_id] = diffs
                reasons.append(f"{dev_id}: {exc}")
                continue
            except ReconcileError as exc:
                misfits += diffs
                reasons.append(f"{dev_id} doesn't fit: {exc}")
                continue
            edits += dev_edits
        if known_limits or misfits:
            self._diverge(
                ctx,
                kind="reconcile",
                message="the shadow can't take the live values (" + "; ".join(reasons) + ")",
                live=live,
                diffs=misfits,
                classified=known_limits,
                entry=entry,
                cls=cls,
            )
        try:
            self.game.replace_state(snapshot)
        except ValueError as exc:
            self._diverge(
                ctx,
                kind="reconcile",
                message=f"the reconciled snapshot doesn't load: {exc}",
                live=live,
                diffs=[d for ds in cls.reconcile.values() for d in ds],
                entry=entry,
                cls=cls,
            )
        new_view = self.game.bot_view()
        again = diff_views(new_view, live, self._ids)
        left = classify(again.diffs, ctx, self._catalog)
        rest = left.unclassified + [
            d for ds in (*left.classified.values(), *left.reconcile.values()) for d in ds
        ]
        if rest:
            # The reconcile didn't account for everything: not explained.
            fixes = ", ".join(cls.reconcile)
            self._diverge(
                ctx,
                kind="reconcile",
                message=f"the reconcile ({fixes}) left {len(rest)} difference(s): "
                + "; ".join(str(d) for d in rest[:3]),
                live=live,
                diffs=rest,
                entry=entry,
                cls=cls,
            )
        for dev_id, diffs in cls.reconcile.items():
            self.explained[dev_id] += len(diffs)
            self.reconciled[dev_id] += 1
        self._shadow_view = new_view
        ctx.post_view = new_view
        self._ids.commit(again.pairs)
        if entry is not None:
            entry["reconcile"] = {"deviations": sorted(cls.reconcile), "edits": edits}
        else:
            self._log.append(
                {"kind": "reconcile", "deviations": sorted(cls.reconcile), "edits": edits}
            )

    def _diverge(
        self,
        ctx: StepContext,
        *,
        kind: str,
        message: str,
        live: dict[str, Any] | None,
        diffs: list[FieldDiff] | None = None,
        classified: dict[str, list[FieldDiff]] | None = None,
        entry: dict[str, Any] | None = None,
        cls: Classification | None = None,
    ) -> None:
        step = self.steps + (0 if ctx.method in ("start", "attach", "gamestate") else 1)
        bundle = self._bundle(ctx, kind, message, live, diffs or [], classified or {}, entry, cls)
        path = self._write_bundle(bundle, ctx) if self.bundle_dir is not None else None
        where = f" (bundle: {path})" if path else ""
        err = DivergenceError(
            f"lockstep divergence at step {step}: {message}{where}",
            kind=kind,
            step=step,
            diffs=diffs,
            classified=classified,
            bundle=bundle,
            bundle_path=path,
        )
        self._broken = err
        raise err

    def _bundle(
        self,
        ctx: StepContext,
        kind: str,
        message: str,
        live: dict[str, Any] | None,
        diffs: list[FieldDiff],
        classified: dict[str, list[FieldDiff]],
        entry: dict[str, Any] | None,
        cls: Classification | None,
    ) -> dict[str, Any]:
        step = None
        if entry is not None:
            step = {k: entry[k] for k in ("kind", "method", "params", "action") if k in entry}
        elif ctx.method not in ("start", "attach"):
            step = {"kind": "check", "method": ctx.method, "params": ctx.params}
        # The entries this step concerns: those whose trigger fired, and the ones
        # the differences were classified under (always-on entries aside).
        candidates = [
            self._deviations[dev_id].to_json()
            for dev_id in dict.fromkeys([*triggered(ctx, self._catalog), *sorted(classified)])
            if dev_id in self._deviations
        ]
        return {
            "format": BUNDLE_FORMAT,
            "created": _now(),
            "jackdaw": _jackdaw_version(),
            "balatrobot": _core.BALATROBOT_VERSION,
            "backend": describe_backend(self._backend),
            "label": self.label,
            "start": self._start,
            "log": list(self._log),
            "step": step,
            "divergence": {
                "kind": kind,
                "message": message,
                "diff": [d.to_json() for d in diffs],
                "classified": {k: [d.to_json() for d in v] for k, v in classified.items()},
                "candidates": candidates,
            },
            "ids": self._ids.to_json(),
            "memory": self._memory.to_json(),
            "explained": dict(self.explained),
            "reconciled": dict(self.reconciled),
            "vetoed": dict(self.vetoed),
            "views": {
                "sim": ctx.post_view,
                "live": live,
                "sim_before": ctx.pre_view,
                "live_before": self._live_view if ctx.pre_view is not None else None,
            },
            "shadow_before": ctx.pre_snapshot if ctx.pre is not None else None,
        }

    def _write_bundle(self, bundle: dict[str, Any], ctx: StepContext) -> Path:
        assert self.bundle_dir is not None
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [stamp, self.label, str(self._start.get("seed", "")), f"step{self.steps + 1}"]
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", "-".join(p for p in parts if p))
        path = self.bundle_dir / name
        n = 1
        while path.exists():
            n += 1
            path = self.bundle_dir / f"{name}-{n}"
        write_bundle(bundle, path)
        return path


def _divergence_message(method: str | None, cls: Classification) -> str:
    parts = []
    if cls.unclassified:
        shown = "; ".join(str(d) for d in cls.unclassified[:3])
        more = len(cls.unclassified) - 3
        parts.append(
            f"{len(cls.unclassified)} unclassified difference(s): {shown}"
            + (f"; and {more} more" if more > 0 else "")
        )
    for dev_id, diffs in cls.classified.items():
        parts.append(f"{len(diffs)} difference(s) classified as {dev_id} (a veto missed them)")
    return f"after {method}: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

_VIEW_FILES = {
    "sim": "sim_view.json",
    "live": "live_view.json",
    "sim_before": "sim_view_before.json",
    "live_before": "live_view_before.json",
}


def write_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Write a bundle as a directory: ``bundle.json`` (start params, the log, the
    divergent step and its field diff), the views, the shadow's snapshot before the
    step and a ``summary.txt``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in bundle.items() if k not in ("views", "shadow_before")}
    head["files"] = {}
    for key, name in _VIEW_FILES.items():
        view = bundle["views"].get(key)
        if view is not None:
            (path / name).write_text(json.dumps(view, indent=1, sort_keys=True), encoding="utf-8")
            head["files"][key] = name
    if bundle.get("shadow_before") is not None:
        from jackdaw.core.game import _to_json

        (path / "shadow_before.json").write_text(
            json.dumps(_to_json(bundle["shadow_before"]), sort_keys=True), encoding="utf-8"
        )
        head["files"]["shadow_before"] = "shadow_before.json"
    (path / "bundle.json").write_text(_dump_head(head), encoding="utf-8")
    (path / "summary.txt").write_text(summary_text(bundle, path), encoding="utf-8")
    return path


def _dump_head(head: dict[str, Any]) -> str:
    """``bundle.json``: indented, with one line per log entry."""
    head = dict(head)
    log = head.pop("log")
    text = json.dumps(head, indent=1)
    entries = ",\n".join(f"  {json.dumps(e)}" for e in log)
    body = f"[\n{entries}\n ]" if log else "[]"
    return f'{text[:-2]},\n "log": {body}\n}}\n'


def read_bundle(path: str | Path) -> dict[str, Any]:
    """A bundle written by :func:`write_bundle` (the directory or its ``bundle.json``)."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    head = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    if head.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path}: not a {BUNDLE_FORMAT} bundle")
    files = head.pop("files", {})
    bundle = dict(head)
    bundle["views"] = {
        key: json.loads((path / name).read_text(encoding="utf-8"))
        for key, name in files.items()
        if key in _VIEW_FILES
    }
    if "shadow_before" in files:
        from jackdaw.core.game import _from_json

        bundle["shadow_before"] = _from_json(
            json.loads((path / files["shadow_before"]).read_text(encoding="utf-8"))
        )
    else:
        bundle["shadow_before"] = None
    return bundle


def summary_text(bundle: dict[str, Any], path: Path | None = None) -> str:
    start = bundle["start"]
    div = bundle["divergence"]
    step = bundle.get("step")
    lines = [
        f"jackdaw lockstep divergence ({div['kind']})",
        f"created {bundle['created']} by jackdaw {bundle['jackdaw']}, "
        f"BalatroBot {bundle['balatrobot']}, live {bundle['backend']}",
        f"run: seed {start.get('seed')}, deck {start.get('deck')}, stake {start.get('stake')}"
        + (f", challenge {start['challenge']}" if start.get("challenge") else ""),
        f"steps before the divergence: {sum(e['kind'] != 'reconcile' for e in bundle['log'])}",
    ]
    if step:
        lines.append(f"step: {step['method']} {json.dumps(step.get('params'))}")
        if step.get("action"):
            lines.append(f"action: {step['action']}")
    lines += ["", div["message"], ""]
    if div["diff"]:
        lines.append("unclassified differences (sim vs live):")
        lines += [f"  {FieldDiff.from_json(d)}" for d in div["diff"]]
    for dev_id, diffs in div["classified"].items():
        lines.append(f"classified as {dev_id}:")
        lines += [f"  {FieldDiff.from_json(d)}" for d in diffs]
    if div["candidates"]:
        lines.append("")
        lines.append("catalog entries active at this step:")
        lines += [
            f"  {c['id']} ({c['mode']}): {c['title']} [{c['source']}]" for c in div["candidates"]
        ]
    lines += ["", f"replay: jackdaw lockstep replay {path or '<bundle>'}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayReport:
    """What :func:`replay_bundle` found."""

    bundle: Path
    #: The log replayed (False: ``error`` says where it stopped).
    replayed: bool = False
    error: str = ""
    #: The replayed shadow equals the bundle's snapshot before the divergent step.
    shadow_matches: bool | None = None
    #: How the replayed sim's view differs from the one the bundle recorded.
    sim_changed: list[FieldDiff] = field(default_factory=list)
    #: The sim refused the divergent request now (its message).
    refused: str = ""
    #: The sim's view now against the recorded live view.
    diffs: list[FieldDiff] = field(default_factory=list)
    classification: Classification | None = None
    kind: str = ""
    message: str = ""

    @property
    def reproduced(self) -> bool:
        """The divergence is still there on the current simulator."""
        if self.kind in ("live-refused", "live-error"):
            return not self.refused
        if self.classification is None:
            return bool(self.diffs)
        c = self.classification
        return bool(c.unclassified or c.classified or c.reconcile)

    def exit_code(self) -> int:
        if not self.replayed:
            return 2
        return 1 if self.reproduced else 0

    def lines(self) -> list[str]:
        out = [f"bundle: {self.bundle}", f"recorded: {self.kind}: {self.message}"]
        if not self.replayed:
            out.append(f"replay failed: {self.error}")
            return out
        if self.shadow_matches is not None:
            out.append(
                "the replayed shadow matches the recorded one"
                if self.shadow_matches
                else "the replayed shadow differs from the recorded one (the simulator changed)"
            )
        if self.refused:
            out.append(f"the simulator now refuses the divergent request: {self.refused}")
        if self.sim_changed:
            out.append(f"the simulator's view changed since the bundle ({len(self.sim_changed)}):")
            out += [f"  {d}" for d in self.sim_changed[:40]]
        c = self.classification
        if c is not None:
            if c.unclassified:
                out.append(f"unclassified differences now ({len(c.unclassified)}):")
                out += [f"  {d}" for d in c.unclassified[:40]]
            for dev_id, diffs in {**c.classified, **c.reconcile}.items():
                out.append(f"{dev_id} explains {len(diffs)}:")
                out += [f"  {d}" for d in diffs[:20]]
            for dev_id, diffs in c.ignored.items():
                out.append(f"{dev_id} drops {len(diffs)}")
        out.append(
            "REPRODUCED: the simulator still disagrees with the live game"
            if self.reproduced
            else "NOT REPRODUCED: the simulator now agrees with the live game"
        )
        return out


def replay_bundle(path: str | Path, catalog: tuple[Deviation, ...] = CATALOG) -> ReplayReport:
    """Rerun a bundle on the simulator alone: the shadow's log, then the divergent
    request, compared with the live view the bundle recorded."""
    bundle = read_bundle(path)
    report = ReplayReport(
        Path(path), kind=bundle["divergence"]["kind"], message=bundle["divergence"]["message"]
    )
    start = bundle["start"]
    game = Game.new(
        start["back"] if start["back"] != "b_challenge" else "b_red",
        start["stake_level"],
        start["seed"],
        start.get("challenge"),
    )
    memory = RunMemory()
    for n, entry in enumerate(bundle["log"], 1):
        if entry["kind"] == "reconcile":
            snapshot = game.to_dict()
            apply_edits(snapshot, entry["edits"])
            game.replace_state(snapshot)
            continue
        pre_view = game.bot_view()
        session = _core.BotSession.from_game(game._g)
        ok, text = session.call(entry["method"], json.dumps(entry["params"]))
        if not ok:
            report.error = f"log entry {n} ({entry['method']} {entry['params']}): {text}"
            return report
        post = Game(session.game())
        ctx = StepContext(
            entry["method"],
            entry["params"],
            pre=game,
            pre_view=pre_view,
            post=post,
            post_view=json.loads(text),
            memory=memory,
        )
        update_memory(ctx, catalog)
        game = post
        rec = entry.get("reconcile")
        if rec:
            snapshot = game.to_dict()
            apply_edits(snapshot, rec["edits"])
            game.replace_state(snapshot)
    report.replayed = True
    if bundle.get("shadow_before") is not None:
        recorded = Game.from_dict(bundle["shadow_before"]).state_hash()
        report.shadow_matches = recorded == game.state_hash()
    step = bundle.get("step")
    views = bundle["views"]
    live_view = views.get("live")
    pre_view = game.bot_view()
    ctx = StepContext(
        step["method"] if step else "start",
        step.get("params", {}) if step else {},
        pre=game,
        pre_view=pre_view,
        memory=memory,
    )
    if step is None or step.get("kind") == "check":
        sim_view = pre_view
        ctx.post, ctx.post_view = game, pre_view
    else:
        session = _core.BotSession.from_game(game._g)
        ok, text = session.call(step["method"], json.dumps(step["params"]))
        if not ok:
            report.refused = json.loads(text).get("message", text)
            return report
        sim_view = json.loads(text)
        ctx.post, ctx.post_view = Game(session.game()), sim_view
    if views.get("sim") is not None:
        report.sim_changed = diff_views(sim_view, views["sim"]).diffs
    if live_view is not None:
        ids = IdMap.from_json(bundle.get("ids", []))
        diff = diff_views(sim_view, live_view, ids)
        report.diffs = diff.diffs
        report.classification = classify(diff.diffs, ctx, catalog)
    return report
