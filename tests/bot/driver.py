"""Drive a ``jackdaw._core.BotSession`` the way a BalatroBot client would."""

from __future__ import annotations

import json
import random
from typing import Any

from jackdaw import _core


class BotCallError(Exception):
    def __init__(self, method: str, params: dict[str, Any], error: dict[str, Any]) -> None:
        super().__init__(f"{method} {params}: {error['name']}: {error['message']}")
        self.error = error


def call(session: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    ok, text = session.call(method, json.dumps(params or {}))
    data = json.loads(text)
    if not ok:
        raise BotCallError(method, params or {}, data)
    return data


def request(session: Any, action: list[Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    """The request for a tape-form action; a ``PlayHand`` / ``Discard`` marker gets a
    random selection (with Cerulean Bell's forced card)."""
    out = session.translate(json.dumps(action))
    assert out is not None, action
    method, params_text = out
    params = json.loads(params_text)
    if action[0] in ("PlayHand", "Discard") and not action[1]:
        view = json.loads(session.view())
        cards = view["hand"]["cards"]
        k = rng.randint(1, min(5, len(cards)))
        picked = sorted(rng.sample(range(len(cards)), k))
        forced = params["cards"][0]
        if cards[forced].get("state", {}) and cards[forced]["state"].get("highlight"):
            picked = sorted({*picked[:4], forced})
        params["cards"] = picked
    return method, params


def _boost(session: Any, view: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """``set`` the round's chips to the blind's score (so the next hand wins) or the money
    to 30, now and then, so random play gets past the first blind and into shops."""
    if view["state"] == "SELECTING_HAND" and rng.random() < 0.3:
        current = [b for b in view["blinds"].values() if b["status"] == "CURRENT"]
        if current and view["round"]["chips"] < current[0]["score"]:
            return call(session, "set", {"chips": current[0]["score"]})
    if view["state"] == "SHOP" and view["money"] < 30 and rng.random() < 0.3:
        return call(session, "set", {"money": 30})
    return None


def random_run(
    seed: str,
    rng: random.Random,
    *,
    deck: str = "RED",
    stake: str = "WHITE",
    max_steps: int = 300,
    boost: bool = True,
    on_view: Any = None,
) -> Any:
    """Play a run with random BALATROBOT-legal requests; ``on_view(method, view)`` sees
    every response. With ``boost``, ``set`` now and then beats a blind or tops up the
    money. Returns the session."""
    session = _core.BotSession()
    view = call(session, "start", {"deck": deck, "stake": stake, "seed": seed})
    if on_view:
        on_view("start", view)
    for _ in range(max_steps):
        if view["state"] == "GAME_OVER" or view.get("won"):
            break
        if boost:
            boosted = _boost(session, view, rng)
            if boosted is not None:
                view = boosted
                if on_view:
                    on_view("set", view)
        legal = json.loads(session.legal_actions("balatrobot"))
        if not legal:
            break
        action = rng.choice(legal)
        method, params = request(session, action, rng)
        try:
            view = call(session, method, params)
        except BotCallError as exc:
            # A random selection can break a vanilla rule (e.g. The Psychic's five cards).
            if method in ("play", "discard"):
                continue
            raise AssertionError(f"{action}: {exc}") from exc
        if on_view:
            on_view(method, view)
    return session
