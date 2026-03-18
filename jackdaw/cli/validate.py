"""Validation subcommands — crash testing, benchmarking, and live comparison.

Replaces ``scripts/validate.py`` with proper CLI integration.

Usage::

    jackdaw validate crash --count 200
    jackdaw validate benchmark --count 1000
    jackdaw validate seed --seed TESTSEED --seeds 5
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any

from jackdaw.bridge.backend import SimBackend
from jackdaw.engine.runner import greedy_play_agent, random_agent, simulate_run

# ---------------------------------------------------------------------------
# Agent dispatch
# ---------------------------------------------------------------------------

_AGENTS = {
    "random": random_agent,
}

# ---------------------------------------------------------------------------
# Crash testing
# ---------------------------------------------------------------------------


def run_crash(count: int, agent_name: str) -> int:
    """Run *count* simulated games, report crashes and stats.

    Returns 0 on success (no crashes, deterministic), 1 otherwise.
    """
    agent = _AGENTS[agent_name]
    wins = 0
    crashes = 0
    total_actions = 0
    total_rounds = 0
    max_actions = 0

    # -- Bridge smoke test ---------------------------------------------------
    print("Bridge smoke test (1 game through SimBackend.handle)...")
    bridge_ok = _bridge_smoke_test()
    print(f"  Bridge: {'OK' if bridge_ok else 'FAIL'}")

    # -- Engine crash test ---------------------------------------------------
    print(f"\nRunning {count} games with {agent_name} agent...")
    t0 = time.time()
    for i in range(count):
        try:
            gs = simulate_run(
                "b_red", 1, f"CRASH_{agent_name}_{i}",
                agent, max_actions=2000,
            )
            if gs.get("won"):
                wins += 1
            total_actions += gs["actions_taken"]
            total_rounds += gs.get("round", 0)
            max_actions = max(max_actions, gs["actions_taken"])
        except Exception as e:
            crashes += 1
            print(f"  CRASH at seed {i}: {e}")
    elapsed = time.time() - t0
    runs_per_sec = count / max(elapsed, 0.001)

    # -- Determinism check ---------------------------------------------------
    print("\nDeterminism check...")
    gs1 = simulate_run("b_red", 1, "DETERMINISM_CHECK", greedy_play_agent, max_actions=500)
    gs2 = simulate_run("b_red", 1, "DETERMINISM_CHECK", greedy_play_agent, max_actions=500)
    deterministic = (
        gs1.get("dollars") == gs2.get("dollars")
        and gs1.get("round") == gs2.get("round")
        and gs1.get("won") == gs2.get("won")
    )
    print(f"  Same seed -> same result: {deterministic}")

    # -- Results -------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Runs:        {count}")
    print(f"Wins:        {wins}")
    print(f"Crashes:     {crashes}")
    print(f"Avg actions: {total_actions / max(count, 1):.1f}")
    print(f"Max actions: {max_actions}")
    print(f"Avg rounds:  {total_rounds / max(count, 1):.1f}")
    print(f"Time:        {elapsed:.2f}s ({runs_per_sec:.0f} runs/sec)")

    ok = crashes == 0 and deterministic and bridge_ok
    print(
        f"\n{'PASS' if ok else 'FAIL'}: {crashes} crashes, "
        f"determinism={'OK' if deterministic else 'FAIL'}, "
        f"bridge={'OK' if bridge_ok else 'FAIL'}"
    )
    return 0 if ok else 1


def _bridge_smoke_test() -> bool:
    """Run one game entirely through SimBackend.handle() calls."""
    from jackdaw.engine.actions import GamePhase, get_legal_actions

    try:
        backend = SimBackend()
        backend.handle("health", None)
        backend.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "BRIDGE_SMOKE"})

        gs = backend._gs
        assert gs is not None
        actions_taken = 0
        max_actions = 500

        while actions_taken < max_actions:
            phase = gs.get("phase")
            if phase == GamePhase.GAME_OVER:
                break
            if gs.get("won") and phase == GamePhase.SHOP:
                break

            legal = get_legal_actions(gs)
            if not legal:
                break

            action = greedy_play_agent(gs, legal)

            # Route through handle() to exercise the bridge
            from jackdaw.bridge.balatrobot_adapter import action_to_rpc

            rpc_call = action_to_rpc(action)
            backend.handle(rpc_call["method"], rpc_call.get("params"))
            actions_taken += 1

        # Verify gamestate serialization works
        backend.handle("gamestate", None)
        backend.handle("menu", None)
        return True
    except Exception as e:
        print(f"  Bridge smoke test failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def run_benchmark(count: int) -> None:
    """Pure performance measurement using simulate_run() directly."""
    agent = random_agent

    print(f"Benchmarking {count} runs...")
    t0 = time.time()
    total_actions = 0
    for i in range(count):
        gs = simulate_run("b_red", 1, f"BENCH_{i}", agent, max_actions=2000)
        total_actions += gs["actions_taken"]
    elapsed = time.time() - t0

    runs_per_sec = count / max(elapsed, 0.001)
    actions_per_sec = total_actions / max(elapsed, 0.001)

    print(f"\n{'=' * 60}")
    print(f"Runs:            {count}")
    print(f"Total actions:   {total_actions}")
    print(f"Avg actions/run: {total_actions / count:.1f}")
    print(f"Time:            {elapsed:.2f}s")
    print(f"Runs/sec:        {runs_per_sec:.0f}")
    print(f"Actions/sec:     {actions_per_sec:.0f}")


# ---------------------------------------------------------------------------
# Seed validation (sim vs live balatrobot)
# ---------------------------------------------------------------------------

# Reverse maps: engine key → balatrobot enum name
_DECK_MAP: dict[str, str] = {
    "b_red": "RED", "b_blue": "BLUE", "b_yellow": "YELLOW",
    "b_green": "GREEN", "b_black": "BLACK", "b_magic": "MAGIC",
    "b_nebula": "NEBULA", "b_ghost": "GHOST", "b_abandoned": "ABANDONED",
    "b_checkered": "CHECKERED", "b_zodiac": "ZODIAC", "b_painted": "PAINTED",
    "b_anaglyph": "ANAGLYPH", "b_plasma": "PLASMA", "b_erratic": "ERRATIC",
}

_STAKE_MAP: dict[int, str] = {
    1: "WHITE", 2: "RED", 3: "GREEN", 4: "BLACK",
    5: "BLUE", 6: "PURPLE", 7: "ORANGE", 8: "GOLD",
}


def _pick_best_hand(hand_cards: list[Any], jokers: list[Any]) -> list[int]:
    """Pick the best 5-card (or fewer) subset from hand_cards by brute force.

    Returns indices into hand_cards. With 8 cards this is C(8,5)=56 combos.
    """
    from itertools import combinations

    from jackdaw.engine.data.hands import HAND_ORDER, HandType
    from jackdaw.engine.hand_eval import evaluate_hand

    n = len(hand_cards)
    if n <= 5:
        return list(range(n))

    best_indices: list[int] = list(range(min(5, n)))
    best_rank = len(HAND_ORDER)  # worst possible
    best_card_sum = 0

    for combo in combinations(range(n), min(5, n)):
        cards = [hand_cards[i] for i in combo]
        result = evaluate_hand(cards, jokers)
        try:
            rank = HAND_ORDER.index(HandType(result.detected_hand))
        except (ValueError, KeyError):
            rank = len(HAND_ORDER)
        card_sum = sum(c.get_chip_bonus() for c in cards if hasattr(c, "get_chip_bonus"))
        if rank < best_rank or (rank == best_rank and card_sum > best_card_sum):
            best_rank = rank
            best_indices = list(combo)
            best_card_sum = card_sum

    return best_indices


# ---------------------------------------------------------------------------
# Validation agent — exercises diverse game actions
# ---------------------------------------------------------------------------


def _validation_agent(gs: dict[str, Any], legal_actions: list[Any]) -> Any:
    """Pick a smart action that exercises diverse code paths.

    Priority order per phase:
    - BLIND_SELECT: always select (never skip — we want to play rounds)
    - SELECTING_HAND: play best hand; discard once per round if available
    - ROUND_EVAL: cash out
    - SHOP: buy cheapest affordable card, reroll once, open a booster, then next round
    - PACK_OPENING: pick first card, or skip if no picks left
    """
    from jackdaw.engine.actions import (
        BuyCard,
        CashOut,
        Discard,
        GamePhase,
        NextRound,
        OpenBooster,
        PickPackCard,
        PlayHand,
        Reroll,
        SelectBlind,
        SkipPack,
    )

    phase = gs.get("phase")
    if isinstance(phase, str):
        phase = GamePhase(phase)

    # -- BLIND_SELECT: always select
    if phase == GamePhase.BLIND_SELECT:
        for a in legal_actions:
            if isinstance(a, SelectBlind):
                return a
        return legal_actions[0]

    # -- SELECTING_HAND: play best hand, discard first round if possible
    if phase == GamePhase.SELECTING_HAND:
        hand = gs.get("hand", [])
        jokers = gs.get("jokers", [])
        cr = gs.get("current_round", {})

        # Discard once at the start of each blind (exercises discard path)
        if cr.get("hands_played", 0) == 0 and cr.get("discards_used", 0) == 0:
            for a in legal_actions:
                if isinstance(a, Discard) and not a.card_indices and hand:
                    # Discard the worst 2 cards (lowest chip value)
                    ranked = sorted(range(len(hand)),
                                    key=lambda i: hand[i].get_chip_bonus()
                                    if hasattr(hand[i], "get_chip_bonus") else 0)
                    n_discard = min(2, len(ranked))
                    return Discard(card_indices=tuple(sorted(ranked[:n_discard])))

        # Play best hand
        for a in legal_actions:
            if isinstance(a, PlayHand) and not a.card_indices and hand:
                best = _pick_best_hand(hand, jokers)
                return PlayHand(card_indices=tuple(best))

        return legal_actions[0]

    # -- ROUND_EVAL: cash out
    if phase == GamePhase.ROUND_EVAL:
        for a in legal_actions:
            if isinstance(a, CashOut):
                return a
        return legal_actions[0]

    # -- SHOP: buy, reroll, open boosters, then leave
    if phase == GamePhase.SHOP:
        # Buy cheapest affordable shop card (joker or consumable)
        buy_actions = [a for a in legal_actions if isinstance(a, BuyCard)]
        if buy_actions:
            shop_cards = gs.get("shop_cards", [])
            cheapest = min(buy_actions, key=lambda a: (
                shop_cards[a.shop_index].cost if a.shop_index < len(shop_cards) else 999
            ))
            if cheapest.shop_index < len(shop_cards):
                card = shop_cards[cheapest.shop_index]
                if card.cost <= gs.get("dollars", 0):
                    return cheapest

        # Open a booster pack if we can afford one
        for a in legal_actions:
            if isinstance(a, OpenBooster):
                boosters = gs.get("shop_boosters", [])
                if a.card_index < len(boosters):
                    if boosters[a.card_index].cost <= gs.get("dollars", 0):
                        return a

        # Reroll once per shop visit (if we haven't bought anything yet)
        cr = gs.get("current_round", {})
        if cr.get("jokers_purchased", 0) == 0:
            for a in legal_actions:
                if isinstance(a, Reroll):
                    cost = cr.get("reroll_cost", 5)
                    if gs.get("dollars", 0) >= cost:
                        return a

        # Leave shop
        for a in legal_actions:
            if isinstance(a, NextRound):
                return a

        return legal_actions[0]

    # -- PACK_OPENING: pick first card or skip
    if phase == GamePhase.PACK_OPENING:
        for a in legal_actions:
            if isinstance(a, PickPackCard):
                return a
        for a in legal_actions:
            if isinstance(a, SkipPack):
                return a
        return legal_actions[0]

    # Fallback
    return legal_actions[0]


def _action_to_rpc_call(action: Any) -> dict[str, Any]:
    """Convert an engine Action to an RPC method+params dict."""
    from jackdaw.bridge.balatrobot_adapter import action_to_rpc
    return action_to_rpc(action)


def _action_description(action: Any) -> str:
    """Short human-readable description of an action."""
    from jackdaw.engine.actions import (
        BuyCard,
        CashOut,
        Discard,
        NextRound,
        OpenBooster,
        PickPackCard,
        PlayHand,
        Reroll,
        SelectBlind,
        SkipBlind,
        SkipPack,
    )

    match action:
        case SelectBlind():
            return "Select Blind"
        case SkipBlind():
            return "Skip Blind"
        case PlayHand(card_indices=idx):
            return f"Play {list(idx)}"
        case Discard(card_indices=idx):
            return f"Discard {list(idx)}"
        case CashOut():
            return "Cash Out"
        case NextRound():
            return "Next Round"
        case BuyCard(shop_index=i):
            return f"Buy card {i}"
        case OpenBooster(card_index=i):
            return f"Open booster {i}"
        case PickPackCard(card_index=i):
            return f"Pick pack card {i}"
        case SkipPack():
            return "Skip Pack"
        case Reroll():
            return "Reroll"
        case _:
            return type(action).__name__


def _card_keys(area: dict | list) -> list[str]:
    """Extract card keys from a serialized area (dict with 'cards' or list)."""
    if isinstance(area, dict):
        return [c.get("key", "") for c in area.get("cards", [])]
    return []


_PACK_VARIANT_RE = re.compile(r"^(p_\w+_normal_)\d+$")


def _is_pack_variant_diff(diff: str) -> bool:
    """Check if a diff is just a booster pack variant difference (_1 vs _2).

    The real game uses math.random (not pseudoseed) to pick between pack
    variants like p_buffoon_normal_1 and p_buffoon_normal_2. These are
    functionally identical — only the sprite differs.
    """
    if not diff.startswith("packs:"):
        return False
    # Extract the two lists from the diff string and normalize variants
    # e.g. "packs: sim=['p_buffoon_normal_1', ...] live=['p_buffoon_normal_2', ...]"
    def normalize_pack_key(key: str) -> str:
        m = _PACK_VARIANT_RE.match(key)
        return f"{m.group(1)}X" if m else key

    # Parse the sim= and live= parts
    try:
        sim_part = diff.split("sim=")[1].split(" live=")[0]
        live_part = diff.split("live=")[1]
        sim_keys = eval(sim_part)  # noqa: S307
        live_keys = eval(live_part)  # noqa: S307
        sim_norm = [normalize_pack_key(k) for k in sim_keys]
        live_norm = [normalize_pack_key(k) for k in live_keys]
        return sim_norm == live_norm
    except Exception:
        return False


def _compare_responses(
    sim_resp: dict,
    live_resp: dict,
    label: str,
    *,
    action_desc: str = "",
    sim_rpc: dict | None = None,
    live_rpc: dict | None = None,
) -> list[str]:
    """Compare serialized sim and live responses, return list of diffs.

    On divergence, prints detailed debug info including the action that
    caused it and the full state of diverging fields.
    """
    diffs: list[str] = []

    def cmp(name: str, s: Any, live: Any) -> None:
        if s != live:
            diffs.append(f"{name}: sim={s} live={live}")

    cmp("state", sim_resp.get("state", ""), live_resp.get("state", ""))
    cmp("money", sim_resp.get("money", 0), live_resp.get("money", 0))
    cmp("ante_num", sim_resp.get("ante_num", 1), live_resp.get("ante_num", 1))

    sim_round = sim_resp.get("round", {})
    live_round = live_resp.get("round", {})

    # Round fields: compare during active play and round_eval (used for earnings).
    # Skip only in SHOP and BLIND_SELECT where they're truly stale.
    state = sim_resp.get("state", "")
    if state in ("SELECTING_HAND", "ROUND_EVAL"):
        cmp("chips", sim_round.get("chips", 0), live_round.get("chips", 0))
        cmp("hands_left", sim_round.get("hands_left", 0), live_round.get("hands_left", 0))
        cmp("discards_left", sim_round.get("discards_left", 0), live_round.get("discards_left", 0))

    # Hand cards (as sets — display order may differ)
    sim_hand_keys = _card_keys(sim_resp.get("hand", {}))
    live_hand_keys = _card_keys(live_resp.get("hand", {}))
    cmp("hand_cards", set(sim_hand_keys), set(live_hand_keys))

    # Deck size
    cmp("deck_size", sim_resp.get("cards", {}).get("count", 0),
        live_resp.get("cards", {}).get("count", 0))

    # Jokers (ordered — position matters for scoring)
    cmp("jokers", _card_keys(sim_resp.get("jokers", {})),
        _card_keys(live_resp.get("jokers", {})))

    # Consumables (as sets)
    cmp("consumables", set(_card_keys(sim_resp.get("consumables", {}))),
        set(_card_keys(live_resp.get("consumables", {}))))

    # Shop (when in SHOP phase)
    if state == "SHOP":
        cmp("shop", _card_keys(sim_resp.get("shop", {})),
            _card_keys(live_resp.get("shop", {})))
        cmp("vouchers", _card_keys(sim_resp.get("vouchers", {})),
            _card_keys(live_resp.get("vouchers", {})))
        cmp("packs", _card_keys(sim_resp.get("packs", {})),
            _card_keys(live_resp.get("packs", {})))

    # Pack (when in pack opening)
    if state == "SMODS_BOOSTER_OPENED":
        cmp("pack_cards", _card_keys(sim_resp.get("pack", {})),
            _card_keys(live_resp.get("pack", {})))

    # Filter out known non-determinism: Buffoon pack variant (_1 vs _2)
    # The real game uses math.random (not pseudoseed) to pick the variant.
    diffs = [d for d in diffs if not _is_pack_variant_diff(d)]

    # -- Output --
    if not diffs:
        print(f"  [{label}] OK")
        return diffs

    print(f"  [{label}] DIVERGED ({len(diffs)} diffs)")
    for d in diffs:
        print(f"    {d}")

    # -- Debug dump on divergence --
    if action_desc:
        print(f"    action: {action_desc}")
    if sim_rpc:
        print(f"    sim_rpc:  {sim_rpc['method']} {sim_rpc.get('params', {})}")
    if live_rpc:
        print(f"    live_rpc: {live_rpc['method']} {live_rpc.get('params', {})}")

    # Show full hand lists (ordered) so we can see ordering differences
    if set(sim_hand_keys) != set(live_hand_keys):
        only_sim = set(sim_hand_keys) - set(live_hand_keys)
        only_live = set(live_hand_keys) - set(sim_hand_keys)
        if only_sim:
            print(f"    hand only in sim:  {sorted(only_sim)}")
        if only_live:
            print(f"    hand only in live: {sorted(only_live)}")
        print(f"    sim hand order:  {sim_hand_keys}")
        print(f"    live hand order: {live_hand_keys}")

    # Show deck first-10 on deck_size mismatch
    if sim_resp.get("cards", {}).get("count", 0) != live_resp.get("cards", {}).get("count", 0):
        sim_dk = _card_keys(sim_resp.get("cards", {}))[:10]
        live_dk = _card_keys(live_resp.get("cards", {}))[:10]
        print(f"    sim deck (first 10):  {sim_dk}")
        print(f"    live deck (first 10): {live_dk}")

    # Show round details (only when round diffs are being compared)
    if state == "SELECTING_HAND" and any(
        d.startswith(("chips:", "hands_left:", "discards_left:")) for d in diffs
    ):
        print(f"    sim round:  {sim_round}")
        print(f"    live round: {live_round}")

    return diffs


def _translate_action_for_live(
    action: Any,
    gs: dict[str, Any],
    live_resp: dict,
) -> dict[str, Any]:
    """Convert a sim action to RPC params, translating card indices for live.

    For PlayHand/Discard, the sim picks indices into its own hand. The live
    hand may have the same cards in a different order, so we translate by
    matching card keys.
    """
    from jackdaw.engine.actions import Discard, PlayHand

    rpc_call = _action_to_rpc_call(action)

    # Translate card indices for play/discard
    if isinstance(action, (PlayHand, Discard)) and action.card_indices:
        sim_hand = gs.get("hand", [])
        live_hand = live_resp.get("hand", {}).get("cards", [])
        live_keys = [c.get("key", "") for c in live_hand]

        # Get the card keys the sim is playing
        play_keys = [
            sim_hand[i].card_key for i in action.card_indices
            if i < len(sim_hand) and hasattr(sim_hand[i], "card_key")
        ]

        # Find those keys in the live hand
        live_indices: list[int] = []
        used: set[int] = set()
        for key in play_keys:
            for i, lk in enumerate(live_keys):
                if i not in used and lk == key:
                    live_indices.append(i)
                    used.add(i)
                    break

        if len(live_indices) == len(play_keys):
            rpc_call["params"]["cards"] = live_indices
        # else: keep original indices as fallback

    return rpc_call


def _seed_run_validation(
    sim: SimBackend,
    live_handle: Any,
    seed: str,
    back_key: str,
    stake: int,
) -> dict[str, Any]:
    """Play a full run using the validation agent, comparing state after each action."""
    from jackdaw.engine.actions import GamePhase, get_legal_actions

    print(f"\n{'=' * 60}")
    print(f"Seed: {seed}  back={back_key}  stake={stake}")
    print(f"{'=' * 60}")

    deck_name = _DECK_MAP.get(back_key, "RED")
    stake_name = _STAKE_MAP.get(stake, "WHITE")

    # Start both
    live_handle("menu", None)
    time.sleep(1.5)
    start_params = {"deck": deck_name, "stake": stake_name, "seed": seed}
    sim.handle("start", start_params)
    live_handle("start", start_params)
    time.sleep(0.5)

    all_diffs: list[list[str]] = []
    step_count = 0
    max_actions = 500
    last_phase = ""

    # Compare initial state
    sim_resp = sim.handle("gamestate", None)
    live_resp = live_handle("gamestate", None)
    init_diffs = _compare_responses(sim_resp, live_resp, "init")
    all_diffs.append(init_diffs)
    if init_diffs:
        return _seed_summary(seed, all_diffs, step_count, "diverged_at_init")

    gs = sim._gs
    assert gs is not None

    while step_count < max_actions:
        phase = gs.get("phase")
        if isinstance(phase, str):
            phase = GamePhase(phase)

        if phase == GamePhase.GAME_OVER:
            won = gs.get("won", False)
            print(f"  {'WIN' if won else 'GAME OVER'} at step {step_count}")
            break

        if gs.get("won") and phase == GamePhase.SHOP:
            print(f"  WIN (in shop) at step {step_count}")
            break

        # Log phase transitions
        phase_str = phase.value if hasattr(phase, "value") else str(phase)
        if phase_str != last_phase:
            blind_name = gs.get("blind_on_deck", "")
            ante = gs.get("round_resets", {}).get("ante", 1)
            print(f"\n--- {phase_str} (ante {ante}, {blind_name}) ---")
            last_phase = phase_str

        legal = get_legal_actions(gs)
        if not legal:
            print(f"  No legal actions at step {step_count}")
            break

        action = _validation_agent(gs, legal)
        desc = _action_description(action)

        # Convert to RPC for both sides
        live_resp_now = live_handle("gamestate", None)
        live_rpc = _translate_action_for_live(action, gs, live_resp_now)
        sim_rpc = _action_to_rpc_call(action)

        # Execute on both
        try:
            sim.handle(sim_rpc["method"], sim_rpc.get("params"))
        except Exception as e:
            print(f"  SIM ERROR on {desc}: {e}")
            print(f"    sim_rpc:  {sim_rpc['method']} {sim_rpc.get('params', {})}")
            print(f"    live_rpc: {live_rpc['method']} {live_rpc.get('params', {})}")
            return _seed_summary(seed, all_diffs, step_count, f"sim_error:{desc}")

        try:
            live_handle(live_rpc["method"], live_rpc.get("params"))
        except Exception as e:
            print(f"  LIVE ERROR on {desc}: {e}")
            print(f"    sim_rpc:  {sim_rpc['method']} {sim_rpc.get('params', {})}")
            print(f"    live_rpc: {live_rpc['method']} {live_rpc.get('params', {})}")
            return _seed_summary(seed, all_diffs, step_count, f"live_error:{desc}")

        # Longer delays for actions that produce visible animations so you
        # can watch the game in the balatrobot window.
        method = sim_rpc["method"]
        if method == "play":
            time.sleep(2.5)     # scoring animation
        elif method == "cash_out":
            time.sleep(3.0)     # round earnings scoreboard
        elif method in ("select", "buy", "reroll"):
            time.sleep(1.0)     # card movement animation
        elif method == "pack":
            time.sleep(1.5)     # pack opening animation
        else:
            time.sleep(0.3)

        # Compare after action
        sim_resp = sim.handle("gamestate", None)
        live_resp = live_handle("gamestate", None)
        diffs = _compare_responses(
            sim_resp, live_resp, f"{step_count}:{desc}",
            action_desc=desc, sim_rpc=sim_rpc, live_rpc=live_rpc,
        )
        all_diffs.append(diffs)
        step_count += 1

        if diffs:
            print(f"\n  Stopping seed {seed} — states diverged at step {step_count - 1}")
            return _seed_summary(seed, all_diffs, step_count, f"diverged:{desc}")

    status = "complete" if step_count < max_actions else "max_actions"
    if gs.get("phase") == GamePhase.GAME_OVER or (gs.get("phase") == "game_over"):
        status = "win" if gs.get("won") else "game_over"
    return _seed_summary(seed, all_diffs, step_count, status)


def _seed_summary(
    seed: str, all_diffs: list[list[str]], step_count: int, status: str,
) -> dict[str, Any]:
    total_diffs = sum(len(d) for d in all_diffs)
    clean = sum(1 for d in all_diffs if not d)
    return {
        "seed": seed,
        "steps": step_count,
        "total_diffs": total_diffs,
        "clean_steps": clean,
        "total_steps": len(all_diffs),
        "status": status,
    }


def run_seed(
    seed_base: str,
    num_seeds: int,
    back: str,
    stake: int,
    host: str,
    port: int,
) -> int:
    """Compare sim vs live balatrobot side-by-side.

    Returns 0 if all seeds clean, 1 otherwise.
    """
    from jackdaw.bridge.backend import LiveBackend, RPCError

    live_backend = LiveBackend(host=host, port=port)

    # Check balatrobot reachable
    try:
        live_backend.handle("health", None)
    except Exception as e:
        print(f"Cannot reach balatrobot at http://{host}:{port}: {e}")
        print("Start it with: uvx balatrobot serve --fast --no-audio --love-path <path>")
        return 1

    results: list[dict[str, Any]] = []
    for i in range(num_seeds):
        seed = f"{seed_base}{i}" if num_seeds > 1 else seed_base
        sim = SimBackend()
        result = _seed_run_validation(sim, live_backend.handle, seed, back, stake)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r["total_diffs"] == 0 else "FAIL"
        print(
            f"  {r['seed']}: {status} ({r['clean_steps']}/{r['total_steps']} clean, "
            f"{r['total_diffs']} diffs, {r['status']})"
        )

    total_clean = sum(r["total_diffs"] == 0 for r in results)
    print(f"\nResult: {total_clean}/{len(results)} seeds fully clean")
    return 0 if total_clean == len(results) else 1
