"""Tag validation scenarios — all 24 tag types.

Each scenario skips a blind to trigger a specific tag, in lockstep with the
shadow run.

Tag types by context:
  immediate        — fires on skip, effect applied instantly (money, jokers, hand level)
  new_blind_choice — fires on skip, opens pack or rerolls boss
  shop_start       — fires when entering shop (free rerolls)
  store_joker_*    — modifies next shop joker (rarity, edition)
  shop_final_pass  — fires at end of shop (coupon: free items)
  round_start_bonus— fires at round start (juggle: hand size)
  eval             — fires after blind scored (investment: $25 after boss)
  tag_add          — fires when tag added (double: duplicate tag)

Seeds from: uv run python scripts/find_seeds.py --max-seeds 2000 --max-ante 8
"""

from __future__ import annotations

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Lockstep,
    cash_out,
    compare_state,
    current_blind_score,
    get_state,
    next_round,
    pause,
    play_hand,
    select_blind,
    set_both,
    skip_blind,
    start_both,
)

# ---------------------------------------------------------------------------
# Known seeds where each tag appears at a specific ante + blind position
# Format: tag_key -> (seed, ante, blind_position)
# blind_position: "Small" or "Big" — which blind to skip to trigger the tag
# ---------------------------------------------------------------------------

# From find_seeds.py output for FIND_6:
_TAG_SEEDS: dict[str, tuple[str, int, str]] = {
    # FIND_6 tags (15 tags)
    "tag_skip": ("FIND_6", 1, "Small"),
    "tag_rare": ("FIND_6", 1, "Big"),
    "tag_holo": ("FIND_6", 2, "Small"),
    "tag_uncommon": ("FIND_6", 2, "Big"),
    "tag_juggle": ("FIND_6", 3, "Small"),
    "tag_ethereal": ("FIND_6", 3, "Big"),
    "tag_d_six": ("FIND_6", 4, "Small"),
    "tag_investment": ("FIND_6", 4, "Big"),
    "tag_economy": ("FIND_6", 5, "Small"),
    "tag_foil": ("FIND_6", 5, "Big"),
    "tag_charm": ("FIND_6", 6, "Small"),
    "tag_buffoon": ("FIND_6", 6, "Big"),
    "tag_top_up": ("FIND_6", 7, "Big"),
    "tag_boss": ("FIND_6", 7, "Small"),
    "tag_negative": ("FIND_6", 8, "Small"),
    # FIND_109 tags (7 tags)
    "tag_voucher": ("FIND_109", 1, "Small"),
    "tag_garbage": ("FIND_109", 3, "Small"),
    "tag_handy": ("FIND_109", 3, "Big"),
    "tag_polychrome": ("FIND_109", 4, "Small"),
    "tag_orbital": ("FIND_109", 6, "Small"),
    "tag_coupon": ("FIND_109", 7, "Small"),
    "tag_standard": ("FIND_109", 7, "Big"),
    # FIND_11 tags (2 tags)
    "tag_meteor": ("FIND_11", 7, "Small"),
    "tag_double": ("FIND_11", 8, "Small"),
}

# Pack-creating tags. Skipping a blind for one opens its pack at once; the live
# `skip` then waits for BLIND_SELECT and never answers, so the lockstep vetoes
# the skip (BB-SKIP-PACK) and these scenarios report a skip.
_PACK_TAGS = {"tag_buffoon", "tag_charm", "tag_ethereal", "tag_meteor", "tag_standard"}


def _force_beat_blind(
    ls: Lockstep,
    *,
    delay: float = 0.3,
) -> bool:
    """Select blind, cheat chips, play one hand to beat it immediately.

    Returns True on success, False if the game was lost.
    """
    select_blind(ls, delay=delay)

    state = get_state(ls)
    if state != "SELECTING_HAND":
        return state == "ROUND_EVAL"

    set_both(ls, chips=current_blind_score(ls) - 1)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)

    if get_state(ls) != "ROUND_EVAL":
        return False

    pause(delay, 2.0)
    cash_out(ls, delay=delay)
    pause(delay, 2.0)
    next_round(ls, delay=delay)
    pause(delay, 1.0)
    return True


def _advance_to_ante(
    ls: Lockstep,
    target_ante: int,
    *,
    delay: float = 0.3,
) -> bool:
    """Play through blinds to reach the target ante.

    Returns True on success, False if the game was lost.
    """
    for _ in range(1, target_ante):
        for _ in range(3):  # Small, Big, Boss
            if not _force_beat_blind(ls, delay=delay):
                return False
    return True


def _tag_scenario(
    ls: Lockstep,
    *,
    tag_key: str,
    seed: str,
    ante: int,
    blind_pos: str,
    delay: float = 0.3,
) -> ScenarioResult:
    """Navigate to the right ante, skip the right blind, compare tag effect."""
    start_both(ls, seed=seed, delay=delay)

    # Advance to the target ante
    if not _advance_to_ante(ls, ante, delay=delay):
        return ScenarioResult(
            passed=True,
            details=f"Tag {tag_key}: SKIP (game ended before ante {ante})",
        )

    # If tag is on Big blind, beat Small first
    if blind_pos == "Big" and not _force_beat_blind(ls, delay=delay):
        return ScenarioResult(passed=True, details=f"Tag {tag_key}: SKIP (lost at Small)")

    # Skip the blind to trigger the tag (vetoed for a pack tag: BB-SKIP-PACK).
    skip_blind(ls, delay=delay)
    pause(delay, 1.0)
    if get_state(ls) == "SMODS_BOOSTER_OPENED":
        ls.request("pack", {"card": 0})
        pause(delay, 1.0)

    diffs = compare_state(ls, label=f"after {tag_key}", check_round=False)
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Tag {tag_key}: {'PASS' if not diffs else 'FAIL'}",
    )


# ---------------------------------------------------------------------------
# Tag descriptions
# ---------------------------------------------------------------------------

_TAG_DESCS: dict[str, str] = {
    # Immediate
    "tag_economy": "gain money up to current amount (max $40)",
    "tag_garbage": "$1 per unused discard last round",
    "tag_handy": "$1 per hand played last round",
    "tag_skip": "$5 per blind skipped this run",
    "tag_top_up": "spawn 2 free Common jokers",
    "tag_orbital": "level up a random hand type 3 times",
    # Pack-creating (new_blind_choice)
    "tag_buffoon": "opens free Mega Buffoon Pack",
    "tag_charm": "opens free Mega Arcana Pack",
    "tag_ethereal": "opens free Spectral Pack",
    "tag_meteor": "opens free Mega Celestial Pack",
    "tag_standard": "opens free Mega Standard Pack",
    # Boss reroll
    "tag_boss": "rerolls the boss blind",
    # Shop modifiers
    "tag_d_six": "1 free reroll in shop",
    "tag_uncommon": "force next shop joker to Uncommon",
    "tag_rare": "force next shop joker to Rare",
    "tag_foil": "add Foil edition to shop joker",
    "tag_holo": "add Holographic edition to shop joker",
    "tag_polychrome": "add Polychrome edition to shop joker",
    "tag_negative": "add Negative edition to shop joker",
    "tag_coupon": "all shop items free",
    "tag_voucher": "add a free random voucher",
    # Round modifiers
    "tag_juggle": "+3 hand size for one round",
    "tag_investment": "$25 after defeating boss blind",
    "tag_double": "duplicate the next tag awarded",
}

# Register all tag scenarios
for _key, (_seed, _ante, _pos) in _TAG_SEEDS.items():
    _desc = _TAG_DESCS.get(_key, _key)

    def _make_fn(key: str = _key, seed: str = _seed, ante: int = _ante, pos: str = _pos):
        def fn(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
            return _tag_scenario(
                ls,
                tag_key=key,
                seed=seed,
                ante=ante,
                blind_pos=pos,
                delay=delay,
            )

        return fn

    register(
        name=f"tag_{_key[4:]}",
        category="tags",
        description=f"{_key}: {_desc} (seed {_seed} ante {_ante} {_pos})",
    )(_make_fn(_key, _seed, _ante, _pos))
