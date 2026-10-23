#!/usr/bin/env python3
"""Generate and check the Balatro 1.0.1o parity ledger (``docs/parity/``).

The ledger is the checklist that proves the Rust engine covers every piece of gameplay logic in
the real game source. It has one Markdown table per scope file under ``docs/parity/ledger/``,
with one row per auditable unit: functions, named branches, RNG sites, joker-context calls,
queued events and game entities. Rows come from parsing the Lua with a real parser
(``luaparser``). Reviewers fill in the ``status``, ``rust``, ``test``, ``gap`` and ``notes``
columns; ``generate`` keeps those columns (merged by row id) every time it rewrites the files.

Usage (luaparser is not a project dependency; uv adds it for the one run)::

    uv run --no-sync --with luaparser==4.2.0 python scripts/parity_ledger.py generate
    uv run --no-sync --with luaparser==4.2.0 python scripts/parity_ledger.py check [--strict]

``balatro_source/`` is proprietary and gitignored. Rows quote only short identifiers and
conditions (at most 100 characters each), never code. The script refuses to run unless
``balatro_source/version.jkr`` starts with ``1.0.1o``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "balatro_source"
PARITY_DIR = ROOT / "docs" / "parity"
LEDGER_DIR = PARITY_DIR / "ledger"
README_PATH = PARITY_DIR / "README.md"
GAPS_PATH = PARITY_DIR / "gaps.md"

REQUIRED_VERSION = "1.0.1o"
LUAPARSER_MAJOR = "4"
TEXT_LIMIT = 100  # max characters of Lua text quoted in one place
CHAIN_ITEM_LIMIT = 72  # max characters of one condition inside a context chain
CONTEXT_LIMIT = 320  # max characters of a rendered context chain

COLUMNS = ("id", "lua", "kind", "subject", "context", "status", "rust", "test", "gap", "notes")
GENERATED_COLUMNS = COLUMNS[:5]
MANUAL_COLUMNS = COLUMNS[5:]
KINDS = ("function", "branch", "rng", "context_call", "event", "entity")
KIND_KEYS = {
    "function": "fn",
    "branch": "branch",
    "rng": "rng",
    "context_call": "ctx",
    "event": "event",
    "entity": "entity",
}
STATUSES = ("unreviewed", "ported", "verified", "gap", "smods-diff", "n/a-ui", "n/a-meta")
GAP_COLUMNS = ("id", "area", "summary", "lua", "python fix", "rust", "test", "status")
GAP_STATUSES = ("open", "in-progress", "python-fixed", "fixed", "wontfix")
GAP_ID_RE = re.compile(r"^[GB]\d{2}$")
SUMMARY_BEGIN = "<!-- BEGIN GENERATED SUMMARY (scripts/parity_ledger.py generate) -->"
SUMMARY_END = "<!-- END GENERATED SUMMARY -->"
AUTO_NOTE = "auto:"

# Gameplay files in scope, relative to balatro_source/.
SCOPE_FILES = (
    "card.lua",
    "blind.lua",
    "back.lua",
    "tag.lua",
    "cardarea.lua",
    "challenges.lua",
    "game.lua",
    "functions/state_events.lua",
    "functions/common_events.lua",
    "functions/misc_functions.lua",
    "functions/button_callbacks.lua",
    "functions/UI_definitions.lua",
)

# game.lua is mostly engine boot, menus and rendering. Only these functions are gameplay.
# Game:update (the state dispatcher, which also moves SELECTING_HAND -> DRAW_TO_HAND when the
# hand is empty) and Game:init_item_prototypes (the P_* tables and pool ordering) are included
# on top of the run-lifecycle functions and the update_* state handlers.
GAME_FUNCTIONS = {
    "Game:init_game_object",
    "Game:start_run",
    "Game:delete_run",
    "Game:init_item_prototypes",
    "Game:update",
}
GAME_FUNCTION_RE = re.compile(r"^Game:update_\w+$")

# UI_definitions.lua is in scope only for functions with a gameplay side effect: an RNG draw,
# card creation, a hand level-up, a game-state ease (ease_value/ease_colour/
# ease_background_colour* are visual tweens and do not count) or an assignment into G.GAME.
UI_DEFS_TRIGGER_CALLS = {
    "pseudorandom",
    "pseudoseed",
    "pseudoshuffle",
    "pseudorandom_element",
    "create_card",
    "level_up_hand",
    "ease_dollars",
    "ease_chips",
    "ease_discard",
    "ease_hands_played",
    "ease_ante",
    "ease_round",
}

# Functions whose if/elseif arms naming a specific entity become `branch` rows.
BRANCH_SCOPE: dict[str, set[str] | str] = {
    "card.lua": {
        "Card:calculate_joker",
        "Card:use_consumeable",
        "Card:can_use_consumeable",
        "Card:set_ability",
        "Card:add_to_deck",
        "Card:remove_from_deck",
        "Card:calculate_dollar_bonus",
        "Card:calculate_seal",
        "Card:get_end_of_round_effect",
        "Card:apply_to_run",
        "Card:open",
        "Card:redeem",
        "Card:set_cost",
        "Card:get_chip_bonus",
        "Card:get_chip_mult",
        "Card:get_chip_x_mult",
        "Card:get_chip_h_mult",
        "Card:get_chip_h_x_mult",
        "Card:get_p_dollars",
        "Card:get_edition",
    },
    "blind.lua": "Blind:*",
    "back.lua": {"Back:apply_to_run", "Back:trigger_effect"},
    "tag.lua": {"Tag:apply_to_run", "Tag:yep"},
    "state_events.lua": {"G.FUNCS.evaluate_play", "end_round", "G.FUNCS.evaluate_round"},
}
# Further gameplay functions whose named arms carry entity-specific rules (per-frame joker
# values, Verdant Leaf on sell, Orbital Tag's hand, edition/seal application, Stone/Wild/
# Smeared/Pareidolia card semantics, Ankh's check, challenge rules, sticker rolls, Illusion,
# Director's Cut/Retcon, Negative buys at full slots).
BRANCH_SCOPE_ADDITIONS: dict[str, set[str]] = {
    "card.lua": {
        "Card:update",
        "Card:sell_card",
        "Card:set_edition",
        "Card:set_seal",
        "Card:get_nominal",
        "Card:get_id",
        "Card:is_face",
        "Card:is_suit",
        "Card:check_use",
        "Card:highlight",
    },
    "tag.lua": {"Tag:set_ability"},
    "cardarea.lua": {"CardArea:update", "CardArea:draw_card_from"},
    "state_events.lua": {
        "new_round",
        "G.FUNCS.draw_from_deck_to_hand",
        "G.FUNCS.discard_cards_from_highlighted",
    },
    "common_events.lua": {
        "draw_card",
        "get_current_pool",
        "create_card",
        "reset_idol_card",
        "reset_mail_rank",
        "reset_castle_card",
    },
    "misc_functions.lua": {"mod_chips", "get_blind_amount"},
    "button_callbacks.lua": {
        "G.FUNCS.can_select_card",
        "G.FUNCS.check_for_buy_space",
        "G.FUNCS.buy_from_shop",
        "G.FUNCS.reroll_boss_button",
    },
    "UI_definitions.lua": {"create_card_for_shop", "create_UIBox_blind_choice"},
}

# Functions that name entities only for display or unlock bookkeeping; left out of the
# entity rows' "named in" cross-references (their own rows are still in the ledger).
REF_EXCLUDE = {
    "Card:generate_UIBox_ability_table",
    "Card:load",
    "check_for_unlock",
    "G.FUNCS.HUD_blind_debuff_prefix",
    "G.FUNCS.change_collab",
    "G.FUNCS.refresh_contrast_mode",
}

RNG_DRAWS = {"pseudorandom", "pseudoseed", "pseudoshuffle", "pseudorandom_element"}
# Which argument carries the stream key (0-based).
RNG_KEY_ARG = {
    "pseudorandom": 0,
    "pseudoseed": 0,
    "pseudoshuffle": 1,
    "pseudorandom_element": 1,
    "create_card": 7,
    "poll_edition": 0,
    "get_pack": 0,
    "get_current_pool": 3,
    "get_next_tag_key": 0,
    ":shuffle": 0,
}
# Helpers that draw from the seeded RNG on behalf of their caller; their call sites decide
# the stream keys, so each call site is an `rng` row as well.
RNG_HELPERS = {
    "create_card",
    "create_playing_card",
    "poll_edition",
    "get_pack",
    "get_current_pool",
    "get_next_tag_key",
    "get_next_voucher_key",
    "get_new_boss",
    "random_string",
    "reset_idol_card",
    "reset_mail_rank",
    "reset_ancient_card",
    "reset_castle_card",
}
# `math.random` inside the arguments of these calls only varies sound pitch or picks a
# speech-bubble quip, so it is not listed.
UI_ARGUMENT_CALLS = {"play_sound", "PLAY_SOUND", ":add_speech_bubble", ":say_stuff"}
CONTEXT_METHODS = {"calculate_joker", "trigger_effect", "calculate_seal", "apply_to_run"}
# Payload keys present in most contexts; left out of context_call ids (kept in the subject).
CONTEXT_PAYLOAD_KEYS = {
    "full_hand",
    "scoring_hand",
    "scoring_name",
    "poker_hands",
    "other_card",
    "card",
    "cards",
}

ENTITY_TABLES = {"P_BLINDS": "blind", "P_TAGS": "tag", "P_STAKES": "stake", "P_SEALS": "seal"}
CENTER_SET_TYPES = {
    "Joker": "joker",
    "Tarot": "tarot",
    "Planet": "planet",
    "Spectral": "spectral",
    "Voucher": "voucher",
    "Booster": "booster",
    "Back": "back",
    "Enhanced": "enhancement",
    "Edition": "edition",
    "Default": "base",
}
EDITION_FLAGS = {"foil", "holo", "polychrome", "negative"}

# --- heuristics for pre-marking obvious n/a rows (conservative: only rows that provably
# --- write no game state, and either never touch game state or are UI/meta by name)
GAMEPLAY_ROOTS = (
    "G.GAME",
    "G.STATE",
    "G.STATES",
    "G.STATE_COMPLETE",
    "G.hand",
    "G.deck",
    "G.play",
    "G.jokers",
    "G.consumeables",
    "G.discard",
    "G.shop_jokers",
    "G.shop_vouchers",
    "G.shop_booster",
    "G.pack_cards",
    "G.playing_cards",
    "G.P_CENTERS",
    "G.P_CENTER_POOLS",
    "G.P_JOKER_RARITY_POOLS",
    "G.P_BLINDS",
    "G.P_TAGS",
    "G.P_SEALS",
    "G.P_STAKES",
    "G.P_CARDS",
)
GAMEPLAY_FIELDS = {
    "ability",
    "base",
    "edition",
    "seal",
    "debuff",
    "facing",
    "cost",
    "sell_cost",
    "extra_cost",
    "playing_card",
    "highlighted",
    "added_to_deck",
    "getting_sliced",
    "sort_id",
    "cards",
    "card_limit",
    "temp_limit",
    "pinned",
    "center",
    "center_key",
    "triggered",
    "disabled",
    "chips",
    "mult",
    "dollars",
    "hands",
    "boss",
    "highlighted_limit",
}
GAMEPLAY_CALLS = {
    "pseudorandom",
    "pseudoseed",
    "pseudoshuffle",
    "pseudorandom_element",
    "ease_dollars",
    "ease_chips",
    "ease_discard",
    "ease_hands_played",
    "ease_ante",
    "ease_round",
    "create_card",
    "copy_card",
    "create_playing_card",
    "add_joker",
    "level_up_hand",
    "add_tag",
    "draw_card",
    "eval_card",
    "end_round",
    "new_round",
    "save_run",
    "mod_chips",
    "mod_mult",
}
MUTATING_METHODS = {
    "set_ability",
    "set_edition",
    "set_seal",
    "set_base",
    "set_debuff",
    "set_cost",
    "set_eternal",
    "set_perishable",
    "set_rental",
    "add_to_deck",
    "remove_from_deck",
    "emplace",
    "remove_card",
    "start_dissolve",
    "shatter",
    "explode",
    "flip",
    "sell_card",
    "redeem",
    "open",
    "apply_to_run",
    "trigger_effect",
    "calculate_joker",
    "calculate_seal",
    "calculate_dollar_bonus",
    "calculate_perishable",
    "calculate_rental",
    "use_consumeable",
    "change_suit",
    "change_size",
    "shuffle",
    "sort",
    "set_blind",
    "defeat",
    "disable",
    "debuff_hand",
    "press_play",
    "drawn_to_hand",
    "stay_flipped",
    "debuff_card",
    "modify_hand",
    "highlight",
    "add_to_highlighted",
    "remove_from_highlighted",
    "unhighlight_all",
    "parse_highlighted",
    "draw_card_from",
    "hard_set_cards",
    "set_ranks",
    "yep",
    "nope",
    "remove_from_game",
    "init_game_object",
    "start_run",
    "delete_run",
    "load",
}
UI_ROOTS = (
    "G.C",
    "G.UIT",
    "G.ROOM",
    "G.ROOM_ATTACH",
    "G.CONTROLLER",
    "G.OVERLAY_MENU",
    "G.SHADERS",
    "G.ASSET_ATLAS",
    "G.ANIMATION_ATLAS",
    "G.TILESIZE",
    "G.TILESCALE",
    "G.HUD",
    "G.HUD_blind",
    "G.HUD_tags",
    "G.MOVEABLES",
    "G.I",
    "G.ARGS",
    "G.CANVAS",
    "love",
)
UI_FIELDS = {
    "T",
    "VT",
    "children",
    "states",
    "juice",
    "tilt_var",
    "ambient_tilt",
    "hover_tilt",
    "role",
    "alignment",
    "sprite_facing",
    "shadow_parrallax",
    "alerted",
}
UI_CALLS = {
    "play_sound",
    "PLAY_SOUND",
    "attention_text",
    "juice_card",
    "juice_card_until",
    "UIBox",
    "Sprite",
    "AnimatedSprite",
    "DynaText",
    "Particles",
    "localize",
    "update_canvas_juice",
    "ease_colour",
    "ease_value",
    "ease_background_colour",
    "ease_background_colour_blind",
    "play_area_status_text",
    "set_alerts",
}
UI_METHODS = {"juice_up", "draw_shader", "draw_self", "draw_boundingrect", "add_speech_bubble"}
META_ROOTS = ("G.PROFILES", "G.ACHIEVEMENTS", "G.DISCOVER_TALLIES", "G.STEAM", "love.filesystem")
META_CALLS = {
    "check_for_unlock",
    "unlock_card",
    "discover_card",
    "unlock_achievement",
    "fetch_achievements",
    "notify_alert",
    "inc_steam_stat",
    "unlock_notify",
    "set_profile_progress",
    "set_discover_tallies",
    "inc_career_stat",
    "set_joker_usage",
    "set_joker_win",
    "set_joker_loss",
    "set_deck_usage",
    "set_deck_win",
    "set_deck_loss",
    "set_challenge_unlock",
    "check_and_set_high_score",
    "compress_and_save",
}
META_NAMES = META_CALLS | {
    "create_unlock_overlay",
    "convert_save_to_meta",
    "get_joker_win_sticker",
    "get_deck_win_stake",
    "get_deck_win_sticker",
    "send_score",
    "send_name",
    "save_progress",
    "save_settings",
    "save_metrics",
    "save_notify",
}
UI_NAME_RE = re.compile(
    r"^(draw|hover|stop_hover|juice_up|align|align_h_popup|move|hard_set_T|update_alert"
    r"|set_sprites|remove_UI|wiggle|change_dim|change_colour|set_text|alert_debuff"
    r"|get_loc_debuff_text|get_uibox_table|generate_UI|generate_UIBox_\w+|generate_card_ui"
    r"|juice_card|juice_card_until|update_canvas_juice|set_screen_positions|ease_colour"
    r"|ease_background_colour\w*|play_area_status_text|get_type_colour|play_sound"
    r"|modulate_sound|PLAY_SOUND|STOP_AUDIO|SET_SFX|MODULATE|RESTART_MUSIC|AMBIENT"
    r"|localize|loc_parse_string|loc_colour|init_localization|number_format"
    r"|score_number_scale|add_to_drawhash|reset_drawhash|adjust_alpha|lighten|darken"
    r"|mix_colours|HEX|get_blind_main_colour|get_stake_sprite|get_front_spriteinfo"
    r"|get_stake_col)$"
)
# Calls that only animate, play audio or pace the event queue. An event whose body does
# nothing else is pre-marked n/a-ui.
UI_EVENT_CALLS = {
    "play_sound",
    "attention_text",
    "juice_card",
    "juice_card_until",
    "play_area_status_text",
    "delay",
    "ease_colour",
    "ease_background_colour",
    "ease_background_colour_blind",
    ":juice_up",
    ":start_materialize",
    ":add_speech_bubble",
    ":say_stuff",
}
UI_ASSIGN_ROOTS = ("G.ROOM", "G.ROOM_ATTACH", "G.CONTROLLER")


class LedgerError(Exception):
    """A ledger, README or gap-register file could not be read or is invalid."""


# =============================================================================================
# Markdown tables (no luaparser needed)
# =============================================================================================


@dataclass
class Row:
    id: str
    lua: str
    kind: str
    subject: str
    context: str
    status: str = "unreviewed"
    rust: str = ""
    test: str = ""
    gap: str = ""
    notes: str = ""

    def cells(self) -> list[str]:
        return [getattr(self, column) for column in COLUMNS]

    def generated(self) -> tuple[str, ...]:
        return tuple(getattr(self, column) for column in GENERATED_COLUMNS)

    def manual(self) -> tuple[str, ...]:
        return tuple(getattr(self, column) for column in MANUAL_COLUMNS)


_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def split_cells(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|") and not inner.endswith("\\|"):
        inner = inner[:-1]
    return [cell.strip().replace("\\|", "|") for cell in _CELL_SPLIT.split(inner)]


def escape_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def table_line(cells: Iterable[str]) -> str:
    return "| " + " | ".join(escape_cell(cell) for cell in cells) + " |"


def is_separator(cells: list[str]) -> bool:
    return all(cell and set(cell) <= set("-: ") for cell in cells)


def parse_tables(text: str, header: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """Every data row of every table with this header, tagged with its section heading."""
    rows: list[tuple[str, list[str]]] = []
    section = ""
    in_table = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        if not line.lstrip().startswith("|"):
            in_table = False
            continue
        cells = split_cells(line)
        if tuple(cells) == header:
            in_table = True
            continue
        if not in_table or is_separator(cells):
            continue
        if len(cells) != len(header):
            raise LedgerError(f"line {lineno}: expected {len(header)} cells, got {len(cells)}")
        rows.append((section, cells))
    return rows


def parse_ledger(text: str) -> tuple[list[Row], list[Row]]:
    """Return (main rows, stale rows) of one ledger file."""
    main: list[Row] = []
    stale: list[Row] = []
    for section, cells in parse_tables(text, COLUMNS):
        (stale if section.lower().startswith("stale rows") else main).append(Row(*cells))
    return main, stale


def ledger_name(rel: str) -> str:
    return Path(rel).name + ".md"


def render_ledger(rel: str, rows: list[Row], stale: list[Row]) -> str:
    counts = Counter(row.kind for row in rows)
    count_text = ", ".join(f"{counts[kind]} {kind}" for kind in KINDS if counts[kind])
    lines = [
        f"# Parity ledger: {Path(rel).name}",
        "",
        "<!-- Generated by scripts/parity_ledger.py. Edit only the status, rust, test, gap and"
        " notes columns; `generate` keeps them (merged by id) and rewrites everything else. -->",
        "",
        f"Source: `balatro_source/{rel}` (Balatro {REQUIRED_VERSION}). {len(rows)} rows:"
        f" {count_text}. Column meanings, statuses and the review protocol are in"
        " [../README.md](../README.md).",
        "",
        table_line(COLUMNS),
        "|" + "---|" * len(COLUMNS),
    ]
    lines += [table_line(row.cells()) for row in rows]
    if stale:
        lines += [
            "",
            "## Stale rows",
            "",
            "The generator no longer produces these ids. Move their review data to the matching"
            " new row, then delete them; `check` fails while any remain.",
            "",
            table_line(COLUMNS),
            "|" + "---|" * len(COLUMNS),
        ]
        lines += [table_line(row.cells()) for row in stale]
    return "\n".join(lines) + "\n"


def render_summary(per_file: dict[str, list[Row]]) -> str:
    lines = [
        table_line(("file", "kind", "rows", *STATUSES)),
        "|---|---|" + "---:|" * (1 + len(STATUSES)),
    ]
    totals: Counter[str] = Counter()
    total_rows = 0
    for name, rows in per_file.items():
        for kind in KINDS:
            kind_rows = [row for row in rows if row.kind == kind]
            if not kind_rows:
                continue
            counts = Counter(row.status for row in kind_rows)
            totals.update(counts)
            total_rows += len(kind_rows)
            lines.append(
                table_line((name, kind, str(len(kind_rows)), *(str(counts[s]) for s in STATUSES)))
            )
    lines.append(table_line(("all", "all", str(total_rows), *(str(totals[s]) for s in STATUSES))))
    return "\n".join(lines)


def replace_summary(readme: str, summary: str) -> str:
    try:
        start = readme.index(SUMMARY_BEGIN) + len(SUMMARY_BEGIN)
        end = readme.index(SUMMARY_END, start)
    except ValueError as exc:
        raise LedgerError(f"{README_PATH} lacks the summary markers") from exc
    return readme[:start] + "\n" + summary + "\n" + readme[end:]


def parse_gaps(text: str) -> dict[str, list[str]]:
    gaps: dict[str, list[str]] = {}
    for _, cells in parse_tables(text, GAP_COLUMNS):
        gap_id, status = cells[0], cells[-1]
        if not GAP_ID_RE.match(gap_id):
            raise LedgerError(f"gaps.md: bad gap id {gap_id!r}")
        if gap_id in gaps:
            raise LedgerError(f"gaps.md: duplicate gap id {gap_id}")
        if status not in GAP_STATUSES:
            raise LedgerError(f"gaps.md: {gap_id} has status {status!r}, not one of {GAP_STATUSES}")
        gaps[gap_id] = cells
    return gaps


def gap_refs(cell: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", cell) if part]


def validate_rows(name: str, rows: list[Row], gaps: dict[str, list[str]]) -> list[str]:
    """Field-level problems in one ledger file (the same rules the pytest suite checks)."""
    problems: list[str] = []
    for row in rows:
        where = f"{name}: {row.id}"
        if row.kind not in KINDS:
            problems.append(f"{where}: unknown kind {row.kind!r}")
        if row.status not in STATUSES:
            problems.append(f"{where}: invalid status {row.status!r}")
        refs = gap_refs(row.gap)
        for ref in refs:
            if ref not in gaps:
                problems.append(f"{where}: gap {ref!r} is not in gaps.md")
        if row.status == "gap" and not refs:
            problems.append(f"{where}: status gap needs a gap id")
        if row.status in ("ported", "verified") and not row.rust:
            problems.append(f"{where}: status {row.status} needs the rust location")
        if row.status == "verified" and not row.test:
            problems.append(f"{where}: status verified needs a test id")
    return problems


# =============================================================================================
# Lua parsing
# =============================================================================================

A: Any = None  # luaparser.astnodes, bound by load_luaparser()
lua_ast: Any = None

BINARY_OPS = {
    "AddOp": "+",
    "SubOp": "-",
    "MultOp": "*",
    "FloatDivOp": "/",
    "FloorDivOp": "//",
    "ModOp": "%",
    "ExpoOp": "^",
    "Concat": "..",
    "EqToOp": "==",
    "NotEqToOp": "~=",
    "LessThanOp": "<",
    "GreaterThanOp": ">",
    "LessOrEqThanOp": "<=",
    "GreaterOrEqThanOp": ">=",
    "AndLoOp": "and",
    "OrLoOp": "or",
    "BAndOp": "&",
    "BOrOp": "|",
    "BXorOp": "~",
    "BShiftLOp": "<<",
    "BShiftROp": ">>",
}
UNARY_OPS = {"ULNotOp": "not ", "UMinusOp": "-", "ULengthOP": "#", "UBNotOp": "~"}
QUOTES = {"SINGLE_QUOTE": ("'", "'"), "DOUBLE_QUOTE": ('"', '"'), "DOUBLE_SQUARE": ("[[", "]]")}


def load_luaparser() -> None:
    global A, lua_ast
    try:
        import luaparser
        from luaparser import ast, astnodes
    except ImportError as exc:
        raise SystemExit(
            "parity_ledger: luaparser is not installed. Run through\n"
            "  uv run --no-sync --with luaparser==4.2.0 python scripts/parity_ledger.py ..."
        ) from exc
    version = getattr(luaparser, "__version__", "?")
    if version.split(".")[0] != LUAPARSER_MAJOR:
        raise SystemExit(f"parity_ledger: needs luaparser {LUAPARSER_MAJOR}.x, found {version}")
    A, lua_ast = astnodes, ast


def check_version(source: Path) -> None:
    version_file = source / "version.jkr"
    if not version_file.is_file():
        raise SystemExit(f"parity_ledger: {version_file} not found (use --source)")
    head = version_file.read_text(encoding="utf-8", errors="replace").strip()
    if not head.startswith(REQUIRED_VERSION):
        first = head.splitlines()[0] if head else ""
        raise SystemExit(
            f"parity_ledger: {version_file} says {first!r}; this ledger is for {REQUIRED_VERSION}"
        )


def children(node: Any) -> Iterator[Any]:
    for key, value in vars(node).items():
        if key.startswith("_") or key == "comments":
            continue
        if isinstance(value, A.Node):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, A.Node):
                    yield item


def walk(node: Any, into_functions: bool = True) -> Iterator[Any]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if not into_functions and current is not node and is_function_node(current):
            continue
        stack.extend(reversed(list(children(current))))


def is_function_node(node: Any) -> bool:
    return isinstance(node, (A.AnonymousFunction, A.Function, A.LocalFunction, A.Method))


def string_value(node: Any) -> str:
    raw = node.s
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def line_of(node: Any) -> int:
    return getattr(node, "line", None) or 0


def end_line_of(node: Any) -> int:
    token = getattr(node, "last_token", None)
    return getattr(token, "line", None) or line_of(node)


def block_end(stmts: list[Any], default: int) -> int:
    return max([end_line_of(stmt) for stmt in stmts] or [default])


def clip(text: str, limit: int = TEXT_LIMIT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def code(text: str, limit: int = TEXT_LIMIT) -> str:
    text = clip(text, limit)
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def slug(text: str, limit: int = 64) -> str:
    text = re.sub(r"[^a-z0-9_]+", "-", text.lower()).strip("-")
    if len(text) > limit:
        text = text[:limit].rsplit("-", 1)[0] or text[:limit]
    return text or "x"


class Source:
    """One parsed scope file."""

    def __init__(self, source_dir: Path, rel: str):
        self.rel = rel
        self.name = Path(rel).name
        self.key = Path(rel).stem
        self.text = (source_dir / rel).read_text(encoding="utf-8")
        try:
            self.tree = lua_ast.parse(self.text)
        except Exception as exc:  # luaparser raises its own syntax errors
            raise SystemExit(f"parity_ledger: luaparser could not parse {rel}: {exc}") from exc

    def render(self, node: Any) -> str:
        """Lua text of an expression, rebuilt from the AST (luaparser's spans are unreliable
        for some nodes, e.g. assignment targets that start with a bare name)."""
        kind = type(node).__name__
        if isinstance(node, A.Name):
            out = node.id
        elif isinstance(node, A.Index):
            base = self.render(node.value)
            if node.notation == A.IndexNotation.DOT:
                out = f"{base}.{node.idx.id}"
            else:
                out = f"{base}[{self.render(node.idx)}]"
        elif isinstance(node, A.String):
            left, right = QUOTES.get(node.delimiter.name, ("'", "'"))
            out = f"{left}{string_value(node)}{right}"
        elif isinstance(node, A.Number):
            start, stop = getattr(node, "start_char", None), getattr(node, "stop_char", None)
            out = self.text[start : stop + 1] if start is not None else repr(node.n)
        elif isinstance(node, A.Nil):
            out = "nil"
        elif isinstance(node, A.TrueExpr):
            out = "true"
        elif isinstance(node, A.FalseExpr):
            out = "false"
        elif isinstance(node, A.Varargs):
            out = "..."
        elif isinstance(node, A.Call):
            out = f"{self.render(node.func)}({', '.join(self.render(a) for a in node.args)})"
        elif isinstance(node, A.Invoke):
            args = ", ".join(self.render(a) for a in node.args)
            out = f"{self.render(node.source)}:{node.func.id}({args})"
        elif kind in BINARY_OPS:
            op = BINARY_OPS[kind]
            sep = op if op == ".." else f" {op} "
            out = f"{self.render(node.left)}{sep}{self.render(node.right)}"
        elif kind in UNARY_OPS:
            out = f"{UNARY_OPS[kind]}{self.render(node.operand)}"
        elif isinstance(node, A.Table):
            out = "{…}" if node.fields else "{}"
        elif isinstance(node, A.AnonymousFunction):
            out = "function(…) … end"
        else:
            out = f"<{kind}>"
        if getattr(node, "wrapped", False):
            out = f"({out})"
        return out

    def call_name(self, node: Any) -> str:
        """`foo`, `G.FUNCS.bar` or `:method` for a Call/Invoke node."""
        if isinstance(node, A.Invoke):
            return f":{node.func.id}"
        return self.render(node.func)


@dataclass
class LuaFunction:
    name: str
    params: list[str]
    start: int
    end: int
    node: Any  # Function / LocalFunction / Method / AnonymousFunction
    body: list[Any]


def top_level_functions(src: Source) -> list[LuaFunction]:
    found: list[LuaFunction] = []

    def scan(stmts: list[Any]) -> None:
        for stmt in stmts:
            name = fn = None
            if isinstance(stmt, A.Method):
                name, fn = f"{src.render(stmt.source)}:{stmt.name.id}", stmt
            elif isinstance(stmt, (A.Function, A.LocalFunction)):
                name, fn = src.render(stmt.name), stmt
            elif (
                isinstance(stmt, (A.Assign, A.LocalAssign))
                and len(stmt.targets) == 1
                and len(stmt.values) == 1
                and isinstance(stmt.values[0], A.AnonymousFunction)
            ):
                name, fn = src.render(stmt.targets[0]), stmt.values[0]
            if fn is not None:
                found.append(
                    LuaFunction(
                        name=name,
                        params=[src.render(arg) for arg in fn.args],
                        start=line_of(stmt),
                        end=end_line_of(stmt),
                        node=fn,
                        body=fn.body.body,
                    )
                )
                continue
            for block in control_blocks(stmt):
                scan(block)

    scan(src.tree.body.body)
    return found


def control_blocks(stmt: Any) -> Iterator[list[Any]]:
    if isinstance(stmt, (A.If, A.ElseIf)):
        yield stmt.body.body
        if isinstance(stmt.orelse, (A.If, A.ElseIf)):
            yield from control_blocks(stmt.orelse)
        elif isinstance(stmt.orelse, A.Block):
            yield stmt.orelse.body
    elif isinstance(stmt, (A.Do, A.While, A.Repeat, A.Fornum, A.Forin)):
        yield stmt.body.body


def in_scope(src: Source, fn: LuaFunction) -> bool:
    if src.name == "game.lua":
        return fn.name in GAME_FUNCTIONS or bool(GAME_FUNCTION_RE.match(fn.name))
    if src.name == "UI_definitions.lua":
        return has_gameplay_trigger(src, fn)
    return True


def has_gameplay_trigger(src: Source, fn: LuaFunction) -> bool:
    for node in walk(fn.node):
        if isinstance(node, A.Call):
            if src.call_name(node) in UI_DEFS_TRIGGER_CALLS:
                return True
        elif isinstance(node, A.Assign):
            if any(src.render(target).startswith("G.GAME") for target in node.targets):
                return True
    return False


def branch_scope(src: Source, fn: LuaFunction) -> bool:
    if fn.name in BRANCH_SCOPE_ADDITIONS.get(src.name, set()):
        return True
    scope = BRANCH_SCOPE.get(src.name)
    if scope is None:
        return False
    if isinstance(scope, str):
        return fn.name.startswith(scope.rstrip("*"))
    return fn.name in scope


# --- entity selectors: what makes a condition "name" a specific card, blind, tag, back, ...


@dataclass(frozen=True)
class Selector:
    kind: str  # name, effect, key, seal, edition, voucher, center, joker, find, config,
    # modifier, debuff
    value: str

    def display(self) -> str:
        if self.kind in ("name", "joker", "key", "center", "voucher"):
            return self.value
        if self.kind == "seal":
            return f"{self.value} seal"
        if self.kind == "effect":
            return f"effect {self.value}"
        if self.kind == "find":
            return f"*{self.value}*"
        if self.kind == "config":
            return f"config.{self.value}"
        if self.kind == "modifier":
            return f"modifiers.{self.value}"
        if self.kind == "debuff":
            return f"debuff.{self.value}"
        return f"edition.{self.value}"


def index_key(node: Any) -> str | None:
    if node.notation == A.IndexNotation.DOT:
        return node.idx.id
    if isinstance(node.idx, A.String):
        return string_value(node.idx)
    return None


@dataclass
class Vocabulary:
    """Entity names, keys and data-driven config keys parsed from the prototype tables. A
    comparison only counts as naming an entity if its literal is one of these (so
    `args.key == 'BACKSPACE'` or `config.name == 'bottom'` are not selectors)."""

    names: set[str] = field(default_factory=set)
    keys: set[str] = field(default_factory=set)
    effects: set[str] = field(default_factory=set)
    config_keys: set[str] = field(default_factory=set)
    debuff_keys: set[str] = field(default_factory=set)

    @classmethod
    def from_entities(cls, rows: Iterable[RawRow]) -> Vocabulary:
        vocab = cls()
        for row in rows:
            info = row.entity
            vocab.names.add(str(info["name"]))
            vocab.keys.add(str(info["key"]))
            if info.get("effect"):
                vocab.effects.add(str(info["effect"]))
            vocab.config_keys |= info.get("config_keys", set())
            vocab.debuff_keys |= info.get("debuff_keys", set())
        return vocab

    def accepts(self, selector: Selector) -> bool:
        kind, value = selector.kind, selector.value
        if kind in ("name", "joker"):
            return value in self.names
        if kind == "effect":
            return value in self.effects or value in self.names
        if kind in ("key", "seal", "voucher", "center"):
            return value in self.keys
        if kind == "edition":
            return value in EDITION_FLAGS or f"e_{value}" in self.keys
        if kind == "find":
            return any(value in name for name in self.names)
        if kind == "config":
            return value in self.config_keys
        if kind == "debuff":
            return value in self.debuff_keys
        return True  # modifiers: challenge rules and stake modifiers set in code


def node_selectors(src: Source, node: Any, vocab: Vocabulary) -> list[Selector]:
    return [s for s in candidate_selectors(src, node) if vocab.accepts(s)]


def candidate_selectors(src: Source, node: Any) -> list[Selector]:
    found: list[Selector] = []
    if isinstance(node, (A.EqToOp, A.NotEqToOp)):
        for subject, other in ((node.left, node.right), (node.right, node.left)):
            if not (isinstance(other, A.String) and isinstance(subject, A.Index)):
                continue
            value = string_value(other)
            key = index_key(subject)
            if not value or key is None:
                continue
            if key in ("name", "effect", "seal"):
                found.append(Selector(key, value))
            elif key in ("key", "center_key"):
                found.append(Selector("key", value))
            elif key == "type" and src.render(subject.value).endswith(".edition"):
                found.append(Selector("edition", value))
    elif isinstance(node, A.Call):
        if src.call_name(node) == "find_joker" and node.args and isinstance(node.args[0], A.String):
            found.append(Selector("joker", string_value(node.args[0])))
    elif isinstance(node, A.Invoke):
        if (
            node.func.id == "find"
            and node.args
            and isinstance(node.args[0], A.String)
            and src.render(node.source).endswith(".name")
        ):
            found.append(Selector("find", string_value(node.args[0])))
    elif isinstance(node, A.Index):
        key = index_key(node)
        if key is None:
            return found
        base = src.render(node.value)
        if base == "G.GAME.used_vouchers":
            found.append(Selector("voucher", key))
        elif base in (
            "G.P_CENTERS",
            "G.P_BLINDS",
            "G.P_TAGS",
            "G.P_SEALS",
            "G.P_STAKES",
            "G.GAME.banned_keys",
            "G.GAME.used_jokers",
        ):
            found.append(Selector("center", key))
        elif base == "G.GAME.modifiers":
            found.append(Selector("modifier", key))
        elif base.endswith(".effect.config"):
            found.append(Selector("config", key))
        elif base == "self.debuff" and src.name == "blind.lua" or base == "G.GAME.blind.debuff":
            found.append(Selector("debuff", key))
        elif base.endswith(".edition") and key in EDITION_FLAGS:
            found.append(Selector("edition", key))
    return found


def expression_selectors(src: Source, expr: Any, vocab: Vocabulary) -> list[Selector]:
    seen: dict[Selector, None] = {}
    for node in walk(expr, into_functions=False):
        for selector in node_selectors(src, node, vocab):
            seen.setdefault(selector)
    return list(seen)


# --- row extraction


@dataclass
class RawRow:
    kind: str
    func: str
    anchor: str | None
    slug: str | None
    line: int
    end: int
    pos: int
    subject: str
    context: str
    auto: tuple[str, str] | None = None
    selectors: tuple[Selector, ...] = ()
    entity: dict[str, Any] = field(default_factory=dict)


EVENT_MARK = "[event]"


def join_chain(chain: list[str]) -> str:
    parts = list(chain)
    text = " & ".join(parts)
    dropped = False
    while len(text) > CONTEXT_LIMIT and len(parts) > 1:
        parts.pop(0)
        dropped = True
        text = "… & " + " & ".join(parts)
    return text if dropped or text else ""


def else_element(prior: list[str]) -> str:
    if len(prior) == 1:
        return f"else({code(prior[0], 60)})"
    return f"else({code(prior[0], 50)} … {code(prior[-1], 50)})"


class FunctionScanner:
    """Walks one in-scope function and emits its branch, rng, context_call and event rows."""

    def __init__(self, src: Source, fn: LuaFunction, emit_branches: bool, vocab: Vocabulary):
        self.src = src
        self.fn = fn
        self.emit_branches = emit_branches
        self.vocab = vocab
        self.rows: list[RawRow] = []
        self.folded: set[int] = set()

    def run(self) -> list[RawRow]:
        self.block(self.fn.body, [], None)
        return self.rows

    def block(self, stmts: list[Any], chain: list[str], anchor: str | None) -> None:
        for stmt in stmts:
            self.visit(stmt, chain, anchor, False)

    def visit(self, node: Any, chain: list[str], anchor: str | None, sound: bool) -> None:
        if isinstance(node, A.If):
            self.visit_if(node, chain, anchor)
            return
        if isinstance(node, (A.Call, A.Invoke)):
            self.visit_call(node, chain, anchor, sound)
            return
        for child in children(node):
            self.visit(child, chain, anchor, sound)

    def visit_if(self, node: Any, chain: list[str], anchor: str | None) -> None:
        arm: Any = node
        prior: list[str] = []
        while True:
            self.visit(arm.test, chain, anchor, False)
            condition = self.src.render(arm.test)
            selectors = expression_selectors(self.src, arm.test, self.vocab)
            inner_anchor = anchor
            if selectors:
                inner_anchor = slug(selectors[0].display(), 40)
                if self.emit_branches:
                    self.add_branch(arm, condition, selectors, chain, anchor)
            self.block(arm.body.body, [*chain, code(condition, CHAIN_ITEM_LIMIT)], inner_anchor)
            prior.append(condition)
            orelse = arm.orelse
            if isinstance(orelse, A.ElseIf):
                arm = orelse
                continue
            if isinstance(orelse, A.Block) and orelse.body:
                self.block(orelse.body, [*chain, else_element(prior)], anchor)
            break

    def add(self, kind: str, node: Any, anchor: str | None, slug_text: str | None, **kw) -> None:
        self.rows.append(
            RawRow(
                kind=kind,
                func=self.fn.name,
                anchor=anchor,
                slug=slug_text,
                line=line_of(node),
                end=kw.pop("end", end_line_of(node)),
                pos=getattr(node, "start_char", None) or 0,
                **kw,
            )
        )

    def add_branch(
        self,
        arm: Any,
        condition: str,
        selectors: list[Selector],
        chain: list[str],
        anchor: str | None,
    ) -> None:
        names = ", ".join(selector.display() for selector in selectors)
        self.add(
            "branch",
            arm,
            anchor,
            "+".join(slug(selector.display(), 40) for selector in selectors)[:120],
            end=block_end(arm.body.body, line_of(arm)),
            subject=f"{clip(names, 160)} — {code(condition)}",
            context=join_chain(chain),
            selectors=tuple(selectors),
        )

    def visit_call(self, node: Any, chain: list[str], anchor: str | None, sound: bool) -> None:
        src = self.src
        name = src.call_name(node)
        is_call = isinstance(node, A.Call)
        if is_call and name in RNG_DRAWS and id(node) not in self.folded:
            self.add_rng(node, name, chain, anchor)
        elif is_call and name == "math.random" and not sound:
            self.add_rng(node, name, chain, anchor)
        elif (is_call and name in RNG_HELPERS) or name == ":shuffle":
            self.add_rng(node, name, chain, anchor)
        if name in (f":{m}" for m in CONTEXT_METHODS) or (is_call and name == "eval_card"):
            self.add_context_call(node, name, chain, anchor)
        child_chain = chain
        if name == ":add_event" and src.render(node.source) == "G.E_MANAGER":
            self.add_event(node, chain, anchor)
            child_chain = [*chain, EVENT_MARK]
        child_sound = sound or name in UI_ARGUMENT_CALLS
        for child in children(node):
            self.visit(child, child_chain, anchor, child_sound)

    def add_rng(self, node: Any, name: str, chain: list[str], anchor: str | None) -> None:
        src = self.src
        key_index = RNG_KEY_ARG.get(name)
        key_text = ""
        if key_index is not None and key_index < len(node.args):
            key_node = node.args[key_index]
            if (
                name in RNG_DRAWS
                and isinstance(key_node, A.Call)
                and src.call_name(key_node) == "pseudoseed"
            ):
                self.folded.add(id(key_node))
                if key_node.args:
                    key_node = key_node.args[0]
            key_text = src.render(key_node)
        elif name == "math.random":
            key_text = ", ".join(src.render(arg) for arg in node.args)
        label = name.lstrip(":")
        self.add(
            "rng",
            node,
            anchor,
            f"{label}:{slug(key_text, 48)}" if key_text and key_text != "nil" else label,
            subject=code(src.render(node)),
            context=join_chain(chain),
        )

    def add_context_call(self, node: Any, name: str, chain: list[str], anchor: str | None) -> None:
        src = self.src
        label = name.lstrip(":")
        arg_index = 1 if label == "eval_card" else 0
        table = node.args[arg_index] if arg_index < len(node.args) else None
        if label == "apply_to_run" and not isinstance(table, A.Table):
            return  # Card:apply_to_run(center) / Back:apply_to_run(): not a context call
        if isinstance(table, A.Table):
            parts = [context_field(src, f) for f in table.fields]
            subject = f"{label}{{{', '.join(parts)}}}"
            flags = [p for p in parts if p.split("=", 1)[0] not in CONTEXT_PAYLOAD_KEYS]
            key = "+".join(sorted(slug(part, 32) for part in flags)) or "payload"
        else:
            arg = src.render(table) if table is not None else ""
            subject = f"{label}({arg})"
            key = slug(arg, 32) if arg else "noargs"
        self.add(
            "context_call",
            node,
            anchor,
            f"{label}:{key}"[:100],
            subject=code(subject),
            context=join_chain(chain),
        )

    def add_event(self, node: Any, chain: list[str], anchor: str | None) -> None:
        src = self.src
        settings: list[str] = []
        trigger = "immediate"
        func_body: list[Any] | None = None
        event = node.args[0] if node.args else None
        if isinstance(event, A.Call) and src.call_name(event) == "Event" and event.args:
            table = event.args[0]
            if isinstance(table, A.Table):
                for fld in table.fields:
                    key = fld.key.id if isinstance(fld.key, A.Name) else None
                    if key in ("trigger", "blocking", "blockable"):
                        value = literal_text(src, fld.value)
                        settings.append(f"{key}={value}")
                        if key == "trigger":
                            trigger = value.strip("'\"")
                    elif key == "func" and isinstance(fld.value, A.AnonymousFunction):
                        func_body = fld.value.body.body
        for position, label in ((1, "queue"), (2, "front")):
            if len(node.args) > position:
                settings.append(f"{label}={literal_text(src, node.args[position])}")
        calls = event_calls(src, func_body or [])
        shown = ", ".join(calls[:4]) + (f" +{len(calls) - 4}" if len(calls) > 4 else "")
        subject = code("Event{" + ", ".join(settings) + "}")
        if shown:
            subject += f" → {shown}"
        auto = None
        if func_body is not None and event_is_ui_only(src, func_body):
            auto = ("n/a-ui", f"{AUTO_NOTE} event only animates, plays sound or waits")
        first = next((c for c in calls if not c.startswith(":")), calls[0] if calls else "")
        event_slug = slug(trigger, 24) + (f":{slug(first, 40)}" if first else "")
        self.add(
            "event",
            node,
            anchor,
            event_slug,
            subject=subject,
            context=join_chain(chain),
            auto=auto,
        )


def literal_text(src: Source, node: Any) -> str:
    if isinstance(node, (A.String, A.Number, A.TrueExpr, A.FalseExpr, A.Nil)):
        return src.render(node)
    return "…"


def context_field(src: Source, fld: Any) -> str:
    if fld.key is None:
        return clip(src.render(fld.value), 24)
    key = fld.key.id if isinstance(fld.key, A.Name) else src.render(fld.key)
    value = fld.value
    if isinstance(value, A.TrueExpr):
        return key
    if isinstance(value, (A.String, A.Number, A.FalseExpr, A.Nil)):
        return f"{key}={src.render(value)}"
    rendered = src.render(value)
    if rendered.startswith("G.") and len(rendered) <= 24:
        return f"{key}={rendered}"
    return key


NOISE_CALLS = {"ipairs", "pairs", "next", "type", "tostring", "tonumber", "math.min", "math.max"}


def event_calls(src: Source, body: list[Any]) -> list[str]:
    """Distinct calls made by an event body (not counting nested events' bodies)."""
    found: dict[str, None] = {}
    stack = list(reversed(body))
    while stack:
        node = stack.pop()
        if isinstance(node, (A.Call, A.Invoke)):
            name = src.call_name(node)
            if name not in NOISE_CALLS and name != "Event" and not name.startswith("math."):
                found.setdefault(name)
            if name == ":add_event":
                continue
        if isinstance(node, A.AnonymousFunction):
            continue
        stack.extend(reversed(list(children(node))))
    return list(found)


def event_is_ui_only(src: Source, body: list[Any]) -> bool:
    if not body:
        return False
    for stmt in body:
        if isinstance(stmt, A.Return):
            if all(isinstance(v, (A.TrueExpr, A.FalseExpr, A.Nil)) for v in stmt.values):
                continue
            return False
        if isinstance(stmt, (A.Call, A.Invoke)):
            if src.call_name(stmt) not in UI_EVENT_CALLS:
                return False
            continue
        if isinstance(stmt, A.Assign):
            for target in stmt.targets:
                path = src.render(target)
                if not path.startswith(UI_ASSIGN_ROOTS) and not has_segment(path, UI_FIELDS):
                    return False
            continue
        return False
    return True


def path_segments(path: str) -> list[str]:
    return [seg for seg in re.split(r"[.:\[\]()]+", path) if seg]


def access_path(src: Source, node: Any, self_is_g: bool = False) -> str:
    """`a.b[].c` for an index chain, with bracketed keys elided (the key expression is not
    part of what is accessed). In Game methods `self` is G, so it is spelled `G`."""
    if isinstance(node, A.Index):
        base = access_path(src, node.value, self_is_g)
        if node.notation == A.IndexNotation.DOT:
            return f"{base}.{node.idx.id}"
        return f"{base}[]"
    if isinstance(node, A.Name):
        return "G" if self_is_g and node.id == "self" else node.id
    return "<expr>"


def touches_game_state(path: str) -> bool:
    """Game state, unless the access goes through a UI object (`x.children.y`, `x.T.w`)."""
    if has_segment(path, UI_FIELDS) or has_root(path, UI_ROOTS):
        return False
    return has_root(path, GAMEPLAY_ROOTS) or has_segment(path, GAMEPLAY_FIELDS)


def has_root(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == r or path.startswith((r + ".", r + "[", r + ":")) for r in roots)


def has_segment(path: str, fields: set[str]) -> bool:
    return any(seg in fields for seg in path_segments(path)[1:])


# --- heuristics


@dataclass
class Features:
    reads: bool = False
    writes: list[str] = field(default_factory=list)
    callees: set[str] = field(default_factory=set)
    ui: bool = False
    meta: bool = False


def function_features(
    src: Source, fn: LuaFunction, scope_names: set[str], methods: dict[str, set[str]]
) -> Features:
    """What a function touches. `methods` maps a method name to the in-scope `Class:method`
    functions it may resolve to; such calls count as gameplay unless every candidate is n/a."""
    feats = Features()
    is_game_method = fn.name.startswith("Game:")  # there `self` is G itself
    for node in walk(fn.node):
        if isinstance(node, A.Index):
            path = access_path(src, node, is_game_method)
            if touches_game_state(path):
                feats.reads = True
            if has_root(path, UI_ROOTS) or has_segment(path, UI_FIELDS):
                feats.ui = True
            if has_root(path, META_ROOTS) or path.startswith("G.SETTINGS.profile"):
                feats.meta = True
        elif isinstance(node, A.Assign):
            for target in node.targets:
                path = access_path(src, target, is_game_method)
                if touches_game_state(path):
                    feats.writes.append(f"assigns {clip(path, 40)}")
        elif isinstance(node, A.Call):
            name = src.call_name(node)
            if name in GAMEPLAY_CALLS:
                feats.writes.append(f"calls {name}")
            elif name in scope_names and name != fn.name:
                feats.callees.add(name)
            if name in UI_CALLS or name.startswith(("create_UIBox", "G.UIDEF.", "love.")):
                feats.ui = True
            if name in META_CALLS:
                feats.meta = True
        elif isinstance(node, A.Invoke):
            method = node.func.id
            receiver = src.render(node.source)
            ui_remove = method == "remove" and re.search(
                r"children|UIBox|alert|popup|OVERLAY|text|button|uibox", receiver
            )
            if method in MUTATING_METHODS and not ui_remove:
                feats.writes.append(f"calls :{method}")
            elif not ui_remove:
                feats.callees.update(methods.get(method, set()) - {fn.name})
            if method in UI_METHODS:
                feats.ui = True
            if receiver in ("G", "self") and method in ("save_progress", "save_metrics"):
                feats.meta = True
    return feats


def classify(fn_name: str, feats: Features, na: set[str]) -> tuple[str, str] | None:
    if feats.writes or any(callee not in na for callee in feats.callees):
        return None
    last = re.split(r"[.:]", fn_name)[-1]
    if last.startswith("can_"):
        return None  # legality checks (buttons, highlighting) are always reviewed by hand
    if last in META_NAMES:
        return (
            "n/a-meta",
            f"{AUTO_NOTE} profile/unlock/achievement helper by name; writes no game state",
        )
    if UI_NAME_RE.match(last):
        return "n/a-ui", f"{AUTO_NOTE} UI/animation function by name; writes no game state"
    if not feats.reads:
        if feats.meta:
            return "n/a-meta", f"{AUTO_NOTE} touches only profile/unlock/achievement state"
        if feats.ui:
            return "n/a-ui", f"{AUTO_NOTE} touches only UI/sound/animation state"
    return None


# --- entities


def table_fields(src: Source, table: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fld in table.fields:
        if isinstance(fld.key, A.Name):
            out[fld.key.id] = fld.value
        elif isinstance(fld.key, A.String):
            out[string_value(fld.key)] = fld.value
    return out


def literal(src: Source, node: Any) -> Any:
    if isinstance(node, A.String):
        return string_value(node)
    if isinstance(node, A.Number):
        return node.n
    if isinstance(node, A.TrueExpr):
        return True
    if isinstance(node, A.FalseExpr):
        return False
    return None


def entity_key(src: Source, fld: Any) -> str | None:
    if isinstance(fld.key, A.Name):
        return fld.key.id
    if isinstance(fld.key, A.String):
        return string_value(fld.key)
    return None


def prototype_entities(src: Source, fn: LuaFunction) -> list[RawRow]:
    rows: list[RawRow] = []
    for stmt in fn.body:
        if not (isinstance(stmt, A.Assign) and len(stmt.targets) == 1):
            continue
        target = src.render(stmt.targets[0])
        table_name = target.removeprefix("self.")
        if not (isinstance(stmt.values[0], A.Table) and table_name.startswith("P_")):
            continue
        if table_name != "P_CENTERS" and table_name not in ENTITY_TABLES:
            continue
        for fld in stmt.values[0].fields:
            key = entity_key(src, fld)
            if key is None or not isinstance(fld.value, A.Table):
                continue
            props = table_fields(src, fld.value)
            if table_name == "P_CENTERS":
                etype = CENTER_SET_TYPES.get(literal(src, props.get("set")) or "")
                if etype is None:
                    continue  # sprite placeholders such as `soul` and `undiscovered_*`
            else:
                etype = ENTITY_TABLES[table_name]
            name = literal(src, props["name"]) if "name" in props else key
            config = props.get("config")
            debuff = props.get("debuff")
            info = {
                "type": etype,
                "key": key,
                "name": name,
                "table": table_name,
                "effect": literal(src, props.get("effect")) if "effect" in props else None,
                "config_keys": set(table_fields(src, config))
                if isinstance(config, A.Table)
                else set(),
                "debuff_keys": set(table_fields(src, debuff))
                if isinstance(debuff, A.Table)
                else set(),
            }
            label = etype.capitalize()
            subject = f"{label}: {name} (`{key}`)" if name != key else f"{label}: `{key}`"
            rows.append(
                RawRow(
                    kind="entity",
                    func=table_name,
                    anchor=None,
                    slug=f"{etype}/{key}",
                    line=line_of(fld),
                    end=end_line_of(fld),
                    pos=getattr(fld, "start_char", None) or 0,
                    subject=subject,
                    context="",
                    entity=info,
                )
            )
    return rows


def challenge_entities(src: Source) -> list[RawRow]:
    rows: list[RawRow] = []
    for stmt in src.tree.body.body:
        if not (isinstance(stmt, A.Assign) and src.render(stmt.targets[0]) == "G.CHALLENGES"):
            continue
        for fld in stmt.values[0].fields:
            if not isinstance(fld.value, A.Table):
                continue
            props = table_fields(src, fld.value)
            cid = literal(src, props.get("id"))
            name = literal(src, props.get("name"))
            rules = (
                table_fields(src, props["rules"]) if isinstance(props.get("rules"), A.Table) else {}
            )

            def ids(node: Any) -> list[str]:
                if not isinstance(node, A.Table):
                    return []
                out = []
                for item in node.fields:
                    if isinstance(item.value, A.Table):
                        value = literal(src, table_fields(src, item.value).get("id"))
                        if value:
                            out.append(str(value))
                return out

            def count(key: str) -> int:
                node = props.get(key)
                return len(node.fields) if isinstance(node, A.Table) else 0

            custom, modifiers = ids(rules.get("custom")), ids(rules.get("modifiers"))
            starts = {k: count(k) for k in ("jokers", "consumeables", "vouchers")}
            parts = []
            if custom:
                parts.append("custom: " + ", ".join(custom))
            if modifiers:
                parts.append("modifiers: " + ", ".join(modifiers))
            parts.append(", ".join(f"{k} {n}" for k, n in starts.items()))
            rows.append(
                RawRow(
                    kind="entity",
                    func="G.CHALLENGES",
                    anchor=None,
                    slug=f"challenge/{cid}",
                    line=line_of(fld),
                    end=end_line_of(fld),
                    pos=getattr(fld, "start_char", None) or 0,
                    subject=f"Challenge: {name} (`{cid}`)",
                    context=clip("; ".join(parts), 200),
                    entity={
                        "type": "challenge",
                        "key": cid,
                        "name": name,
                        "rules": set(custom) | set(modifiers),
                        "starts": starts,
                    },
                )
            )
    return rows


# --- whole-source extraction


@dataclass
class Extraction:
    rows: dict[str, list[Row]]  # ledger file name -> rows in file order
    raw: dict[str, list[RawRow]]


KIND_ORDER = {kind: index for index, kind in enumerate(KINDS)}


FunctionKey = tuple[str, str, int]  # (scope file, function name, first line)


def classify_functions(
    sources: list[Source], functions: dict[str, list[LuaFunction]]
) -> tuple[dict[FunctionKey, Features], dict[FunctionKey, tuple[str, str]]]:
    """Features of every in-scope function and the n/a verdicts, as a fixed point over the
    calls between in-scope functions (a call is harmless only if its callee is n/a too)."""
    scope_names = {fn.name for fns in functions.values() for fn in fns}
    methods: dict[str, set[str]] = defaultdict(set)
    for name in scope_names:
        if ":" in name:
            methods[name.rsplit(":", 1)[1]].add(name)
    feats = {
        (src.rel, fn.name, fn.start): function_features(src, fn, scope_names, methods)
        for src in sources
        for fn in functions[src.rel]
    }
    # Greatest fixed point: start from every function that would be n/a if all its in-scope
    # callees were, then drop functions that call anything outside the set until stable.
    # (UI code is mutually recursive, e.g. Card:draw <-> CardArea:draw.)
    everything = {key[1] for key in feats}
    verdicts = {
        key: verdict
        for key, feat in feats.items()
        if (verdict := classify(key[1], feat, everything)) is not None
    }
    while True:
        na_names = {key[1] for key in verdicts}
        kept = {
            key: verdict
            for key, verdict in verdicts.items()
            if classify(key[1], feats[key], na_names) is not None
        }
        if kept.keys() == verdicts.keys():
            return feats, verdicts
        verdicts = kept


def load_sources(source_dir: Path) -> tuple[list[Source], dict[str, list[LuaFunction]]]:
    load_luaparser()
    check_version(source_dir)
    sources = [Source(source_dir, rel) for rel in SCOPE_FILES]
    functions = {
        src.rel: [fn for fn in top_level_functions(src) if in_scope(src, fn)] for src in sources
    }
    return sources, functions


def extract(source_dir: Path) -> Extraction:
    sources, functions = load_sources(source_dir)
    _, verdicts = classify_functions(sources, functions)

    # Entities first: their names and keys decide which conditions "name" an entity.
    by_rel = {src.rel: src for src in sources}
    prototypes = next(fn for fn in functions["game.lua"] if fn.name == "Game:init_item_prototypes")
    entity_rows = {
        "game.lua": prototype_entities(by_rel["game.lua"], prototypes),
        "challenges.lua": challenge_entities(by_rel["challenges.lua"]),
    }
    vocab = Vocabulary.from_entities(row for rows in entity_rows.values() for row in rows)

    raw: dict[str, list[RawRow]] = {}
    refs: dict[Selector, dict[str, bool]] = defaultdict(dict)  # fn name -> has branch rows
    for src in sources:
        file_rows: list[RawRow] = list(entity_rows.get(src.rel, []))
        for fn in functions[src.rel]:
            verdict = verdicts.get((src.rel, fn.name, fn.start))
            params = ", ".join(fn.params)
            file_rows.append(
                RawRow(
                    kind="function",
                    func=fn.name,
                    anchor=None,
                    slug=None,
                    line=fn.start,
                    end=fn.end,
                    pos=-1,
                    subject=code(f"{fn.name}({params})"),
                    context="",
                    auto=verdict,
                )
            )
            body_rows = FunctionScanner(src, fn, branch_scope(src, fn), vocab).run()
            if verdict:
                inherited = (verdict[0], f"{AUTO_NOTE} inside {fn.name}, pre-marked {verdict[0]}")
                for row in body_rows:
                    row.auto = inherited
            file_rows += body_rows
            # Entity cross-references point into gameplay code only.
            if not verdict and fn.name not in REF_EXCLUDE:
                for node in walk(fn.node):
                    for selector in node_selectors(src, node, vocab):
                        refs[selector].setdefault(fn.name, branch_scope(src, fn))
        raw[src.rel] = file_rows

    link_entities(raw, refs)
    rows = {ledger_name(rel): finalize(src_rel_key(rel), rel, raw[rel]) for rel in SCOPE_FILES}
    return Extraction(rows=rows, raw=raw)


def src_rel_key(rel: str) -> str:
    return Path(rel).stem


def link_entities(raw: dict[str, list[RawRow]], refs: dict[Selector, dict[str, bool]]) -> None:
    """Fill each entity row's context with the functions that name it (functions with branch
    rows first)."""
    entities = [row for rows in raw.values() for row in rows if row.kind == "entity"]
    by_name: dict[str, list[RawRow]] = defaultdict(list)
    by_key: dict[str, RawRow] = {}
    for row in entities:
        by_name[str(row.entity["name"])].append(row)
        by_key[str(row.entity["key"])] = row
    found: dict[int, dict[str, bool]] = defaultdict(dict)

    def mark(targets: Iterable[RawRow], funcs: dict[str, bool]) -> None:
        for target in targets:
            for func, has_branches in funcs.items():
                found[id(target)].setdefault(func, has_branches)

    for selector, funcs in refs.items():
        value = selector.value
        if selector.kind in ("name", "joker"):
            targets = by_name.get(value, [])
            if selector.kind == "joker":
                targets = [t for t in targets if t.entity["type"] == "joker"]
            mark(targets, funcs)
        elif selector.kind == "effect":
            mark(
                [
                    e
                    for e in entities
                    if e.entity.get("effect") == value or e.entity["name"] == value
                ],
                funcs,
            )
        elif selector.kind in ("key", "center", "voucher", "seal"):
            if value in by_key:
                mark([by_key[value]], funcs)
        elif selector.kind == "edition":
            if f"e_{value}" in by_key:
                mark([by_key[f"e_{value}"]], funcs)
        elif selector.kind == "find":
            mark([e for e in entities if value in str(e.entity["name"])], funcs)
        elif selector.kind == "config":
            mark([e for e in entities if value in e.entity.get("config_keys", ())], funcs)
        elif selector.kind == "debuff":
            mark([e for e in entities if value in e.entity.get("debuff_keys", ())], funcs)
        elif selector.kind == "modifier":
            mark([e for e in entities if value in e.entity.get("rules", ())], funcs)
    for row in entities:
        named = found.get(id(row), {})
        funcs = [f for f, branched in named.items() if branched]
        funcs += [f for f, branched in named.items() if not branched]
        shown = ", ".join(funcs[:6]) + (f" +{len(funcs) - 6} more" if len(funcs) > 6 else "")
        ref_text = f"named in {shown}" if funcs else "not named in code (data-driven)"
        row.context = f"{row.context}; {ref_text}" if row.context else ref_text


def finalize(file_key: str, rel: str, raw_rows: list[RawRow]) -> list[Row]:
    name = Path(rel).name
    ordered = sorted(raw_rows, key=lambda r: (r.line, r.pos, KIND_ORDER[r.kind]))
    seen: Counter[str] = Counter()
    rows: list[Row] = []
    for raw in ordered:
        parts = [file_key, KIND_KEYS[raw.kind]]
        if raw.kind != "entity":
            parts.append(raw.func)
        if raw.anchor:
            parts.append(raw.anchor)
        if raw.slug:
            parts.append(raw.slug)
        base = "/".join(parts)
        seen[base] += 1
        row_id = base if seen[base] == 1 else f"{base}#{seen[base]}"
        lua = f"{name}:{raw.line}" + (f"-{raw.end}" if raw.end > raw.line else "")
        row = Row(row_id, lua, raw.kind, raw.subject, raw.context)
        if raw.auto:
            row.status, row.notes = raw.auto
        rows.append(row)
    return rows


# =============================================================================================
# generate / check
# =============================================================================================


def read_ledger(path: Path) -> tuple[list[Row], list[Row]]:
    if not path.is_file():
        return [], []
    try:
        return parse_ledger(path.read_text(encoding="utf-8"))
    except LedgerError as exc:
        raise LedgerError(f"{path.name}: {exc}") from exc


def merge(
    generated: list[Row], existing: list[Row], stale: list[Row]
) -> tuple[list[Row], list[Row]]:
    old: dict[str, Row] = {}
    for row in [*existing, *stale]:
        if row.id in old:
            raise LedgerError(f"duplicate id {row.id}")
        old[row.id] = row
    merged: list[Row] = []
    for row in generated:
        previous = old.pop(row.id, None)
        # Review data wins. A row nobody has touched keeps following the generator, so an
        # improved pre-mark heuristic reaches it (and a withdrawn pre-mark is cleared).
        if previous is not None and not untouched(previous):
            row.status, row.rust, row.test, row.gap, row.notes = previous.manual()
        merged.append(row)
    return merged, list(old.values())


def untouched(row: Row) -> bool:
    """Only generator output in the review columns: unreviewed with no notes, or a status
    whose note is still the generator's `auto:` note; rust, test and gap empty."""
    if row.rust or row.test or row.gap:
        return False
    if row.notes.startswith(AUTO_NOTE):
        return True
    return row.status == "unreviewed" and not row.notes


def build(extraction: Extraction, prune_stale: bool) -> dict[Path, str]:
    """Expected contents of every generated file, merged with the current review data."""
    outputs: dict[Path, str] = {}
    per_file: dict[str, list[Row]] = {}
    for rel in SCOPE_FILES:
        name = ledger_name(rel)
        path = LEDGER_DIR / name
        existing, stale = read_ledger(path)
        rows, leftover = merge(extraction.rows[name], existing, stale)
        if prune_stale:
            leftover = [row for row in leftover if not untouched(row)]
        outputs[path] = render_ledger(rel, rows, leftover)
        per_file[Path(rel).name] = rows
    if not README_PATH.is_file():
        raise LedgerError(f"{README_PATH} is missing")
    readme = README_PATH.read_text(encoding="utf-8")
    outputs[README_PATH] = replace_summary(readme, render_summary(per_file))
    return outputs


def cmd_generate(args: argparse.Namespace) -> int:
    extraction = extract(args.source)
    outputs = build(extraction, args.prune_stale)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    changed = 0
    for path, text in outputs.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text:
            path.write_text(text, encoding="utf-8", newline="\n")
            changed += 1
    total = sum(len(rows) for rows in extraction.rows.values())
    stale = sum(len(parse_ledger(t)[1]) for p, t in outputs.items() if p.parent == LEDGER_DIR)
    print(f"parity_ledger: {total} rows in {len(SCOPE_FILES)} files; {changed} file(s) written")
    if stale:
        print(f"parity_ledger: {stale} stale row(s) kept; `check` fails until they are resolved")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    extraction = extract(args.source)
    problems: list[str] = []
    try:
        gaps = parse_gaps(GAPS_PATH.read_text(encoding="utf-8"))
    except (OSError, LedgerError) as exc:
        gaps = {}
        problems.append(f"gaps.md: {exc}")
    try:
        outputs = build(extraction, prune_stale=False)
    except LedgerError as exc:
        print(f"parity_ledger check: {exc}", file=sys.stderr)
        return 1
    expected_files = {path.name for path in outputs if path.parent == LEDGER_DIR}
    for path in sorted(LEDGER_DIR.glob("*")):
        if path.name not in expected_files:
            problems.append(f"{path.name}: not a ledger file for any scope file")
    unreviewed = 0
    for path, text in outputs.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current is None:
            problems.append(f"{path.name}: missing; run generate")
            continue
        if current.replace("\r\n", "\n") != text:
            problems.append(f"{path.name}: out of sync with the Lua source; run generate")
            problems += describe_diff(path, current, text)
        if path.parent == LEDGER_DIR:
            main, stale = parse_ledger(text)
            problems += [f"{path.name}: stale row {row.id}" for row in stale]
            ids = Counter(row.id for row in main)
            problems += [f"{path.name}: duplicate id {i}" for i, n in ids.items() if n > 1]
            problems += validate_rows(path.name, main, gaps)
            unreviewed += sum(row.status == "unreviewed" for row in main)
    if args.strict and unreviewed:
        problems.append(f"{unreviewed} row(s) are still unreviewed (--strict)")
    if problems:
        for problem in problems[:200]:
            print(f"parity_ledger check: {problem}", file=sys.stderr)
        if len(problems) > 200:
            print(f"parity_ledger check: ... and {len(problems) - 200} more", file=sys.stderr)
        return 1
    total = sum(len(rows) for rows in extraction.rows.values())
    print(f"parity_ledger check: OK ({total} rows, {unreviewed} unreviewed)")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Why each matching function was (or was not) pre-marked n/a."""
    sources, functions = load_sources(args.source)
    feats, verdicts = classify_functions(sources, functions)
    na_names = {key[1] for key in verdicts}
    pattern = re.compile(args.pattern)
    for key, feat in feats.items():
        rel, name, start = key
        if not pattern.search(name):
            continue
        verdict = verdicts.get(key)
        print(f"{Path(rel).name}:{start} {name}: {verdict[0] if verdict else 'review by hand'}")
        if verdict:
            print(f"    {verdict[1]}")
        print(f"    reads game state: {feat.reads}; UI: {feat.ui}; profile/meta: {feat.meta}")
        if feat.writes:
            print("    writes: " + ", ".join(sorted(set(feat.writes))[:12]))
        blocking = sorted(callee for callee in feat.callees if callee not in na_names)
        if blocking:
            print("    calls gameplay functions: " + ", ".join(blocking[:12]))
    return 0


def describe_diff(path: Path, current: str, expected: str) -> list[str]:
    if path.parent != LEDGER_DIR:
        return [f"{path.name}: the generated summary block differs"]
    try:
        have_main, _ = parse_ledger(current)
    except LedgerError as exc:
        return [f"{path.name}: {exc}"]
    want_main, _ = parse_ledger(expected)
    have = {row.id: row.generated() for row in have_main}
    want = {row.id: row.generated() for row in want_main}
    out = [f"{path.name}: new row {i}" for i in want if i not in have][:10]
    out += [f"{path.name}: row {i} no longer generated" for i in have if i not in want][:10]
    changed = [i for i in want if i in have and have[i] != want[i]]
    for row_id in changed[:10]:
        pairs = zip(GENERATED_COLUMNS, have[row_id], want[row_id], strict=True)
        columns = ", ".join(column for column, old, new in pairs if old != new)
        out.append(f"{path.name}: row {row_id} differs in {columns}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write/merge docs/parity/ledger/*.md and the summary")
    gen.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    gen.add_argument(
        "--prune-stale",
        action="store_true",
        help="drop stale rows that carry no review data (unreviewed or auto-marked only)",
    )
    chk = sub.add_parser("check", help="fail if the ledger is out of sync or has invalid fields")
    chk.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    chk.add_argument("--strict", action="store_true", help="also fail on unreviewed rows")
    exp = sub.add_parser("explain", help="show why functions were (not) pre-marked n/a")
    exp.add_argument("pattern", help="regex matched against function names")
    exp.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    commands = {"generate": cmd_generate, "check": cmd_check, "explain": cmd_explain}
    try:
        return commands[args.command](args)
    except LedgerError as exc:
        print(f"parity_ledger: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
