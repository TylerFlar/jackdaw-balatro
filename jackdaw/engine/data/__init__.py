"""The game's data tables, extracted from Balatro 1.0.1o's Lua source.

``scripts/extract_prototypes.lua`` writes the prototype tables (``centers``, ``cards``,
``blinds``, ``tags``, ``stakes``, ``seals``) and ``scripts/extract_game_tables.lua`` the rest
(``hands``, ``challenges``, ``blind_amounts``). ``cargo xtask codegen`` turns them into
``crates/jackdaw-data``, which the engine uses; Python reads them with :func:`load`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent


@cache
def load(name: str) -> dict[str, Any]:
    """One table by name: ``load("centers")`` is ``centers.json``. Cached; don't modify it."""
    return json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
