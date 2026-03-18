"""Balatrobot protocol bridge.

Converts between jackdaw engine Actions and balatrobot JSON-RPC calls,
and between balatrobot game state responses and engine game_state dicts.
"""

from jackdaw.bridge.balatrobot_adapter import (
    action_to_rpc,
    bot_state_to_game_state,
    extract_comparison_keys,
)

__all__ = [
    "action_to_rpc",
    "bot_state_to_game_state",
    "extract_comparison_keys",
]
