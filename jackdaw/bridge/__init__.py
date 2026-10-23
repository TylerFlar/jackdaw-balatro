"""Balatrobot protocol bridge.

Converts between jackdaw engine Actions and balatrobot JSON-RPC calls,
and between balatrobot game state responses and engine game_state dicts.
The live lockstep (:class:`LiveAdapter`) plays a live BalatroBot game in step
with a shadow Rust run; see ``docs/live.md``.
"""

from jackdaw.bridge.backend import Backend, LiveBackend, RPCError, SimBackend, to_bot_add_params
from jackdaw.bridge.balatrobot_adapter import (
    action_to_rpc,
    bot_accepts,
    bot_refusal,
    bot_state_to_game_state,
    extract_comparison_keys,
    game_state_to_bot_response,
)
from jackdaw.bridge.deserializer import rpc_to_action
from jackdaw.bridge.deviations import CATALOG, Deviation, Mode
from jackdaw.bridge.live import (
    DivergenceError,
    LiveAdapter,
    ReplayReport,
    VetoedActionError,
    read_bundle,
    replay_bundle,
)
from jackdaw.bridge.serializer import (
    serialize_area,
    serialize_card,
    serialize_hands,
)
from jackdaw.bridge.viewdiff import FieldDiff, diff_views

__all__ = [
    # adapter (original)
    "action_to_rpc",
    "bot_accepts",
    "bot_refusal",
    "bot_state_to_game_state",
    "extract_comparison_keys",
    "game_state_to_bot_response",
    # deserializer
    "rpc_to_action",
    # serializer
    "serialize_area",
    "serialize_card",
    "serialize_hands",
    # backend
    "Backend",
    "LiveBackend",
    "RPCError",
    "SimBackend",
    "to_bot_add_params",
    # the live lockstep
    "CATALOG",
    "Deviation",
    "DivergenceError",
    "FieldDiff",
    "LiveAdapter",
    "Mode",
    "ReplayReport",
    "VetoedActionError",
    "diff_views",
    "read_bundle",
    "replay_bundle",
]
