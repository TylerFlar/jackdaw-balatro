"""Balatrobot protocol bridge.

Backends for BalatroBot's JSON-RPC API: :class:`SimBackend` runs it on the Rust
simulator in process, :class:`LiveBackend` talks to a live game (or ``jackdaw
serve``) over HTTP.  The live lockstep (:class:`LiveAdapter`) plays a BalatroBot
game in step with a shadow Rust run; see ``docs/live.md``.
"""

from jackdaw.bridge.backend import Backend, LiveBackend, RPCError, SimBackend, to_bot_add_params
from jackdaw.bridge.deviations import CATALOG, Deviation, Mode
from jackdaw.bridge.live import (
    DivergenceError,
    LiveAdapter,
    ReplayReport,
    VetoedActionError,
    read_bundle,
    replay_bundle,
)
from jackdaw.bridge.viewdiff import FieldDiff, diff_views

__all__ = [
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
