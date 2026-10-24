"""Gymnasium environment for Balatro RL research.

Canonical usage::

    from jackdaw.env import BalatroEnvironment, DirectAdapter, balatro_game_spec

    spec = balatro_game_spec()
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    obs, mask, info = env.reset()

``DirectAdapter`` runs the Rust engine (``jackdaw.core.Game``) in process;
``LiveAdapter`` plays the live game through BalatroBot, in step with a Rust
shadow run.  ``BalatroVectorEnv`` steps many environments in parallel in Rust.
"""

from jackdaw.bridge.live import LiveAdapter
from jackdaw.env.action_space import (
    ACTION_VERSION,
    DEFAULT_MAX_FREE_MOVES,
    FREE_ACTION_TYPES,
    NUM_ACTION_TYPES,
    ActionMask,
    ActionType,
    FactoredAction,
    Legality,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
    get_consumable_target_info,
)
from jackdaw.env.agents import Agent, RandomAgent
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.consumable_targets import (
    ConsumableTargetSpec,
    get_consumable_target_spec,
    get_valid_target_cards,
    validate_card_targets,
)
from jackdaw.env.game_interface import (
    BridgeAdapter,
    DirectAdapter,
    GameAdapter,
    GameState,
)
from jackdaw.env.game_spec import (
    GameActionMask,
    GameObservation,
    GameSpec,
)
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    NUM_CENTER_KEYS,
    Observation,
    encode_observation,
)
from jackdaw.env.rewards import (
    RewardFn,
    ShapedReward,
    Transition,
    blind_progress_reward,
    sparse_reward,
)
from jackdaw.env.vector_env import BalatroVectorEnv

__all__ = [
    "ACTION_VERSION",
    "ActionMask",
    "ActionType",
    "Agent",
    "BalatroEnvironment",
    "BalatroGymnasiumEnv",
    "BalatroVectorEnv",
    "BridgeAdapter",
    "ConsumableTargetSpec",
    "D_CONSUMABLE",
    "D_GLOBAL",
    "D_JOKER",
    "D_PLAYING_CARD",
    "D_SHOP",
    "DEFAULT_MAX_FREE_MOVES",
    "DirectAdapter",
    "FREE_ACTION_TYPES",
    "FactoredAction",
    "GameActionMask",
    "GameAdapter",
    "GameObservation",
    "GameSpec",
    "GameState",
    "Legality",
    "LiveAdapter",
    "NUM_ACTION_TYPES",
    "NUM_CENTER_KEYS",
    "Observation",
    "RandomAgent",
    "RewardFn",
    "ShapedReward",
    "Transition",
    "balatro_game_spec",
    "blind_progress_reward",
    "encode_observation",
    "engine_action_to_factored",
    "factored_to_engine_action",
    "get_action_mask",
    "get_consumable_target_info",
    "get_consumable_target_spec",
    "get_valid_target_cards",
    "sparse_reward",
    "validate_card_targets",
]
