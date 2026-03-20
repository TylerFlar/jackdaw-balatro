"""Gymnasium environment for Balatro RL research.

Quick start::

    from jackdaw.env import DirectAdapter, RandomAgent

    env = DirectAdapter()
    agent = RandomAgent()

    env.reset(back_key="b_red", stake=1, seed="MYSEED")
    agent.reset()

    while not env.done:
        gs = env.raw_state
        mask = get_action_mask(gs)
        info = {"raw_state": gs, "legal_actions": env.get_legal_actions()}
        fa = agent.act({}, mask, info)
        engine_action = factored_to_engine_action(fa, gs)
        env.step(engine_action)

Architecture::

    Agent -> GameAdapter -> DirectAdapter  -> Engine      (training, fast)
                         -> BridgeAdapter  -> SimBackend  (bridge validation)

    Observation: entity-based, variable-length
        (hand_cards, jokers, consumables, shop, pack) + global context
    Action: factored 21-type with entity selection and card targeting
    Reward: configurable (dense for training, sparse for evaluation)
"""

from jackdaw.env.action_space import (
    NUM_ACTION_TYPES,
    ActionMask,
    ActionType,
    FactoredAction,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
    get_consumable_target_info,
)
from jackdaw.env.agents import Agent, RandomAgent
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
    DenseRewardWrapper,
    RewardCalculator,
    RewardConfig,
    SparseRewardWrapper,
)

__all__ = [
    "ActionMask",
    "ActionType",
    "Agent",
    "BridgeAdapter",
    "ConsumableTargetSpec",
    "D_CONSUMABLE",
    "D_GLOBAL",
    "D_JOKER",
    "D_PLAYING_CARD",
    "D_SHOP",
    "DenseRewardWrapper",
    "DirectAdapter",
    "FactoredAction",
    "GameAdapter",
    "GameState",
    "NUM_ACTION_TYPES",
    "NUM_CENTER_KEYS",
    "Observation",
    "RandomAgent",
    "RewardCalculator",
    "RewardConfig",
    "SparseRewardWrapper",
    "encode_observation",
    "engine_action_to_factored",
    "factored_to_engine_action",
    "get_action_mask",
    "get_consumable_target_info",
    "get_consumable_target_spec",
    "get_valid_target_cards",
    "validate_card_targets",
]
