"""Performance profiling tools for the env layer.

Measures time spent in each component of the env step loop to ensure
the env layer does not bottleneck the engine (target: <1ms total
overhead per env.step excluding engine time).

Usage::

    python -m jackdaw.env.profiling
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jackdaw.env.action_space import (
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.observation import encode_observation
from jackdaw.env.rewards import DenseRewardWrapper


def _resolve_action(action: Any, gs: dict[str, Any]) -> Any:
    """Fill in card indices for marker PlayHand/Discard actions."""
    import random as _rng

    from jackdaw.engine.actions import Discard, PlayHand

    hand = gs.get("hand", [])
    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = _rng.randint(1, n)
        indices = tuple(sorted(_rng.sample(range(len(hand)), count)))
        return PlayHand(card_indices=indices)
    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = _rng.randint(1, n)
        indices = tuple(sorted(_rng.sample(range(len(hand)), count)))
        return Discard(card_indices=indices)
    return action

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class TimingSummary:
    """Summary statistics for a set of timing measurements."""

    name: str
    samples: list[float] = field(default_factory=list, repr=False)

    @property
    def mean_us(self) -> float:
        return statistics.mean(self.samples) * 1e6 if self.samples else 0.0

    @property
    def p50_us(self) -> float:
        return _percentile(self.samples, 50) * 1e6

    @property
    def p95_us(self) -> float:
        return _percentile(self.samples, 95) * 1e6

    @property
    def p99_us(self) -> float:
        return _percentile(self.samples, 99) * 1e6

    def __str__(self) -> str:
        return (
            f"  {self.name:<30s}  "
            f"mean={self.mean_us:8.1f}us  "
            f"p50={self.p50_us:8.1f}us  "
            f"p95={self.p95_us:8.1f}us  "
            f"p99={self.p99_us:8.1f}us"
        )


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def _time_call(fn, *args, **kwargs) -> tuple[Any, float]:
    """Call fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Helper: run game steps and collect game states
# ---------------------------------------------------------------------------


def _collect_game_states(n_steps: int = 1000) -> list[dict[str, Any]]:
    """Play a game with random actions and collect game states at each step."""
    import random

    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "PROFILE_SEED_42")
    states: list[dict[str, Any]] = []

    for _ in range(n_steps * 3):  # overshoot to collect enough non-terminal states
        if adapter.done:
            adapter.reset("b_red", 1, f"PROFILE_SEED_{random.randint(0, 99999)}")

        # Snapshot state before step
        gs = adapter.raw_state
        states.append(gs)
        if len(states) >= n_steps:
            break

        # Take a random legal action
        legal = adapter.get_legal_actions()
        if not legal:
            adapter.reset("b_red", 1, f"PROFILE_SEED_{random.randint(0, 99999)}")
            continue
        action = _resolve_action(random.choice(legal), gs)
        adapter.step(action)

    return states[:n_steps]


# ---------------------------------------------------------------------------
# profile_env_step
# ---------------------------------------------------------------------------


def profile_env_step(n_steps: int = 1000) -> dict[str, TimingSummary]:
    """Profile each component of an env step.

    Measures:
      - engine.step()
      - encode_observation()
      - get_action_mask()
      - reward calculation
      - action conversion (FactoredAction -> engine Action)

    Returns dict of TimingSummary keyed by component name.
    """
    import random

    adapter = DirectAdapter()
    reward_wrapper = DenseRewardWrapper()

    t_engine = TimingSummary("engine_step")
    t_encode = TimingSummary("encode_observation")
    t_mask = TimingSummary("get_action_mask")
    t_reward = TimingSummary("reward_calculation")
    t_action_conv = TimingSummary("action_conversion")
    t_total = TimingSummary("total_env_step")

    adapter.reset("b_red", 1, "PROFILE_SEED_0")
    reward_wrapper.reset()
    steps_done = 0

    while steps_done < n_steps:
        if adapter.done:
            adapter.reset("b_red", 1, f"PROFILE_SEED_{random.randint(0, 99999)}")
            reward_wrapper.reset()

        gs_prev = adapter.raw_state

        # Get legal actions and pick one randomly
        legal = adapter.get_legal_actions()
        if not legal:
            adapter.reset("b_red", 1, f"PROFILE_SEED_{random.randint(0, 99999)}")
            reward_wrapper.reset()
            continue

        engine_action = _resolve_action(random.choice(legal), gs_prev)

        # Convert to factored and back (measure action conversion)
        from jackdaw.env.action_space import engine_action_to_factored

        try:
            fa, dt_conv1 = _time_call(engine_action_to_factored, engine_action, gs_prev)
        except ValueError:
            # Some engine actions (complex permutations) can't be factored
            from jackdaw.env.action_space import ActionType, FactoredAction

            fa = FactoredAction(action_type=ActionType.SelectBlind)
            dt_conv1 = 0.0

        t_total_start = time.perf_counter()

        # Engine step
        _, dt_eng = _time_call(adapter.step, engine_action)
        t_engine.samples.append(dt_eng)

        gs = adapter.raw_state

        # Encode observation
        _, dt_enc = _time_call(encode_observation, gs)
        t_encode.samples.append(dt_enc)

        # Action mask
        _, dt_mask = _time_call(get_action_mask, gs)
        t_mask.samples.append(dt_mask)

        # Reward
        _, dt_rew = _time_call(reward_wrapper.reward, gs_prev, fa, gs)
        t_reward.samples.append(dt_rew)

        # Action conversion (factored -> engine)
        try:
            _, dt_conv2 = _time_call(factored_to_engine_action, fa, gs_prev)
            t_action_conv.samples.append(dt_conv1 + dt_conv2)
        except (ValueError, IndexError):
            t_action_conv.samples.append(dt_conv1)

        t_total.samples.append(time.perf_counter() - t_total_start)
        steps_done += 1

    # Report
    results = {
        "engine_step": t_engine,
        "encode_observation": t_encode,
        "get_action_mask": t_mask,
        "reward_calculation": t_reward,
        "action_conversion": t_action_conv,
        "total_env_step": t_total,
    }

    total_wall = sum(t_total.samples)
    sps = n_steps / total_wall if total_wall > 0 else 0

    print(f"  Steps: {n_steps}")
    print(f"  Total wall time: {total_wall:.3f}s")
    print(f"  Steps/sec: {sps:.0f}")
    print()
    for ts in results.values():
        print(ts)

    overhead_us = t_encode.mean_us + t_mask.mean_us + t_reward.mean_us + t_action_conv.mean_us
    print(f"\n  Env overhead (excl engine): {overhead_us:.1f}us mean")
    print(f"  Target: <1000us  {'PASS' if overhead_us < 1000 else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
# profile_encoding
# ---------------------------------------------------------------------------


def profile_encoding(n_steps: int = 1000) -> dict[str, TimingSummary]:
    """Drill into encode_observation timing.

    Measures:
      - global_context encoding time
      - Per-entity-type encoding time (hand, jokers, consumables, shop, pack)
      - numpy array allocation overhead
    """
    from jackdaw.env.observation import (
        encode_consumable,
        encode_global_context,
        encode_jokers_batch,
        encode_playing_cards_batch,
        encode_shop_item,
    )

    states = _collect_game_states(n_steps)

    t_global = TimingSummary("global_context")
    t_hand = TimingSummary("hand_cards_batch")
    t_jokers = TimingSummary("jokers_batch")
    t_consumables = TimingSummary("consumables")
    t_shop = TimingSummary("shop_items")
    t_pack = TimingSummary("pack_cards_batch")
    t_total = TimingSummary("total_encode")

    hand_counts: list[int] = []
    joker_counts: list[int] = []

    for gs in states:
        t0 = time.perf_counter()

        _, dt = _time_call(encode_global_context, gs)
        t_global.samples.append(dt)

        hand = gs.get("hand", [])
        hand_counts.append(len(hand))
        _, dt = _time_call(encode_playing_cards_batch, hand, gs)
        t_hand.samples.append(dt)

        jokers = gs.get("jokers", [])
        joker_counts.append(len(jokers))
        _, dt = _time_call(encode_jokers_batch, jokers, gs)
        t_jokers.samples.append(dt)

        consumables = gs.get("consumables", [])
        if consumables:
            _, dt = _time_call(
                lambda c, g: np.stack([encode_consumable(x, g) for x in c]),
                consumables,
                gs,
            )
        else:
            dt = 0.0
        t_consumables.samples.append(dt)

        shop_items = (
            gs.get("shop_cards", []) + gs.get("shop_vouchers", []) + gs.get("shop_boosters", [])
        )
        if shop_items:
            _, dt = _time_call(
                lambda items, g: np.stack([encode_shop_item(x, g) for x in items]),
                shop_items,
                gs,
            )
        else:
            dt = 0.0
        t_shop.samples.append(dt)

        pack_cards = gs.get("pack_cards", [])
        _, dt = _time_call(encode_playing_cards_batch, pack_cards, gs)
        t_pack.samples.append(dt)

        t_total.samples.append(time.perf_counter() - t0)

    results = {
        "global_context": t_global,
        "hand_cards_batch": t_hand,
        "jokers_batch": t_jokers,
        "consumables": t_consumables,
        "shop_items": t_shop,
        "pack_cards_batch": t_pack,
        "total_encode": t_total,
    }

    print(f"  Samples: {len(states)}")
    print(f"  Avg hand size: {statistics.mean(hand_counts):.1f}")
    print(f"  Avg joker count: {statistics.mean(joker_counts):.1f}")
    print()
    for ts in results.values():
        print(ts)

    return results


# ---------------------------------------------------------------------------
# profile_training_step
# ---------------------------------------------------------------------------


def profile_training_step() -> dict[str, float]:
    """Profile one PPO training update end-to-end.

    Measures:
      - Rollout collection time (env interaction)
      - GAE computation
      - Forward pass time
      - Backward pass time
      - Optimizer step time
      - GPU utilization if CUDA available
    """
    try:
        import torch
    except ImportError:
        print("  PyTorch not installed — skipping training profiling")
        return {}

    from jackdaw.env.training.ppo import PPOConfig, PPOTrainer

    config = PPOConfig(
        num_envs=2,
        num_steps=32,
        total_timesteps=64,
        update_epochs=1,
        num_minibatches=2,
        embed_dim=64,
        num_heads=2,
        num_layers=1,
    )

    trainer = PPOTrainer(config)
    trainer.policy.train()

    # Initial reset
    reset_data = trainer.envs.reset()
    trainer._current_obs = [d[0] for d in reset_data]
    trainer._current_masks = [d[1] for d in reset_data]
    trainer._current_infos = [d[2] for d in reset_data]

    results: dict[str, float] = {}

    # Rollout collection
    t0 = time.perf_counter()
    buffer = trainer._collect_rollouts()
    results["rollout_collection_ms"] = (time.perf_counter() - t0) * 1000

    # GAE computation
    with torch.no_grad():
        from jackdaw.env.policy.policy import PolicyInput, collate_policy_inputs

        policy_inputs = [
            PolicyInput(obs=obs, action_mask=mask)
            for obs, mask in zip(trainer._current_obs, trainer._current_masks)
        ]
        batch = collate_policy_inputs(policy_inputs, device=trainer.device)
        out = trainer.policy.forward(batch)
        last_value = out["value"].squeeze(-1).cpu()

    last_done = np.zeros(config.num_envs, dtype=np.bool_)
    t0 = time.perf_counter()
    buffer.compute_returns(last_value, last_done, config.gamma, config.gae_lambda)
    results["gae_computation_ms"] = (time.perf_counter() - t0) * 1000

    # PPO update (forward + backward + optimizer)
    trainer.policy.train()
    t_forward_total = 0.0
    t_backward_total = 0.0
    t_optim_total = 0.0
    n_batches = 0

    for mb in buffer.get_batches(config.num_minibatches, trainer.device):
        t0 = time.perf_counter()
        new_log_probs, entropy, new_values = trainer.policy.evaluate_actions(mb.batch, mb.actions)
        advantages = mb.advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_ratio = new_log_probs - mb.old_log_probs
        ratio = log_ratio.exp()
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        vf_loss = 0.5 * ((new_values - mb.returns) ** 2).mean()
        ent_loss = entropy.mean()
        loss = pg_loss - config.ent_coef * ent_loss + config.vf_coef * vf_loss
        t_forward_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        trainer.optimizer.zero_grad()
        loss.backward()
        t_backward_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), config.max_grad_norm)
        trainer.optimizer.step()
        t_optim_total += time.perf_counter() - t0

        n_batches += 1

    results["forward_pass_ms"] = t_forward_total * 1000
    results["backward_pass_ms"] = t_backward_total * 1000
    results["optimizer_step_ms"] = t_optim_total * 1000
    results["n_minibatches"] = float(n_batches)

    # GPU utilization
    if torch.cuda.is_available():
        results["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        results["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
        results["gpu_utilization"] = -1.0  # nvidia-smi needed for real util

    total = sum(v for k, v in results.items() if k.endswith("_ms"))
    results["total_training_step_ms"] = total

    print(f"  Rollout collection:  {results['rollout_collection_ms']:8.1f}ms")
    print(f"  GAE computation:     {results['gae_computation_ms']:8.1f}ms")
    print(f"  Forward pass:        {results['forward_pass_ms']:8.1f}ms")
    print(f"  Backward pass:       {results['backward_pass_ms']:8.1f}ms")
    print(f"  Optimizer step:      {results['optimizer_step_ms']:8.1f}ms")
    print(f"  Total:               {total:8.1f}ms  ({n_batches} minibatches)")

    if torch.cuda.is_available():
        print(f"  GPU memory alloc:    {results['gpu_memory_allocated_mb']:.1f}MB")
        print(f"  GPU memory reserved: {results['gpu_memory_reserved_mb']:.1f}MB")

    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Env Step Profiling ===")
    profile_env_step()
    print("\n=== Encoding Profiling ===")
    profile_encoding()
    print("\n=== Training Step Profiling ===")
    profile_training_step()
