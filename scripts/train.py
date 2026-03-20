#!/usr/bin/env python3
"""Train a Balatro RL agent.

Usage:
    # Quick test (CPU, ~2 min)
    uv run python scripts/train.py --quick

    # Standard curriculum run (GPU, ~1-2 hours)
    uv run python scripts/train.py --curriculum

    # Custom config
    uv run python scripts/train.py --timesteps 5000000 --embed-dim 256 --num-layers 4

    # Resume from checkpoint
    uv run python scripts/train.py --resume checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch

from jackdaw.env.agents import HeuristicAgent, RandomAgent, evaluate_agent
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.training.curriculum import default_curriculum
from jackdaw.env.training.ppo import EvalMetrics, PPOConfig, PPOTrainer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DECK_NAMES = {
    "b_red": "Red Deck",
    "b_blue": "Blue Deck",
    "b_yellow": "Yellow Deck",
    "b_green": "Green Deck",
    "b_black": "Black Deck",
    "b_magic": "Magic Deck",
    "b_nebula": "Nebula Deck",
    "b_ghost": "Ghost Deck",
    "b_abandoned": "Abandoned Deck",
    "b_checkered": "Checkered Deck",
    "b_zodiac": "Zodiac Deck",
    "b_painted": "Painted Deck",
    "b_anaglyph": "Anaglyph Deck",
    "b_plasma": "Plasma Deck",
    "b_erratic": "Erratic Deck",
}

STAKE_NAMES = {
    1: "White Stake",
    2: "Red Stake",
    3: "Green Stake",
    4: "Black Stake",
    5: "Blue Stake",
    6: "Purple Stake",
    7: "Orange Stake",
    8: "Gold Stake",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Balatro RL agent with PPO")

    # Core
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)

    # Architecture
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)

    # Environment
    p.add_argument("--deck", type=str, default="b_red")
    p.add_argument(
        "--stake",
        type=str,
        default="1",
        help="Stake level(s), comma-separated (e.g. '1' or '1,2,3')",
    )
    p.add_argument("--device", type=str, default="auto")

    # Training modes
    p.add_argument("--curriculum", action="store_true", help="Use default curriculum")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--quick", action="store_true", help="Quick test preset")

    # Logging / saving
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--max-checkpoints", type=int, default=5)

    # Final evaluation
    p.add_argument(
        "--final-eval-episodes",
        type=int,
        default=200,
        help="Episodes for final evaluation (0 to skip)",
    )

    return p.parse_args()


def _apply_quick_preset(args: argparse.Namespace) -> None:
    """Override args with quick-test values."""
    args.timesteps = 50_000
    args.num_envs = 2
    args.embed_dim = 32
    args.num_heads = 2
    args.num_layers = 1
    args.eval_episodes = 10
    args.save_interval = 25
    args.eval_interval = 25
    args.final_eval_episodes = 20
    if args.device == "auto":
        args.device = "cpu"


def _parse_stakes(stake_str: str) -> list[int]:
    """Parse comma-separated stake string into list of ints."""
    if stake_str == "all":
        return list(range(1, 9))
    return [int(s.strip()) for s in stake_str.split(",")]


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages periodic, best, and latest checkpoints."""

    def __init__(
        self,
        save_dir: Path,
        max_keep: int = 5,
    ) -> None:
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.best_avg_ante: float = 0.0
        self._periodic: list[Path] = []
        save_dir.mkdir(parents=True, exist_ok=True)

    def save_periodic(self, trainer: PPOTrainer, update: int) -> None:
        path = self.save_dir / f"checkpoint_{update}.pt"
        trainer.save_checkpoint(str(path), extra={"best_avg_ante": self.best_avg_ante})
        self._periodic.append(path)
        # Prune old checkpoints
        while len(self._periodic) > self.max_keep:
            old = self._periodic.pop(0)
            old.unlink(missing_ok=True)

    def save_latest(self, trainer: PPOTrainer) -> None:
        path = self.save_dir / "latest.pt"
        trainer.save_checkpoint(str(path), extra={"best_avg_ante": self.best_avg_ante})

    def save_best(self, trainer: PPOTrainer, avg_ante: float) -> bool:
        """Save if avg_ante is a new best. Returns True if saved."""
        if avg_ante > self.best_avg_ante:
            self.best_avg_ante = avg_ante
            path = self.save_dir / "best.pt"
            trainer.save_checkpoint(str(path), extra={"best_avg_ante": self.best_avg_ante})
            return True
        return False


# ---------------------------------------------------------------------------
# Banner and logging
# ---------------------------------------------------------------------------


def _print_banner(
    trainer: PPOTrainer,
    args: argparse.Namespace,
    run_name: str,
    resumed: bool = False,
) -> None:
    cfg = trainer.config
    n_params = sum(p.numel() for p in trainer.policy.parameters())

    device_str = str(trainer.device)
    if trainer.device.type == "cuda":
        device_str = f"cuda ({torch.cuda.get_device_name(trainer.device)})"

    deck_key = args.deck
    if deck_key == "all":
        deck_str = "All Decks"
    else:
        deck_str = DECK_NAMES.get(deck_key, deck_key)
    stakes = _parse_stakes(args.stake)
    if len(stakes) == 1:
        stake_str = STAKE_NAMES.get(stakes[0], f"Stake {stakes[0]}")
    else:
        stake_str = ", ".join(STAKE_NAMES.get(s, str(s)) for s in stakes)

    cur_str = "Off"
    if trainer.curriculum is not None:
        stage = trainer.curriculum.current_stage
        n_stages = len(cfg.curriculum.stages)
        cur_str = f"Stage {trainer.curriculum.stage_index + 1}/{n_stages} ({stage.name})"

    w = 50
    border = "=" * w
    print()
    print(border)
    title = "Jackdaw PPO Training"
    if resumed:
        title += " (Resumed)"
    print(f"  {title}")
    print(border)
    print(f"  Device:     {device_str}")
    print(f"  Envs:       {cfg.num_envs}")
    print(
        f"  Model:      {cfg.embed_dim}d / {cfg.num_heads}h / {cfg.num_layers}L "
        f"({n_params:,} params)"
    )
    print(f"  Timesteps:  {cfg.total_timesteps:,}")
    print(f"  Deck:       {deck_str}, {stake_str}")
    print(f"  Curriculum: {cur_str}")
    print(f"  Run:        {run_name}")
    if resumed:
        print(f"  Resumed:    step {trainer.global_step:,}, update {trainer.update_count}")
    print(border)
    print()


def _print_progress(
    trainer: PPOTrainer,
    metrics: dict[str, float],
    num_updates: int,
) -> None:
    """Print a detailed progress summary."""
    update = int(metrics["update"])
    step = int(metrics["global_step"])
    episodes = int(metrics["episodes"])
    sps = metrics["steps_per_second"]

    print(
        f"[Update {update:>5d} / {num_updates}] "
        f"step={step:>9,d}  episodes={episodes:>6,d}  SPS={sps:,.0f}"
    )
    print(
        f"  Losses: pg={metrics['policy_loss']:.4f}  "
        f"vf={metrics['value_loss']:.4f}  "
        f"ent={metrics['entropy']:.3f}  "
        f"clip={metrics['clip_fraction']:.3f}"
    )

    # Episode performance from rolling window
    if trainer._episode_antes:
        import numpy as np

        avg_ante = np.mean(trainer._episode_antes)
        max_ante = max(trainer._episode_antes)
        win_rate = np.mean(trainer._episode_wins) if trainer._episode_wins else 0.0
        print(
            f"  Perf:   avg_ante={avg_ante:.2f}  max_ante={int(max_ante)}  win_rate={win_rate:.1%}"
        )

    # Curriculum info
    if trainer.curriculum is not None:
        cm = trainer.curriculum
        stage = cm.current_stage
        cm_metrics = cm.get_metrics()
        rate = cm_metrics["curriculum/target_rate"]
        print(
            f"  Curriculum: Stage {cm.stage_index + 1} ({stage.name}) "
            f"-- {rate:.0%} reaching ante {stage.target_ante} "
            f"(target: {stage.target_rate:.0%})"
        )


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------


def _run_final_eval(
    trainer: PPOTrainer,
    n_episodes: int,
    save_dir: Path,
) -> None:
    """Run final evaluation and compare against baselines."""
    from jackdaw.env.training.ppo import _PolicyAgent

    back_key = trainer.back_keys[0]
    stake = trainer.stakes[0]
    max_steps = trainer.config.max_steps_per_episode

    print()
    print("=" * 50)
    print("  Final Evaluation")
    print("=" * 50)

    # Trained agent
    print(f"\n  Evaluating trained agent ({n_episodes} episodes)...")
    agent = _PolicyAgent(trainer.policy, trainer.device)
    trained_result = evaluate_agent(
        agent,
        n_episodes=n_episodes,
        back_key=back_key,
        stake=stake,
        max_steps=max_steps,
    )

    # Random baseline
    print(f"  Evaluating RandomAgent ({n_episodes} episodes)...")
    random_result = evaluate_agent(
        RandomAgent(),
        n_episodes=n_episodes,
        back_key=back_key,
        stake=stake,
        max_steps=max_steps,
    )

    # Heuristic baseline
    print(f"  Evaluating HeuristicAgent ({n_episodes} episodes)...")
    heuristic_result = evaluate_agent(
        HeuristicAgent(),
        n_episodes=n_episodes,
        back_key=back_key,
        stake=stake,
        max_steps=max_steps,
    )

    # Print comparison table
    print()
    header = f"  {'Agent':<20s} {'Win Rate':>10s} {'Avg Ante':>10s} {'Avg Actions':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, result in [
        ("Trained (PPO)", trained_result),
        ("HeuristicAgent", heuristic_result),
        ("RandomAgent", random_result),
    ]:
        print(
            f"  {name:<20s} {result.win_rate:>9.1%} "
            f"{result.avg_ante:>10.2f} {result.avg_actions:>12.0f}"
        )
    print()

    # Save results
    eval_data: dict[str, Any] = {}
    for name, result in [
        ("trained", trained_result),
        ("heuristic", heuristic_result),
        ("random", random_result),
    ]:
        eval_data[name] = result.summary()

    eval_path = save_dir / "eval_results.json"
    eval_path.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
    print(f"  Results saved to {eval_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.quick:
        _apply_quick_preset(args)

    # Generate run name
    run_name = args.run_name
    if run_name is None:
        deck = args.deck if args.deck != "all" else "all"
        run_name = f"balatro_{deck}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Build PPOConfig
    curriculum = default_curriculum() if args.curriculum else None

    ppo_config = PPOConfig(
        game_spec=balatro_game_spec(),
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        back_keys=args.deck,
        stake=_parse_stakes(args.stake),
        device=args.device,
        log_dir=args.log_dir,
        run_name=run_name,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        curriculum=curriculum,
    )

    # Create trainer
    trainer = PPOTrainer(ppo_config)

    # Checkpoint management
    save_dir = Path(args.save_dir) / run_name
    ckpt_mgr = CheckpointManager(save_dir, max_keep=args.max_checkpoints)

    # Resume from checkpoint
    resumed = False
    if args.resume:
        print(f"[Jackdaw] Resuming from {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        ckpt_mgr.best_avg_ante = checkpoint.get("best_avg_ante", 0.0)
        resumed = True

    _print_banner(trainer, args, run_name, resumed=resumed)

    # Graceful shutdown handler
    shutdown_requested = False

    def _signal_handler(sig: int, frame: Any) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n[Jackdaw] Force quit.")
            sys.exit(1)
        shutdown_requested = True
        print("\n[Jackdaw] Ctrl+C received — saving checkpoint and exiting...")

    signal.signal(signal.SIGINT, _signal_handler)

    # --- Training loop ---
    cfg = trainer.config
    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    start_update = trainer.update_count
    start_time = time.time()

    # Initial reset
    reset_data = trainer.envs.reset()
    trainer._current_obs = [d[0] for d in reset_data]
    trainer._current_masks = [d[1] for d in reset_data]
    trainer._current_infos = [d[2] for d in reset_data]

    for update in range(start_update + 1, num_updates + 1):
        if shutdown_requested:
            break

        # Collect rollouts
        buffer = trainer._collect_rollouts()

        # Update policy
        update_metrics = trainer._update(buffer)
        trainer.update_count = update

        # Logging
        if update % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            total_elapsed = elapsed  # approximate
            sps = (trainer.global_step - (start_update * cfg.num_steps * cfg.num_envs)) / max(
                elapsed, 1e-8
            )
            metrics = {
                "update": float(update),
                "global_step": float(trainer.global_step),
                "episodes": float(trainer.episode_count),
                "steps_per_second": sps,
                "wall_time": total_elapsed,
                **update_metrics,
            }
            _print_progress(trainer, metrics, num_updates)

            if trainer.writer is not None:
                gs = trainer.global_step
                trainer.writer.add_scalar("charts/SPS", sps, gs)
                trainer.writer.add_scalar("charts/episodes", trainer.episode_count, gs)
                um = update_metrics
                trainer.writer.add_scalar("losses/policy_loss", um["policy_loss"], gs)
                trainer.writer.add_scalar("losses/value_loss", um["value_loss"], gs)
                trainer.writer.add_scalar("losses/entropy", um["entropy"], gs)
                trainer.writer.add_scalar("losses/total_loss", um["total_loss"], gs)
                trainer.writer.add_scalar("losses/clip_fraction", um["clip_fraction"], gs)
                trainer.writer.add_scalar("losses/approx_kl", um["approx_kl"], gs)
                trainer.writer.add_scalar("losses/explained_variance", um["explained_variance"], gs)
                trainer.writer.add_scalar("losses/learning_rate", um["learning_rate"], gs)

                if trainer.curriculum is not None:
                    for k, v in trainer.curriculum.get_metrics().items():
                        trainer.writer.add_scalar(k, v, gs)

        # Evaluation
        if update % cfg.eval_interval == 0:
            eval_metrics = trainer._evaluate()
            trainer._log_eval(eval_metrics, update)

            if trainer.writer is not None:
                gs = trainer.global_step
                trainer.writer.add_scalar("eval/win_rate", eval_metrics.win_rate, gs)
                trainer.writer.add_scalar("eval/avg_ante", eval_metrics.avg_ante, gs)
                trainer.writer.add_scalar("eval/avg_length", eval_metrics.avg_length, gs)

            # Best model tracking
            is_best = ckpt_mgr.save_best(trainer, eval_metrics.avg_ante)
            if is_best:
                print(
                    f"  ** New best model: avg_ante={eval_metrics.avg_ante:.2f} "
                    f"(was {ckpt_mgr.best_avg_ante:.2f})"
                )

        # Periodic checkpoint
        if update % cfg.save_interval == 0:
            ckpt_mgr.save_periodic(trainer, update)
            ckpt_mgr.save_latest(trainer)

    # --- End of training ---
    ckpt_mgr.save_latest(trainer)

    if trainer.writer is not None:
        trainer.writer.close()

    elapsed = time.time() - start_time
    print()
    print("=" * 50)
    print("  Training Complete")
    print("=" * 50)
    print(f"  Total steps:    {trainer.global_step:,}")
    print(f"  Total updates:  {trainer.update_count}")
    print(f"  Total episodes: {trainer.episode_count:,}")
    print(f"  Wall time:      {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    if elapsed > 0:
        print(f"  Avg SPS:        {trainer.global_step / elapsed:,.0f}")
    print(f"  Best avg_ante:  {ckpt_mgr.best_avg_ante:.2f}")
    print(f"  Checkpoints:    {save_dir}")
    print()

    # Final evaluation
    if args.final_eval_episodes > 0 and not shutdown_requested:
        _run_final_eval(trainer, args.final_eval_episodes, save_dir)


if __name__ == "__main__":
    main()
