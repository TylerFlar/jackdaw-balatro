"""Train Balatro agent with factored policy PPO.

Uses a structured action decomposition (type → entity pointer → card selection)
instead of the flat Discrete(500) enumeration used by SB3's MaskablePPO.

Requires the ``train`` optional dependency group::

    uv sync --extra train

Usage::

    python scripts/train_factored.py --total-timesteps 1000000
    python scripts/train_factored.py --n-envs 8 --total-timesteps 50000000
    python scripts/train_factored.py --resume runs/balatro_factored/checkpoint_1000000.pt
"""

from __future__ import annotations

import argparse
import functools
import random
from pathlib import Path

import numpy as np
import torch

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.rl.env_wrapper import FactoredBalatroEnv
from jackdaw.rl.network import FactoredPolicy
from jackdaw.rl.trainer import BalatroTrainer
from jackdaw.rl.vec_env import SubprocVecEnv


def _make_env(max_steps: int, seed_prefix: str, env_idx: int) -> FactoredBalatroEnv:
    """Top-level factory function (must be picklable for multiprocessing)."""
    return FactoredBalatroEnv(
        adapter_factory=DirectAdapter,
        reward_shaping=True,
        max_steps=max_steps,
        seed_prefix=f"{seed_prefix}_ENV{env_idx}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Balatro with factored policy PPO")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--log-dir", type=str, default="runs/balatro_factored")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ent-coef", type=float, default=0.2)
    parser.add_argument("--entropy-target", type=float, default=2.0)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create env factories (functools.partial is picklable, lambdas are not)
    seed_prefix = f"FACTORED_{args.seed}"
    env_fns = [
        functools.partial(_make_env, args.max_steps, seed_prefix, i) for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)

    network = FactoredPolicy()

    trainer = BalatroTrainer(
        vec_env=vec_env,
        network=network,
        lr=args.lr,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        entropy_target=args.entropy_target,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        log_dir=args.log_dir,
        total_timesteps=args.total_timesteps,
        checkpoint_interval=args.checkpoint_interval,
    )

    resume_step = 0
    if args.resume:
        resume_step = trainer.load_checkpoint(args.resume)

    trainer.train(total_timesteps=args.total_timesteps, resume_step=resume_step)

    # Save model
    save_path = args.save_path or str(log_path / "factored_policy.pt")
    torch.save(network.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
