"""Train MaskablePPO on the Balatro gymnasium environment.

Requires the ``train`` optional dependency group::

    uv pip install -e ".[train]"

Usage::

    python scripts/train_ppo.py --total-timesteps 100000
    python scripts/train_ppo.py --total-timesteps 50000 --log-dir runs/exp1 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


def make_env(seed: int = 0, max_steps: int = 10_000) -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=max_steps,
        seed_prefix=f"PPO_{seed}",
        reward_shaping=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Balatro")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--log-dir", type=str, default="runs/balatro_ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    env = make_env(seed=args.seed, max_steps=args.max_steps)

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(log_path),
    )

    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)

    save_path = log_path / "balatro_ppo"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
