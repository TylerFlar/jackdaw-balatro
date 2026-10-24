"""Train MaskablePPO (sb3-contrib) on jackdaw's Balatro simulator.

    uv sync --group train
    uv run --no-sync python examples/train_maskable_ppo.py

Writes ``runs/maskable_ppo/<run>/``: TensorBoard logs (``tb/``), checkpoints
(``checkpoints/``), ``best_model.zip`` (most blinds beaten on the validation seeds),
``final_model.zip`` and ``config.json``.  ``runs/`` is gitignored.  Creating a file named
``STOP`` in the run directory ends training early and still saves the final model.  The
defaults are the demo run's settings; see examples/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

#: Validation seeds for the periodic evaluation during training (the final evaluation in
#: evaluate_agents.py uses other seeds).
VALIDATION_SEED_BASE = 20_000_000

#: ShapedReward, progress-heavy and without the loss penalty.  With the default weights a
#: policy gains loss * (1 - gamma) per step by putting off the end of a lost run, which
#: matches step_cost at gamma = 0.995: free moves (swapping or sorting cards) stall for nothing.
REWARD = "shaped:loss=0,score_progress=0.1,blind_beaten=0.5,boss_beaten=0.3"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--timesteps", type=int, default=60_000_000, help="env steps to train for")
    p.add_argument("--n-envs", type=int, default=32, help="parallel envs (BalatroVectorEnv)")
    p.add_argument("--n-steps", type=int, default=256, help="steps per env per rollout")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--net", default="256,256", help="hidden layers of the policy and value MLPs")
    p.add_argument(
        "--reward",
        default=REWARD,
        help='"shaped" (ShapedReward\'s defaults), "blind_progress", "sparse", or "shaped:" '
        "with weights to change (default: %(default)s)",
    )
    p.add_argument("--max-steps", type=int, default=2_000, help="episode cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=6, help="torch / BLAS threads (cap)")
    p.add_argument("--env-threads", type=int, default=2, help="the vector env's threads")
    p.add_argument("--run-dir", default="runs/maskable_ppo")
    p.add_argument("--run-name", default=None, help="default: seed<seed>-<date>-<time>")
    p.add_argument("--checkpoint-freq", type=int, default=5_000_000, help="in env steps")
    p.add_argument("--eval-freq", type=int, default=2_000_000, help="in env steps")
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--resume", default=None, help="a saved model .zip to continue training")
    return p.parse_args(argv)


def cap_threads(n: int) -> None:
    """Cap the BLAS and OpenMP thread pools; call before numpy or torch is imported."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cap_threads(args.threads)

    # numpy, torch and the env load after the thread caps are set.
    import balatro_sb3 as demo
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    run_name = args.run_name or time.strftime(f"seed{args.seed}-%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    from jackdaw.env import ACTION_VERSION

    # The action space the policy learns: evaluate_agents.py refuses a model on another one.
    config = {**vars(args), "action_version": ACTION_VERSION}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    env = demo.make_training_env(
        args.n_envs, args.seed, args.reward, args.max_steps, args.env_threads
    )
    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device="cpu")
        model.tensorboard_log = str(run_dir / "tb")
    else:
        hidden = [int(n) for n in args.net.split(",")]
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            policy_kwargs={
                "net_arch": {"pi": hidden, "vf": hidden},
                "activation_fn": torch.nn.ReLU,
            },
            tensorboard_log=str(run_dir / "tb"),
            stats_window_size=200,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
    print(model.policy, flush=True)

    validation_seeds = [VALIDATION_SEED_BASE + k for k in range(args.eval_episodes)]
    callbacks = CallbackList(
        [
            demo.BalatroMetricsCallback(),
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq // args.n_envs),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="maskable_ppo",
            ),
            demo.FixedSeedEvalCallback(
                validation_seeds,
                eval_freq=args.eval_freq,
                save_dir=run_dir,
                reward=args.reward,
                max_steps=args.max_steps,
            ),
            demo.StopFileCallback(run_dir / "STOP"),
        ]
    )

    start = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="MaskablePPO",
            reset_num_timesteps=not args.resume,
        )
    finally:
        model.save(run_dir / "final_model")
        minutes = (time.perf_counter() - start) / 60
        print(f"{model.num_timesteps:,} steps in {minutes:.1f} min; saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
