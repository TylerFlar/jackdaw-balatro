"""Evaluate a trained MaskablePPO model against two baselines on the same seeds.

    uv run --no-sync python examples/evaluate_agents.py \
        --model runs/maskable_ppo/<run>/final_model.zip --out examples/results/eval_results.json

Agents, each playing one episode from every seed ``EVAL_SEED_BASE + k`` (k < --episodes):

- ``ppo_greedy``: the model's most likely legal action;
- ``ppo_sampled``: an action sampled from the model's masked distribution, as in training;
- ``random``: a uniformly random legal action;
- ``smart``: ``jackdaw.cli.smart_agent``, the repo's hand-written agent.  It reads the Python
  engine's state, so it plays through ``OracleAdapter``, which plays exactly as the Rust engine.
  It picks from the same action table as the other agents.

All agents use the reward and episode cap the model was trained with (``config.json`` next to
the model, else the flags).  Prints a Markdown table and writes the summary as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="a MaskablePPO .zip")
    p.add_argument("--episodes", type=int, default=1000, help="seeds per agent")
    p.add_argument("--smart-episodes", type=int, default=None, help="default: --episodes")
    p.add_argument(
        "--agents", default="ppo_greedy,ppo_sampled,random,smart", help="comma-separated"
    )
    p.add_argument("--reward", default=None, help="default: the model's config.json")
    p.add_argument("--max-steps", type=int, default=None, help="default: the model's config")
    p.add_argument("--seed", type=int, default=0, help="seeds the sampled and random agents")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None, help="write the summary JSON here")
    return p.parse_args(argv)


def cap_threads(n: int) -> None:
    """Cap the BLAS and OpenMP thread pools; call before numpy or torch is imported."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)


def smart_chooser() -> Any:
    """``jackdaw.cli.smart_agent`` as a chooser over the env's action table (OracleAdapter)."""
    from jackdaw.cli.smart_agent import smart_agent
    from jackdaw.engine import actions as A
    from jackdaw.env import FactoredAction, engine_action_to_factored

    def with_forced_card(action: Any, gs: dict[str, Any]) -> Any:
        # Cerulean Bell: vanilla always selects the forced card (as tests/golden does).
        if not isinstance(action, (A.PlayHand, A.Discard)):
            return action
        hand = gs.get("hand", [])
        forced = [i for i, c in enumerate(hand) if c.ability.get("forced_selection")]
        if not forced or forced[0] in action.card_indices:
            return action
        kept = list(action.card_indices)[:4]
        return type(action)(card_indices=tuple(sorted({*kept, forced[0]})))

    def closest(table: list[FactoredAction], wanted: FactoredAction) -> int:
        """The table entry for ``wanted``, else the nearest one of its type (same target,
        most cards in common): the table holds at most 200 card selections per action."""
        want = set(wanted.card_target or ())
        best, best_score = None, None
        for i, a in enumerate(table):
            if a.action_type != wanted.action_type:
                continue
            have = set(a.card_target or ())
            score = (a.entity_target == wanted.entity_target, len(want & have) - len(want ^ have))
            if best_score is None or score > best_score:
                best, best_score = i, score
        return 0 if best is None else best

    def choose(envs: list[Any], obs: Any, masks: Any) -> list[int]:
        actions = []
        for env in envs:
            gym_env = env.unwrapped
            adapter = gym_env._inner._adapter  # OracleAdapter: the Python engine's state
            gs = adapter.raw_state
            legal = adapter.get_legal_actions()
            table = list(gym_env._action_table)
            wanted = engine_action_to_factored(with_forced_card(smart_agent(gs, legal), gs), gs)
            if wanted.card_target == ():
                wanted = FactoredAction(wanted.action_type, None, wanted.entity_target)
            actions.append(table.index(wanted) if wanted in table else closest(table, wanted))
        return actions

    return choose


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cap_threads(args.threads)

    import balatro_sb3 as demo
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO

    from jackdaw.env import OracleAdapter

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    model_path = Path(args.model)
    config_path = model_path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    reward = args.reward or config.get("reward", "shaped")
    max_steps = args.max_steps or config.get("max_steps", demo.MAX_STEPS)
    model = MaskablePPO.load(model_path, device="cpu")

    seeds = [demo.EVAL_SEED_BASE + k for k in range(args.episodes)]
    smart_seeds = seeds[: args.smart_episodes or args.episodes]
    agents = {
        "ppo_greedy": (demo.model_chooser(model, deterministic=True), seeds, {}),
        "ppo_sampled": (demo.model_chooser(model, deterministic=False), seeds, {}),
        "random": (demo.random_chooser(args.seed), seeds, {}),
        "smart": (smart_chooser(), smart_seeds, {"adapter_factory": OracleAdapter}),
    }
    summary: dict[str, Any] = {
        "model": model_path.as_posix(),
        "model_timesteps": int(model.num_timesteps),
        "reward": reward,
        "max_steps": max_steps,
        "seeds": f"{seeds[0]}..{seeds[-1]}",
        "agents": {},
    }
    for name in args.agents.split(","):
        choose, agent_seeds, kwargs = agents[name]
        start = time.perf_counter()
        results = demo.run_episodes(
            choose, agent_seeds, reward=reward, max_steps=max_steps, **kwargs
        )
        stats = demo.summarize(results)
        blinds = np.array([r.blinds_beaten for r in results], dtype=float)
        stats["blinds_beaten_sem"] = float(blinds.std(ddof=1) / np.sqrt(len(blinds)))
        stats["max_ante"] = float(max(r.ante_reached for r in results))
        stats["seconds"] = time.perf_counter() - start
        summary["agents"][name] = stats
        print(f"{name:12s} {demo.format_summary(stats)} ({stats['seconds']:.0f} s)", flush=True)

    print()
    print("| Agent | Episodes | Mean ante reached | Blinds beaten | Win rate | Mean reward | "
          f"Episode length | Hit the {max_steps:,}-step cap |")  # fmt: skip
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in summary["agents"].items():
        print(
            f"| {name} | {s['episodes']:.0f} | {s['ante_reached']:.2f} | "
            f"{s['blinds_beaten']:.2f} ± {s['blinds_beaten_sem']:.2f} | "
            f"{100 * s['win_rate']:.1f}% | {s['reward']:+.3f} | {s['length']:.1f} | "
            f"{100 * s['truncated']:.1f}% |"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
