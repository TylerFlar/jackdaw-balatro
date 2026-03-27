"""Run a trained factored policy on a single Balatro game.

Usage::

    # Against the internal engine (fast, no Balatro needed)
    python scripts/play.py runs/balatro_factored/checkpoint_1000000.pt
    python scripts/play.py checkpoint.pt --n-games 10 --verbose

    # Against live Balatro (requires balatrobot running)
    python scripts/play.py checkpoint.pt --live
    python scripts/play.py checkpoint.pt --live --host 127.0.0.1 --port 12346
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.game_spec import FactoredAction
from jackdaw.rl.env_wrapper import FactoredBalatroEnv
from jackdaw.rl.network import (
    ENTITY_MAX_COUNTS,
    NEEDS_CARDS,
    NEEDS_ENTITY,
    FactoredPolicy,
)

_SPEC = balatro_game_spec()
HAND_CARD_MAX = _SPEC.entity_types[0].max_count


def _pad_mask(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[:target_len].astype(bool)
    padded = np.zeros(target_len, dtype=bool)
    padded[: len(arr)] = arr
    return padded


def _obs_to_device(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in obs.items()}


def _masks_to_device(mask, device: torch.device) -> dict:
    type_mask = mask.type_mask.astype(bool)
    card_mask = _pad_mask(mask.card_mask, HAND_CARD_MAX)

    entity_masks = {}
    for atype, emask in mask.entity_masks.items():
        etype_idx = _SPEC.entity_type_for_action(atype)
        if etype_idx >= 0:
            entity_masks[atype] = (
                torch.from_numpy(_pad_mask(emask, ENTITY_MAX_COUNTS[etype_idx]))
                .bool()
                .unsqueeze(0)
                .to(device)
            )

    return {
        "type_mask": torch.from_numpy(type_mask).bool().unsqueeze(0).to(device),
        "card_mask": torch.from_numpy(card_mask).bool().unsqueeze(0).to(device),
        "entity_masks": entity_masks,
        "min_card_select": torch.tensor([mask.min_card_select], dtype=torch.long, device=device),
        "max_card_select": torch.tensor([mask.max_card_select], dtype=torch.long, device=device),
    }


def play_game(
    network: FactoredPolicy,
    device: torch.device,
    adapter_factory=None,
    seed: str = "PLAY_0",
    max_steps: int = 10_000,
    verbose: bool = False,
    live: bool = False,
) -> dict:
    """Play a single game with the trained policy. Returns game stats."""
    if adapter_factory is None:
        adapter_factory = DirectAdapter
    env = FactoredBalatroEnv(
        adapter_factory=adapter_factory,
        reward_shaping=False,
        max_steps=max_steps,
        seed_prefix=seed,
    )
    # Disable episode truncation for evaluation
    env.max_episode_steps = max_steps

    network.eval()
    obs, mask, info = env.reset()
    total_reward = 0.0
    step_count = 0
    prev_ante = 1
    prev_round = 0
    errors_in_a_row = 0

    while True:
        obs_t = _obs_to_device(obs, device)
        masks_t = _masks_to_device(mask, device)

        with torch.no_grad():
            out = network(obs_t, masks_t)

        action_type = out["action_type"].item()
        entity_target = out["entity_target"].item()
        card_target_arr = out["card_target"][0].cpu().numpy()
        value = out["value"].item()

        ct = None
        et = None
        if action_type in NEEDS_ENTITY and entity_target >= 0:
            et = entity_target
        if action_type in NEEDS_CARDS:
            selected = np.nonzero(card_target_arr)[0]
            if len(selected) > 0:
                ct = tuple(int(i) for i in selected)

        fa = FactoredAction(action_type=action_type, card_target=ct, entity_target=et)

        if live and verbose:
            from jackdaw.env.action_space import ActionType

            at_name = ActionType(action_type).name if action_type < 21 else str(action_type)
            print(f"    action={at_name} entity={et} cards={ct}")

        try:
            next_obs, reward, terminated, truncated, next_mask, info = env.step(fa)
            errors_in_a_row = 0
        except Exception as e:
            if not live:
                raise
            errors_in_a_row += 1
            if verbose:
                print(f"  [!] Action rejected: {e} (retrying with re-fetched state)")
            if errors_in_a_row > 10:
                print("  Too many consecutive errors, aborting game")
                return {
                    "won": False,
                    "ante_reached": prev_ante,
                    "rounds_beaten": prev_round,
                    "steps": step_count,
                    "reward": total_reward,
                }
            # Re-fetch state from the live game and retry
            try:
                reobs, remask, reinfo = env._inner.reobserve()
                from jackdaw.rl.env_wrapper import _remap_shop_masks

                shop_splits = reinfo.get("shop_splits", (0, 0, 0))
                remask = _remap_shop_masks(remask, shop_splits)
                obs = env._build_obs(reobs)
                mask = remask
            except Exception:
                pass
            time.sleep(0.2)
            continue

        total_reward += reward
        step_count += 1
        done = terminated or truncated

        # Track progress for verbose output
        gs = info.get("raw_state", {})
        ante = gs.get("round_resets", {}).get("ante", prev_ante)
        round_num = gs.get("round", prev_round)

        if verbose and (ante > prev_ante or round_num > prev_round):
            phase = gs.get("phase", "?")
            dollars = gs.get("dollars", 0)
            print(
                f"  Step {step_count}: ante={ante} round={round_num} "
                f"phase={phase} ${dollars} value={value:.3f}"
            )
            prev_ante = ante
            prev_round = round_num

        if done:
            won = env.episode_won
            final_ante = info.get("balatro/ante_reached", ante)
            final_rounds = info.get("balatro/rounds_beaten", round_num)
            if verbose:
                result = "WON!" if won else "Lost"
                print(
                    f"  {result} at ante {final_ante} "
                    f"(rounds beaten: {final_rounds}, steps: {step_count})"
                )
            return {
                "won": won,
                "ante_reached": final_ante,
                "rounds_beaten": final_rounds,
                "steps": step_count,
                "reward": total_reward,
            }

        obs = next_obs
        mask = next_mask

        if live:
            time.sleep(0.05)  # small delay so you can watch the game


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Balatro with a trained policy")
    parser.add_argument("model_path", help="Path to .pt checkpoint or model file")
    parser.add_argument("--seed", default="PLAY_0", help="Game seed prefix")
    parser.add_argument("--n-games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--verbose", action="store_true", help="Print game progress")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--live", action="store_true", help="Play against live Balatro via balatrobot"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Balatrobot host (with --live)")
    parser.add_argument("--port", type=int, default=12346, help="Balatrobot port (with --live)")
    args = parser.parse_args()

    device = torch.device(args.device)
    network = FactoredPolicy()

    # Load model — handle both full checkpoints and raw state dicts
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "network" in ckpt:
        network.load_state_dict(ckpt["network"])
        print(f"Loaded checkpoint (step {ckpt.get('global_step', '?')})")
    else:
        network.load_state_dict(ckpt)
        print("Loaded model state dict")

    network.to(device)

    # Set up adapter factory
    adapter_factory = None
    if args.live:
        from jackdaw.bridge.backend import LiveBackend
        from jackdaw.env.game_interface import BridgeAdapter

        backend = LiveBackend(host=args.host, port=args.port)
        # Test connection and go to menu
        try:
            backend.handle("health", {})
            print(f"Connected to balatrobot at {args.host}:{args.port}")
            backend.handle("menu", {})
            print("Returned to menu")
        except Exception as e:
            print(f"Cannot connect to balatrobot at {args.host}:{args.port}: {e}")
            print("Make sure balatrobot is running:")
            print("  uvx balatrobot serve --fast --no-audio --love-path <path>")
            return

        def adapter_factory():
            return BridgeAdapter(backend)

    results = []
    for i in range(args.n_games):
        seed = f"{args.seed}_{i}"
        if args.verbose or args.live:
            print(f"\n--- Game {i + 1}/{args.n_games} (seed: {seed}) ---")
        result = play_game(
            network,
            device,
            adapter_factory=adapter_factory,
            seed=seed,
            verbose=args.verbose or args.live,
            live=args.live,
        )
        results.append(result)

    # Summary
    antes = [r["ante_reached"] for r in results]
    wins = sum(1 for r in results if r["won"])
    steps = [r["steps"] for r in results]
    print(f"\n{'=' * 40}")
    print(f"Games: {len(results)}")
    print(f"Win rate: {wins}/{len(results)} ({100 * wins / len(results):.1f}%)")
    print(f"Ante reached: mean={np.mean(antes):.1f} max={np.max(antes)} min={np.min(antes)}")
    print(f"Steps: mean={np.mean(steps):.0f}")


if __name__ == "__main__":
    main()
