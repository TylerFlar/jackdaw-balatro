---
layout: default
title: Evaluating on Jackdaw Arena
---

# Evaluating on Jackdaw Arena

[Jackdaw Arena](https://arena.tylerflar.com) is a hosted, verifiable benchmark for agents
trained on jackdaw. Your agent keeps running on your machine and plays over HTTPS; the episodes
run on the server, on seeds the server picks.

A local evaluation picks its own seeds, in a simulator that can snapshot and look ahead. The
Arena takes that freedom away:

- **Seeds nobody has seen.** Each episode gets a fresh 16-character seed from a CSPRNG, kept
  secret until the episode ends. The server holds the only copy of the game.
- **Every episode counts.** Each agent has a fixed budget per track, 500 on `red-1`, and is
  ranked once all of them have ended. An abandoned or timed-out episode keeps the score it had
  reached.
- **Verified and public.** The server replays every finished episode and checks that it ends on
  the same `state_hash`. Then the seed and the tape (the engine actions, in order) are
  published, so anyone can replay the episode with jackdaw.

## Evaluate an agent

1. Sign in on [the site](https://arena.tylerflar.com) with GitHub, then create an agent and a
   token (it's shown once). Make the token available as `ARENA_TOKEN`, or save it to
   `~/.jackdaw-arena/token`.
2. Install the client. It doesn't depend on jackdaw:

   ```bash
   uv add "jackdaw-arena @ git+https://github.com/TylerFlar/jackdaw-arena#subdirectory=client"
   ```

3. Play. `ArenaEnv` is a drop-in for `BalatroGymnasiumEnv`: the same observation space,
   `Discrete(500)` with `action_masks()`, and the same `balatro/*` info keys.

   ```python
   from jackdaw_arena import ArenaEnv

   env = ArenaEnv()
   obs, info = env.reset()           # the server picks the seed
   done = False
   while not done:
       action = model.predict(obs, action_masks=env.action_masks())[0]
       obs, reward, terminated, truncated, info = env.step(int(action))
       done = terminated or truncated
   print(info["balatro/rounds_beaten"], info["arena/seed"])   # revealed at the end
   ```

   To spend a whole budget in batches, `evaluate()` takes the chooser signature of
   [`examples/balatro_sb3.py`](https://github.com/TylerFlar/jackdaw-balatro/blob/dev/examples/balatro_sb3.py),
   so a model trained there plugs in unchanged. It never starts more episodes than you ask for,
   and an interrupted run picks up where it stopped.

   ```python
   from sb3_contrib import MaskablePPO
   from jackdaw_arena import evaluate

   model = MaskablePPO.load("examples/results/maskable_ppo_balatro.zip")

   def choose(envs, obs, masks):
       actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
       return actions

   print(evaluate(choose, episodes=500, batch=32).summary())
   ```

## The track

`red-1` is the protocol of `examples/`: Red Deck, white stake, a 2,000-step cap, and episodes
that end on a win. The rank is the mean number of blinds beaten, with its standard error and
95% interval. An episode that ends because the engine refused an action scores 0, as in
`run_episodes`. The house baselines are listed apart: `random`, and `smart_agent`, which reads
face-down cards.

Results belong to an engine version. A jackdaw release that changes the game starts a new
season.

## Verify an episode

Once an episode has ended, its seed and tape are public. With the same jackdaw version
installed:

```bash
jackdaw-arena tape <episode-id> -o tape.json
jackdaw-arena verify tape.json
```

`verify` starts a game from the tape's deck, stake and seed, plays every action, and compares
the final `state_hash` with the one the server published. The same in Python:

```python
import json
from jackdaw.core import Game

tape = json.load(open("tape.json"))
game = Game.new(tape["back"], int(tape["stake"]), tape["seed"])
for action in tape["tape"]:
    game.step(action)
assert game.state_hash() == tape["final_state_hash"]
```
