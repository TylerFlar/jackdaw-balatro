<div align="center">
  <h1>Jackdaw</h1>
  <div><img src="docs/jackdaw.png" alt="Jackdaw Logo" width="300" height="300"/></div>
  <br/>
  <a href="https://tylerflar.github.io/jackdaw-balatro/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Documentation"/></a>
</div>

---

Jackdaw is a Balatro simulator built for RL research: a bit-exact Balatro 1.0.1o engine written in Rust with a Python interface, Gymnasium-style environments (single and vectorized) with entity-based observations and a factored action space, and a bridge that plays live Balatro in lockstep through [BalatroBot](https://github.com/coder/balatrobot).

## Motivation

With [BalatroBot](https://github.com/coder/balatrobot) already existing, why build a separate simulator?

BalatroBot drives the real game, which runs at the game's speed and can race or crash when pushed. RL training needs many fast, parallel environments: one jackdaw environment steps about 180,000 times a second, and `BalatroVectorEnv` over a million times a second on 8 threads. And the Python API exposes every part of the game, with read-only views, snapshots, exact score previews and a state builder, which helps with debugging and research. See the [Python API](https://tylerflar.github.io/jackdaw-balatro/api).

The engine follows the game's Lua source function by function, down to the RNG. It was checked step by step against a Python reference engine, against LuaJIT for the RNG and hand evaluation, and in lockstep against the live game: 200 organic games and every validation scenario without an unexplained difference.

## Install

### Add to your project

```bash
uv add git+https://github.com/TylerFlar/jackdaw-balatro.git --branch main   # the latest release
uv add git+https://github.com/TylerFlar/jackdaw-balatro.git --tag v0.2.0    # a given release
```

`main` holds the releases; the default branch, `dev`, is the work toward the next one. Installing from git compiles the Rust engine, so it needs a Rust toolchain ([rustup](https://rustup.rs)). The [releases](https://github.com/TylerFlar/jackdaw-balatro/releases) also have prebuilt wheels for Linux, macOS and Windows (CPython 3.12+), which don't.

### For development

```bash
git clone https://github.com/TylerFlar/jackdaw-balatro
cd jackdaw-balatro
uv sync --dev
```

`uv sync` builds the `jackdaw._core` extension with maturin. See [development](https://tylerflar.github.io/jackdaw-balatro/development) for the toolchain and [CONTRIBUTING.md](CONTRIBUTING.md) for the branch flow.

## Quick Start

```python
from jackdaw.env import BalatroEnvironment, DirectAdapter

env = BalatroEnvironment(adapter_factory=DirectAdapter)
obs, mask, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = your_model.act(obs, mask)
    obs, terminated, truncated, mask, info = env.step(action)
```

## Gymnasium Wrapper

`BalatroGymnasiumEnv` exposes the game as a flat `Discrete(500)` action space with an `action_masks()` method, so it works with action-masking RL libraries such as sb3-contrib's MaskablePPO. `BalatroVectorEnv` steps many of them at once in Rust. Jackdaw doesn't depend on any RL library; install the one you use in your own project.

[`examples/`](examples) has a complete MaskablePPO demo: training with TensorBoard logs and checkpoints, an evaluation against a random agent and the hand-written `smart_agent`, and its results. Its dependencies are the `train` dependency group (`uv sync --group train`), which isn't part of the package.

See the [RL Project Guide](https://tylerflar.github.io/jackdaw-balatro/rl-guide) for the observations, actions, rewards and environments.

## Live Validation

Swap the adapter to play against real Balatro via [BalatroBot](https://github.com/coder/balatrobot) — the interface is identical. `LiveAdapter` keeps a shadow run in the simulator, checks every live answer against it, and takes observations and masks from it, so an agent sees exactly what it saw in training:

```python
from jackdaw.bridge import LiveAdapter, LiveBackend
from jackdaw.env import BalatroEnvironment

env = BalatroEnvironment(
    adapter_factory=lambda: LiveAdapter(LiveBackend("127.0.0.1", 12346))
)
# Same obs and actions; the mask only offers what BalatroBot can execute
```

`jackdaw lockstep` plays organic games on one or more BalatroBot instances this way and writes a repro bundle for every divergence; `jackdaw lockstep replay` reruns one on the simulator. See [Lockstep against the live game](https://tylerflar.github.io/jackdaw-balatro/live).

## Architecture

```
crates/
  jackdaw-core/     The engine: Lua numerics and RNG, cards, scoring, every joker,
                    consumable, tag, voucher, boss, deck and challenge rule, snapshots
  jackdaw-data/     Generated tables (centers, blinds, tags, hands...) from the game's prototypes
  jackdaw-env/      Observation encoder, action masks, Gym enumeration, vector env, rewards
  jackdaw-bot/      BalatroBot v1.5.2's API: view, endpoints, validator, `jackdaw serve`
  jackdaw-py/       The PyO3 extension, jackdaw._core

jackdaw/
  core/             The Python API: Game, read-only views, GameBuilder
  engine/           The action types and the game's data tables (codegen input)
  env/              BalatroEnvironment, BalatroGymnasiumEnv, BalatroVectorEnv, rewards, agents
  bridge/           BalatroBot backends; LiveAdapter, the deviation catalog, lockstep
  cli/              The jackdaw command and its validation scenarios
```

## CLI

```bash
jackdaw validate                            # all validation scenarios against live Balatro
jackdaw validate --category jokers          # joker scenarios only
jackdaw validate --scenario joker_jolly     # one scenario
jackdaw lockstep --instances 4 --seeds 200  # organic games on 4 BalatroBot instances
jackdaw lockstep replay BUNDLE              # rerun a divergence on the simulator
jackdaw serve                               # BalatroBot's API on the simulator
```

> One of the biggest ways to contribute is by writing validation scenarios! See the [validation docs](https://tylerflar.github.io/jackdaw-balatro/validation) to get started.

## Development

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Rust.

```bash
uv sync --dev          # install with dev dependencies (builds the Rust extension)
uv run pytest          # Python tests
cargo test             # Rust tests
uv run pytest -m benchmark   # performance benchmarks
uv run pytest -m live        # against live Balatro (needs BalatroBot)
uv run pytest -m lockstep    # long sim-against-sim lockstep runs over jackdaw serve
uv run ruff check . && uv run ruff format --check .
```

Pull requests go to the `dev` branch; `main` holds the releases. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [changelog](CHANGELOG.md).

## License

MIT
