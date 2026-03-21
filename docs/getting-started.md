---
layout: default
title: Getting Started
---

# Getting Started

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

No game client needed for training. For live validation you'll also need Balatro + [BalatroBot](https://github.com/coder/balatrobot).

## Install

```bash
git clone https://github.com/TylerFlar/jackdaw-balatro.git
cd jackdaw-balatro
uv sync --dev
```

## First Training Loop

```python
from jackdaw.env import BalatroEnvironment, DirectAdapter

env = BalatroEnvironment(adapter_factory=DirectAdapter)
obs, mask, info = env.reset()

while not info.get("done"):
    action = your_model.act(obs, mask)   # FactoredAction
    obs, terminated, truncated, mask, info = env.step(action)
```

`DirectAdapter` runs the engine in-process at ~1,400 runs/sec.

## Tests

```bash
pytest                    # unit tests
pytest --cov=jackdaw      # with coverage
pytest -m benchmark       # benchmarks
pytest -m live            # live validation (needs BalatroBot)
```

## Lint

```bash
ruff check .
ruff format .
```

## Project Structure

```
jackdaw/
  engine/       Deterministic game simulator (30 modules)
  env/          RL environment — observations, actions
  bridge/       Validation bridge to live Balatro
  cli/          CLI tools & 250+ validation scenarios
tests/          Test suite & benchmarks
scripts/        Data extraction utilities
```
