---
layout: default
title: Home
---

# Jackdaw

Bit-exact headless Balatro v1.0.1o simulator for reinforcement learning research.
**1,400+ runs/sec** · Pure Python · No game client needed · Gymnasium compatible

---

## Quick Start

```bash
pip install jackdaw
```

```python
from jackdaw.env import BalatroEnvironment, DirectAdapter

env = BalatroEnvironment(adapter_factory=DirectAdapter)
obs, mask, info = env.reset()

while not info.get("done"):
    action = your_model.act(obs, mask)
    obs, terminated, truncated, mask, info = env.step(action)
```

Swap one line to validate against real Balatro:

```python
from jackdaw.bridge import LiveBackend, BridgeAdapter

env = BalatroEnvironment(
    adapter_factory=lambda: BridgeAdapter(LiveBackend("127.0.0.1", 12346))
)
```

Same observations, same masks, same interface.

---

## At a Glance

| | |
|---|---|
| **150** jokers | **250+** validation scenarios |
| **21** action types | **30** engine modules |
| **14**-phase scoring | **8** stakes, **15** deck backs |
| Bit-exact with LuaJIT 2.1 | Deterministic from any seed string |

---

## Pages

- [Getting Started](getting-started) — Install, first training loop, project structure
- [Engine](engine) — State machine, RNG, scoring, jokers
- [RL Environment](environment) — Observations, actions, validation
