---
layout: default
title: Home
---

<div align="center">
  <img src="jackdaw.png" alt="Jackdaw Logo" width="300" height="300"/>
  <h1>Jackdaw</h1>
</div>

Bit-exact headless Balatro v1.0.1o simulator for reinforcement learning research.

---

## Pages

- [Python API](api) — The Rust-backed `Game`: actions, legal moves, views, previews, snapshots, `GameBuilder`
- [RL Project Guide](rl-guide) — Observations, actions, masks, rewards, and the Gymnasium wrapper (works with MaskablePPO)
- [Validation & Scenarios](validation) — Validating the engine against live Balatro, writing your own test scenarios
- [Lockstep against the live game](live) — `jackdaw lockstep`, divergence bundles, replay, and the deviation catalog
- [BalatroBot on the simulator](balatrobot) — `jackdaw serve`, `SimBackend`, and how the simulator maps onto BalatroBot's API
