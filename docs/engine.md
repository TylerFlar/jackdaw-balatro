---
layout: default
title: Engine
---

# Engine

The engine is a deterministic, bit-exact reimplementation of Balatro v1.0.1o in pure Python. Every RNG draw, score, and state transition matches the original LuaJIT 2.1 behavior.

## Core Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `initialize_run()` | `run_init.py` | Create a seeded game state from `(back, stake, seed)` |
| `step()` | `game.py` | Apply one action, return next state |
| `simulate_run()` | `runner.py` | Run a full game with an agent |

## Game Phases

| Phase | Actions |
|-------|---------|
| Blind Select | SelectBlind, SkipBlind |
| Play | PlayHand, Discard, UseConsumable, SellJoker, SellConsumable |
| Cash Out | CashOut |
| Shop | BuyCard, Reroll, NextRound, SellJoker, UseConsumable, RedeemVoucher |
| Pack Open | PickPackCard, SkipPack |
| Game Over | Terminal |

## RNG System

Three-layer PRNG matching Balatro exactly:

1. **pseudohash** — seed string → numeric seed
2. **pseudoseed** — master seed → independent streams (`shop`, `joker`, `tarot`, etc.)
3. **TamedWild223** — Tausworthe PRNG producing actual random values

Each stream is consumed in identical order to the Lua implementation.

## Scoring Pipeline

Scoring resolves in 14 phases:

1. Identify hand type
2. Base chips + mult from hand level
3. Card-level bonuses (enhancement, edition)
4. Played-card triggers (left to right)
5. Held-card triggers
6. Joker triggers (left to right — order matters)
7. Retrigger passes (Red Seal, Hack, Dusk, etc.)
8. Boss blind effects
9. Edition multipliers (Foil, Holographic, Polychrome)
10. Seal effects
11. Consumable on-score effects
12. Final `chips × mult`
13. Glass card destruction rolls
14. Post-score cleanup

## Key Modules

| Module | What it does |
|--------|-------------|
| `jokers.py` | 150 joker effects, dispatch by center key |
| `scoring.py` | 14-phase scoring pipeline |
| `rng.py` | Bit-exact 3-layer PRNG |
| `hand_eval.py` | Poker hand detection (respects wilds, debuffs) |
| `shop.py` | Shop population, reroll, buy/sell |
| `consumables.py` | Tarot, planet, spectral usage |
| `blind.py` | Blind scaling and boss effects |
| `economy.py` | Interest, slots, reroll costs |

## Data Files

Game data extracted from Balatro's Lua source lives in `engine/data/`:

| File | Contents |
|------|----------|
| `centers.json` (117 KB) | ~299 card prototypes (jokers, tarots, planets, spectrals, vouchers) |
| `blinds.json` | Blind definitions and boss effects |
| `cards.json` | Standard playing card specs |
| `tags.json` | Tag definitions |
| `stakes.json` | 8 stake levels |
| `seals.json` | Seal effects |
