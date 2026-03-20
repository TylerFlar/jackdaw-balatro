# Action Space Coverage Report

Empirical verification that every legal engine action is reachable
through the factored action space, and vice versa.

## Forward: Engine → Factored

For every legal action at every step of 200 episodes (100 Random +
100 Heuristic), attempt `engine_action_to_factored()`.

| Action Type | Total Seen | Convertible | Failed | Coverage |
|-------------|-----------|-------------|--------|----------|
| BuyAndUse            |         2 |           2 |      0 |     100% |
| BuyCard              |      1128 |        1128 |      0 |     100% |
| CashOut              |       483 |         483 |      0 |     100% |
| Discard              |      3680 |        3680 |      0 |     100% |
| NextRound            |      1205 |        1205 |      0 |     100% |
| OpenBooster          |      1402 |        1402 |      0 |     100% |
| PickPackCard         |       610 |         610 |      0 |     100% |
| PlayHand             |      4274 |        4274 |      0 |     100% |
| RedeemVoucher        |       182 |         182 |      0 |     100% |
| ReorderHand          |      4274 |        4274 |      0 |     100% |
| ReorderJokers        |      2321 |        2321 |      0 |     100% |
| Reroll               |       853 |         853 |      0 |     100% |
| SelectBlind          |       773 |         773 |      0 |     100% |
| SellCard             |      3884 |        3884 |      0 |     100% |
| SkipBlind            |       569 |         569 |      0 |     100% |
| SkipPack             |       187 |         187 |      0 |     100% |
| SortHand             |      8548 |        8548 |      0 |     100% |
| UseConsumable        |       358 |         358 |      0 |     100% |

**Total: 34,733 actions seen, 34,733 convertible (100.0%), 0 failed**

All action types achieve 100% forward conversion.

**ReorderHand / ReorderJokers:** Empty-permutation markers from
`get_legal_actions()` map to SwapLeft markers. Full permutations
decompose into the first adjacent swap of a bubble-sort pass.
The RL agent achieves arbitrary reorderings through repeated swaps.

**BuyAndUse:** Composite engine action maps to BuyCard. The RL
agent achieves the same result via separate BuyCard + UseConsumable.

## Reverse: Factored → Engine

For each ActionType, generate random valid FactoredActions from
real game states and verify `factored_to_engine_action()` succeeds.

| ActionType | Tested | OK | Failed | Coverage |
|------------|--------|------|--------|----------|
| BuyCard              |     33 |   33 |      0 |     100% |
| CashOut              |     50 |   50 |      0 |     100% |
| Discard              |     50 |   50 |      0 |     100% |
| NextRound            |     50 |   50 |      0 |     100% |
| OpenBooster          |     32 |   32 |      0 |     100% |
| PickPackCard         |     50 |   50 |      0 |     100% |
| PlayHand             |     50 |   50 |      0 |     100% |
| RedeemVoucher        |      0 |    0 |      0 |      N/A |
| Reroll               |     21 |   21 |      0 |     100% |
| SelectBlind          |     50 |   50 |      0 |     100% |
| SellConsumable       |      4 |    4 |      0 |     100% |
| SellJoker            |     39 |   39 |      0 |     100% |
| SkipBlind            |     35 |   35 |      0 |     100% |
| SkipPack             |     50 |   50 |      0 |     100% |
| SortHandRank         |     50 |   50 |      0 |     100% |
| SortHandSuit         |     50 |   50 |      0 |     100% |
| SwapHandLeft         |     50 |   14 |     36 |      28% |
| SwapHandRight        |     50 |   13 |     37 |      26% |
| SwapJokersLeft       |     24 |   24 |      0 |     100% |
| SwapJokersRight      |     30 |   30 |      0 |     100% |
| UseConsumable        |      0 |    0 |      0 |      N/A |

**Total: 768 tested, 695 OK (90.5%), 73 failed**
