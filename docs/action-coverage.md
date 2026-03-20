# Action Space Coverage Report

Empirical verification that every legal engine action is reachable
through the factored action space, and vice versa.

## Forward: Engine → Factored

For every legal action at every step of 200 episodes (100 Random +
100 Heuristic), attempt `engine_action_to_factored()`.

| Action Type | Total Seen | Convertible | Failed | Coverage |
|-------------|-----------|-------------|--------|----------|
| BuyAndUse            |         2 |           2 |      0 |     100% |
| BuyCard              |      1130 |        1130 |      0 |     100% |
| CashOut              |       484 |         484 |      0 |     100% |
| Discard              |      3829 |        3829 |      0 |     100% |
| NextRound            |      1207 |        1207 |      0 |     100% |
| OpenBooster          |      1403 |        1403 |      0 |     100% |
| PickPackCard         |       610 |         610 |      0 |     100% |
| PlayHand             |      4399 |        4399 |      0 |     100% |
| RedeemVoucher        |       182 |         182 |      0 |     100% |
| ReorderHand          |      4399 |        4399 |      0 |     100% |
| ReorderJokers        |      2321 |        2321 |      0 |     100% |
| Reroll               |       854 |         854 |      0 |     100% |
| SelectBlind          |       772 |         772 |      0 |     100% |
| SellCard             |      3884 |        3884 |      0 |     100% |
| SkipBlind            |       567 |         567 |      0 |     100% |
| SkipPack             |       187 |         187 |      0 |     100% |
| SortHand             |      8798 |        8798 |      0 |     100% |
| UseConsumable        |       358 |         358 |      0 |     100% |

**Total: 35,386 actions seen, 35,386 convertible (100.0%), 0 failed**

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
| BuyCard              |     32 |   32 |      0 |     100% |
| CashOut              |     50 |   50 |      0 |     100% |
| Discard              |     50 |   50 |      0 |     100% |
| NextRound            |     50 |   50 |      0 |     100% |
| OpenBooster          |     29 |   29 |      0 |     100% |
| PickPackCard         |     50 |   50 |      0 |     100% |
| PlayHand             |     50 |   50 |      0 |     100% |
| RedeemVoucher        |      0 |    0 |      0 |      N/A |
| Reroll               |     21 |   21 |      0 |     100% |
| SelectBlind          |     50 |   50 |      0 |     100% |
| SellConsumable       |      4 |    4 |      0 |     100% |
| SellJoker            |     36 |   36 |      0 |     100% |
| SkipBlind            |     36 |   36 |      0 |     100% |
| SkipPack             |     50 |   50 |      0 |     100% |
| SortHandRank         |     50 |   50 |      0 |     100% |
| SortHandSuit         |     50 |   50 |      0 |     100% |
| SwapHandLeft         |     50 |   50 |      0 |     100% |
| SwapHandRight        |     50 |   50 |      0 |     100% |
| SwapJokersLeft       |     25 |   25 |      0 |     100% |
| SwapJokersRight      |     27 |   27 |      0 |     100% |
| UseConsumable        |      0 |    0 |      0 |      N/A |

**Total: 760 tested, 760 OK (100.0%), 0 failed**
