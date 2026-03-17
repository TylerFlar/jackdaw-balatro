# Balatrobot ↔ Jackdaw Action Mapping

## Action Mapping

| Balatrobot RPC | Params | Our Action Type | Notes |
|---|---|---|---|
| `play` | `cards: int[]` | `PlayHand(card_indices)` | 0-based, selection order preserved |
| `discard` | `cards: int[]` | `Discard(card_indices)` | 0-based |
| `select` | *(none)* | `SelectBlind()` | Selects current blind_on_deck |
| `skip` | *(none)* | `SkipBlind()` | Only for Small/Big |
| `buy` | `card: int` | `BuyCard(shop_index)` | Buys from shop card area |
| `buy` | `voucher: int` | `RedeemVoucher(card_index)` | Buys + applies voucher |
| `buy` | `pack: int` | `OpenBooster(card_index)` | Buys + opens pack |
| `sell` | `joker: int` | `SellCard('jokers', idx)` | |
| `sell` | `consumable: int` | `SellCard('consumables', idx)` | |
| `use` | `consumable: int, cards?: int[]` | `UseConsumable(idx, targets)` | `cards` = hand targets |
| `reroll` | *(none)* | `Reroll()` | |
| `next_round` | *(none)* | `NextRound()` | Leave shop |
| `cash_out` | *(none)* | `CashOut()` | Collect round earnings |
| `pack` | `card: int` | `PickPackCard(card_index)` | Non-targeted pick |
| `pack` | `card: int, targets: int[]` | `PickPackCard(idx, targets)` | Arcana/Spectral targeting |
| `pack` | `skip: true` | `SkipPack()` | Skip remaining picks |
| `rearrange` | `hand: int[]` | `ReorderHand(new_order)` | Permutation |
| `rearrange` | `jokers: int[]` | `ReorderJokers(new_order)` | Permutation |
| `rearrange` | `consumables: int[]` | *(not implemented)* | Consumable order doesn't affect gameplay |

## Actions We Have That Balatrobot Doesn't

| Our Action | Notes |
|---|---|
| `SortHand(mode)` | Balatrobot uses `rearrange hand` instead of a sort mode |
| `BuyAndUse(idx, targets)` | We model buy+use as one action; balatrobot does `buy card` then `use consumable` |

## Balatrobot Actions We're Missing

| Balatrobot RPC | Notes |
|---|---|
| `start` | Run initialization — we use `initialize_run()` directly |
| `menu` | Return to menu — no sim equivalent (terminal) |
| `save` / `load` | Save/load — not needed for sim |
| `add` | Debug: add card — not needed for sim |
| `set` | Debug: modify values — not needed for sim |
| `screenshot` | Visual capture — not applicable |
| `rearrange consumables` | Consumable order doesn't affect gameplay |

## State Mapping

| Balatrobot State | Our GamePhase |
|---|---|
| `BLIND_SELECT` | `GamePhase.BLIND_SELECT` |
| `SELECTING_HAND` | `GamePhase.SELECTING_HAND` |
| `ROUND_EVAL` | `GamePhase.ROUND_EVAL` |
| `SHOP` | `GamePhase.SHOP` |
| `SMODS_BOOSTER_OPENED` | `GamePhase.PACK_OPENING` |
| `GAME_OVER` | `GamePhase.GAME_OVER` |
| `MENU` | *(not mapped — pre-game)* |

## Card Index Conventions

Both use 0-based indices. Balatrobot indices reference the same card
arrays as our `gs["hand"]`, `gs["jokers"]`, `gs["consumables"]`,
`gs["shop_cards"]`, `gs["shop_vouchers"]`, `gs["shop_boosters"]`.
