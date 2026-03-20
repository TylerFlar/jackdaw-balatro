# M18 Baseline Agent Performance

Baselines recorded by `scripts/baselines.py`. These are the numbers
the RL agent needs to beat.

## Summary

| Agent | Episodes | Win Rate | Avg Ante | Avg Rounds | Avg Actions | Min Ante | Max Ante |
|-------|----------|----------|----------|------------|-------------|----------|----------|
| RandomAgent / b_red / stake 1 | 200 | 0.0% | 1.00 | 0.0 | 25 | 1 | 1 |
| HeuristicAgent / b_red / stake 1 | 200 | 0.0% | 2.17 | 5.2 | 47 | 1 | 7 |
| HeuristicAgent / b_blue / stake 1 | 50 | 0.0% | 2.34 | 6.0 | 58 | 1 | 5 |
| HeuristicAgent / b_yellow / stake 1 | 50 | 0.0% | 2.42 | 6.3 | 58 | 1 | 5 |
| HeuristicAgent / b_green / stake 1 | 50 | 0.0% | 2.00 | 4.7 | 43 | 1 | 4 |

## Ante Distributions

### RandomAgent / b_red / stake 1

| Ante | Count | % |
|------|-------|---|
| 1 | 200 | 100.0% |

### HeuristicAgent / b_red / stake 1

| Ante | Count | % |
|------|-------|---|
| 1 | 78 | 39.0% |
| 2 | 52 | 26.0% |
| 3 | 34 | 17.0% |
| 4 | 31 | 15.5% |
| 5 | 4 | 2.0% |
| 7 | 1 | 0.5% |

### HeuristicAgent / b_blue / stake 1

| Ante | Count | % |
|------|-------|---|
| 1 | 16 | 32.0% |
| 2 | 12 | 24.0% |
| 3 | 12 | 24.0% |
| 4 | 9 | 18.0% |
| 5 | 1 | 2.0% |

### HeuristicAgent / b_yellow / stake 1

| Ante | Count | % |
|------|-------|---|
| 1 | 15 | 30.0% |
| 2 | 14 | 28.0% |
| 3 | 9 | 18.0% |
| 4 | 9 | 18.0% |
| 5 | 3 | 6.0% |

### HeuristicAgent / b_green / stake 1

| Ante | Count | % |
|------|-------|---|
| 1 | 25 | 50.0% |
| 2 | 8 | 16.0% |
| 3 | 9 | 18.0% |
| 4 | 8 | 16.0% |
