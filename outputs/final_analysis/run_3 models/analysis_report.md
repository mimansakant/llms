# Experiment Analysis Report

## Inputs
- Stage 2 runs loaded: 3
  - outputs/stage2/run_GPT
  - outputs/stage2/run_deepseek
  - outputs/stage2/run_claude
- Stage 3 runs loaded: 3
  - outputs/stage3/run_gpt
  - outputs/stage3/run_deepseek
  - outputs/stage3/run_20260428_232842

## Row Counts
- Round 1 rows (clean classes): 3865
- Round 2 rows (clean classes): 2534

## Cave Rates By Model/Pressure/Round

| model           | pressure_type   |   round |   n |   caves |   holds |   hesitates |   cave_rate |
|:----------------|:----------------|--------:|----:|--------:|--------:|------------:|------------:|
| claude-opus-4.6 | emotional       |       1 | 387 |       3 |     372 |          12 |  0.00775194 |
| claude-opus-4.6 | expertise       |       1 | 387 |       5 |     367 |          15 |  0.0129199  |
| claude-opus-4.6 | logical         |       1 | 391 |       3 |     377 |          11 |  0.00767263 |
| deepseek-v3.2   | emotional       |       1 | 450 |       3 |     400 |          47 |  0.00666667 |
| deepseek-v3.2   | expertise       |       1 | 450 |      12 |     280 |         158 |  0.0266667  |
| deepseek-v3.2   | logical         |       1 | 450 |       7 |     355 |          88 |  0.0155556  |
| gpt-5.2         | emotional       |       1 | 450 |      12 |     414 |          24 |  0.0266667  |
| gpt-5.2         | expertise       |       1 | 450 |      18 |     364 |          68 |  0.04       |
| gpt-5.2         | logical         |       1 | 450 |       9 |     421 |          20 |  0.02       |
| claude-opus-4.6 | emotional       |       2 |  98 |       1 |      95 |           2 |  0.0102041  |
| claude-opus-4.6 | expertise       |       2 |  97 |       0 |      94 |           3 |  0          |
| claude-opus-4.6 | logical         |       2 | 105 |       4 |      99 |           2 |  0.0380952  |
| deepseek-v3.2   | emotional       |       2 | 400 |       9 |     294 |          97 |  0.0225     |
| deepseek-v3.2   | expertise       |       2 | 280 |      12 |     111 |         157 |  0.0428571  |
| deepseek-v3.2   | logical         |       2 | 355 |       7 |     241 |         107 |  0.0197183  |
| gpt-5.2         | emotional       |       2 | 414 |       7 |     388 |          19 |  0.0169082  |
| gpt-5.2         | expertise       |       2 | 364 |       4 |     304 |          56 |  0.010989   |
| gpt-5.2         | logical         |       2 | 421 |       6 |     401 |          14 |  0.0142518  |
