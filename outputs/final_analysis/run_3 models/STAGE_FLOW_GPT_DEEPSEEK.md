# Stage-by-Stage Analysis (GPT + DeepSeek)

## Stage 1: Baseline Filter (1000-question run)
- Common pool across model runs: **1000**
- Passed all 3 models (Claude+GPT+DeepSeek): **595** (59.50%)
- Source merge run: `outputs/stage1/merged/run_20260429_042524`

## Stage 2: Round 1 Pressure (sample-size 595)
- Each model ran 595 questions × 3 pressure types = 1785 trials.
- DeepSeek Round 1: HOLDS=1572, CAVES=128, HESITATES=85, ERROR=0
  - DeepSeek cave rate: **7.17%**; hold rate: **88.07%**
- GPT Round 1 (after retry merge): HOLDS=1546, CAVES=216, HESITATES=23, ERROR=0
  - GPT cave rate: **12.10%**; hold rate: **86.61%**
- GPT required `stage2_retry_errors.py` due OpenAI 429 mid-run; merged output has 0 ERROR.

## Stage 3: Round 2 Sustained Pressure
- DeepSeek Round 2 rows: 1572 (matches Stage 2 HOLDS)
  - HOLDS=1322, CAVES=84, HESITATES=166, ERROR=0
  - Round 2 cave rate: **5.34%** (delta vs Round 1: -1.83%)
- GPT Round 2 rows: 1546 (matches Stage 2 HOLDS)
  - HOLDS=1459, CAVES=60, HESITATES=27, ERROR=0
  - Round 2 cave rate: **3.88%** (delta vs Round 1: -8.22%)

## Interpretation
- GPT shows high susceptibility in Round 1 and a strong drop in caving in Round 2 among retained HOLDS cases.
- DeepSeek shows low cave rates in both rounds, with a moderate increase under sustained pressure.
- Because Round 2 only includes Round-1 HOLDS, Round-1 and Round-2 cave rates are not directly from identical trial pools; report deltas with this caveat.

## Key Files
- Stage1 merged: `outputs/stage1/merged/run_20260429_042524`
- DeepSeek Stage2: `outputs/stage2/run_deepseek_2`
- DeepSeek Stage3: `outputs/stage3/run_20260429_055150`
- GPT Stage2 fixed: `outputs/stage2/run_gpt_2_fixed`
- GPT Stage3: `outputs/stage3/run_20260429_064321`