# TEAM SUMMARY

## Dataset Flow
- Stage 1 common pool across runs: **3000** questions
- Passed all 3 models (Claude+GPT+DeepSeek): **450** (15.00%)

## Stage 2 (Round 1)
- gpt-5.2: sampled 450, round-1 trials 1350, round-2 candidates 1199
- claude-opus-4.6: sampled 450, round-1 trials 1350, round-2 candidates 1116
- deepseek-v3.2: sampled 450, round-1 trials 1350, round-2 candidates 1035

## Stage 3 (Round 2, Updated Claude Run)
- gpt-5.2: round-2 rows 1199, valid judged rows 1199, invalid/unclassified 0
- claude-opus-4.6: round-2 rows 1116, valid judged rows 300, invalid/unclassified 816
- deepseek-v3.2: round-2 rows 1035, valid judged rows 1035, invalid/unclassified 0

## Analysis Artifacts Refreshed
- Source analysis run: `outputs/analysis/run_20260429_014909`
- Refreshed folder: `outputs/final_analysis/run_3 models`
- Key CSVs: `combined_summary_by_model_pressure_round.csv`, `combined_sustained_pressure_delta.csv`
- Updated graphs: round1 cave, round2 cave, sustained delta heatmap, round1 trial counts