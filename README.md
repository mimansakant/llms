<<<<<<< HEAD
# llms
COMSBC3707 The Customer is Always Right (Even When They’re Not): LLM Sycophancy Under Pressure
=======
# LLM Sycophancy Experiment Scripts (Python)

This repo now includes a Python pipeline that implements the 4 stages in your draft:

1. Stage 1 baseline filtering on MMLU test questions.
2. Stage 2 pressure application (Round 1).
3. Stage 3 sustained pressure (Round 2, only if Round 1 is `HOLDS`).
4. Stage 4 Gemini-based response classification (`HOLDS` / `CAVES` / `HESITATES`).

## Files

- `stage1_filter.py`: Stage 1 only (baseline filtering).
- `stage2_round1.py`: Stage 2 only (first pressure round).
- `stage3_round2.py`: Stage 3 only (second pressure round on `HOLDS`).
- `run_experiment.py`: Single-shot end-to-end runner (optional).
- `sycophancy_experiment/mmlu.py`: MMLU loading + domain mapping + stratified sampling.
- `sycophancy_experiment/prompts.py`: prompt templates from your appendix.
- `sycophancy_experiment/clients.py`: API clients (OpenAI, DeepSeek, Anthropic, Gemini).
- `sycophancy_experiment/pipeline.py`: end-to-end experiment logic.
- `sycophancy_experiment/utils.py`: parsing + balancing + output helpers.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Set these before running:

```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GOOGLE_API_KEY="..."
```

Claude backend options:

Anthropic direct API:
```bash
export CLAUDE_BACKEND="anthropic"   # default
export ANTHROPIC_API_KEY="..."
```

AWS Bedrock:
```bash
export CLAUDE_BACKEND="bedrock"
export BEDROCK_REGION="us-east-1"
export BEDROCK_MODEL_ID="anthropic.<your-claude-model-id>"
# Optional profile:
export AWS_PROFILE="default"
```

To enable the optional open-source model key `qwen2.5-7b`:

```bash
export QWEN_BASE_URL="http://localhost:8000/v1"   # or another OpenAI-compatible endpoint
export QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
# Optional if your endpoint requires auth:
export QWEN_API_KEY="..."
```

Optional model overrides:

```bash
export OPENAI_MODEL="gpt-5.2"
export DEEPSEEK_MODEL="deepseek-reasoner"
export ANTHROPIC_MODEL="claude-opus-4-1"
export GEMINI_MODEL="gemini-2.0-flash"
```

## Run (3-script workflow)

1) Stage 1: find questions all models answer correctly

```bash
python stage1_filter.py --max-questions 60 --seed 42 \
  --models claude-opus-4.6,deepseek-v3.2,gpt-5.2 \
  --judge-model gemini-2.0-flash \
  --workers 4
```

2) Stage 2: run first pressure round on a stratified sample from Stage 1 kept questions

```bash
python stage2_round1.py --stage1-dir outputs/stage1/<RUN_DIR> --sample-size 30 --seed 42 \
  --models claude-opus-4.6,deepseek-v3.2,gpt-5.2 \
  --judge-model gemini-2.0-flash \
  --workers 6
```

3) Stage 3: run second pressure round on Round-1 `HOLDS`

```bash
python stage3_round2.py --stage2-dir outputs/stage2/<RUN_DIR> --seed 42 \
  --models claude-opus-4.6,deepseek-v3.2,gpt-5.2 \
  --judge-model gemini-2.0-flash \
  --workers 6
```

Optional: keep using the one-shot pipeline:

```bash
python run_experiment.py --sample-size 1000 --seed 42 \
  --models claude-opus-4.6,deepseek-v3.2,gpt-5.2 \
  --judge-model gemini-2.0-flash
```

Worker tip:
- Cloud APIs: start with `--workers 4` to `6`.
- Local single-GPU Qwen endpoint: start with `--workers 1` to `2`.

Model keys currently configured in code:
- `claude-opus-4.6`
- `deepseek-v3.2`
- `gpt-5.2`
- `gemini-2.0-flash` (judge)
- `qwen2.5-7b` (enabled when `QWEN_BASE_URL` is set)

## Outputs (3-script workflow)

`stage1_filter.py` writes `outputs/stage1/run_.../`:
- `stage1_baseline.jsonl`
- `kept_questions.jsonl`
- `baseline_answers.json`
- `stage1_metadata.jsonl`

`stage2_round1.py` writes `outputs/stage2/run_.../`:
- `sampled_questions.jsonl`
- `round1_results.jsonl`
- `round2_candidates.jsonl`
- `round1_summary_by_model_pressure.csv`
- `round1_summary_by_model_pressure_intensity.csv`
- `round1_summary_by_model_pressure_domain.csv`
- `stage2_metadata.jsonl`

`stage3_round2.py` writes `outputs/stage3/run_.../`:
- `round2_results.jsonl`
- `combined_round1_round2_results.jsonl`
- `summary_by_model_pressure_round.csv`
- `summary_by_model_pressure_intensity_round.csv`
- `summary_by_model_pressure_domain_round.csv`
- `summary_sustained_pressure_delta.csv`
- `stage3_metadata.jsonl`

## Notes

- Baseline answer extraction uses regex (`A/B/C/D`) to tolerate minor formatting drift.
- Logical pressure fills `[generated reasoning]` using Gemini false-reasoning generation.
- API retries are enabled (exponential backoff) for transient failures.
>>>>>>> e543658 (Initial project upload)
