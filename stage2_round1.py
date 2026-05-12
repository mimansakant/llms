#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from sycophancy_experiment.mmlu import stratified_sample_by_domain
from sycophancy_experiment.pipeline import RunConfig, SycophancyPipeline
from sycophancy_experiment.prompts import MCQ, PRESSURE_PROMPTS, PRESSURE_TYPES, build_round1_messages
from sycophancy_experiment.utils import (
    aggregate_rates,
    balanced_intensities,
    choose_two_distinct,
    ensure_jsonl,
    read_jsonl,
    utc_run_dir,
)


def parse_models_arg(raw: str) -> tuple[str, ...]:
    models = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not models:
        raise ValueError("`--models` cannot be empty.")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: run first pressure round.")
    parser.add_argument("--stage1-dir", type=str, required=True, help="Directory containing stage1 outputs.")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument(
        "--models",
        type=str,
        default="claude-opus-4.6,deepseek-v3.2,gpt-5.2",
        help="Comma-separated target model keys to evaluate in Round 1.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.0-flash",
        help="Model key used for classification/false reasoning generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel worker threads for round-1 API calls.",
    )
    return parser.parse_args()


def build_intensity_assignments(
    questions: List[MCQ], seed: int, target_models: Tuple[str, ...]
) -> Dict[Tuple[str, str, str], str]:
    assignments: Dict[Tuple[str, str, str], str] = {}
    for model_idx, model_name in enumerate(target_models):
        for pressure_idx, pressure_type in enumerate(PRESSURE_TYPES):
            intensities = balanced_intensities(
                n=len(questions),
                seed=seed + model_idx * 100 + pressure_idx * 17,
            )
            for q_idx, mcq in enumerate(questions):
                assignments[(mcq.question_id, model_name, pressure_type)] = intensities[q_idx]
    return assignments


def main() -> None:
    load_dotenv()
    args = parse_args()

    kept_rows = read_jsonl(os.path.join(args.stage1_dir, "kept_questions.jsonl"))
    with open(os.path.join(args.stage1_dir, "baseline_answers.json"), "r", encoding="utf-8") as f:
        baseline_answers = json.load(f)

    questions = [
        MCQ(
            question_id=row["question_id"],
            subject=row["subject"],
            domain=row["domain"],
            question=row["question"],
            choices=tuple(row["choices"]),
            correct_letter=row["correct_letter"],
        )
        for row in kept_rows
    ]

    sampled_questions = stratified_sample_by_domain(questions, sample_size=args.sample_size, seed=args.seed)

    requested_models = parse_models_arg(args.models)
    cfg = RunConfig(
        seed=args.seed,
        output_root=args.output_root,
        target_models=requested_models,
        judge_model=args.judge_model,
    )
    pipeline = SycophancyPipeline(cfg)
    target_models = tuple([m for m in requested_models if m in pipeline.clients])

    if not target_models:
        raise ValueError("None of the requested models are configured in available clients.")

    intensity_assignments = build_intensity_assignments(sampled_questions, args.seed, target_models)

    out_dir = utc_run_dir(os.path.join(args.output_root, "stage2"))
    sampled_rows = [
        {
            "question_id": q.question_id,
            "subject": q.subject,
            "domain": q.domain,
            "question": q.question,
            "choices": list(q.choices),
            "correct_letter": q.correct_letter,
        }
        for q in sampled_questions
    ]

    round1_rows: List[dict] = []
    generated_reasoning_tag = re.compile(r"\[(?:generated\s+reasoning[^\]]*)\]", re.IGNORECASE)

    deduped_specs: List[tuple[MCQ, str, str, str, str, str, str]] = []
    for mcq in sampled_questions:
        for model_name in target_models:
            baseline_answer = baseline_answers.get(mcq.question_id, {}).get(model_name, "").strip().upper()
            if baseline_answer not in {"A", "B", "C", "D"}:
                continue
            for pressure_type in PRESSURE_TYPES:
                intensity = intensity_assignments[(mcq.question_id, model_name, pressure_type)]
                options = PRESSURE_PROMPTS[pressure_type][intensity]
                rng = random.Random(f"{args.seed}-{mcq.question_id}-{model_name}-{pressure_type}")
                template_r1, template_r2 = choose_two_distinct(options, rng)
                deduped_specs.append((mcq, model_name, pressure_type, intensity, baseline_answer, template_r1, template_r2))

    def run_round1_trial(spec: tuple[MCQ, str, str, str, str, str, str]) -> dict:
        mcq, model_name, pressure_type, intensity, baseline_answer, template_r1, template_r2 = spec
        pressure_r1 = template_r1
        pressure_r2 = template_r2
        render_error = ""
        try:
            pressure_r1 = pipeline._render_pressure_prompt(mcq, template_r1)
            pressure_r2 = pipeline._render_pressure_prompt(mcq, template_r2)
        except Exception as exc:  # noqa: BLE001
            # Do not crash the entire stage if false-reasoning generation quota is exhausted.
            fallback = "there is a strong flaw in that argument."
            pressure_r1 = generated_reasoning_tag.sub(fallback, template_r1).strip()
            pressure_r2 = generated_reasoning_tag.sub(fallback, template_r2).strip()
            render_error = str(exc)

        messages = build_round1_messages(mcq, baseline_answer, pressure_r1)

        try:
            response = pipeline._chat(model_name, messages, temperature=0.0, max_tokens=1024)
            judged = pipeline._judge(mcq, baseline_answer, pressure_r1, response)
            cls = judged["classification"]
            evidence = judged.get("evidence", "")
            judge_raw = judged.get("judge_raw", "")
        except Exception as exc:  # noqa: BLE001
            response = f"[API_ERROR] {exc}"
            cls = "ERROR"
            evidence = f"render_error={render_error}" if render_error else ""
            judge_raw = ""
        else:
            if render_error:
                evidence = f"render_error={render_error}; {evidence}".strip("; ")

        return {
            "question_id": mcq.question_id,
            "subject": mcq.subject,
            "domain": mcq.domain,
            "question": mcq.question,
            "choices": list(mcq.choices),
            "correct_letter": mcq.correct_letter,
            "model": model_name,
            "pressure_type": pressure_type,
            "intensity": intensity,
            "round": 1,
            "baseline_answer": baseline_answer,
            "pressure_text": pressure_r1,
            "pressure_text_round2": pressure_r2,
            "response": response,
            "classification": cls,
            "evidence": evidence,
            "judge_raw": judge_raw,
        }

    max_workers = max(1, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_round1_trial, deduped_specs)
        for row in tqdm(results, total=len(deduped_specs), desc="Stage 2 round 1"):
            round1_rows.append(row)

    candidates = [row for row in round1_rows if row.get("classification") == "HOLDS"]
    clean_rows = [row for row in round1_rows if row.get("classification") in {"HOLDS", "CAVES", "HESITATES"}]

    ensure_jsonl(os.path.join(out_dir, "sampled_questions.jsonl"), sampled_rows)
    ensure_jsonl(os.path.join(out_dir, "round1_results.jsonl"), round1_rows)
    ensure_jsonl(os.path.join(out_dir, "round2_candidates.jsonl"), candidates)

    summary_main = aggregate_rates(clean_rows, keys=["model", "pressure_type", "round"])
    summary_intensity = aggregate_rates(clean_rows, keys=["model", "pressure_type", "intensity", "round"])
    summary_domain = aggregate_rates(clean_rows, keys=["model", "pressure_type", "domain", "round"])

    pd.DataFrame(summary_main).to_csv(os.path.join(out_dir, "round1_summary_by_model_pressure.csv"), index=False)
    pd.DataFrame(summary_intensity).to_csv(
        os.path.join(out_dir, "round1_summary_by_model_pressure_intensity.csv"), index=False
    )
    pd.DataFrame(summary_domain).to_csv(os.path.join(out_dir, "round1_summary_by_model_pressure_domain.csv"), index=False)

    metadata = {
        "seed": args.seed,
        "sample_size_target": args.sample_size,
        "num_kept_questions_input": len(questions),
        "num_sampled_questions": len(sampled_questions),
        "num_round1_trials": len(round1_rows),
        "num_round2_candidates": len(candidates),
        "target_models": list(target_models),
        "judge_model": args.judge_model,
        "workers": args.workers,
    }
    ensure_jsonl(os.path.join(out_dir, "stage2_metadata.jsonl"), [metadata])

    print(f"Stage 2 complete: {len(round1_rows)} trials, {len(candidates)} round-2 candidates")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
