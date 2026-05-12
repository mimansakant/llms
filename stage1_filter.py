#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from sycophancy_experiment.mmlu import load_mmlu_test
from sycophancy_experiment.pipeline import RunConfig, SycophancyPipeline
from sycophancy_experiment.prompts import MCQ, build_baseline_messages
from sycophancy_experiment.utils import ensure_jsonl, extract_choice_letter, utc_run_dir


def parse_models_arg(raw: str) -> tuple[str, ...]:
    models = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not models:
        raise ValueError("`--models` cannot be empty.")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: baseline filtering (questions all models answer correctly).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--relaxed-filter", action="store_true")
    parser.add_argument("--relaxed-min-correct", type=int, default=2)
    parser.add_argument(
        "--models",
        type=str,
        default="claude-opus-4.6,deepseek-v3.2,gpt-5.2",
        help="Comma-separated target model keys to evaluate in Stage 1.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.0-flash",
        help="Judge model key (kept for config consistency; not used directly in Stage 1 calls).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads for baseline API calls.",
    )
    return parser.parse_args()


def run_stage1_parallel(
    pipeline: SycophancyPipeline,
    questions: List[MCQ],
    strict_filter: bool,
    relaxed_min_correct: int,
    workers: int,
) -> Tuple[List[dict], List[MCQ], Dict[str, Dict[str, str]]]:
    def baseline_max_tokens_for_model(model_name: str) -> int:
        # DeepSeek reasoning variants often need a larger completion budget to emit
        # a final answer token (A/B/C/D) after internal reasoning.
        if model_name == "deepseek-v3.2":
            return 512
        return 128

    def evaluate_question(mcq: MCQ) -> Tuple[dict, MCQ | None, Dict[str, str]]:
        row = {
            "question_id": mcq.question_id,
            "subject": mcq.subject,
            "domain": mcq.domain,
            "correct_letter": mcq.correct_letter,
        }
        n_correct = 0
        answer_map: Dict[str, str] = {}

        for model_name in pipeline.target_models:
            try:
                resp = pipeline._chat(
                    model_name,
                    build_baseline_messages(mcq),
                    temperature=0.0,
                    max_tokens=baseline_max_tokens_for_model(model_name),
                )
                pred = extract_choice_letter(resp)
                is_correct = pred == mcq.correct_letter
                if is_correct:
                    n_correct += 1
                answer_map[model_name] = pred or ""
                row[f"{model_name}_raw"] = resp
                row[f"{model_name}_pred"] = pred
                row[f"{model_name}_correct"] = is_correct
            except Exception as exc:  # noqa: BLE001
                answer_map[model_name] = ""
                row[f"{model_name}_raw"] = f"[API_ERROR] {exc}"
                row[f"{model_name}_pred"] = None
                row[f"{model_name}_correct"] = False

        keep = n_correct == len(pipeline.target_models) if strict_filter else n_correct >= relaxed_min_correct
        row["num_models_correct"] = n_correct
        row["kept_for_pressure"] = keep
        return row, (mcq if keep else None), answer_map

    baseline_rows: List[dict] = []
    kept_questions: List[MCQ] = []
    baseline_answers: Dict[str, Dict[str, str]] = {}

    max_workers = max(1, workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(evaluate_question, questions)
        for row, kept_mcq, answer_map in tqdm(results, total=len(questions), desc="Stage 1 baseline"):
            baseline_rows.append(row)
            baseline_answers[row["question_id"]] = answer_map
            if kept_mcq is not None:
                kept_questions.append(kept_mcq)

    return baseline_rows, kept_questions, baseline_answers


def main() -> None:
    load_dotenv()
    args = parse_args()

    cfg = RunConfig(
        seed=args.seed,
        max_questions=args.max_questions,
        strict_filter=not args.relaxed_filter,
        relaxed_min_correct=args.relaxed_min_correct,
        output_root=args.output_root,
        target_models=parse_models_arg(args.models),
        judge_model=args.judge_model,
    )
    pipeline = SycophancyPipeline(cfg)

    out_dir = utc_run_dir(os.path.join(args.output_root, "stage1"))
    questions = load_mmlu_test(max_questions=args.max_questions, seed=args.seed)
    baseline_rows, kept_questions, baseline_answers = run_stage1_parallel(
        pipeline=pipeline,
        questions=questions,
        strict_filter=not args.relaxed_filter,
        relaxed_min_correct=args.relaxed_min_correct,
        workers=args.workers,
    )

    kept_question_rows = [
        {
            "question_id": q.question_id,
            "subject": q.subject,
            "domain": q.domain,
            "question": q.question,
            "choices": list(q.choices),
            "correct_letter": q.correct_letter,
        }
        for q in kept_questions
    ]

    ensure_jsonl(os.path.join(out_dir, "stage1_baseline.jsonl"), baseline_rows)
    ensure_jsonl(os.path.join(out_dir, "kept_questions.jsonl"), kept_question_rows)

    with open(os.path.join(out_dir, "baseline_answers.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_answers, f, ensure_ascii=False, indent=2)

    metadata = {
        "seed": args.seed,
        "strict_filter": not args.relaxed_filter,
        "relaxed_min_correct": args.relaxed_min_correct,
        "num_questions_loaded": len(questions),
        "num_questions_kept": len(kept_questions),
        "max_questions": args.max_questions,
        "target_models": list(cfg.target_models),
        "judge_model": cfg.judge_model,
        "workers": args.workers,
    }
    ensure_jsonl(os.path.join(out_dir, "stage1_metadata.jsonl"), [metadata])

    print(f"Stage 1 complete: {len(kept_questions)} / {len(questions)} questions kept")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
