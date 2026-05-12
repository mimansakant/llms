#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from sycophancy_experiment.pipeline import RunConfig, SycophancyPipeline
from sycophancy_experiment.prompts import MCQ, build_round2_messages
from sycophancy_experiment.utils import ensure_jsonl, read_jsonl, utc_run_dir


VALID_CLASSES = {"HOLDS", "CAVES", "HESITATES"}


def parse_models_arg(raw: str) -> tuple[str, ...]:
    models = tuple(x.strip() for x in raw.split(",") if x.strip())
    if raw.strip() and not models:
        raise ValueError("`--models` cannot be empty when provided.")
    return models


def key_for_row(row: dict) -> Tuple[str, str, str, str]:
    return (
        row.get("question_id", ""),
        row.get("model", ""),
        row.get("pressure_type", ""),
        row.get("intensity", ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry only Stage 3 rows that previously failed with classification=ERROR."
    )
    parser.add_argument("--stage2-dir", required=True, help="Stage 2 output directory (source of HOLDS candidates).")
    parser.add_argument(
        "--stage3-dir",
        required=True,
        help="Existing Stage 3 output directory that contains round2_results.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model keys to filter retries.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.0-flash",
        help="Judge model key.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers for retries. Keep low to reduce throttling risk.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    selected_models = parse_models_arg(args.models) if args.models.strip() else tuple()

    round1_rows = read_jsonl(os.path.join(args.stage2_dir, "round1_results.jsonl"))
    old_round2_rows = read_jsonl(os.path.join(args.stage3_dir, "round2_results.jsonl"))

    candidates = [row for row in round1_rows if row.get("classification") == "HOLDS"]
    if selected_models:
        candidates = [row for row in candidates if row.get("model") in selected_models]

    old_error_keys = {key_for_row(r) for r in old_round2_rows if r.get("classification") == "ERROR"}
    retry_candidates = [row for row in candidates if key_for_row(row) in old_error_keys]

    if not retry_candidates:
        print("No retry candidates found (no matching ERROR rows).")
        return

    target_models_for_config = selected_models if selected_models else tuple(
        sorted({row.get("model") for row in retry_candidates if row.get("model")})
    )
    cfg = RunConfig(
        seed=args.seed,
        output_root=args.output_root,
        target_models=target_models_for_config,
        judge_model=args.judge_model,
    )
    pipeline = SycophancyPipeline(cfg)

    def run_round2_trial(row: dict) -> dict:
        mcq = MCQ(
            question_id=row["question_id"],
            subject=row["subject"],
            domain=row["domain"],
            question=row["question"],
            choices=tuple(row["choices"]),
            correct_letter=row["correct_letter"],
        )
        model_name = row["model"]
        baseline_answer = row["baseline_answer"]
        pressure_r1 = row["pressure_text"]
        response_r1 = row["response"]
        pressure_r2 = row.get("pressure_text_round2", "").strip()

        if not pressure_r2:
            return {
                "question_id": mcq.question_id,
                "subject": mcq.subject,
                "domain": mcq.domain,
                "question": mcq.question,
                "choices": list(mcq.choices),
                "correct_letter": mcq.correct_letter,
                "model": model_name,
                "pressure_type": row["pressure_type"],
                "intensity": row["intensity"],
                "round": 2,
                "baseline_answer": baseline_answer,
                "pressure_text": pressure_r2,
                "response": "[SKIPPED] Missing round-2 prompt",
                "classification": "ERROR",
                "evidence": "",
                "judge_raw": "",
            }

        messages = build_round2_messages(
            mcq=mcq,
            baseline_answer=baseline_answer,
            pressure_round1=pressure_r1,
            response_round1=response_r1,
            pressure_round2=pressure_r2,
        )

        try:
            response_r2 = pipeline._chat(model_name, messages, temperature=0.0, max_tokens=512)
            judged = pipeline._judge(mcq, baseline_answer, pressure_r2, response_r2)
            cls = judged["classification"]
            evidence = judged.get("evidence", "")
            judge_raw = judged.get("judge_raw", "")
        except Exception as exc:  # noqa: BLE001
            response_r2 = f"[API_ERROR] {exc}"
            cls = "ERROR"
            evidence = ""
            judge_raw = ""

        return {
            "question_id": mcq.question_id,
            "subject": mcq.subject,
            "domain": mcq.domain,
            "question": mcq.question,
            "choices": list(mcq.choices),
            "correct_letter": mcq.correct_letter,
            "model": model_name,
            "pressure_type": row["pressure_type"],
            "intensity": row["intensity"],
            "round": 2,
            "baseline_answer": baseline_answer,
            "pressure_text": pressure_r2,
            "response": response_r2,
            "classification": cls,
            "evidence": evidence,
            "judge_raw": judge_raw,
        }

    retried_rows: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(run_round2_trial, retry_candidates)
        for row in tqdm(results, total=len(retry_candidates), desc="Stage 3 retry ERROR rows"):
            retried_rows.append(row)

    retried_map: Dict[Tuple[str, str, str, str], dict] = {key_for_row(r): r for r in retried_rows}
    merged_round2_rows: List[dict] = []
    for old in old_round2_rows:
        k = key_for_row(old)
        merged_round2_rows.append(retried_map.get(k, old))

    merged_error_count = sum(1 for r in merged_round2_rows if r.get("classification") == "ERROR")
    merged_valid_count = sum(1 for r in merged_round2_rows if r.get("classification") in VALID_CLASSES)
    retry_success_count = sum(1 for r in retried_rows if r.get("classification") in VALID_CLASSES)

    out_dir = utc_run_dir(os.path.join(args.output_root, "stage3_retry"))
    ensure_jsonl(os.path.join(out_dir, "retried_rows.jsonl"), retried_rows)
    ensure_jsonl(os.path.join(out_dir, "round2_results_merged.jsonl"), merged_round2_rows)
    ensure_jsonl(
        os.path.join(out_dir, "retry_metadata.jsonl"),
        [
            {
                "stage2_dir": args.stage2_dir,
                "stage3_dir": args.stage3_dir,
                "retry_candidates": len(retry_candidates),
                "retry_success_valid_classification": retry_success_count,
                "merged_round2_total_rows": len(merged_round2_rows),
                "merged_round2_valid_rows": merged_valid_count,
                "merged_round2_error_rows": merged_error_count,
                "judge_model": args.judge_model,
                "workers": args.workers,
                "models_filter": list(selected_models),
            }
        ],
    )

    print(f"Retry complete: {len(retry_candidates)} attempted, {retry_success_count} now valid.")
    print(f"Merged round2 rows: {len(merged_round2_rows)} total, {merged_valid_count} valid, {merged_error_count} ERROR.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
