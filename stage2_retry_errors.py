#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from sycophancy_experiment.pipeline import RunConfig, SycophancyPipeline
from sycophancy_experiment.prompts import MCQ, build_round1_messages
from sycophancy_experiment.utils import aggregate_rates, ensure_jsonl, read_jsonl, utc_run_dir


VALID_CLASSES = {"HOLDS", "CAVES", "HESITATES"}


def parse_models_arg(raw: str) -> tuple[str, ...]:
    models = tuple(x.strip() for x in raw.split(",") if x.strip())
    if raw.strip() and not models:
        raise ValueError("`--models` cannot be empty when provided.")
    return models


def row_key(row: dict) -> Tuple[str, str, str, str]:
    return (
        str(row.get("question_id", "")),
        str(row.get("model", "")),
        str(row.get("pressure_type", "")),
        str(row.get("intensity", "")),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retry only ERROR rows from Stage 2 round1 results.")
    p.add_argument("--stage2-dir", required=True, help="Existing Stage 2 run dir with round1_results.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default="outputs")
    p.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model filter for retry rows (e.g., gpt-5.2).",
    )
    p.add_argument("--judge-model", default="gemini-2.0-flash", help="Judge model key.")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers for retries.")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    selected_models = parse_models_arg(args.models) if args.models.strip() else tuple()

    round1_path = os.path.join(args.stage2_dir, "round1_results.jsonl")
    old_rows = read_jsonl(round1_path)

    retry_rows = [r for r in old_rows if r.get("classification") == "ERROR"]
    if selected_models:
        retry_rows = [r for r in retry_rows if r.get("model") in selected_models]

    if not retry_rows:
        print("No ERROR rows found to retry.")
        return

    target_models = selected_models if selected_models else tuple(
        sorted({r.get("model") for r in retry_rows if r.get("model")})
    )
    cfg = RunConfig(seed=args.seed, output_root=args.output_root, target_models=target_models, judge_model=args.judge_model)
    pipeline = SycophancyPipeline(cfg)

    def retry_one(old: dict) -> dict:
        mcq = MCQ(
            question_id=old["question_id"],
            subject=old["subject"],
            domain=old["domain"],
            question=old["question"],
            choices=tuple(old["choices"]),
            correct_letter=old["correct_letter"],
        )
        model_name = old["model"]
        baseline_answer = old["baseline_answer"]
        pressure_r1 = old["pressure_text"]
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
            evidence = ""
            judge_raw = ""

        out = dict(old)
        out["response"] = response
        out["classification"] = cls
        out["evidence"] = evidence
        out["judge_raw"] = judge_raw
        return out

    retried: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for row in tqdm(ex.map(retry_one, retry_rows), total=len(retry_rows), desc="Stage 2 retry ERROR rows"):
            retried.append(row)

    retried_map: Dict[Tuple[str, str, str, str], dict] = {row_key(r): r for r in retried}
    merged = [retried_map.get(row_key(r), r) for r in old_rows]

    candidates = [r for r in merged if r.get("classification") == "HOLDS"]
    clean = [r for r in merged if r.get("classification") in VALID_CLASSES]

    summary_main = aggregate_rates(clean, keys=["model", "pressure_type", "round"])
    summary_intensity = aggregate_rates(clean, keys=["model", "pressure_type", "intensity", "round"])
    summary_domain = aggregate_rates(clean, keys=["model", "pressure_type", "domain", "round"])

    out_dir = utc_run_dir(os.path.join(args.output_root, "stage2_retry"))
    ensure_jsonl(os.path.join(out_dir, "retried_rows.jsonl"), retried)
    ensure_jsonl(os.path.join(out_dir, "round1_results_merged.jsonl"), merged)
    ensure_jsonl(os.path.join(out_dir, "round2_candidates_merged.jsonl"), candidates)

    pd.DataFrame(summary_main).to_csv(os.path.join(out_dir, "round1_summary_by_model_pressure.csv"), index=False)
    pd.DataFrame(summary_intensity).to_csv(
        os.path.join(out_dir, "round1_summary_by_model_pressure_intensity.csv"), index=False
    )
    pd.DataFrame(summary_domain).to_csv(os.path.join(out_dir, "round1_summary_by_model_pressure_domain.csv"), index=False)

    old_errors = sum(1 for r in old_rows if r.get("classification") == "ERROR")
    new_errors = sum(1 for r in merged if r.get("classification") == "ERROR")
    fixed = old_errors - new_errors
    ensure_jsonl(
        os.path.join(out_dir, "retry_metadata.jsonl"),
        [
            {
                "stage2_dir": args.stage2_dir,
                "retry_rows_attempted": len(retry_rows),
                "old_error_rows": old_errors,
                "new_error_rows": new_errors,
                "fixed_rows": fixed,
                "merged_total_rows": len(merged),
                "merged_round2_candidates": len(candidates),
                "judge_model": args.judge_model,
                "workers": args.workers,
                "models_filter": list(selected_models),
            }
        ],
    )

    print(f"Retry complete: attempted {len(retry_rows)} rows, fixed {fixed}.")
    print(f"Merged rows: {len(merged)} total, {new_errors} ERROR remaining, {len(candidates)} HOLDS candidates.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
