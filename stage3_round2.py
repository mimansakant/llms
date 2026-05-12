#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from sycophancy_experiment.pipeline import RunConfig, SycophancyPipeline
from sycophancy_experiment.prompts import MCQ, build_round2_messages
from sycophancy_experiment.utils import aggregate_rates, ensure_jsonl, read_jsonl, utc_run_dir


def parse_models_arg(raw: str) -> tuple[str, ...]:
    models = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not models:
        raise ValueError("`--models` cannot be empty.")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: run second pressure round on HOLDS from round 1.")
    parser.add_argument("--stage2-dir", type=str, required=True, help="Directory containing stage2 outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model keys to filter round-2 execution (default: use all candidates).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.0-flash",
        help="Model key used for classification in round 2.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel worker threads for round-2 API calls.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    round1_rows = read_jsonl(os.path.join(args.stage2_dir, "round1_results.jsonl"))
    candidates = [row for row in round1_rows if row.get("classification") == "HOLDS"]
    selected_models = parse_models_arg(args.models) if args.models.strip() else tuple()
    if selected_models:
        candidates = [row for row in candidates if row.get("model") in selected_models]

    target_models_for_config = selected_models if selected_models else tuple(
        sorted({row.get("model") for row in candidates if row.get("model")})
    )
    cfg = RunConfig(
        seed=args.seed,
        output_root=args.output_root,
        target_models=target_models_for_config,
        judge_model=args.judge_model,
    )
    pipeline = SycophancyPipeline(cfg)

    out_dir = utc_run_dir(os.path.join(args.output_root, "stage3"))

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
            response_r2 = pipeline._chat(model_name, messages, temperature=0.0, max_tokens=1024)
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

    round2_rows: List[dict] = []
    max_workers = max(1, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_round2_trial, candidates)
        for row in tqdm(results, total=len(candidates), desc="Stage 3 round 2"):
            round2_rows.append(row)

    combined = round1_rows + round2_rows
    clean_rows = [row for row in combined if row.get("classification") in {"HOLDS", "CAVES", "HESITATES"}]

    ensure_jsonl(os.path.join(out_dir, "round2_results.jsonl"), round2_rows)
    ensure_jsonl(os.path.join(out_dir, "combined_round1_round2_results.jsonl"), combined)

    summary_main = aggregate_rates(clean_rows, keys=["model", "pressure_type", "round"])
    summary_intensity = aggregate_rates(clean_rows, keys=["model", "pressure_type", "intensity", "round"])
    summary_domain = aggregate_rates(clean_rows, keys=["model", "pressure_type", "domain", "round"])

    pd.DataFrame(summary_main).to_csv(os.path.join(out_dir, "summary_by_model_pressure_round.csv"), index=False)
    pd.DataFrame(summary_intensity).to_csv(
        os.path.join(out_dir, "summary_by_model_pressure_intensity_round.csv"), index=False
    )
    pd.DataFrame(summary_domain).to_csv(
        os.path.join(out_dir, "summary_by_model_pressure_domain_round.csv"), index=False
    )

    if summary_main:
        df = pd.DataFrame(summary_main)
        r1 = df[df["round"] == 1][["model", "pressure_type", "cave_rate"]].rename(
            columns={"cave_rate": "round1_cave_rate"}
        )
        r2 = df[df["round"] == 2][["model", "pressure_type", "cave_rate"]].rename(
            columns={"cave_rate": "round2_cave_rate"}
        )
        merged = r1.merge(r2, on=["model", "pressure_type"], how="outer")
        merged["sustained_pressure_delta"] = merged["round2_cave_rate"] - merged["round1_cave_rate"]
        merged.to_csv(os.path.join(out_dir, "summary_sustained_pressure_delta.csv"), index=False)

    metadata = {
        "seed": args.seed,
        "num_round1_rows": len(round1_rows),
        "num_round2_candidates": len(candidates),
        "num_round2_rows": len(round2_rows),
        "num_combined_rows": len(combined),
        "target_models_filter": list(selected_models),
        "judge_model": args.judge_model,
        "workers": args.workers,
    }
    ensure_jsonl(os.path.join(out_dir, "stage3_metadata.jsonl"), [metadata])

    print(f"Stage 3 complete: {len(round2_rows)} round-2 trials")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
