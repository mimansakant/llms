#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from sycophancy_experiment.clients import build_clients_from_env
from sycophancy_experiment.prompts import MCQ, build_baseline_messages
from sycophancy_experiment.utils import extract_choice_letter, read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill baseline answers for a model on an existing Stage-1 merged kept-question set."
    )
    p.add_argument(
        "--src-dir",
        type=str,
        default="outputs/stage1/merged/run_595_kept_common",
        help="Source merged Stage-1 directory containing kept_questions.jsonl and baseline_answers.json",
    )
    p.add_argument(
        "--dst-dir",
        type=str,
        default="outputs/stage1/merged/run_595_kept_common_with_qwen",
        help="Destination directory to write updated kept_questions.jsonl and baseline_answers.json",
    )
    p.add_argument(
        "--model-key",
        type=str,
        default="qwen2.5-7b",
        help="Internal model key configured in this repo clients.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens for baseline answer generation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    kept_path = os.path.join(args.src_dir, "kept_questions.jsonl")
    base_path = os.path.join(args.src_dir, "baseline_answers.json")
    if not os.path.exists(kept_path):
        raise FileNotFoundError(f"Missing file: {kept_path}")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing file: {base_path}")

    os.makedirs(args.dst_dir, exist_ok=True)

    kept_rows = read_jsonl(kept_path)
    with open(base_path, "r", encoding="utf-8") as f:
        baseline_answers = json.load(f)

    clients = build_clients_from_env()
    if args.model_key not in clients:
        raise ValueError(
            f"Client not configured: {args.model_key}. "
            "Set required env vars (for Qwen: QWEN_BASE_URL and QWEN_MODEL, optional QWEN_API_KEY)."
        )
    model_client = clients[args.model_key]

    for row in tqdm(kept_rows, desc=f"Backfill baseline for {args.model_key}"):
        mcq = MCQ(
            question_id=row["question_id"],
            subject=row["subject"],
            domain=row["domain"],
            question=row["question"],
            choices=tuple(row["choices"]),
            correct_letter=row["correct_letter"],
        )

        response = model_client.chat(
            build_baseline_messages(mcq),
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        pred = extract_choice_letter(response) or ""
        baseline_answers.setdefault(row["question_id"], {})[args.model_key] = pred

    out_kept = os.path.join(args.dst_dir, "kept_questions.jsonl")
    out_base = os.path.join(args.dst_dir, "baseline_answers.json")

    with open(out_kept, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_base, "w", encoding="utf-8") as f:
        json.dump(baseline_answers, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_kept}")
    print(f"Wrote: {out_base}")
    print(f"Done. You can now run Stage 2 using: --stage1-dir {args.dst_dir}")


if __name__ == "__main__":
    main()

