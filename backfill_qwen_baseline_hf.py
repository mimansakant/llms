#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sycophancy_experiment.prompts import MCQ, build_baseline_messages
from sycophancy_experiment.utils import extract_choice_letter, read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill qwen2.5-7b baseline answers on an existing merged kept-question set using Hugging Face."
    )
    p.add_argument(
        "--src-dir",
        type=str,
        default="outputs/stage1/merged/run_595_kept_common",
        help="Source dir containing kept_questions.jsonl and baseline_answers.json",
    )
    p.add_argument(
        "--dst-dir",
        type=str,
        default="outputs/stage1/merged/run_595_kept_common_with_qwen",
        help="Destination dir to write updated files",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--model-key",
        type=str,
        default="qwen2.5-7b",
        help="Key to write into baseline_answers.json",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Generation length for baseline answer extraction",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    kept_path = os.path.join(args.src_dir, "kept_questions.jsonl")
    baseline_path = os.path.join(args.src_dir, "baseline_answers.json")
    if not os.path.exists(kept_path):
        raise FileNotFoundError(f"Missing file: {kept_path}")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Missing file: {baseline_path}")

    os.makedirs(args.dst_dir, exist_ok=True)

    kept_rows = read_jsonl(kept_path)
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_answers = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    for row in tqdm(kept_rows, desc=f"Backfill baseline ({args.model_key})"):
        mcq = MCQ(
            question_id=row["question_id"],
            subject=row["subject"],
            domain=row["domain"],
            question=row["question"],
            choices=tuple(row["choices"]),
            correct_letter=row["correct_letter"],
        )
        messages = build_baseline_messages(mcq)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
    print(f"Done. Run stage2 with: --stage1-dir {args.dst_dir}")


if __name__ == "__main__":
    main()

