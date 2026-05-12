#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from sycophancy_experiment.utils import ensure_jsonl, read_jsonl, utc_run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge separate Stage 1 runs into one combined baseline dataset.")
    p.add_argument("--deepseek-run-dir", required=True, help="Stage 1 run dir containing DeepSeek results.")
    p.add_argument("--gpt-run-dir", required=True, help="Stage 1 run dir containing GPT results.")
    p.add_argument("--claude-run-dir", required=True, help="Stage 1 run dir containing Claude results.")
    p.add_argument("--output-root", default="outputs/stage1", help="Root output dir for merged run.")
    return p.parse_args()


def _index_by_qid(path: str) -> Dict[str, dict]:
    rows = read_jsonl(path)
    return {r["question_id"]: r for r in rows}


def _load_question_bank(run_dirs: List[str]) -> Dict[str, dict]:
    bank: Dict[str, dict] = {}
    for run_dir in run_dirs:
        kp = os.path.join(run_dir, "kept_questions.jsonl")
        if not os.path.exists(kp):
            continue
        for row in read_jsonl(kp):
            qid = row.get("question_id")
            question = row.get("question")
            choices = row.get("choices")
            if not qid or question is None or choices is None:
                continue
            if not isinstance(choices, list) or len(choices) != 4:
                continue
            bank[qid] = {
                "question": question,
                "choices": choices,
                "subject": row.get("subject"),
                "domain": row.get("domain"),
                "correct_letter": row.get("correct_letter"),
            }
    return bank


def main() -> None:
    args = parse_args()

    d_path = os.path.join(args.deepseek_run_dir, "stage1_baseline.jsonl")
    g_path = os.path.join(args.gpt_run_dir, "stage1_baseline.jsonl")
    c_path = os.path.join(args.claude_run_dir, "stage1_baseline.jsonl")

    deepseek = _index_by_qid(d_path)
    gpt = _index_by_qid(g_path)
    claude = _index_by_qid(c_path)
    question_bank = _load_question_bank([args.deepseek_run_dir, args.gpt_run_dir, args.claude_run_dir])

    common_qids = sorted(set(deepseek) & set(gpt) & set(claude))
    merged_rows: List[dict] = []
    kept_rows: List[dict] = []
    baseline_answers: Dict[str, Dict[str, str]] = {}

    for qid in common_qids:
        d = deepseek[qid]
        g = gpt[qid]
        c = claude[qid]

        row = {
            "question_id": qid,
            "subject": d.get("subject"),
            "domain": d.get("domain"),
            "correct_letter": d.get("correct_letter"),
            "deepseek-v3.2_raw": d.get("deepseek-v3.2_raw"),
            "deepseek-v3.2_pred": d.get("deepseek-v3.2_pred"),
            "deepseek-v3.2_correct": d.get("deepseek-v3.2_correct"),
            "gpt-5.2_raw": g.get("gpt-5.2_raw"),
            "gpt-5.2_pred": g.get("gpt-5.2_pred"),
            "gpt-5.2_correct": g.get("gpt-5.2_correct"),
            "claude-opus-4.6_raw": c.get("claude-opus-4.6_raw"),
            "claude-opus-4.6_pred": c.get("claude-opus-4.6_pred"),
            "claude-opus-4.6_correct": c.get("claude-opus-4.6_correct"),
        }

        n_correct = sum(
            1
            for k in (
                "deepseek-v3.2_correct",
                "gpt-5.2_correct",
                "claude-opus-4.6_correct",
            )
            if row.get(k) is True
        )
        row["num_models_correct"] = n_correct
        row["kept_for_pressure"] = n_correct == 3
        merged_rows.append(row)

        baseline_answers[qid] = {
            "deepseek-v3.2": str(row.get("deepseek-v3.2_pred") or ""),
            "gpt-5.2": str(row.get("gpt-5.2_pred") or ""),
            "claude-opus-4.6": str(row.get("claude-opus-4.6_pred") or ""),
        }

        if row["kept_for_pressure"]:
            qmeta = question_bank.get(qid, {})
            kept_rows.append(
                {
                    "question_id": qid,
                    "subject": row["subject"],
                    "domain": row["domain"],
                    "question": qmeta.get("question"),
                    "choices": qmeta.get("choices"),
                    "correct_letter": row["correct_letter"],
                }
            )

    out_dir = utc_run_dir(os.path.join(args.output_root, "merged"))
    ensure_jsonl(os.path.join(out_dir, "stage1_baseline.jsonl"), merged_rows)
    ensure_jsonl(os.path.join(out_dir, "kept_questions.jsonl"), kept_rows)
    with open(os.path.join(out_dir, "baseline_answers.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_answers, f, ensure_ascii=False, indent=2)

    meta = {
        "deepseek_run_dir": args.deepseek_run_dir,
        "gpt_run_dir": args.gpt_run_dir,
        "claude_run_dir": args.claude_run_dir,
        "num_common_questions": len(common_qids),
        "num_kept_questions": len(kept_rows),
        "num_question_bank_rows": len(question_bank),
        "num_kept_with_missing_question_or_choices": sum(
            1 for r in kept_rows if (r.get("question") is None or r.get("choices") is None)
        ),
    }
    ensure_jsonl(os.path.join(out_dir, "merge_metadata.jsonl"), [meta])

    print(f"Merged common questions: {len(common_qids)}")
    print(f"Final kept questions (all 3 correct): {len(kept_rows)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
