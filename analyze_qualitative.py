#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

FINAL_STAGE2_DIRS = [
    "outputs/stage2/run_20260505_021830",
    "outputs/stage2/run_595_deepseek_stage2",
    "outputs/stage2/run_595_gpt_stage2",
]

FINAL_STAGE3_DIRS = [
    "outputs/stage3/run_595_deepseek_stage3",
    "outputs/stage3/run_595_gpt_stage3",
]

LABELS = ("HOLDS", "CAVES", "HESITATES")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize qualitative behavior and print representative examples from final runs."
    )
    p.add_argument("--stage2-dirs", nargs="*", default=None, help="Override default final Stage2 run dirs.")
    p.add_argument("--stage3-dirs", nargs="*", default=None, help="Override default final Stage3 run dirs.")
    p.add_argument(
        "--examples-per-class",
        type=int,
        default=2,
        help="How many representative examples to print per model/class.",
    )
    p.add_argument(
        "--max-response-chars",
        type=int,
        default=220,
        help="Clip response text in printed examples.",
    )
    return p.parse_args()


def load_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pick_examples(rows: Iterable[dict], n: int) -> List[dict]:
    # Prefer substantive but still concise examples for report excerpts.
    filtered = [r for r in rows if (r.get("response") or "").strip()]
    substantive = [r for r in filtered if len(str(r.get("response", "")).strip()) >= 40]
    pool = substantive if substantive else filtered
    pool.sort(key=lambda r: len(str(r.get("response", ""))))
    return pool[:n]


def clip(text: str, k: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= k:
        return t
    return t[: k - 3] + "..."


def summarize_round(rows: List[dict], round_name: str, max_examples: int, max_chars: int) -> None:
    rows_clean = [r for r in rows if r.get("classification") != "ERROR"]
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for r in rows_clean:
        by_model[str(r.get("model", "unknown"))].append(r)

    print(f"## {round_name} Quantitative Snapshot (non-ERROR rows)")
    print("| model | n | caves | holds | hesitates | cave_rate |")
    print("|---|---:|---:|---:|---:|---:|")
    for model in sorted(by_model.keys()):
        model_rows = by_model[model]
        c = Counter(str(r.get("classification", "")) for r in model_rows)
        n = len(model_rows)
        cave_rate = c["CAVES"] / max(1, n)
        print(f"| {model} | {n} | {c['CAVES']} | {c['HOLDS']} | {c['HESITATES']} | {100*cave_rate:.2f}% |")

    print()
    print(f"## {round_name} Representative Examples")
    for model in sorted(by_model.keys()):
        print(f"\n### {model}")
        model_rows = by_model[model]
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for r in model_rows:
            grouped[str(r.get("classification", ""))].append(r)
        for label in LABELS:
            examples = pick_examples(grouped.get(label, []), max_examples)
            if not examples:
                continue
            print(f"- {label}:")
            for ex in examples:
                print(
                    "  "
                    + f"[{ex.get('question_id')}] type={ex.get('pressure_type')} intensity={ex.get('intensity')} "
                    + f"resp=\"{clip(str(ex.get('response', '')), max_chars)}\""
                )


def main() -> None:
    args = parse_args()
    stage2_dirs = args.stage2_dirs if args.stage2_dirs is not None else [d for d in FINAL_STAGE2_DIRS if os.path.isdir(d)]
    stage3_dirs = args.stage3_dirs if args.stage3_dirs is not None else [d for d in FINAL_STAGE3_DIRS if os.path.isdir(d)]

    round1_rows: List[dict] = []
    for d in stage2_dirs:
        p = os.path.join(d, "round1_results.jsonl")
        if os.path.exists(p):
            round1_rows.extend(load_jsonl(p))

    round2_rows: List[dict] = []
    for d in stage3_dirs:
        p = os.path.join(d, "round2_results.jsonl")
        if os.path.exists(p):
            round2_rows.extend(load_jsonl(p))

    if round1_rows:
        summarize_round(round1_rows, "Round 1", args.examples_per_class, args.max_response_chars)
        print()
    else:
        print("No Round 1 rows found.")

    if round2_rows:
        summarize_round(round2_rows, "Round 2", args.examples_per_class, args.max_response_chars)
    else:
        print("No Round 2 rows found.")


if __name__ == "__main__":
    main()
