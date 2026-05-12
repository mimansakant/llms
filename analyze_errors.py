#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List

FINAL_STAGE2_DIRS = [
    "outputs/stage2/run_20260505_021830",  # Claude (final Stage 2)
    "outputs/stage2/run_595_deepseek_stage2",
    "outputs/stage2/run_595_gpt_stage2",
]

FINAL_STAGE3_DIRS = [
    "outputs/stage3/run_595_deepseek_stage3",
    "outputs/stage3/run_595_gpt_stage3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize Stage2/Stage3 error rates, causes, and cave-rate bounds."
    )
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="Analyze all discovered run_* directories instead of the curated final runs.",
    )
    p.add_argument(
        "--stage2-dirs",
        nargs="*",
        default=None,
        help="Explicit Stage2 run dirs. Overrides defaults.",
    )
    p.add_argument(
        "--stage3-dirs",
        nargs="*",
        default=None,
        help="Explicit Stage3 run dirs. Overrides defaults.",
    )
    return p.parse_args()


def discover(default_glob: str, given: List[str] | None) -> List[str]:
    if given:
        return given
    return sorted(glob.glob(default_glob))


def classify_error_cause(msg: str) -> str:
    s = (msg or "").lower()
    if "throttling" in s or "rate limit" in s or "too many tokens" in s:
        return "throttling"
    if "no longer available" in s or "not_found" in s or "not found" in s:
        return "model_unavailable"
    if "connection refused" in s or "max retries exceeded" in s:
        return "connection_refused"
    if "timed out" in s or "timeout" in s:
        return "timeout"
    if "[api_error]" in s:
        return "api_error_other"
    return "other"


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_run(run_dir: str, stage: str, file_name: str) -> Dict[str, object] | None:
    path = os.path.join(run_dir, file_name)
    if not os.path.exists(path):
        return None

    rows = load_jsonl(path)
    total = len(rows)
    cls = Counter(r.get("classification", "") for r in rows)
    errors = cls.get("ERROR", 0)
    caves = cls.get("CAVES", 0)
    holds = cls.get("HOLDS", 0)
    hesitates = cls.get("HESITATES", 0)

    non_error = max(1, total - errors)
    cave_excl_error = caves / non_error
    cave_best_case = caves / max(1, total)
    cave_worst_case = (caves + errors) / max(1, total)

    cause_counts: Counter[str] = Counter()
    errors_by_pressure: Counter[str] = Counter()
    total_by_pressure: Counter[str] = Counter()

    for r in rows:
        pressure = str(r.get("pressure_type", "unknown"))
        total_by_pressure[pressure] += 1
        if r.get("classification") == "ERROR":
            errors_by_pressure[pressure] += 1
            cause_counts[classify_error_cause(str(r.get("response", "")))] += 1

    pressure_error_rates = {
        p: errors_by_pressure[p] / max(1, total_by_pressure[p]) for p in sorted(total_by_pressure.keys())
    }

    return {
        "run_dir": run_dir,
        "stage": stage,
        "total": total,
        "caves": caves,
        "holds": holds,
        "hesitates": hesitates,
        "errors": errors,
        "error_rate": errors / max(1, total),
        "cave_rate_excl_error": cave_excl_error,
        "cave_rate_best_case": cave_best_case,
        "cave_rate_worst_case": cave_worst_case,
        "error_causes": dict(cause_counts),
        "pressure_error_rates": pressure_error_rates,
    }


def fmt_pct(x: float) -> str:
    return f"{100*x:.2f}%"


def main() -> None:
    args = parse_args()

    if args.stage2_dirs is not None:
        stage2_dirs = args.stage2_dirs
    elif args.all_runs:
        stage2_dirs = discover("outputs/stage2/run_*", None)
    else:
        stage2_dirs = [d for d in FINAL_STAGE2_DIRS if os.path.isdir(d)]

    if args.stage3_dirs is not None:
        stage3_dirs = args.stage3_dirs
    elif args.all_runs:
        stage3_dirs = discover("outputs/stage3/run_*", None)
    else:
        stage3_dirs = [d for d in FINAL_STAGE3_DIRS if os.path.isdir(d)]

    summaries: List[Dict[str, object]] = []
    for d in stage2_dirs:
        s = summarize_run(d, "stage2", "round1_results.jsonl")
        if s:
            summaries.append(s)
    for d in stage3_dirs:
        s = summarize_run(d, "stage3", "round2_results.jsonl")
        if s:
            summaries.append(s)

    if not summaries:
        print("No runs found.")
        return

    mode = "ALL discovered runs" if args.all_runs else "FINAL run set"
    print(f"# Error Analysis ({mode})")
    print()
    print("| stage | run_dir | total | errors | error_rate | cave_excl_error | cave_best | cave_worst |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s['stage']} | {s['run_dir']} | {s['total']} | {s['errors']} | {fmt_pct(float(s['error_rate']))} | "
            f"{fmt_pct(float(s['cave_rate_excl_error']))} | {fmt_pct(float(s['cave_rate_best_case']))} | {fmt_pct(float(s['cave_rate_worst_case']))} |"
        )

    print("\n## Error Causes")
    for s in summaries:
        print(f"\n### {s['run_dir']} ({s['stage']})")
        causes: Dict[str, int] = dict(s["error_causes"])
        if not causes:
            print("- none")
            continue
        for k, v in sorted(causes.items(), key=lambda kv: kv[1], reverse=True):
            print(f"- {k}: {v}")

    print("\n## Error Rate by Pressure Type")
    for s in summaries:
        rates: Dict[str, float] = dict(s["pressure_error_rates"])
        if not rates:
            continue
        pretty = ", ".join(f"{k}={fmt_pct(v)}" for k, v in rates.items())
        print(f"- {s['run_dir']} ({s['stage']}): {pretty}")


if __name__ == "__main__":
    main()
