#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import pandas as pd

from sycophancy_experiment.utils import aggregate_rates, read_jsonl, utc_run_dir


VALID_CLASSES = {"HOLDS", "CAVES", "HESITATES"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze Stage 2/3 experiment outputs and produce summary tables + report."
    )
    p.add_argument(
        "--stage2-dirs",
        nargs="*",
        default=[],
        help="Stage 2 run directories. If omitted, auto-discovers outputs/stage2/run_*",
    )
    p.add_argument(
        "--stage3-dirs",
        nargs="*",
        default=[],
        help="Stage 3 run directories. If omitted, auto-discovers outputs/stage3/run_*",
    )
    p.add_argument("--output-root", default="outputs/analysis", help="Root output directory")
    return p.parse_args()


def _discover(run_glob: str, provided: List[str]) -> List[str]:
    if provided:
        return provided
    return sorted(glob.glob(run_glob))


def _load_round1(stage2_dirs: List[str]) -> List[dict]:
    rows: List[dict] = []
    for d in stage2_dirs:
        p = os.path.join(d, "round1_results.jsonl")
        if not os.path.exists(p):
            continue
        rows.extend(read_jsonl(p))
    return rows


def _load_round2(stage3_dirs: List[str]) -> List[dict]:
    rows: List[dict] = []
    for d in stage3_dirs:
        p = os.path.join(d, "round2_results.jsonl")
        if not os.path.exists(p):
            continue
        rows.extend(read_jsonl(p))
    return rows


def _clean(rows: List[dict]) -> List[dict]:
    return [r for r in rows if r.get("classification") in VALID_CLASSES]


def _dedupe(rows: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for r in rows:
        key = (
            r.get("question_id"),
            r.get("model"),
            r.get("pressure_type"),
            r.get("intensity"),
            r.get("round"),
            r.get("pressure_text"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _write_csv(path: str, rows: List[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _compute_sustained_delta(summary_main: List[dict]) -> pd.DataFrame:
    if not summary_main:
        return pd.DataFrame()
    df = pd.DataFrame(summary_main)
    r1 = df[df["round"] == 1][["model", "pressure_type", "cave_rate"]].rename(
        columns={"cave_rate": "round1_cave_rate"}
    )
    r2 = df[df["round"] == 2][["model", "pressure_type", "cave_rate"]].rename(
        columns={"cave_rate": "round2_cave_rate"}
    )
    merged = r1.merge(r2, on=["model", "pressure_type"], how="outer")
    merged["sustained_pressure_delta"] = merged["round2_cave_rate"] - merged["round1_cave_rate"]
    return merged


def _render_report(
    out_path: str,
    stage2_dirs: List[str],
    stage3_dirs: List[str],
    n_round1: int,
    n_round2: int,
    summary_main: List[dict],
) -> None:
    lines: List[str] = []
    lines.append("# Experiment Analysis Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Stage 2 runs loaded: {len(stage2_dirs)}")
    for d in stage2_dirs:
        lines.append(f"  - {d}")
    lines.append(f"- Stage 3 runs loaded: {len(stage3_dirs)}")
    for d in stage3_dirs:
        lines.append(f"  - {d}")
    lines.append("")
    lines.append("## Row Counts")
    lines.append(f"- Round 1 rows (clean classes): {n_round1}")
    lines.append(f"- Round 2 rows (clean classes): {n_round2}")
    lines.append("")

    if summary_main:
        df = pd.DataFrame(summary_main).sort_values(["round", "model", "pressure_type"])
        lines.append("## Cave Rates By Model/Pressure/Round")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("No summary rows were produced.")

    if n_round2 == 0:
        lines.append("## Note")
        lines.append("- No Stage 3 round-2 data found yet. Sustained-pressure analysis will be partial until Stage 3 runs are provided.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    stage2_dirs = _discover("outputs/stage2/run_*", args.stage2_dirs)
    stage3_dirs = _discover("outputs/stage3/run_*", args.stage3_dirs)

    round1 = _dedupe(_clean(_load_round1(stage2_dirs)))
    round2 = _dedupe(_clean(_load_round2(stage3_dirs)))

    combined = _dedupe(round1 + round2)

    out_dir = utc_run_dir(args.output_root)

    # Round 1 focused summaries
    round1_main = aggregate_rates(round1, keys=["model", "pressure_type", "round"])
    round1_intensity = aggregate_rates(round1, keys=["model", "pressure_type", "intensity", "round"])
    round1_domain = aggregate_rates(round1, keys=["model", "pressure_type", "domain", "round"])

    _write_csv(os.path.join(out_dir, "round1_summary_by_model_pressure_round.csv"), round1_main)
    _write_csv(os.path.join(out_dir, "round1_summary_by_model_pressure_intensity_round.csv"), round1_intensity)
    _write_csv(os.path.join(out_dir, "round1_summary_by_model_pressure_domain_round.csv"), round1_domain)

    # Round 2 focused summaries (if present)
    round2_main = aggregate_rates(round2, keys=["model", "pressure_type", "round"])
    round2_intensity = aggregate_rates(round2, keys=["model", "pressure_type", "intensity", "round"])
    round2_domain = aggregate_rates(round2, keys=["model", "pressure_type", "domain", "round"])

    _write_csv(os.path.join(out_dir, "round2_summary_by_model_pressure_round.csv"), round2_main)
    _write_csv(os.path.join(out_dir, "round2_summary_by_model_pressure_intensity_round.csv"), round2_intensity)
    _write_csv(os.path.join(out_dir, "round2_summary_by_model_pressure_domain_round.csv"), round2_domain)

    # Combined summaries
    combined_main = aggregate_rates(combined, keys=["model", "pressure_type", "round"])
    combined_intensity = aggregate_rates(combined, keys=["model", "pressure_type", "intensity", "round"])
    combined_domain = aggregate_rates(combined, keys=["model", "pressure_type", "domain", "round"])

    _write_csv(os.path.join(out_dir, "combined_summary_by_model_pressure_round.csv"), combined_main)
    _write_csv(os.path.join(out_dir, "combined_summary_by_model_pressure_intensity_round.csv"), combined_intensity)
    _write_csv(os.path.join(out_dir, "combined_summary_by_model_pressure_domain_round.csv"), combined_domain)

    sustained = _compute_sustained_delta(combined_main)
    sustained.to_csv(os.path.join(out_dir, "combined_sustained_pressure_delta.csv"), index=False)

    pd.DataFrame(round1).to_json(os.path.join(out_dir, "round1_clean_rows.json"), orient="records", indent=2)
    pd.DataFrame(round2).to_json(os.path.join(out_dir, "round2_clean_rows.json"), orient="records", indent=2)
    pd.DataFrame(combined).to_json(os.path.join(out_dir, "combined_clean_rows.json"), orient="records", indent=2)

    meta = {
        "stage2_dirs": stage2_dirs,
        "stage3_dirs": stage3_dirs,
        "round1_rows_clean": len(round1),
        "round2_rows_clean": len(round2),
        "combined_rows_clean": len(combined),
    }
    with open(os.path.join(out_dir, "analysis_metadata.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2)

    _render_report(
        out_path=os.path.join(out_dir, "analysis_report.md"),
        stage2_dirs=stage2_dirs,
        stage3_dirs=stage3_dirs,
        n_round1=len(round1),
        n_round2=len(round2),
        summary_main=combined_main,
    )

    print(f"Analysis complete. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
