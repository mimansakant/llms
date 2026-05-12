#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from sycophancy_experiment.pipeline import RunConfig, SycophancyPipeline


def parse_models_arg(raw: str) -> tuple[str, ...]:
    models = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not models:
        raise ValueError("`--models` cannot be empty.")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM sycophancy pressure experiment pipeline.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Target number of questions after Stage 1 filtering.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and template assignment.")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on loaded MMLU test questions (useful for pilot/debug runs).",
    )
    parser.add_argument(
        "--relaxed-filter",
        action="store_true",
        help="Use relaxed Stage 1 filtering (>= relaxed-min-correct models correct).",
    )
    parser.add_argument(
        "--relaxed-min-correct",
        type=int,
        default=2,
        help="Minimum number of models that must be correct in relaxed mode.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Directory where run output folder is created.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="claude-opus-4.6,deepseek-v3.2,gpt-5.2",
        help="Comma-separated target model keys to evaluate.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.0-flash",
        help="Model key used for response classification and false reasoning generation.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    cfg = RunConfig(
        sample_size=args.sample_size,
        seed=args.seed,
        max_questions=args.max_questions,
        strict_filter=not args.relaxed_filter,
        relaxed_min_correct=args.relaxed_min_correct,
        output_root=args.output_root,
        target_models=parse_models_arg(args.models),
        judge_model=args.judge_model,
    )

    pipeline = SycophancyPipeline(cfg)
    out_dir = pipeline.run()
    print(f"Run completed. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
