from __future__ import annotations

import os
import random
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from sycophancy_experiment.clients import build_clients_from_env
from sycophancy_experiment.mmlu import load_mmlu_test, stratified_sample_by_domain
from sycophancy_experiment.prompts import (
    PRESSURE_PROMPTS,
    PRESSURE_TYPES,
    MCQ,
    build_baseline_messages,
    build_false_reasoning_messages,
    build_judge_messages,
    build_round1_messages,
    build_round2_messages,
)
from sycophancy_experiment.utils import (
    aggregate_rates,
    balanced_intensities,
    choose_two_distinct,
    ensure_jsonl,
    extract_choice_letter,
    parse_judge_json,
    utc_run_dir,
)


DEFAULT_TARGET_MODELS = ("claude-opus-4.6", "deepseek-v3.2", "gpt-5.2")
DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"


@dataclass
class RunConfig:
    sample_size: int = 1000
    seed: int = 42
    max_questions: int | None = None
    strict_filter: bool = True
    relaxed_min_correct: int = 2
    output_root: str = "outputs"
    target_models: Tuple[str, ...] = DEFAULT_TARGET_MODELS
    judge_model: str = DEFAULT_JUDGE_MODEL


class SycophancyPipeline:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.clients = build_clients_from_env()
        self.false_reasoning_cache: Dict[str, str] = {}
        self.false_reasoning_lock = threading.Lock()
        self.generated_reasoning_tag = re.compile(r"\[(?:generated\s+reasoning[^\]]*)\]", re.IGNORECASE)
        self.target_models = tuple(cfg.target_models)
        self.judge_model = cfg.judge_model

        for name in self.target_models + (self.judge_model,):
            if name not in self.clients:
                raise ValueError(f"Client not configured: {name}")

    def _chat(self, model_name: str, messages: List[dict], **kwargs) -> str:
        client = self.clients[model_name]
        return client.chat(messages, **kwargs)

    def _judge(self, mcq: MCQ, baseline_answer: str, pressure_text: str, response_text: str) -> Dict[str, str]:
        judge_messages = build_judge_messages(
            mcq=mcq,
            baseline_answer=baseline_answer,
            pressure_text=pressure_text,
            response_to_pressure=response_text,
        )
        judge_raw = self._chat(self.judge_model, judge_messages, temperature=0.0, max_output_tokens=300)
        parsed = parse_judge_json(judge_raw)
        parsed["judge_raw"] = judge_raw
        return parsed

    def _pick_target_wrong_letter(self, mcq: MCQ) -> str:
        options = [x for x in ("A", "B", "C", "D") if x != mcq.correct_letter]
        rng = random.Random(f"{self.cfg.seed}-{mcq.question_id}-wrong-target")
        return rng.choice(options)

    def _get_false_reasoning(self, mcq: MCQ) -> str:
        with self.false_reasoning_lock:
            if mcq.question_id in self.false_reasoning_cache:
                return self.false_reasoning_cache[mcq.question_id]

        target_wrong_letter = self._pick_target_wrong_letter(mcq)
        messages = build_false_reasoning_messages(mcq, target_wrong_letter=target_wrong_letter)
        reasoning = self._chat(self.judge_model, messages, temperature=0.3, max_output_tokens=200)
        reasoning = " ".join(reasoning.split())
        with self.false_reasoning_lock:
            self.false_reasoning_cache[mcq.question_id] = reasoning
        return reasoning

    def _render_pressure_prompt(self, mcq: MCQ, template: str) -> str:
        if "generated reasoning" not in template.lower():
            return template
        reasoning = self._get_false_reasoning(mcq)
        return self.generated_reasoning_tag.sub(reasoning, template).strip()

    def run(self) -> str:
        os.makedirs(self.cfg.output_root, exist_ok=True)
        out_dir = utc_run_dir(self.cfg.output_root)

        questions = load_mmlu_test(max_questions=self.cfg.max_questions, seed=self.cfg.seed)
        baseline_rows, kept_questions, baseline_answers = self._run_stage1(questions)

        sampled_questions = stratified_sample_by_domain(
            kept_questions,
            sample_size=self.cfg.sample_size,
            seed=self.cfg.seed,
        )
        sampled_qids = {q.question_id for q in sampled_questions}

        filtered_rows = [
            {
                "question_id": q.question_id,
                "subject": q.subject,
                "domain": q.domain,
                "question": q.question,
                "choices": list(q.choices),
                "correct_letter": q.correct_letter,
            }
            for q in sampled_questions
        ]

        pressure_rows = self._run_pressure_stages(sampled_questions, baseline_answers)

        ensure_jsonl(os.path.join(out_dir, "stage1_baseline.jsonl"), baseline_rows)
        ensure_jsonl(os.path.join(out_dir, "stage1_sampled_questions.jsonl"), filtered_rows)
        ensure_jsonl(os.path.join(out_dir, "pressure_results.jsonl"), pressure_rows)

        self._write_summaries(out_dir, pressure_rows)
        self._write_run_metadata(out_dir, questions, kept_questions, sampled_questions)

        # Keep only stage1 rows for sampled set in a separate file for convenience.
        sampled_stage1 = [row for row in baseline_rows if row["question_id"] in sampled_qids]
        ensure_jsonl(os.path.join(out_dir, "stage1_sampled_baseline.jsonl"), sampled_stage1)

        return out_dir

    def _run_stage1(self, questions: List[MCQ]) -> Tuple[List[dict], List[MCQ], Dict[str, Dict[str, str]]]:
        baseline_rows: List[dict] = []
        kept_questions: List[MCQ] = []
        baseline_answers: Dict[str, Dict[str, str]] = {}

        for mcq in tqdm(questions, desc="Stage 1 baseline"):
            row = {
                "question_id": mcq.question_id,
                "subject": mcq.subject,
                "domain": mcq.domain,
                "correct_letter": mcq.correct_letter,
            }
            n_correct = 0
            baseline_answers[mcq.question_id] = {}

            for model_name in self.target_models:
                try:
                    resp = self._chat(model_name, build_baseline_messages(mcq), temperature=0.0, max_tokens=128)
                    pred = extract_choice_letter(resp)
                    is_correct = pred == mcq.correct_letter
                    if is_correct:
                        n_correct += 1
                    baseline_answers[mcq.question_id][model_name] = pred or ""
                    row[f"{model_name}_raw"] = resp
                    row[f"{model_name}_pred"] = pred
                    row[f"{model_name}_correct"] = is_correct
                except Exception as exc:  # noqa: BLE001
                    baseline_answers[mcq.question_id][model_name] = ""
                    row[f"{model_name}_raw"] = f"[API_ERROR] {exc}"
                    row[f"{model_name}_pred"] = None
                    row[f"{model_name}_correct"] = False

            if self.cfg.strict_filter:
                keep = n_correct == len(self.target_models)
            else:
                keep = n_correct >= self.cfg.relaxed_min_correct

            row["num_models_correct"] = n_correct
            row["kept_for_pressure"] = keep
            baseline_rows.append(row)

            if keep:
                kept_questions.append(mcq)

        return baseline_rows, kept_questions, baseline_answers

    def _run_pressure_stages(self, questions: List[MCQ], baseline_answers: Dict[str, Dict[str, str]]) -> List[dict]:
        rows: List[dict] = []

        intensity_assignments: Dict[Tuple[str, str, str], str] = {}
        for model_idx, model_name in enumerate(self.target_models):
            for pressure_idx, pressure_type in enumerate(PRESSURE_TYPES):
                intensities = balanced_intensities(
                    n=len(questions),
                    seed=self.cfg.seed + model_idx * 100 + pressure_idx * 17,
                )
                for q_idx, mcq in enumerate(questions):
                    intensity_assignments[(mcq.question_id, model_name, pressure_type)] = intensities[q_idx]

        for mcq in tqdm(questions, desc="Stage 2/3 pressure"):
            for model_name in self.target_models:
                baseline_answer = baseline_answers.get(mcq.question_id, {}).get(model_name, "").strip().upper()
                if baseline_answer not in {"A", "B", "C", "D"}:
                    continue

                for pressure_type in PRESSURE_TYPES:
                    intensity = intensity_assignments[(mcq.question_id, model_name, pressure_type)]
                    options = PRESSURE_PROMPTS[pressure_type][intensity]
                    rng = random.Random(f"{self.cfg.seed}-{mcq.question_id}-{model_name}-{pressure_type}")
                    template_r1, template_r2 = choose_two_distinct(options, rng)

                    pressure_r1 = self._render_pressure_prompt(mcq, template_r1)
                    r1_messages = build_round1_messages(mcq, baseline_answer, pressure_r1)

                    try:
                        r1_response = self._chat(model_name, r1_messages, temperature=0.0, max_tokens=1024)
                        r1_judged = self._judge(mcq, baseline_answer, pressure_r1, r1_response)
                        r1_class = r1_judged["classification"]
                        r1_evidence = r1_judged.get("evidence", "")
                        r1_judge_raw = r1_judged.get("judge_raw", "")
                    except Exception as exc:  # noqa: BLE001
                        r1_response = f"[API_ERROR] {exc}"
                        r1_class = "ERROR"
                        r1_evidence = ""
                        r1_judge_raw = ""

                    row_r1 = {
                        "question_id": mcq.question_id,
                        "subject": mcq.subject,
                        "domain": mcq.domain,
                        "model": model_name,
                        "pressure_type": pressure_type,
                        "intensity": intensity,
                        "round": 1,
                        "correct_letter": mcq.correct_letter,
                        "baseline_answer": baseline_answer,
                        "pressure_text": pressure_r1,
                        "response": r1_response,
                        "classification": r1_class,
                        "evidence": r1_evidence,
                        "judge_raw": r1_judge_raw,
                    }
                    rows.append(row_r1)

                    if r1_class != "HOLDS":
                        continue

                    pressure_r2 = self._render_pressure_prompt(mcq, template_r2)
                    r2_messages = build_round2_messages(
                        mcq=mcq,
                        baseline_answer=baseline_answer,
                        pressure_round1=pressure_r1,
                        response_round1=r1_response,
                        pressure_round2=pressure_r2,
                    )

                    try:
                        r2_response = self._chat(model_name, r2_messages, temperature=0.0, max_tokens=1024)
                        r2_judged = self._judge(mcq, baseline_answer, pressure_r2, r2_response)
                        r2_class = r2_judged["classification"]
                        r2_evidence = r2_judged.get("evidence", "")
                        r2_judge_raw = r2_judged.get("judge_raw", "")
                    except Exception as exc:  # noqa: BLE001
                        r2_response = f"[API_ERROR] {exc}"
                        r2_class = "ERROR"
                        r2_evidence = ""
                        r2_judge_raw = ""

                    row_r2 = {
                        "question_id": mcq.question_id,
                        "subject": mcq.subject,
                        "domain": mcq.domain,
                        "model": model_name,
                        "pressure_type": pressure_type,
                        "intensity": intensity,
                        "round": 2,
                        "correct_letter": mcq.correct_letter,
                        "baseline_answer": baseline_answer,
                        "pressure_text": pressure_r2,
                        "response": r2_response,
                        "classification": r2_class,
                        "evidence": r2_evidence,
                        "judge_raw": r2_judge_raw,
                    }
                    rows.append(row_r2)

        return rows

    def _write_summaries(self, out_dir: str, pressure_rows: List[dict]) -> None:
        clean_rows = [r for r in pressure_rows if r.get("classification") in {"HOLDS", "CAVES", "HESITATES"}]

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

    def _write_run_metadata(
        self,
        out_dir: str,
        all_questions: List[MCQ],
        kept_questions: List[MCQ],
        sampled_questions: List[MCQ],
    ) -> None:
        meta = {
            "seed": self.cfg.seed,
            "sample_size_target": self.cfg.sample_size,
            "strict_filter": self.cfg.strict_filter,
            "relaxed_min_correct": self.cfg.relaxed_min_correct,
            "max_questions_input": self.cfg.max_questions,
            "num_questions_loaded": len(all_questions),
            "num_questions_kept_after_stage1": len(kept_questions),
            "num_questions_sampled": len(sampled_questions),
            "target_models": list(self.target_models),
            "judge_model": self.judge_model,
        }
        ensure_jsonl(os.path.join(out_dir, "run_metadata.jsonl"), [meta])
