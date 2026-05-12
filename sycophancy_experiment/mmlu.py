from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List

from datasets import get_dataset_config_names, load_dataset

from sycophancy_experiment.prompts import MCQ


SUBJECT_TO_DOMAIN: Dict[str, str] = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "social_sciences",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "social_sciences",
    "marketing": "social_sciences",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "other",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "stem",
    "world_religions": "humanities",
}


ANSWER_LETTERS = ("A", "B", "C", "D")


def _get_domain(subject: str) -> str:
    return SUBJECT_TO_DOMAIN.get(subject, "other")


def _load_first_available_split(subject: str):
    # Some cached/local variants expose only train; prefer test when available.
    split_order = ("test", "validation", "dev", "train")
    last_err: Exception | None = None
    for split_name in split_order:
        try:
            split = load_dataset("cais/mmlu", subject, split=split_name)
            if len(split) > 0:
                return split, split_name
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise ValueError(f"No usable split found for subject={subject}. Last error: {last_err}")


def load_mmlu_test(max_questions: int | None = None, seed: int = 42) -> List[MCQ]:
    subjects = [s for s in get_dataset_config_names("cais/mmlu") if s in SUBJECT_TO_DOMAIN]
    if not subjects:
        subjects = sorted(SUBJECT_TO_DOMAIN.keys())
    subjects.sort()

    all_rows: List[MCQ] = []
    for subject in subjects:
        try:
            split, split_name = _load_first_available_split(subject)
        except Exception:
            # Skip unavailable configs in partially cached environments.
            continue
        for idx, row in enumerate(split):
            answer_index = int(row["answer"])
            all_rows.append(
                MCQ(
                    question_id=f"{subject}-{split_name}-{idx}",
                    subject=subject,
                    domain=_get_domain(subject),
                    question=row["question"],
                    choices=(
                        row["choices"][0],
                        row["choices"][1],
                        row["choices"][2],
                        row["choices"][3],
                    ),
                    correct_letter=ANSWER_LETTERS[answer_index],
                )
            )

    if not all_rows:
        raise ValueError(
            "Could not load any MMLU rows. Check network/cache access for `cais/mmlu` or dataset permissions."
        )

    if max_questions is None or max_questions >= len(all_rows):
        return all_rows

    rng = random.Random(seed)
    return rng.sample(all_rows, max_questions)


def stratified_sample_by_domain(rows: Iterable[MCQ], sample_size: int, seed: int = 42) -> List[MCQ]:
    rows = list(rows)
    if sample_size >= len(rows):
        return rows

    rng = random.Random(seed)
    by_domain: Dict[str, List[MCQ]] = defaultdict(list)
    for row in rows:
        by_domain[row.domain].append(row)

    domains = sorted(by_domain.keys())
    per_domain = sample_size // len(domains)
    remainder = sample_size % len(domains)

    sampled: List[MCQ] = []
    for i, domain in enumerate(domains):
        domain_rows = by_domain[domain]
        rng.shuffle(domain_rows)
        take = per_domain + (1 if i < remainder else 0)
        sampled.extend(domain_rows[: min(take, len(domain_rows))])

    if len(sampled) < sample_size:
        remaining_pool = [x for x in rows if x.question_id not in {y.question_id for y in sampled}]
        rng.shuffle(remaining_pool)
        sampled.extend(remaining_pool[: sample_size - len(sampled)])

    rng.shuffle(sampled)
    return sampled
