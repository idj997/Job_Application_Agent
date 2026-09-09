from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.datasets.annotation_candidates import write_candidate_records
from app.datasets.requirement_dataset import RequirementExample


SOURCE_LABELS = ("CORE", "IMPORTANT", "PREFERRED")

NEGATION_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "not_required",
        "Previous experience with {requirement} is not required for this role.",
    ),
    (
        "training_provided",
        "You do not need prior experience with {requirement}; training will be provided.",
    ),
    (
        "not_prerequisite",
        "Knowledge of {requirement} is not a prerequisite for this position.",
    ),
    (
        "without_experience",
        "Candidates without {requirement} experience are encouraged to apply.",
    ),
    (
        "not_necessary",
        "No previous {requirement} experience is necessary.",
    ),
    (
        "not_needed",
        "Prior knowledge of {requirement} is not needed to succeed in this role.",
    ),
)


@dataclass(frozen=True)
class CounterfactualGenerationSummary:
    source_examples: int
    eligible_examples: int
    grouped_examples_skipped: int
    candidates: int
    output_path: Path


def generate_counterfactual_candidates(
    examples: Iterable[RequirementExample],
    output_path: str | Path,
    *,
    variants_per_example: int = 1,
    seed: int = 42,
    source_labels: Iterable[str] = SOURCE_LABELS,
) -> CounterfactualGenerationSummary:
    if not 1 <= variants_per_example <= len(NEGATION_TEMPLATES):
        raise ValueError(
            f"variants_per_example must be between 1 and {len(NEGATION_TEMPLATES)}"
        )

    reviewed_examples = list(examples)
    allowed_labels = {label.upper() for label in source_labels}
    unknown_labels = allowed_labels - set(SOURCE_LABELS)
    if unknown_labels:
        raise ValueError(
            "counterfactual source labels must be positive requirement labels: "
            + ", ".join(sorted(unknown_labels))
        )

    eligible = [
        example
        for example in reviewed_examples
        if example.label in allowed_labels
        and not example.is_synthetic
        and example.group_operator is None
    ]
    grouped_examples_skipped = sum(
        example.label in allowed_labels
        and not example.is_synthetic
        and example.group_operator is not None
        for example in reviewed_examples
    )
    records: list[dict[str, object]] = []
    for example in eligible:
        rng = random.Random(f"{seed}:{example.example_id}")
        templates = rng.sample(list(NEGATION_TEMPLATES), variants_per_example)
        for template_index, (template_name, template) in enumerate(templates, start=1):
            evidence = template.format(requirement=example.requirement)
            identity = (
                f"{example.example_id}:{template_name}:{template_index}:{seed}"
            )
            records.append(
                {
                    "candidate_id": hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()[:20],
                    "annotation_status": "needs_review",
                    "reviewed_label": None,
                    "review_notes": "",
                    "job_id": example.job_id,
                    "job_title": example.job_title,
                    # Evidence-only context prevents a negated sentence from
                    # contradicting other positive mentions in the source JD.
                    "job_description": evidence,
                    "requirement": example.requirement,
                    "proposed_label": "NOT_REQUIREMENT",
                    "evidence_span": evidence,
                    "proposal_confidence": 1.0,
                    "proposal_rule": "counterfactual_explicit_negation",
                    "annotation_source": "counterfactual_candidate",
                    "role_family": example.role_family,
                    "source": example.source,
                    "source_url": example.source_url,
                    "is_synthetic": True,
                    "synthetic_type": f"explicit_negation:{template_name}",
                    "source_example_id": example.example_id,
                    "source_label": example.label,
                    "original_evidence_span": example.evidence_span,
                    "generation_seed": seed,
                }
            )

    records.sort(key=lambda record: str(record["candidate_id"]))
    destination = write_candidate_records(records, output_path)
    return CounterfactualGenerationSummary(
        source_examples=len(reviewed_examples),
        eligible_examples=len(eligible),
        grouped_examples_skipped=grouped_examples_skipped,
        candidates=len(records),
        output_path=destination,
    )
