from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.datasets.requirement_dataset import LABELS, RequirementExample


@dataclass(frozen=True)
class CoverageReportSummary:
    examples: int
    unique_jobs: int
    blockers: int
    warnings: int
    ready_for_training: bool
    output_path: Path


def generate_coverage_report(
    examples: Iterable[RequirementExample],
    output_path: str | Path,
    *,
    required_role_families: Iterable[str],
    minimum_per_label: int = 20,
    minimum_per_role: int = 5,
    minimum_unique_jobs: int = 30,
    minimum_sources: int = 3,
    maximum_label_share: float = 0.5,
) -> CoverageReportSummary:
    reviewed = list(examples)
    required_roles = tuple(dict.fromkeys(required_role_families))
    if not reviewed:
        raise ValueError("coverage report requires at least one reviewed example")
    if minimum_per_label < 0 or minimum_per_role < 0:
        raise ValueError("minimum coverage counts cannot be negative")
    if minimum_unique_jobs < 1 or minimum_sources < 1:
        raise ValueError("minimum_unique_jobs and minimum_sources must be positive")
    if not 0 < maximum_label_share <= 1:
        raise ValueError("maximum_label_share must be greater than 0 and at most 1")

    label_counts = Counter(example.label for example in reviewed)
    role_counts = Counter(example.role_family or "unassigned" for example in reviewed)
    source_counts = Counter(example.source or "unassigned" for example in reviewed)
    group_operator_counts = Counter(
        example.group_operator or "ATOMIC" for example in reviewed
    )
    unique_jobs = len({example.job_id for example in reviewed})
    dominant_label, dominant_count = label_counts.most_common(1)[0]
    dominant_share = dominant_count / len(reviewed)
    gaps: list[dict[str, object]] = []

    for label in LABELS:
        count = label_counts.get(label, 0)
        if count < minimum_per_label:
            gaps.append(
                {
                    "severity": "blocker",
                    "dimension": "label",
                    "name": label,
                    "observed": count,
                    "target": minimum_per_label,
                    "needed": minimum_per_label - count,
                }
            )

    for role_family in required_roles:
        count = role_counts.get(role_family, 0)
        if count < minimum_per_role:
            gaps.append(
                {
                    "severity": "blocker",
                    "dimension": "role_family",
                    "name": role_family,
                    "observed": count,
                    "target": minimum_per_role,
                    "needed": minimum_per_role - count,
                }
            )

    if unique_jobs < minimum_unique_jobs:
        gaps.append(
            {
                "severity": "blocker",
                "dimension": "unique_jobs",
                "name": "unique_jobs",
                "observed": unique_jobs,
                "target": minimum_unique_jobs,
                "needed": minimum_unique_jobs - unique_jobs,
            }
        )
    represented_sources = sum(
        source != "unassigned" and count > 0
        for source, count in source_counts.items()
    )
    if represented_sources < minimum_sources:
        gaps.append(
            {
                "severity": "blocker",
                "dimension": "sources",
                "name": "distinct_sources",
                "observed": represented_sources,
                "target": minimum_sources,
                "needed": minimum_sources - represented_sources,
            }
        )
    if dominant_share > maximum_label_share:
        gaps.append(
            {
                "severity": "blocker",
                "dimension": "label_balance",
                "name": dominant_label,
                "observed": round(dominant_share, 4),
                "target_maximum": maximum_label_share,
            }
        )
    if role_counts.get("unassigned", 0):
        gaps.append(
            {
                "severity": "warning",
                "dimension": "role_family",
                "name": "unassigned",
                "observed": role_counts["unassigned"],
                "target": 0,
            }
        )

    severity_counts = Counter(str(gap["severity"]) for gap in gaps)
    payload = {
        "ready_for_training": severity_counts["blocker"] == 0,
        "summary": {
            "examples": len(reviewed),
            "unique_jobs": unique_jobs,
            "distinct_sources": represented_sources,
            "real_examples": sum(not example.is_synthetic for example in reviewed),
            "synthetic_examples": sum(example.is_synthetic for example in reviewed),
            "grouped_examples": sum(
                example.group_operator is not None for example in reviewed
            ),
            "blockers": severity_counts["blocker"],
            "warnings": severity_counts["warning"],
        },
        "targets": {
            "minimum_per_label": minimum_per_label,
            "minimum_per_role": minimum_per_role,
            "minimum_unique_jobs": minimum_unique_jobs,
            "minimum_sources": minimum_sources,
            "maximum_label_share": maximum_label_share,
            "required_role_families": list(required_roles),
        },
        "distribution": {
            "labels": {label: label_counts.get(label, 0) for label in LABELS},
            "role_families": dict(sorted(role_counts.items())),
            "sources": dict(sorted(source_counts.items())),
            "requirement_structures": dict(sorted(group_operator_counts.items())),
        },
        "gaps": gaps,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return CoverageReportSummary(
        examples=len(reviewed),
        unique_jobs=unique_jobs,
        blockers=severity_counts["blocker"],
        warnings=severity_counts["warning"],
        ready_for_training=severity_counts["blocker"] == 0,
        output_path=destination,
    )
