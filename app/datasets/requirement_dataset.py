from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LABELS = (
    "CORE",
    "IMPORTANT",
    "PREFERRED",
    "CONTEXTUAL",
    "NOT_REQUIREMENT",
)

GROUP_OPERATORS = (
    "ANY_OF",
    "ALL_OF",
)


class DatasetValidationError(ValueError):
    """Raised when annotated examples would produce an unsound dataset."""


def _required_text(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _optional_text(payload: dict[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetValidationError(f"{field_name} must be a string or null")
    return value.strip() or None


def _optional_bool(payload: dict[str, Any], field_name: str) -> bool:
    value = payload.get(field_name, False)
    if not isinstance(value, bool):
        raise DatasetValidationError(f"{field_name} must be a boolean")
    return value


def _optional_int(payload: dict[str, Any], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(f"{field_name} must be an integer or null")
    return value


def _optional_text_list(
    payload: dict[str, Any],
    field_name: str,
) -> tuple[str, ...]:
    value = payload.get(field_name)
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise DatasetValidationError(f"{field_name} must be an array or null")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetValidationError(
                f"{field_name} must contain only non-empty strings"
            )
        member = item.strip()
        key = _normalized_text(member)
        if key not in seen:
            cleaned.append(member)
            seen.add(key)
    return tuple(cleaned)


@dataclass(frozen=True)
class RequirementExample:
    job_id: str
    job_title: str
    job_description: str
    requirement: str
    label: str
    evidence_span: str
    annotation_source: str = "human"
    review_notes: str | None = None
    role_family: str | None = None
    source: str | None = None
    source_url: str | None = None
    is_synthetic: bool = False
    synthetic_type: str | None = None
    source_example_id: str | None = None
    source_label: str | None = None
    original_evidence_span: str | None = None
    generation_seed: int | None = None
    requirement_group: str | None = None
    group_operator: str | None = None
    group_members: tuple[str, ...] = ()
    group_label: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RequirementExample:
        label_value = _required_text(payload, "label").upper()
        if label_value not in LABELS:
            allowed = ", ".join(LABELS)
            raise DatasetValidationError(
                f"label must be one of {allowed}; received {label_value!r}"
            )

        raw_group_operator = _optional_text(payload, "group_operator")
        raw_group_label = _optional_text(payload, "group_label")
        example = cls(
            job_id=_required_text(payload, "job_id"),
            job_title=_required_text(payload, "job_title"),
            job_description=_required_text(payload, "job_description"),
            requirement=_required_text(payload, "requirement"),
            label=label_value,
            evidence_span=_required_text(payload, "evidence_span"),
            annotation_source=str(payload.get("annotation_source") or "human").strip(),
            review_notes=_optional_text(payload, "review_notes"),
            role_family=_optional_text(payload, "role_family"),
            source=_optional_text(payload, "source"),
            source_url=_optional_text(payload, "source_url"),
            is_synthetic=_optional_bool(payload, "is_synthetic"),
            synthetic_type=_optional_text(payload, "synthetic_type"),
            source_example_id=_optional_text(payload, "source_example_id"),
            source_label=_optional_text(payload, "source_label"),
            original_evidence_span=_optional_text(
                payload,
                "original_evidence_span",
            ),
            generation_seed=_optional_int(payload, "generation_seed"),
            requirement_group=_optional_text(payload, "requirement_group"),
            group_operator=(
                raw_group_operator.upper() if raw_group_operator else None
            ),
            group_members=_optional_text_list(payload, "group_members"),
            group_label=raw_group_label.upper() if raw_group_label else None,
        )
        example.validate()
        return example

    @property
    def example_id(self) -> str:
        identity = json.dumps(
            {
                "job_id": self.job_id,
                "requirement": _normalized_text(self.requirement),
                "label": self.label,
                "evidence_span": _normalized_text(self.evidence_span),
                "requirement_group": (
                    _normalized_text(self.requirement_group)
                    if self.requirement_group
                    else None
                ),
                "group_operator": self.group_operator,
                "group_members": [
                    _normalized_text(member) for member in self.group_members
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return _stable_hash(identity)[:20]

    @property
    def conflict_key(self) -> tuple[str, str, str]:
        return (
            self.job_id,
            _normalized_text(self.requirement),
            _normalized_text(self.evidence_span),
        )

    def validate(self) -> None:
        if self.evidence_span not in self.job_description:
            raise DatasetValidationError(
                "evidence_span must occur verbatim in job_description "
                f"for job {self.job_id!r}"
            )
        if len(self.requirement) > 200:
            raise DatasetValidationError(
                f"requirement is implausibly long for job {self.job_id!r}"
            )
        if self.is_synthetic and not self.synthetic_type:
            raise DatasetValidationError(
                f"synthetic_type is required for synthetic job {self.job_id!r}"
            )
        if self.is_synthetic and not self.source_example_id:
            raise DatasetValidationError(
                f"source_example_id is required for synthetic job {self.job_id!r}"
            )
        group_values_present = any(
            (
                self.requirement_group,
                self.group_operator,
                self.group_members,
                self.group_label,
            )
        )
        if not group_values_present:
            return
        if not self.requirement_group:
            raise DatasetValidationError(
                "requirement_group is required when group metadata is present"
            )
        if self.group_operator not in GROUP_OPERATORS:
            allowed = ", ".join(GROUP_OPERATORS)
            raise DatasetValidationError(
                f"group_operator must be one of {allowed} for grouped requirements"
            )
        if len(self.group_members) < 2:
            raise DatasetValidationError(
                "group_members must contain at least two distinct alternatives"
            )
        if _normalized_text(self.requirement) not in {
            _normalized_text(member) for member in self.group_members
        }:
            raise DatasetValidationError(
                "requirement must be one of group_members"
            )
        if self.group_label not in LABELS:
            allowed = ", ".join(LABELS)
            raise DatasetValidationError(
                f"group_label must be one of {allowed} for grouped requirements"
            )
        if self.group_label != self.label:
            raise DatasetValidationError(
                "group_label must equal label because the label applies to the group"
            )

    def to_record(
        self,
        *,
        variant: str,
        job_description: str | None = None,
        noise_types: Iterable[str] = (),
        variant_index: int = 0,
    ) -> dict[str, Any]:
        description = job_description or self.job_description
        if self.evidence_span not in description:
            raise DatasetValidationError(
                f"variant removed evidence for example {self.example_id}"
            )

        noise = list(noise_types)
        variant_identity = f"{self.example_id}:{variant}:{variant_index}"
        return {
            "example_id": _stable_hash(variant_identity)[:20],
            "parent_example_id": self.example_id,
            "job_id": self.job_id,
            "job_title": self.job_title,
            "job_description": description,
            "requirement": self.requirement,
            "label": self.label,
            "evidence_span": self.evidence_span,
            "annotation_source": self.annotation_source,
            "review_notes": self.review_notes,
            "role_family": self.role_family,
            "source": self.source,
            "source_url": self.source_url,
            "is_synthetic": self.is_synthetic,
            "synthetic_type": self.synthetic_type,
            "source_example_id": self.source_example_id,
            "source_label": self.source_label,
            "original_evidence_span": self.original_evidence_span,
            "generation_seed": self.generation_seed,
            "requirement_group": self.requirement_group,
            "group_operator": self.group_operator,
            "group_members": list(self.group_members),
            "group_label": self.group_label,
            "variant": variant,
            "is_noisy": bool(noise),
            "noise_types": noise,
        }


@dataclass(frozen=True)
class DatasetBuildResult:
    output_dir: Path
    manifest: dict[str, Any]


def load_annotations(paths: Iterable[str | Path]) -> list[RequirementExample]:
    examples: list[RequirementExample] = []
    seen_ids: set[str] = set()
    conflict_labels: dict[tuple[str, str, str], str] = {}

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise DatasetValidationError(f"annotation file does not exist: {path}")

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"invalid JSON in {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise DatasetValidationError(
                    f"annotation at {path}:{line_number} must be a JSON object"
                )

            try:
                example = RequirementExample.from_dict(payload)
            except DatasetValidationError as exc:
                raise DatasetValidationError(f"{path}:{line_number}: {exc}") from exc

            previous_label = conflict_labels.get(example.conflict_key)
            if previous_label is not None and previous_label != example.label:
                raise DatasetValidationError(
                    f"conflicting labels at {path}:{line_number}: "
                    f"{previous_label} and {example.label}"
                )
            conflict_labels[example.conflict_key] = example.label

            if example.example_id in seen_ids:
                continue
            seen_ids.add(example.example_id)
            examples.append(example)

    if not examples:
        raise DatasetValidationError("no annotated examples were found")
    return examples


class RequirementDatasetBuilder:
    def __init__(
        self,
        *,
        seed: int = 42,
        evaluation_ratio: float = 0.2,
        noise_ratio: float = 0.3,
        minimum_per_label: int = 1,
        maximum_synthetic_ratio: float = 0.4,
    ):
        if not 0 <= evaluation_ratio < 1:
            raise ValueError("evaluation_ratio must be at least 0 and less than 1")
        if not 0 <= noise_ratio < 1:
            raise ValueError("noise_ratio must be at least 0 and less than 1")
        if minimum_per_label < 0:
            raise ValueError("minimum_per_label cannot be negative")
        if not 0 <= maximum_synthetic_ratio <= 1:
            raise ValueError("maximum_synthetic_ratio must be between 0 and 1")

        self.seed = seed
        self.evaluation_ratio = evaluation_ratio
        self.noise_ratio = noise_ratio
        self.minimum_per_label = minimum_per_label
        self.maximum_synthetic_ratio = maximum_synthetic_ratio

    def build(
        self,
        examples: Iterable[RequirementExample],
        output_dir: str | Path,
    ) -> DatasetBuildResult:
        source_examples = list(examples)
        self._validate_corpus(source_examples)
        train_examples, evaluation_examples = self._split_by_job(source_examples)

        train_records = [
            example.to_record(variant="real_world")
            for example in train_examples
        ]
        train_records.extend(self._training_noise(train_examples))
        train_records.sort(key=lambda record: record["example_id"])

        evaluation_real_examples = [
            example for example in evaluation_examples if not example.is_synthetic
        ]
        evaluation_synthetic_examples = [
            example for example in evaluation_examples if example.is_synthetic
        ]
        evaluation_real_world = [
            example.to_record(variant="real_world")
            for example in evaluation_real_examples
        ]
        evaluation_clean = [
            example.to_record(
                variant="clean_context",
                job_description=example.evidence_span,
            )
            for example in evaluation_real_examples
        ]
        evaluation_noisy = [
            self._noisy_record(example, variant_index=1)
            for example in evaluation_real_examples
        ]
        evaluation_synthetic = [
            example.to_record(variant="synthetic_counterfactual")
            for example in evaluation_synthetic_examples
        ]

        for records in (
            evaluation_real_world,
            evaluation_clean,
            evaluation_noisy,
            evaluation_synthetic,
        ):
            records.sort(key=lambda record: record["example_id"])

        output_path = Path(output_dir)
        evaluation_path = output_path / "evaluation"
        evaluation_path.mkdir(parents=True, exist_ok=True)

        self._write_jsonl(output_path / "train.jsonl", train_records)
        self._write_jsonl(evaluation_path / "clean.jsonl", evaluation_clean)
        self._write_jsonl(
            evaluation_path / "real_world.jsonl",
            evaluation_real_world,
        )
        self._write_jsonl(evaluation_path / "noisy.jsonl", evaluation_noisy)
        self._write_jsonl(
            evaluation_path / "synthetic.jsonl",
            evaluation_synthetic,
        )

        manifest = self._manifest(
            source_examples=source_examples,
            train_examples=train_examples,
            evaluation_examples=evaluation_examples,
            train_records=train_records,
            evaluation_clean=evaluation_clean,
            evaluation_real_world=evaluation_real_world,
            evaluation_noisy=evaluation_noisy,
            evaluation_synthetic=evaluation_synthetic,
        )
        self._write_json(output_path / "manifest.json", manifest)
        return DatasetBuildResult(output_dir=output_path, manifest=manifest)

    def _validate_corpus(self, examples: list[RequirementExample]) -> None:
        if not examples:
            raise DatasetValidationError("cannot build a dataset without examples")

        conflict_labels: dict[tuple[str, str, str], str] = {}
        for example in examples:
            example.validate()
            previous_label = conflict_labels.get(example.conflict_key)
            if previous_label is not None and previous_label != example.label:
                raise DatasetValidationError(
                    f"conflicting labels for example {example.example_id}: "
                    f"{previous_label} and {example.label}"
                )
            conflict_labels[example.conflict_key] = example.label

        counts = Counter(example.label for example in examples)
        insufficient = {
            label: counts.get(label, 0)
            for label in LABELS
            if counts.get(label, 0) < self.minimum_per_label
        }
        if insufficient:
            details = ", ".join(
                f"{label}={count}"
                for label, count in insufficient.items()
            )
            raise DatasetValidationError(
                "dataset does not meet minimum examples per label: " + details
            )

        synthetic_ratio = sum(example.is_synthetic for example in examples) / len(
            examples
        )
        if synthetic_ratio > self.maximum_synthetic_ratio:
            raise DatasetValidationError(
                f"synthetic example ratio {synthetic_ratio:.3f} exceeds maximum "
                f"{self.maximum_synthetic_ratio:.3f}"
            )

    def _split_by_job(
        self,
        examples: list[RequirementExample],
    ) -> tuple[list[RequirementExample], list[RequirementExample]]:
        by_job: dict[str, list[RequirementExample]] = defaultdict(list)
        for example in examples:
            by_job[example.job_id].append(example)

        job_ids = sorted(
            by_job,
            key=lambda job_id: _stable_hash(f"{self.seed}:{job_id}"),
        )
        if self.evaluation_ratio == 0 or len(job_ids) < 2:
            evaluation_job_ids: set[str] = set()
        else:
            evaluation_jobs = max(1, round(len(job_ids) * self.evaluation_ratio))
            evaluation_jobs = min(evaluation_jobs, len(job_ids) - 1)
            remaining_label_counts = Counter(
                example.label
                for example in examples
            )
            job_label_counts = {
                job_id: Counter(example.label for example in by_job[job_id])
                for job_id in job_ids
            }
            evaluation_label_counts: Counter[str] = Counter()
            evaluation_job_ids = set()

            while len(evaluation_job_ids) < evaluation_jobs:
                eligible = [
                    job_id
                    for job_id in job_ids
                    if job_id not in evaluation_job_ids
                    and all(
                        remaining_label_counts[label] - count
                        >= self.minimum_per_label
                        for label, count in job_label_counts[job_id].items()
                    )
                ]
                if not eligible:
                    break

                selected = min(
                    eligible,
                    key=lambda job_id: (
                        -sum(
                            evaluation_label_counts[label] == 0
                            for label in job_label_counts[job_id]
                        ),
                        _stable_hash(f"{self.seed}:{job_id}"),
                    ),
                )
                evaluation_job_ids.add(selected)
                remaining_label_counts.subtract(job_label_counts[selected])
                evaluation_label_counts.update(job_label_counts[selected])

        train = [
            example
            for example in examples
            if example.job_id not in evaluation_job_ids
        ]
        evaluation = [
            example
            for example in examples
            if example.job_id in evaluation_job_ids
        ]
        return train, evaluation

    def _training_noise(
        self,
        examples: list[RequirementExample],
    ) -> list[dict[str, Any]]:
        if not examples or self.noise_ratio == 0:
            return []

        requested = round(len(examples) * self.noise_ratio / (1 - self.noise_ratio))
        if requested == 0:
            requested = 1

        ordered = sorted(
            examples,
            key=lambda example: _stable_hash(
                f"{self.seed}:training-noise:{example.example_id}"
            ),
        )
        return [
            self._noisy_record(
                ordered[index % len(ordered)],
                variant_index=(index // len(ordered)) + 1,
            )
            for index in range(requested)
        ]

    def _noisy_record(
        self,
        example: RequirementExample,
        *,
        variant_index: int,
    ) -> dict[str, Any]:
        rng = random.Random(
            f"{self.seed}:{example.example_id}:{variant_index}"
        )
        candidates = self._noise_blocks(example, rng)
        block_count = rng.randint(1, min(3, len(candidates)))
        selected = rng.sample(candidates, block_count)

        prefix: list[str] = []
        suffix: list[str] = []
        noise_types: list[str] = []
        for noise_type, block in selected:
            noise_types.append(noise_type)
            (prefix if rng.choice((True, False)) else suffix).append(block)

        description_parts = [*prefix, example.job_description, *suffix]
        description = "\n\n".join(description_parts)
        return example.to_record(
            variant="noisy",
            job_description=description,
            noise_types=noise_types,
            variant_index=variant_index,
        )

    @staticmethod
    def _noise_blocks(
        example: RequirementExample,
        rng: random.Random,
    ) -> list[tuple[str, str]]:
        unrelated_technologies = [
            technology
            for technology in ("COBOL", "Ruby on Rails", "MATLAB", "Unity", "SAP ABAP")
            if technology.casefold() not in example.requirement.casefold()
        ]
        unrelated = rng.choice(unrelated_technologies)
        return [
            (
                "recruiter_boilerplate",
                "Recruitment agencies: please do not contact the hiring team directly.",
            ),
            (
                "salary_noise",
                "Compensation may include salary, bonus, pension, and location-based benefits.",
            ),
            (
                "duplicated_heading",
                "ABOUT THE ROLE\nABOUT THE ROLE",
            ),
            (
                "irrelevant_technology_mention",
                f"Elsewhere in the organisation, legacy teams may still support {unrelated}.",
            ),
            (
                "marketing_copy",
                "Join our award-winning journey and help us shape an exciting future together.",
            ),
            (
                "odd_formatting",
                "*** VACANCY DETAILS ***\n\n• • •\nReference: INTERNAL-CAREERS-PAGE",
            ),
        ]

    def _manifest(
        self,
        *,
        source_examples: list[RequirementExample],
        train_examples: list[RequirementExample],
        evaluation_examples: list[RequirementExample],
        train_records: list[dict[str, Any]],
        evaluation_clean: list[dict[str, Any]],
        evaluation_real_world: list[dict[str, Any]],
        evaluation_noisy: list[dict[str, Any]],
        evaluation_synthetic: list[dict[str, Any]],
    ) -> dict[str, Any]:
        train_job_ids = {example.job_id for example in train_examples}
        evaluation_job_ids = {example.job_id for example in evaluation_examples}
        overlap = sorted(train_job_ids & evaluation_job_ids)
        if overlap:
            raise DatasetValidationError(
                "job-level split leakage detected: " + ", ".join(overlap)
            )

        return {
            "schema_version": 1,
            "seed": self.seed,
            "evaluation_ratio": self.evaluation_ratio,
            "requested_training_noise_ratio": self.noise_ratio,
            "maximum_synthetic_ratio": self.maximum_synthetic_ratio,
            "labels": list(LABELS),
            "source_examples": len(source_examples),
            "source_jobs": len({example.job_id for example in source_examples}),
            "train_jobs": len(train_job_ids),
            "evaluation_jobs": len(evaluation_job_ids),
            "job_split_overlap": overlap,
            "counts": {
                "train_total": len(train_records),
                "train_real_world": sum(
                    record["variant"] == "real_world"
                    for record in train_records
                ),
                "train_noisy": sum(record["is_noisy"] for record in train_records),
                "evaluation_clean": len(evaluation_clean),
                "evaluation_real_world": len(evaluation_real_world),
                "evaluation_noisy": len(evaluation_noisy),
                "evaluation_synthetic": len(evaluation_synthetic),
            },
            "label_distribution": {
                "source": self._label_counts(source_examples),
                "train": self._record_label_counts(train_records),
                "evaluation": self._record_label_counts(evaluation_real_world),
            },
            "role_distribution": {
                "source": self._role_counts(source_examples),
                "train": self._record_role_counts(train_records),
                "evaluation": self._record_role_counts(evaluation_real_world),
            },
            "source_distribution": {
                "source": self._source_counts(source_examples),
                "train": self._record_source_counts(train_records),
                "evaluation": self._record_source_counts(evaluation_real_world),
            },
            "provenance": {
                "source_real": sum(
                    not example.is_synthetic for example in source_examples
                ),
                "source_synthetic": sum(
                    example.is_synthetic for example in source_examples
                ),
                "source_synthetic_ratio": (
                    sum(example.is_synthetic for example in source_examples)
                    / len(source_examples)
                ),
                "train_synthetic": sum(
                    bool(record.get("is_synthetic")) for record in train_records
                ),
                "evaluation_synthetic": sum(
                    bool(record.get("is_synthetic"))
                    for record in evaluation_synthetic
                ),
                "source_with_review_notes": sum(
                    bool(example.review_notes) for example in source_examples
                ),
            },
        }

    @staticmethod
    def _label_counts(examples: Iterable[RequirementExample]) -> dict[str, int]:
        counts = Counter(example.label for example in examples)
        return {label: counts.get(label, 0) for label in LABELS}

    @staticmethod
    def _record_label_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(record["label"] for record in records)
        return {label: counts.get(label, 0) for label in LABELS}

    @staticmethod
    def _role_counts(examples: Iterable[RequirementExample]) -> dict[str, int]:
        counts = Counter(example.role_family or "unassigned" for example in examples)
        return dict(sorted(counts.items()))

    @staticmethod
    def _record_role_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(record.get("role_family") or "unassigned" for record in records)
        return dict(sorted(counts.items()))

    @staticmethod
    def _source_counts(examples: Iterable[RequirementExample]) -> dict[str, int]:
        counts = Counter(example.source or "unassigned" for example in examples)
        return dict(sorted(counts.items()))

    @staticmethod
    def _record_source_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(record.get("source") or "unassigned" for record in records)
        return dict(sorted(counts.items()))

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
        content = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
