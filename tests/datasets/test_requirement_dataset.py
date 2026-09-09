import json
from pathlib import Path

import pytest

from app.datasets.requirement_dataset import (
    LABELS,
    DatasetValidationError,
    RequirementDatasetBuilder,
    RequirementExample,
    load_annotations,
)


EVIDENCE = {
    "CORE": ("SQL", "Strong SQL experience is essential."),
    "IMPORTANT": ("PySpark", "You will build production pipelines using PySpark."),
    "PREFERRED": ("Airflow", "Experience with Airflow would be advantageous."),
    "CONTEXTUAL": ("Scala", "Our adjacent platform team currently uses Scala."),
    "NOT_REQUIREMENT": ("Kubernetes", "Previous Kubernetes experience is not required."),
}


def make_examples(jobs_per_label: int = 3) -> list[RequirementExample]:
    examples = []
    for label in LABELS:
        requirement, evidence = EVIDENCE[label]
        for index in range(jobs_per_label):
            examples.append(
                RequirementExample(
                    job_id=f"{label.lower()}-{index}",
                    job_title="Data Engineer",
                    job_description=(
                        "We are hiring for a growing data team.\n\n"
                        f"{evidence}\n\n"
                        "The role works closely with product and engineering."
                    ),
                    requirement=requirement,
                    label=label,
                    evidence_span=evidence,
                    review_notes=(
                        "Reviewer considered surrounding responsibilities."
                        if index == 0
                        else None
                    ),
                    role_family=f"role-{index % 2}",
                    source=f"source-{index % 2}",
                )
            )
    return examples


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_annotation_loader_normalizes_labels_and_removes_exact_duplicates(tmp_path):
    path = tmp_path / "requirements.jsonl"
    payload = {
        "job_id": "job-1",
        "job_title": "Data Engineer",
        "job_description": "Strong SQL experience is essential.",
        "requirement": "SQL",
        "label": "core",
        "evidence_span": "Strong SQL experience is essential.",
    }
    path.write_text(
        json.dumps(payload) + "\n" + json.dumps(payload) + "\n",
        encoding="utf-8",
    )

    examples = load_annotations([path])

    assert len(examples) == 1
    assert examples[0].label == "CORE"


def test_grouped_requirement_metadata_is_validated_and_preserved(tmp_path):
    path = tmp_path / "requirements.jsonl"
    payload = {
        "job_id": "job-group",
        "job_title": "Data Engineer",
        "job_description": "Advanced knowledge of Java, Python, or Scala.",
        "requirement": "Java",
        "label": "IMPORTANT",
        "evidence_span": "Advanced knowledge of Java, Python, or Scala.",
        "requirement_group": "programming language",
        "group_operator": "any_of",
        "group_members": ["Java", "Python", "Scala"],
        "group_label": "important",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    example = load_annotations([path])[0]
    record = example.to_record(variant="real_world")

    assert example.group_operator == "ANY_OF"
    assert example.group_label == "IMPORTANT"
    assert record["group_members"] == ["Java", "Python", "Scala"]


def test_grouped_requirement_rejects_member_outside_group():
    with pytest.raises(DatasetValidationError, match="one of group_members"):
        RequirementExample.from_dict(
            {
                "job_id": "job-group",
                "job_title": "Data Engineer",
                "job_description": "Advanced knowledge of Java or Python.",
                "requirement": "Scala",
                "label": "IMPORTANT",
                "evidence_span": "Advanced knowledge of Java or Python.",
                "requirement_group": "programming language",
                "group_operator": "ANY_OF",
                "group_members": ["Java", "Python"],
                "group_label": "IMPORTANT",
            }
        )


def test_annotation_loader_rejects_missing_evidence_and_label_conflicts(tmp_path):
    missing_evidence = tmp_path / "missing.jsonl"
    missing_evidence.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "job_title": "Data Engineer",
                "job_description": "SQL is essential.",
                "requirement": "SQL",
                "label": "CORE",
                "evidence_span": "Python is essential.",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="verbatim"):
        load_annotations([missing_evidence])

    conflicts = tmp_path / "conflicts.jsonl"
    base = {
        "job_id": "job-2",
        "job_title": "Data Engineer",
        "job_description": "Experience with Airflow would be advantageous.",
        "requirement": "Airflow",
        "evidence_span": "Experience with Airflow would be advantageous.",
    }
    conflicts.write_text(
        json.dumps({**base, "label": "PREFERRED"})
        + "\n"
        + json.dumps({**base, "label": "CORE"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="conflicting labels"):
        load_annotations([conflicts])


def test_builder_is_reproducible_preserves_evidence_and_prevents_job_leakage(
    tmp_path,
):
    examples = make_examples()
    builder = RequirementDatasetBuilder(
        seed=123,
        evaluation_ratio=0.2,
        noise_ratio=0.3,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = builder.build(examples, first_dir)
    second = builder.build(examples, second_dir)

    assert first.manifest == second.manifest
    for first_path in sorted(path for path in first_dir.rglob("*") if path.is_file()):
        relative_path = first_path.relative_to(first_dir)
        assert first_path.read_bytes() == (second_dir / relative_path).read_bytes()

    train = read_jsonl(first_dir / "train.jsonl")
    clean = read_jsonl(first_dir / "evaluation" / "clean.jsonl")
    real_world = read_jsonl(first_dir / "evaluation" / "real_world.jsonl")
    noisy = read_jsonl(first_dir / "evaluation" / "noisy.jsonl")

    train_jobs = {record["job_id"] for record in train}
    evaluation_jobs = {record["job_id"] for record in real_world}
    assert train_jobs.isdisjoint(evaluation_jobs)
    assert first.manifest["job_split_overlap"] == []
    assert set(first.manifest["role_distribution"]["source"]) == {
        "role-0",
        "role-1",
    }
    assert set(first.manifest["source_distribution"]["source"]) == {
        "source-0",
        "source-1",
    }
    assert first.manifest["provenance"]["source_with_review_notes"] == 5
    assert len(clean) == len(real_world) == len(noisy) == 3

    noisy_records = [record for record in train if record["is_noisy"]] + noisy
    assert noisy_records
    assert all(record["noise_types"] for record in noisy_records)
    assert all(
        record["evidence_span"] in record["job_description"]
        for record in train + clean + real_world + noisy
    )
    assert all(record["label"] in LABELS for record in train + noisy)

    counts = first.manifest["counts"]
    actual_noise_ratio = counts["train_noisy"] / counts["train_total"]
    assert actual_noise_ratio == pytest.approx(0.3, abs=0.05)


def test_builder_requires_every_label_by_default(tmp_path):
    incomplete = [
        example
        for example in make_examples(jobs_per_label=1)
        if example.label != "NOT_REQUIREMENT"
    ]
    builder = RequirementDatasetBuilder()

    with pytest.raises(DatasetValidationError, match="NOT_REQUIREMENT=0"):
        builder.build(incomplete, tmp_path / "dataset")


def test_job_split_keeps_a_rare_label_in_training(tmp_path):
    examples = make_examples(jobs_per_label=1)
    builder = RequirementDatasetBuilder(
        seed=7,
        evaluation_ratio=0.4,
        noise_ratio=0,
    )

    result = builder.build(examples, tmp_path / "dataset")

    assert result.manifest["evaluation_jobs"] == 0
    assert result.manifest["label_distribution"]["train"] == {
        label: 1 for label in LABELS
    }
