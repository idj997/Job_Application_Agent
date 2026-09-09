import json
from pathlib import Path

import pytest

from app.datasets.annotation_candidates import promote_approved_candidates
from app.datasets.counterfactual_candidates import generate_counterfactual_candidates
from app.datasets.requirement_dataset import (
    DatasetValidationError,
    RequirementDatasetBuilder,
    RequirementExample,
    load_annotations,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "datasets" / "requirements.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_counterfactuals_only_derive_from_real_positive_annotations(tmp_path):
    source_examples = load_annotations([FIXTURE])
    output = tmp_path / "counterfactuals.jsonl"

    summary = generate_counterfactual_candidates(
        source_examples,
        output,
        variants_per_example=2,
        seed=17,
    )
    records = read_jsonl(output)
    source_by_id = {example.example_id: example for example in source_examples}

    assert summary.source_examples == 10
    assert summary.eligible_examples == 6
    assert summary.candidates == 12
    assert all(record["proposed_label"] == "NOT_REQUIREMENT" for record in records)
    assert all(record["annotation_status"] == "needs_review" for record in records)
    assert all(record["is_synthetic"] for record in records)
    assert all(record["job_description"] == record["evidence_span"] for record in records)
    assert all(record["source_label"] in {"CORE", "IMPORTANT", "PREFERRED"} for record in records)
    assert all(record["source_example_id"] in source_by_id for record in records)
    assert all(
        record["job_id"] == source_by_id[record["source_example_id"]].job_id
        for record in records
    )


def test_counterfactual_generation_is_deterministic(tmp_path):
    examples = load_annotations([FIXTURE])
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    generate_counterfactual_candidates(examples, first, seed=99)
    generate_counterfactual_candidates(reversed(examples), second, seed=99)

    assert first.read_bytes() == second.read_bytes()


def test_counterfactual_generation_skips_any_of_group_members(tmp_path):
    grouped = RequirementExample.from_dict(
        {
            "job_id": "grouped-job",
            "job_title": "Data Engineer",
            "job_description": "Advanced knowledge of Java or Python is expected.",
            "requirement": "Java",
            "label": "IMPORTANT",
            "evidence_span": "Advanced knowledge of Java or Python is expected.",
            "requirement_group": "programming language",
            "group_operator": "ANY_OF",
            "group_members": ["Java", "Python"],
            "group_label": "IMPORTANT",
        }
    )
    output = tmp_path / "counterfactuals.jsonl"

    summary = generate_counterfactual_candidates([grouped], output)

    assert summary.eligible_examples == 0
    assert summary.grouped_examples_skipped == 1
    assert summary.candidates == 0
    assert output.read_text(encoding="utf-8") == ""


def test_reviewed_counterfactual_preserves_provenance_through_promotion(tmp_path):
    examples = load_annotations([FIXTURE])
    queue = tmp_path / "queue.jsonl"
    generate_counterfactual_candidates(examples, queue)
    records = read_jsonl(queue)
    records[0]["annotation_status"] = "approved"
    records[0]["reviewed_label"] = "NOT_REQUIREMENT"
    queue.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    approved = tmp_path / "approved.jsonl"

    promote_approved_candidates(queue, approved)
    promoted = load_annotations([approved])

    assert len(promoted) == 1
    assert promoted[0].label == "NOT_REQUIREMENT"
    assert promoted[0].is_synthetic is True
    assert promoted[0].synthetic_type.startswith("explicit_negation:")
    assert promoted[0].source_example_id
    assert promoted[0].source_label in {"CORE", "IMPORTANT", "PREFERRED"}
    assert promoted[0].original_evidence_span
    assert promoted[0].generation_seed == 42


def test_dataset_builder_rejects_excessive_synthetic_content(tmp_path):
    real_examples = load_annotations([FIXTURE])
    queue = tmp_path / "queue.jsonl"
    generate_counterfactual_candidates(real_examples, queue)
    synthetic_examples = [
        RequirementExample.from_dict(
            {
                **record,
                "label": "NOT_REQUIREMENT",
            }
        )
        for record in read_jsonl(queue)
    ]
    builder = RequirementDatasetBuilder(maximum_synthetic_ratio=0.1)

    with pytest.raises(DatasetValidationError, match="synthetic example ratio"):
        builder.build(real_examples + synthetic_examples, tmp_path / "dataset")


def test_synthetic_evaluation_is_separate_without_job_leakage(tmp_path):
    real_examples = load_annotations([FIXTURE])
    queue = tmp_path / "queue.jsonl"
    generate_counterfactual_candidates(real_examples, queue)
    synthetic_examples = [
        RequirementExample.from_dict({**record, "label": "NOT_REQUIREMENT"})
        for record in read_jsonl(queue)
    ]
    output = tmp_path / "dataset"
    builder = RequirementDatasetBuilder(
        seed=11,
        evaluation_ratio=0.9,
        noise_ratio=0,
    )

    result = builder.build(real_examples + synthetic_examples, output)

    real_world = read_jsonl(output / "evaluation" / "real_world.jsonl")
    synthetic = read_jsonl(output / "evaluation" / "synthetic.jsonl")
    train = read_jsonl(output / "train.jsonl")
    assert synthetic
    assert all(not record["is_synthetic"] for record in real_world)
    assert all(record["is_synthetic"] for record in synthetic)
    assert {record["job_id"] for record in train}.isdisjoint(
        {record["job_id"] for record in real_world + synthetic}
    )
    assert result.manifest["counts"]["evaluation_synthetic"] == len(synthetic)
