import json

from app.datasets.coverage_report import generate_coverage_report
from app.datasets.requirement_dataset import LABELS, RequirementExample


def example(index: int, label: str, role_family: str) -> RequirementExample:
    evidence = f"Requirement {index} is essential."
    return RequirementExample(
        job_id=f"job-{index}",
        job_title="Engineer",
        job_description=evidence,
        requirement=f"Requirement {index}",
        label=label,
        evidence_span=evidence,
        role_family=role_family,
        source=f"source-{index % 3}",
    )


def test_coverage_report_lists_label_and_role_gaps(tmp_path):
    examples = [
        example(index, "IMPORTANT", "data_engineering")
        for index in range(6)
    ]
    output = tmp_path / "coverage.json"

    summary = generate_coverage_report(
        examples,
        output,
        required_role_families=["data_engineering", "gpu_engineering"],
        minimum_per_label=2,
        minimum_per_role=2,
        minimum_unique_jobs=5,
        minimum_sources=2,
        maximum_label_share=0.6,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert summary.ready_for_training is False
    assert payload["distribution"]["labels"] == {
        label: (6 if label == "IMPORTANT" else 0) for label in LABELS
    }
    assert any(
        gap["dimension"] == "role_family"
        and gap["name"] == "gpu_engineering"
        for gap in payload["gaps"]
    )
    assert any(
        gap["dimension"] == "label_balance" for gap in payload["gaps"]
    )


def test_coverage_report_can_pass_configured_thresholds(tmp_path):
    examples = [
        example(index, label, f"role-{index % 2}")
        for index, label in enumerate(LABELS)
    ]

    summary = generate_coverage_report(
        examples,
        tmp_path / "coverage.json",
        required_role_families=["role-0", "role-1"],
        minimum_per_label=1,
        minimum_per_role=2,
        minimum_unique_jobs=5,
        minimum_sources=3,
        maximum_label_share=0.5,
    )

    assert summary.ready_for_training is True
    assert summary.blockers == 0
