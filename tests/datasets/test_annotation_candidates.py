import json

from app.datasets.annotation_candidates import (
    candidates_from_job,
    generate_candidate_queue,
    promote_approved_candidates,
    validate_candidate_reviews,
)
from app.datasets.requirement_dataset import load_annotations


def sample_job() -> dict:
    description = "\n".join(
        [
            "Strong SQL experience is essential.",
            "You will build production pipelines using PySpark.",
            "Experience with Apache Airflow would be advantageous.",
            "Our platform team uses Scala.",
            "Previous Kubernetes experience is not required.",
            "Python was mentioned in a company blog last year.",
        ]
    )
    return {
        "id": "job-123",
        "title": "Data Engineer",
        "description": description,
        "source": "test",
        "source_url": "https://example.com/jobs/123",
        "metadata": {
            "collection": {
                "role_family": "data_engineering",
                "query": "data engineer",
            }
        },
    }


def test_candidate_generation_uses_explicit_cues_and_requires_review():
    records = candidates_from_job(sample_job())
    labels = {
        (record["requirement"], record["proposed_label"])
        for record in records
    }

    assert ("SQL", "CORE") in labels
    assert ("PySpark", "IMPORTANT") in labels
    assert ("Apache Airflow", "PREFERRED") in labels
    assert ("Scala", "CONTEXTUAL") in labels
    assert ("Kubernetes", "NOT_REQUIREMENT") in labels
    assert not any(record["requirement"] == "Python" for record in records)
    assert all(record["annotation_status"] == "needs_review" for record in records)
    assert all(
        record["evidence_span"] in record["job_description"]
        for record in records
    )
    assert all(record["role_family"] == "data_engineering" for record in records)


def test_candidate_generation_accepts_common_experience_requirement_wording():
    payload = sample_job()
    payload["description"] = "Experience with Python in production environments."

    records = candidates_from_job(payload)

    assert len(records) == 1
    assert records[0]["requirement"] == "Python"
    assert records[0]["proposed_label"] == "IMPORTANT"
    assert records[0]["proposal_confidence"] == 0.78


def test_section_heading_applies_to_following_skill_requirements():
    payload = sample_job()
    payload["description"] = "\n".join(
        [
            "Preferred Qualifications",
            "Experience with Python in production environments.",
            "Required Qualifications",
            "Knowledge of SQL databases and query optimisation.",
        ]
    )

    labels = {
        (record["requirement"], record["proposed_label"])
        for record in candidates_from_job(payload)
    }

    assert ("Python", "PREFERRED") in labels
    assert ("SQL", "CORE") in labels


def test_queue_generation_is_deterministic_and_skips_broken_jobs(tmp_path):
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_text(json.dumps(sample_job()), encoding="utf-8")
    invalid.write_text("not-json", encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_summary = generate_candidate_queue([invalid, valid], first)
    second_summary = generate_candidate_queue([valid, invalid], second)

    assert first_summary.jobs_read == 1
    assert first_summary.jobs_failed == 1
    assert first_summary.candidates == 5
    assert first.read_bytes() == second.read_bytes()


def test_readable_json_queue_can_be_promoted(tmp_path):
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job()), encoding="utf-8")
    queue = tmp_path / "candidates.json"
    generate_candidate_queue([job_path], queue)
    records = json.loads(queue.read_text(encoding="utf-8"))
    assert isinstance(records, list)
    assert queue.read_text(encoding="utf-8").startswith("[\n  {")
    assert "job_description" not in records[0]
    assert records[0]["job_description_ref"] == str(job_path)
    assert isinstance(records[0]["review_context"], list)
    assert records[0]["evidence_span"] in "\n".join(
        records[0]["review_context"]
    )

    records[0]["annotation_status"] = "approved"
    records[0]["reviewed_label"] = records[0]["proposed_label"]
    queue.write_text(json.dumps(records, indent=2), encoding="utf-8")
    generate_candidate_queue([job_path], queue)
    records = json.loads(queue.read_text(encoding="utf-8"))
    approved_record = next(
        record
        for record in records
        if record["annotation_status"] == "approved"
    )
    assert approved_record["reviewed_label"]
    approved = tmp_path / "approved.jsonl"

    summary = promote_approved_candidates(queue, approved)

    assert summary.approved == 1


def test_generation_merges_additional_review_queue_into_readable_json(tmp_path):
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job()), encoding="utf-8")
    extra_path = tmp_path / "negative.jsonl"
    extra = {
        "candidate_id": "negative-1",
        "annotation_status": "needs_review",
        "reviewed_label": None,
        "review_notes": "",
        "job_id": "negative-job",
        "job_title": "Data Engineer",
        "job_description": "Previous Python experience is not required.",
        "requirement": "Python",
        "proposed_label": "NOT_REQUIREMENT",
        "evidence_span": "Previous Python experience is not required.",
    }
    extra_path.write_text(json.dumps(extra) + "\n", encoding="utf-8")
    output = tmp_path / "combined.json"

    summary = generate_candidate_queue(
        [job_path],
        output,
        additional_candidate_paths=[extra_path],
    )
    records = json.loads(output.read_text(encoding="utf-8"))
    merged = next(record for record in records if record["candidate_id"] == "negative-1")

    assert summary.candidates == 6
    assert merged["job_description"] == extra["job_description"]


def test_only_explicitly_reviewed_labels_can_be_promoted(tmp_path):
    records = candidates_from_job(sample_job())
    records[0]["annotation_status"] = "approved"
    records[0]["reviewed_label"] = records[0]["proposed_label"]
    queue = tmp_path / "candidates.jsonl"
    queue.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    approved = tmp_path / "requirements.jsonl"

    summary = promote_approved_candidates(queue, approved)
    examples = load_annotations([approved])

    assert summary.candidates_read == 5
    assert summary.approved == 1
    assert summary.pending == 4
    assert len(examples) == 1
    assert examples[0].annotation_source == "human_review"
    assert examples[0].role_family == "data_engineering"


def test_review_notes_and_rejections_are_preserved_in_decision_log(tmp_path):
    records = candidates_from_job(sample_job())
    records[0]["annotation_status"] = "approved"
    records[0]["reviewed_label"] = records[0]["proposed_label"]
    records[0]["review_notes"] = "Wording is explicit, but production scale matters."
    records[1]["annotation_status"] = "rejected"
    records[1]["review_notes"] = "This mention describes another team, not the candidate."
    queue = tmp_path / "candidates.json"
    queue.write_text(json.dumps(records, indent=2), encoding="utf-8")
    approved = tmp_path / "requirements.jsonl"
    decisions = tmp_path / "review_decisions.json"

    summary = promote_approved_candidates(
        queue,
        approved,
        decision_log_path=decisions,
    )
    examples = load_annotations([approved])
    decision_payload = json.loads(decisions.read_text(encoding="utf-8"))

    assert summary.approved == 1
    assert summary.rejected == 1
    assert summary.pending == 3
    assert examples[0].review_notes == (
        "Wording is explicit, but production scale matters."
    )
    assert decision_payload["summary"]["with_review_notes"] == 2
    rejected = next(
        decision
        for decision in decision_payload["decisions"]
        if decision["annotation_status"] == "rejected"
    )
    assert "another team" in rejected["review_notes"]


def test_title_case_approval_accepts_proposed_label(tmp_path):
    records = candidates_from_job(sample_job())
    records[0]["annotation_status"] = "Approved"
    queue = tmp_path / "candidates.json"
    queue.write_text(json.dumps(records, indent=2), encoding="utf-8")
    approved = tmp_path / "requirements.jsonl"
    decisions = tmp_path / "review_decisions.json"

    summary = promote_approved_candidates(
        queue,
        approved,
        decision_log_path=decisions,
    )
    examples = load_annotations([approved])
    decision_payload = json.loads(decisions.read_text(encoding="utf-8"))

    assert summary.approved == 1
    assert examples[0].label == records[0]["proposed_label"]
    approved_decision = next(
        decision
        for decision in decision_payload["decisions"]
        if decision["annotation_status"] == "approved"
    )
    assert approved_decision["annotation_status_raw"] == "Approved"
    assert approved_decision["label_source"] == "proposed_label"


def test_review_validation_flags_rejection_that_looks_like_label_correction(
    tmp_path,
):
    records = candidates_from_job(sample_job())
    records[0]["annotation_status"] = "Reject"
    records[0]["review_notes"] = "SQL should be classified as core here."
    queue = tmp_path / "candidates.json"
    queue.write_text(json.dumps(records, indent=2), encoding="utf-8")
    report = tmp_path / "review_validation.json"

    summary = validate_candidate_reviews(queue, report)
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert summary.errors == 0
    correction = next(
        issue
        for issue in payload["issues"]
        if issue["code"] == "possible_label_correction_rejected"
    )
    assert correction["suggested_status"] == "approved"
    assert correction["suggested_reviewed_label"] == "CORE"


def test_grouped_review_metadata_survives_promotion(tmp_path):
    records = candidates_from_job(sample_job())
    java_like = records[0]
    java_like["annotation_status"] = "approved"
    java_like["reviewed_label"] = java_like["proposed_label"]
    java_like["requirement_group"] = "database technology"
    java_like["group_operator"] = "ANY_OF"
    java_like["group_members"] = [java_like["requirement"], "PostgreSQL"]
    java_like["group_label"] = java_like["proposed_label"]
    queue = tmp_path / "candidates.jsonl"
    queue.write_text(json.dumps(java_like) + "\n", encoding="utf-8")
    approved = tmp_path / "requirements.jsonl"

    promote_approved_candidates(queue, approved)
    promoted = json.loads(approved.read_text(encoding="utf-8"))

    assert promoted["requirement_group"] == "database technology"
    assert promoted["group_operator"] == "ANY_OF"
    assert promoted["group_members"] == [java_like["requirement"], "PostgreSQL"]
    assert promoted["group_label"] == java_like["proposed_label"]
