import json
from pathlib import Path

import pytest

from app.applications.models import (
    CandidateProfile, ConstraintCheck, CVEntry, CVEvaluation, CVSection,
    MatchAssessment, Preferences, RequirementMatch, TailoredCV,
)
from app.applications.tracking import ApplicationLedger, job_key
from app.applications.workflow import ApplicationWorkflow
from app.providers.openai import OpenAIProviderError


MASTER_CV = """Alex Example
alex@example.test
London
Data Engineer | Example Ltd | 2022–present
Built Python and SQL pipelines for data quality reporting.
BSc Computer Science, Example University, 2021.
"""
DESCRIPTION = """Data Engineer at Hiring Ltd. Python is required. SQL experience is expected.
The role is based in London. You will help maintain reporting pipelines and work
with colleagues to understand the data that their teams need. The successful
candidate will have time to learn the existing systems, discuss improvements,
and document the changes that they make. We welcome applications from people
with a range of professional backgrounds and provide a supportive environment.
"""


class QueuedProvider:
    model = "offline-test-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_structured(self, *, prompt, schema, system_prompt):
        self.calls.append({"payload": json.loads(prompt), "schema": schema, "system": system_prompt})
        if not self.responses:
            raise AssertionError("The workflow made an unexpected model request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, schema)
        return response


class RecordingExporter:
    def __init__(self):
        self.calls = []

    def __call__(self, markdown, directory):
        directory = Path(directory)
        self.calls.append((markdown, directory))
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for key, suffix in (("markdown", "md"), ("docx", "docx"), ("pdf", "pdf")):
            path = directory / f"cv.{suffix}"
            path.write_text(markdown if suffix == "md" else "offline export stub", encoding="utf-8")
            paths[key] = str(path.resolve())
        return paths


@pytest.fixture
def job():
    return {
        "title": "Data Engineer", "company": "Hiring Ltd", "source": "greenhouse",
        "external_id": "123", "source_url": "https://boards.greenhouse.io/hiring/jobs/123",
        "description": DESCRIPTION, "metadata": {"description_type": "full"},
    }


@pytest.fixture
def profile():
    return CandidateProfile(cv_path="/private/master_cv.md", name="Alex Example", email="alex@example.test")


def assessment(*, sql_status="MET", core_status="MET", checks=()):
    evidence = "Built Python and SQL pipelines for data quality reporting."
    return MatchAssessment(
        description_complete=True,
        requirements=[
            RequirementMatch(
                requirement="Python", importance="CORE", job_evidence="Python is required.",
                status=core_status, cv_evidence=[evidence] if core_status in {"MET", "PARTIAL"} else [],
                explanation="Compared with the supplied CV.",
            ),
            RequirementMatch(
                requirement="SQL", importance="IMPORTANT", job_evidence="SQL experience is expected.",
                status=sql_status, cv_evidence=[evidence] if sql_status in {"MET", "PARTIAL"} else [],
                explanation="Compared with the supplied CV.",
            ),
        ],
        constraint_checks=list(checks), recommended_verdict="APPLY",
        rationale="The documented skills fit the role.", uncertainties=[],
    )


def tailored_cv():
    def entry(text, style="paragraph"):
        return CVEntry(text=text, style=style, evidence_quotes=[text])

    return TailoredCV(sections=[
        CVSection(heading="Contact", entries=[entry("Alex Example"), entry("alex@example.test")]),
        CVSection(heading="Experience", entries=[
            entry("Data Engineer | Example Ltd | 2022–present"),
            entry("Built Python and SQL pipelines for data quality reporting.", "bullet"),
        ]),
        CVSection(heading="Education", entries=[entry("BSc Computer Science, Example University, 2021.")]),
    ])


def evaluation(**changes):
    values = dict(
        requirement_coverage=90, clarity=90, factual_consistency=True,
        unsupported_claims=[], missing_critical_information=[], improvements=[],
    )
    values.update(changes)
    return CVEvaluation(**values)


def make_workflow(tmp_path, responses, **options):
    provider = QueuedProvider(responses)
    exporter = RecordingExporter()
    ledger = ApplicationLedger(tmp_path / "applications.sqlite3")
    workflow = ApplicationWorkflow(provider, ledger, tmp_path / "output", exporter=exporter, **options)
    return workflow, provider, exporter


def test_apply_match_tailors_audits_and_exports_ready_package(tmp_path, job, profile):
    workflow, provider, exporter = make_workflow(tmp_path, [assessment(), tailored_cv(), evaluation()])

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "ready"
    assert record["decision"]["verdict"] == "APPLY"
    assert record["decision"]["score"] == 100
    assert record["cv_score"] == 90
    assert [call["schema"] for call in provider.calls] == [MatchAssessment, TailoredCV, CVEvaluation]
    assert provider.responses == []
    assert provider.calls[2]["payload"]["master_cv"] == MASTER_CV
    assert provider.calls[2]["payload"]["proposed_cv"] == exporter.calls[0][0]
    markdown = Path(record["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert markdown.startswith("# Alex Example\n\n")
    assert "## Experience\n\n" in markdown
    assert "- Built Python and SQL pipelines" in markdown
    assert markdown.endswith("\n")
    assert record["validation_errors"] == []
    assert len(exporter.calls) == 1
    report = json.loads(Path(record["artifacts"]["assessment"]).read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    saved = workflow.ledger.get(job_key(job))
    assert saved["artifacts"] == record["artifacts"]
    assert "cv_path" not in saved["profile"]
    assert "preferences" not in saved["profile"]


def test_low_match_score_skips_without_tailoring(tmp_path, job, profile):
    workflow, provider, exporter = make_workflow(tmp_path, [assessment(sql_status="MISSING")])

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "skipped"
    assert record["decision"]["score"] < 70
    assert len(provider.calls) == 1
    assert not exporter.calls


@pytest.mark.parametrize("unknown", ["constraint", "core"])
def test_unknown_mandatory_information_needs_review(tmp_path, job, profile, unknown):
    if unknown == "constraint":
        profile.preferences = Preferences(requires_sponsorship=True)
        response = assessment(checks=[ConstraintCheck(
            constraint_id="sponsorship", status="UNKNOWN", job_evidence=None,
            explanation="Sponsorship is not addressed in this advertisement.",
        )])
    else:
        response = assessment(core_status="UNKNOWN")
    workflow, provider, exporter = make_workflow(tmp_path, [response])

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert record["decision"]["verdict"] == "REVIEW"
    assert record["reasons"]
    assert len(provider.calls) == 1
    assert not exporter.calls


@pytest.mark.parametrize("incomplete", ["summary", "short"])
def test_incomplete_job_never_spends_model_calls(tmp_path, job, profile, incomplete):
    if incomplete == "summary":
        job["metadata"]["description_type"] = "summary"
    else:
        job["description"] = "Python developer wanted."
    workflow, provider, exporter = make_workflow(tmp_path, [])

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert workflow.ledger.get(job_key(job))["status"] == "review"
    assert not provider.calls
    assert not exporter.calls


@pytest.mark.parametrize("bad_evidence", [[], ["I have invented Python qualifications."]])
def test_unverified_match_evidence_cannot_trigger_tailoring(tmp_path, job, profile, bad_evidence):
    response = assessment()
    response.requirements[0].cv_evidence = bad_evidence
    workflow, provider, exporter = make_workflow(tmp_path, [response])

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert len(provider.calls) == 1
    assert not exporter.calls


def test_invented_cv_source_evidence_blocks_ready_even_with_high_audit_score(tmp_path, job, profile):
    cv = tailored_cv()
    cv.sections[1].entries[1].evidence_quotes = ["Invented employer evidence."]
    workflow, provider, _ = make_workflow(tmp_path, [assessment(), cv, evaluation()], max_revisions=0)

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert record["cv_score"] == 90
    assert record["validation_errors"]
    assert len(provider.calls) == 3


@pytest.mark.parametrize("audit", [
    {"factual_consistency": False},
    {"unsupported_claims": ["The proposed performance metric has no source."]},
    {"missing_critical_information": ["The email address is missing."]},
])
def test_failed_factual_audit_cannot_be_overridden_by_high_score(tmp_path, job, profile, audit):
    workflow, _, _ = make_workflow(
        tmp_path, [assessment(), tailored_cv(), evaluation(**audit)], max_revisions=0,
    )

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert record["cv_score"] == 90


def test_revision_receives_failed_audit_feedback_and_can_reach_ready(tmp_path, job, profile):
    failed = evaluation(clarity=30, requirement_coverage=50, improvements=["Make the relevant SQL work clearer."])
    workflow, provider, exporter = make_workflow(
        tmp_path, [assessment(), tailored_cv(), failed, tailored_cv(), evaluation()], max_revisions=1,
    )

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "ready"
    assert len(record["cv_attempts"]) == 2
    assert len(provider.calls) == 5
    assert provider.calls[1]["payload"]["feedback"] is None
    assert provider.calls[3]["payload"]["feedback"]["evaluation"] == failed.model_dump()
    assert len(exporter.calls) == 1


def test_revision_budget_stops_unproductive_generation(tmp_path, job, profile):
    failed = evaluation(clarity=20, requirement_coverage=20)
    responses = [assessment()]
    for _ in range(3):
        responses.extend([tailored_cv(), failed])
    workflow, provider, _ = make_workflow(tmp_path, responses, max_revisions=2)

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert len(record["cv_attempts"]) == 3
    assert len(provider.calls) == 7
    assert not provider.responses


def test_ready_rerun_uses_saved_artifacts_without_model_calls(tmp_path, job, profile):
    workflow, provider, exporter = make_workflow(tmp_path, [assessment(), tailored_cv(), evaluation()])
    first = workflow.prepare(job, profile, MASTER_CV)

    repeated = workflow.prepare(job, profile, MASTER_CV)

    assert repeated["cached"] is True
    assert repeated["status"] == "ready"
    assert repeated["artifacts"] == first["artifacts"]
    assert len(provider.calls) == 3
    assert len(exporter.calls) == 1


@pytest.mark.parametrize("state", ["submitted", "submission_unknown", "submission_attempted"])
def test_previously_submitted_or_attempted_job_never_regenerates(tmp_path, job, profile, state):
    workflow, provider, exporter = make_workflow(tmp_path, [])
    key = job_key(job)
    previous = {"status": "ready", "artifacts": {"pdf": "/existing/cv.pdf"}}
    if state == "submission_unknown":
        previous["status"] = state
    elif state == "submission_attempted":
        previous.update(status="opened", submission_attempted=True)
    workflow.ledger.save(key, previous)
    if state == "submitted":
        workflow.ledger.mark_submitted(key, "Employer receipt 123")

    record = workflow.prepare(job, profile, MASTER_CV + "Changed CV text", retry=True)

    assert record["cached"] is True
    assert record["artifacts"] == previous["artifacts"]
    assert not provider.calls
    assert not exporter.calls
    assert workflow.ledger.get(key)["status"] == record["status"]


@pytest.mark.parametrize("stage", ["match", "evaluation"])
def test_unexpected_api_failure_is_saved_without_exception_secrets(tmp_path, job, profile, stage):
    secret = "sk-secret-do-not-persist"
    failure = RuntimeError(f"Request failed with credential {secret} and confidential response body")
    responses = [failure] if stage == "match" else [assessment(), tailored_cv(), failure]
    workflow, _, exporter = make_workflow(tmp_path, responses)

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "failed"
    assert "RuntimeError" in record["reasons"][0]
    persisted = workflow.ledger.get(job_key(job))
    assert persisted["status"] == "failed"
    assert secret not in json.dumps(record)
    assert secret not in json.dumps(persisted)
    assert "confidential response body" not in json.dumps(persisted)
    assert not exporter.calls


def test_provider_error_keeps_its_sanitized_actionable_advice(tmp_path, job, profile):
    message = "OpenAI request failed (HTTP 429). Check API quota/rate limits and retry later."
    workflow, _, _ = make_workflow(tmp_path, [OpenAIProviderError(message)])

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "failed"
    assert record["reasons"] == [message]


@pytest.mark.parametrize("options", [
    {"max_revisions": -1}, {"max_revisions": 3},
    {"match_threshold": -1}, {"match_threshold": 101},
    {"cv_threshold": -1}, {"cv_threshold": 101},
])
def test_invalid_budgets_or_thresholds_fail_before_model_calls(tmp_path, options):
    with pytest.raises(ValueError):
        make_workflow(tmp_path, [], **options)
