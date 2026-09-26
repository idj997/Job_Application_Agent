import json
from pathlib import Path

import pytest

from app.applications.cv_evidence import ReferencedCV
from app.applications.models import (
    CandidateProfile, ConstraintCheck, CVEvaluation, MatchAssessment, Preferences,
    RequirementMatch, TailoredCV,
)
from app.applications.tracking import ApplicationLedger, job_key
from app.applications.workflow import ApplicationWorkflow
from app.providers.openai import OpenAIProviderError
from app.providers.base import ModelProviderError


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


def referenced_cv():
    def entry(text, source_id, style="paragraph"):
        return {"text": text, "style": style, "evidence_ids": [source_id]}

    return ReferencedCV.model_validate({"sections": [
        {"heading": "Contact", "entries": [
            entry("Alex Example", "S0001"), entry("alex@example.test", "S0002"),
        ]},
        {"heading": "Experience", "entries": [
            entry("Data Engineer | Example Ltd | 2022–present", "S0004"),
            entry("Built Python and SQL pipelines for data quality reporting.", "S0005", "bullet"),
        ]},
        {"heading": "Education", "entries": [
            entry("BSc Computer Science, Example University, 2021.", "S0006"),
        ]},
    ]})


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


class RecordingMatcher:
    name = "classifier"

    def __init__(self, result=None, version="offline-v1"):
        self.result = result or assessment()
        self.fingerprint = {"version": version, "model": "offline-classifier"}
        self.diagnostics = {"source": "offline test", "pairs": 2}
        self.calls = []

    def assess(self, job, profile, master_cv):
        self.calls.append((job, profile, master_cv))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_classifier_matching_calls_generator_only_for_cv_and_audit(tmp_path, job, profile):
    match = assessment()
    match.recommended_verdict = "SKIP"  # Ignored by classifier-mode deterministic decision.
    matcher = RecordingMatcher(match)
    workflow, provider, _ = make_workflow(
        tmp_path, [referenced_cv(), evaluation()], matcher=matcher,
    )

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "ready"
    assert record["decision"]["verdict"] == "APPLY"
    assert record["matcher"] == "classifier"
    assert record["matcher_config"] == matcher.fingerprint
    assert record["matcher_diagnostics"] == matcher.diagnostics
    assert matcher.calls == [(job, profile, MASTER_CV)]
    assert [call["schema"] for call in provider.calls] == [ReferencedCV, CVEvaluation]


@pytest.mark.parametrize("problem", ["core_gap", "uncertainty", "invented_evidence"])
def test_classifier_uncertainty_and_bad_evidence_never_start_qwen(tmp_path, job, profile, problem):
    match = assessment(core_status="UNKNOWN" if problem == "core_gap" else "MET")
    if problem == "uncertainty":
        match.uncertainties = ["A complex eligibility clause needs human review."]
    if problem == "invented_evidence":
        match.requirements[0].cv_evidence = ["Invented experience not found in the CV."]
    workflow, provider, exporter = make_workflow(tmp_path, [], matcher=RecordingMatcher(match))

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert provider.calls == [] and exporter.calls == []


def test_match_only_can_run_without_any_generation_provider(tmp_path, job, profile):
    matcher = RecordingMatcher()
    ledger = ApplicationLedger(tmp_path / "applications.sqlite3")
    workflow = ApplicationWorkflow(None, ledger, tmp_path / "output", matcher=matcher)

    record = workflow.prepare(job, profile, MASTER_CV, match_only=True)

    assert record["status"] == "matched"
    assert record["match_only"] is True
    assert record["decision"]["verdict"] == "APPLY"
    assert record["artifacts"] == {}
    with pytest.raises(ValueError, match="not ready"):
        ledger.claim_open(record["job_key"])


def test_match_only_cache_cannot_replace_full_preparation(tmp_path, job, profile):
    matcher = RecordingMatcher()
    workflow, provider, _ = make_workflow(tmp_path, [referenced_cv(), evaluation()], matcher=matcher)

    first = workflow.prepare(job, profile, MASTER_CV, match_only=True)
    repeated = workflow.prepare(job, profile, MASTER_CV, match_only=True)
    full = workflow.prepare(job, profile, MASTER_CV)

    assert first["status"] == repeated["status"] == "matched"
    assert repeated["cached"] is True
    assert full["status"] == "ready" and not full.get("cached")
    assert len(matcher.calls) == 2
    assert len(provider.calls) == 2


def test_classifier_failure_does_not_fall_back_to_model_matching(tmp_path, job, profile):
    matcher = RecordingMatcher(ModelProviderError("Local classifier weights are unavailable."))
    workflow, provider, exporter = make_workflow(tmp_path, [], matcher=matcher)

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "failed"
    assert record["reasons"] == ["Local classifier weights are unavailable."]
    assert provider.calls == [] and exporter.calls == []


def test_classifier_configuration_change_invalidates_unattempted_cache(tmp_path, job, profile):
    matcher = RecordingMatcher()
    workflow, _, _ = make_workflow(tmp_path, [], matcher=matcher)
    first = workflow.prepare(job, profile, MASTER_CV, match_only=True)
    matcher.fingerprint = {**matcher.fingerprint, "version": "offline-v2"}

    second = workflow.prepare(job, profile, MASTER_CV, match_only=True)

    assert second["input_fingerprint"] != first["input_fingerprint"]
    assert not second.get("cached")
    assert len(matcher.calls) == 2


def test_switching_to_classifier_never_releases_an_attempted_application(tmp_path, job, profile):
    workflow, provider, _ = make_workflow(tmp_path, [assessment(), referenced_cv(), evaluation()])
    initial = workflow.prepare(job, profile, MASTER_CV)
    workflow.ledger.claim_open(initial["job_key"])
    matcher = RecordingMatcher()
    workflow.matcher = matcher

    record = workflow.prepare(job, profile, MASTER_CV, match_only=True, retry=True)

    assert record["cached"] is True
    assert record["submission_attempted"] is True
    assert matcher.calls == []
    assert len(provider.calls) == 3


def test_explicit_legacy_match_only_stops_after_one_assessment(tmp_path, job, profile):
    workflow, provider, _ = make_workflow(tmp_path, [assessment()])
    record = workflow.prepare(job, profile, MASTER_CV, match_only=True)
    assert record["status"] == "matched"
    assert [call["schema"] for call in provider.calls] == [MatchAssessment]


def test_apply_match_tailors_audits_and_exports_ready_package(tmp_path, job, profile):
    workflow, provider, exporter = make_workflow(tmp_path, [assessment(), referenced_cv(), evaluation()])

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "ready"
    assert record["decision"]["verdict"] == "APPLY"
    assert record["decision"]["score"] == 100
    assert record["cv_score"] == 90
    assert [call["schema"] for call in provider.calls] == [MatchAssessment, ReferencedCV, CVEvaluation]
    assert provider.responses == []
    source_catalog = [
        {"source_id": f"S{index:04d}", "text": line}
        for index, line in enumerate(MASTER_CV.splitlines(), start=1)
    ]
    tailor_payload = provider.calls[1]["payload"]
    assert tailor_payload["master_cv_sources"] == source_catalog
    assert "master_cv" not in tailor_payload
    assert tailor_payload["tailoring_policy"]
    assert provider.calls[2]["payload"]["master_cv"] == MASTER_CV
    assert provider.calls[2]["payload"]["proposed_cv"] == record["cv_attempts"][0]["cv"]
    markdown = Path(record["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert markdown.startswith("# Alex Example\n\n")
    assert "## Experience\n\n" in markdown
    assert "- Built Python and SQL pipelines" in markdown
    assert markdown.endswith("\n")
    assert record["validation_errors"] == []
    assert record["cv_attempts"][0]["referenced_cv"] == referenced_cv().model_dump()
    resolved = TailoredCV.model_validate(record["cv_attempts"][0]["cv"])
    for section in resolved.sections:
        for entry in section.entries:
            assert entry.evidence_quotes == [entry.text]
    assert json.loads(Path(record["artifacts"]["source_catalog"]).read_text()) == source_catalog
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


def test_unknown_cv_source_id_fails_closed_before_audit_or_export(tmp_path, job, profile):
    cv = referenced_cv()
    cv.sections[1].entries[1].evidence_ids = ["S9999"]
    workflow, provider, exporter = make_workflow(tmp_path, [assessment(), cv, evaluation()], max_revisions=0)

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "failed"
    assert "cv_score" not in record
    assert record["reasons"]
    assert [call["schema"] for call in provider.calls] == [MatchAssessment, ReferencedCV]
    assert len(provider.responses) == 1
    assert not exporter.calls
    assert "draft_json" not in record["artifacts"]
    assert "referenced_cv" not in record["artifacts"]
    assert json.loads(Path(record["artifacts"]["failed_referenced_cv"]).read_text()) == cv.model_dump()


def test_exact_claim_relinks_valid_wrong_id_but_preserves_raw_output_and_runs_audit(tmp_path, job, profile):
    model_cv = referenced_cv()
    model_cv.sections[1].entries[1].evidence_ids = ["S0003"]  # London, not the achievement.
    raw_output = model_cv.model_dump()
    workflow, provider, exporter = make_workflow(
        tmp_path, [assessment(), model_cv, evaluation()], max_revisions=0,
    )

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "ready"
    assert [call["schema"] for call in provider.calls] == [MatchAssessment, ReferencedCV, CVEvaluation]
    assert not provider.responses and len(exporter.calls) == 1
    attempt = record["cv_attempts"][0]
    assert attempt["model_referenced_cv"] == raw_output == model_cv.model_dump()
    assert attempt["referenced_cv"] == referenced_cv().model_dump()
    raw_path = Path(record["artifacts"]["model_referenced_cv"])
    canonical_path = Path(record["artifacts"]["referenced_cv"])
    assert raw_path.name == "cv_model_references_1.json"
    assert canonical_path.name == "cv_references_1.json"
    assert json.loads(raw_path.read_text()) == raw_output
    assert json.loads(canonical_path.read_text()) == referenced_cv().model_dump()
    assert attempt["evidence_reconciliations"] == [{
        "section_index": 2, "entry_index": 2,
        "original_ids": ["S0003"], "resolved_ids": ["S0005"],
        "reason": "exact_source_line_match",
    }]
    assert record["evidence_reconciliations"] == attempt["evidence_reconciliations"]
    for raw_section, resolved_section in zip(raw_output["sections"], attempt["cv"]["sections"]):
        assert [entry["text"] for entry in raw_section["entries"]] == [
            entry["text"] for entry in resolved_section["entries"]
        ]
    audit_entry = provider.calls[2]["payload"]["proposed_cv"]["sections"][1]["entries"][1]
    assert audit_entry == {
        "text": "Built Python and SQL pipelines for data quality reporting.",
        "style": "bullet",
        "evidence_quotes": ["Built Python and SQL pipelines for data quality reporting."],
    }
    assert attempt["evaluation"] == evaluation().model_dump()


def test_nonexact_unsupported_claim_keeps_ids_and_negative_audit_blocks_high_score(tmp_path, job, profile):
    model_cv = referenced_cv()
    unsupported_claim = "Led production C++ optimization that improved CPU performance by 40%."
    model_cv.sections[1].entries[1].text = unsupported_claim
    model_cv.sections[1].entries[1].evidence_ids = ["S0003"]
    rejected = evaluation(
        requirement_coverage=99, clarity=99, factual_consistency=False,
        unsupported_claims=["The C++ responsibility and 40% metric are unsupported."],
    )
    workflow, provider, _ = make_workflow(
        tmp_path, [assessment(), model_cv, rejected], max_revisions=0,
    )

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review" and record["cv_score"] == 99
    attempt = record["cv_attempts"][0]
    assert attempt["model_referenced_cv"] == attempt["referenced_cv"] == model_cv.model_dump()
    assert attempt["evidence_reconciliations"] == record["evidence_reconciliations"] == []
    assert [call["schema"] for call in provider.calls] == [MatchAssessment, ReferencedCV, CVEvaluation]
    assert provider.calls[2]["payload"]["proposed_cv"]["sections"][1]["entries"][1] == {
        "text": unsupported_claim, "style": "bullet", "evidence_quotes": ["London"],
    }
    assert attempt["evaluation"] == rejected.model_dump()
    assert rejected.unsupported_claims[0] in record["reasons"]
    with pytest.raises(ValueError, match="not ready"):
        workflow.ledger.claim_open(record["job_key"])


@pytest.mark.parametrize("audit", [
    {"factual_consistency": False},
    {"unsupported_claims": ["The proposed performance metric has no source."]},
    {"missing_critical_information": ["The email address is missing."]},
])
def test_failed_factual_audit_cannot_be_overridden_by_high_score(tmp_path, job, profile, audit):
    workflow, _, _ = make_workflow(
        tmp_path, [assessment(), referenced_cv(), evaluation(**audit)], max_revisions=0,
    )

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert record["cv_score"] == 90


def test_revision_receives_failed_audit_feedback_and_can_reach_ready(tmp_path, job, profile):
    failed = evaluation(clarity=30, requirement_coverage=50, improvements=["Make the relevant SQL work clearer."])
    workflow, provider, exporter = make_workflow(
        tmp_path, [assessment(), referenced_cv(), failed, referenced_cv(), evaluation()], max_revisions=1,
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
        responses.extend([referenced_cv(), failed])
    workflow, provider, _ = make_workflow(tmp_path, responses, max_revisions=2)

    record = workflow.prepare(job, profile, MASTER_CV)

    assert record["status"] == "review"
    assert len(record["cv_attempts"]) == 3
    assert len(provider.calls) == 7
    assert not provider.responses


def test_ready_rerun_uses_saved_artifacts_without_model_calls(tmp_path, job, profile):
    workflow, provider, exporter = make_workflow(tmp_path, [assessment(), referenced_cv(), evaluation()])
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
    responses = [failure] if stage == "match" else [assessment(), referenced_cv(), failure]
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


def test_local_provider_uses_identical_generation_and_factual_gates(tmp_path, job, profile):
    workflow, provider, exporter = make_workflow(tmp_path, [assessment(), referenced_cv(), evaluation()])
    provider.name = "ollama"
    provider.model = "qwen3:8b"
    record = workflow.prepare(job, profile, MASTER_CV)
    assert record["status"] == "ready"
    assert record["provider"] == "ollama"
    assert record["model"] == "qwen3:8b"
    assert [call["schema"] for call in provider.calls] == [MatchAssessment, ReferencedCV, CVEvaluation]
    assert exporter.calls


def test_changing_backend_invalidates_cached_assessment_even_with_same_model_name(tmp_path, job, profile):
    workflow, provider, _ = make_workflow(tmp_path, [assessment(core_status="UNKNOWN")])
    provider.name = "openai"
    first = workflow.prepare(job, profile, MASTER_CV)
    provider.name = "ollama"
    provider.responses = [assessment(core_status="UNKNOWN")]
    second = workflow.prepare(job, profile, MASTER_CV)
    assert second["provider"] == "ollama"
    assert not second.get("cached")
    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert len(provider.calls) == 2


def test_common_provider_error_preserves_safe_local_advice(tmp_path, job, profile):
    message = "Ollama is unavailable. Start the local server and retry."
    workflow, _, _ = make_workflow(tmp_path, [ModelProviderError(message)])
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
