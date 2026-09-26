"""Single-job manual tailoring is auditable and never authorizes submission."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.applications.cv_evidence import ReferencedCV
from app.applications.models import CandidateProfile, CVEvaluation, MatchAssessment, TailoredCV
from app.applications.tracking import ApplicationLedger, job_key
from app.applications.workflow import ApplicationWorkflow
from scripts import apply_jobs


CV = "Alex Example\nalex@example.test\nEngineer | Example Ltd | 2022-present\nBuilt Python pipelines.\nBSc Computer Science, Example University, 2021."
DESCRIPTION = "Python is required. " + "Maintain useful systems and collaborate with the engineering team. " * 8
NOTE = "I reviewed this specific vacancy and approve a CV-tailoring test only."


def draft():
    return ReferencedCV.model_validate({"sections": [
        {"heading": "Contact", "entries": [
            {"text": value, "style": "paragraph", "evidence_ids": [source_id]}
            for source_id, value in [("S0001", "Alex Example"), ("S0002", "alex@example.test")]]},
        {"heading": "Experience", "entries": [
            {"text": value, "style": "paragraph", "evidence_ids": [source_id]}
            for source_id, value in [("S0003", "Engineer | Example Ltd | 2022-present"),
                                     ("S0004", "Built Python pipelines.")]]},
        {"heading": "Education", "entries": [{
            "text": "BSc Computer Science, Example University, 2021.", "style": "paragraph",
            "evidence_ids": ["S0005"]}]},
    ]})


def resolved_draft():
    sources = {f"S{index:04d}": text for index, text in enumerate(CV.splitlines(), start=1)}
    result = draft().model_dump()
    for section in result["sections"]:
        for entry in section["entries"]:
            entry["evidence_quotes"] = [sources[source_id] for source_id in entry.pop("evidence_ids")]
    return TailoredCV.model_validate(result)


def audit():
    return CVEvaluation(requirement_coverage=95, clarity=95, factual_consistency=True,
                        unsupported_claims=[], missing_critical_information=[], improvements=[])


class Matcher:
    name = "classifier"
    fingerprint = {"version": "offline-only"}
    diagnostics = {"pairs": 0}

    def __init__(self):
        self.calls = 0

    def assess(self, *args):
        self.calls += 1
        return MatchAssessment.model_validate({
            "description_complete": True,
            "requirements": [{"requirement": "Python", "importance": "CORE",
                              "job_evidence": "Python is required.", "status": "UNKNOWN",
                              "cv_evidence": [], "explanation": "Parser needs review."}],
            "constraint_checks": [], "recommended_verdict": "REVIEW",
            "rationale": "Needs manual review.", "uncertainties": ["Parser needs review."],
        })


class Provider:
    name = "ollama"
    model = "offline-qwen"
    usage = {"requests": 0}

    def __init__(self):
        self.responses = [draft(), audit()]
        self.calls = []
        self.before_call = None

    def generate_structured(self, *, schema, **kwargs):
        self.calls.append(schema)
        if self.before_call:
            self.before_call(schema)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, schema)
        return result


def export(markdown, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "cv.pdf"
    path.write_text("Offline PDF placeholder")
    return {"pdf": str(path.resolve())}


@pytest.fixture
def case(tmp_path):
    job = {"title": "Engineer", "company": "Example", "description": DESCRIPTION,
           "source_url": "https://employer.example/jobs/one", "metadata": {"description_type": "full"}}
    profile = CandidateProfile(name="Alex Example")
    provider, matcher = Provider(), Matcher()
    ledger = ApplicationLedger(tmp_path / "applications.db")
    workflow = ApplicationWorkflow(provider, ledger, tmp_path, matcher=matcher, exporter=export, max_revisions=0)
    previous = workflow.prepare(job, profile, CV, match_only=True)
    assert previous["decision"]["verdict"] == "REVIEW"
    return SimpleNamespace(job=job, profile=profile, provider=provider, matcher=matcher,
                           ledger=ledger, workflow=workflow, previous=previous, root=tmp_path)


def run(case, **kwargs):
    return case.workflow.prepare(case.job, case.profile, CV, manual_approval_note=NOTE, **kwargs)


def legacy_review(case):
    previous = deepcopy(case.previous)
    previous.pop("source_fingerprint")
    previous["input_fingerprint"] = hashlib.sha256(json.dumps({
        "job": case.job, "cv": CV, "profile": case.profile.model_dump(),
        "match_threshold": 70, "cv_threshold": 75, "max_revisions": 0,
        "model": previous["model"], "provider": previous["provider"],
        "workflow_version": 3, "matcher": previous["matcher"],
        "matcher_config": previous["matcher_config"], "match_only": True,
    }, sort_keys=True).encode()).hexdigest()
    return previous


def test_manual_approval_preserves_classifier_and_never_becomes_ready(case):
    record = run(case)
    assert record["status"] == "review"
    assert record["decision"] == case.previous["decision"]
    assert record["match"] == case.previous["match"]
    assert record["matcher_diagnostics"] == case.previous["matcher_diagnostics"]
    assert case.matcher.calls == 1
    assert case.provider.calls == [ReferencedCV, CVEvaluation]
    approval = record["manual_approval"]
    assert approval["verdict"] == "APPLY" and approval["scope"] == "tailoring_only"
    assert approval["note"] == NOTE and approval["approved_at"]
    assert approval["original_assessment"]["decision"] == case.previous["decision"]
    assert record["requires_manual_cv_review"] and record["draft_only"]
    assert record["cv_score"] == 95 and not record["validation_errors"]
    assert record["cv_attempts"][0]["referenced_cv"] == draft().model_dump()
    assert record["cv_attempts"][0]["cv"] == resolved_draft().model_dump()
    assert json.loads(Path(record["artifacts"]["source_catalog"]).read_text()) == [
        {"source_id": f"S{index:04d}", "text": text}
        for index, text in enumerate(CV.splitlines(), start=1)
    ]
    with pytest.raises(ValueError, match="not ready"):
        case.ledger.claim_open(record["job_key"])
    with pytest.raises(ValueError, match="Only ready"):
        case.ledger.mark_submitted(record["job_key"], "Not a real receipt")
    report = json.loads(Path(record["artifacts"]["assessment"]).read_text())
    assert report["manual_approval"] == approval


@pytest.mark.parametrize("change", ["job", "cv", "profile", "legacy_job", "legacy_profile"])
def test_manual_approval_rejects_changed_sources_without_generation(case, change):
    job, profile, cv = deepcopy(case.job), case.profile.model_copy(deep=True), CV
    if change.startswith("legacy_"):
        old = legacy_review(case)
        case.ledger.save(old["job_key"], old)
    if change.endswith("job"):
        job["description"] += " Additional requirements."
    elif change.endswith("profile"):
        profile.preferences.requires_sponsorship = True
    else:
        cv += " A changed CV."
    with pytest.raises(ValueError, match="exact job, profile and CV"):
        case.workflow.prepare(job, profile, cv, manual_approval_note=NOTE)
    assert not case.provider.calls and case.matcher.calls == 1


def test_legacy_review_can_use_new_generation_settings_without_rematching(case):
    previous = legacy_review(case)
    case.ledger.save(previous["job_key"], previous)
    case.provider.model = "another-offline-model"
    case.matcher.fingerprint = {"version": "new-config"}
    record = run(case)
    assert record["status"] == "review"
    assert case.matcher.calls == 1
    assert record["model"] == "another-offline-model"
    assert record["matcher_config"] == previous["matcher_config"]


@pytest.mark.parametrize("problem", ["no_review", "skip", "incomplete", "malformed", "llm", "failed", "opened", "attempted", "submitted", "unknown", "blank", "long", "match_only"])
def test_ineligible_manual_approvals_do_not_generate(case, problem):
    previous = deepcopy(case.previous)
    note, options = NOTE, {}
    if problem == "no_review":
        case.job["source_url"] += "-different"
    elif problem == "skip":
        previous["decision"]["verdict"] = "SKIP"
    elif problem == "incomplete":
        previous["match"]["description_complete"] = False
    elif problem == "malformed":
        previous["match"] = None
    elif problem == "llm":
        case.workflow.matcher = None
    elif problem == "failed":
        previous["status"] = "failed"
    elif problem == "opened":
        previous["opened_at"] = "2026-01-01T00:00:00Z"
    elif problem == "attempted":
        previous["submission_attempted"] = True
    elif problem in {"submitted", "unknown"}:
        previous["status"] = "ready" if problem == "submitted" else "submission_unknown"
    elif problem == "blank":
        note = "  "
    elif problem == "long":
        note = "a" * 2001
    else:
        options["match_only"] = True
    case.ledger.save(previous["job_key"], previous)
    if problem == "submitted":
        case.ledger.mark_submitted(previous["job_key"], "Offline receipt")
    with pytest.raises(ValueError):
        case.workflow.prepare(case.job, case.profile, CV, manual_approval_note=note, **options)
    assert not case.provider.calls


@pytest.mark.parametrize("stage", ["generation", "evaluation", "export"])
def test_manual_failure_keeps_approval_and_any_generated_draft(case, stage):
    def verify_checkpoint(schema):
        saved = case.ledger.get(job_key(case.job))
        assert saved["manual_approval"]["scope"] == "tailoring_only"
        if schema is CVEvaluation:
            assert saved["cv_attempts"][0]["evaluation"] is None
            assert saved["cv_attempts"][0]["referenced_cv"] == draft().model_dump()
            assert saved["cv_attempts"][0]["cv"] == resolved_draft().model_dump()
            assert Path(saved["artifacts"]["draft_markdown"]).is_file()
            assert Path(saved["artifacts"]["source_catalog"]).is_file()
    case.provider.before_call = verify_checkpoint
    error = RuntimeError("private-token-must-not-leak")
    if stage == "generation":
        case.provider.responses = [error]
    elif stage == "evaluation":
        case.provider.responses = [draft(), error]
    else:
        def fail_export(*args):
            raise error
        case.workflow.exporter = fail_export
    record = run(case)
    assert record["status"] == "failed"
    assert record["manual_approval"]["original_assessment"]["decision"] == case.previous["decision"]
    assert "private-token" not in json.dumps(record)
    if stage != "generation":
        assert Path(record["artifacts"]["draft_markdown"]).read_text().startswith("# Alex Example")
        assert json.loads(Path(record["artifacts"]["draft_json"]).read_text()) == resolved_draft().model_dump()
        assert record["cv_attempts"][0]["evaluation"] == (audit().model_dump() if stage == "export" else None)
    with pytest.raises(ValueError):
        case.ledger.claim_open(record["job_key"])


def test_explicit_retry_after_manual_failure_keeps_original_approval_history(case):
    case.provider.responses = [draft(), RuntimeError("Offline evaluation failed")]
    first = run(case)
    case.provider.responses = [draft(), audit()]
    second = run(case, retry=True)
    assert second["status"] == "review"
    assert second["manual_approval_history"] == [first["manual_approval"]]
    assert second["manual_approval"]["original_assessment"] == first["manual_approval"]["original_assessment"]
    assert case.matcher.calls == 1


def test_invalid_revision_keeps_prior_draft_and_matching_reference_checkpoint(case):
    first_audit = audit().model_copy(update={
        "requirement_coverage": 40, "clarity": 30,
        "improvements": ["Make the relevant pipeline work clearer."],
    })
    invalid_revision = draft()
    invalid_revision.sections[1].entries[1].text = "A different unverified revised claim."
    invalid_revision.sections[1].entries[1].evidence_ids = ["S9999"]
    case.workflow.max_revisions = 1
    case.provider.responses = [draft(), first_audit, invalid_revision, audit()]
    original_checkpoint = {}
    export_calls = []

    def capture_prior_checkpoint(schema):
        if case.provider.calls == [ReferencedCV, CVEvaluation, ReferencedCV]:
            saved = case.ledger.get(job_key(case.job))
            original_checkpoint["attempts"] = saved["cv_attempts"]
            original_checkpoint["paths"] = {
                key: saved["artifacts"][key]
                for key in ("draft_markdown", "draft_json", "referenced_cv", "model_referenced_cv")
            }
            original_checkpoint["contents"] = {
                key: Path(path).read_text()
                for key, path in original_checkpoint["paths"].items()
            }

    def recording_export(markdown, folder):
        export_calls.append((markdown, folder))
        return export(markdown, folder)

    case.provider.before_call = capture_prior_checkpoint
    case.workflow.exporter = recording_export

    record = run(case)

    assert record["status"] == "failed"
    assert record["decision"] == case.previous["decision"]
    assert case.provider.calls == [ReferencedCV, CVEvaluation, ReferencedCV]
    assert case.provider.responses == [audit()]
    assert export_calls == [] and case.matcher.calls == 1
    assert len(record["cv_attempts"]) == 1
    assert record["cv_attempts"] == original_checkpoint["attempts"]
    assert record["cv_attempts"][0]["evaluation"] == first_audit.model_dump()
    assert record["cv_attempts"][0]["cv"] == resolved_draft().model_dump()
    for key, old_path in original_checkpoint["paths"].items():
        assert record["artifacts"][key] == old_path
        assert Path(old_path).read_text() == original_checkpoint["contents"][key]
    active_reference = Path(record["artifacts"]["referenced_cv"])
    failed_reference = Path(record["artifacts"]["failed_referenced_cv"])
    assert active_reference.name == "cv_references_1.json"
    assert failed_reference.name == "cv_model_references_2.json"
    assert json.loads(active_reference.read_text()) == draft().model_dump()
    assert json.loads(failed_reference.read_text()) == invalid_revision.model_dump()
    assert not {"pdf", "docx", "markdown"} & record["artifacts"].keys()
    assert case.ledger.get(record["job_key"])["artifacts"] == record["artifacts"]
    with pytest.raises(ValueError, match="not ready"):
        case.ledger.claim_open(record["job_key"])


@pytest.mark.parametrize("options", [
    [], ["--job", "one.json", "--job", "two.json", "--limit", "1", "--no-enrich"],
    ["--job", "one.json", "--no-enrich"], ["--job", "one.json", "--limit", "1"],
    ["--job", "one.json", "--limit", "1", "--no-enrich", "--match-only"],
    ["--url", "https://employer.example/one", "--limit", "1", "--no-enrich"],
    ["--fetch", "--limit", "1", "--no-enrich"],
    ["--job", "one.json", "--limit", "1", "--no-enrich", "--description-file", "changed.txt"],
])
def test_cli_rejects_unscoped_approval_before_cv_model_or_network(tmp_path, monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid scope must fail before reading CV or building a client")
    monkeypatch.setattr(apply_jobs, "load_profile", forbidden)
    monkeypatch.setattr(apply_jobs, "build_provider", forbidden)
    with pytest.raises(SystemExit) as error:
        apply_jobs.main(["--output", str(tmp_path / "untouched"), "prepare", "--manual-approval-note", NOTE, *options])
    assert error.value.code == 2
    assert not (tmp_path / "untouched").exists()


def test_cli_passes_single_job_manual_approval_and_never_opens_browser(case, monkeypatch):
    import dotenv
    from app.applications import browser
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(apply_jobs, "build_matcher", lambda name: case.matcher)
    monkeypatch.setattr(apply_jobs, "build_provider", lambda *args: case.provider)
    monkeypatch.setattr("app.applications.workflow.export_cv", export)
    monkeypatch.setattr(apply_jobs, "load_profile", lambda *args: (case.profile, case.root / "master.txt"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Manual tailoring must not access employer pages or browser")
    monkeypatch.setattr("app.applications.jobs.prepare_job", lambda job, **kwargs: job)
    monkeypatch.setattr(browser, "assist_application", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    cv_path = case.root / "master.txt"
    cv_path.write_text(CV)
    job_path = case.root / "job.json"
    job_path.write_text(json.dumps(case.job))
    assert apply_jobs.main([
        "--output", str(case.root), "prepare", "--provider", "ollama", "--model", case.provider.model,
        "--matcher", "classifier", "--cv", str(cv_path), "--job", str(job_path),
        "--no-enrich", "--limit", "1", "--max-revisions", "0", "--manual-approval-note", NOTE,
    ]) == 0
    record = case.ledger.get(case.previous["job_key"])
    assert record["manual_approval"]["note"] == NOTE
    assert record["status"] == "review" and record["draft_only"]
    assert case.provider.calls == [ReferencedCV, CVEvaluation]
