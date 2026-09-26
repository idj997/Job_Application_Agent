"""No inference: prove writer/critic separation, revision bounds and selection."""
import json
from pathlib import Path

import pytest

from app.applications.cv_critique import CVCritique
from app.applications.cv_evidence import ReferencedCV, build_cv_sources
from app.applications.models import CandidateProfile, MatchAssessment
from app.applications.tracking import ApplicationLedger
from app.applications.workflow import ApplicationWorkflow
from app.providers.base import ModelProviderError


MASTER = "Alex Example\nEngineer | Example Ltd | 2022-present\nBuilt Python pipelines.\nIndependent C++ benchmarking, ongoing."
JOB = {"title": "Engineer", "company": "Example", "source_url": "https://example.test/job/critic",
       "description": "Python required. " + "Maintain reliable systems and collaborate with the team. " * 10}
PROFILE = CandidateProfile(name="Alex Example")


def draft(*, revision=False):
    sections = [
        {"heading": "Contact", "entries": [
            {"text": "Alex Example", "style": "paragraph", "evidence_ids": ["S0001"]}]},
        {"heading": "Experience", "entries": [
            {"text": "Engineer | Example Ltd | 2022-present", "style": "paragraph", "evidence_ids": ["S0002"]},
            {"text": "Built Python pipelines.", "style": "bullet", "evidence_ids": ["S0003"]}]},
    ]
    if revision:
        sections.append({"heading": "Projects", "entries": [
            {"text": "Independent C++ benchmarking, ongoing.", "style": "paragraph", "evidence_ids": ["S0004"]}]})
    return ReferencedCV.model_validate({"sections": sections})


def critique(score=90, revision=False, **evaluation_changes):
    evaluation = {"requirement_coverage": score, "clarity": score, "factual_consistency": True,
                  "unsupported_claims": [], "missing_critical_information": [], "improvements": []}
    evaluation.update(evaluation_changes)
    return CVCritique.model_validate({
        "evaluation": evaluation, "revision_needed": revision,
        "suggestions": ([{"section": "Projects", "entry_index": None, "action": "add",
                          "message": "Add the supplied independent benchmarking project, preserving its scope.",
                          "evidence_ids": ["S0004"], "job_evidence": None}] if revision else []),
    })


class Provider:
    name = "ollama"

    def __init__(self, model, responses):
        self.model = model
        self.responses = list(responses)
        self.calls = []
        self.fingerprint = {"model": model, "num_ctx": 12000}
        self.model_digest = None

    def generate_structured(self, *, prompt, schema, system_prompt):
        self.calls.append({"payload": json.loads(prompt), "schema": schema,
                           "system": system_prompt, "raw_prompt": prompt})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class Matcher:
    name = "classifier"
    fingerprint = {"offline": True}

    def __init__(self, status="MET"):
        self.calls = 0
        self.status = status

    def assess(self, *args):
        self.calls += 1
        return MatchAssessment.model_validate({
            "description_complete": True, "recommended_verdict": "APPLY", "rationale": "Source supported",
            "uncertainties": [], "constraint_checks": [], "requirements": [{
                "requirement": "Python", "importance": "CORE", "job_evidence": "Python required.",
                "status": self.status, "cv_evidence": ["Built Python pipelines."] if self.status == "MET" else [],
                "explanation": "Compared with source",
            }],
        })


def exporter(markdown, directory):
    path = Path(directory) / "cv.md"
    path.write_text(markdown)
    return {"markdown": str(path.resolve())}


def workflow(tmp_path, drafts=None, critiques=None, **kwargs):
    writer = Provider("qwen3:8b", drafts if drafts is not None else [draft()])
    critic = Provider("gemma3:4b", critiques if critiques is not None else [critique()])
    ledger = ApplicationLedger(tmp_path / "applications.db")
    matcher = kwargs.pop("matcher", Matcher())
    result = ApplicationWorkflow(writer, ledger, tmp_path / "output", matcher=matcher,
                                 exporter=kwargs.pop("exporter", exporter), critic_provider=critic, **kwargs)
    return result, writer, critic


def test_critic_replaces_self_audit_and_gets_complete_unscored_catalog(tmp_path):
    flow, writer, critic = workflow(tmp_path)
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "ready"
    assert [call["schema"] for call in writer.calls] == [ReferencedCV]
    assert [call["schema"] for call in critic.calls] == [CVCritique]
    payload = critic.calls[0]["payload"]
    assert payload["master_cv_sources"] == [item.model_dump() for item in build_cv_sources(MASTER)]
    assert payload["job"]["description"] == JOB["description"]
    assert payload["proposed_cv"] == draft().model_dump()
    assert set(payload) == {"master_cv_sources", "job", "proposed_cv", "tailoring_policy", "validation_errors", "quality_metrics"}
    assert "master_cv" not in payload and "evaluation" not in payload
    assert record["critic_model"] == "gemma3:4b" and record["model"] == "qwen3:8b"
    assert record["decision"]["verdict"] == "APPLY" and flow.matcher.calls == 1
    assert record["selected_attempt"] == 1
    assert json.loads(Path(record["artifacts"]["critique"]).read_text()) == record["critique"]


def test_substantive_revision_happens_even_above_score_threshold(tmp_path):
    flow, writer, critic = workflow(tmp_path, [draft(), draft(revision=True)], [critique(90, True), critique(95)])
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "ready" and record["cv_score"] == 95
    assert record["selected_attempt"] == 2 and len(writer.calls) == len(critic.calls) == 2
    feedback = writer.calls[1]["payload"]["feedback"]
    assert feedback["critic_feedback"] == critique(90, True).model_dump()
    assert feedback["previous_cv"] == draft().model_dump()
    assert json.loads(Path(record["cv_attempts"][0]["artifacts"]["writer_feedback"]).read_text()) is None
    assert json.loads(Path(record["cv_attempts"][1]["artifacts"]["writer_feedback"]).read_text()) == feedback
    assert "Independent C++" in Path(record["artifacts"]["markdown"]).read_text()
    assert all("attempt_02" in record["artifacts"][name] for name in (
        "markdown", "referenced_cv", "model_referenced_cv", "draft_json", "critique", "evaluation", "writer_feedback"))
    assert Path(record["cv_attempts"][0]["artifacts"]["draft_markdown"]).exists()


def test_writer_and_critic_json_is_compact_without_changing_source_text(tmp_path):
    flow, writer, critic = workflow(tmp_path, [draft(), draft(revision=True)],
                                   [critique(90, True), critique(95)])
    job = {**JOB, "description": JOB["description"] + "Preserve  two spaces, commas, colons: and Unicode – exactly."}
    record = flow.prepare(job, PROFILE, MASTER)
    assert record["status"] == "ready"
    for call in writer.calls + critic.calls:
        assert call["raw_prompt"] == json.dumps(call["payload"], ensure_ascii=False, separators=(",", ":"))
        assert call["payload"]["job"]["description"] == job["description"]
        assert [item["text"] for item in call["payload"]["master_cv_sources"]] == MASTER.splitlines()


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_revision_budget_is_bounded_and_unresolved_feedback_blocks_ready(tmp_path, budget):
    flow, writer, critic = workflow(tmp_path, [draft()] * (budget + 1),
                                   [critique(95, True)] * (budget + 1), max_revisions=budget)
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert len(writer.calls) == len(critic.calls) == budget + 1
    assert record["status"] == "review" and record["critic_revision_pending"]


def test_later_lower_score_preserves_best_draft_and_matching_artifacts(tmp_path):
    flow, writer, critic = workflow(tmp_path, [draft(), draft(revision=True)], [critique(95, True), critique(85)])
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["selected_attempt"] == 1 and record["cv_score"] == 95
    assert record["status"] == "review"  # The retained draft still has a substantive suggestion.
    assert "Independent C++" not in Path(record["artifacts"]["markdown"]).read_text()
    assert all("attempt_01" in record["artifacts"][name] for name in (
        "markdown", "referenced_cv", "draft_json", "critique", "evaluation"))
    assert len(record["cv_attempts"]) == 2


@pytest.mark.parametrize("problem", ["factual", "unsupported", "missing", "length"])
def test_higher_score_invalid_revision_never_replaces_valid_draft(tmp_path, problem):
    second = draft(revision=True)
    changes = {}
    if problem == "factual":
        changes["factual_consistency"] = False
    elif problem == "unsupported":
        changes["unsupported_claims"] = ["Unsupported professional C++ scope."]
    elif problem == "missing":
        changes["missing_critical_information"] = ["Employment chronology missing."]
    else:
        second.sections[1].entries[-1].text = "Python " * 660
    flow, _, _ = workflow(tmp_path, [draft(), second], [critique(85, True), critique(100, **changes)])
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["selected_attempt"] == 1 and record["cv_score"] == 85
    assert record["status"] == "review"


def test_all_invalid_audits_remain_review_and_cannot_be_opened(tmp_path):
    flow, _, _ = workflow(tmp_path, critiques=[critique(100, unsupported_claims=["Unsupported claim."])], max_revisions=0)
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "review" and record["cv_score"] == 100
    with pytest.raises(ValueError, match="not ready"):
        flow.ledger.claim_open(record["job_key"])


@pytest.mark.parametrize("stage", ["writer", "critic", "references", "feedback"])
def test_later_failures_preserve_earlier_audited_draft_as_review(tmp_path, stage):
    second = draft(revision=True)
    reviews = [critique(90, True), critique(99)]
    if stage == "writer":
        second = ModelProviderError("Writer unavailable.")
    elif stage == "critic":
        reviews[-1] = ModelProviderError("Critic unavailable.")
    elif stage == "references":
        second.sections[1].entries[0].evidence_ids = ["S9999"]
    else:
        reviews[-1] = critique(99, True)
        reviews[-1].suggestions[0].evidence_ids = ["S9999"]
    flow, writer, critic = workflow(tmp_path, [draft(), second], reviews)
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "review" and record["selected_attempt"] == 1
    assert record["cv_score"] == 90 and record["critic_run_failure"]
    assert Path(record["artifacts"]["markdown"]).exists()
    assert record["cv_attempts"][1]["failure"]
    saved_feedback = Path(record["cv_attempts"][1]["artifacts"]["writer_feedback"])
    assert json.loads(saved_feedback.read_text()) == writer.calls[1]["payload"]["feedback"]
    assert len(writer.calls) == 2 and len(critic.calls) <= 2
    with pytest.raises(ValueError, match="not ready"):
        flow.ledger.claim_open(record["job_key"])


def test_invalid_first_critique_fails_without_revision_or_export(tmp_path):
    invalid = critique(95, True)
    invalid.suggestions[0].evidence_ids = ["S9999"]
    flow, writer, critic = workflow(tmp_path, critiques=[invalid])
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "failed"
    assert len(writer.calls) == len(critic.calls) == 1
    assert "selected_attempt" not in record and "markdown" not in record["artifacts"]
    assert Path(record["artifacts"]["unaudited_draft_markdown"]).exists()
    assert record["cv_attempts"][0]["evaluation"] is None
    assert Path(record["cv_attempts"][0]["artifacts"]["raw_critique"]).exists()
    assert json.loads(Path(record["cv_attempts"][0]["artifacts"]["writer_feedback"]).read_text()) is None


def test_writer_and_critic_failures_do_not_fall_back(tmp_path):
    flow, writer, critic = workflow(tmp_path, critiques=[ModelProviderError("Critic offline.")])
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "failed" and len(writer.calls) == len(critic.calls) == 1
    assert record["critic_run_failure"]["message"] == "Critic offline."


def test_identical_instances_or_names_are_rejected_before_calls(tmp_path):
    writer = Provider("qwen3:8b", [])
    for critic in (writer, Provider("qwen3:8b", [])):
        with pytest.raises(ValueError, match="different model"):
            ApplicationWorkflow(writer, ApplicationLedger(tmp_path / "ledger.db"), tmp_path, critic_provider=critic)
    assert not writer.calls


def test_different_tags_with_identical_weight_digests_fail_closed(tmp_path):
    flow, writer, critic = workflow(tmp_path)
    writer.model_digest = critic.model_digest = "same-installed-model-digest"
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "failed" and "same model weights" in record["critic_run_failure"]["message"]
    assert record["cv_attempts"][0]["evaluation"] is None


def test_cache_includes_critic_runtime_and_excludes_lazy_digest(tmp_path):
    flow, writer, critic = workflow(tmp_path, [draft(), draft()], [critique(), critique()])
    first = flow.prepare(JOB, PROFILE, MASTER)
    critic.model_digest = "newly-known-real-digest"
    repeated = flow.prepare(JOB, PROFILE, MASTER)
    assert repeated["cached"] and repeated["input_fingerprint"] == first["input_fingerprint"]
    critic.fingerprint = {**critic.fingerprint, "num_ctx": 16000}
    changed = flow.prepare(JOB, PROFILE, MASTER)
    assert not changed.get("cached") and changed["input_fingerprint"] != first["input_fingerprint"]
    assert len(writer.calls) == len(critic.calls) == 2


def test_critic_model_change_invalidates_cache(tmp_path):
    flow, _, critic = workflow(tmp_path, [draft(), draft()], [critique(), critique()])
    first = flow.prepare(JOB, PROFILE, MASTER)
    critic.model = "another-small-model"
    second = flow.prepare(JOB, PROFILE, MASTER)
    assert not second.get("cached") and first["input_fingerprint"] != second["input_fingerprint"]


def test_manual_approval_keeps_classifier_review_and_draft_only(tmp_path):
    flow, writer, critic = workflow(tmp_path, matcher=Matcher("UNKNOWN"))
    previous = flow.prepare(JOB, PROFILE, MASTER, match_only=True)
    assert writer.calls == critic.calls == []
    record = flow.prepare(JOB, PROFILE, MASTER, manual_approval_note="Approve this tailoring test only.")
    assert record["status"] == "review" and record["draft_only"] and record["requires_manual_cv_review"]
    assert record["match"] == previous["match"] and record["decision"] == previous["decision"]
    assert flow.matcher.calls == 1
    with pytest.raises(ValueError, match="not ready"):
        flow.ledger.claim_open(record["job_key"])


def test_skipped_and_match_only_jobs_never_call_writer_or_critic(tmp_path):
    flow, writer, critic = workflow(tmp_path, drafts=[], critiques=[], matcher=Matcher("UNKNOWN"))
    assert flow.prepare(JOB, PROFILE, MASTER)["status"] == "review"
    assert flow.prepare(JOB, PROFILE, MASTER, match_only=True)["status"] == "review"
    assert not writer.calls and not critic.calls


def test_submitted_record_is_not_reprepared(tmp_path):
    flow, writer, critic = workflow(tmp_path)
    first = flow.prepare(JOB, PROFILE, MASTER)
    flow.ledger.mark_submitted(first["job_key"], "Test receipt")
    second = flow.prepare(JOB, PROFILE, MASTER, retry=True)
    assert second["status"] == "submitted" and second["cached"]
    assert len(writer.calls) == len(critic.calls) == 1


def test_export_failure_keeps_valid_draft_unapproved(tmp_path):
    def fail_export(*args):
        raise RuntimeError("PRIVATE SOURCE CONTENT")

    flow, _, _ = workflow(tmp_path, exporter=fail_export)
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["status"] == "review" and record["critic_run_failure"]["stage"] == "export"
    assert "PRIVATE SOURCE" not in json.dumps(record)
    assert Path(record["artifacts"]["draft_markdown"]).exists()


def test_same_score_prefers_revision_with_resolved_feedback(tmp_path):
    flow, _, _ = workflow(tmp_path, [draft(), draft(revision=True)], [critique(90, True), critique(90)])
    record = flow.prepare(JOB, PROFILE, MASTER)
    assert record["selected_attempt"] == 2 and record["status"] == "ready"
