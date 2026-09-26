import json

from app.applications.cv_evidence import ReferencedCV
from app.applications.models import CandidateProfile, CVEvaluation, MatchAssessment
from app.applications.tracking import ApplicationLedger
from app.applications.workflow import ApplicationWorkflow


def test_high_self_score_cannot_override_length_and_audit_receives_resolved_quotes(tmp_path):
    source_line = " ".join(["Python"] * 650)
    master = "Alex\n" + source_line
    draft = ReferencedCV.model_validate({"sections": [
        {"heading": "Contact", "entries": [{"text": "Alex", "style": "paragraph", "evidence_ids": ["S0001"]}]},
        {"heading": "Experience", "entries": [{"text": source_line, "style": "bullet", "evidence_ids": ["S0002"]}]},
    ]})
    audit = CVEvaluation(requirement_coverage=100, clarity=100, factual_consistency=True,
                         unsupported_claims=[], missing_critical_information=[], improvements=[])

    class Matcher:
        name = "classifier"
        fingerprint = {"version": "offline"}

        def assess(self, *args):
            return MatchAssessment.model_validate({
                "description_complete": True, "recommended_verdict": "APPLY", "rationale": "Supported",
                "uncertainties": [], "constraint_checks": [], "requirements": [{
                    "requirement": "Python", "importance": "CORE", "job_evidence": "Python required.",
                    "status": "MET", "cv_evidence": ["Python"], "explanation": "Source supports Python",
                }],
            })

    class Provider:
        model = "offline"
        calls = []

        def generate_structured(self, prompt, schema, system_prompt):
            self.calls.append(json.loads(prompt))
            response = draft if schema is ReferencedCV else audit
            assert isinstance(response, schema)
            return response

    def exporter(markdown, folder):
        path = folder / "cv.md"
        path.write_text(markdown)
        return {"markdown": str(path.resolve())}

    provider = Provider()
    ledger = ApplicationLedger(tmp_path / "applications.db")
    workflow = ApplicationWorkflow(provider, ledger, tmp_path, matcher=Matcher(), exporter=exporter, max_revisions=0)
    job = {"title": "Engineer", "description": "Python required. " + "Maintain reliable systems. " * 20,
           "source_url": "https://employer.example/job/length-test"}
    record = workflow.prepare(job, CandidateProfile(), master)

    assert record["decision"]["verdict"] == "APPLY"
    assert record["cv_score"] == 100
    assert record["status"] == "review"
    assert record["validation_errors"] == ["CV word count 651 exceeds the tailoring limit of 650."]
    assert provider.calls[1]["validation_errors"] == record["validation_errors"]
    assert provider.calls[1]["quality_metrics"]["word_count"] == 651
    assert provider.calls[1]["master_cv"] == master
    assert provider.calls[1]["proposed_cv"]["sections"][1]["entries"][0]["evidence_quotes"] == [source_line]
