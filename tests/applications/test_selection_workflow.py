"""Reviewed slots bound both writer modes without changing matching or sources."""
import json
from pathlib import Path

import pytest

from app.applications.cv_critique import CVCritique
from app.applications.cv_evidence import build_cv_sources
from app.applications.cv_selection import (
    SelectionPlan, prepared_job_fingerprint, source_catalog_fingerprint,
)
from app.applications.models import CandidateProfile, CVEvaluation, MatchAssessment
from app.applications.tracking import ApplicationLedger
from app.applications.workflow import ApplicationWorkflow


MASTER = "Alex Example\nEngineer | Example Ltd | 2022-present\nBuilt Python pipelines.\nUnselected testing detail."
JOB = {"title": "Engineer", "company": "Example", "source_url": "https://example.test/job/selection",
       "description": "Python required. " + "Maintain reliable systems and collaborate with the team. " * 10}


def plan(max_words=20):
    return SelectionPlan.model_validate({
        "source_catalog_sha256": source_catalog_fingerprint(build_cv_sources(MASTER)),
        "job_sha256": prepared_job_fingerprint(JOB),
        "sections": [
            {"heading": "Contact", "slots": [
                {"slot_id": "name", "evidence_ids": ["S0001"], "mode": "fixed", "style": "paragraph", "max_words": 10}]},
            {"heading": "Experience", "slots": [
                {"slot_id": "role", "evidence_ids": ["S0002"], "mode": "fixed", "style": "paragraph", "max_words": 20},
                {"slot_id": "achievement", "evidence_ids": ["S0003"], "mode": "rewrite", "style": "bullet", "max_words": max_words}]},
        ],
        "roles": [{"heading_slot_id": "role", "achievement_slot_ids": ["achievement"], "primary": True}],
        "project_heading_slot_ids": [],
    })


def evaluation():
    return CVEvaluation(requirement_coverage=90, clarity=90, factual_consistency=True,
                        unsupported_claims=[], missing_critical_information=[], improvements=[])


class Matcher:
    name = "classifier"
    fingerprint = {"offline": True}

    def __init__(self):
        self.calls = 0

    def assess(self, *args):
        self.calls += 1
        return MatchAssessment.model_validate({
            "description_complete": True, "recommended_verdict": "APPLY", "rationale": "Supported",
            "uncertainties": [], "constraint_checks": [], "requirements": [{
                "requirement": "Python", "importance": "CORE", "job_evidence": "Python required.",
                "status": "MET", "cv_evidence": ["Built Python pipelines."], "explanation": "Supported",
            }],
        })


class Provider:
    name = "ollama"

    def __init__(self, model, text="Built Python pipelines."):
        self.model = model
        self.text = text
        self.calls = []

    def generate_structured(self, *, prompt, schema, system_prompt):
        self.calls.append({"payload": json.loads(prompt), "schema": schema, "system": system_prompt})
        if schema is CVEvaluation:
            return evaluation()
        if schema is CVCritique:
            return CVCritique(evaluation=evaluation(), revision_needed=False, suggestions=[])
        return schema.model_validate({"achievement": self.text})


def exporter(markdown, directory):
    path = directory / "cv.md"
    path.write_text(markdown)
    return {"markdown": str(path.resolve())}


def workflow(tmp_path, *, separate, selection=None, text="Built Python pipelines."):
    writer = Provider("writer", text)
    critic = Provider("critic") if separate else None
    flow = ApplicationWorkflow(writer, ApplicationLedger(tmp_path / "applications.db"), tmp_path / "output",
                               matcher=Matcher(), critic_provider=critic, selection_plan=selection or plan(),
                               exporter=exporter, max_revisions=0)
    return flow, writer, critic


@pytest.mark.parametrize("separate", [False, True])
def test_both_modes_assemble_fixed_slots_and_preserve_full_catalog(tmp_path, separate):
    flow, writer, critic = workflow(tmp_path, separate=separate)
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["status"] == "ready", record["reasons"]
    assert flow.matcher.calls == 1
    request = writer.calls[0]["payload"]
    assert request["master_cv_sources"] == [s.model_dump() for s in build_cv_sources(MASTER)]
    assert request["selection_plan"] == {
        **plan().model_dump(),
        "limits": {"slot_count": 3, "rewrite_slots": 1, "fixed_slots": 2,
                   "word_budget": 28, "noncontact_entries": 2},
    }
    assert set(writer.calls[0]["schema"].model_json_schema()["properties"]) == {"achievement"}
    assert "Engineer | Example Ltd | 2022-present" in Path(record["artifacts"]["markdown"]).read_text()
    assert "Unselected testing detail" not in Path(record["artifacts"]["markdown"]).read_text()
    audit_request = critic.calls[0]["payload"] if separate else writer.calls[1]["payload"]
    entry = audit_request["proposed_cv"]["sections"][1]["entries"][1]
    assert entry.get("evidence_ids", entry.get("evidence_quotes")) == (["S0003"] if separate else ["Built Python pipelines."])
    assert record["evidence_reconciliations"] == []
    assert Path(record["artifacts"]["selection_plan"]).exists()
    assert json.loads(Path(record["artifacts"]["selection_response"]).read_text()) == {"achievement": "Built Python pipelines."}
    assert len(writer.calls) == (1 if separate else 2)


@pytest.mark.parametrize("separate", [False, True])
def test_overlong_slot_response_is_saved_but_never_audited_or_exported(tmp_path, separate):
    flow, writer, critic = workflow(tmp_path, separate=separate, text=" ".join(["Python"] * 21))
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["status"] == "failed"
    assert len(writer.calls) == 1
    assert critic is None or critic.calls == []
    assert "markdown" not in record["artifacts"]
    artifacts = record["cv_attempts"][0]["artifacts"] if separate else record["artifacts"]
    response_key = "selection_response" if separate else "failed_selection_response"
    assert json.loads(Path(artifacts[response_key]).read_text())["achievement"].split() == ["Python"] * 21
    if not separate:
        assert "selection_response" not in artifacts
    assert any("selection plan" in reason for reason in record["reasons"])


def test_rejected_self_audit_revision_keeps_accepted_response_and_references_aligned(tmp_path, monkeypatch):
    flow, writer, _ = workflow(tmp_path, separate=False)
    flow.max_revisions = 1
    drafts = 0

    def generate(*, prompt, schema, system_prompt):
        nonlocal drafts
        writer.calls.append({"payload": json.loads(prompt), "schema": schema, "system": system_prompt})
        if schema is CVEvaluation:
            return evaluation().model_copy(update={"requirement_coverage": 30, "clarity": 30})
        drafts += 1
        return schema.model_validate({
            "achievement": "Built Python pipelines." if drafts == 1 else " ".join(["Python"] * 21),
        })

    monkeypatch.setattr(writer, "generate_structured", generate)
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["status"] == "failed"
    assert len(writer.calls) == 3
    assert [call["schema"] for call in writer.calls].count(CVEvaluation) == 1
    artifacts = record["artifacts"]
    assert "markdown" not in artifacts
    assert Path(artifacts["selection_response"]).name == "selection_response_1.json"
    assert Path(artifacts["failed_selection_response"]).name == "selection_response_2.json"
    assert Path(artifacts["referenced_cv"]).name == "cv_references_1.json"
    assert Path(artifacts["model_referenced_cv"]).name == "cv_model_references_1.json"
    accepted = json.loads(Path(artifacts["selection_response"]).read_text())
    rejected = json.loads(Path(artifacts["failed_selection_response"]).read_text())
    references = json.loads(Path(artifacts["referenced_cv"]).read_text())
    assert accepted["achievement"] == references["sections"][1]["entries"][1]["text"] == "Built Python pipelines."
    assert rejected["achievement"].split() == ["Python"] * 21
    assert flow.ledger.get(record["job_key"])["artifacts"] == artifacts


@pytest.mark.parametrize("change", ["cv", "job"])
def test_stale_plan_fails_before_matching_or_writing(tmp_path, change):
    flow, writer, critic = workflow(tmp_path, separate=True)
    with pytest.raises(ValueError):
        flow.prepare({**JOB, "title": "Changed"} if change == "job" else JOB, CandidateProfile(),
                     MASTER + "\nNew fact" if change == "cv" else MASTER)
    assert flow.matcher.calls == 0 and writer.calls == [] and critic.calls == []
    assert flow.ledger.list() == [] and not (tmp_path / "output").exists()


def test_selection_cap_changes_cache_identity(tmp_path):
    first, _, _ = workflow(tmp_path, separate=True, selection=plan(20))
    original = first.prepare(JOB, CandidateProfile(), MASTER)
    same, writer, critic = workflow(tmp_path, separate=True, selection=plan(20))
    cached = same.prepare(JOB, CandidateProfile(), MASTER)
    assert cached["cached"] and writer.calls == [] and critic.calls == []
    changed, writer, _ = workflow(tmp_path, separate=True, selection=plan(19))
    updated = changed.prepare(JOB, CandidateProfile(), MASTER)
    assert len(writer.calls) == 1
    assert updated["input_fingerprint"] != original["input_fingerprint"]
    assert updated["selection_plan_sha256"] != original["selection_plan_sha256"]


def test_exact_source_reconciliation_cannot_expand_a_planned_slot(tmp_path):
    flow, _, critic = workflow(tmp_path, separate=True, text="Unselected testing detail.")
    # This tests reference assignment, not semantic approval: the mock audit is
    # intentionally permissive. Real reviewers must reject this unsupported edit.
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["evidence_reconciliations"] == []
    entry = critic.calls[0]["payload"]["proposed_cv"]["sections"][1]["entries"][1]
    assert entry["evidence_ids"] == ["S0003"]


def test_constructor_copies_reviewed_plan(tmp_path):
    original = plan()
    flow, _, _ = workflow(tmp_path, separate=True, selection=original)
    original.sections[1].slots[1].max_words = 1
    assert flow.selection_plan.sections[1].slots[1].max_words == 20


class EntryProvider(Provider):
    def __init__(self, model, texts=None):
        super().__init__(model)
        self.texts = list(texts or ["Built Python pipelines."])

    def generate_structured(self, *, prompt, schema, system_prompt):
        from app.applications.entry_writer import EntryText
        if schema is EntryText:
            self.calls.append({"payload": json.loads(prompt), "schema": schema, "system": system_prompt})
            return EntryText(text=self.texts.pop(0))
        return super().generate_structured(prompt=prompt, schema=schema, system_prompt=system_prompt)


def entry_workflow(tmp_path, separate=True, texts=None):
    writer = EntryProvider("writer", texts)
    critic = Provider("critic") if separate else None
    flow = ApplicationWorkflow(writer, ApplicationLedger(tmp_path / "applications.db"), tmp_path / "output",
                               matcher=Matcher(), critic_provider=critic, selection_plan=plan(),
                               exporter=exporter, max_revisions=0, entry_by_entry=True)
    return flow, writer, critic


@pytest.mark.parametrize("separate", [False, True])
def test_entry_writing_is_source_isolated_but_audit_is_complete(tmp_path, separate):
    flow, writer, critic = entry_workflow(tmp_path, separate)
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["status"] == "ready", record["reasons"]
    assert record["entry_by_entry"] is True and record["entry_attempt_limit"] == 2
    assert flow.matcher.calls == 1
    payload = writer.calls[0]["payload"]
    assert payload["source_lines"] == [{"source_id": "S0003", "text": "Built Python pipelines."}]
    assert "master_cv_sources" not in payload and "job" not in payload and "feedback" not in payload
    assert "Alex Example" not in json.dumps(payload) and "Unselected testing detail" not in json.dumps(payload)
    audit = critic.calls[0]["payload"] if separate else writer.calls[1]["payload"]
    if separate:
        assert audit["master_cv_sources"] == [s.model_dump() for s in build_cv_sources(MASTER)]
    else:
        assert audit["master_cv"] == MASTER
    assert record["evidence_reconciliations"] == []
    assert Path(record["artifacts"]["entry_writer_checkpoints"], "manifest.json").exists()
    assert json.loads(Path(record["artifacts"]["selection_response"]).read_text()) == {"achievement": "Built Python pipelines."}


@pytest.mark.parametrize("separate", [False, True])
def test_only_failed_entry_is_repaired_before_audit(tmp_path, separate):
    rejected_text = " ".join(["Python"] * 21)
    flow, writer, critic = entry_workflow(tmp_path, separate, [rejected_text, "Built Python pipelines."])
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["status"] == "ready", record["reasons"]
    assert len(writer.calls) == (2 if separate else 3)
    assert critic is None or len(critic.calls) == 1
    first, repair = (call["payload"] for call in writer.calls[:2])
    assert first["source_lines"] == repair["source_lines"]
    assert first["max_words"] == repair["max_words"] == 20
    assert "previous_text_unverified" not in repair
    assert rejected_text not in json.dumps(repair)
    assert {key: value for key, value in repair.items() if key != "validation_feedback"} == first
    assert repair["validation_feedback"] == {
        "accepted": False, "word_count": 21, "max_words": 20,
        "errors": ["word_budget_exceeded"],
    }
    assert "critic_feedback" not in json.dumps(repair)
    # Rejected prose remains available privately without being repeated to the
    # writer, anchoring a retry, or reaching the complete-CV audit.
    checkpoints = Path(record["artifacts"]["entry_writer_checkpoints"])
    manifest = json.loads((checkpoints / "manifest.json").read_text())
    rejected = checkpoints / manifest["slots"][0]["attempts"][0]["response_file"]
    assert json.loads(rejected.read_text())["text"] == rejected_text
    audit = critic.calls[0]["payload"] if separate else writer.calls[-1]["payload"]
    assert rejected_text not in json.dumps(audit)


@pytest.mark.parametrize("separate", [False, True])
def test_exhausted_entry_repair_never_exports_or_audits(tmp_path, separate):
    flow, writer, critic = entry_workflow(tmp_path, separate, [" ".join(["Python"] * 21)] * 2)
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["status"] == "failed" and len(writer.calls) == 2
    assert critic is None or critic.calls == []
    assert "markdown" not in record["artifacts"] and "selection_response" not in record["artifacts"]
    artifacts = record["cv_attempts"][0]["artifacts"] if separate else record["artifacts"]
    assert Path(artifacts["entry_writer_checkpoints"], "manifest.json").exists()


def test_entry_mode_has_distinct_cache_identity(tmp_path):
    old, _, _ = workflow(tmp_path, separate=True)
    old_record = old.prepare(JOB, CandidateProfile(), MASTER)
    flow, writer, _ = entry_workflow(tmp_path)
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert writer.calls and record["input_fingerprint"] != old_record["input_fingerprint"]
    cached_flow, cached_writer, _ = entry_workflow(tmp_path)
    assert cached_flow.prepare(JOB, CandidateProfile(), MASTER)["cached"]
    assert cached_writer.calls == []


@pytest.mark.parametrize("separate", [False, True])
def test_all_fixed_entry_plan_skips_drafting_but_audits_and_saves_complete_artifacts(tmp_path, separate):
    flow, writer, critic = entry_workflow(tmp_path, separate)
    flow.selection_plan.sections[1].slots[1].mode = "fixed"
    record = flow.prepare(JOB, CandidateProfile(), MASTER)
    assert record["status"] == "ready", record["reasons"]
    assert flow.matcher.calls == 1
    # The only writer request in self-audit mode is the complete-CV audit.
    assert len(writer.calls) == (0 if separate else 1)
    assert critic is None or len(critic.calls) == 1
    audit = critic.calls[0] if separate else writer.calls[0]
    assert audit["schema"] is (CVCritique if separate else CVEvaluation)
    if separate:
        assert audit["payload"]["master_cv_sources"] == [s.model_dump() for s in build_cv_sources(MASTER)]
    else:
        assert audit["payload"]["master_cv"] == MASTER
    artifacts = record["artifacts"]
    checkpoints = Path(artifacts["entry_writer_checkpoints"])
    manifest = json.loads((checkpoints / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["slots"] == []
    assert json.loads((checkpoints / "accepted_entries.json").read_text()) == {}
    assert json.loads(Path(artifacts["selection_response"]).read_text()) == {}
    assert not list(checkpoints.glob("*_input.json"))
    assert "Built Python pipelines." in Path(artifacts["markdown"]).read_text()
    assert record["evidence_reconciliations"] == []


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("prompt_name", ["ENTRY_TAILOR", "ENTRY_REPAIR", "ENTRY_GENERAL_REPAIR"])
def test_changed_entry_prompt_invalidates_only_entry_mode_cache(tmp_path, monkeypatch, separate, prompt_name):
    from app.applications import entry_writer, workflow as workflow_module

    whole_root, entry_root = tmp_path / "whole", tmp_path / "entry"
    whole_flow, _, _ = workflow(whole_root, separate=separate)
    whole_record = whole_flow.prepare(JOB, CandidateProfile(), MASTER)
    entry_flow, _, _ = entry_workflow(entry_root, separate)
    entry_record = entry_flow.prepare(JOB, CandidateProfile(), MASTER)

    # Both imports change together, just as they would after deploying a changed
    # prompt and starting a fresh process; the checkpoint and request stay aligned.
    marker = "Preserve the exact source facts for this cache probe."
    changed_prompt = getattr(entry_writer, prompt_name) + "\n" + marker
    monkeypatch.setattr(entry_writer, prompt_name, changed_prompt)
    monkeypatch.setattr(workflow_module, prompt_name, changed_prompt)
    texts = {
        "ENTRY_REPAIR": [" ".join(["Python"] * 21), "Built Python pipelines."],
        "ENTRY_GENERAL_REPAIR": ["- Built Python pipelines.", "Built Python pipelines."],
    }.get(prompt_name)
    changed_flow, changed_writer, changed_critic = entry_workflow(entry_root, separate, texts)
    changed_record = changed_flow.prepare(JOB, CandidateProfile(), MASTER)
    assert changed_record["input_fingerprint"] != entry_record["input_fingerprint"]
    prompt_index = 0 if prompt_name == "ENTRY_TAILOR" else 1
    assert changed_writer.calls and marker in changed_writer.calls[prompt_index]["system"]
    if prompt_name != "ENTRY_TAILOR":
        assert marker not in changed_writer.calls[0]["system"]
    assert changed_critic is None or len(changed_critic.calls) == 1

    cached_entry, cached_writer, cached_critic = entry_workflow(entry_root, separate)
    assert cached_entry.prepare(JOB, CandidateProfile(), MASTER)["cached"]
    assert cached_writer.calls == [] and (cached_critic is None or cached_critic.calls == [])
    cached_whole, whole_writer, whole_critic = workflow(whole_root, separate=separate)
    cached_record = cached_whole.prepare(JOB, CandidateProfile(), MASTER)
    assert cached_record["cached"] and cached_record["input_fingerprint"] == whole_record["input_fingerprint"]
    assert whole_writer.calls == [] and (whole_critic is None or whole_critic.calls == [])


@pytest.mark.parametrize("with_plan,revisions", [(False, 0), (True, 1), (True, 2)])
def test_entry_mode_rejects_missing_plan_or_critic_revision_budget(tmp_path, with_plan, revisions):
    with pytest.raises(ValueError, match="Entry-by-entry"):
        ApplicationWorkflow(Provider("writer"), ApplicationLedger(tmp_path / "applications.db"), tmp_path,
                            entry_by_entry=True, selection_plan=plan() if with_plan else None, max_revisions=revisions)


def test_entry_mode_revalidates_changed_revision_budget(tmp_path):
    flow, writer, critic = entry_workflow(tmp_path)
    flow.max_revisions = 1
    with pytest.raises(ValueError, match="Entry-by-entry"):
        flow.prepare(JOB, CandidateProfile(), MASTER)
    assert writer.calls == [] and critic.calls == [] and flow.ledger.list() == []
