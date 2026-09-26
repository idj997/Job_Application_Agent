"""Offline trust-boundary tests for independent, source-linked feedback."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.applications.cv_critique import CVCritique, validate_critique
from app.applications.cv_evidence import ReferencedCV, build_cv_sources


MASTER = "Alex Example\nBuilt Python pipelines.\nIndependent C++ benchmarking, ongoing."
JOB = "Python pipelines and systems benchmarking are relevant."


def draft():
    return ReferencedCV.model_validate({"sections": [
        {"heading": "Contact", "entries": [
            {"text": "Alex Example", "style": "paragraph", "evidence_ids": ["S0001"]}]},
        {"heading": "Experience", "entries": [
            {"text": "Built Python pipelines.", "style": "bullet", "evidence_ids": ["S0002"]}]},
    ]})


def payload():
    return {
        "evaluation": {"requirement_coverage": 80, "clarity": 85, "factual_consistency": True,
                       "unsupported_claims": [], "missing_critical_information": [], "improvements": []},
        "revision_needed": True,
        "suggestions": [{"section": "Projects", "entry_index": None, "action": "add",
                         "message": "Include the independent benchmarking project with its ongoing scope.",
                         "evidence_ids": ["S0003"], "job_evidence": "systems benchmarking"}],
    }


def test_new_section_suggestion_is_source_linked_and_copied():
    critique = CVCritique.model_validate(payload())
    checked = validate_critique(critique, build_cv_sources(MASTER), draft(), JOB)
    assert checked == critique and checked is not critique


@pytest.mark.parametrize("field,value", [
    ("section", "Invented heading"), ("entry_index", 0), ("entry_index", -1),
    ("entry_index", True), ("entry_index", "1"), ("action", "apply"),
    ("message", ""), ("message", "   "), ("message", "x" * 601), ("message", 5),
    ("evidence_ids", ["S999"]), ("evidence_ids", ["S0003", "S0003"]),
    ("evidence_ids", "S0003"), ("job_evidence", ""), ("job_evidence", "  "),
])
def test_malformed_suggestions_fail_schema(field, value):
    data = payload()
    data["suggestions"][0][field] = value
    with pytest.raises(ValidationError):
        CVCritique.model_validate(data)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_revision_boolean_is_not_coerced(value):
    data = payload()
    data["revision_needed"] = value
    with pytest.raises(ValidationError):
        CVCritique.model_validate(data)


def test_suggestions_are_bounded_and_verdicts_are_forbidden():
    data = payload()
    data["suggestions"] *= 7
    with pytest.raises(ValidationError):
        CVCritique.model_validate(data)
    data = payload()
    data["verdict"] = "APPLY"
    with pytest.raises(ValidationError):
        CVCritique.model_validate(data)


@pytest.mark.parametrize("changes", [
    {"evidence_ids": ["S9999"]},
    {"action": "add", "evidence_ids": []},
    {"action": "rewrite", "evidence_ids": []},
    {"action": "restore", "evidence_ids": []},
    {"section": "Experience", "entry_index": 2},
    {"section": "Projects", "entry_index": 1},
    {"section": "overall", "entry_index": 1},
    {"job_evidence": "Systems benchmarking"},
    {"job_evidence": "No such job text"},
])
def test_unresolvable_suggestion_fails_closed(changes):
    data = payload()
    data["suggestions"][0].update(changes)
    critique = CVCritique.model_validate(data)
    with pytest.raises(ValueError):
        validate_critique(critique, build_cv_sources(MASTER), draft(), JOB)


@pytest.mark.parametrize("action", ["remove", "reorder"])
def test_presentation_actions_can_omit_evidence(action):
    data = payload()
    data["suggestions"][0].update(section="Experience", entry_index=1, action=action,
                                  evidence_ids=[], job_evidence=None)
    validate_critique(CVCritique.model_validate(data), build_cv_sources(MASTER), draft(), JOB)


def test_duplicate_sections_cannot_target_an_ambiguous_entry():
    cv = draft()
    cv.sections.append(cv.sections[-1].model_copy(deep=True))
    data = payload()
    data["suggestions"][0].update(section="Experience", entry_index=1)
    with pytest.raises(ValueError, match="unique"):
        validate_critique(CVCritique.model_validate(data), build_cv_sources(MASTER), cv, JOB)


def test_mutated_models_are_revalidated_without_echoing_input():
    critique = CVCritique.model_validate(payload())
    critique.suggestions[0].message = "SECRET " * 100
    with pytest.raises(ValueError) as exc:
        validate_critique(critique, build_cv_sources(MASTER), draft(), JOB)
    assert "SECRET" not in str(exc.value)


def test_mutated_source_catalog_and_unvalidated_dict_are_rejected():
    sources = build_cv_sources(MASTER)
    sources[1].source_id = "S0001"
    with pytest.raises(ValueError):
        validate_critique(CVCritique.model_validate(payload()), sources, draft(), JOB)
    with pytest.raises(ValueError):
        validate_critique(deepcopy(payload()), build_cv_sources(MASTER), draft(), JOB)
