"""Reviewed source selection constrains structure, never proves claim truth."""
from copy import deepcopy
import hashlib
import json

import pytest
from pydantic import ValidationError

from app.applications.cv_evidence import CVSource, build_cv_sources, resolve_cv_references
from app.applications.cv_quality import cv_quality_metrics
from app.applications.cv_selection import (
    SELECTED_TAILOR, SelectionPlan, compile_selection_plan, contains_required_span, plan_fingerprint,
    prepared_job_fingerprint, source_catalog_fingerprint,
)


JOB = {"title": "CPU engineer", "description": "Measure power and validate systems.",
       "source_url": "https://employer.example/job/1", "metadata": {"remote": False}}
SOURCES = build_cv_sources(
    "Alex Example\nalex@example.test\nCPU Engineer | Example Devices\n"
    "Cambridge | Oct 2024–Present\nExtended an existing framework.\n"
    "Achieved a 6–10% power benefit without compromising performance.\n"
    "QA Engineer | Earlier Employer\nLondon | 2020–2024\nAutomated tests.\n"
    "MSc Computer Science\nExample University | 2020\nPython and C++\n"
    "Cache benchmark\nIndependent project | 2026 | Ongoing\nCompared memory access patterns."
)


def slot(identifier, ids, *, mode="rewrite", style="bullet", words=30):
    return {"slot_id": identifier, "evidence_ids": ids, "mode": mode,
            "style": style, "max_words": words}


def plan_dict():
    return {
        "source_catalog_sha256": source_catalog_fingerprint(SOURCES),
        "job_sha256": prepared_job_fingerprint(JOB),
        "sections": [
            {"heading": "Contact", "slots": [
                slot("name", ["S0001"], mode="fixed", style="paragraph", words=2),
                slot("email", ["S0002"], mode="fixed", style="paragraph", words=1),
            ]},
            {"heading": "Professional Summary", "slots": [
                slot("summary", ["S0003", "S0005"], style="paragraph", words=40),
            ]},
            {"heading": "Experience", "slots": [
                slot("current_heading", ["S0003", "S0004"], mode="fixed", style="paragraph"),
                slot("current_achievement", ["S0005", "S0006"]),
                slot("earlier_heading", ["S0007", "S0008"], mode="fixed", style="paragraph"),
                slot("earlier_achievement", ["S0009"]),
            ]},
            {"heading": "Education", "slots": [
                slot("degree", ["S0010", "S0011"], mode="fixed", style="paragraph"),
            ]},
            {"heading": "Skills", "slots": [
                slot("skills", ["S0012"], style="paragraph", words=10),
            ]},
            {"heading": "Projects", "slots": [
                slot("project_heading", ["S0013", "S0014"], mode="fixed", style="paragraph"),
                slot("project_achievement", ["S0015"], words=20),
            ]},
        ],
        "roles": [
            {"heading_slot_id": "current_heading", "achievement_slot_ids": ["current_achievement"], "primary": True},
            {"heading_slot_id": "earlier_heading", "achievement_slot_ids": ["earlier_achievement"], "primary": False},
        ],
        "project_heading_slot_ids": ["project_heading"],
    }


def compile_plan(payload=None):
    return compile_selection_plan(SelectionPlan.model_validate(payload or plan_dict()), SOURCES, JOB)


def valid_response():
    return {
        "summary": "CPU engineer extending existing frameworks.",
        "current_achievement": "Extended an existing framework; achieved a 6–10% power benefit without compromising performance.",
        "earlier_achievement": "Automated tests.", "skills": "Python and C++",
        "project_achievement": "Compared memory access patterns.",
    }


def test_exact_source_and_job_hashes_use_full_ordered_json_inputs():
    expected = hashlib.sha256(json.dumps([s.model_dump() for s in SOURCES], sort_keys=True).encode()).hexdigest()
    assert source_catalog_fingerprint(SOURCES) == expected
    assert source_catalog_fingerprint(list(reversed(SOURCES))) != expected
    changed_job = deepcopy(JOB)
    changed_job["metadata"]["remote"] = True
    assert prepared_job_fingerprint(changed_job) != prepared_job_fingerprint(JOB)
    assert prepared_job_fingerprint(dict(reversed(list(JOB.items())))) == prepared_job_fingerprint(JOB)


def test_compile_and_assemble_preserve_fixed_facts_references_and_order():
    compiled = compile_plan()
    response = compiled.response_schema.model_validate(valid_response())
    cv = compiled.assemble(response)
    assert [s.heading for s in cv.sections] == [s["heading"] for s in plan_dict()["sections"]]
    current = cv.sections[2].entries[0]
    assert current.text == "CPU Engineer | Example Devices | Cambridge | Oct 2024–Present"
    assert current.evidence_ids == ["S0003", "S0004"]
    assert current.style == "paragraph"
    assert cv.sections[5].entries[0].text == "Cache benchmark | Independent project | 2026 | Ongoing"
    assert cv.sections[0].entries[0].text == "Alex Example"
    assert len(cv.sections[2].entries) == 4
    assert cv_quality_metrics(resolve_cv_references(cv, SOURCES))["word_count"] <= compiled.summary["word_budget"]
    assert compiled.instructions == SELECTED_TAILOR
    assert compiled.summary == {"slot_count": 11, "rewrite_slots": 5, "fixed_slots": 6,
                                "word_budget": 169, "noncontact_entries": 9}


def test_schema_only_requests_required_rewrite_strings_and_forbids_other_fields():
    schema = compile_plan().response_schema.model_json_schema()
    assert set(schema["properties"]) == set(valid_response())
    assert set(schema["required"]) == set(valid_response())
    assert schema["additionalProperties"] is False
    assert all(value["type"] == "string" for value in schema["properties"].values())
    assert "evidence_ids" not in schema["properties"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("summary"),
    lambda r: r.update(extra="Added claim"),
    lambda r: r.update(name="Other person"),
    lambda r: r.update(summary={"text": "Claim", "evidence_ids": ["S0006"]}),
    lambda r: r.update(summary=42),
    lambda r: r.update(summary=None),
    lambda r: r.update(summary=""),
    lambda r: r.update(field_0000="Internal alias bypass"),
])
def test_invalid_or_changed_response_shape_is_rejected(mutation):
    response = valid_response()
    mutation(response)
    with pytest.raises(ValueError, match="missing, extra or invalid"):
        compile_plan().assemble(response)


@pytest.mark.parametrize("bad", [" ", "\t", "line\nline", "line\rline", "hidden\x00control", "hidden\x7fcontrol"])
def test_response_text_is_never_silently_cleaned(bad):
    response = valid_response()
    response["summary"] = bad
    with pytest.raises(ValueError, match="plain-text"):
        compile_plan().assemble(response)


def test_outer_whitespace_is_rejected_not_silently_trimmed():
    response = valid_response()
    response["summary"] = "  CPU engineer  "
    with pytest.raises(ValueError, match="outer whitespace"):
        compile_plan().assemble(response)
    assert response["summary"] == "  CPU engineer  "


def test_over_budget_response_is_rejected_not_truncated():
    response = valid_response()
    response["summary"] = " ".join(["word"] * 41)
    with pytest.raises(ValueError, match="summary exceeds its 40-word budget"):
        compile_plan().assemble(response)
    assert len(response["summary"].split()) == 41


def test_exact_budget_response_is_allowed():
    response = valid_response()
    response["summary"] = " ".join(["word"] * 40)
    assert len(compile_plan().assemble(response).sections[1].entries[0].text.split()) == 40


def test_mutated_response_model_is_revalidated():
    compiled = compile_plan()
    response = compiled.response_schema.model_validate(valid_response())
    response.field_0000 = 123
    with pytest.raises(ValueError, match="invalid rewrite"):
        compiled.assemble(response)


def test_payload_summary_and_caller_input_mutations_do_not_change_compiled_contract():
    plan = SelectionPlan.model_validate(plan_dict())
    sources = deepcopy(SOURCES)
    compiled = compile_selection_plan(plan, sources, JOB)
    plan.sections[1].slots[0].max_words = 650
    sources[0].text = "New person"
    compiled.payload["sections"][1]["slots"][0]["max_words"] = 650
    compiled.summary["word_budget"] = 9999
    assert compiled.assemble(valid_response()).sections[0].entries[0].text == "Alex Example"
    assert compiled.payload["sections"][1]["slots"][0]["max_words"] == 40
    assert compiled.summary["word_budget"] < 650


@pytest.mark.parametrize("key", ["source_catalog_sha256", "job_sha256"])
def test_stale_bindings_fail_before_any_drafting(key):
    data = plan_dict()
    data[key] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        compile_plan(data)


@pytest.mark.parametrize("changes", [
    {"max_words": True}, {"max_words": "20"}, {"max_words": 0}, {"max_words": 651},
    {"mode": "generate"}, {"style": "markdown"}, {"evidence_ids": []},
    {"slot_id": "UPPER"}, {"slot_id": "padded "}, {"slot_id": "bad\n"},
    {"fixed_text": "Private invented fact"},
])
def test_plan_slot_types_and_fields_are_strict(changes):
    data = plan_dict()
    data["sections"][1]["slots"][0].update(changes)
    with pytest.raises(ValidationError) as exc:
        SelectionPlan.model_validate(data)
    assert "Private invented fact" not in str(exc.value)


@pytest.mark.parametrize("mutation,problem", [
    (lambda p: p["sections"].reverse(), "Contact first"),
    (lambda p: p["sections"].append(deepcopy(p["sections"][1])), "unique section"),
    (lambda p: p["sections"][1]["slots"][0].update(slot_id="name"), "duplicate slot"),
    (lambda p: p["sections"][1]["slots"][0].update(evidence_ids=["S9999"]), "unknown source"),
    (lambda p: p["sections"][1]["slots"][0].update(evidence_ids=["S0001", "S0001"]), "duplicate evidence"),
    (lambda p: p["sections"][0]["slots"][0].update(mode="rewrite"), "contact and education"),
    (lambda p: p["sections"][3]["slots"][0].update(mode="rewrite"), "contact and education"),
    (lambda p: p["sections"][2]["slots"][0].update(mode="rewrite"), "employment headings"),
    (lambda p: p["sections"][2]["slots"][1].update(style="paragraph"), "explicit bullet"),
    (lambda p: p["roles"].clear(), "Every Experience slot"),
    (lambda p: p["roles"].reverse(), "ordered, contiguous"),
    (lambda p: p["roles"][0]["achievement_slot_ids"].clear(), "Every Experience slot"),
    (lambda p: p["roles"][0]["achievement_slot_ids"].append("current_achievement"), "exactly one"),
    (lambda p: p["roles"][0].update(primary=False), "exactly one explicitly primary"),
    (lambda p: p["roles"][1].update(primary=True), "exactly one explicitly primary"),
    (lambda p: p["roles"][0].update(heading_slot_id="unknown"), "unknown slot"),
    (lambda p: p["project_heading_slot_ids"].clear(), "project heading slots"),
    (lambda p: p["sections"][5]["slots"][0].update(mode="rewrite"), "date/status headings"),
    (lambda p: p["sections"][0]["slots"][0].update(max_words=1), "fixed source join"),
    (lambda p: p["sections"][1]["slots"][0].update(max_words=71), "70-word"),
])
def test_invalid_plan_structure_fails_closed(mutation, problem):
    data = plan_dict()
    mutation(data)
    with pytest.raises(ValueError, match=problem):
        compile_plan(data)


@pytest.mark.parametrize("primary,count", [(True, 9), (False, 4)])
def test_per_role_caps_use_explicit_role_metadata(primary, count):
    data = plan_dict()
    role_index = 0 if primary else 1
    role = data["roles"][role_index]
    existing = role["achievement_slot_ids"][0]
    added = [slot(f"added_{i}", ["S0005"], words=1) for i in range(count - 1)]
    position = 2 if primary else 4
    data["sections"][2]["slots"][position:position] = added
    role["achievement_slot_ids"] = [existing, *[s["slot_id"] for s in added]]
    with pytest.raises(ValueError, match="role bullet limit"):
        compile_plan(data)


def test_total_budgets_not_only_actual_current_response_must_fit():
    data = plan_dict()
    data["sections"][2]["slots"][1]["max_words"] = 650
    with pytest.raises(ValueError, match="fixed words plus rewrite budgets"):
        compile_plan(data)


def test_skills_and_total_entry_caps_apply_across_plan():
    data = plan_dict()
    data["sections"][4]["slots"] = [slot(f"skill_{i}", ["S0012"], words=1) for i in range(6)]
    with pytest.raises(ValueError, match="five Skills"):
        compile_plan(data)
    data = plan_dict()
    data["sections"].append({"heading": "Certifications", "slots": [
        slot(f"cert_{i}", ["S0012"], words=1) for i in range(24)
    ]})
    with pytest.raises(ValueError, match="32 non-contact"):
        compile_plan(data)


def test_all_fixed_plan_has_empty_strict_response_and_no_required_optional_sections():
    data = plan_dict()
    data.update(sections=[data["sections"][0]], roles=[], project_heading_slot_ids=[])
    compiled = compile_plan(data)
    assert compiled.response_schema.model_json_schema()["properties"] == {}
    assert compiled.assemble({}).sections[0].entries[0].text == "Alex Example"


def test_slot_ids_cannot_shadow_response_model_methods_or_bypass_aliases():
    data = plan_dict()
    data["sections"][1]["slots"][0]["slot_id"] = "model_dump"
    compiled = compile_plan(data)
    response = valid_response()
    response["model_dump"] = response.pop("summary")
    instance = compiled.response_schema.model_validate(response)
    assert callable(instance.model_dump)
    assert compiled.assemble(instance).sections[1].entries[0].text == response["model_dump"]


@pytest.mark.parametrize("mutation", [
    lambda p: setattr(p.sections[1].slots[0], "max_words", True),
    lambda p: setattr(p, "sections", []),
])
def test_mutated_plan_is_revalidated_and_safe_errors_hide_values(mutation):
    plan = SelectionPlan.model_validate(plan_dict())
    mutation(plan)
    with pytest.raises(ValueError, match="does not satisfy"):
        compile_selection_plan(plan, SOURCES, JOB)


@pytest.mark.parametrize("sources", [None, [], ["Private source"], [CVSource.model_construct(source_id="bad", text="Private source")]])
def test_invalid_source_catalog_does_not_expose_private_text(sources):
    with pytest.raises(ValueError) as exc:
        source_catalog_fingerprint(sources)
    assert "Private source" not in str(exc.value)


def test_duplicate_catalog_ids_fail_even_if_text_matches():
    with pytest.raises(ValueError, match="duplicate IDs"):
        source_catalog_fingerprint(SOURCES + [SOURCES[0]])


@pytest.mark.parametrize("job", [None, [], {}, {"x": float("nan")}, {"x": object()}])
def test_invalid_job_fingerprint_input_fails_closed(job):
    with pytest.raises(ValueError):
        prepared_job_fingerprint(job)


def test_every_plan_change_affects_fingerprint():
    plan = SelectionPlan.model_validate(plan_dict())
    original = plan_fingerprint(plan)
    plan.sections[1].slots[0].max_words -= 1
    assert plan_fingerprint(plan) != original


def test_valid_selection_is_not_semantic_proof():
    response = valid_response()
    response["summary"] = "Invented expertise."
    cv = compile_plan().assemble(response)
    assert cv.sections[1].entries[0].text == "Invented expertise."
    # Source quotes are still supplied for the independent factual audit.
    assert resolve_cv_references(cv, SOURCES).sections[1].entries[0].evidence_quotes == [
        SOURCES[2].text, SOURCES[4].text,
    ]


def test_required_spans_default_empty_without_changing_rewrite_task_shape():
    plan = SelectionPlan.model_validate(plan_dict())
    assert all(slot.required_spans == [] for section in plan.sections for slot in section.slots)
    assert all("required_source_spans" not in task for task in compile_plan().rewrite_tasks())


def test_required_spans_are_frozen_source_scoped_and_exposed_only_when_present():
    data = plan_dict()
    required = [{"source_id": "S0003", "text": "CPU Engineer"},
                {"source_id": "S0005", "text": "existing framework"}]
    data["sections"][1]["slots"][0]["required_spans"] = deepcopy(required)
    plan = SelectionPlan.model_validate(data)
    sources = deepcopy(SOURCES)
    compiled = compile_selection_plan(plan, sources, JOB)
    task = compiled.rewrite_tasks()[0]
    assert task["required_source_spans"] == required
    assert all("required_source_spans" not in other for other in compiled.rewrite_tasks()[1:])
    plan.sections[1].slots[0].required_spans[0].text = "Changed title"
    sources[2].text = "Changed source"
    task["required_source_spans"][0]["text"] = "Changed task"
    compiled.payload["sections"][1]["slots"][0]["required_spans"].clear()
    assert compiled.rewrite_tasks()[0]["required_source_spans"] == required
    response = valid_response()
    response["summary"] = "CPU engineer extending an existing framework."
    assert compiled.assemble(response).sections[1].entries[0].text == response["summary"]
    response["summary"] = "CPU engineer."
    with pytest.raises(ValueError, match="required source span"):
        compiled.assemble(response)


@pytest.mark.parametrize("span", [
    {"source_id": "S0003", "text": 3}, {"source_id": "S0003", "text": None},
    {"source_id": "S0003", "text": True}, {"source_id": "wrong", "text": "CPU Engineer"},
    {"source_id": "S0003", "text": "CPU Engineer", "extra": "Private extra"},
    {"text": "CPU Engineer"}, {"source_id": "S0003"}, "CPU Engineer",
])
def test_required_span_schema_is_strict_and_hides_input(span):
    data = plan_dict()
    data["sections"][1]["slots"][0]["required_spans"] = [span]
    with pytest.raises(ValidationError) as exc:
        SelectionPlan.model_validate(data)
    assert "Private extra" not in str(exc.value)


@pytest.mark.parametrize("text", [
    "", " ", "\t", "CPU\nEngineer", "CPU\rEngineer", "CPU\u2028Engineer",
    "CPU\u2029Engineer", "CPU\x00Engineer", "CPU\u200bEngineer", "cpu engineer",
    "CPU  Engineer", "Private invented title",
])
def test_required_span_must_be_exact_nonblank_single_line_source_substring(text):
    data = plan_dict()
    data["sections"][1]["slots"][0]["required_spans"] = [{"source_id": "S0003", "text": text}]
    with pytest.raises(ValueError, match="exact source substring") as exc:
        compile_plan(data)
    assert "Private invented title" not in str(exc.value)


@pytest.mark.parametrize("identifier", ["S0006", "S9999"])
def test_required_span_cannot_borrow_unassigned_or_unknown_sources(identifier):
    data = plan_dict()
    data["sections"][1]["slots"][0]["required_spans"] = [{"source_id": identifier, "text": "power benefit"}]
    with pytest.raises(ValueError, match="assigned to its slot"):
        compile_plan(data)


def test_duplicate_required_spans_fail_before_drafting():
    data = plan_dict()
    data["sections"][1]["slots"][0]["required_spans"] = [{"source_id": "S0003", "text": "CPU Engineer"}] * 2
    with pytest.raises(ValueError, match="duplicate required source spans"):
        compile_plan(data)


def test_individually_impossible_required_span_is_rejected_before_drafting():
    data = plan_dict()
    data["sections"][4]["slots"][0].update(
        max_words=2, required_spans=[{"source_id": "S0012", "text": "Python and C++"}],
    )
    with pytest.raises(ValueError, match="span exceeds its slot word budget"):
        compile_plan(data)


@pytest.mark.parametrize("text,accepted", [
    ("CPU Engineer", True), ("cpu engineer", True), ("A CPU Engineer, measuring power.", True),
    ("CPU Engineering", False), ("CPU Engineer2", False), ("microCPU Engineer", False),
    ("CPU Engineers", False), ("CPU  Engineer", False), ("CPU–Engineer", False),
])
def test_required_title_span_uses_case_insensitive_standalone_word_boundaries(text, accepted):
    data = plan_dict()
    data["sections"][1]["slots"][0]["required_spans"] = [{"source_id": "S0003", "text": "CPU Engineer"}]
    response = valid_response()
    response["summary"] = text
    if accepted:
        assert compile_plan(data).assemble(response).sections[1].entries[0].text == text
    else:
        with pytest.raises(ValueError, match="required source span"):
            compile_plan(data).assemble(response)


@pytest.mark.parametrize("text,span,accepted", [
    ("Measured a 6–10% power benefit.", "6–10%", True),
    ("Measured a 16–10% power benefit.", "6–10%", False),
    ("Used C++.", "C++", True), ("Used C++11.", "C++", False),
    ("Used Python (Pandas).", "Python (Pandas)", True),
    ("Used Python XPandas.", "Python (Pandas)", False),
])
def test_required_span_matching_escapes_regex_and_preserves_punctuation(text, span, accepted):
    assert contains_required_span(text, span) is accepted


def test_fixed_spans_are_checked_without_sending_fixed_slots_to_writer():
    data = plan_dict()
    data["sections"][0]["slots"][0]["required_spans"] = [{"source_id": "S0001", "text": "Alex"}]
    compiled = compile_plan(data)
    assert compiled.assemble(valid_response()).sections[0].entries[0].text == "Alex Example"
    assert all(task["slot_id"] != "name" for task in compiled.rewrite_tasks())
    # Even an exact source substring must occur as a standalone retained span.
    data["sections"][0]["slots"][0]["required_spans"][0]["text"] = "Al"
    with pytest.raises(ValueError, match="fixed source join omits"):
        compile_plan(data)


def test_required_anchor_change_affects_plan_fingerprint():
    plan = SelectionPlan.model_validate(plan_dict())
    original = plan_fingerprint(plan)
    data = plan.model_dump()
    data["sections"][1]["slots"][0]["required_spans"] = [{"source_id": "S0003", "text": "CPU Engineer"}]
    assert plan_fingerprint(SelectionPlan.model_validate(data)) != original


def test_required_spans_do_not_prove_actor_or_causal_relationships():
    data = plan_dict()
    data["sections"][2]["slots"][1]["required_spans"] = [
        {"source_id": "S0005", "text": "Extended an existing framework"},
        {"source_id": "S0006", "text": "6–10% power benefit"},
    ]
    response = valid_response()
    response["current_achievement"] = "Extended an existing framework, which directly caused a 6–10% power benefit."
    # The invented direct causal link is NOT established by the two independent
    # source lines. Anchors cannot replace a source-by-source factual audit.
    assert compile_plan(data).assemble(response).sections[2].entries[1].text == response["current_achievement"]
