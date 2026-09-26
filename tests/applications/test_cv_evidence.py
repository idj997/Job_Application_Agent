import json
from typing import get_args

import pytest
from pydantic import ValidationError

from app.applications.cv_evidence import (
    CVSource, ReferencedCV, ReferencedCVEntry, ReferencedCVSection,
    build_cv_sources, reconcile_exact_cv_references, resolve_cv_references,
)
from app.applications.models import CVSection, TailoredCV
from app.scoring import validate_cv


MASTER_CV = (
    "Alex Example\n\nCPU Engineer\nExample Devices\n"
    "Cambridge\nOct 2024–Present\nImproved power efficiency by 6–10%.\n"
)


def referenced_cv(evidence_ids=None):
    return ReferencedCV(sections=[
        ReferencedCVSection(heading="Contact", entries=[
            ReferencedCVEntry(text="Alex Example", style="paragraph", evidence_ids=["S0001"]),
        ]),
        ReferencedCVSection(heading="Experience", entries=[
            ReferencedCVEntry(
                text="CPU Engineer | Example Devices | Cambridge | Oct 2024–Present",
                style="paragraph",
                evidence_ids=evidence_ids if evidence_ids is not None else [
                    "S0002", "S0003", "S0004", "S0005",
                ],
            ),
        ]),
    ])


def test_source_catalog_numbers_all_nonblank_lines_without_combining_them():
    sources = build_cv_sources(MASTER_CV)
    assert [source.source_id for source in sources] == [f"S{x:04d}" for x in range(1, 7)]
    assert [source.text for source in sources] == [line for line in MASTER_CV.splitlines() if line]
    assert all(source.text in MASTER_CV for source in sources)


def test_source_catalog_is_stable_and_blank_separators_do_not_change_ids():
    source = " A name \n\nCPU Engineer\n"
    assert build_cv_sources(source) == build_cv_sources("\nA name\n\n\nCPU Engineer\n\n")
    assert build_cv_sources(source) == build_cv_sources(source)


def test_duplicate_source_text_retains_distinct_ids_and_order():
    sources = build_cv_sources("Python\nSQL\nPython\n")
    assert [(source.source_id, source.text) for source in sources] == [
        ("S0001", "Python"), ("S0002", "SQL"), ("S0003", "Python"),
    ]


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r"])
def test_line_endings_do_not_change_source_text(separator):
    sources = build_cv_sources(separator.join(["Name", "Role", "Degree"]))
    assert [source.text for source in sources] == ["Name", "Role", "Degree"]


def test_unicode_and_interior_table_tabs_are_preserved_exactly():
    source = "  María González\t工程師  \n\tMSc\t2023–2024\tDistinction\t\n e\u0301 ≠ é; 6–10% power benefit. "
    sources = build_cv_sources(source)
    assert [item.text for item in sources] == [
        "María González\t工程師", "MSc\t2023–2024\tDistinction", "e\u0301 ≠ é; 6–10% power benefit.",
    ]
    assert all(item.text in source for item in sources)


def test_long_source_line_is_never_truncated_or_rewritten():
    line = "Very long supported project detail: " + "CPU/GPU — measurement; " * 10_000
    sources = build_cv_sources("Name\n" + line + "\nFinal qualification")
    assert len(sources) == 3
    assert sources[1].text == line.strip()
    assert sources[2].text == "Final qualification"


def test_large_catalog_does_not_drop_sources_and_ids_extend_past_four_digits():
    sources = build_cv_sources("\n".join(f"Source {index}" for index in range(10_001)))
    assert len(sources) == 10_001
    assert sources[-2].source_id == "S10000"
    assert sources[-1].source_id == "S10001"
    assert sources[-1].text == "Source 10000"


@pytest.mark.parametrize("source", ["", " \t\n\r\n ", "\u2003\n\u00a0"])
def test_empty_source_is_rejected(source):
    with pytest.raises(ValueError, match="no nonblank source lines"):
        build_cv_sources(source)


@pytest.mark.parametrize("source", [None, 42, b"Name", ["Name"]])
def test_nonstring_master_cv_is_rejected_without_echoing_input(source):
    with pytest.raises(TypeError, match="Master CV must be a string"):
        build_cv_sources(source)


def test_joined_role_header_resolves_to_separate_exact_source_quotes():
    cv = resolve_cv_references(referenced_cv(), build_cv_sources(MASTER_CV))
    assert isinstance(cv, TailoredCV)
    entry = cv.sections[1].entries[0]
    assert entry.text == "CPU Engineer | Example Devices | Cambridge | Oct 2024–Present"
    assert entry.evidence_quotes == ["CPU Engineer", "Example Devices", "Cambridge", "Oct 2024–Present"]
    assert entry.text not in MASTER_CV
    assert all(quote in MASTER_CV for quote in entry.evidence_quotes)
    assert validate_cv(cv, MASTER_CV) == []


def test_evidence_order_follows_selected_ids_and_source_catalog_is_not_mutated():
    sources = build_cv_sources(MASTER_CV)
    original = [source.model_dump() for source in sources]
    cv = resolve_cv_references(referenced_cv(["S0003", "S0002"]), sources)
    assert cv.sections[1].entries[0].evidence_quotes == ["Example Devices", "CPU Engineer"]
    assert [source.model_dump() for source in sources] == original


def test_same_source_id_can_be_used_in_different_entries():
    model = referenced_cv(["S0001"])
    cv = resolve_cv_references(model, build_cv_sources(MASTER_CV))
    assert cv.sections[0].entries[0].evidence_quotes == cv.sections[1].entries[0].evidence_quotes


def test_unknown_source_ids_fail_closed_without_exposing_cv_text():
    with pytest.raises(ValueError, match="section 2, entry 1 references an unknown source ID") as error:
        resolve_cv_references(referenced_cv(["S9999"]), build_cv_sources(MASTER_CV))
    assert "Alex" not in str(error.value)
    assert "CPU Engineer" not in str(error.value)


@pytest.mark.parametrize("ids", [
    [], [""], [" "], [None], [12], [True], [b"S0001"], ["S0001 "],
    [" S0001"], ["S0001\n"], ("S0001",), "S0001", None,
])
def test_evidence_ids_are_required_nonempty_strict_unpadded_strings(ids):
    with pytest.raises(ValidationError):
        ReferencedCVEntry(text="A claim", style="bullet", evidence_ids=ids)


@pytest.mark.parametrize("identifier", ["s0001", "1", "S1", "S-001", "person@example.com"])
def test_malformed_ids_are_rejected_without_echoing_values(identifier):
    with pytest.raises(ValidationError) as error:
        ReferencedCVEntry(text="A private claim", style="bullet", evidence_ids=[identifier])
    assert "person@example.com" not in str(error.value)
    assert "A private claim" not in str(error.value)


def test_missing_evidence_ids_are_rejected():
    with pytest.raises(ValidationError):
        ReferencedCVEntry(text="A claim", style="bullet")


def test_duplicate_evidence_ids_within_one_entry_are_rejected():
    with pytest.raises(ValidationError, match="duplicate evidence IDs"):
        ReferencedCVEntry(text="A claim", style="bullet", evidence_ids=["S0001", "S0001"])


def test_model_cannot_add_its_own_evidence_quotes():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as error:
        ReferencedCVEntry(
            text="A claim", style="bullet", evidence_ids=["S0001"],
            evidence_quotes=["Invented confidential information"],
        )
    assert "Invented confidential information" not in str(error.value)


def test_duplicate_catalog_ids_are_rejected_even_when_text_is_the_same():
    sources = build_cv_sources(MASTER_CV)
    with pytest.raises(ValueError, match="duplicate source IDs"):
        resolve_cv_references(referenced_cv(), sources + [sources[0]])


@pytest.mark.parametrize("sources", [[], None, {}, ["S0001"], [{"source_id": "S0001", "text": "Private name"}]])
def test_invalid_catalog_is_rejected_without_exposing_source_text(sources):
    with pytest.raises(ValueError) as error:
        resolve_cv_references(referenced_cv(), sources)
    assert "Private name" not in str(error.value)


@pytest.mark.parametrize("ids", [["S0001", "S0001"], [""], [], [42]])
def test_mutated_reference_model_is_revalidated(ids):
    model = referenced_cv()
    model.sections[0].entries[0].evidence_ids = ids
    with pytest.raises(ValueError, match="referenced CV schema"):
        resolve_cv_references(model, build_cv_sources(MASTER_CV))


def test_mutated_catalog_entry_is_revalidated():
    sources = build_cv_sources(MASTER_CV)
    sources[0].source_id = "Private invalid ID"
    with pytest.raises(ValueError, match="invalid source entry") as error:
        resolve_cv_references(referenced_cv(), sources)
    assert "Private invalid ID" not in str(error.value)


@pytest.mark.parametrize("text", ["", " \t", 123, ["Private name"]])
def test_mutated_catalog_text_is_revalidated_without_printing_warnings(text, recwarn):
    sources = build_cv_sources(MASTER_CV)
    sources[0].text = text
    with pytest.raises(ValueError, match="invalid source entry") as error:
        resolve_cv_references(referenced_cv(), sources)
    assert "Private name" not in str(error.value)
    assert not recwarn.list


def test_plain_dictionary_cannot_bypass_reference_schema():
    with pytest.raises(ValueError, match="referenced CV schema"):
        resolve_cv_references(referenced_cv().model_dump(), build_cv_sources(MASTER_CV))


def test_reference_schema_has_same_headings_as_existing_cv_schema():
    assert get_args(ReferencedCVSection.model_fields["heading"].annotation) == get_args(
        CVSection.model_fields["heading"].annotation,
    )


@pytest.mark.parametrize("payload", [
    {"sections": []},
    {"sections": [{"heading": "Contact", "entries": []}]},
    {"sections": [{"heading": "Made-up Heading", "entries": [
        {"text": "A claim", "style": "bullet", "evidence_ids": ["S0001"]},
    ]}]},
])
def test_required_cv_structure_is_enforced(payload):
    with pytest.raises(ValidationError):
        ReferencedCV.model_validate(payload)


def test_json_schema_has_only_reference_evidence_and_forbids_extra_fields():
    schema = ReferencedCV.model_json_schema()
    entry = schema["$defs"]["ReferencedCVEntry"]
    assert set(entry["properties"]) == {"text", "style", "evidence_ids"}
    assert set(entry["required"]) == {"text", "style", "evidence_ids"}
    assert entry["additionalProperties"] is False
    assert entry["properties"]["evidence_ids"]["type"] == "array"
    assert entry["properties"]["evidence_ids"]["minItems"] == 1
    assert schema["additionalProperties"] is False
    json.dumps(schema)


def test_reference_round_trip_is_compatible_with_structured_json_output():
    original = referenced_cv()
    restored = ReferencedCV.model_validate_json(original.model_dump_json())
    assert restored == original
    assert resolve_cv_references(restored, build_cv_sources(MASTER_CV)) == resolve_cv_references(
        original, build_cv_sources(MASTER_CV),
    )


def test_valid_ids_are_traceability_not_a_factual_entailment_check():
    model = referenced_cv()
    model.sections[1].entries[0].text = "Invented production RTL design experience."
    cv = resolve_cv_references(model, build_cv_sources(MASTER_CV))
    # Correct source IDs prevent fabricated quotations, not unsupported claims.
    # The audit/human review remains necessary and must not be replaced by this.
    assert cv.sections[1].entries[0].text == "Invented production RTL design experience."
    assert cv.sections[1].entries[0].evidence_quotes == [
        "CPU Engineer", "Example Devices", "Cambridge", "Oct 2024–Present",
    ]


def test_exact_reconciliation_relinks_a_whole_source_line_with_safe_audit_notes():
    model = referenced_cv(["S0002", "S0003"])
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    reconciled, notes = reconcile_exact_cv_references(model, build_cv_sources(MASTER_CV))
    assert reconciled.sections[1].entries[0].evidence_ids == ["S0006"]
    assert notes == [{
        "section_index": 2,
        "entry_index": 1,
        "original_ids": ["S0002", "S0003"],
        "resolved_ids": ["S0006"],
        "reason": "exact_source_line_match",
    }]
    assert "Improved" not in json.dumps(notes)
    assert "Alex" not in json.dumps(notes)
    assert resolve_cv_references(reconciled, build_cv_sources(MASTER_CV)).sections[1].entries[0].evidence_quotes == [
        "Improved power efficiency by 6–10%.",
    ]


def test_exact_reconciliation_preserves_every_other_field_and_caller_objects():
    model = referenced_cv(["S0002", "S0003"])
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    sources = build_cv_sources(MASTER_CV)
    before_model = model.model_dump()
    before_sources = [source.model_dump() for source in sources]
    reconciled, notes = reconcile_exact_cv_references(model, sources)
    expected = model.model_dump()
    expected["sections"][1]["entries"][0]["evidence_ids"] = ["S0006"]
    assert reconciled.model_dump() == expected
    assert model.model_dump() == before_model
    assert [source.model_dump() for source in sources] == before_sources
    reconciled.sections[0].entries[0].text = "Changed copy"
    notes[0]["original_ids"].append("S9999")
    notes[0]["resolved_ids"].append("S9999")
    assert model.model_dump() == before_model
    assert reconciled.sections[1].entries[0].evidence_ids == ["S0006"]


def test_exact_reconciliation_keeps_existing_exact_citation_and_other_selected_ids():
    model = referenced_cv(["S0002", "S0006", "S0003"])
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    reconciled, notes = reconcile_exact_cv_references(model, build_cv_sources(MASTER_CV))
    assert reconciled == model
    assert reconciled is not model
    assert notes == []


def test_exact_reconciliation_uses_first_matching_catalog_entry_not_lowest_id():
    model = referenced_cv(["S0002"])
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    sources = build_cv_sources(MASTER_CV)
    sources.insert(0, CVSource(source_id="S0100", text="Improved power efficiency by 6–10%."))
    reconciled, notes = reconcile_exact_cv_references(model, sources)
    assert reconciled.sections[1].entries[0].evidence_ids == ["S0100"]
    assert notes[0]["resolved_ids"] == ["S0100"]


def test_exact_reconciliation_does_not_replace_a_later_identical_source_already_cited():
    model = referenced_cv(["S0006"])
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    sources = build_cv_sources(MASTER_CV)
    sources.insert(0, CVSource(source_id="S0100", text="Improved power efficiency by 6–10%."))
    reconciled, notes = reconcile_exact_cv_references(model, sources)
    assert reconciled == model
    assert notes == []


@pytest.mark.parametrize("text", [
    "Improved power efficiency by 6–10%",  # punctuation differs
    "Improved power efficiency by 6-10%.",  # dash differs
    "Improved  power efficiency by 6–10%.",  # whitespace differs
    "Improved\tpower efficiency by 6–10%.",
    " Improved power efficiency by 6–10%.",
    "Improved power efficiency by 6–10%. ",
    "improved power efficiency by 6–10%.",  # case differs
    "power efficiency by 6–10%.",  # substring only
    "Improved power efficiency by 6–10%. Led the entire programme.",
    "Improved CPU speed by 6–10%.",  # different claim, related words
])
def test_exact_reconciliation_never_normalizes_or_repairs_nonidentical_claims(text):
    model = referenced_cv(["S0002"])
    # Assignment deliberately preserves edge spaces that model construction
    # normally strips, verifying this helper itself performs no normalization.
    model.sections[1].entries[0].text = text
    reconciled, notes = reconcile_exact_cv_references(model, build_cv_sources(MASTER_CV))
    assert reconciled == model
    assert reconciled.sections[1].entries[0].text == text
    assert notes == []


def test_exact_reconciliation_never_repairs_unknown_ids_even_for_verbatim_claims():
    model = referenced_cv(["S9999"])
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    with pytest.raises(ValueError, match="unknown source ID"):
        reconcile_exact_cv_references(model, build_cv_sources(MASTER_CV))
    assert model.sections[1].entries[0].evidence_ids == ["S9999"]


@pytest.mark.parametrize("ids", [[""], [], ["S0002", "S0002"], [None], [" S0002"]])
def test_exact_reconciliation_never_repairs_invalid_ids_even_for_verbatim_claims(ids):
    model = referenced_cv()
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    model.sections[1].entries[0].evidence_ids = ids
    with pytest.raises(ValueError, match="referenced CV schema"):
        reconcile_exact_cv_references(model, build_cv_sources(MASTER_CV))


def test_exact_reconciliation_rejects_invalid_catalog_before_linking():
    model = referenced_cv(["S0002"])
    model.sections[1].entries[0].text = "Improved power efficiency by 6–10%."
    sources = build_cv_sources(MASTER_CV)
    with pytest.raises(ValueError, match="duplicate source IDs"):
        reconcile_exact_cv_references(model, sources + [sources[-1]])


def test_exact_reconciliation_is_idempotent_and_preserves_multi_line_headers():
    model = referenced_cv()
    model.sections[0].entries[0].evidence_ids = ["S0002"]
    first, first_notes = reconcile_exact_cv_references(model, build_cv_sources(MASTER_CV))
    second, second_notes = reconcile_exact_cv_references(first, build_cv_sources(MASTER_CV))
    assert first == second
    assert len(first_notes) == 1
    assert second_notes == []
    assert first.sections[1] == model.sections[1]
