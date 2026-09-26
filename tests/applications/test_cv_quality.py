"""Presentation limits never replace source/semantic validation."""

import pytest

from app.applications.cv_quality import TAILORING_POLICY, cv_quality_metrics, validate_tailoring
from app.applications.models import TailoredCV
from app.scoring import validate_cv


def cv_with(sections):
    return TailoredCV.model_validate({"sections": [
        {"heading": "Contact", "entries": [{"text": "Alex", "style": "paragraph", "evidence_quotes": ["Alex"]}]},
        *[{"heading": heading, "entries": [
            {"text": text, "style": "bullet", "evidence_quotes": [text]} for text in entries
        ]} for heading, entries in sections],
    ]})


def test_short_source_is_not_padded_to_a_minimum():
    cv = cv_with([("Experience", ["Built Python tools."])])
    assert validate_tailoring(cv) == []
    assert cv_quality_metrics(cv) == {
        "word_count": 4, "summary_words": 0, "skill_entries": 0, "noncontact_entries": 1,
    }


@pytest.mark.parametrize("section,count,problem", [
    ("Experience", 650, "CV word count 651"),
    ("Professional Summary", 71, "Summary word count 71"),
])
def test_word_limits(section, count, problem):
    errors = validate_tailoring(cv_with([(section, [" ".join(["word"] * count)])]))
    assert any(problem in error for error in errors)


def test_exact_word_limit_is_allowed():
    cv = cv_with([("Experience", [" ".join(["word"] * (TAILORING_POLICY["max_words"] - 1))])])
    assert validate_tailoring(cv) == []


def test_skill_and_entry_counts_are_bounded():
    cv = cv_with([("Skills", ["Python"] * 6), ("Experience", ["Built tools"] * 27)])
    assert validate_tailoring(cv) == [
        "Skills entry count 6 exceeds the tailoring limit of 5.",
        "Non-contact entry count 33 exceeds the tailoring limit of 32.",
    ]


def test_quality_gate_does_not_bless_unsupported_source_quotes():
    cv = cv_with([("Experience", ["Invented expertise"] )])
    assert validate_tailoring(cv) == []
    assert validate_cv(cv, "Alex\nActual experience")
