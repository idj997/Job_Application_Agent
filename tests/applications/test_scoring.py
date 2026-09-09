import pytest
from pydantic import ValidationError

from app.applications.models import (
    ConstraintCheck, CVEntry, CVEvaluation, CVSection, MatchAssessment,
    RequirementMatch, TailoredCV,
)
from app.scoring import cv_ready, cv_score, decide, is_quote, validate_cv


CV = "Alex Example\nPython and SQL pipelines.\nBSc Computer Science, 2021."
JOB = (
    "Python is required. SQL is expected. Airflow is helpful. Java is not required. "
    "The role is in London. We cannot sponsor visas."
)


def requirement(name="Python", importance="CORE", status="MET"):
    quotes = {
        "Python": "Python is required.", "SQL": "SQL is expected.",
        "Airflow": "Airflow is helpful.", "Java": "Java is not required.",
    }
    return RequirementMatch(
        requirement=name, importance=importance, job_evidence=quotes[name], status=status,
        cv_evidence=["Python and SQL pipelines."] if status in {"MET", "PARTIAL"} else [],
        explanation="Evidence checked against the supplied source.",
    )


def assessment(requirements=None, checks=None, **changes):
    values = dict(
        description_complete=True, requirements=requirements if requirements is not None else [requirement()],
        constraint_checks=checks or [], recommended_verdict="APPLY",
        rationale="Documented experience meets the requirements.", uncertainties=[],
    )
    values.update(changes)
    return MatchAssessment(**values)


def tailored_cv():
    return TailoredCV(sections=[
        CVSection(heading="Contact", entries=[
            CVEntry(text="Alex Example", style="paragraph", evidence_quotes=["Alex Example"]),
        ]),
        CVSection(heading="Experience", entries=[
            CVEntry(text="Python and SQL pipelines.", style="bullet", evidence_quotes=["Python and SQL pipelines."]),
        ]),
    ])


def evaluation(**changes):
    values = dict(
        requirement_coverage=100, clarity=100, factual_consistency=True,
        unsupported_claims=[], missing_critical_information=[], improvements=[],
    )
    values.update(changes)
    return CVEvaluation(**values)


def test_score_weights_real_requirements_and_ignores_negated_requirements():
    result = decide(assessment(requirements=[
        requirement(), requirement("SQL", "IMPORTANT", "PARTIAL"),
        requirement("Airflow", "PREFERRED", "MISSING"),
        requirement("Java", "NOT_REQUIREMENT", "MISSING"),
    ]), CV, JOB, set())

    assert result.score == 72  # (5 + 3 * 0.5 + 0) / (5 + 3 + 1)
    assert result.verdict == "APPLY"


def test_match_below_threshold_skips_only_after_mandatory_checks_pass():
    result = decide(assessment(requirements=[
        requirement(), requirement("SQL", "IMPORTANT", "MISSING"),
    ]), CV, JOB, set(), threshold=70)

    assert result.score == 62
    assert result.verdict == "SKIP"


@pytest.mark.parametrize("status", ["UNKNOWN", "MISSING", "PARTIAL"])
def test_unresolved_core_requirement_requires_review_before_score_decision(status):
    result = decide(assessment(requirements=[requirement(status=status)]), CV, JOB, set())

    assert result.verdict == "REVIEW"
    assert any("mandatory" in reason.lower() for reason in result.reasons)


@pytest.mark.parametrize("status,expected", [("FAIL", "SKIP"), ("UNKNOWN", "REVIEW")])
def test_required_user_constraint_cannot_be_overridden_by_perfect_match_score(status, expected):
    check = ConstraintCheck(
        constraint_id="sponsorship", status=status,
        job_evidence="We cannot sponsor visas." if status == "FAIL" else None,
        explanation="Visa sponsorship must be resolved before applying.",
    )

    result = decide(assessment(checks=[check]), CV, JOB, {"sponsorship"})

    assert result.score == 100
    assert result.verdict == expected


@pytest.mark.parametrize("check_ids", [[], ["location", "location"], ["location", "salary"], ["salary"]])
def test_every_configured_constraint_must_be_checked_exactly_once(check_ids):
    checks = [ConstraintCheck(
        constraint_id=identifier, status="PASS", job_evidence="The role is in London.",
        explanation="The location fits.",
    ) for identifier in check_ids]

    result = decide(assessment(checks=checks), CV, JOB, {"location"})

    assert result.verdict == "REVIEW"
    assert any("every configured constraint exactly once" in reason for reason in result.reasons)


@pytest.mark.parametrize("evidence", [None, "The employer has a secret London office."])
def test_constraint_pass_requires_verifiable_job_evidence(evidence):
    check = ConstraintCheck(
        constraint_id="location", status="PASS", job_evidence=evidence,
        explanation="The location fits.",
    )

    result = decide(assessment(checks=[check]), CV, JOB, {"location"})

    assert result.verdict == "REVIEW"


@pytest.mark.parametrize("field,bad_value", [
    ("job_evidence", "An invented requirement from another advertisement."),
    ("cv_evidence", ["Ten years of Python experience."]),
    ("cv_evidence", []),
    ("cv_evidence", [""]),
])
def test_model_verdict_cannot_accept_missing_or_invented_evidence(field, bad_value):
    response = assessment()
    setattr(response.requirements[0], field, bad_value)

    result = decide(response, CV, JOB, set())

    assert result.score == 100
    assert result.verdict == "REVIEW"
    assert result.reasons


def test_duplicate_requirements_cannot_inflate_match_score():
    duplicate = requirement()
    duplicate.requirement = "PYTHON"

    result = decide(assessment(requirements=[requirement(), duplicate]), CV, JOB, set())

    assert result.verdict == "REVIEW"
    assert any("Duplicate requirements" in reason for reason in result.reasons)


@pytest.mark.parametrize("response", [
    assessment(description_complete=False),
    assessment(requirements=[]),
    assessment(requirements=[requirement("Java", "CONTEXTUAL", "MISSING")]),
])
def test_incomplete_or_unscored_assessment_cannot_approve(response):
    assert decide(response, CV, JOB, set()).verdict == "REVIEW"


@pytest.mark.parametrize("recommendation", ["SKIP", "REVIEW"])
def test_model_concerns_remain_visible_despite_high_numeric_score(recommendation):
    response = assessment(recommended_verdict=recommendation, rationale="Material concerns remain.")

    result = decide(response, CV, JOB, set())

    assert result.verdict == recommendation
    assert result.reasons == ["Material concerns remain."]


def test_verbatim_evidence_tolerates_document_whitespace_but_not_added_claims():
    assert is_quote("Python and SQL", "Experience: Python\n  and\tSQL pipelines")
    assert not is_quote("Python and Spark", CV)
    assert not is_quote("", CV)
    assert not is_quote(" \n\t", CV)


def test_valid_cv_lines_have_verifiable_source_evidence():
    assert validate_cv(tailored_cv(), CV) == []


@pytest.mark.parametrize("evidence", [["Invented employer and qualification."], [""]])
def test_unverifiable_cv_source_evidence_is_rejected(evidence):
    cv = tailored_cv()
    cv.sections[1].entries[0].evidence_quotes = evidence

    assert any("source evidence could not be verified" in error for error in validate_cv(cv, CV))


def test_cv_entry_schema_requires_at_least_one_source_quote():
    with pytest.raises(ValidationError):
        CVEntry(text="A claimed achievement", style="bullet", evidence_quotes=[])


@pytest.mark.parametrize("text", ["Python\n## New section", "Python\rNew section", "Python\x1bSQL", "Python\x00SQL"])
def test_generated_cv_entries_cannot_inject_extra_lines_or_control_characters(text):
    cv = tailored_cv()
    cv.sections[1].entries[0].text = text

    assert any("one plain-text line" in error for error in validate_cv(cv, CV))


def test_cv_must_start_with_contact_and_have_no_duplicate_sections():
    cv = tailored_cv()
    cv.sections.reverse()
    cv.sections.append(cv.sections[0].model_copy(deep=True))

    errors = validate_cv(cv, CV)

    assert any("start with contact" in error for error in errors)
    assert any("duplicate sections" in error for error in errors)


def test_cv_quality_rubric_has_reproducible_weighting_and_threshold():
    below = evaluation(requirement_coverage=80, clarity=60)
    boundary = evaluation(requirement_coverage=81, clarity=61)

    assert cv_score(below) == 74
    assert cv_ready(below, 75) is False
    assert cv_score(boundary) == 75
    assert cv_ready(boundary, 75) is True


@pytest.mark.parametrize("changes", [
    {"factual_consistency": False},
    {"unsupported_claims": ["Unverified achievement."]},
    {"missing_critical_information": ["Employment dates are absent."]},
])
def test_high_cv_score_never_overrides_failed_factual_or_completeness_checks(changes):
    review = evaluation(**changes)

    assert cv_score(review) == 100
    assert cv_ready(review, 75) is False
