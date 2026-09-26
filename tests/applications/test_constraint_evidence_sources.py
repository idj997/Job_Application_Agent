import pytest

from app.applications.models import ConstraintCheck, MatchAssessment, RequirementMatch
from app.scoring import decide


def assess(check):
    return MatchAssessment(
        description_complete=True,
        requirements=[RequirementMatch(requirement="Python", importance="CORE",
            job_evidence="Python is required.", status="MET", cv_evidence=["Python development."],
            explanation="Documented Python experience.")],
        constraint_checks=[check], recommended_verdict="APPLY", rationale="Supported.", uncertainties=[],
    )


def decision(check, *, legacy=False, context=None):
    return decide(assess(check), "Python development.", "Python is required.", {check.constraint_id},
                  use_recommendation=legacy, job_context=context)


@pytest.mark.parametrize("field,identifier,value", [
    ("title", "role", "Machine Learning Engineer"),
    ("location", "location", "London, United Kingdom"),
])
def test_classifier_structured_evidence_is_validated_against_named_field(field, identifier, value):
    check = ConstraintCheck(constraint_id=identifier, status="PASS", job_evidence=value,
                            job_evidence_source=field, explanation="Matches configured preference.")
    assert decision(check, context={field: value}).verdict == "APPLY"
    assert decision(check).verdict == "REVIEW"
    assert decision(check, context={field: "Different value"}).verdict == "REVIEW"
    assert decision(check, legacy=True, context={field: value}).verdict == "REVIEW"


def test_structured_evidence_cannot_cherry_pick_allowed_country():
    check = ConstraintCheck(constraint_id="location", status="PASS", job_evidence="UK",
                            job_evidence_source="location", explanation="UK matches.")
    assert decision(check, context={"location": "UK or India"}).verdict == "REVIEW"


@pytest.mark.parametrize("identifier,field,value", [
    ("sponsorship", "location", "India"), ("salary", "title", "Salary GBP 80000"),
    ("work_pattern", "location", "Remote"), ("location", "title", "UK Engineer"),
    ("role", "location", "Machine Learning Engineer"),
])
def test_structured_sources_cannot_be_repurposed_for_other_constraints(identifier, field, value):
    check = ConstraintCheck(constraint_id=identifier, status="PASS", job_evidence=value,
                            job_evidence_source=field, explanation="Forged source.")
    assert decision(check, context={field: value}).verdict == "REVIEW"


def test_legacy_constraint_evidence_defaults_to_description():
    check = ConstraintCheck(constraint_id="role", status="PASS", job_evidence="Python is required.",
                            explanation="Legacy description evidence.")
    assert check.job_evidence_source == "description"
    assert decision(check, legacy=True).verdict == "APPLY"


def test_legacy_missing_source_cannot_verify_quote_from_structured_context():
    check = ConstraintCheck(constraint_id="role", status="PASS", job_evidence="Machine Learning Engineer",
                            explanation="No source tag.")
    assert decision(check, context={"title": "Machine Learning Engineer"}).verdict == "REVIEW"
