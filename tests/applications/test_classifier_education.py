"""Synthetic education examples; no candidate's private CV is included."""

import pytest

from app.applications.classifier_education import match_degree


REQUIREMENT = "Undergraduate or higher degree in Computer Science, Engineering, Operations Research, or other quantitative discipline."


@pytest.mark.parametrize("credential", [
    "Bachelor of Engineering — Information Technology",
    "BEng in Information Technology",
    "MSc Big Data Science",
    "Bachelor of Science in Computer Science",
    "PhD in Mathematics",
])
def test_shared_degree_alternatives_accept_completed_relevant_education(credential):
    cv = f"EDUCATION\n{credential}\nNorthbridge University | 2015 – 2019\nDistinction\nEXPERIENCE\nBuilt software."
    status, evidence, explanation = match_degree(REQUIREMENT, cv)
    assert status == "MET"
    assert evidence and all(quote in cv for quote in evidence)
    assert credential in evidence[0]
    assert "2015 – 2019" in evidence[0]
    assert "EXPERIENCE" not in evidence[0]
    assert "completed degree" in explanation


def test_markdown_exact_source_quote_is_not_rewritten():
    cv = "## Education\n\n### MSc Big Data Science\n\nNorthbridge University | 2015 – 2019\n\n**Distinction**\n\n## Awards\nPrize winner."
    status, evidence, _ = match_degree(REQUIREMENT, cv)
    assert status == "MET"
    assert evidence == ["### MSc Big Data Science\n\nNorthbridge University | 2015 – 2019\n\n**Distinction**"]


@pytest.mark.parametrize("cv", [
    "I earned a BSc in Computer Science.",
    "I was awarded a BSc in Computer Science.",
    "Completed a Bachelor of Engineering in Information Technology.",
    "I hold an MSc in Big Data Science.",
    "Graduated with a PhD in Statistics.",
    "BEng in Information Technology | 2010–2014",
    "Education\nMSc Big Data Science\nAwarded 2019",
])
def test_explicit_completed_frames_or_past_credential_dates(cv):
    status, evidence, _ = match_degree(REQUIREMENT, cv)
    assert status == "MET"
    assert all(quote in cv for quote in evidence)


@pytest.mark.parametrize("cv", [
    "Python, Computer Science, Engineering and Mathematics skills.",
    "EDUCATION\nMSc Big Data Science",
    "EDUCATION\nMSc Big Data Science\nNorthbridge University | 2024 – Present",
    "EDUCATION\nMSc Big Data Science\nNorthbridge University | 2024 – 2999",
    "EDUCATION\nMSc Big Data Science\nExpected graduation 2019",
    "EDUCATION\nMSc Big Data Science\nNot completed | 2015 – 2019",
    "EDUCATION\nMSc Big Data Science\nIn progress | 2015 – 2019",
    "EDUCATION\nMSc Big Data Science\nStudent at Northbridge University | 2015 – 2019",
    "EDUCATION\nMSc Big Data Science\nWithdrawn | 2015 – 2019",
    "EDUCATION\nMSc Big Data Science\nCertificate course | 2015 – 2019",
    "Pursuing an MSc Big Data Science | 2015 – 2019",
    "Planning to complete a BSc in Computer Science.",
    "I never earned a BSc in Computer Science.",
    "My sister completed a BSc in Computer Science.",
    "She earned a BSc in Computer Science.",
    "Awarded a BSc in Computer Science to my colleague.",
    "Awarded a BSc in Computer Science to Alice.",
    "Hold a BSc in Computer Science.",
    "Awarded an honorary PhD in Mathematics.",
    "I completed a BSc in Computer Science course.",
    "I completed a BSc in Computer Science if I passed the exams.",
    "Example: Completed a BSc in Computer Science.",
    "BSc in Biology with a Computer Science module | 2015–2019",
    "I earned an MSc in Literature.",
    "I earned a Master of Science.",
    "I earned a BSc in Computer Science and Business.",
    "I earned a BSc in Computer Science Fiction.",
    "I earned a BSc in Computer Science for Beginners.",
])
def test_uncertain_unfinished_other_person_or_wrong_subject_never_pass(cv):
    status, evidence, _ = match_degree(REQUIREMENT, cv)
    assert status == "UNKNOWN"
    assert evidence == []


@pytest.mark.parametrize("requirement", [
    "First-class degree in Computer Science.",
    "Accredited undergraduate degree in Engineering.",
    "Degree in Computer Science with a minimum 3.5 GPA.",
    "Degree in Computer Science and three years of experience.",
    "Degree in Computer Science or equivalent professional experience.",
    "Degree in Computer Science and Engineering.",
    "Degree in Computer Science, Engineering.",
    "Degree in Computer Science/Engineering.",
    "Undergraduate degree in Biology or related field.",
    "Degree in Computer Science unless experienced.",
    "No degree in Computer Science is required.",
    "Python is required.",
])
def test_unsupported_requirement_grammar_abstains_without_ignoring_qualifiers(requirement):
    assert match_degree(requirement, "I earned a BSc in Computer Science.") is None


@pytest.mark.parametrize("requirement,cv,expected", [
    ("BSc or MSc in Mathematics is required.", "I earned a BSc in Physics.", "UNKNOWN"),
    ("BSc or MSc in Mathematics is required.", "I earned an MSc in Mathematics.", "MET"),
    ("Master's degree in Computer Science is required.", "I earned a BSc in Computer Science.", "UNKNOWN"),
    ("Bachelor's degree in Computer Science is required.", "I earned an MSc in Computer Science.", "UNKNOWN"),
    ("Bachelor's degree or higher in Computer Science is required.", "I earned an MSc in Computer Science.", "MET"),
    ("Undergraduate or higher degree in Computer Science.", "I earned a BEng in Information Technology.", "UNKNOWN"),
    ("Degree in Engineering or Mathematics.", "I earned a BEng in Information Technology.", "MET"),
    ("Degree in Information Technology.", "I earned a BEng in Information Technology.", "MET"),
    ("Degree in CS or Maths is required.", "Maths skills.", "UNKNOWN"),
])
def test_degree_level_and_subject_qualifiers_are_preserved(requirement, cv, expected):
    status, evidence, _ = match_degree(requirement, cv)
    assert status == expected
    assert all(quote in cv for quote in evidence)


def test_next_degree_does_not_borrow_previous_completion_dates():
    cv = "Education\nBSc in Literature\nNorthbridge University | 2010–2014\nMSc in Computer Science\nNorthbridge University"
    assert match_degree(REQUIREMENT, cv)[0] == "UNKNOWN"


def test_current_year_date_range_is_not_automatically_completed():
    from datetime import date

    cv = f"Education\nMSc Big Data Science\nNorthbridge University | 2024–{date.today().year}"
    assert match_degree(REQUIREMENT, cv)[0] == "UNKNOWN"


@pytest.mark.parametrize("heading", ["Work Experience", "Employment History", "Professional Experience", "Projects", "Skills"])
def test_education_does_not_borrow_dates_from_unrelated_section(heading):
    cv = f"Education\nMSc Big Data Science\nNorthbridge University\n{heading}\nEngineer | 2015–2019"
    assert match_degree(REQUIREMENT, cv)[0] == "UNKNOWN"
