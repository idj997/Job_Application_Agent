import pytest

from app.applications.classifier_clauses import decompose_shared_skills


@pytest.mark.parametrize("text,combination,items", [
    ("Experience with Python and SQL.", "all", ("Python", "SQL")),
    ("Experience with Python, SQL, Java.", "all", ("Python", "SQL", "Java")),
    ("Experience with Python, SQL and Java.", "all", ("Python", "SQL", "Java")),
    ("Experience with Python, SQL, and Java.", "all", ("Python", "SQL", "Java")),
    ("Experience with Python or SQL.", "any", ("Python", "SQL")),
    ("Experience with Python or SQL or Java.", "any", ("Python", "SQL", "Java")),
    ("Experience with Python, SQL or Java.", "any", ("Python", "SQL", "Java")),
    ("Experience with Python, SQL, or Java.", "any", ("Python", "SQL", "Java")),
    ("Experience with Python & SQL.", "all", ("Python", "SQL")),
    ("Experience with C++ and C# is required.", "all", ("C++", "C#")),
    ("Experience with Node.js and Python.", "all", ("Node.js", "Python")),
])
def test_simple_shared_list_preserves_every_item(text, combination, items):
    parsed = decompose_shared_skills(text)
    assert parsed is not None
    assert parsed.atoms == tuple(f"Experience with {item}" for item in items)
    assert parsed.combination == combination
    assert parsed.uncertainty is None


@pytest.mark.parametrize("prefix", [
    "Professional experience with", "Commercial experience in", "Hands-on experience with",
    "Proven hands-on production experience with", "Demonstrated industry experience in",
    "Working knowledge of", "Strong familiarity with", "Deep understanding of", "Proficiency in",
])
def test_every_atom_retains_every_shared_qualifier(prefix):
    parsed = decompose_shared_skills(f"{prefix} Python and SQL")
    assert parsed is not None
    assert parsed.atoms == (f"{prefix} Python", f"{prefix} SQL")
    assert parsed.combination == "all"
    assert parsed.uncertainty is None


def test_open_ended_scientific_list_exposes_only_explicit_members_and_retains_uncertainty():
    parsed = decompose_shared_skills(
        "Experience with scientific computing, statistics, optimization, time series, panel data, etc."
    )
    assert parsed is not None
    assert parsed.atoms == (
        "Experience with scientific computing", "Experience with statistics", "Experience with optimization",
        "Experience with time series", "Experience with panel data",
    )
    assert parsed.combination == "all"
    assert parsed.uncertainty and "unspecified remainder" in parsed.uncertainty


@pytest.mark.parametrize("text", [
    "Professional Python or SQL experience",
    "Experience with Python and SQL in production",
    "Experience with Python and commercial SQL",
    "Experience with Python and SQL for financial clients",
    "Experience with Python and proven SQL",
    "Experience with Python and hands-on SQL",
    "Experience with Python and SQL with production experience",
    "Experience with Python and SQL or Java",
    "Experience with Python or SQL and Java",
    "Experience with Python or SQL, Java",
    "Experience with Python, and SQL, Java",
    "Experience with Python, SQL and/or Java",
    "Experience with Python (preferred) or C++",
    "Experience with Python or SQL is preferred",
    "Full stack experience with Python (preferred) or C++, Spark/Scala, SQL or other distributed data processing technologies as well as experience working comfortably building and deploying services and models in containerized environments",
    "Experience with Python and SQL unless waived",
    "Experience with either Python or SQL",
    "No experience with Python and SQL",
    "Experience with Python and SQL not required",
    "Experience with Python and SQL equivalent",
    "Experience with Python and other languages",
    "Experience with Python, SQL, e.g. Java",
    "Experience with Python, SQL, including Java",
    "Experience with Python and SQL for at least three years",
    "Three years of experience with Python and SQL",
    "Experience with Python and SQL for 3 years",
    "Experience with Python and SQL with UK work authorization",
    "Experience with Python and fluent English",
    "Knowledge of Python and eligibility",
    "Knowledge of BSc and MSc qualifications",
    "Experience with Python and master's degree",
    "Experience with Python and SQL; experience with Rust",
    "Experience with Python and SQL. Production experience",
    "Experience with Python and SQL. Kubernetes",
    "Experience with Python and SQL...",
    "Experience with Python and SQL\nProfessional experience",
    "Experience with Python and SQL and",
    "Experience with Python,, SQL",
    "Experience with Python and Python",
    "Experience with Python",
    "Experience with Python, etc.",
    "Experience with A, B, C, D, E, F, G, H, I",
    "",
])
def test_unsupported_or_ambiguous_lists_abstain(text):
    assert decompose_shared_skills(text) is None


def test_interior_case_and_qualifier_wording_are_not_normalized():
    parsed = decompose_shared_skills("Proven Hands On experience with PYTHON and SQL")
    assert parsed is not None
    assert parsed.atoms == ("Proven Hands On experience with PYTHON", "Proven Hands On experience with SQL")


def test_non_string_input_abstains():
    assert decompose_shared_skills(None) is None
