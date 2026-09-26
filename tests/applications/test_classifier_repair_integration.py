"""Synthetic regression of real parser/rules/scorer/ledger, with offline NLI.

These are workflow correctness checks, not a classifier-accuracy benchmark.
"""

from app.applications.classifier_matching import ClassifierMatcher
from app.applications.models import CandidateProfile, Preferences
from app.applications.tracking import ApplicationLedger
from app.applications.workflow import ApplicationWorkflow


class OfflineNLI:
    def classify(self, context, hypothesis):
        return {"label": "entailment", "confidence": 0.97,
                "probabilities": {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01}}


class ForbiddenGenerator:
    name = "ollama"
    model = "offline-test"

    def generate_structured(self, **kwargs):
        raise AssertionError("Matching-only must never invoke the CV generator")


def run(tmp_path, *, complete):
    profile = CandidateProfile(preferences=Preferences(
        target_roles=["Machine Learning Engineer"], locations=["United Kingdom"],
        requires_sponsorship=True,
    ))
    description = (
        "Specializing in reliable products, our business delivers strong results.\n"
        "The foundation of our success is the experience of our people.\n"
        "Description\nWe are seeking a ML Engineer to join our team.\n"
        "Our company builds data products for customers around the world.\n"
        + ("We provide visa sponsorship.\n" if complete else "")
        + "Requirements\n"
        "Undergraduate or higher degree in Computer Science, Engineering, or another quantitative discipline\n"
        "Experience with Python and SQL\n"
        "Experience with C++ is a great plus\n"
        + ("" if complete else "3+ years of machine learning experience is required.\n")
    )
    cv = (
        "Education\nMSc Data Science\nNorthern University | 2018 - 2020\n"
        "Professional Experience\nWorked with Python.\n"
        + ("Worked with SQL.\n" if complete else "")
    )
    job = {"title": "Machine Learning Engineer", "location": "London, England, United Kingdom",
           "description": description, "metadata": {"description_type": "full"},
           "source_url": "https://jobs.example.test/repair-regression"}
    workflow = ApplicationWorkflow(ForbiddenGenerator(), ApplicationLedger(tmp_path / "ledger.db"),
                                   tmp_path, matcher=ClassifierMatcher(classifier=OfflineNLI()))
    return workflow.prepare(job, profile, cv, match_only=True)


def test_repaired_matching_retains_real_uncertainties_without_calling_generator(tmp_path):
    record = run(tmp_path, complete=False)
    assert record["status"] == "review"
    assert record["decision"]["verdict"] == "REVIEW"
    assert {check["constraint_id"]: check["status"] for check in record["match"]["constraint_checks"]} == {
        "role": "PASS", "location": "PASS", "sponsorship": "UNKNOWN",
    }
    requirements = record["match"]["requirements"]
    assert [item["status"] for item in requirements] == ["MET", "PARTIAL", "MISSING", "UNKNOWN"]
    assert requirements[2]["importance"] == "PREFERRED"
    assert record["matcher_diagnostics"]["nli_pairs"] > 0
    assert not any("could not be verified" in reason for reason in record["reasons"])
    assert record["artifacts"] == {}
    assert not record.get("submission_attempted")


def test_explicitly_supported_match_can_pass_without_waiving_evidence_gates(tmp_path):
    record = run(tmp_path, complete=True)
    assert record["status"] == "matched"
    assert record["decision"]["verdict"] == "APPLY"
    assert record["match"]["uncertainties"] == []
    assert record["artifacts"] == {}
    assert not record.get("submission_attempted")
