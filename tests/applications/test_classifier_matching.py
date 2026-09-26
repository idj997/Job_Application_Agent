"""Classifier matching and adapter boundaries with no downloaded model weights."""

from contextlib import nullcontext
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from app.applications import classifier_matching as matching
from app.applications.models import CandidateProfile, ConstraintCheck, Preferences
from app.classifiers.requirement_nli import NLIInputTooLong, RequirementNLIClassifier
from app.providers.base import ModelProviderError


INTRO = (
    "Our organization makes products for customers around the country. "
    "The team meets regularly to discuss its plans and listen to customer feedback. "
    "This advertisement describes a permanent position in our central office. "
    "Applications are considered by the hiring team on an ongoing basis. "
    "Further details about the business are available on our company website."
)


def job(*requirements, heading="Requirements"):
    return {"title": "Data Engineer", "description": INTRO + "\n" + heading + ":\n" + "\n".join(requirements), "metadata": {"description_type": "full"}}


def prediction(entailment=0.97, contradiction=0.01):
    values = {"entailment": entailment, "contradiction": contradiction, "neutral": 1 - entailment - contradiction}
    label = max(values, key=values.get)
    return {"label": label, "confidence": values[label], "probabilities": values}


class FakeClassifier:
    fingerprint = {"model": "offline-test-baseline", "revision": "test", "device": "cpu"}

    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else prediction()

    def classify(self, context, hypothesis):
        self.calls.append((context, hypothesis))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in ("REQUIREMENT_NLI_MODEL", "REQUIREMENT_NLI_REVISION", "REQUIREMENT_NLI_DEVICE"):
        monkeypatch.delenv(name, raising=False)


def assess(record, cv="Built Python data pipelines.", classifier=None):
    matcher = matching.ClassifierMatcher(classifier=classifier or FakeClassifier())
    return matcher, matcher.assess(record, CandidateProfile(), cv)


def test_simple_requirement_uses_nli_and_preserves_exact_source_quotes():
    record = job("Python is required.")
    cv = "Alex Example\nBuilt Python data pipelines."
    matcher, result = assess(record, cv)
    assert result.requirements[0].status == "MET"
    assert result.requirements[0].importance == "CORE"
    assert result.requirements[0].job_evidence in record["description"]
    assert all(quote in cv for quote in result.requirements[0].cv_evidence)
    assert matcher.classifier.calls
    assert result.recommended_verdict == "APPLY"
    assert result.uncertainties == []
    assert "not custom-trained" in matcher.diagnostics["baseline"]
    assert "baseline" not in " ".join(result.uncertainties)


@pytest.mark.parametrize("quote", [
    "Contributed to a Python-based education system.",
    "Worked on Python reporting pipelines.",
    "Worked with Python and SQL.",
    "Worked as a Python developer.",
    "Worked at a company using Python.",
    "Worked in Python application development.",
])
def test_subjectless_achievement_normalization_keeps_original_source_evidence(quote):
    matcher, result = assess(job("Python is required."), quote)
    assert matcher.classifier.calls[0][0] == "The candidate " + quote[0].lower() + quote[1:]
    assert result.requirements[0].cv_evidence == [quote]
    assert result.requirements[0].status == "MET"
    assert matcher.diagnostics["normalized_subject_pairs"] == 1
    assert matching.ENTAILMENT_THRESHOLD == 0.90


@pytest.mark.parametrize("quote", [
    "Build Python pipelines.",
    "Automated Python scripts were built by Alice.",
    "Contributed Python code was supplied by a colleague.",
    "Never used Python at work.",
    "Currently learning to build Python pipelines.",
    "Alice automated Python reporting.",
    "The team built Python tools.",
    "Supported Jane who wrote Python code.",
    "Python, SQL and Java.",
    "Automated Python scripts would improve reporting.",
    "Supported Python versions include Python 3.",
    "Developed Python software could replace our legacy systems.",
    "Built entirely by Alice using Python.",
    "Automated daily by Alice using Python.",
    "Used only by Alice for Python development.",
    "Built Python pipelines?",
    "Used Python for any project?",
    "Automated experiment execution and analysis using Python, Pandas and NumPy.",
    "Built Python pipelines for reporting.",
    "Extended a Python framework with a custom neural-network surrogate.",
    "Automated Python reports help Alice track sales.",
    "Supported Python platforms run on Windows.",
])
def test_unsafe_or_non_achievement_text_is_never_normalized_or_promoted(quote):
    assert matching._nli_premise(quote) == (quote, False)


@pytest.mark.parametrize("quote", [
    "Alice automated Python reporting.", "ALICE built Python tools.",
    "My colleague built Python services.", "They built Python reports.",
    "The team built Python tools.",
])
def test_other_peoples_work_cannot_be_assigned_to_candidate_by_entailing_fake(quote):
    matcher, result = assess(job("Python is required."), quote, FakeClassifier(prediction(entailment=1, contradiction=0)))
    assert result.requirements[0].status == "UNKNOWN"
    assert not matcher.classifier.calls


def test_python_like_does_not_receive_python_based_anchor_expansion():
    matcher, result = assess(job("Python is required."), "Built a Python-like teaching language.")
    assert result.requirements[0].status == "UNKNOWN"
    assert not matcher.classifier.calls


@pytest.mark.parametrize("quote", ["Contributed to Python reporting pipelines.", "I built Python pipelines for reporting."])
def test_coherent_achievements_rank_before_short_keyword_fragments(quote):
    cv = "Python\nProgramming: Python, SQL.\nPython project\n" + quote
    matcher, result = assess(job("Python is required."), cv)
    assert matcher.classifier.calls[0][0] == matching._nli_premise(quote)[0]
    assert result.requirements[0].cv_evidence == [quote]
    assert len(matcher.classifier.calls) == 1


def test_overlapping_windows_do_not_crowd_out_distinct_achievement_quotes():
    first = "Worked on Python reporting pipelines."
    second = "Contributed to complex processing jobs using Python scripts."
    cv = "Python\n" + first + "\nPython projects\n" + second
    classifier = FakeClassifier(prediction(entailment=0, contradiction=0))
    matcher, _ = assess(job("Python is required."), cv, classifier)
    first_two = [context for context, _ in matcher.classifier.calls[:2]]
    assert first_two == [matching._nli_premise(first)[0], matching._nli_premise(second)[0]]


def test_extraction_includes_degree_and_years_beyond_skill_keywords():
    _, result = assess(job("BSc in Computer Science is required.", "At least three years of Python experience is required."),
                       "Education\nBSc in Computer Science\nNorthern University | 2018 - 2021\nBuilt Python services for a client.")
    assert len(result.requirements) == 2
    assert result.requirements[0].status == "MET"
    assert result.requirements[1].status == "UNKNOWN"
    assert "duration" in result.requirements[1].explanation
    assert result.recommended_verdict == "REVIEW"


@pytest.mark.parametrize("cv", ["Skills: Python, SQL.", "Python\nBuilt SQL pipelines.", "I want to learn Python.", "I have no Python experience."])
def test_skill_mentions_aspirations_and_negation_cannot_prove_experience(cv):
    matcher, result = assess(job("Professional Python experience is required."), cv)
    assert result.requirements[0].status == "UNKNOWN"
    assert result.requirements[0].cv_evidence == []
    assert result.uncertainties
    assert not matcher.classifier.calls


def test_positive_explicit_professional_evidence_can_be_verified():
    _, result = assess(job("Professional Python experience is required."), "Built production Python pipelines for a commercial client.")
    assert result.requirements[0].status == "MET"


@pytest.mark.parametrize("cv", [
    "Coursework: Built Python systems.",
    "Built Python systems as a side project.",
    "University project: Built Python systems for a simulated client.",
    "Built Python systems.",
    "Personal projects:\nBuilt Python systems for a commercial client.",
])
def test_perfect_nli_cannot_turn_project_work_into_professional_experience(cv):
    classifier = FakeClassifier(prediction(entailment=1, contradiction=0))
    matcher, result = assess(job("Commercial Python experience is required."), cv, classifier)
    assert result.requirements[0].status == "UNKNOWN"
    assert result.uncertainties
    assert not matcher.classifier.calls


@pytest.mark.parametrize("eligibility", [
    "British citizens only", "UK nationals only", "Eligible for security clearance",
    "Full driving licence", "Fluent German", "Right to work in the UK",
])
def test_unheaded_eligibility_is_retained_and_never_inferred_from_cv(eligibility):
    record = {"description": INTRO + "\n" + eligibility, "metadata": {"description_type": "full"}}
    matcher, result = assess(record, "London, UK.\nBritish citizen.\nFluent German.\nFull driving licence.",
                             FakeClassifier(prediction(entailment=1, contradiction=0)))
    assert any(requirement.requirement == eligibility for requirement in result.requirements)
    assert all(requirement.status == "UNKNOWN" for requirement in result.requirements)
    assert result.uncertainties
    assert not matcher.classifier.calls


@pytest.mark.parametrize("section", ["What we're looking for:", "Additional candidate criteria:", "Candidate qualities"])
def test_unrecognized_sections_cannot_hide_bare_skill_obligations(section):
    record = {"description": INTRO + "\nPython is required.\n" + section + "\n- Rust\n- SQL",
              "metadata": {"description_type": "full"}}
    _, result = assess(record)
    assert {"Rust", "SQL"}.issubset({requirement.requirement for requirement in result.requirements})
    assert result.recommended_verdict == "REVIEW"
    assert result.uncertainties


def test_unclassified_bare_bullet_is_not_silently_omitted():
    record = {"description": INTRO + "\nPython is required.\n- Rust", "metadata": {"description_type": "full"}}
    _, result = assess(record)
    assert any(requirement.requirement == "Rust" and requirement.status == "UNKNOWN" for requirement in result.requirements)
    assert result.uncertainties


@pytest.mark.parametrize("bullet", ["1. ", "2) ", "- ", "• "])
def test_ordered_and_unordered_obligations_preserve_complete_source_quotes(bullet):
    record = {"description": INTRO + "\nPython is required.\n" + bullet + "Rust",
              "metadata": {"description_type": "full"}}
    _, result = assess(record)
    requirement = next(item for item in result.requirements if item.requirement == "Rust")
    assert requirement.job_evidence in record["description"]
    assert requirement.status == "UNKNOWN"
    assert result.uncertainties


@pytest.mark.parametrize("cv", ["I don't know Python.", "I don’t know Python.", "Missing Python skills.", "I am learning Python."])
def test_negative_or_aspirational_evidence_cannot_prove_a_simple_skill(cv):
    matcher, result = assess(job("Python is required."), cv, FakeClassifier(prediction(entailment=1, contradiction=0)))
    assert result.requirements[0].status == "UNKNOWN"
    assert not matcher.classifier.calls


def test_unfinished_study_cannot_prove_a_completed_degree():
    matcher, result = assess(job("BSc in Computer Science is required."), "Currently studying a BSc in Computer Science.",
                             FakeClassifier(prediction(entailment=1, contradiction=0)))
    assert result.requirements[0].status == "UNKNOWN"
    assert not matcher.classifier.calls


def test_known_benefits_section_does_not_turn_benefit_bullets_into_obligations():
    _, result = assess(job("Python is required.\nBenefits:\n- Coffee\n- Generous holiday allowance"))
    assert [requirement.requirement for requirement in result.requirements] == ["Python is required."]
    assert result.recommended_verdict == "APPLY"


FORM_TAIL = (
    "Create a Job Alert\n"
    "Interested in building your career at Graham Capital Management, L.P.? "
    "Get future opportunities sent straight to your email.\n*\n"
    "indicates a required field\n"
    "Accepted file types: pdf, doc, docx, txt, rtf\n"
    "Accepted file types: pdf, doc, docx, txt, rtf\n"
    "Click here to view GCM’s Privacy Notice.\nPowered by"
)


def test_corroborated_form_tail_is_not_scored_but_requires_boundary_review():
    record = job("Python is required.\n" + FORM_TAIL)
    source = record["description"]
    matcher, result = assess(record)
    assert [item.requirement for item in result.requirements] == ["Python is required."]
    assert result.requirements[0].status == "MET"
    assert result.requirements[0].job_evidence in source
    assert result.recommended_verdict == "REVIEW"
    assert any("application-form UI tail" in note and "unchanged source" in note for note in result.uncertainties)
    assert matcher.diagnostics["excluded_form_tail"]["source_start"] == source.index("Create a Job Alert")
    assert matcher.diagnostics["excluded_form_tail"]["characters"] == len(FORM_TAIL)
    assert record["description"] == source
    assert all("required field" not in hypothesis for _, hypothesis in matcher.classifier.calls)


@pytest.mark.parametrize("tail", [
    "Create a Job Alert\nSQL is required.",
    "Create a Job Alert\nAccepted file types: pdf, doc\nAccepted file types: pdf, doc\nSQL is required.",
    "Apply your skills\nindicates a required field\nAccepted file types: pdf, doc\nSQL is required.",
    "We create a job alert system.\nindicates a required field\nAccepted file types: pdf, doc\nSQL is required.",
])
def test_unproven_ui_boundaries_do_not_discard_subsequent_requirements(tail):
    matcher, result = assess(job("Python is required.\n" + tail))
    assert any(item.requirement == "SQL is required." for item in result.requirements)
    assert "excluded_form_tail" not in matcher.diagnostics


def test_role_content_after_corroborated_form_boundary_cannot_create_false_apply():
    _, result = assess(job("Python is required.\n" + FORM_TAIL + "\nRust is required."),
                       classifier=FakeClassifier(prediction(entailment=1, contradiction=0)))
    assert result.requirements[0].status == "MET"
    assert result.recommended_verdict == "REVIEW"
    assert any("no role content was omitted" in note for note in result.uncertainties)


@pytest.mark.parametrize("requirement,cv", [
    ("Degree in CS or Maths is required.", "Maths skills."),
    ("BSc or MSc in Mathematics is required.", "BSc in Physics."),
    ("Professional Python or SQL experience is required.", "Python skills."),
])
def test_shared_qualifiers_are_never_dropped_from_compound_requirements(requirement, cv):
    matcher, result = assess(job(requirement), cv, FakeClassifier(prediction(entailment=1, contradiction=0)))
    assert result.requirements[0].status == "UNKNOWN"
    assert not result.requirements[0].cv_evidence
    assert not matcher.classifier.calls


def test_python_or_java_is_one_group_and_accepts_supported_alternative():
    _, result = assess(job("Python or Java is required."), "Built Java services for customers.")
    assert len(result.requirements) == 1
    assert result.requirements[0].status == "MET"


def test_python_and_sql_requires_evidence_for_both_members():
    _, result = assess(job("Python and SQL are required."), "Built Python pipelines.")
    assert result.requirements[0].status == "PARTIAL"
    assert result.requirements[0].cv_evidence == ["Built Python pipelines."]
    assert result.uncertainties
    _, complete = assess(job("Python and SQL are required."), "Built Python pipelines.\nWrote SQL reporting queries.")
    assert complete.requirements[0].status == "MET"


def test_qualified_list_keeps_qualifiers_and_credits_only_verified_components():
    matcher, result = assess(job("Knowledge of Python and SQL is required."), "Python skills.")
    item = result.requirements[0]
    assert item.status == "PARTIAL"
    assert item.cv_evidence == ["Python skills."]
    assert matcher.classifier.calls[0][1] == "The candidate has Knowledge of Python."
    assert result.recommended_verdict == "REVIEW"


def test_shared_experience_list_can_recover_partial_evidence_with_spelling_variants():
    cv = "Applied statistical testing to experiment results.\nWorked on optimisation methods."
    requirement = "Experience with statistics, optimization and time series"
    matcher, result = assess(job(requirement), cv)
    item = result.requirements[0]
    assert item.status == "PARTIAL"
    assert item.job_evidence == requirement
    assert len(item.cv_evidence) == 2
    assert all(quote in cv for quote in item.cv_evidence)
    assert all("Experience with" in hypothesis for _, hypothesis in matcher.classifier.calls)
    assert result.recommended_verdict == "REVIEW"


def test_open_ended_list_never_becomes_met_even_if_every_named_skill_is_verified():
    matcher, result = assess(job("Experience with Python, SQL, etc."),
                             "Worked with Python.\nWorked with SQL.")
    assert matcher.diagnostics["nli_pairs"] > 0
    assert result.requirements[0].status == "PARTIAL"
    assert "unspecified remainder" in result.requirements[0].explanation
    assert result.recommended_verdict == "REVIEW"


def test_shared_professional_qualifier_cannot_be_satisfied_by_coursework():
    matcher, result = assess(job("Professional experience with Python and SQL"),
                             "Coursework: Built Python and SQL pipelines.")
    assert result.requirements[0].status == "UNKNOWN"
    assert not matcher.classifier.calls


def test_neutral_nli_on_decomposed_skills_still_gives_no_positive_credit():
    matcher, result = assess(job("Experience with Python and SQL"),
                             "Worked with Python.\nWorked with SQL.",
                             FakeClassifier(prediction(entailment=0.2, contradiction=0.1)))
    assert matcher.classifier.calls
    assert result.requirements[0].status == "UNKNOWN"
    assert not result.requirements[0].cv_evidence


def test_narrative_heading_and_employer_background_are_not_scored_requirements():
    text = (
        "Specializing in reliable systems, Example is dedicated to delivering strong results.\n"
        "The foundation of our success is the experience of our people.\n"
        "Description\nOur company is seeking a software engineer.\n"
        "We build products for customers.\n"
        "Python is required.\nResponsibilities\nDevelop SQL services.\n"
        "Requirements\nExperience with production serving is a great plus."
    )
    units = matching._requirements(text)
    assert [unit.text for unit in units] == [
        "Python is required.", "Develop SQL services.",
        "Experience with production serving is a great plus.",
    ]
    assert [unit.importance for unit in units] == ["CORE", "IMPORTANT", "PREFERRED"]
    assert all(unit.quote in text for unit in units)


@pytest.mark.parametrize("heading", ["Description", "Job description", "About the role"])
def test_narrative_sections_still_preserve_candidate_obligations(heading):
    units = matching._requirements(heading + "\nYou must know Python.\nFluent German\n- Rust")
    assert {"You must know Python.", "Fluent German", "Rust"}.issubset({unit.text for unit in units})


@pytest.mark.parametrize("requirement", [
    "Extensive commercial experience with Python.", "A degree in Computer Science.",
    "Hands-on experience with Python.", "Three years of relevant Python experience.",
    "Our applicants need experience with Python.",
])
def test_unheaded_qualification_prefixes_are_not_silently_lost(requirement):
    units = matching._requirements("Description:\n" + requirement + "\nRequirements:\nSQL is required.")
    assert requirement in {unit.text for unit in units}


def test_internal_preference_cannot_waive_a_following_obligation():
    _, result = assess(job("SQL is required.", "Python preferred and Rust."),
                       "Worked with SQL.\nWorked with Python.")
    assert result.requirements[1].importance == "CORE"
    assert result.requirements[1].status == "UNKNOWN"
    assert result.recommended_verdict == "REVIEW"


def test_parenthesized_language_preference_does_not_downgrade_stack_requirement():
    clause = "Full-stack experience with Python (preferred) or C++ and SQL"
    _, result = assess(job(clause), "Worked with Python, C++ and SQL.")
    assert result.requirements[0].importance == "CORE"
    assert result.requirements[0].status == "UNKNOWN"
    assert "scope" in result.requirements[0].explanation


def test_whole_clause_great_plus_is_optional_even_under_requirements_heading():
    _, result = assess(job("Experience writing production code for multi-client systems is a great plus"),
                       "Worked on unrelated Java applications.")
    assert result.requirements[0].importance == "PREFERRED"
    assert result.requirements[0].status == "MISSING"


def test_complex_conjunction_is_not_declared_met_from_one_skill():
    matcher, result = assess(job("Python and the ability to lead a distributed engineering department are required."))
    assert result.requirements[0].status == "UNKNOWN"
    assert "conjunction" in result.requirements[0].explanation
    assert not matcher.classifier.calls


@pytest.mark.parametrize("requirement,cv", [
    ("Advanced Python or SQL is required.", "SQL skills."),
    ("Expertise in Python or Java is required.", "Java skills."),
    ("Maintain scalable Python and SQL services.", "SQL services."),
])
def test_fallback_cannot_drop_shared_qualifiers_or_actions(requirement, cv):
    matcher, result = assess(job(requirement), cv,
                             FakeClassifier(prediction(entailment=1, contradiction=0)))
    assert result.requirements[0].status == "UNKNOWN"
    assert not matcher.classifier.calls


def test_explicit_waiver_is_not_promoted_to_a_required_skill():
    _, result = assess(job("No Python experience is required.", "SQL is required."), "Built SQL reports.")
    assert result.requirements[0].importance == "NOT_REQUIREMENT"
    assert result.requirements[1].status == "MET"


def test_mixed_negation_cannot_waive_another_required_skill():
    _, result = assess(job("Python is not required but Java is required."), "Built Python projects.")
    assert result.requirements[0].importance == "CORE"
    assert result.requirements[0].status == "UNKNOWN"
    assert result.uncertainties


@pytest.mark.parametrize("requirement", [
    "Python experience is not required if you know Java.",
    "No Python experience is required unless working on production systems.",
    "Python experience is not required except for senior roles.",
])
def test_conditional_waiver_cannot_remove_a_real_obligation(requirement):
    _, result = assess(job("SQL is required.", requirement), "Worked with SQL.")
    assert result.requirements[1].importance == "CORE"
    assert result.requirements[1].status == "UNKNOWN"
    assert result.recommended_verdict == "REVIEW"


def test_familiarity_under_requirements_is_not_discarded_as_contextual():
    _, result = assess(job("Familiarity with Kubernetes."), "Kubernetes skills.")
    assert result.requirements[0].importance == "CORE"
    assert result.requirements[0].status == "MET"


def test_preference_heading_has_no_scored_heading_and_missing_optional_skill_is_missing():
    _, result = assess(job("Python.", heading="Preferred skills"), "Java development.")
    assert len(result.requirements) == 1
    assert result.requirements[0].importance == "PREFERRED"
    assert result.requirements[0].status == "MISSING"
    assert result.uncertainties == []


def test_mixed_required_preferred_wording_is_ambiguous():
    _, result = assess(job("Python is required but Java is preferred."))
    assert result.requirements[0].status == "UNKNOWN"
    assert result.uncertainties


def test_generic_nli_entailment_cannot_equate_java_with_javascript():
    matcher, result = assess(job("Java is required."), "Built JavaScript browser applications.")
    assert result.requirements[0].status == "UNKNOWN"
    assert not matcher.classifier.calls


def test_degree_field_cannot_be_met_by_a_different_degree():
    _, result = assess(job("BSc in Computer Science is required."), "BSc in Mathematics.")
    assert result.requirements[0].status == "UNKNOWN"


def test_pairs_are_short_full_quotes_and_never_the_whole_cv():
    cv = "Name Example\n" + "\n".join(f"Unrelated entry number {i}." for i in range(30)) + "\nBuilt Python pipelines."
    matcher, result = assess(job("Python experience is required."), cv)
    assert result.requirements[0].status == "MET"
    assert all(len(context) <= matching.MAX_WINDOW_CHARS + len("The candidate ") for context, _ in matcher.classifier.calls)
    assert all(quote in cv for quote in result.requirements[0].cv_evidence)
    assert matcher.classifier.calls[0][0] == matching._nli_premise(result.requirements[0].cv_evidence[0])[0]
    assert all(context != cv for context, _ in matcher.classifier.calls)


def test_pair_budget_exhaustion_is_reported_and_prevents_apply(monkeypatch):
    monkeypatch.setattr(matching, "MAX_PAIRS", 1)
    matcher, result = assess(job("Python is required.", "SQL is required."), "Built Python pipelines.\nSQL skills.")
    assert len(matcher.classifier.calls) == 1
    assert matcher.diagnostics["pair_budget_exhausted"] is True
    assert result.recommended_verdict == "REVIEW"
    assert any("budget" in note for note in result.uncertainties)


def test_long_cv_unit_is_not_silently_truncated_or_ignored():
    matcher, result = assess(job("Python is required."), "Built Python systems " + "word " * 200)
    assert result.recommended_verdict == "REVIEW"
    assert any("not truncated" in note for note in result.uncertainties)
    assert not matcher.classifier.calls


def test_overlong_tokenized_pairs_need_review():
    classifier = FakeClassifier(NLIInputTooLong("No input was truncated"))
    matcher, result = assess(job("Python is required."), classifier=classifier)
    assert result.recommended_verdict == "REVIEW"
    assert matcher.diagnostics["overlong_pairs"] > 0
    assert any("token allowance" in note for note in result.uncertainties)


def test_excess_requirement_count_is_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(matching, "MAX_REQUIREMENTS", 1)
    matcher, result = assess(job("Python is required.", "SQL is required."))
    assert result.requirements == []
    assert result.recommended_verdict == "REVIEW"
    assert not matcher.classifier.calls


def test_repeated_requirements_are_merged_with_stronger_importance():
    _, result = assess(job("Python.\nPreferred skills:\nPython."))
    assert len(result.requirements) == 1
    assert result.requirements[0].importance == "CORE"


def test_constraint_checks_are_delegated_without_generation(monkeypatch):
    expected = ConstraintCheck(constraint_id="sponsorship", status="UNKNOWN", job_evidence=None, explanation="Not stated")
    observed = []
    def check(record, profile):
        observed.append((record, profile))
        return [expected]
    monkeypatch.setattr("app.applications.classifier_constraints.check_constraints", check)
    record = job("Python is required.")
    profile = CandidateProfile(preferences=Preferences(requires_sponsorship=True))
    result = matching.ClassifierMatcher(classifier=FakeClassifier()).assess(record, profile, "Python skills.")
    assert result.constraint_checks == [expected]
    assert observed == [(record, profile)]
    assert result.recommended_verdict == "REVIEW"


@pytest.mark.parametrize("bad", [
    {"label": "LABEL_1", "confidence": 0.99, "probabilities": {"LABEL_1": 0.99}},
    {"label": "entailment", "confidence": 0.99, "probabilities": {"entailment": float("nan"), "neutral": 0.01, "contradiction": 0}},
    {"label": "neutral", "confidence": 0.97, "probabilities": {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01}},
])
def test_bad_classifier_labels_or_probabilities_stop_matching(bad):
    with pytest.raises(ModelProviderError, match="invalid/ambiguous"):
        assess(job("Python is required."), classifier=FakeClassifier(bad))


def test_neutral_nli_result_never_becomes_a_positive_match():
    _, result = assess(job("Python is required."), classifier=FakeClassifier(prediction(entailment=0.2, contradiction=0.1)))
    assert result.requirements[0].status == "UNKNOWN"


def test_default_matcher_is_lazy_and_fingerprint_includes_resolved_configuration(monkeypatch):
    monkeypatch.setenv("REQUIREMENT_NLI_REVISION", "cached-revision")
    monkeypatch.setenv("REQUIREMENT_NLI_DEVICE", "cpu")
    matcher = matching.ClassifierMatcher()
    assert matcher.classifier.model is None
    assert matcher.classifier.local_files_only is True
    assert matcher.fingerprint["classifier"]["revision"] == "cached-revision"
    assert matcher.fingerprint["classifier"]["device"] == "cpu"
    json.dumps(matcher.fingerprint)


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self


class FakeTokenizer:
    model_max_length = 512

    def __init__(self, count=20):
        self.count = count
        self.calls = []

    def __call__(self, context, hypothesis, **kwargs):
        self.calls.append((context, hypothesis, kwargs))
        return {"input_ids": Tensor([1] * self.count)}


class FakeModel:
    def __init__(self, labels=None):
        self.config = SimpleNamespace(id2label=labels or {0: "contradiction", 1: "entailment", 2: "neutral"}, num_labels=3, max_position_embeddings=512)
        self.calls = 0
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=Tensor([0.01, 0.97, 0.02]))


@pytest.fixture
def fake_torch(monkeypatch):
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value
    fake = SimpleNamespace(inference_mode=nullcontext, softmax=lambda logits, dim: [[Scalar(value) for value in logits.values]])
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_adapter_uses_full_pair_cpu_explicit_label_mapping(fake_torch):
    tokenizer, model = FakeTokenizer(), FakeModel()
    classifier = RequirementNLIClassifier(tokenizer=tokenizer, model=model)
    actual = classifier.classify("Worked with Python.", "The candidate has Python skills.")
    assert actual["label"] == "entailment"
    assert actual["confidence"] == 0.97
    assert model.device == "cpu"
    assert tokenizer.calls[0][2]["truncation"] is False
    assert "max_length" not in tokenizer.calls[0][2]


def test_adapter_rejects_overlong_pairs_before_model_forward(fake_torch):
    tokenizer, model = FakeTokenizer(count=257), FakeModel()
    classifier = RequirementNLIClassifier(tokenizer=tokenizer, model=model)
    with pytest.raises(NLIInputTooLong, match="no input was truncated"):
        classifier.classify("Complete CV window", "Complete requirement")
    assert model.calls == 0


@pytest.mark.parametrize("labels", [{0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, {0: "entailment", 1: "entailment", 2: "neutral"}, {0: "entailment", 1: "neutral"}])
def test_adapter_rejects_ambiguous_label_configuration(fake_torch, labels):
    classifier = RequirementNLIClassifier(tokenizer=FakeTokenizer(), model=FakeModel(labels))
    with pytest.raises(ModelProviderError, match="label mappings"):
        classifier.classify("context", "hypothesis")


def test_adapter_enforces_local_only_and_revision_during_load(fake_torch, monkeypatch):
    calls = []
    def tokenizer(model_name, **kwargs):
        calls.append(("tokenizer", model_name, kwargs))
        return FakeTokenizer()
    def model(model_name, **kwargs):
        calls.append(("model", model_name, kwargs))
        return FakeModel()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model),
    ))
    classifier = RequirementNLIClassifier(revision="cached-version")
    assert calls == []
    classifier.classify("context", "hypothesis")
    assert len(calls) == 2
    for _, _, options in calls:
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
        assert options["revision"] == "cached-version"


def test_missing_local_weights_has_sanitized_error_without_download_fallback(fake_torch, monkeypatch):
    calls = []
    def missing(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("private directory or remote error details")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=missing),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=missing),
    ))
    with pytest.raises(ModelProviderError, match="automatic model downloads are disabled") as error:
        RequirementNLIClassifier().classify("context", "hypothesis")
    assert len(calls) == 1
    assert "private directory" not in str(error.value)


def test_importing_adapter_does_not_import_torch_or_transformers(monkeypatch):
    # Reload only this small adapter with heavyweight dependencies unavailable.
    module = importlib.import_module("app.classifiers.requirement_nli")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    # The module body has no heavyweight import; construction must stay lazy too.
    classifier = module.RequirementNLIClassifier()
    assert classifier.model is None
