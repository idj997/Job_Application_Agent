from app.applications.central_requirements import (
    CentralMatcher,
    CentralRequirementSelection,
    select_central_requirements,
)
from app.applications.classifier_constraints import role_family
from app.applications.models import MatchAssessment, RequirementMatch
from app.scoring import decide


CV = (
    "Built Python regression and statistical analysis pipelines. "
    "Designed hypothesis-driven experiments and evaluated noisy measurements. "
    "Profiled Linux workloads using PMU counters and tracing."
)

ZOPA_JOB = (
    "Python is required. Regression or classification experience is required. "
    "Strong statistical fundamentals and hypothesis testing are required. "
    "Experimental design experience is required."
)

OPENAI_JOB = (
    "Performance profiling experience is required. "
    "You will diagnose distributed systems behavior. "
    "You will debug networking bottlenecks in large-scale systems."
)


def req(text, evidence, *, status="MET", central=True, importance="IMPORTANT"):
    return RequirementMatch(
        requirement=text,
        importance=importance,
        central=central,
        job_evidence=evidence,
        status=status,
        cv_evidence=[CV] if status in {"MET", "PARTIAL"} else [],
        explanation="Test evidence.",
    )


def assessment(requirements, family):
    return MatchAssessment(
        description_complete=True,
        requirements=requirements,
        constraint_checks=[],
        recommended_verdict="APPLY",
        rationale="Deterministic central-policy test.",
        uncertainties=[],
        central_policy_applied=True,
        profile_family=family,
    )


def test_zopa_style_data_science_role_is_direct_when_all_central_requirements_are_met():
    result = decide(
        assessment([
            req("Python", "Python is required.", importance="CORE"),
            req("Regression or classification", "Regression or classification experience is required.", importance="CORE"),
            req("Statistics and hypothesis testing", "Strong statistical fundamentals and hypothesis testing are required.", importance="CORE"),
            req("Experimental design", "Experimental design experience is required.", importance="CORE"),
        ], "data_science_applied_ml"),
        CV,
        ZOPA_JOB,
        set(),
        threshold=80,
        use_recommendation=False,
    )

    assert result.verdict == "APPLY"
    assert result.fit_class == "DIRECT"
    assert result.profile_family == "data_science_applied_ml"


def test_openai_platform_style_role_is_stretch_when_distributed_systems_is_a_central_gap():
    result = decide(
        assessment([
            req("Performance profiling", "Performance profiling experience is required.", importance="CORE"),
            req("Distributed systems", "You will diagnose distributed systems behavior.", status="MISSING"),
            req("Networking", "You will debug networking bottlenecks in large-scale systems.", status="MET"),
        ], "ai_ml_research_engineering"),
        CV,
        OPENAI_JOB,
        set(),
        threshold=80,
        use_recommendation=False,
    )

    assert result.verdict == "REVIEW"
    assert result.fit_class == "STRETCH"
    assert any("Distributed systems" in reason for reason in result.reasons)


def test_two_missing_central_requirements_are_poor_fit():
    result = decide(
        assessment([
            req("Performance profiling", "Performance profiling experience is required.", importance="CORE"),
            req("Distributed systems", "You will diagnose distributed systems behavior.", status="MISSING"),
            req("Networking", "You will debug networking bottlenecks in large-scale systems.", status="MISSING"),
        ], "ai_ml_research_engineering"),
        CV,
        OPENAI_JOB,
        set(),
        threshold=80,
        use_recommendation=False,
    )

    assert result.verdict == "SKIP"
    assert result.fit_class == "POOR"


def test_role_families_do_not_collapse_data_science_and_ml_engineering():
    assert role_family("Senior Data Scientist") == "data_science_applied_ml"
    assert role_family("Machine Learning Engineer") == "ai_ml_research_engineering"
    assert role_family("CPU Performance Engineer") == "cpu_system_performance"
    assert role_family("CUDA Engineer") is None
    assert role_family("Distributed Systems Engineer") is None


class FakeSelector:
    name = "fake"
    model = "test"

    def __init__(self, indexes):
        self.indexes = indexes

    def generate_structured(self, prompt, schema, system_prompt=None):
        return CentralRequirementSelection(requirement_indexes=self.indexes, rationale="test")


def test_central_selector_can_only_choose_existing_scored_requirements():
    base = MatchAssessment(
        description_complete=True,
        requirements=[
            req("Python", "Python is required.", central=False, importance="CORE"),
            req("Statistics", "Strong statistical fundamentals and hypothesis testing are required.", central=False, importance="CORE"),
        ],
        constraint_checks=[],
        recommended_verdict="APPLY",
        rationale="test",
        uncertainties=[],
    )
    chosen, _ = select_central_requirements(
        FakeSelector([0, 1]),
        {"title": "Data Scientist", "description": ZOPA_JOB},
        base,
    )
    assert chosen == {0, 1}
