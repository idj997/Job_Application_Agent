from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Preferences(StrictModel):
    target_roles: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    work_patterns: list[Literal["remote", "hybrid", "onsite"]] = Field(default_factory=list)
    minimum_salary: int | None = Field(default=None, ge=0)
    salary_currency: str = "GBP"
    requires_sponsorship: bool | None = None
    hard_constraints: list[str] = Field(default_factory=list)

    def constraints(self) -> dict[str, str]:
        checks = {}
        if self.target_roles:
            checks["role"] = "Role must fit one of: " + ", ".join(self.target_roles)
        if self.locations:
            checks["location"] = "Must be able to work from: " + ", ".join(self.locations)
        if self.work_patterns:
            checks["work_pattern"] = "Acceptable working arrangements: " + ", ".join(self.work_patterns)
        if self.minimum_salary is not None:
            checks["salary"] = (
                f"Annual salary must allow at least {self.minimum_salary} {self.salary_currency}. "
                "Predicted salary, missing currency or unclear pay period is UNKNOWN."
            )
        if self.requires_sponsorship is True:
            checks["sponsorship"] = "Employer must explicitly offer suitable visa sponsorship."
        for index, constraint in enumerate(self.hard_constraints):
            checks[f"custom_{index + 1}"] = constraint
        return checks


class CandidateProfile(StrictModel):
    cv_path: str | None = None
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    preferences: Preferences = Field(default_factory=Preferences)


class RequirementMatch(StrictModel):
    requirement: str = Field(min_length=1)
    importance: Literal["CORE", "IMPORTANT", "PREFERRED", "CONTEXTUAL", "NOT_REQUIREMENT"]
    job_evidence: str = Field(min_length=1)
    status: Literal["MET", "PARTIAL", "MISSING", "UNKNOWN"]
    cv_evidence: list[str]
    explanation: str


class ConstraintCheck(StrictModel):
    constraint_id: str
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    job_evidence: str | None
    explanation: str


class MatchAssessment(StrictModel):
    description_complete: bool
    requirements: list[RequirementMatch]
    constraint_checks: list[ConstraintCheck]
    recommended_verdict: Literal["APPLY", "SKIP", "REVIEW"]
    rationale: str
    uncertainties: list[str]


class CVEntry(StrictModel):
    text: str = Field(min_length=1)
    style: Literal["paragraph", "bullet"]
    evidence_quotes: list[str] = Field(min_length=1)


class CVSection(StrictModel):
    heading: Literal[
        "Contact", "Professional Summary", "Experience", "Education", "Skills",
        "Projects", "Certifications", "Languages", "Publications", "Volunteering",
    ]
    entries: list[CVEntry] = Field(min_length=1)


class TailoredCV(StrictModel):
    sections: list[CVSection] = Field(min_length=1)


class CVEvaluation(StrictModel):
    requirement_coverage: int = Field(ge=0, le=100)
    clarity: int = Field(ge=0, le=100)
    factual_consistency: bool
    unsupported_claims: list[str]
    missing_critical_information: list[str]
    improvements: list[str]


class Decision(StrictModel):
    verdict: Literal["APPLY", "SKIP", "REVIEW"]
    score: int = Field(ge=0, le=100)
    reasons: list[str]
