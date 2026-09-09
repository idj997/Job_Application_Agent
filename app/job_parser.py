from typing import Literal
from pydantic import BaseModel, Field


class JobDescription(BaseModel):

    job_title: str | None = None
    company: str | None = None
    location: str | None = None

    salary_min: int | None = None
    salary_max: int | None = None
    currency: str | None = None

    required_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical technical skill names only. "
            "Examples: SQL, PySpark, Azure Databricks, Delta Lake."
        )
    )

    preferred_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical technical skill names only. "
            "Do not include phrases such as 'experience with'."
        )
    )

    minimum_experience_years: float | None = None

    experience_area: str | None = None

    work_pattern: Literal[
        "remote",
        "hybrid",
        "onsite",
        "unknown"
    ] = "unknown"

    office_days_per_week: int | None = None

    sponsorship: Literal[
        "available",
        "unavailable",
        "unknown"
    ] = "unknown"

    security_clearance: Literal[
        "required",
        "not_required",
        "not_mentioned"
    ] = "not_mentioned"