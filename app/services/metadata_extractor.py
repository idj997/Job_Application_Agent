from pydantic import BaseModel


class JobMetadata(BaseModel):
    job_title: str | None = None
    company: str | None = None
    location: str | None = None

    salary_min: int | None = None
    salary_max: int | None = None
    currency: str | None = None

    minimum_experience_years: float | None = None
    experience_area: str | None = None

    work_pattern: str = "unknown"
    office_days_per_week: int | None = None