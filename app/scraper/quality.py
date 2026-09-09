from __future__ import annotations

from app.scraper.exceptions import InvalidJobError
from app.scraper.sources.base import ExtractedJob


class JobQualityValidator:
    def __init__(self, minimum_description_length: int = 300):
        self.minimum_description_length = minimum_description_length

    def validate(self, job: ExtractedJob) -> None:
        if not job.title or not job.title.strip():
            raise InvalidJobError("Job title is missing")

        if not job.description or not job.description.strip():
            raise InvalidJobError("Job description is missing")

        if len(job.description.strip()) < self.minimum_description_length:
            raise InvalidJobError(
                "Job description is shorter than "
                f"{self.minimum_description_length} characters"
            )

    def is_valid(self, job: ExtractedJob) -> bool:
        try:
            self.validate(job)
        except InvalidJobError:
            return False
        return True
