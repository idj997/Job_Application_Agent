import pytest

from app.scraper.exceptions import InvalidJobError
from app.scraper.quality import JobQualityValidator
from app.scraper.sources.base import ExtractedJob


def test_quality_gate_accepts_a_realistic_description():
    job = ExtractedJob(title="Data Engineer", description="Useful detail. " * 30)
    assert JobQualityValidator().is_valid(job)


@pytest.mark.parametrize(
    "job",
    [
        ExtractedJob(description="A" * 400),
        ExtractedJob(title="Data Engineer"),
        ExtractedJob(title="Data Engineer", description="Apply now"),
    ],
)
def test_quality_gate_rejects_incomplete_extractions(job):
    with pytest.raises(InvalidJobError):
        JobQualityValidator().validate(job)
