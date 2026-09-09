from app.scraper.deduplicator import JobDeduplicator
from app.scraper.sources.base import ExtractedJob


def test_tracking_params_are_removed():
    url_a = (
        "https://example.com/jobs/123"
        "?utm_source=indeed&utm_medium=cpc"
    )

    url_b = "https://example.com/jobs/123"

    assert JobDeduplicator.is_same_url(
        url_a,
        url_b,
    )


def test_trailing_slash_is_removed():
    url_a = "https://example.com/jobs/123/"
    url_b = "https://example.com/jobs/123"

    assert JobDeduplicator.is_same_url(
        url_a,
        url_b,
    )


def test_different_jobs_are_not_same_url():
    url_a = "https://example.com/jobs/123"
    url_b = "https://example.com/jobs/456"

    assert not JobDeduplicator.is_same_url(
        url_a,
        url_b,
    )


def test_identical_jobs_have_same_content_fingerprint():
    job_a = ExtractedJob(
        title="Senior Data Engineer",
        company="Example Ltd",
        location="London",
        description="Build pipelines using Databricks and PySpark.",
        source="test",
        source_url="https://example.com/jobs/123",
    )

    job_b = ExtractedJob(
        title="Senior Data Engineer",
        company="Example Ltd",
        location="London",
        description="Build pipelines using Databricks and PySpark.",
        source="test",
        source_url="https://example.com/jobs/999",
    )

    assert JobDeduplicator.is_exact_duplicate(
        job_a,
        job_b,
    )