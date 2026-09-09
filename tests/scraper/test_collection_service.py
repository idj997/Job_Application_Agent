import json
from pathlib import Path
from unittest.mock import Mock

import requests

from app.scraper.cleaner import JobCleaner
from app.scraper.sources.base import (
    ExtractedJob,
    JobReference,
    JobSource,
    RawJobPage,
    RawJobPayload,
)
from app.scraper.sources.arbeitnow import ArbeitnowSource
from app.scraper.storage import JobStorage
from app.services.job_collection_service import JobCollectionService


FIXTURES = Path(__file__).parents[1] / "fixtures" / "api"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def load_pages():
    return [
        json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        for name in ("arbeitnow_page_1.json", "arbeitnow_page_2.json")
    ]


def make_source():
    session = Mock()
    session.get.side_effect = [FakeResponse(payload) for payload in load_pages()]
    return ArbeitnowSource(session=session)


class StaticSource(JobSource):
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url

    def discover_jobs(
        self,
        query: str | None = None,
        location: str | None = None,
        limit: int = 50,
    ) -> list[JobReference]:
        return [JobReference(source=self.name, url=self.url)][:limit]

    def fetch(self, url: str) -> RawJobPayload:
        return RawJobPayload(
            source=self.name,
            source_url=url,
            data={
                "title": "Data Engineer",
                "company": "Example Ltd",
                "location": "London",
                "description": "Detailed data engineering requirements. " * 20,
                "source": self.name,
                "source_url": url,
            },
        )

    def extract(self, raw: RawJobPage | RawJobPayload) -> ExtractedJob:
        assert isinstance(raw, RawJobPayload)
        assert isinstance(raw.data, dict)
        return ExtractedJob(**raw.data)


class BrokenSource(StaticSource):
    def discover_jobs(
        self,
        query: str | None = None,
        location: str | None = None,
        limit: int = 50,
    ) -> list[JobReference]:
        raise RuntimeError("unexpected discovery failure")


def test_collection_saves_jobs_rejects_low_quality_and_preserves_raw(tmp_path):
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    service = JobCollectionService(storage=storage, cleaner=JobCleaner())

    summary = service.collect(make_source(), query="data engineer", limit=3)

    assert summary.discovered == 3
    assert summary.saved == 2
    assert summary.rejected == 1
    assert summary.failed == 0
    assert len(list((tmp_path / "raw").glob("*.json"))) == 2

    processed_paths = list((tmp_path / "processed").glob("*.json"))
    assert len(processed_paths) == 2
    saved = json.loads(processed_paths[0].read_text(encoding="utf-8"))
    assert saved["raw_description"].startswith("<p>")
    assert "<p>" not in saved["description"]
    assert saved["metadata"]


def test_second_collection_counts_url_duplicates(tmp_path):
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    service = JobCollectionService(storage=storage, cleaner=JobCleaner())

    first = service.collect(make_source(), query="data engineer", limit=2)
    second = service.collect(make_source(), query="data engineer", limit=2)

    assert first.saved == 2
    assert second.saved == 0
    assert second.duplicates == 2


def test_source_failure_returns_a_summary_instead_of_crashing(tmp_path):
    session = Mock()
    session.get.side_effect = requests.ConnectionError("offline")
    source = ArbeitnowSource(session=session)
    service = JobCollectionService(
        storage=JobStorage(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        ),
        cleaner=JobCleaner(),
    )

    summary = service.collect(source, limit=1)

    assert summary.discovered == 0
    assert summary.failed == 1
    assert "offline" in summary.source_error


def test_collect_many_shares_deduplication_and_isolates_source_failures(tmp_path):
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    service = JobCollectionService(storage=storage, cleaner=JobCleaner())
    sources = [
        StaticSource("source-a", "https://source-a.example/jobs/123"),
        BrokenSource("broken", "https://broken.example/jobs/999"),
        StaticSource("source-b", "https://source-b.example/jobs/456"),
    ]

    result = service.collect_many(
        sources,
        query="data engineer",
        location="London",
        limit_per_source=1,
    )

    assert [summary.source for summary in result.summaries] == [
        "source-a",
        "broken",
        "source-b",
    ]
    assert result.discovered == 2
    assert result.saved == 1
    assert result.duplicates == 1
    assert result.rejected == 0
    assert result.failed == 1
    assert "unexpected discovery failure" in result.summaries[1].source_error
    assert len(list((tmp_path / "raw").glob("*.json"))) == 1
    assert len(list((tmp_path / "processed").glob("*.json"))) == 1
