import json

from app.scraper.cleaner import JobCleaner
from app.scraper.role_catalog import RoleFamily
from app.scraper.sources.base import (
    ExtractedJob,
    JobReference,
    JobSource,
    RawJobPage,
    RawJobPayload,
)
from app.scraper.storage import JobStorage
from app.services.job_collection_service import JobCollectionService
from app.services.role_corpus_service import RoleCorpusCollectionService


class QuerySource(JobSource):
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.query = ""

    def discover_jobs(
        self,
        query: str | None = None,
        location: str | None = None,
        limit: int = 50,
    ) -> list[JobReference]:
        if self.fail:
            raise RuntimeError("source offline")
        self.query = query or "unknown"
        slug = self.query.replace(" ", "-")
        return [JobReference(source=self.name, url=f"https://example.com/{slug}")]

    def fetch(self, url: str) -> RawJobPayload:
        return RawJobPayload(
            source=self.name,
            source_url=url,
            data={"query": self.query},
        )

    def extract(self, raw: RawJobPage | RawJobPayload) -> ExtractedJob:
        assert isinstance(raw, RawJobPayload)
        query = raw.data["query"]
        return ExtractedJob(
            title=query.title(),
            company="Example Ltd",
            location="London",
            description=f"Detailed responsibilities for {query}. " * 20,
            source=self.name,
            source_url=raw.source_url,
        )


def test_role_collection_isolates_a_source_and_records_query_provenance(tmp_path):
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    service = RoleCorpusCollectionService(
        JobCollectionService(storage=storage, cleaner=JobCleaner())
    )
    family = RoleFamily(
        name="data_engineering",
        display_name="Data Engineering",
        queries=("data engineer", "analytics engineer"),
    )

    result = service.collect(
        sources=[QuerySource("broken", fail=True), QuerySource("working")],
        role_families=[family],
        limit_per_query=1,
    )

    assert len(result.queries) == 3
    assert result.failed == 1
    assert result.saved == 2
    assert [item.collection.source for item in result.queries] == [
        "broken",
        "working",
        "working",
    ]

    saved = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "processed").glob("*.json")
    ]
    assert {item["metadata"]["collection"]["query"] for item in saved} == {
        "data engineer",
        "analytics engineer",
    }
    assert all(
        item["metadata"]["collection"]["role_family"] == "data_engineering"
        for item in saved
    )

