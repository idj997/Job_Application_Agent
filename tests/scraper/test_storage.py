import json

from app.scraper.sources.base import ExtractedJob, RawJobPayload
from app.scraper.storage import JobStorage


def test_storage_saves_api_payload_and_processed_metadata(tmp_path):
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    url = "https://example.com/jobs/123?utm_source=test"
    raw = RawJobPayload(
        source="test",
        source_url=url,
        data={"id": 123, "description": "raw"},
    )
    job = ExtractedJob(
        external_id="123",
        title="Data Engineer",
        company="Example Ltd",
        location="London",
        raw_description="<p>raw</p>",
        description="Detailed requirements. " * 20,
        source="test",
        source_url=url,
        metadata={"remote": True},
    )

    raw_path = storage.save_raw_payload(raw)
    processed_path = storage.save_job(job)
    payload = json.loads(processed_path.read_text(encoding="utf-8"))

    assert json.loads(raw_path.read_text(encoding="utf-8"))["id"] == 123
    assert payload["external_id"] == "123"
    assert payload["canonical_url"] == "https://example.com/jobs/123"
    assert payload["content_hash"]
    assert payload["scraped_at"]


def test_storage_detects_an_exact_content_duplicate_at_another_url(tmp_path):
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    original = ExtractedJob(
        title="Data Engineer",
        company="Example Ltd",
        location="London",
        description="Detailed requirements. " * 20,
        source="source-a",
        source_url="https://source-a.example/jobs/123",
    )
    duplicate = ExtractedJob(
        title=original.title,
        company=original.company,
        location=original.location,
        description=original.description,
        source="source-b",
        source_url="https://source-b.example/vacancies/456",
    )

    storage.save_job(original)

    assert storage.duplicate_kind(duplicate) == "content"
