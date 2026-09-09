import json
from pathlib import Path
from unittest.mock import Mock

from app.scraper.cleaner import JobCleaner
from app.scraper.sources.greenhouse import GreenhouseSource
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


def test_collection_saves_full_job_and_rejects_short_extraction(tmp_path):
    session = Mock()
    session.get.return_value = FakeResponse(
        json.loads(
            (FIXTURES / "greenhouse_jobs.json").read_text(encoding="utf-8")
        )
    )
    source = GreenhouseSource(
        board_token="example",
        company="Example Ltd",
        session=session,
    )
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )

    summary = JobCollectionService(
        storage=storage,
        cleaner=JobCleaner(),
    ).collect(source, query="data engineer", limit=5)

    assert summary.discovered == 2
    assert summary.saved == 1
    assert summary.rejected == 1
    assert summary.failed == 0
    assert len(list((tmp_path / "raw").glob("*.json"))) == 1
    assert len(list((tmp_path / "processed").glob("*.json"))) == 1
