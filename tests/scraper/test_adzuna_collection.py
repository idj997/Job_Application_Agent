import json
from pathlib import Path
from unittest.mock import Mock

from app.scraper.cleaner import JobCleaner
from app.scraper.sources.adzuna import AdzunaSource
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


def test_collection_saves_valid_summaries_and_rejects_short_one(tmp_path):
    pages = [
        json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        for name in ("adzuna_page_1.json", "adzuna_page_2.json")
    ]
    session = Mock()
    session.get.side_effect = [FakeResponse(page) for page in pages]
    source = AdzunaSource(
        app_id="test-app-id",
        app_key="test-app-key",
        session=session,
    )
    source.MAX_RESULTS_PER_PAGE = 2
    storage = JobStorage(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )

    summary = JobCollectionService(
        storage=storage,
        cleaner=JobCleaner(),
    ).collect(source, query="data engineer", limit=3)

    assert summary.discovered == 3
    assert summary.saved == 2
    assert summary.rejected == 1
    assert summary.failed == 0
    assert len(list((tmp_path / "raw").glob("*.json"))) == 2
    assert len(list((tmp_path / "processed").glob("*.json"))) == 2
