import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.scraper.cleaner import JobCleaner
from app.scraper.sources.remotive import RemotiveSource
from app.scraper.sources.the_muse import TheMuseSource
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


@pytest.mark.parametrize("source_name", ["the_muse", "remotive"])
def test_collection_saves_valid_job_and_rejects_short_one(tmp_path, source_name):
    session = Mock()
    if source_name == "the_muse":
        session.get.side_effect = [
            FakeResponse(
                json.loads(
                    (FIXTURES / "the_muse_page_0.json").read_text(encoding="utf-8")
                )
            ),
            FakeResponse(
                json.loads(
                    (FIXTURES / "the_muse_page_1.json").read_text(encoding="utf-8")
                )
            ),
        ]
        source = TheMuseSource(session=session)
        location = "London, United Kingdom"
    else:
        session.get.return_value = FakeResponse(
            json.loads(
                (FIXTURES / "remotive_jobs.json").read_text(encoding="utf-8")
            )
        )
        source = RemotiveSource(session=session)
        location = "UK"

    storage = JobStorage(
        raw_dir=str(tmp_path / source_name / "raw"),
        processed_dir=str(tmp_path / source_name / "processed"),
    )
    summary = JobCollectionService(
        storage=storage,
        cleaner=JobCleaner(),
    ).collect(
        source,
        query="data engineer",
        location=location,
        limit=5,
    )

    assert summary.discovered == 2
    assert summary.saved == 1
    assert summary.rejected == 1
    assert summary.failed == 0
