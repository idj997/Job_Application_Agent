import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.scraper.cleaner import JobCleaner
from app.scraper.exceptions import SourceConfigurationError
from app.scraper.sources.greenhouse import GreenhouseSource


FIXTURES = Path(__file__).parents[1] / "fixtures" / "api"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def make_source(*responses, **kwargs):
    session = Mock()
    session.get.side_effect = [FakeResponse(response) for response in responses]
    source = GreenhouseSource(
        board_token="example",
        company="Example Ltd",
        session=session,
        **kwargs,
    )
    return source, session


def test_discovery_requests_full_content_and_filters_locally():
    source, session = make_source(load_fixture("greenhouse_jobs.json"))

    references = source.discover_jobs(
        query="data engineer",
        location="London",
        limit=5,
    )

    assert [reference.external_id for reference in references] == ["7001"]
    assert references[0].company == "Example Ltd"
    assert session.get.call_count == 1
    assert session.get.call_args.args[0].endswith("/example/jobs")
    assert session.get.call_args.kwargs["params"] == {"content": "true"}


def test_company_name_is_loaded_from_board_when_not_configured():
    session = Mock()
    session.get.side_effect = [
        FakeResponse(load_fixture("greenhouse_board.json")),
        FakeResponse(load_fixture("greenhouse_jobs.json")),
    ]
    source = GreenhouseSource(
        board_token="example",
        company="",
        session=session,
    )

    reference = source.discover_jobs(limit=1)[0]

    assert source.company == "Example Greenhouse Company"
    assert reference.company == "Example Greenhouse Company"
    assert session.get.call_args_list[0].args[0].endswith("/example")
    assert session.get.call_args_list[1].args[0].endswith("/example/jobs")


def test_extract_preserves_greenhouse_fields_and_full_raw_content():
    source, _ = make_source(load_fixture("greenhouse_jobs.json"))
    reference = source.discover_jobs(limit=1)[0]

    job = source.extract(source.fetch_job(reference))
    cleaned = JobCleaner().clean(job)

    assert cleaned.title == "Senior Data Engineer"
    assert cleaned.company == "Example Ltd"
    assert cleaned.location == "London, United Kingdom"
    assert cleaned.external_id == "7001"
    assert cleaned.source == "greenhouse"
    assert cleaned.date_posted is None
    assert cleaned.raw_description.startswith("&lt;p&gt;")
    assert "Join the data platform group" in cleaned.description
    assert "&lt;p&gt;" not in cleaned.description
    assert cleaned.metadata["updated_at"] == "2026-08-18T10:55:28Z"
    assert cleaned.metadata["internal_job_id"] == 17001
    assert cleaned.metadata["requisition_id"] == "DATA-101"
    assert cleaned.metadata["departments"][0]["name"] == "Data Engineering"
    assert cleaned.metadata["offices"][0]["name"] == "London"


def test_board_url_is_accepted_in_place_of_bare_token():
    source = GreenhouseSource(
        board_token="https://job-boards.greenhouse.io/example/",
        company="Example Ltd",
    )

    assert source.board_token == "example"


def test_missing_board_token_fails_before_http_request(monkeypatch):
    monkeypatch.delenv("GREENHOUSE_BOARD_TOKEN", raising=False)
    session = Mock()
    source = GreenhouseSource(board_token="", session=session)

    with pytest.raises(SourceConfigurationError, match="board token"):
        source.discover_jobs(limit=1)

    session.get.assert_not_called()


def test_query_does_not_match_incidental_description_text():
    record = {
        "title": "Engineering Manager, Search Storage",
        "content": "You will collaborate with machine learning engineers.",
        "departments": [{"name": "Backend Engineering"}],
    }

    assert not GreenhouseSource._matches(
        record,
        query="machine learning engineer",
        location=None,
    )
