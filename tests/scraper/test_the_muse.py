import json
from pathlib import Path
from unittest.mock import Mock

import requests

from app.scraper.cleaner import JobCleaner
from app.scraper.exceptions import SourceUnavailableError
from app.scraper.sources.the_muse import TheMuseSource


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
    return TheMuseSource(session=session, **kwargs), session


def test_discovery_uses_zero_based_pagination_and_local_query_filter():
    source, session = make_source(
        load_fixture("the_muse_page_0.json"),
        load_fixture("the_muse_page_1.json"),
    )

    references = source.discover_jobs(
        query="data engineer",
        location="London, United Kingdom",
        limit=5,
    )

    assert [reference.external_id for reference in references] == ["801", "803"]
    assert session.get.call_count == 2
    assert session.get.call_args_list[0].kwargs["params"]["page"] == 0
    assert session.get.call_args_list[1].kwargs["params"]["page"] == 1


def test_api_filters_and_optional_key_are_sent():
    source, session = make_source(
        load_fixture("the_muse_page_0.json"),
        api_key="test-key",
        category="Data and Analytics",
        level="Senior Level",
        company_filter="Example Muse Company",
    )

    source.discover_jobs(location="London, United Kingdom", limit=1)

    params = session.get.call_args.kwargs["params"]
    assert params["api_key"] == "test-key"
    assert params["category"] == "Data and Analytics"
    assert params["level"] == "Senior Level"
    assert params["company"] == "Example Muse Company"
    assert params["location"] == "London, United Kingdom"
    assert params["descending"] == "true"


def test_extract_normalizes_muse_job_and_preserves_landing_page():
    source, _ = make_source(load_fixture("the_muse_page_0.json"))
    reference = source.discover_jobs(limit=1)[0]

    job = source.extract(source.fetch_job(reference))
    cleaned = JobCleaner().clean(job)

    assert cleaned.title == "Senior Data Engineer"
    assert cleaned.company == "Example Muse Company"
    assert cleaned.location == "London, United Kingdom; Flexible / Remote"
    assert cleaned.external_id == "801"
    assert cleaned.source_url == reference.url
    assert cleaned.date_posted == "2026-08-18T10:00:00Z"
    assert cleaned.raw_description.startswith("<p>")
    assert cleaned.metadata["categories"] == [
        "Data and Analytics",
        "Software Engineering",
    ]
    assert cleaned.metadata["levels"] == ["Senior Level"]
    assert cleaned.metadata["company_id"] == 91


def test_request_error_does_not_expose_api_key():
    session = Mock()
    session.get.side_effect = requests.HTTPError(
        "403 for https://www.themuse.com/api/public/jobs?api_key=super-secret"
    )
    source = TheMuseSource(api_key="super-secret", session=session)

    try:
        source.discover_jobs(limit=1)
    except SourceUnavailableError as exc:
        assert "super-secret" not in str(exc)
    else:
        raise AssertionError("Expected SourceUnavailableError")
