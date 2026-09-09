import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.scraper.cleaner import JobCleaner
from app.scraper.exceptions import SourceConfigurationError
from app.scraper.sources.lever import LeverSource


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
    source = LeverSource(
        company_slug="example",
        company="Example Ltd",
        session=session,
        **kwargs,
    )
    return source, session


def test_discovery_paginates_and_filters_title_and_location():
    source, session = make_source(
        load_fixture("lever_page_1.json"),
        load_fixture("lever_page_2.json"),
    )
    source.MAX_RESULTS_PER_PAGE = 2

    references = source.discover_jobs(
        query="data engineer",
        location="London",
        limit=5,
    )

    assert [reference.external_id for reference in references] == ["lever-501"]
    assert session.get.call_count == 2
    assert session.get.call_args_list[0].kwargs["params"]["skip"] == 0
    assert session.get.call_args_list[1].kwargs["params"]["skip"] == 2
    assert session.get.call_args_list[0].kwargs["params"]["limit"] == 2


def test_source_specific_filters_are_sent_to_lever():
    source, session = make_source(
        load_fixture("lever_page_1.json"),
        commitment="Full-time",
        team="Data Platform",
        department="Engineering",
    )

    source.discover_jobs(limit=1)

    params = session.get.call_args.kwargs["params"]
    assert params["mode"] == "json"
    assert params["commitment"] == "Full-time"
    assert params["team"] == "Data Platform"
    assert params["department"] == "Engineering"


def test_extract_assembles_lists_and_normalizes_structured_fields():
    source, _ = make_source(load_fixture("lever_page_1.json"))
    reference = source.discover_jobs(limit=1)[0]

    job = source.extract(source.fetch_job(reference))
    cleaned = JobCleaner().clean(job)

    assert cleaned.title == "Senior Data Engineer"
    assert cleaned.company == "Example Ltd"
    assert cleaned.location == "London, United Kingdom"
    assert cleaned.external_id == "lever-501"
    assert cleaned.source == "lever"
    assert cleaned.salary == "£65,000 - £80,000 per year"
    assert cleaned.date_posted == "2026-08-17T18:56:11+00:00"
    assert "<h2>What you will do</h2>" in cleaned.raw_description
    assert "What you will do" in cleaned.description
    assert "Build production pipelines" in cleaned.description
    assert cleaned.metadata["commitment"] == "Full-time"
    assert cleaned.metadata["department"] == "Engineering"
    assert cleaned.metadata["team"] == "Data Platform"
    assert cleaned.metadata["workplace_type"] == "hybrid"
    assert cleaned.metadata["all_locations"] == [
        "London, United Kingdom",
        "Remote - United Kingdom",
    ]


def test_company_url_is_accepted_in_place_of_slug():
    source = LeverSource(
        company_slug="https://jobs.lever.co/example/",
        company="Example Ltd",
    )

    assert source.company_slug == "example"


def test_eu_region_uses_eu_api_host():
    source, session = make_source(load_fixture("lever_page_1.json"), region="eu")

    source.discover_jobs(limit=1)

    assert session.get.call_args.args[0].startswith("https://api.eu.lever.co/")


def test_missing_company_slug_fails_before_http_request(monkeypatch):
    monkeypatch.delenv("LEVER_COMPANY_SLUG", raising=False)
    session = Mock()
    source = LeverSource(company_slug="", session=session)

    with pytest.raises(SourceConfigurationError, match="company slug"):
        source.discover_jobs(limit=1)

    session.get.assert_not_called()


def test_invalid_region_fails_before_http_request():
    session = Mock()
    source = LeverSource(company_slug="example", region="moon", session=session)

    with pytest.raises(SourceConfigurationError, match="global.*eu"):
        source.discover_jobs(limit=1)

    session.get.assert_not_called()


def test_query_does_not_match_incidental_description_text():
    record = {
        "text": "Engineering Manager",
        "description": "You will manage data engineers.",
        "categories": {
            "team": "Platform",
            "department": "Engineering",
        },
    }

    assert not LeverSource._matches(record, "data engineer", None)
