import json
from pathlib import Path
from unittest.mock import Mock

from app.scraper.cleaner import JobCleaner
from app.scraper.sources.remotive import RemotiveSource


FIXTURES = Path(__file__).parents[1] / "fixtures" / "api"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def load_fixture():
    return json.loads(
        (FIXTURES / "remotive_jobs.json").read_text(encoding="utf-8")
    )


def make_source(**kwargs):
    session = Mock()
    session.get.return_value = FakeResponse(load_fixture())
    return RemotiveSource(session=session, **kwargs), session


def test_discovery_sends_filters_and_accepts_worldwide_for_location():
    source, session = make_source(
        category="software-dev",
        company_name="Remote Example",
    )

    references = source.discover_jobs(
        query="data engineer",
        location="UK",
        limit=5,
    )

    assert [reference.external_id for reference in references] == ["901", "903"]
    params = session.get.call_args.kwargs["params"]
    assert params["search"] == "data engineer"
    assert params["category"] == "software-dev"
    assert params["company_name"] == "Remote Example"
    assert "limit" not in params


def test_discovery_sends_limit_when_no_client_location_filter_is_needed():
    source, session = make_source()

    source.discover_jobs(limit=2)

    assert session.get.call_args.kwargs["params"]["limit"] == 2


def test_query_uses_server_search_but_filters_incidental_description_matches():
    source, session = make_source()

    references = source.discover_jobs(query="data engineer", limit=5)

    assert [reference.external_id for reference in references] == ["901", "903"]
    assert session.get.call_args.kwargs["params"]["search"] == "data engineer"
    assert "limit" not in session.get.call_args.kwargs["params"]


def test_extract_normalizes_remote_job_fields():
    source, _ = make_source()
    reference = source.discover_jobs(limit=1)[0]

    job = source.extract(source.fetch_job(reference))
    cleaned = JobCleaner().clean(job)

    assert cleaned.title == "Senior Data Engineer"
    assert cleaned.company == "Remote Example"
    assert cleaned.location == "Worldwide"
    assert cleaned.external_id == "901"
    assert cleaned.salary == "$80,000 - $100,000"
    assert cleaned.date_posted == "2026-08-18T10:23:26"
    assert cleaned.source_url.startswith("https://remotive.com/")
    assert cleaned.metadata == {
        "remote": True,
        "category": "Software Development",
        "employment_type": "full_time",
        "company_logo": "https://remotive.com/job/901/logo",
    }
