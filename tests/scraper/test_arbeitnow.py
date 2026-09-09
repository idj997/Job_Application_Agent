import json
from pathlib import Path
from unittest.mock import Mock

from app.scraper.sources.arbeitnow import ArbeitnowSource


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
    return ArbeitnowSource(session=session, **kwargs), session


def test_discovery_filters_and_paginates_to_the_requested_limit():
    source, session = make_source(
        load_fixture("arbeitnow_page_1.json"),
        load_fixture("arbeitnow_page_2.json"),
    )

    references = source.discover_jobs(query="data engineer", limit=2)

    assert [reference.external_id for reference in references] == ["101", "201"]
    assert session.get.call_count == 2
    assert session.get.call_args_list[0].kwargs["params"]["page"] == 1
    assert session.get.call_args_list[1].kwargs["params"]["page"] == 2


def test_extract_normalizes_the_cached_api_record():
    source, _ = make_source(load_fixture("arbeitnow_page_1.json"))
    reference = source.discover_jobs(query="data engineer", limit=1)[0]

    job = source.extract(source.fetch_job(reference))

    assert job.title == "Data Engineer"
    assert job.company == "Example Analytics"
    assert job.location == "London, United Kingdom"
    assert job.external_id == "101"
    assert job.source == "arbeitnow"
    assert job.source_url == reference.url
    assert job.raw_description.startswith("<p>")
    assert job.metadata == {
        "remote": True,
        "visa_sponsorship": True,
        "tags": ["Python", "SQL", "Data"],
        "employment_type": ["full_time"],
    }


def test_visa_sponsorship_filter_is_sent_to_the_api():
    source, session = make_source(
        load_fixture("arbeitnow_page_1.json"),
        visa_sponsorship=True,
    )

    source.discover_jobs(limit=1)

    assert session.get.call_args.kwargs["params"]["visa_sponsorship"] == "true"


def test_epoch_created_at_is_normalized_to_iso_utc():
    payload = load_fixture("arbeitnow_page_1.json")
    payload["data"][0]["created_at"] = 1786992971
    source, _ = make_source(payload)
    reference = source.discover_jobs(limit=1)[0]

    job = source.extract(source.fetch_job(reference))

    assert job.date_posted == "2026-08-17T18:56:11+00:00"


def test_query_terms_must_form_a_phrase_not_appear_independently():
    record = {
        "title": "Customer Support",
        "company_name": "Example",
        "description": "Work with data supplied by an engineer.",
        "tags": [],
    }

    assert not ArbeitnowSource._matches(record, "data engineer", None)


def test_uk_location_filter_accepts_city_level_locations_from_uk_api():
    record = {"location": "London"}

    assert ArbeitnowSource._matches(record, None, "United Kingdom")
