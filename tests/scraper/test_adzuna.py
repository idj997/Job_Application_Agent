import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from app.scraper.exceptions import (
    SourceConfigurationError,
    SourceUnavailableError,
)
from app.scraper.sources.adzuna import AdzunaSource


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
    source = AdzunaSource(
        app_id="test-app-id",
        app_key="test-app-key",
        session=session,
        **kwargs,
    )
    return source, session


def test_discovery_paginates_and_sends_supported_filters():
    source, session = make_source(
        load_fixture("adzuna_page_1.json"),
        load_fixture("adzuna_page_2.json"),
        salary_min=60000.0,
        salary_max=80000,
        full_time=True,
        permanent=True,
    )
    source.MAX_RESULTS_PER_PAGE = 2

    references = source.discover_jobs(
        query="data engineer",
        location="London",
        limit=3,
    )

    assert [reference.external_id for reference in references] == ["501", "502", "503"]
    assert session.get.call_count == 2
    assert session.get.call_args_list[0].args[0].endswith("/gb/search/1")
    assert session.get.call_args_list[1].args[0].endswith("/gb/search/2")
    params = session.get.call_args_list[0].kwargs["params"]
    assert params["what"] == "data engineer"
    assert params["where"] == "London"
    assert params["salary_min"] == 60000
    assert params["salary_max"] == 80000
    assert params["full_time"] == "1"
    assert params["permanent"] == "1"
    assert params["results_per_page"] == 2


def test_extract_marks_description_as_summary_and_normalizes_salary():
    source, _ = make_source(load_fixture("adzuna_page_1.json"))
    reference = source.discover_jobs(limit=1)[0]

    job = source.extract(source.fetch_job(reference))

    assert job.title == "Senior Data Engineer"
    assert job.company == "Example Data Ltd"
    assert job.location == "London"
    assert job.external_id == "501"
    assert job.salary == "£60,000 - £75,000"
    assert job.source == "adzuna"
    assert job.raw_description == job.description
    assert job.metadata["description_type"] == "summary"
    assert job.metadata["contract_time"] == "full_time"
    assert job.metadata["contract_type"] == "permanent"
    assert job.metadata["category"]["tag"] == "it-jobs"
    assert job.metadata["salary_is_predicted"] == 0


def test_location_falls_back_to_reversed_area_hierarchy():
    source, _ = make_source(
        load_fixture("adzuna_page_1.json"),
        load_fixture("adzuna_page_2.json"),
    )
    source.MAX_RESULTS_PER_PAGE = 2
    reference = source.discover_jobs(limit=3)[2]

    job = source.extract(source.fetch_job(reference))

    assert job.location == "Edinburgh, Scotland, UK"


def test_missing_credentials_fail_before_an_http_request(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    session = Mock()
    source = AdzunaSource(app_id="", app_key="", session=session)

    with pytest.raises(SourceConfigurationError, match="ADZUNA_APP_ID"):
        source.discover_jobs(limit=1)

    session.get.assert_not_called()


def test_request_failure_does_not_expose_credentials():
    session = Mock()
    session.get.side_effect = requests.HTTPError(
        "401 for https://api.adzuna.com?app_key=super-secret"
    )
    source = AdzunaSource(
        app_id="test-app-id",
        app_key="super-secret",
        session=session,
    )

    with pytest.raises(SourceUnavailableError) as exc_info:
        source.discover_jobs(limit=1)

    assert "super-secret" not in str(exc_info.value)


def test_fractional_salary_filter_is_rejected_before_request():
    with pytest.raises(SourceConfigurationError, match="salary_min must be an integer"):
        AdzunaSource(
            app_id="test-app-id",
            app_key="test-app-key",
            salary_min=60000.50,
        )


@pytest.mark.parametrize("location", ["UK", "U.K.", "United Kingdom", "Great Britain"])
def test_countrywide_uk_location_is_not_sent_as_a_where_filter(location):
    source, session = make_source(load_fixture("adzuna_page_1.json"))

    source.discover_jobs(
        query="data engineer",
        location=location,
        limit=1,
    )

    params = session.get.call_args.kwargs["params"]
    assert params["what"] == "data engineer"
    assert "where" not in params
