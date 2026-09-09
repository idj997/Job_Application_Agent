import pytest

from app.scraper.source_registry import build_source
from app.scraper.sources.adzuna import AdzunaSource
from app.scraper.sources.greenhouse import GreenhouseSource
from app.scraper.sources.lever import LeverSource
from app.scraper.sources.remotive import RemotiveSource
from app.scraper.sources.the_muse import TheMuseSource


def test_registry_builds_adzuna_with_source_options():
    source = build_source(
        "adzuna",
        app_id="test-app-id",
        app_key="test-app-key",
        salary_min=60000,
        permanent=True,
    )

    assert isinstance(source, AdzunaSource)
    assert source.salary_min == 60000
    assert source.permanent is True


def test_registry_reports_available_sources_for_unknown_name():
    with pytest.raises(ValueError, match="adzuna"):
        build_source("unknown")


def test_registry_builds_greenhouse_with_board_configuration():
    source = build_source(
        "greenhouse",
        board_token="example",
        company="Example Ltd",
    )

    assert isinstance(source, GreenhouseSource)
    assert source.board_token == "example"
    assert source.company == "Example Ltd"


def test_registry_builds_lever_with_company_configuration():
    source = build_source(
        "lever",
        company_slug="example",
        company="Example Ltd",
    )

    assert isinstance(source, LeverSource)
    assert source.company_slug == "example"
    assert source.company == "Example Ltd"


def test_registry_builds_the_muse_and_remotive():
    muse = build_source("the_muse", category="Data and Analytics")
    remotive = build_source("remotive", category="software-dev")

    assert isinstance(muse, TheMuseSource)
    assert muse.category == "Data and Analytics"
    assert isinstance(remotive, RemotiveSource)
    assert remotive.category == "software-dev"
