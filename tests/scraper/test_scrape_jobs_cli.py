import sys
from unittest.mock import Mock

from app.services.job_collection_service import (
    CollectionSummary,
    MultiSourceCollectionSummary,
)
from scripts import scrape_jobs


def test_main_accepts_multiple_sources_and_a_per_source_limit(monkeypatch):
    collect = Mock()
    monkeypatch.setattr(scrape_jobs, "scrape_sources", collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_jobs",
            "--sources",
            "arbeitnow",
            "adzuna",
            "--query",
            "data engineer",
            "--location",
            "United Kingdom",
            "--limit-per-source",
            "100",
        ],
    )

    scrape_jobs.main()

    kwargs = collect.call_args.kwargs
    assert kwargs["source_names"] == ["arbeitnow", "adzuna"]
    assert kwargs["query"] == "data engineer"
    assert kwargs["location"] == "United Kingdom"
    assert kwargs["limit_per_source"] == 100


def test_multi_source_report_includes_totals_and_source_error(capsys):
    summary = MultiSourceCollectionSummary(
        summaries=[
            CollectionSummary(source="arbeitnow", discovered=5, saved=4, duplicates=1),
            CollectionSummary(
                source="adzuna",
                failed=1,
                source_error="service unavailable",
            ),
        ]
    )

    scrape_jobs.print_multi_source_summary(summary)

    output = capsys.readouterr().out
    assert "arbeitnow" in output
    assert "adzuna" in output
    assert "Total" in output
    assert "service unavailable" in output

