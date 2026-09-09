from __future__ import annotations

import argparse
from pprint import pprint

from app.scraper.cleaner import JobCleaner
from app.scraper.source_registry import build_source
from app.scraper.sources.base import JobSource
from app.scraper.storage import JobStorage
from app.services.job_collection_service import (
    CollectionSummary,
    JobCollectionService,
    MultiSourceCollectionSummary,
)
from app.scraper.sources.dwp import DwpSource
from app.scraper.sources.generic_jsonld import GenericJsonLdSource


SOURCE_CHOICES = [
    "adzuna",
    "arbeitnow",
    "dwp",
    "greenhouse",
    "lever",
    "remotive",
    "the_muse",
]


def scrape_single_url(url: str) -> None:
    source = GenericJsonLdSource()
    cleaner = JobCleaner()
    storage = JobStorage()

    print(f"Fetching: {url}")

    page = source.fetch(url)

    storage.save_raw_page(page)

    print(
        f"Downloaded {len(page.html):,} characters "
        f"(HTTP {page.status_code})"
    )

    job = source.extract(page)
    job = cleaner.clean(job)

    storage.save_job(job)

    pprint(job)


def scrape_dwp(
    query: str,
    location: str,
    limit: int,
) -> None:

    source = DwpSource()
    cleaner = JobCleaner()
    storage = JobStorage()

    print(
        f"Discovering DWP jobs: "
        f"{query!r} in {location!r}"
    )

    try:
        urls = source.discover_jobs(
            query=query,
            location=location,
            limit=limit,
        )

    except Exception as exc:
        print(
            f"DWP source unavailable: {exc}"
        )
        return

    print(
        f"Discovered {len(urls)} job URLs."
    )


def build_configured_source(
    source_name: str,
    visa_sponsorship: bool | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    full_time: bool | None = None,
    permanent: bool | None = None,
    board_token: str | None = None,
    company: str | None = None,
    company_slug: str | None = None,
    lever_region: str = "global",
    commitment: str | None = None,
    team: str | None = None,
    department: str | None = None,
    category: str | None = None,
    level: str | None = None,
) -> JobSource:
    source_options = {}
    if source_name == "arbeitnow":
        source_options["visa_sponsorship"] = visa_sponsorship
    elif source_name == "adzuna":
        source_options.update(
            {
                "salary_min": salary_min,
                "salary_max": salary_max,
                "full_time": full_time,
                "permanent": permanent,
            }
        )
    elif source_name == "greenhouse":
        source_options.update(
            {
                "board_token": board_token,
                "company": company,
            }
        )
    elif source_name == "lever":
        source_options.update(
            {
                "company_slug": company_slug,
                "company": company,
                "region": lever_region,
                "commitment": commitment,
                "team": team,
                "department": department,
            }
        )
    elif source_name == "the_muse":
        source_options.update(
            {
                "category": category,
                "level": level,
                "company_filter": company,
            }
        )
    elif source_name == "remotive":
        source_options.update(
            {
                "category": category,
                "company_name": company,
            }
        )

    return build_source(source_name, **source_options)


def scrape_source(
    source_name: str,
    query: str | None,
    location: str,
    limit: int,
    **source_configuration: object,
) -> CollectionSummary:
    source = build_configured_source(source_name, **source_configuration)
    service = JobCollectionService(
        storage=JobStorage(),
        cleaner=JobCleaner(),
    )
    summary = service.collect(
        source=source,
        query=query,
        location=location,
        limit=limit,
    )

    print_collection_summary(summary)

    return summary


def scrape_sources(
    source_names: list[str],
    query: str | None,
    location: str,
    limit_per_source: int,
    **source_configuration: object,
) -> MultiSourceCollectionSummary:
    service = JobCollectionService(
        storage=JobStorage(),
        cleaner=JobCleaner(),
    )

    configured: list[JobSource | CollectionSummary] = []
    for source_name in source_names:
        try:
            configured.append(
                build_configured_source(source_name, **source_configuration)
            )
        except Exception as exc:
            configured.append(
                CollectionSummary(
                    source=source_name,
                    failed=1,
                    source_error=str(exc),
                )
            )

    collected = service.collect_many(
        sources=[
            source
            for source in configured
            if isinstance(source, JobSource)
        ],
        query=query,
        location=location,
        limit_per_source=limit_per_source,
    )
    collected_summaries = iter(collected.summaries)
    result = MultiSourceCollectionSummary(
        summaries=[
            item if isinstance(item, CollectionSummary) else next(collected_summaries)
            for item in configured
        ]
    )

    print_multi_source_summary(result)
    return result


def print_collection_summary(summary: CollectionSummary) -> None:
    print(f"Source: {summary.source}")
    print()
    print(f"Discovered: {summary.discovered:>4}")
    print(f"Saved:       {summary.saved:>4}")
    print(f"Duplicates:  {summary.duplicates:>4}")
    print(f"Rejected:    {summary.rejected:>4}")
    print(f"Failed:      {summary.failed:>4}")
    if summary.source_error:
        print(f"Error: {summary.source_error}")


def print_multi_source_summary(summary: MultiSourceCollectionSummary) -> None:
    print(
        f"{'Source':<14}"
        f"{'Discovered':>12}"
        f"{'Saved':>8}"
        f"{'Duplicates':>12}"
        f"{'Rejected':>10}"
        f"{'Failed':>8}"
    )
    print("-" * 64)
    for item in summary.summaries:
        print(
            f"{item.source:<14}"
            f"{item.discovered:>12}"
            f"{item.saved:>8}"
            f"{item.duplicates:>12}"
            f"{item.rejected:>10}"
            f"{item.failed:>8}"
        )
    print("-" * 64)
    print(
        f"{'Total':<14}"
        f"{summary.discovered:>12}"
        f"{summary.saved:>8}"
        f"{summary.duplicates:>12}"
        f"{summary.rejected:>10}"
        f"{summary.failed:>8}"
    )

    for item in summary.summaries:
        if item.source_error:
            print(f"Error ({item.source}): {item.source_error}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "url",
        nargs="?",
    )

    source_group = parser.add_mutually_exclusive_group()

    source_group.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
    )

    source_group.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCE_CHOICES,
        help="Collect from multiple sources in one failure-isolated run.",
    )

    parser.add_argument(
        "--query",
    )

    parser.add_argument(
        "--location",
        default="",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--limit-per-source",
        type=int,
        help="Maximum jobs to discover from each source in --sources mode.",
    )

    parser.add_argument(
        "--visa-sponsorship",
        action="store_true",
        default=None,
        help="Only request jobs marked as offering visa sponsorship.",
    )

    parser.add_argument(
        "--salary-min",
        type=int,
        help="Minimum annual salary for sources that support it.",
    )

    parser.add_argument(
        "--salary-max",
        type=int,
        help="Maximum annual salary for sources that support it.",
    )

    parser.add_argument(
        "--full-time",
        action="store_true",
        default=None,
        help="Only request full-time jobs from sources that support it.",
    )

    parser.add_argument(
        "--permanent",
        action="store_true",
        default=None,
        help="Only request permanent jobs from sources that support it.",
    )

    parser.add_argument(
        "--board-token",
        help="Employer job-board token for ATS sources such as Greenhouse.",
    )

    parser.add_argument(
        "--company",
        help="Company name override for employer-specific ATS sources.",
    )

    parser.add_argument(
        "--company-slug",
        help="Employer site slug for ATS sources such as Lever.",
    )

    parser.add_argument(
        "--lever-region",
        choices=["global", "eu"],
        default="global",
        help="Lever hosting region.",
    )

    parser.add_argument(
        "--commitment",
        help="Exact Lever commitment filter, such as Full-time.",
    )

    parser.add_argument(
        "--team",
        help="Exact Lever team filter.",
    )

    parser.add_argument(
        "--department",
        help="Exact Lever department filter.",
    )

    parser.add_argument(
        "--category",
        help="Source-specific job category filter.",
    )

    parser.add_argument(
        "--level",
        help="Experience-level filter for sources that support it.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.limit_per_source is not None and args.limit_per_source < 1:
        parser.error("--limit-per-source must be at least 1")

    if args.url:
        scrape_single_url(
            args.url
        )
        return

    selected_sources = args.sources or ([args.source] if args.source else [])
    if "dwp" in selected_sources and not args.query:
        parser.error(
            "--query is required when using the dwp source"
        )

    if args.source == "dwp":
        scrape_dwp(
            query=args.query,
            location=args.location,
            limit=args.limit,
        )

        return

    source_configuration = {
        "visa_sponsorship": args.visa_sponsorship,
        "salary_min": args.salary_min,
        "salary_max": args.salary_max,
        "full_time": args.full_time,
        "permanent": args.permanent,
        "board_token": args.board_token,
        "company": args.company,
        "company_slug": args.company_slug,
        "lever_region": args.lever_region,
        "commitment": args.commitment,
        "team": args.team,
        "department": args.department,
        "category": args.category,
        "level": args.level,
    }

    if args.source:
        scrape_source(
            source_name=args.source,
            query=args.query,
            location=args.location,
            limit=args.limit,
            **source_configuration,
        )
        return

    if args.sources:
        scrape_sources(
            source_names=args.sources,
            query=args.query,
            location=args.location,
            limit_per_source=args.limit_per_source or args.limit,
            **source_configuration,
        )
        return

    parser.error(
        "Provide either a URL or "
        "--source/--sources."
    )


if __name__ == "__main__":
    main()
