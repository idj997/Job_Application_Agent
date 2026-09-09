from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from app.scraper.cleaner import JobCleaner
from app.scraper.role_catalog import DEFAULT_ROLE_CATALOG, RoleCatalog, RoleCatalogError
from app.scraper.storage import JobStorage
from app.services.job_collection_service import CollectionSummary, JobCollectionService
from app.services.role_corpus_service import (
    RoleCorpusCollectionService,
    RoleCorpusSummary,
    RoleQuerySummary,
)
from scripts.scrape_jobs import SOURCE_CHOICES, build_configured_source


def _print_catalog(catalog: RoleCatalog) -> None:
    for family in catalog.select():
        print(f"{family.name:<22} {family.display_name}")
        for query in family.queries:
            print(f"  - {query}")


def _print_plan(
    source_names: list[str],
    families,
    location: str,
    limit_per_query: int,
) -> None:
    query_count = sum(len(family.queries) for family in families)
    print(f"Sources:             {', '.join(source_names)}")
    print(f"Role families:       {len(families)}")
    print(f"Distinct queries:    {query_count}")
    print(f"Query/source pairs:  {query_count * len(source_names)}")
    print(f"Limit per pair:      {limit_per_query}")
    print(f"Location:            {location or 'any'}")
    print()
    for family in families:
        print(f"{family.display_name} ({family.name})")
        for query in family.queries:
            print(f"  - {query}")


def _print_summary(summary: RoleCorpusSummary) -> None:
    print(
        f"{'Source':<13}"
        f"{'Role family':<21}"
        f"{'Query':<38}"
        f"{'Found':>7}"
        f"{'Saved':>7}"
        f"{'Dupes':>7}"
        f"{'Reject':>8}"
        f"{'Failed':>8}"
    )
    print("-" * 109)
    for item in summary.queries:
        collection = item.collection
        print(
            f"{collection.source:<13}"
            f"{item.role_family:<21}"
            f"{item.query[:36]:<38}"
            f"{collection.discovered:>7}"
            f"{collection.saved:>7}"
            f"{collection.duplicates:>7}"
            f"{collection.rejected:>8}"
            f"{collection.failed:>8}"
        )
        if collection.source_error:
            print(f"  Error: {collection.source_error}")
    print("-" * 109)
    print(
        f"{'Total':<72}"
        f"{summary.discovered:>7}"
        f"{summary.saved:>7}"
        f"{summary.duplicates:>7}"
        f"{summary.rejected:>8}"
        f"{summary.failed:>8}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect a diverse job-description corpus by role family."
    )
    parser.add_argument("--sources", nargs="+", choices=SOURCE_CHOICES)
    role_group = parser.add_mutually_exclusive_group()
    role_group.add_argument("--roles", nargs="+", metavar="ROLE_FAMILY")
    role_group.add_argument("--all-roles", action="store_true")
    parser.add_argument("--list-roles", action="store_true")
    parser.add_argument(
        "--role-config",
        type=Path,
        default=DEFAULT_ROLE_CATALOG,
    )
    parser.add_argument("--location", default="")
    parser.add_argument("--limit-per-query", type=int, default=5)
    parser.add_argument(
        "--queries-per-role",
        type=int,
        default=4,
        help="Use at most this many catalog queries from each selected role family.",
    )
    parser.add_argument(
        "--max-query-source-pairs",
        type=int,
        default=50,
        help="Safety cap on planned query/source combinations.",
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help="Allow a plan larger than --max-query-source-pairs.",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--visa-sponsorship", action="store_true", default=None)
    parser.add_argument("--salary-min", type=int)
    parser.add_argument("--salary-max", type=int)
    parser.add_argument("--full-time", action="store_true", default=None)
    parser.add_argument("--permanent", action="store_true", default=None)
    parser.add_argument("--board-token")
    parser.add_argument("--company")
    parser.add_argument("--company-slug")
    parser.add_argument("--lever-region", choices=["global", "eu"], default="global")
    parser.add_argument("--commitment")
    parser.add_argument("--team")
    parser.add_argument("--department")
    parser.add_argument("--category")
    parser.add_argument("--level")
    args = parser.parse_args(argv)

    try:
        catalog = RoleCatalog.load(args.role_config)
    except RoleCatalogError as exc:
        parser.error(str(exc))

    if args.list_roles:
        _print_catalog(catalog)
        return
    if not args.sources:
        parser.error("--sources is required unless --list-roles is used")
    if not args.roles and not args.all_roles:
        parser.error("provide --roles or --all-roles")
    if args.limit_per_query < 1:
        parser.error("--limit-per-query must be at least 1")
    if args.queries_per_role < 1:
        parser.error("--queries-per-role must be at least 1")

    try:
        families = catalog.select(None if args.all_roles else args.roles)
    except RoleCatalogError as exc:
        parser.error(str(exc))
    families = [
        replace(family, queries=family.queries[: args.queries_per_role])
        for family in families
    ]

    pair_count = len(args.sources) * sum(
        len(family.queries) for family in families
    )
    if pair_count > args.max_query_source_pairs and not args.allow_large_run:
        parser.error(
            f"planned run has {pair_count} query/source pairs, exceeding the "
            f"safety cap of {args.max_query_source_pairs}; select fewer roles "
            "or pass --allow-large-run"
        )

    _print_plan(
        source_names=args.sources,
        families=families,
        location=args.location,
        limit_per_query=args.limit_per_query,
    )
    if args.dry_run:
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
    sources = []
    configuration_failures: list[RoleQuerySummary] = []
    for source_name in args.sources:
        try:
            sources.append(
                build_configured_source(source_name, **source_configuration)
            )
        except Exception as exc:
            configuration_failures.append(
                RoleQuerySummary(
                    role_family="configuration",
                    query="source setup",
                    collection=CollectionSummary(
                        source=source_name,
                        failed=1,
                        source_error=str(exc),
                    ),
                )
            )
    collection_service = JobCollectionService(
        storage=JobStorage(),
        cleaner=JobCleaner(),
    )
    corpus_service = RoleCorpusCollectionService(collection_service)
    summary = corpus_service.collect(
        sources=sources,
        role_families=families,
        location=args.location,
        limit_per_query=args.limit_per_query,
    )
    summary.queries[0:0] = configuration_failures
    print()
    _print_summary(summary)


if __name__ == "__main__":
    main()
