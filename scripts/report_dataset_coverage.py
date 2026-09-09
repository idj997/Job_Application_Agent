from __future__ import annotations

import argparse
from pathlib import Path

from app.datasets.coverage_report import generate_coverage_report
from app.datasets.requirement_dataset import DatasetValidationError, load_annotations
from app.scraper.role_catalog import RoleCatalog, RoleCatalogError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Report label, role, source, and job coverage before training."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[Path("data/labelled/requirements.jsonl")],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labelled/coverage_report.json"),
    )
    parser.add_argument("--minimum-per-label", type=int, default=20)
    parser.add_argument("--minimum-per-role", type=int, default=5)
    parser.add_argument("--minimum-unique-jobs", type=int, default=30)
    parser.add_argument("--minimum-sources", type=int, default=3)
    parser.add_argument("--maximum-label-share", type=float, default=0.5)
    parser.add_argument(
        "--role-catalog",
        type=Path,
        default=Path("config/job_role_queries.json"),
    )
    args = parser.parse_args(argv)

    try:
        examples = load_annotations(args.input)
        role_catalog = RoleCatalog.load(args.role_catalog)
        summary = generate_coverage_report(
            examples,
            args.output,
            required_role_families=role_catalog.names,
            minimum_per_label=args.minimum_per_label,
            minimum_per_role=args.minimum_per_role,
            minimum_unique_jobs=args.minimum_unique_jobs,
            minimum_sources=args.minimum_sources,
            maximum_label_share=args.maximum_label_share,
        )
    except (DatasetValidationError, RoleCatalogError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Coverage report:    {summary.output_path}")
    print(f"Reviewed examples:  {summary.examples}")
    print(f"Unique jobs:        {summary.unique_jobs}")
    print(f"Blocking gaps:      {summary.blockers}")
    print(f"Warnings:           {summary.warnings}")
    print(
        "Ready for training: "
        + ("yes" if summary.ready_for_training else "no")
    )


if __name__ == "__main__":
    main()
