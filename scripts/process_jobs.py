from __future__ import annotations

import argparse
from pathlib import Path

from app.datasets.annotation_candidates import (
    generate_candidate_queue,
    promote_approved_candidates,
    validate_candidate_reviews,
)
from app.datasets.requirement_dataset import DatasetValidationError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conservative requirement annotation candidates from "
            "processed jobs. Candidates still require human review."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed_jobs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labelled/requirement_candidates.json"),
    )
    parser.add_argument(
        "--include-candidates",
        nargs="+",
        type=Path,
        default=[],
        help=(
            "Additional candidate queues to merge into generation, such as "
            "reviewable counterfactuals."
        ),
    )
    parser.add_argument(
        "--promote-approved",
        type=Path,
        metavar="CANDIDATE_FILE",
        help="Promote explicitly approved candidates instead of generating a queue.",
    )
    parser.add_argument(
        "--validate-reviews",
        type=Path,
        metavar="CANDIDATE_FILE",
        help="Audit review statuses, labels, and notes without changing the queue.",
    )
    parser.add_argument(
        "--review-report",
        type=Path,
        default=Path("data/labelled/review_validation.json"),
        help="Readable JSON report written by --validate-reviews.",
    )
    parser.add_argument(
        "--approved-output",
        type=Path,
        default=Path("data/labelled/requirements.jsonl"),
    )
    parser.add_argument(
        "--decision-log",
        type=Path,
        default=Path("data/labelled/review_decisions.json"),
        help="Audit log containing approved, rejected, and pending review feedback.",
    )
    args = parser.parse_args(argv)

    if args.promote_approved and args.validate_reviews:
        parser.error("--promote-approved and --validate-reviews are mutually exclusive")

    if args.validate_reviews:
        try:
            summary = validate_candidate_reviews(
                args.validate_reviews,
                args.review_report,
            )
        except DatasetValidationError as exc:
            parser.error(str(exc))
        print(f"Review report: {summary.output_path}")
        print(f"Candidates:    {summary.candidates_read}")
        print(f"Errors:        {summary.errors}")
        print(f"Warnings:      {summary.warnings}")
        print(f"Information:   {summary.information}")
        return

    if args.promote_approved:
        try:
            summary = promote_approved_candidates(
                args.promote_approved,
                args.approved_output,
                decision_log_path=args.decision_log,
            )
        except DatasetValidationError as exc:
            parser.error(str(exc))
        print(f"Approved annotations: {summary.output_path}")
        print(f"Candidates read:      {summary.candidates_read}")
        print(f"Approved:             {summary.approved}")
        print(f"Rejected:             {summary.rejected}")
        print(f"Still pending:        {summary.pending}")
        print(f"Decision log:         {summary.decision_log_path}")
        return

    input_paths = sorted(args.input_dir.glob("*.json"))
    if not input_paths:
        parser.error(f"no processed job JSON files found in {args.input_dir}")

    summary = generate_candidate_queue(
        input_paths,
        args.output,
        additional_candidate_paths=args.include_candidates,
    )
    print(f"Candidate queue: {summary.output_path}")
    print(f"Jobs read:       {summary.jobs_read}")
    print(f"Jobs failed:     {summary.jobs_failed}")
    print(f"Candidates:      {summary.candidates}")
    for label, count in summary.label_counts.items():
        print(f"  {label:<16} {count}")
    print("Review status:   all candidates require human approval")


if __name__ == "__main__":
    main()
