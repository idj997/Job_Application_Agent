from __future__ import annotations

import argparse
from pathlib import Path

from app.datasets.counterfactual_candidates import (
    NEGATION_TEMPLATES,
    SOURCE_LABELS,
    generate_counterfactual_candidates,
)
from app.datasets.requirement_dataset import DatasetValidationError, load_annotations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate explicit NOT_REQUIREMENT counterfactual candidates from "
            "reviewed positive annotations. All outputs require another review."
        )
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
        default=Path("data/labelled/not_requirement_candidates.json"),
    )
    parser.add_argument(
        "--variants-per-example",
        type=int,
        default=1,
        choices=range(1, len(NEGATION_TEMPLATES) + 1),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-labels",
        nargs="+",
        choices=SOURCE_LABELS,
        default=list(SOURCE_LABELS),
    )
    args = parser.parse_args(argv)

    try:
        examples = load_annotations(args.input)
        summary = generate_counterfactual_candidates(
            examples,
            args.output,
            variants_per_example=args.variants_per_example,
            seed=args.seed,
            source_labels=args.source_labels,
        )
    except (DatasetValidationError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Counterfactual queue: {summary.output_path}")
    print(f"Reviewed inputs:      {summary.source_examples}")
    print(f"Eligible positives:   {summary.eligible_examples}")
    print(f"Grouped skills skipped: {summary.grouped_examples_skipped}")
    print(f"Candidates generated: {summary.candidates}")
    print("Proposed label:       NOT_REQUIREMENT")
    print("Review status:        all candidates require human approval")


if __name__ == "__main__":
    main()
