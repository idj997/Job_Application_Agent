from __future__ import annotations

import argparse
from pathlib import Path

from app.datasets.requirement_dataset import (
    DatasetValidationError,
    RequirementDatasetBuilder,
    load_annotations,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe requirement-classification training and "
            "evaluation datasets from reviewed JSONL annotations."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[Path("data/labelled/requirements.jsonl")],
        help="One or more reviewed annotation JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets/requirements"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-ratio",
        type=float,
        default=0.2,
        help="Fraction of unique jobs reserved for evaluation.",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.3,
        help="Target fraction of noisy variants in the mixed training file.",
    )
    parser.add_argument(
        "--minimum-per-label",
        type=int,
        default=1,
        help="Fail if any of the five labels has fewer reviewed examples.",
    )
    parser.add_argument(
        "--maximum-synthetic-ratio",
        type=float,
        default=0.4,
        help="Fail if synthetic annotations exceed this fraction of the corpus.",
    )
    args = parser.parse_args(argv)

    try:
        examples = load_annotations(args.input)
        builder = RequirementDatasetBuilder(
            seed=args.seed,
            evaluation_ratio=args.evaluation_ratio,
            noise_ratio=args.noise_ratio,
            minimum_per_label=args.minimum_per_label,
            maximum_synthetic_ratio=args.maximum_synthetic_ratio,
        )
        result = builder.build(examples, args.output_dir)
    except (DatasetValidationError, ValueError) as exc:
        parser.error(str(exc))

    counts = result.manifest["counts"]
    print(f"Dataset written to: {result.output_dir}")
    print(f"Reviewed examples:  {result.manifest['source_examples']}")
    print(f"Unique jobs:         {result.manifest['source_jobs']}")
    print(f"Training examples:   {counts['train_total']}")
    print(f"  Real-world:        {counts['train_real_world']}")
    print(f"  Noisy:             {counts['train_noisy']}")
    print(f"Evaluation jobs:     {result.manifest['evaluation_jobs']}")
    print("Job split overlap:   0")


if __name__ == "__main__":
    main()
