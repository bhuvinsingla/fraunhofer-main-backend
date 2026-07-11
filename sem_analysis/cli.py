#!/usr/bin/env python3
"""CLI entry point for SEM tip radius analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sem_analysis.pipeline import SEMAnalysisPipeline, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SEM image analysis pipeline for tip radius measurement",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to SEM image (PNG, TIFF, JPG)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for reports and annotated images",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "-g", "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground-truth CSV/JSON for validation",
    )
    parser.add_argument(
        "--no-alt-methods",
        action="store_true",
        help="Skip alternative brainstorming measurement methods",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print JSON results to stdout",
    )

    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 1

    pipeline = SEMAnalysisPipeline(config_path=args.config)
    result = pipeline.analyze(
        args.image,
        output_dir=args.output,
        ground_truth_path=args.ground_truth,
        run_alternative_methods=not args.no_alt_methods,
    )

    if args.json_only:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Analyzed: {result.source_path}")
        print(f"Calibration: {result.nm_per_pixel:.4f} nm/px")
        print(f"Shapes detected: {result.shapes_detected} ({result.shapes_passed} passed)")
        if result.aggregation.get("mean_radius_nm") is not None:
            print(
                f"Mean radius: {result.aggregation['mean_radius_nm']:.2f} ± "
                f"{result.aggregation['std_radius_nm']:.2f} nm"
            )
            print(f"Tip condition: {result.tip_condition}")
        if result.alternative_methods:
            print("\nAlternative methods:")
            for name, data in result.alternative_methods.items():
                print(f"  {name}: {data}")
        print(f"\nAnnotated image: {result.annotated_image_path}")
        if result.validation:
            print(f"Validation metrics: {result.validation.get('metrics')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
