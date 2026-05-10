"""
run_pipeline.py — CLI entry point for gridpoint_ml.

Usage:
    python run_pipeline.py config/example_config.toml
"""
import argparse
import logging
import sys

from gridpoint_ml import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ML models for every gridpoint in a spatiotemporal grid."
    )
    parser.add_argument("config", help="Path to TOML configuration file.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = Pipeline(args.config)
    results = pipeline.run()

    failed = [r for r in results if not r.success]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
