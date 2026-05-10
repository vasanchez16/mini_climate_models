"""
run_predictions.py — CLI entry point for the gridpoint_ml prediction pipeline.

Usage:
    python run_predictions.py config/example_predict_config.toml
    python run_predictions.py config/example_predict_config.toml --verbose
"""
import argparse
import logging
import sys

from gridpoint_ml.predict_pipeline import PredictPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predictions for every trained gridpoint model."
    )
    parser.add_argument("config", help="Path to prediction TOML configuration file.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = PredictPipeline(args.config)
    results = pipeline.run()

    failed = [r for r in results if not r.success]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
