#!/usr/bin/env python3

"""A script to track pitch with multiple algorithms according to a YAML file."""

import argparse
import logging
import sys
from collections.abc import Sequence

logger = logging.getLogger("pitch_garden")


def get_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Track pitch with multiple algorithms and vote their outputs. "
            "The settings are given by a YAML file; see configs/*.yaml."
        )
    )
    parser.add_argument("config", type=str, help="The YAML configuration file.")
    parser.add_argument(
        "source",
        type=str,
        help="The input audio file, directory, or text file listing audio files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="The directory where the results are saved.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="If set, suppresses non-error output during processing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Perform the main pitch tracking process."""
    args = get_arguments(argv)

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from pitch_garden.config import ConfigError, load_config
    from pitch_garden.runner import run_batch

    try:
        config = load_config(args.config)
    except ConfigError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(2)

    num_errors = run_batch(config, args.source, args.output_dir, quiet=args.quiet)
    if num_errors > 0:
        logger.error(f"{num_errors} files were skipped due to errors.")
        sys.exit(1)
    logger.info("All files processed successfully.")


if __name__ == "__main__":
    main()
