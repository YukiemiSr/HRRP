"""Evaluate a fixed checkpoint on the original BUAA test only; never select checkpoints."""

from __future__ import annotations
import argparse
from pathlib import Path

if __package__:
    from .utils import common_arguments
    from .val import run_evaluation
else:
    from utils import common_arguments
    from val import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only two records per class; not a formal score.",
    )
    run_evaluation(parser.parse_args(), "test")


if __name__ == "__main__":
    main()
