"""Optional raw BUAA .dat -> original L2 cache, using the bundled loader."""

from __future__ import annotations
import argparse
from pathlib import Path

if __package__:
    from .dataset import prepare_internal_cache
else:
    from dataset import prepare_internal_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="JSON: raw-label string -> class name.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty cache output directory.")
    print(prepare_internal_cache(args.raw_dir, args.labels, args.output))


if __name__ == "__main__":
    main()
