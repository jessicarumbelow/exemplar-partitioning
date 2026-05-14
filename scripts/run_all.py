"""Compatibility entrypoint — forwards to build_partitions.

Use --eval to run SAEBench evals after building, e.g.:
  python -m scripts.run_all --eval all
  python -m scripts.run_all --eval sparse_probing
"""

from __future__ import annotations

import sys


def main() -> None:
    from scripts.build_partitions import main as target_main

    sys.argv = [sys.argv[0], *sys.argv[1:]]
    target_main()


if __name__ == "__main__":
    main()
