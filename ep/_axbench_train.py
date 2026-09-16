"""Wrapper that runs AxBench's train.py through ``ep._axbench_bootstrap``.

The wrapper loads only the AxBench modules needed for this evaluation before
train.py imports ``axbench``. This avoids optional upstream imports with
additional ``transformers`` dependencies.

Invoked as ``python -m ep._axbench_train`` in place of
``python -m axbench.scripts.train``.
"""
from ep import _axbench_bootstrap  # noqa: F401  must run before `import axbench`

from axbench.scripts import train as _tr


if __name__ == "__main__":
    _tr.main()
