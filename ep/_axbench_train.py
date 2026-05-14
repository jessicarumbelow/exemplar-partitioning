"""Wrapper that runs AxBench's train.py through ``ep._axbench_bootstrap``.

No source patching here — train.py already respects ``max_concepts`` and
doesn't touch ``master_data_dir``. The only reason this wrapper exists is
to install our trimmed ``axbench`` package (which skips broken upstream
star-imports like ``models.hypernet.modeling_hypernet`` that depend on
private ``transformers`` utilities) before train.py runs ``import axbench``.

Invoked as ``python -m ep._axbench_train`` in place of
``python -m axbench.scripts.train``.
"""
from ep import _axbench_bootstrap  # noqa: F401  must run before `import axbench`

from axbench.scripts import train as _tr


if __name__ == "__main__":
    _tr.main()
