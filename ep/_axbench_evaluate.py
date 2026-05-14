# ruff: noqa: E402
"""Wrapper that runs AxBench's evaluate.py with our master_data_dir override.

Upstream ``axbench.scripts.evaluate.eval_steering_single_task`` builds the
LMJudge ``LanguageModel`` with ``master_data_dir="axbench/data"``, ignoring
``args.master_data_dir``. On Modal that resolves under the read-only mount,
so ``LanguageModel.__init__`` crashes when it ``mkdir``\\s the
``persist_lm_cache`` inside it.

We monkey-patch the function at import time, in-memory, by re-execing its
source with ``"axbench/data"`` swapped for an env-var read. ``build_partitions``
sets ``AXBENCH_MASTER_DATA_DIR`` to the writable ``/vol`` path before launching
this wrapper. The submodule files on disk are never modified.

Invoked as ``python -m ep._axbench_evaluate`` in place of
``python -m axbench.scripts.evaluate``. Works because ``evaluate.main()``
dispatches to ``eval_steering_single_task`` via the module's globals at call
time, so rebinding the name in ``_ev.__dict__`` is enough.

The eager ``stanza.Pipeline(...)`` at the top of evaluate.py is neutralised
by ``ep._axbench_bootstrap`` (stubs ``stanza`` to a no-op ``Pipeline``).
Our sweep doesn't use Rule evaluators, so ``nlp`` is never consumed.
"""
from ep import _axbench_bootstrap  # noqa: F401  must run before `import axbench`

import inspect

from axbench.scripts import evaluate as _ev


_NEEDLE = '        master_data_dir="axbench/data",\n'
_REPLACEMENT = (
    '        master_data_dir=os.environ.get("AXBENCH_MASTER_DATA_DIR", "axbench/data"),\n'
)


def _patched(fn):
    src = inspect.getsource(fn)
    if _NEEDLE not in src:
        raise RuntimeError(
            f"AxBench {fn.__name__}: master_data_dir patch needle not found. "
            f"Upstream may have moved; re-pin _NEEDLE."
        )
    ns = dict(_ev.__dict__)
    exec(src.replace(_NEEDLE, _REPLACEMENT), ns)
    return ns[fn.__name__]


_ev.eval_steering_single_task = _patched(_ev.eval_steering_single_task)


if __name__ == "__main__":
    _ev.main()
