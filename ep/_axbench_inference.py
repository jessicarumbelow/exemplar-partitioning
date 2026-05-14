# ruff: noqa: E402
"""Wrapper that runs AxBench's inference.py with our ``max_concepts`` cap.

Upstream AxBench wires ``max_concepts`` into ``train.py`` (df_list slicing) but
not ``inference.py``, so a smoke run with ``--axbench-max-concepts=N`` trains
N rows of weights then tries to look up ``concept_id=N+1`` in
``predict_latent`` → ``IndexError: index N is out of bounds for dimension 2
with size N``.

We patch the four ``infer_*`` functions in
``axbench.scripts.inference`` at import time, in-memory, by re-execing each
function's source with one extra ``concept_ids = concept_ids[:max_concepts]``
line spliced in. The submodule files on disk are never modified, so the
AxBench checkout stays pristine — important because we publish this repo and
the submodule pin should match upstream byte-for-byte.

Invoked as ``python -m ep._axbench_inference`` in place of
``python -m axbench.scripts.inference``. Works because ``inference.main()``
looks up ``infer_latent`` etc. via the module's globals at call time, so
rebinding the names in ``_inf.__dict__`` is enough.
"""
from ep import _axbench_bootstrap  # noqa: F401  must run before `import axbench`

import inspect

from axbench.scripts import inference as _inf


_NEEDLE = (
    '    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]\n'
)
_CAP = _NEEDLE + (
    '    if getattr(args, "max_concepts", None):\n'
    '        concept_ids = concept_ids[:args.max_concepts]\n'
)


def _patched(fn):
    src = inspect.getsource(fn)
    if _NEEDLE not in src:
        raise RuntimeError(
            f"AxBench {fn.__name__}: max_concepts patch needle not found. "
            f"Upstream may have moved; re-pin _NEEDLE."
        )
    src = src.replace(_NEEDLE, _CAP)
    # Enable LM cache so persist_lm_cache fills across runs. Upstream
    # hardcodes ``use_cache=False`` at the inference DatasetFactory site
    # (inference.py:610), which makes ``save_cache`` a no-op via the
    # internal ``if self.use_cache:`` guard, so atexit-registered cache
    # dumps silently drop everything. Re-spending the LLM-judge bill on
    # every restart is the symptom; this flip is the fix.
    src = src.replace("use_cache=False,", "use_cache=True,")
    ns = dict(_inf.__dict__)
    exec(src, ns)
    return ns[fn.__name__]


_inf.infer_steering = _patched(_inf.infer_steering)
_inf.infer_latent = _patched(_inf.infer_latent)
_inf.infer_latent_imbalance = _patched(_inf.infer_latent_imbalance)
_inf.infer_latent_on_train_data = _patched(_inf.infer_latent_on_train_data)


# Fix upstream save_state/load_state path mismatch. save_state writes to
# ``<dump_dir>/<partition>_<STATE_FILE>_rank_<rank>`` but load_state reads
# from ``<dump_dir>/inference/<mode>_<STATE_FILE>_rank_<rank>``. The state
# pkl is therefore never found on resume, and every restart re-processes
# all concepts from 0. Override save_state to write where load_state will
# look.
import os as _os
import pickle as _pickle
from pathlib import Path as _Path


def _patched_save_state(dump_dir, state, partition, rank):
    if not isinstance(dump_dir, _Path):
        dump_dir = _Path(dump_dir)
    state_dir = dump_dir / "inference"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{partition}_{_inf.STATE_FILE}_rank_{rank}"
    with open(state_path, "wb") as f:
        _pickle.dump(state, f)


_inf.save_state = _patched_save_state


if __name__ == "__main__":
    _inf.main()
