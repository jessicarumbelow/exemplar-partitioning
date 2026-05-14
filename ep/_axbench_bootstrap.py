"""Pre-import bootstrap for AxBench. Run before any ``import axbench``.

Three things happen here, all purely in-process so the AxBench submodule
on disk stays byte-identical with upstream:

1. Stub ``stanza`` in ``sys.modules``. Upstream ``axbench.scripts.evaluate``
   eagerly calls ``stanza.Pipeline('en', ...)`` at module top, which forces
   a stanza install + model download. Our sweep doesn't use Rule
   evaluators (the only consumers of ``nlp``), so we hand back a no-op
   ``Pipeline`` and skip the cost.

2. Pre-load the ``axbench`` package with a trimmed ``__init__.py``.
   Upstream's init star-imports several modules — sft, lora, reft,
   lsreft, steering_vector, ig, random, mean, bow, probe,
   preference_lora/reft, concept_lora/reft, preference_vector,
   concept_vector, hypersteer, hypernet, plus winrate/latent_stats
   evaluators — that we don't use. Some of them break in the wild
   depending on which transformers/peft minor version is installed,
   so eagerly loading them just to throw the symbols away is both
   wasteful and a recurring source of ImportError surprises.

3. Monkey-patch ``LanguageModel.chat_completion`` to retry transient
   OpenAI errors (429/5xx) with exponential backoff and return a
   sentinel ``[OPENAI_SKIP]`` on permanent client errors (400/403/422)
   instead of raising. Upstream propagates exceptions through
   ``asyncio.gather`` which kills the entire eval on a single bad
   prompt — one content-policy 403 out of 9742 calls aborted a 6-hour
   p2 run after 101/500 concepts. Also bumps the default
   ``chat_completions`` ``batch_size`` from 32 → 128 for higher
   OpenAI concurrency. Sentinels propagate through the parquet, and
   the downstream mean computation in ``_read_axbench_metrics`` is
   unaffected because skipped concepts simply don't appear in
   ``latent.jsonl``.

Idempotent: a second call sees ``axbench._ep_bootstrapped = True`` and
no-ops. If something else has already imported (untrimmed) axbench before
us, we bail and trust upstream.
"""
import importlib.util
import sys
import types
from pathlib import Path


_TRIMMED_INIT = """
from .utils.plot_utils import *
from .utils.dataset import *
from .utils.constants import *
from .utils.prompt_utils import *
from .utils.model_utils import *

from .templates.html_templates import *
from .templates.prompt_templates import *

from .evaluators.aucroc import *
from .evaluators.ppl import *
from .evaluators.lm_judge import *
from .evaluators.hard_negative import *

from .models.sae import *
from .models.language_models import *
from .models.prompt import *
from .models.ep import *

from .scripts.args.eval_args import *
from .scripts.args.training_args import *
from .scripts.args.dataset_args import *

from .scripts.evaluate import *
from .scripts.inference import *
"""


def _stub_stanza() -> None:
    if "stanza" in sys.modules:
        return
    fake = types.ModuleType("stanza")
    fake.Pipeline = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["stanza"] = fake


def _preload_axbench() -> None:
    existing = sys.modules.get("axbench")
    if existing is not None:
        # Either we already bootstrapped, or someone else imported untrimmed
        # axbench before us. Either way, do nothing.
        return
    spec = importlib.util.find_spec("axbench")
    if spec is None or not spec.submodule_search_locations:
        return  # axbench not on path; let the import fail naturally elsewhere
    mod = types.ModuleType("axbench")
    mod.__path__ = list(spec.submodule_search_locations)
    mod.__file__ = str(Path(spec.submodule_search_locations[0]) / "__init__.py")
    mod.__package__ = "axbench"
    mod.__spec__ = spec
    mod._ep_bootstrapped = True  # type: ignore[attr-defined]
    sys.modules["axbench"] = mod
    exec(_TRIMMED_INIT, mod.__dict__)


def _patch_language_model() -> None:
    """Wrap LanguageModel.chat_completion with retry/skip and bump batch_size."""
    import asyncio
    import logging

    axbench = sys.modules.get("axbench")
    if axbench is None or not getattr(axbench, "_ep_bootstrapped", False):
        return  # bootstrap didn't load axbench; nothing to patch

    try:
        from axbench.models import language_models as _lm
    except Exception:
        return

    if getattr(_lm.LanguageModel, "_ep_patched", False):
        return  # already patched in this process

    _orig_chat = _lm.LanguageModel.chat_completion
    _orig_batch = _lm.LanguageModel.chat_completions

    _TRANSIENT = ("429", "500", "502", "503", "504")
    _PERMANENT = ("400", "403", "404", "422")

    async def _patched_chat_completion(self, client, prompt, api_name):
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                return await _orig_chat(self, client, prompt, api_name)
            except Exception as e:  # noqa: BLE001
                last_err = e
                err = str(e)
                if any(c in err for c in _TRANSIENT):
                    await asyncio.sleep(2 ** attempt)
                    continue
                if any(c in err for c in _PERMANENT):
                    logging.warning(
                        "[ep patch] OpenAI %s — returning [OPENAI_SKIP] for prompt=%r",
                        err[:120], prompt[:60],
                    )
                    return ("[OPENAI_SKIP]", None)
                # Unknown: retry once with backoff
                await asyncio.sleep(2 ** attempt)
        logging.warning(
            "[ep patch] OpenAI exhausted retries (%s) — returning [OPENAI_SKIP]",
            last_err,
        )
        return ("[OPENAI_SKIP]", None)

    async def _patched_chat_completions(self, api_names, prompts, batch_size=128):
        return await _orig_batch(self, api_names, prompts, batch_size=batch_size)

    _lm.LanguageModel.chat_completion = _patched_chat_completion
    _lm.LanguageModel.chat_completions = _patched_chat_completions
    _lm.LanguageModel._ep_patched = True


_stub_stanza()
_preload_axbench()
_patch_language_model()
