"""Label partitions in a saved dictionary using LLM autointerp.

For each partition, formats its sample prompts with per-token importance
highlighting (<<...>> markers around high-activation tokens), sends them
to Claude, and stores the returned label on the Partition object.

Usage:
    uv run python scripts/label_dictionary.py path/to/dict.pkl \
        --model-name google/gemma-2-2b \
        --hook blocks.12.hook_resid_post

    # Dry run — print formatted prompts without calling the LLM:
    uv run python scripts/label_dictionary.py path/to/dict.pkl \
        --model-name google/gemma-2-2b \
        --hook blocks.12.hook_resid_post \
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import transformer_lens as tl

logger = logging.getLogger(__name__)

ACT_THRESHOLD_FRAC = 0.01  # match SAEBench: 1% of max activation

SYSTEM_PROMPT = (
    "We're studying neurons in a neural network. Each neuron activates on some "
    "particular word/words/substring/concept in a short document. The activating "
    "words in each document are indicated with << ... >>. We will give you a list "
    "of documents on which the neuron activates, in order from most strongly "
    "activating to least strongly activating. Look at the parts of the document "
    "the neuron activates for and summarize in a single sentence what the neuron "
    "is activating on. Try not to be overly specific in your explanation. Note "
    "that some neurons will activate only on specific words or substrings, but "
    "others will activate on most/all words in a sentence provided that sentence "
    "contains some particular concept. Your explanation should cover most or all "
    "activating words (for example, don't give an explanation which is specific "
    "to a single word if all words in a sentence cause the neuron to activate). "
    "Pay attention to things like the capitalization and punctuation of the "
    "activating words or concepts, if that seems relevant. Keep the explanation "
    "as short and simple as possible, limited to 20 words or less. Omit "
    "punctuation and formatting. You should avoid giving long lists of words.\n\n"
    'Some examples: "This neuron activates on the word \'knows\' in rhetorical '
    'questions", and "This neuron activates on verbs related to decision-making '
    'and preferences", and "This neuron activates on the substring \'Ent\' at '
    'the start of words", and "This neuron activates on text about government '
    'economic policy".'
)


def compute_token_importance(model, texts, hook_name, exemplar_direction, center, device):
    """Per-token cosine similarity to a partition exemplar direction.

    Per-token activations are first centered by the dictionary's activation
    center, then cosine-normalised, then dotted with the partition's
    (already-unit-norm) exemplar direction. Tokenization and the forward
    pass are batched; the device→host transfer is one tensor.
    """
    if not texts:
        return []

    batch = model.to_tokens(texts, prepend_bos=True, padding_side="right")
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    lengths = (batch != pad_id).sum(dim=1)

    acts: dict = {}

    def cache_hook(act, hook):
        acts[hook.name] = act.detach()

    model.reset_hooks()
    model.add_hook(hook_name, cache_hook, "fwd")
    with torch.no_grad():
        model(batch)
    model.reset_hooks()

    # Transfer in source dtype, upcast on CPU — saves bandwidth for bf16 models.
    all_acts = acts[hook_name].cpu().float().numpy()
    lengths_np = lengths.cpu().numpy()
    batch_np = batch.cpu().numpy()
    e_dir = np.asarray(exemplar_direction)

    results = []
    for i in range(len(texts)):
        L = int(lengths_np[i])
        if L < 2:
            results.append(([], np.zeros(0, dtype=np.float32)))
            continue
        pos_acts = all_acts[i, 1:L] - center
        norms = np.linalg.norm(pos_acts, axis=1, keepdims=True) + 1e-12
        sims = (pos_acts / norms) @ e_dir
        token_strs = [model.tokenizer.decode([int(t)]) for t in batch_np[i, 1:L]]
        results.append((token_strs, sims))

    return results


def format_examples(importance_results: list, threshold_frac: float = ACT_THRESHOLD_FRAC) -> str:
    lines = []
    for idx, (token_strs, importance) in enumerate(importance_results, 1):
        if len(importance) == 0:
            continue
        threshold = threshold_frac * float(importance.max())
        parts = []
        for tok, imp in zip(token_strs, importance):
            if float(imp) > threshold:
                parts.append(f"<<{tok}>>")
            else:
                parts.append(tok)
        lines.append(f"{idx}. {''.join(parts)}")
    return "\n".join(lines)


def label_partition(client, examples_str: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=60,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"The activating documents are given below:\n\n{examples_str}",
        }],
    )
    return response.content[0].text.strip()


def label_dictionary(
    dictionary_path: Path,
    model_name: str,
    hook_name: str,
    device: str = "cpu",
    min_members: int = 5,
    max_partitions: int | None = None,
    dry_run: bool = False,
):
    with dictionary_path.open("rb") as f:
        dictionary = pickle.load(f)

    n_partitions = len(dictionary.partitions)
    logger.info("Loaded dictionary with %d partitions from %s",
                n_partitions, dictionary_path)

    logger.info("Loading model %s", model_name)
    model = tl.HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    ranked = sorted(
        range(n_partitions),
        key=lambda i: dictionary.partitions[i].member_count,
        reverse=True,
    )

    if not dry_run:
        import anthropic
        client = anthropic.Anthropic()

    labelled = 0
    for rank, idx in enumerate(ranked):
        p = dictionary.partitions[idx]
        if p.member_count < min_members:
            continue
        if max_partitions is not None and labelled >= max_partitions:
            break

        sorted_prompts = sorted(p.sample_prompts)
        texts = [text for _, text, pos in sorted_prompts]
        if not texts:
            continue

        importance_results = compute_token_importance(
            model, texts, hook_name, p.exemplar_direction, dictionary.center, device,
        )
        examples_str = format_examples(importance_results)

        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"Partition {idx} (rank {rank + 1}, {p.member_count} members)")
            print(f"{'=' * 60}")
            print(examples_str)
            labelled += 1
            continue

        label = label_partition(client, examples_str)
        p.label = label
        labelled += 1
        logger.info("  [%d/%d] partition %d (%d members): %s",
                    labelled, n_partitions, idx, p.member_count, label)

    if not dry_run:
        out_path = dictionary_path.with_stem(dictionary_path.stem + "_labelled")
        with out_path.open("wb") as f:
            pickle.dump(dictionary, f)
        logger.info("Saved labelled dictionary to %s", out_path)

    return dictionary


def main():
    parser = argparse.ArgumentParser(
        description="Label partitions in a saved dictionary via autointerp",
    )
    parser.add_argument("dictionary", type=Path, help="Path to dictionary .pkl")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--hook", type=str, required=True,
                        help="Hook name, e.g. blocks.4.hook_resid_post")
    parser.add_argument("--device", type=str,
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--min-members", type=int, default=5)
    parser.add_argument("--max-partitions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    label_dictionary(
        dictionary_path=args.dictionary,
        model_name=args.model_name,
        hook_name=args.hook,
        device=args.device,
        min_members=args.min_members,
        max_partitions=args.max_partitions,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
