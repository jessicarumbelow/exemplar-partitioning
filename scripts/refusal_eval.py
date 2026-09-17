"""Generation and fluency diagnostics shared by the region-ablation experiment."""

from collections import Counter

import numpy as np
import torch


def format_chat(model, prompt: str) -> str:
    try:
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def ensure_pad_token(tok):
    """Keep padding distinct from eos so real special tokens remain visible."""
    if tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id:
        return tok.pad_token_id
    for cand in ("<|finetune_right_pad_id|>", "<pad>",
                 "<|reserved_special_token_0|>"):
        i = tok.convert_tokens_to_ids(cand)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id \
                and i != tok.eos_token_id:
            tok.pad_token = cand
            return i
    raise ValueError(f"{tok.name_or_path}: no pad token distinct from eos")


def generate_hooked(model, prompts, hooks, max_new_tokens, batch_size):
    """Greedy generation with intervention functions and no padded batches."""
    tok = model.tokenizer
    ensure_pad_token(tok)
    formatted = [format_chat(model, p) for p in prompts]
    by_len = {}
    for i, text in enumerate(formatted):
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        by_len.setdefault(n, []).append(i)
    groups = [indices[k:k + batch_size] for n in sorted(by_len)
              for indices in [by_len[n]] for k in range(0, len(indices), batch_size)]

    out_by_idx = {}
    for grp in groups:
        tok.padding_side = "left"
        enc = tok([formatted[i] for i in grp], return_tensors="pt",
                  padding=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.cfg.device)
        assert int((input_ids == tok.pad_token_id).sum()) == 0, "padding leaked in"
        model.reset_hooks()
        for name, hook in hooks:
            model.add_hook(name, hook, "fwd")
        try:
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=0.0, verbose=False)
            new = out[:, input_ids.shape[1]:]
        finally:
            model.reset_hooks()
        for i, row in zip(grp, new):
            out_by_idx[i] = tok.decode(row, skip_special_tokens=True)
    return [out_by_idx[i] for i in range(len(prompts))]


def _coherence(text: str) -> tuple[float, int]:
    words = text.split()
    if len(words) < 5:
        return (1.0 if not words else len(set(words)) / len(words)), 0
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return len(set(words)) / len(words), max(grams.values())


def score(generations):
    ratios, repeats = zip(*(_coherence(g) for g in generations)) if generations else ((), ())
    return {
        "n": len(generations),
        "unique_token_ratio": float(np.mean(ratios)),
        "max_repeat_4gram": float(np.mean(repeats)),
    }
