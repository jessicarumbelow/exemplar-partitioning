"""Activation extraction.

Two extractors:

- `extract_per_position`: every position in the prompt, one activation per
  position. Primary path for corpus-scale discovery.
- `extract_final_position`: one activation per prompt at the final token.
  Useful for structured tasks (IOI, SVO) where the prompt-final position is
  the locus of computation.

Both are forward-only: no backward, no gradient. Sequences are length-grouped
and batched for GPU efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class ExtractionResult:
    """Result from an activation extractor."""

    x: np.ndarray                                   # (N, D)
    prompt_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    position_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    n_forward_passes: int = 0
    n_tokens: int = 0


def extract_per_position(
    model,
    prompts: Iterable[str],
    hook_name: str,
    max_positions_per_prompt: int | None = None,
    batch_size: int = 128,
) -> ExtractionResult:
    """Extract per-source-position activations.

    For each prompt of length L, harvests positions 1..L-1 (skipping BOS).
    Sequences in a batch are padded to the longest; each batch is one
    forward pass. Tokenization is batched; activations are gathered on the
    model's existing device into a flat (n_kept, D) tensor before a single
    host transfer.
    """
    import torch

    prompts = list(prompts)
    model.eval()

    if not prompts:
        d = model.cfg.d_model
        return ExtractionResult(x=np.zeros((0, d)))

    # One batched tokenize for all prompts; right-padded so BOS sits at 0
    # and content occupies positions [0, L).
    tokens = model.to_tokens(prompts, prepend_bos=True, padding_side="right")
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    lengths = (tokens != pad_id).sum(dim=1)
    if max_positions_per_prompt is not None:
        lengths = torch.clamp(lengths, max=max_positions_per_prompt)
    total_tokens = int(lengths.sum().item())

    all_x: list[np.ndarray] = []
    all_prompt_ids: list[np.ndarray] = []
    all_position_ids: list[np.ndarray] = []
    n_fwd = 0

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        sub_tokens = tokens[start:end]
        sub_lengths = lengths[start:end]
        B, max_len = sub_tokens.shape

        # Trim to the longest non-pad position in this sub-batch to avoid
        # paying for pad columns left over from the global pad.
        sub_max = int(sub_lengths.max().item())
        if sub_max < max_len:
            sub_tokens = sub_tokens[:, :sub_max]
            max_len = sub_max

        acts_ref: dict = {}

        def fwd(act, hook):
            acts_ref[hook.name] = act

        model.reset_hooks()
        model.add_hook(hook_name, fwd, "fwd")
        with torch.no_grad():
            model(sub_tokens)
        model.reset_hooks()
        n_fwd += 1

        act_tensor = acts_ref[hook_name]  # (B, max_len, D), on-device

        # Build a keep-mask on-device: positions in [1, L_i).
        positions = torch.arange(max_len, device=act_tensor.device)
        keep_mask = (positions[None, :] < sub_lengths[:, None]) & (
            positions[None, :] >= 1
        )
        # Single contiguous gather + single host transfer. Defer the fp32
        # upcast until CPU so bf16 models transfer at half the bandwidth.
        flat = act_tensor[keep_mask].detach().cpu().float().numpy()
        all_x.append(flat)

        # Build prompt/position ids on the CPU from sub_lengths.
        sub_lengths_np = sub_lengths.cpu().numpy()
        for i, L in enumerate(sub_lengths_np):
            L = int(L)
            if L < 2:
                continue
            all_prompt_ids.append(np.full(L - 1, start + i, dtype=np.int64))
            all_position_ids.append(np.arange(1, L, dtype=np.int64))

        del acts_ref, act_tensor, sub_tokens

    d = model.cfg.d_model
    empty = np.zeros((0, d))
    x = np.concatenate(all_x) if all_x else empty
    prompt_ids = (
        np.concatenate(all_prompt_ids) if all_prompt_ids
        else np.array([], dtype=np.int64)
    )
    position_ids = (
        np.concatenate(all_position_ids) if all_position_ids
        else np.array([], dtype=np.int64)
    )

    return ExtractionResult(
        x=x,
        prompt_ids=prompt_ids,
        position_ids=position_ids,
        n_forward_passes=n_fwd,
        n_tokens=total_tokens,
    )


def extract_final_position(
    model,
    prompts: Iterable[str],
    hook_name: str,
    batch_size: int = 256,
) -> ExtractionResult:
    """Extract one activation per prompt at the final token position.

    Tokenization is batched once for the whole prompt list; the final-position
    activations are gathered on-device into a single (B, D) tensor before one
    host transfer per sub-batch.
    """
    import torch

    prompts = list(prompts)
    d = model.cfg.d_model
    if not prompts:
        return ExtractionResult(x=np.zeros((0, d)))

    model.eval()

    tokens = model.to_tokens(prompts, prepend_bos=True, padding_side="right")
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    lengths = (tokens != pad_id).sum(dim=1)
    total_tokens = int(lengths.sum().item())
    keep_mask = lengths >= 1
    if not bool(keep_mask.all()):
        # Drop empty prompts but preserve original prompt indices.
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        tokens = tokens[keep_idx]
        lengths = lengths[keep_idx]
        prompt_indices = keep_idx.cpu().numpy()
    else:
        prompt_indices = np.arange(len(prompts), dtype=np.int64)
    if tokens.shape[0] == 0:
        return ExtractionResult(x=np.zeros((0, d)))

    all_x: list[np.ndarray] = []
    all_prompt_ids: list[np.ndarray] = []
    all_position_ids: list[np.ndarray] = []
    n_fwd = 0

    for start in range(0, tokens.shape[0], batch_size):
        end = min(start + batch_size, tokens.shape[0])
        sub_tokens = tokens[start:end]
        sub_lengths = lengths[start:end]
        sub_max = int(sub_lengths.max().item())
        if sub_max < sub_tokens.shape[1]:
            sub_tokens = sub_tokens[:, :sub_max]

        acts: dict = {}

        def fwd(act, hook):
            acts[hook.name] = act

        model.reset_hooks()
        model.add_hook(hook_name, fwd, "fwd")
        with torch.no_grad():
            model(sub_tokens)
        model.reset_hooks()
        n_fwd += 1

        final_pos = sub_lengths - 1
        batch_idx = torch.arange(sub_tokens.shape[0], device=acts[hook_name].device)
        # Transfer in source dtype, upcast on CPU — saves bandwidth for bf16 models.
        x = acts[hook_name][batch_idx, final_pos].detach().cpu().float().numpy()
        all_x.append(x)
        all_prompt_ids.append(prompt_indices[start:end])
        all_position_ids.append(final_pos.cpu().numpy().astype(np.int64))

        del acts, sub_tokens

    return ExtractionResult(
        x=np.concatenate(all_x),
        prompt_ids=np.concatenate(all_prompt_ids).astype(np.int64),
        position_ids=np.concatenate(all_position_ids).astype(np.int64),
        n_forward_passes=n_fwd,
        n_tokens=total_tokens,
    )


def normalize_extractor_output(result: ExtractionResult | tuple) -> ExtractionResult:
    """Coerce custom-extractor output to an ExtractionResult.

    Built-in extractors already return ExtractionResult. For ergonomic
    custom extractors, plain ``np.ndarray`` and tuples of either ``(x,)`` or
    ``(x, prompt_ids, position_ids)`` are also accepted.
    """
    if isinstance(result, ExtractionResult):
        return result
    if isinstance(result, np.ndarray):
        return ExtractionResult(x=result)
    if isinstance(result, tuple):
        if len(result) == 1:
            return ExtractionResult(x=result[0])
        if len(result) == 3:
            x, prompt_ids, position_ids = result
            return ExtractionResult(x=x, prompt_ids=prompt_ids, position_ids=position_ids)
    raise TypeError(f"Cannot normalize extractor output: {type(result).__name__}")
