# ruff: noqa: E402
# =====================================================================
# Vendored module: ships into baselines/axbench/axbench/models/ep.py.
#
# The relative imports below (`.interventions`, `.mean`, `.model`, `.probe`,
# `..utils.model_utils`) only resolve when this file is sitting inside
# AxBench's package tree, so this module is not importable from ep itself.
# It is the version-controlled source of truth — `_ensure_axbench_ep_module`
# in scripts/build_partitions.py copies it (verbatim) into the AxBench
# checkout before any AxBench subprocess starts.
# =====================================================================
"""Exemplar Partitioning (EP) — unsupervised concept discovery.

Loads a pre-built EP dictionary and selects, per AxBench concept, the
partition with the strongest positive-vs-negative cosine-contrast on the
synthetic per-concept training data.

We go through the same `EPDictionarySAE` adapter the SAEBench and
compare-sae paths use, configured with `readout="cosine"`: pure cosine
similarity in centered space, magnitude removed so high-norm tokens cannot
dominate the per-concept mean. The chosen unit direction is then plugged
into AxBench's pyvene `AdditionIntervention` / `SubspaceIntervention` as
the steering vector.

- "exemplar":  Partition.exemplar_direction      (first-arrival, immutable)
- "mean":      Partition.mean_member_direction   (spherical mean of members)

Two environment switches, both set by scripts/build_partitions.py:

EP_AXBENCH_SELECTION  how the region is chosen from the training examples.
  "auroc" (the protocol reported in the paper): mirrors AxBench's
  GemmaScopeSAEMaxAUC — max cosine over the positions of each training
  sequence, then the region with the highest training-set AUROC. Same
  pooling as the evaluator uses at test time.
  "contrast" (the token-mean contrast rule, paper appendix F): mean cosine over all positive
  tokens minus mean cosine over all negative tokens, argmax over regions.

EP_ACT_CACHE_DIR  if set, residual activations for every AxBench sequence
  (train and latent-eval) are stored under this directory, keyed by the
  sequence's token ids, and read back instead of re-running the model.
  Selection rules and bases can then be re-run in minutes.
"""
import atexit
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from pyvene import IntervenableConfig, IntervenableModel
from torch.utils.data import DataLoader

from ep.saebench_adapter import EPDictionarySAE

from .interventions import AdditionIntervention, SubspaceIntervention
from .mean import LogisticRegressionModel
from .model import Model
from .probe import make_data_module
from ..scripts.inference import prepare_df
from ..utils.model_utils import (
    gather_residual_activations,
    set_decoder_norm_to_unit_norm,
)

import logging
logger = logging.getLogger(__name__)


_ADAPTER_CACHE: dict[tuple[str, str, str], EPDictionarySAE] = {}


def _load_adapter(path: str, basis: str, device: torch.device) -> EPDictionarySAE:
    """Return a cached cosine-readout adapter for this (dictionary, basis, device)."""
    key = (path, basis, str(device))
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    with open(path, "rb") as f:
        dictionary = pickle.load(f)
    adapter = EPDictionarySAE(
        dictionary=dictionary,
        model_name="",
        hook_layer=0,
        device=device,
        dtype=torch.float32,
        basis=basis,
        readout="cosine",
    )
    adapter.eval()
    _ADAPTER_CACHE[key] = adapter
    logger.warning(
        "Loaded EP library (%s basis): %d centroids, dim=%d, threshold=%.4f",
        basis, len(dictionary.partitions), dictionary.center.shape[0],
        dictionary.threshold,
    )
    return adapter


_SELECTION = os.environ.get("EP_AXBENCH_SELECTION", "auroc")
_ACT_CACHE_DIR = os.environ.get("EP_ACT_CACHE_DIR") or None
# One stem (split + concept) resident at a time: {seq_key: [T, D] activations}.
_ACT_CACHE: dict[str, dict[str, torch.Tensor]] = {}
_ACT_CACHE_DIRTY: set[str] = set()


def _act_cache_path(stem: str) -> Path:
    return Path(_ACT_CACHE_DIR) / f"{stem}.pt"


def _act_cache_flush() -> None:
    for stem in list(_ACT_CACHE_DIRTY):
        path = _act_cache_path(stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(_ACT_CACHE[stem], tmp)
        os.replace(tmp, path)
    _ACT_CACHE_DIRTY.clear()


atexit.register(_act_cache_flush)


def _act_cache_get(stem: str) -> dict[str, torch.Tensor]:
    if stem not in _ACT_CACHE:
        _act_cache_flush()
        _ACT_CACHE.clear()
        path = _act_cache_path(stem)
        _ACT_CACHE[stem] = torch.load(path) if path.exists() else {}
    return _ACT_CACHE[stem]


def _seq_key(ids: torch.Tensor) -> str:
    return hashlib.sha1(ids.to(torch.int64).cpu().numpy().tobytes()).hexdigest()


def _residual_activations(model, layer, input_ids, attention_mask, stem):
    """Residual activations at `layer` for each row's non-pad tokens, one
    [T_i, D] tensor per row, on the model's device. Served from the activation
    cache when EP_ACT_CACHE_DIR is set; otherwise (or on a miss) the model runs."""
    mask = attention_mask.bool()
    rows = range(input_ids.shape[0])
    if _ACT_CACHE_DIR is None:
        acts = gather_residual_activations(
            model, layer, {"input_ids": input_ids, "attention_mask": attention_mask})
        return [acts[i][mask[i]] for i in rows]
    store = _act_cache_get(stem)
    keys = [_seq_key(input_ids[i][mask[i]]) for i in rows]
    out = [store.get(k) for k in keys]
    if any(a is None for a in out):
        acts = gather_residual_activations(
            model, layer, {"input_ids": input_ids, "attention_mask": attention_mask})
        for i, k in enumerate(keys):
            if out[i] is None:
                out[i] = acts[i][mask[i]].cpu()
                store[k] = out[i]
        _ACT_CACHE_DIRTY.add(stem)
    return [a.to(input_ids.device) for a in out]


def _column_auroc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AUROC of every column of `scores` [n, K] against boolean `labels` [n],
    via the Mann-Whitney statistic with tie-averaged ranks (matches
    sklearn.metrics.roc_auc_score)."""
    ranks = rankdata(scores.cpu().numpy(), axis=0)
    y = labels.cpu().numpy()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    u = ranks[y].sum(axis=0) - n_pos * (n_pos + 1) / 2
    return torch.from_numpy((u / (n_pos * n_neg)).astype(np.float32)).to(scores.device)


def _stem(examples, split: str) -> str:
    """Cache file stem for one per-concept dataframe. The latent-eval frames
    carry a concept_id column; the training frames don't (prepare_df_combined
    drops it), so those are keyed by a hash of their input texts instead."""
    if "concept_id" in examples:
        return f"{split}_c{int(examples['concept_id'].iloc[0])}"
    digest = hashlib.sha1("\x00".join(examples["input"].tolist()).encode()).hexdigest()[:16]
    return f"{split}_{digest}"


class _EPBase(Model):
    """Shared logic for exemplar-partition selection. Subclasses set BASIS."""

    BASIS: str = "exemplar"

    def make_model(self, **kwargs):
        model_params = kwargs.get("model_params", None)
        ep_library_path = kwargs.get(
            "ep_library_path",
            getattr(model_params, "ep_library_path", None),
        ) or os.environ.get("EP_LIBRARY_PATH")
        assert ep_library_path is not None, f"{self.__str__()} requires ep_library_path"

        self.adapter = _load_adapter(ep_library_path, self.BASIS, self.device)
        # W_dec rows are the unit basis directions in centered space — same
        # tensor we'd have stacked manually as `centroids`.
        self.centroids = self.adapter.W_dec
        embed_dim = self.centroids.shape[1]

        mode = kwargs.get("mode", "train")
        intervention_type = kwargs.get("intervention_type", "addition")
        low_rank_dimension = kwargs.get("low_rank_dimension", 1)

        if mode == "steering":
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=embed_dim, low_rank_dimension=low_rank_dimension,
                )
            elif intervention_type == "clamping":
                ax = SubspaceIntervention(
                    embed_dim=embed_dim, low_rank_dimension=low_rank_dimension,
                )
            else:
                raise ValueError(f"Intervention type {intervention_type} not supported")
            self.ax = ax
            self.ax.train()
            layers = self.steering_layers if self.steering_layers else [self.layer]
            ax_config = IntervenableConfig(representations=[{
                "layer": lyr,
                "component": f"model.layers[{lyr}].output",
                "low_rank_dimension": low_rank_dimension,
                "intervention": self.ax,
            } for lyr in layers])
            ax_model = IntervenableModel(ax_config, self.model)
            ax_model.set_device(self.device)
            self.ax_model = ax_model
        else:
            ax = LogisticRegressionModel(embed_dim, low_rank_dimension)
            ax.to(self.device)
            self.ax = ax

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        return DataLoader(
            data_module["train_dataset"],
            shuffle=True,
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
        )

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        # Base Model.predict_steer has its perplexity block commented out, so
        # the inherited result lacks the `_perplexity` column that
        # PerplexityEvaluator requires. Re-add it post-hoc, mirroring
        # PromptSteering: response-only perplexity under the unintervened LM.
        out = super().predict_steer(examples, **kwargs)

        self.model.eval()
        self.tokenizer.padding_side = "left"
        batch_size = kwargs.get("batch_size", 64)
        generations = out["steered_generation"]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        all_perplexities: list[float] = []
        for i in range(0, len(generations), batch_size):
            batch_texts = generations[i:i + batch_size]
            input_ids = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True,
            ).input_ids.to(self.device)
            attn = (input_ids != self.tokenizer.pad_token_id).float()
            outputs = self.model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits[:, :-1, :].contiguous()
            target_ids = input_ids[:, 1:].contiguous()
            token_losses = loss_fct(
                logits.view(-1, logits.size(-1)), target_ids.view(-1),
            ).view(input_ids.size(0), -1)
            mask = attn[:, 1:].contiguous()
            seq_lengths = mask.sum(dim=1).clamp(min=1)
            seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
            all_perplexities.extend(torch.exp(seq_losses).tolist())
        out["perplexity"] = all_perplexities
        return out

    @torch.no_grad()
    def train(self, examples, **kwargs):
        prefix_length = kwargs.get("prefix_length", 1)
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        stem = _stem(examples, "train")

        # One [T_i - prefix, D] tensor per training sequence, plus its label.
        seq_acts, seq_labels = [], []
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                acts = _residual_activations(
                    self.model, self.layer, inputs["input_ids"],
                    inputs["attention_mask"], stem,
                )
                for a, label in zip(acts, inputs["labels"]):
                    seq_acts.append(a[prefix_length:].float())
                    seq_labels.append(bool(label == 1))

        lengths = [a.shape[0] for a in seq_acts]
        # Cosine-readout adapter: centers internally, normalises, projects
        # onto the (already unit) basis directions -> [n_tokens, K].
        # Magnitude is removed so high-norm tokens cannot dominate.
        scores = self.adapter.encode(torch.cat(seq_acts))
        labels = torch.tensor(seq_labels, device=scores.device)

        if _SELECTION == "contrast":
            token_labels = torch.repeat_interleave(
                labels, torch.tensor(lengths, device=labels.device))
            stat = scores[token_labels].mean(dim=0) - scores[~token_labels].mean(dim=0)
        elif _SELECTION == "auroc":
            seq_max = torch.stack([s.max(dim=0).values for s in scores.split(lengths)])
            stat = _column_auroc(seq_max, labels)
        else:
            raise ValueError(f"EP_AXBENCH_SELECTION={_SELECTION!r}; expected contrast or auroc")
        best = int(stat.argmax().item())

        chosen = self.centroids[best].unsqueeze(0).to(self.ax.proj.weight.dtype)
        self.ax.proj.weight.data = chosen
        if self.ax.proj.bias is not None:
            self.ax.proj.bias.data = torch.zeros_like(self.ax.proj.bias.data)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning(
            "%s selected centroid %d (%s=%.4f) of %d",
            self.__str__(), best, _SELECTION, stat[best].item(), self.centroids.shape[0],
        )

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        # Upstream Model.predict_latent, with the forward pass routed through
        # the activation cache. Falls back to upstream when no cache is set.
        if _ACT_CACHE_DIR is None:
            return super().predict_latent(examples, **kwargs)
        self.ax.eval()
        batch_size = kwargs.get("batch_size", 32)
        return_max_act_only = kwargs.get("return_max_act_only", False)
        is_chat_model = kwargs.get("is_chat_model", False)
        eager_prepare_df = kwargs.get("eager_prepare_df", False)
        overwrite_concept_id = kwargs.get("overwrite_concept_id", None)
        prefix_length = kwargs["prefix_length"]
        stem = _stem(examples, "latent")

        all_acts, all_max_act, all_max_act_idx, all_max_token, all_tokens = [], [], [], [], []
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            if eager_prepare_df:
                batch = prepare_df(batch, self.tokenizer, is_chat_model)
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", padding=True,
                add_special_tokens=True,
            ).to(self.device)
            seq_acts = _residual_activations(
                self.model, self.layer, inputs["input_ids"], inputs["attention_mask"], stem,
            )
            for a, row in zip(seq_acts, batch.itertuples()):
                cid = overwrite_concept_id if overwrite_concept_id is not None else row.concept_id
                acts = self.ax(a[prefix_length:])[:, cid].flatten().float().cpu().numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                all_max_act.append(max_act)
                if not return_max_act_only:
                    max_act_idx = [j for j, x in enumerate(acts) if x == max_act][0]
                    tokens = self.tokenizer.tokenize(row.input)[prefix_length - 1:]
                    all_acts.append(acts)
                    all_max_act_idx.append(max_act_idx)
                    all_max_token.append(tokens[max_act_idx])
                    all_tokens.append(tokens)
            torch.cuda.empty_cache()

        if return_max_act_only:
            return {"max_act": all_max_act}
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens,
        }


class EPExemplar(_EPBase):
    BASIS = "exemplar"

    def __str__(self):
        return "EPExemplar"


class EPMean(_EPBase):
    BASIS = "mean"

    def __str__(self):
        return "EPMean"
