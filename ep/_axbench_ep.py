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
"""
import os
import pickle

import torch
from pyvene import IntervenableConfig, IntervenableModel
from torch.utils.data import DataLoader

from ep.saebench_adapter import EPDictionarySAE

from .interventions import AdditionIntervention, SubspaceIntervention
from .mean import LogisticRegressionModel
from .model import Model
from .probe import make_data_module
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

        pos_acts, neg_acts = [], []
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer,
                    {"input_ids": inputs["input_ids"],
                     "attention_mask": inputs["attention_mask"]},
                ).detach().float()
                nonbos_mask = inputs["attention_mask"][:, prefix_length:].bool()
                activations = activations[:, prefix_length:][nonbos_mask]
                labels = inputs["labels"].unsqueeze(1).repeat(
                    1, inputs["input_ids"].shape[1] - prefix_length,
                )[nonbos_mask]
                pos_acts.append(activations[labels == 1])
                neg_acts.append(activations[labels != 1])

        pos = torch.cat(pos_acts, dim=0)
        neg = torch.cat(neg_acts, dim=0)

        # Cosine-readout adapter: centers internally, normalises x_c, projects
        # onto the (already unit) basis directions. Result is per-input
        # cosine vectors of length K. Magnitude is removed so high-norm
        # tokens cannot dominate the per-concept mean.
        with torch.no_grad():
            pos_score = self.adapter.encode(pos)
            neg_score = self.adapter.encode(neg)
        contrast = pos_score.mean(dim=0) - neg_score.mean(dim=0)
        best = int(contrast.argmax().item())

        chosen = self.centroids[best].unsqueeze(0).to(self.ax.proj.weight.dtype)
        self.ax.proj.weight.data = chosen
        if self.ax.proj.bias is not None:
            self.ax.proj.bias.data = torch.zeros_like(self.ax.proj.bias.data)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning(
            "%s selected centroid %d (contrast=%.4f) of %d",
            self.__str__(), best, contrast[best].item(), self.centroids.shape[0],
        )


class EPExemplar(_EPBase):
    BASIS = "exemplar"

    def __str__(self):
        return "EPExemplar"


class EPMean(_EPBase):
    BASIS = "mean"

    def __str__(self):
        return "EPMean"
