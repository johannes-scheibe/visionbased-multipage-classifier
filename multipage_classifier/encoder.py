import warnings
from inspect import signature
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW, AutoModel

ORDER_NAMES = ["None", "Pred", "Succ", "Same"]

class EncoderForEmbedding(pl.LightningModule):
    encoder: nn.Module
    metrics: torch.nn.ModuleDict
    # confmat: torch.nn.ModuleDict
    heads: torch.nn.ModuleDict
    weights: torch.nn.ParameterDict

    def __init__(self, encoder: nn.Module, hidden_dim):
        super().__init__()
        #self.save_hyperparameters()

        self.encoder = encoder

        self.hidden_dim = hidden_dim
        self.model_input_keys = list(signature(self.encoder.forward).parameters.keys())

        # create order head
        self.heads = torch.nn.ModuleDict({})
        self.heads["order"] = torch.nn.Linear(self.hidden_dim, len(ORDER_NAMES))
        self.weights = torch.nn.ParameterDict({})
        self.weights["order"] = torch.nn.parameter.Parameter(
            data=torch.Tensor([0.1, 1, 1, 1]).float(), requires_grad=False
        )

        self.warned_nan = False

        # self.dropout = (
        #     torch.nn.Dropout(self.config.dropout)
        #     if self.config.dropout is not None
        #     else torch.nn.Identity()
        # )

    def forward(self, batch, **kwargs) -> torch.Tensor:
        model_input = {k: v for k, v in batch.items() if k in self.model_input_keys}
        cls_embeddings = self.encoder(**model_input)["last_hidden_state"]#[:, 0, :]  # [batch_size, seq_len, hidden_size]

        return cls_embeddings

    def step(self, batch: Any):
        emb = self.forward(batch)
        bs = len(emb)

        preds: Dict[str, torch.Tensor] = {}
        ground_truth = {}
        
        # Compute order head input
        diff = (emb.unsqueeze(1) - emb.unsqueeze(0)).view(-1, self.hidden_dim)
        #diff = torch.cat([diff, diff], -1)

        preds["order"] = torch.log_softmax(self.heads["order"](diff), dim=-1)

        same_doc = (batch["doc_id"].unsqueeze(1) == batch["doc_id"].unsqueeze(0)).int()
        same_letter = (batch["letter_id"].unsqueeze(1) == batch["letter_id"].unsqueeze(0)).int()
        less = (batch["page_nr"].unsqueeze(1) < batch["page_nr"].unsqueeze(0)).int() * 1
        greater = (batch["page_nr"].unsqueeze(1) > batch["page_nr"].unsqueeze(0)).int() * 2
        same = (batch["page_nr"].unsqueeze(1) == batch["page_nr"].unsqueeze(0)).int() * 3

        ground_truth["order"] = (
            ((less + greater + same) * same_doc * same_letter).long().view(-1)
        )  # [ones]

        return self.calc_metrics(preds, ground_truth)

    def calc_metrics(self, preds, gt):
        losses = {}
        metrics = {}
        res = []

        for k, v in preds.items():
            res.append(v)
            losses[f"{k}_loss"] = torch.nn.NLLLoss(weight=self.weights[k])(v, gt[k])
            metrics[f"{k}_acc"] = (v.argmax(dim=-1) == gt[k]).float().mean()

        return (
            losses,
            metrics,
            res,
        )

    def training_step(self, batch: Any, batch_idx: int):
        losses, metrics, _ = self.step(batch)
        metrics["loss"] = sum(losses.values())
        metrics.update({(f"loss_{k}" if "loss" not in k else k): v for k, v in losses.items()})
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch: Any, batch_idx: int):
        losses, metrics, _ = self.step(batch)
        metrics["val_loss"] = sum(losses.values())
        metrics.update({(f"loss_{k}" if "loss" not in k else k): v for k, v in losses.items()})
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: Any, batch_idx: int):
        losses, metrics, _ = self.step(batch)
        metrics["val_loss"] = sum(losses.values())
        metrics.update({(f"loss_{k}" if "loss" not in k else k): v for k, v in losses.items()})
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)

        return dict(
            optimizer=optimizer,
        )
