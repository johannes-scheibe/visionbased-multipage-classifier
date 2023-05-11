from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torchmetrics.classification import MulticlassConfusionMatrix
from multipage_classifier.page_classifier import (MultipageClassifier,
                                                  MultipageClassifierConfig)

class MultipageClassifierPLModule(pl.LightningModule):
    
    def __init__(self, config: MultipageClassifierConfig):
        super().__init__()

        self.config = config

        self.classifier = MultipageClassifier(config)

        self.save_hyperparameters()

    def step(self, batch: Any):
        batch_size = len(batch["doc_id"])

        # Predictions
        pred = self.classifier.forward(batch["pixel_values"])

        # Ground truth
        doc_ids = batch["doc_id"]
        doc_ids = torch.cat(
            [
                doc_ids,
                torch.tensor([-1] * (self.config.max_pages -
                             len(doc_ids)), device=self.device),
            ]
        )
        ground_truth = (doc_ids[:batch_size].view(-1, 1)
                        == doc_ids.view(1, -1)).float()

        loss = sigmoid_focal_loss(pred, ground_truth)

        return pred, ground_truth, loss  # TODO: Check this

    def training_step(self, batch: Any, batch_idx: int):
        _, _, loss = self.step(batch)
        metrics = {"loss": loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch: Any, batch_idx: int):
        pred, _, loss = self.step(batch)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch: Any, batch_idx: int):
        _, _, loss = self.step(batch)
        metrics = {"test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        lr = 3e-5

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return [optimizer]


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.50, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs  # .sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean() * 100
