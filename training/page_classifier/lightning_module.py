from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from multipage_classifier.page_classifier import (MultipageClassifier,
                                                  MultipageClassifierConfig)
from utils.lightning import BaseLightningModule

class MultipageClassifierPLModule(BaseLightningModule):
    
    def __init__(self, config: MultipageClassifierConfig):
        super().__init__()

        self.config = config

        self.classifier = MultipageClassifier(config)

        self.set_default_metrics("doc_class", task="multiclass", num_classes=self.config.num_classes)        
        self.set_default_metrics("page_nr", task="multiclass", num_classes=self.config.max_page_nr)        
        self.set_default_metrics("doc_id", task="multilabel", num_labels=self.config.max_pages, confmat=False)        

        self.save_hyperparameters()

    def step(self, batch: Any, *_) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]:
        batch_size = len(batch["doc_id"])

        # Predictions
        preds = self.classifier.forward(batch["pixel_values"])

        # Ground truth
        gt = {}
        gt.update({k: batch[k] for k in self.classifier.separator.heads.keys()})
        
        doc_ids = batch["doc_id"]
        doc_ids = torch.cat(
            [
                doc_ids,
                torch.tensor([-1] * (self.config.max_pages -
                             len(doc_ids)), device=self.device),
            ]
        )
        gt["doc_id"] = (doc_ids[:batch_size].view(-1, 1) == doc_ids.view(1, -1)).int()

        losses = {}
        for k, v in preds.items():
            if k == "doc_id":
                losses["doc_id_loss"] = sigmoid_focal_loss(v, gt[k].float())
                continue
            losses[f"{k}_loss"] = torch.nn.NLLLoss()(v, batch[k])
        return preds, gt, losses

    def configure_optimizers(self):
        lr: float = 2e-5
        weight_decay: float = 1e-4
        lr_decay: float = 1
        
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
        return [optimizer], [scheduler]


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
