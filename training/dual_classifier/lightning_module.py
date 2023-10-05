
from inspect import signature
from typing import Any, cast

import torch
from multipage_classifier.dual_classifier import DualClassifier, DualClassifierConfig, ORDER_NAMES
from utils.lightning import BaseLightningModule, Mode



class DualClassifierPLModule(BaseLightningModule):
    model: DualClassifier

    def __init__(self, cfg: DualClassifierConfig):
        super().__init__()
        
        self.save_hyperparameters()

        self.model: DualClassifier = DualClassifier(cfg)

        self.model_input_keys = list(signature(self.model.forward).parameters.keys())

        self.config = cfg
        
        self.set_default_metrics("order", task="multiclass", num_classes=len(ORDER_NAMES))        
        self.set_default_metrics("page_nr", task="multiclass", num_classes=self.config.max_page_nr)        
        self.set_default_metrics("doc_id", task="multilabel", num_labels=self.config.max_page_nr, confmat=False)        
        self.set_default_metrics("doc_class", task="multiclass", num_labels=self.config.num_classes, confmat=False)        

    def step(self, batch: Any, *_):
        pred= self.model.forward(batch["pixel_values"])

        gt = {}
        losses = {}

        ### Classification Head ###
        gt["doc_class"] = batch["doc_class"]
        losses["doc_class_loss"] = torch.nn.NLLLoss()(pred["doc_class"], gt["doc_class"])

        
        # Compute the order ground truth
        same_doc = (batch["doc_id"].unsqueeze(1) == batch["doc_id"].unsqueeze(0)).int()
        same_letter = (batch["letter_id"].unsqueeze(1) == batch["letter_id"].unsqueeze(0)).int()
        less = (batch["page_nr"].unsqueeze(1) < batch["page_nr"].unsqueeze(0)).int() * 1
        greater = (batch["page_nr"].unsqueeze(1) > batch["page_nr"].unsqueeze(0)).int() * 2
        same = (batch["page_nr"].unsqueeze(1) == batch["page_nr"].unsqueeze(0)).int() * 3

        gt["order"] = (
            ((less + greater + same) * same_doc * same_letter).long().view(-1)
        )  # [ones]

        losses["order_loss"] = torch.nn.NLLLoss()(pred["order"], gt["order"])

        return (
            pred,
            gt,
            losses
        )
    
    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        self.mode = Mode.VALID
        preds, gt, losses = self.step(batch, batch_idx)

        preds = self.model.postprocess(preds)
        gt["doc_id"] = batch["doc_id"]
        gt["page_nr"] = batch["page_nr"]

        self.update_metrics(preds, gt)
        return self.log_losses(losses)

    def configure_optimizers(self):
        lr = 2e-5
        weight_decay = 1e-4

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return [optimizer]
