from inspect import signature
from typing import Any

import torch
from multipage_classifier.page_comparison_encoder import (
    PageComparisonEncoder,
    PageComparisonConfig,
    ORDER_NAMES,
)
from utils.lightning import BaseLightningModule, Mode


class PageComparisonEncoderPLModule(BaseLightningModule):
    model: PageComparisonEncoder

    def __init__(self, cfg: PageComparisonConfig):
        super().__init__()

        self.save_hyperparameters()

        self.model: PageComparisonEncoder = PageComparisonEncoder(cfg)

        self.model_input_keys = list(signature(self.model.forward).parameters.keys())

        self.config = cfg

        self.set_default_metrics(
            "order", task="multiclass", num_classes=len(ORDER_NAMES)
        )
        self.set_default_metrics(
            "page_nr", task="multiclass", num_classes=self.config.max_page_nr
        )
        self.set_default_metrics(
            "doc_id",
            task="multiclass",
            num_classes=self.config.max_page_nr,
            confmat=False,
        )
        self.set_default_metrics(
            "doc_class",
            task="multiclass",
            num_classes=self.config.num_classes,
            confmat=False,
        )

    def step(self, batch: Any, *_):
        pred = self.model.forward(batch["pixel_values"])

        gt = {}
        losses = {}

        ### Classification Head ###
        gt["doc_class"] = batch["doc_class"]
        losses["doc_class_loss"] = torch.nn.NLLLoss()(
            pred["doc_class"], gt["doc_class"]
        )

        # Compute the order ground truth
        same_doc = (batch["doc_id"].unsqueeze(1) == batch["doc_id"].unsqueeze(0)).int()
        same_letter = (
            batch["letter_id"].unsqueeze(1) == batch["letter_id"].unsqueeze(0)
        ).int()
        less = (batch["page_nr"].unsqueeze(1) < batch["page_nr"].unsqueeze(0)).int() * 1
        greater = (
            batch["page_nr"].unsqueeze(1) > batch["page_nr"].unsqueeze(0)
        ).int() * 2
        same = (
            batch["page_nr"].unsqueeze(1) == batch["page_nr"].unsqueeze(0)
        ).int() * 3

        gt["order"] = (
            ((less + greater + same) * same_doc * same_letter).long().view(-1)
        )  # [ones]

        losses["order_loss"] = torch.nn.NLLLoss()(pred["order"], gt["order"])

        return (pred, gt, losses)

    def eval_step(self, batch, batch_idx, mode: Mode):
        self.mode = mode
        preds, gt, losses = self.step(batch, batch_idx)

        preds = self.model.postprocess(preds)

        gt["doc_id"] = batch["doc_id"]
        gt["page_nr"] = batch["page_nr"]

        for k in ["doc_id", "page_nr"]:
            pred = preds[k]

            # Create an empty target tensor with the desired shape
            target_tensor = torch.zeros(
                (len(pred), self.config.max_page_nr), device=self.device
            )

            # Use scatter_ to set 1 at specific indices for each row
            target_tensor.scatter_(1, pred.unsqueeze(1), 1)

            preds[k] = target_tensor
        self.update_metrics(preds, gt)
        return self.log_losses(losses)

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, Mode.VALID)

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, Mode.TEST)

    def configure_optimizers(self):
        lr = 2e-5
        weight_decay = 1e-4

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return [optimizer]
