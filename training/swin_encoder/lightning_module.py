
from inspect import signature
from typing import Any

import torch
from multipage_classifier.encoder.swin_encoder import SwinEncoder, SwinEncoderConfig
from utils.lightning import BaseLightningModule

ORDER_NAMES = ["None", "Pred", "Succ", "Same"]

class SwinEncoderPLModule(BaseLightningModule):
    encoder: SwinEncoder

    def __init__(self, config: SwinEncoderConfig):
        super().__init__()
        
        self.save_hyperparameters()

        self.encoder = SwinEncoder(**config.dict())

        self.hidden_dim = self.encoder.hidden_dim

        self.model_input_keys = list(signature(self.encoder.forward).parameters.keys())

        self.order_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, len(ORDER_NAMES)),
            torch.nn.ReLU(),
            torch.nn.Linear(len(ORDER_NAMES), len(ORDER_NAMES))
        )

        self.metrics = torch.nn.ModuleDict({})
        self.confmat = torch.nn.ModuleDict({})
        self.metrics["order"], self.confmat["order"] = self.get_metrics("order", task="multiclass", num_classes=len(ORDER_NAMES))        


    def forward(self, batch, **kwargs) -> torch.Tensor:
        model_input = {k: v for k, v in batch.items() if k in self.model_input_keys}

        cls_embeddings = self.encoder(**model_input)

        return cls_embeddings


    def step(self, batch: Any, *_):
        emb = self.forward(batch)

        # Compute order head input
        bs = len(emb)
        diff = emb.unsqueeze(0).repeat(bs, 1, 1)
        diff = torch.cat([diff, diff.permute(1, 0, 2)], -1).view(-1, self.hidden_dim * 2)
        
        # Compute the prediction
        pred = {} 
        pred["order"] = torch.log_softmax(self.order_head(diff), dim=-1)
        # Compute the ground truth
        gt = {}
        same_doc = (batch["doc_id"].unsqueeze(1) == batch["doc_id"].unsqueeze(0)).int()
        same_letter = (batch["letter_id"].unsqueeze(1) == batch["letter_id"].unsqueeze(0)).int()
        less = (batch["page_nr"].unsqueeze(1) < batch["page_nr"].unsqueeze(0)).int() * 1
        greater = (batch["page_nr"].unsqueeze(1) > batch["page_nr"].unsqueeze(0)).int() * 2
        same = (batch["page_nr"].unsqueeze(1) == batch["page_nr"].unsqueeze(0)).int() * 3

        gt["order"] = (
            ((less + greater + same) * same_doc * same_letter).long().view(-1)
        )  # [ones]

        loss = {"order_loss": torch.nn.NLLLoss()(pred["order"], gt["order"])}

        return (
            pred,
            gt,
            loss
        )

    def configure_optimizers(self):
        lr = 2e-5
        weight_decay = 1e-4

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return [optimizer]
