from functools import partial
import warnings
from inspect import signature
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW, AutoModel

from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock, SwinTransformerBlockV2, Permute
from pydantic import BaseModel

class SwinEncoderConfig(BaseModel):
    patch_size = [4,4]
    embed_dim = 128
    depths = [2, 2, 14, 2]
    num_heads = [4, 8, 16, 32]
    window_size = [10,10]


class SwinEncoder(nn.Module):
    """
    Adapts pytorch's implementation of the Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.permute = Permute([0, 3, 1, 2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pixel_values):
        x = self.features(pixel_values)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

ORDER_NAMES = ["None", "Pred", "Succ", "Same"]

class EncoderPLModule(pl.LightningModule):
    encoder: SwinEncoder

    def __init__(self, config: SwinEncoderConfig):
        super().__init__()
        
        self.save_hyperparameters()

        self.encoder = SwinEncoder(**config.dict())

        self.hidden_dim = self.encoder.num_features

        self.model_input_keys = list(signature(self.encoder.forward).parameters.keys())

        self.order_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, len(ORDER_NAMES)),
            torch.nn.ReLU(),
            torch.nn.Linear(len(ORDER_NAMES), len(ORDER_NAMES))
        )



    def forward(self, batch, **kwargs) -> torch.Tensor:
        model_input = {k: v for k, v in batch.items() if k in self.model_input_keys}

        cls_embeddings = self.encoder(**model_input)

        return cls_embeddings

    def step(self, batch: Any):
        emb = self.forward(batch)

        # Compute order head input
        bs = len(emb)
        diff = emb.unsqueeze(0).repeat(bs, 1, 1)
        diff = torch.cat([diff, diff.permute(1, 0, 2)], -1).view(-1, self.hidden_dim * 2)

        same_doc = (batch["doc_id"].unsqueeze(1) == batch["doc_id"].unsqueeze(0)).int()
        same_letter = (batch["letter_id"].unsqueeze(1) == batch["letter_id"].unsqueeze(0)).int()
        less = (batch["page_nr"].unsqueeze(1) < batch["page_nr"].unsqueeze(0)).int() * 1
        greater = (batch["page_nr"].unsqueeze(1) > batch["page_nr"].unsqueeze(0)).int() * 2
        same = (batch["page_nr"].unsqueeze(1) == batch["page_nr"].unsqueeze(0)).int() * 3

        ground_truth = (
            ((less + greater + same) * same_doc * same_letter).long().view(-1)
        )  # [ones]

        pred = torch.log_softmax(self.order_head(diff), dim=-1)

        loss = torch.nn.NLLLoss()(pred, ground_truth)

        return pred, ground_truth, loss



    def training_step(self, batch: Any, batch_idx: int):
        pred, gt, loss = self.step(batch)
        metrics = {"loss": loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch: Any, batch_idx: int):
        pred, gt, loss = self.step(batch)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: Any, batch_idx: int):
        pred, gt, loss = self.step(batch)
        metrics = {"test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        lr = 2e-5
        weight_decay = 1e-4

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return [optimizer]
