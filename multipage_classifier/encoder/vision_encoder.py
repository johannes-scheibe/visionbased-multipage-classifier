from typing import Type, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pydantic import BaseModel
from torchvision import transforms

from torchvision.transforms.functional import resize, rotate
from transformers import AutoConfig, DonutSwinModel, Swinv2Model, SwinModel
from transformers import ViTModel, ViTConfig


class VisionEncoderConfig(BaseModel):
    image_size: tuple[int, int]
    pretrained_model_name_or_path: str
    dropout: float | None = 0.2
    # TODO custom params if no path is specified


class VisionEncoder(nn.Module):
    """
    Wrapper for the transformers SwinModel.
    """

    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.config = cfg

        config: ViTConfig = AutoConfig.from_pretrained(
            cfg.pretrained_model_name_or_path
        )
        config.image_size = cfg.image_size
        model = ViTModel.from_pretrained(
            cfg.pretrained_model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
        assert isinstance(model, ViTModel)
        self.model: ViTModel = model

        self.hidden_dim = config.hidden_size

        self.dropout = (
            torch.nn.Dropout(self.config.dropout)
            if self.config.dropout is not None
            else torch.nn.Identity()
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self, pixel_values: torch.Tensor, return_pooled_output: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        out = self.model(pixel_values)

        embs: torch.Tensor = out.last_hidden_state

        embs = embs.permute((0, 2, 1))

        embs = self.avgpool(embs)

        embs = embs.flatten(1)

        # embs = out.pooler_output

        embs = self.dropout(embs)

        return embs

    def prepare_input(self, img: Image.Image, align_long_axis=True) -> torch.Tensor:
        img = img.convert("RGB")

        w, h = img.size
        if align_long_axis and (w > h) != (
            self.model.config.image_size[1] > self.model.config.image_size[0]
        ):
            img = img.rotate(90, expand=True)

        img.thumbnail(
            (self.model.config.image_size[1], self.model.config.image_size[0])
        )
        delta_width = self.model.config.image_size[1] - img.width
        delta_height = self.model.config.image_size[0] - img.height

        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

        img = ImageOps.expand(img, padding, fill="white")

        pixel_values = transforms.ToTensor()(img)

        return pixel_values
