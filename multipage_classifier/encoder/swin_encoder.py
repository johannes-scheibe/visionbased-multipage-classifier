from typing import Type, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pydantic import BaseModel
from torchvision import transforms

from torchvision.transforms.functional import resize, rotate
from transformers import AutoConfig, DonutSwinModel, Swinv2Model, SwinModel

class SwinEncoderConfig(BaseModel):
    image_size: tuple[int, int]
    pretrained_model_name_or_path: str
    pretrained_model_type: Type[SwinModel] | Type[Swinv2Model] | Type[DonutSwinModel] = SwinModel

    # TODO custom params if no path is specified

class SwinEncoder(nn.Module):
    """
    Wrapper for the transformers SwinModel.
    """
    
    def __init__(
        self,
        cfg: SwinEncoderConfig
    ):
        super().__init__()
        
        config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path)
        config.image_size = cfg.image_size
        self.model = cast(cfg.pretrained_model_type, cfg.pretrained_model_type.from_pretrained(cfg.pretrained_model_name_or_path, config=config))
        
        self.hidden_dim = self.model.num_features
        
    def forward(self, pixel_values: torch.Tensor, return_pooled_output: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        out = self.model(pixel_values)
        if return_pooled_output:
            return out.pooler_output
        return out.last_hidden_state

    def prepare_input(self, img: Image.Image, align_long_axis = True) -> torch.Tensor:
        img = img.convert("RGB")

        w, h = img.size
        if align_long_axis and (w > h) != (self.model.config.image_size[1] > self.model.config.image_size[0]):
            img = img.rotate(90, expand=True)

        img.thumbnail((self.model.config.image_size[1], self.model.config.image_size[0]))
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