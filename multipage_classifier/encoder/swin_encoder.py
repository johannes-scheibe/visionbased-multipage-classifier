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

        self.cfg = cfg
        
        config = AutoConfig.from_pretrained(self.cfg.pretrained_model_name_or_path)
        config.image_size = cfg.image_size
        self.model = cast(self.cfg.pretrained_model_type, self.cfg.pretrained_model_type.from_pretrained(cfg.pretrained_model_name_or_path, config=config))
        
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

    def prepare_input(self, img: Image.Image, random_padding: bool = False, align_long_axis = False) -> torch.Tensor:
        img = img.convert("RGB")
        if align_long_axis and (
            (self.model.config.image_size[1] > self.model.config.image_size[0] and img.width < img.height)
            or (self.model.config.image_size[1] < self.model.config.image_size[0] and img.width > img.height)
        ):
            img = rotate(img, angle=-90, expand=True) # type: ignore
        img = resize(img, min(self.model.config.image_size)) # type: ignore
        img.thumbnail((self.model.config.image_size[1], self.model.config.image_size[0]))
        delta_width = self.model.config.image_size[1] - img.width
        delta_height = self.model.config.image_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

        pixel_values = transforms.ToTensor()(ImageOps.expand(img, padding))

        return pixel_values
