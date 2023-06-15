import os
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pydantic import BaseModel
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from transformers import MBartConfig, MBartForCausalLM, MBartTokenizer

from multipage_classifier.decoder.separator import DocumentSeparator, DocumentSeparatorConfig
from multipage_classifier.encoder.multipage_encoder import MultipageEncoder
from multipage_classifier.encoder.swin_encoder import SwinEncoder, SwinEncoderConfig


class MultipageClassifierConfig(BaseModel):
    input_size: List[int] = [2560, 1920]
    num_classes: int 
    max_pages: int = 64
    max_seq_len: int = 768

    # Encoder params    
    encoder_cfg: BaseModel

    # Decoder params
    

class MultipageClassifier(nn.Module):

    def __init__(self, config: MultipageClassifierConfig):
        super().__init__()

        self.config = config

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")

        page_encoder = SwinEncoder(
            **self.config.encoder_cfg.dict()
        )
        self.encoder = MultipageEncoder(page_encoder, self.config.max_pages)

        sep_config = DocumentSeparatorConfig(
            embedding_size=self.encoder.hidden_dim,
            num_classes=self.config.num_classes,
            max_pages=self.config.max_pages
        )
        self.separator = DocumentSeparator(
            sep_config
        )


    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(pixel_values) # TODO do this for every item in Batch
        preds = self.separator(
            encoder_outputs
        )
        return preds

    def predict(self, pixel_values: torch.Tensor):
        preds = self.forward(pixel_values)
        preds = self.separator.postprocess(preds)
        return preds

    def prepare_input(self, img: Image.Image, random_padding: bool = False, align_long_axis = False) -> torch.Tensor:
        img = img.convert("RGB")
        if align_long_axis and (
            (self.config.input_size[0] > self.config.input_size[1] and img.height > img.width)
            or (self.config.input_size[0] < self.config.input_size[1] and img.height < img.width)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.config.input_size))
        img.thumbnail((self.config.input_size[0], self.config.input_size[1]))
        delta_width = self.config.input_size[0] - img.width
        delta_height = self.config.input_size[1] - img.height
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

        pixel_values = torch.Tensor(self.to_tensor(ImageOps.expand(img, padding)))

        return pixel_values
    