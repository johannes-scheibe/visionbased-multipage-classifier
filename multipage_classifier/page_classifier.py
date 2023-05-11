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
from multipage_classifier.encoder.swin_encoder import SwinEncoder


class MultipageClassifierConfig(BaseModel):
    input_size: List[int] = [2560, 1920]
    max_pages: int = 64
    max_seq_len: int = 768

    # Encoder params    
    patch_size: List[int] = [8,8]
    embed_dim = 96
    depths: List[int] = [2, 2, 10, 2]
    num_heads: List[int] = [4, 8, 16, 32]
    window_size: List[int] = [7,7]

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
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            depths=self.config.depths,
            num_heads=self.config.num_heads,
            window_size=self.config.window_size,
        )
        self.encoder = MultipageEncoder(page_encoder)

        sep_config = DocumentSeparatorConfig(
            embedding_size=self.encoder.hidden_dim,
            max_pages=self.config.max_pages
        )
        self.separator = DocumentSeparator(
            sep_config
        )


    def forward(self, pixel_values: torch.Tensor):
        encoder_outputs = self.encoder(pixel_values) # TODO do this for every item in Batch
        
        pred = self.separator(
            encoder_outputs
        )
        return pred

    def predict(self, pixel_values: torch.Tensor):
        pred = self.forward(pixel_values)
        pred = self.separator.postprocess(pred)
        return pred

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
    