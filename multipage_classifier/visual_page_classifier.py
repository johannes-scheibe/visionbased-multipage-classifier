import os
import re
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pydantic import BaseModel
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from transformers.models.mbart import MBartTokenizer
from transformers.models.swin import SwinModel
from transformers.models.swinv2 import Swinv2Model

from multipage_classifier.decoder.separator import DocumentSeparator, DocumentSeparatorConfig
from multipage_classifier.encoder.multipage_encoder import MultipageEncoder
from multipage_classifier.encoder.swin_encoder import SwinEncoder, SwinEncoderConfig


class VisualPageClassifierConfig(BaseModel):
    num_classes: int 
    max_pages: int = 64
    max_page_nr: int = 96 # TODO adjust in dataset
    max_seq_len: int = 768

    # Encoder params    
    encoder_cfg: SwinEncoderConfig | None = None
    pretrained_encoder: str | None = None
    detached: bool = True
    

class VisualPageClassifier(nn.Module):

    def __init__(self, config: VisualPageClassifierConfig):
        super().__init__()

        self.config = config

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
        
        assert self.config.pretrained_encoder or self.config.encoder_cfg

        if self.config.pretrained_encoder:
            page_encoder = torch.load(self.config.pretrained_encoder)
        else:
            page_encoder = SwinEncoder(
                config.encoder_cfg # type: ignore
            )

        self.encoder = MultipageEncoder(page_encoder, self.config.max_pages, self.config.detached)

        sep_config = DocumentSeparatorConfig(
            embedding_size=self.encoder.hidden_dim,
            num_classes=self.config.num_classes,
            max_pages=self.config.max_pages,
            max_page_nr=self.config.max_page_nr

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
