from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
from torch import nn
from transformers import MBartConfig, MBartForCausalLM


class MBartDecoder(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.model = MBartForCausalLM(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
