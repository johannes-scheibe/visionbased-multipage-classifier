from typing import cast
from pydantic import BaseModel
import torch
import torch.nn as nn

from multipage_classifier.encoder.swin_encoder import SwinEncoder

class EncoderOutput(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        
    last_hidden_state: torch.Tensor


class MultipageEncoder(nn.Module):
    def __init__(
        self,
        encoder: SwinEncoder,  # TODO support general encoders
        max_pages: int,
    ):
        super().__init__()

        self.page_encoder = encoder

        self.hidden_dim: int = self.page_encoder.hidden_dim

        self.pos_embedding_layer = nn.Embedding(
            max_pages, embedding_dim=self.hidden_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=8, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:

        embeddings = []
        for px in pixel_values: # iterate over the batch and compute embs for each stack in batch
            page_embeddings = self.page_encoder.forward(px)

            pos_embeddings = self.pos_embedding_layer(
                torch.arange(0, len(page_embeddings), device=page_embeddings.device)
            )
            document_embeddings = self.transformer_encoder(
                page_embeddings.unsqueeze(0) + pos_embeddings.unsqueeze(0)
            ).view(-1, self.hidden_dim).unsqueeze(0)

            embeddings.append(document_embeddings)

        return EncoderOutput(
            last_hidden_state=torch.cat(embeddings),
        )
