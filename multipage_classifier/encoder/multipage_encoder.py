import torch
import torch.nn as nn

from multipage_classifier.encoder.swin_encoder import SwinEncoder

class MultipageEncoder(nn.Module):

    def __init__(
        self,
        encoder: SwinEncoder, # TODO support general encoders
        max_pages: int = 64,
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

        

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values.size()
        page_embeddings = self.page_encoder(pixel_values) # TODO calc batchwise
        pos_embedding = self.pos_embedding_layer(
            torch.arange(0, len(page_embeddings), device=page_embeddings.device)
        )

        document_embeddings = self.transformer_encoder(page_embeddings.unsqueeze(0) + pos_embedding.unsqueeze(0))

        return document_embeddings.squeeze(0)