from typing import Tuple
import numpy as np
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn


 
class DocumentSeparatorConfig(BaseModel):
    batch_size: int = 8
    max_pages: int = 64
    doc_nr_group_threshold: float = 0.5

    embedding_size: int 


class DocumentSeparator(nn.Module):

    def __init__(
        self, 
        config: DocumentSeparatorConfig,
    ):
        super().__init__()
        self.config = config

        self.doc_id = nn.Sequential(
            nn.Linear(self.config.embedding_size, int(self.config.embedding_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.config.embedding_size/2), self.config.max_pages),

        )
        self.doc_id = nn.Sequential(
            nn.Linear(self.config.embedding_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.max_pages)
        )

        self.cluster_prediction = DBSCAN(
            eps=self.config.doc_nr_group_threshold, min_samples=1, metric="precomputed"
        )
        
    def forward(self, embeddings: torch.Tensor):
        pred = self.doc_id(embeddings).sigmoid()
        return pred
    
    def get_groups(self, probability_matrix) -> np.ndarray:
        cost = 1 - ((probability_matrix + probability_matrix.transpose(1, 0)) / (2))
        return self.cluster_prediction.fit_predict(cost)   
    
    def postprocess(self, prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = prediction[:, : len(prediction)]
        doc_id = self.get_groups(prediction.cpu().data.numpy())
        doc_id_prob = (prediction - 0.5).abs().mean() * 2

        return torch.tensor(doc_id, device=prediction.device), doc_id_prob