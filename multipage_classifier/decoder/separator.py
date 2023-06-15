from typing import Tuple
import numpy as np
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn


 
class DocumentSeparatorConfig(BaseModel):
    num_classes: int
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

        self.heads = nn.ModuleDict({})
        self.heads["doc_class"] = self.get_head([self.config.embedding_size, self.config.num_classes])
        self.heads["page_nr"] = self.get_head([self.config.embedding_size, self.config.max_pages])
        self.heads["doc_id"] = self.get_head([self.config.embedding_size, self.config.max_pages])

        self.cluster_prediction = DBSCAN(
            eps=self.config.doc_nr_group_threshold, min_samples=1, metric="precomputed"
        )

    def get_head(self, dims) -> nn.Sequential:
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers[:-1])
         
    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        preds: dict[str, torch.Tensor] = {}
        for k, v in self.heads.items():
            if k == "doc_id":
                continue
            preds[k] = torch.log_softmax(v(embeddings), dim=-1)
        preds["doc_id"] = self.heads["doc_id"](embeddings).sigmoid()

        return preds
    
    def get_groups(self, probability_matrix) -> np.ndarray:
        cost = 1 - ((probability_matrix + probability_matrix.transpose(1, 0)) / (2))
        return self.cluster_prediction.fit_predict(cost)   
    
    def postprocess(self, preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        out.update(preds)

        preds["doc_id"] = preds["doc_id"][:, : len(preds["doc_id"])]
        out["doc_id"] = torch.tensor(
            self.get_groups(preds["doc_id"].cpu().data.numpy()), device=preds["doc_id"].device
        )
        out["doc_id_prob"] = (preds["doc_id"] - 0.5).abs().mean() * 2

        return out