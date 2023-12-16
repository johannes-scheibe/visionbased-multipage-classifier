from collections import defaultdict, deque
import os
import re
from typing import Any, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pydantic import BaseModel
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from transformers.models.mbart import MBartTokenizer

from transformers import AutoConfig, DonutSwinModel, Swinv2Model, SwinModel

from multipage_classifier.decoder.separator import DocumentSeparator, DocumentSeparatorConfig
from multipage_classifier.encoder.multipage_encoder import MultipageEncoder
from multipage_classifier.encoder.swin_encoder import SwinEncoder, SwinEncoderConfig
from sklearn.cluster import DBSCAN

ORDER_NAMES = ["None", "Pred", "Succ", "Same"]

class DualClassifierConfig(BaseModel):
    num_classes: int 
    max_page_nr: int 
    encoder_cfg: SwinEncoderConfig
    

class DualClassifier(nn.Module):

    def __init__(self, config: DualClassifierConfig):
        super().__init__()

        self.config = config

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.encoder = SwinEncoder(
            config.encoder_cfg
        )

        # self.order_head = torch.nn.Sequential(
        #     torch.nn.Linear(self.encoder.hidden_dim * 2,  len(ORDER_NAMES)),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear( len(ORDER_NAMES), len(ORDER_NAMES))
        # )

        # self.classification_head = torch.nn.Sequential(
        #     torch.nn.Linear(self.encoder.hidden_dim, self.config.num_classes),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.config.num_classes, self.config.num_classes)
        # )

        self.order_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.hidden_dim * 2, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, len(ORDER_NAMES))
        )
        
        
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.hidden_dim, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, self.config.num_classes)
        )
        

        
        self.cluster_prediction = DBSCAN(
            min_samples=1, metric="precomputed"
        )


    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_outputs: torch.Tensor = self.encoder(pixel_values)
        
        preds = {} 

        preds["doc_class"] = torch.log_softmax(self.classification_head(encoder_outputs), dim=-1)
        
        bs = len(encoder_outputs)  
        diff = encoder_outputs.unsqueeze(0).repeat(bs, 1, 1)
        diff = torch.cat([diff, diff.permute(1, 0, 2)], -1).view(-1, self.encoder.hidden_dim * 2)

        preds["order"] = torch.log_softmax(self.order_head(diff), dim=-1)

        return preds

    def predict(self, pixel_values: torch.Tensor):
        preds = self.forward(pixel_values)
        preds = self.postprocess(preds)
        return preds

    def postprocess(self, preds) -> dict[str, torch.Tensor]:
        bs =len(preds["doc_class"])

        order_mat = torch.exp(preds["order"]).view(-1, bs, 4)

        preds["doc_id"] = torch.tensor(self.compute_doc_ids(order_mat), device=order_mat.device)

        # iterate over subdocs
        page_nrs = []
        for doc_id in torch.unique(preds["doc_id"]):
            indices = torch.nonzero(preds["doc_id"] == doc_id).reshape(-1)
            sub_order_mat = order_mat[indices][:,indices]   
            sub_page_nrs = self.compute_page_nrs(sub_order_mat)
            page_nrs.extend(sub_page_nrs)

        preds["page_nr"] = torch.tensor(page_nrs, device=order_mat.device)

        return preds

    def compute_doc_ids(self, order_matrix: torch.Tensor) -> list[int]:
        # dim 0 = order: "None". NOTE already inverted 1 -> no relationship 
        doc_id_probs: torch.Tensor = order_matrix[:, :, 0] 

        # make symetric
        doc_id_probs = (doc_id_probs + doc_id_probs.transpose(1, 0)) / (2)

        doc_ids = self.cluster_prediction.fit_predict(doc_id_probs.cpu().data.numpy())
        
        return doc_ids.tolist()
    
    
    def compute_page_nrs(self, order_matrix: torch.Tensor) -> list[int]:
        # extract "pred" and "succ" prediction. NOTE this changes the indices -> ["Pred", "Succ"]
        page_nr_probs = order_matrix[:, :, 1:3] 
 
        # Make "page order symetric": 0 maps to 1 and the other way arround
        bs = len(page_nr_probs)
        for i in range(bs):
            for j in range(i, bs):
                avg = (page_nr_probs[i, j] + page_nr_probs[j, i].flip(0)) / 2
                page_nr_probs[i, j] = avg
                page_nr_probs[j, i] = avg.flip(0)

        page_nr_preds = page_nr_probs.argmax(2)

        # Create a graph representation using a dictionary
        graph = defaultdict(list)
        for i in range(bs):
            for j in range(i, bs):
                if i != j:
                    if page_nr_preds[i, j] == 0:  # i is predecessor
                        graph[i].append(j)
                    elif page_nr_preds[i, j] == 1:  # i is successor
                        graph[j].append(i)

        def clean_graph(graph, visited: list[int], successors: list[int]):
            # Remove circles for a specific node. Because the graph contains all successors for a node we can just remove occurences in the successors
            for succ in successors:
                for node in visited:
                    if node in graph[succ]: 
                        graph[succ].remove(node)
                    
                graph = clean_graph(graph, visited+[succ], graph[succ])
            return graph

        # TODO clean based on probs and distance between pages
        for node in range(bs):
            graph = clean_graph(graph, [node], graph[node])
        
        # Perform topological sorting using an adapted Kahn's algorithm 

        # Count the number of incoming edges
        in_degree = [0] * bs
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                in_degree[neighbor] += 1

        # List of "start nodes" which have no incoming edges
        queue = deque([node for node, degree in enumerate(in_degree) if degree == 0])
        topological_order: list[int] = []
        while queue:
            node = queue.popleft()

            topological_order.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        assert len(topological_order) == len(graph)

        return topological_order