"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import math
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, default_collate
from utils.lightning import BaseLightningModule

from multipage_classifier.multipage_transformer import (
    MultipageTransformer,
    MultipageTransformerConfig,
)
from multipage_classifier.datasets.transformer_dataset import TransformerDataset


class MultipageTransformerPLModule(BaseLightningModule):
    def __init__(self, config: MultipageTransformerConfig):
        super().__init__()

        self.save_hyperparameters()

        self.config = config

        self.model = MultipageTransformer(config=self.config)

    def training_step(self, batch, *args):
        image_tensors = batch[0]
        decoder_input_ids = batch[1][:, :-1]
        decoder_labels = batch[2][:, 1:]

        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, *args):
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [
                input_id[: end_idx + 1]
                for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
            ],
            batch_first=True,
        )

        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
        )["predictions"]

        scores = list()
        for pred, answer in zip(preds, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.model.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

        return preds, answers, scores

    def validation_epoch_end(self, validation_step_outputs):
        cnt = 0
        total_metric = 0
        for _, _, scores in validation_step_outputs:
            cnt += len(scores)
            total_metric += np.sum(scores)
        val_metric = total_metric / cnt

        self.log_dict({"val_metric": val_metric}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)

        return [optimizer]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)


class MultipagePLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        ds_file,
        model: MultipageTransformer,
        split: list = [0.8, 0.2],
        num_workers: int = 1,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.ds_file = ds_file

        self.model = model

        assert (
            math.fsum(split) == 1.0
        ), f"{split} doesn't add up to 1 ({math.fsum(split)})"
        assert len(split) in [
            2,
            3,
        ], f"Length of {split} is not in [2, 3] ({len(split)})"
        self.split = split

        self.num_workers = num_workers

    def prepare_data(self):
        labels = json.load(open(self.dataset_path / self.ds_file))["labels"]

        self.classes = []
        for l in labels:
            if l["type"] not in self.classes:
                self.classes.append(l["type"])

        n = len(labels)
        sizes = [int(n * p) for p in self.split]
        slices = []
        start = 0
        for size in sizes:
            end = start + size
            slices.append(labels[start:end])
            start = end

        self.train_labels = slices[0]
        self.val_labels = slices[1]
        self.test_labels = slices[2] if len(slices) == 3 else slices[1]

    def setup(self, stage=None):
        self.train_dataset = TransformerDataset(
            self.dataset_path,
            self.train_labels,
            self.model,
            self.model.config.max_pages,
            self.model.config.max_seq_len,
            split="train",
            task_start_token="<s_test>",
            sort_json_key=False,
        )

        self.val_dataset = TransformerDataset(
            self.dataset_path,
            self.val_labels,
            self.model,
            self.model.config.max_pages,
            self.model.config.max_seq_len,
            split="val",
            task_start_token="<s_test>",
            sort_json_key=False,
        )

        self.test_dataset = TransformerDataset(
            self.dataset_path,
            self.test_labels,
            self.model,
            self.model.config.max_pages,
            self.model.config.max_seq_len,
            split="test",
            task_start_token="<s_test>",
            sort_json_key=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # collate_fn=collate,
        )
