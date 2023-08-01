from functools import partial
import math
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
from multipage_classifier.datasets.transformer_dataset import TransformerSample, TransformerDataset
from multipage_classifier.datasets.utils import Bucket


class MultipageTransformerPLModule(BaseLightningModule):
    def __init__(self, config: MultipageTransformerConfig):
        super().__init__()

        self.save_hyperparameters()

        self.config = config

        self.model = MultipageTransformer(config=self.config)

        self.validation_step_outputs = []

    def training_step(self, batch, *args):
        image_tensors = batch["pixel_values"]
        decoder_input_ids = batch["decoder_input_ids"][:, :-1]
        decoder_labels = batch["decoder_labels"][:, 1:]

        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, *args):
        image_tensors = batch["pixel_values"]
        decoder_input_ids = batch["decoder_input_ids"]
        prompt_end_idxs = batch["prompt_end_index"]
        answers = batch["target_sequence"]

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
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

        self.validation_step_outputs.append(scores)

        return preds, answers, scores

    def on_validation_epoch_end(self):
        cnt = sum(len(scores) for scores in self.validation_step_outputs)
        val_metric = sum(np.sum(scores) for scores in self.validation_step_outputs)

        self.log_dict({"val_metric": val_metric / cnt}, sync_dist=True)

        self.validation_step_outputs.clear()  # free memory

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
        self, dataset_path: Path, model: MultipageTransformer, task_prompt: str, num_workers: int = 0
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.model = model
        self.num_workers = num_workers
        self.task_prompt = task_prompt
        
    def setup(self, stage=None):
        self.train_dataset = TransformerDataset(
            self.dataset_path,
            Bucket.Training,
            self.model,
            split="train",
            task_start_token=self.task_prompt,
            sort_json_key=False,
        )

        self.val_dataset = TransformerDataset(
            self.dataset_path,
            Bucket.Validation,
            self.model,
            split="train",
            task_start_token=self.task_prompt,
            sort_json_key=False,
        )

        self.test_dataset = TransformerDataset(
            self.dataset_path,
            Bucket.Testing,
            self.model,
            split="train",
            task_start_token=self.task_prompt,
            sort_json_key=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate
        )


def collate(samples: List[TransformerSample]):
    # offset = random.randint(0, max(0, (len(batch) - max_pages_per_batch))) NOTE: cant apply offset because this would require changes of the doc_ids for example

    batch = [s.dict() for s in samples]

    ret = default_collate(batch)

    return ret
