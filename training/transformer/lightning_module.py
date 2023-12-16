from collections import defaultdict
from functools import partial
import math
import re
from pathlib import Path
from typing import Any, List

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

        assert len(image_tensors) == 1 and len(decoder_input_ids) == 1, "batch size > 1 not supported"

        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, *args):
        image_tensors = batch["pixel_values"]
        decoder_input_ids = batch["decoder_input_ids"]
        prompt_end_idxs = batch["prompt_end_index"]
        answers = batch["target_sequence"]
        ground_truths = batch["ground_truth"]
        
        decoder_prompts = pad_sequence(
            [
                input_id[: end_idx + 1]
                for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
            ],
            batch_first=True,
        )

        assert len(image_tensors) == 1 and len(decoder_prompts) == 1, "batch size > 1 not supported"

        preds = self.model.inference(
            image_tensors=image_tensors[0],
            prompt_tensors=decoder_prompts[0],
            return_json=False,
        )["predictions"]

        def check_pred(pred: Any, dst_len) -> list[dict[str, str]]:
            if type(pred) == list:
                res = [element if isinstance(element, dict) else {} for element in pred]
                res.extend([{}]*(dst_len-len(pred)))
                return res
            return [{}]*dst_len

        out = defaultdict(list)
        for pred, answer, gt in zip(preds, answers, ground_truths):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
            
            out["scores"].append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            page_preds = check_pred(self.model.token2json(pred), len(gt)) 
            for truth, p in zip(gt, page_preds):
                out["acc_doc_id"].append(truth.get("doc_id", None) == p.get("doc_id", None))
                out["acc_doc_class"].append(truth.get("doc_class", None) == p.get("doc_class", None))
                out["acc_page_nr"].append(truth.get("pag_nr", None) == p.get("page_nr", None))
            
        self.validation_step_outputs.append(out)

        return preds, answers, out["scores"]

    def on_validation_epoch_end(self):
        metric = {}
        flattend = defaultdict(list)
        for dct in self.validation_step_outputs:
            for key, value in dct.items():
                flattend[key].extend(value)

        metric["val/edit_distance"] = sum(flattend["scores"]) / len(flattend["scores"])
        metric["val/acc_doc_id"] = sum(flattend["acc_doc_id"]) / len(flattend["acc_doc_id"])
        metric["val/acc_doc_class"] = sum(flattend["acc_doc_class"]) / len(flattend["acc_doc_class"])
        metric["val/acc_page_nr"] = sum(flattend["acc_page_nr"]) / len(flattend["acc_page_nr"])


        self.log_dict(metric, sync_dist=True)

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
    batch = []
    gt = []
    for s in samples:
        data = s.dict()
        gt.append(data.pop("ground_truth")) # dont collate ground_tuth
        batch.append(data)

    ret = default_collate(batch)
    assert "ground_truth" not in ret
    ret["ground_truth"] = gt
    return ret
