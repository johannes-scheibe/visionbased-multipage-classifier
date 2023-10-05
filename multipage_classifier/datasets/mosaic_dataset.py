from functools import partial
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from multipage_classifier.datasets.utils import Bucket

import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, default_collate
from PIL import Image


class MosaicDataset(Dataset):
    sample_info_file_name: str = "sample.json"

    def __init__(
        self,
        path: Path,
        bucket: Bucket,
        classes: list,
        max_pages: int,
        prepare_function,
    ):
        super().__init__()

        self.path = path
        self.bucket = bucket

        self.prepare_function = prepare_function

        with (self.path / f"{bucket.value.lower()}.txt").open("r") as file:
            self.inventory = [Path(line.rstrip()) for line in file.readlines()]

        self.classes = classes
        self.id2class = {idx: str(label) for idx, label in enumerate(classes)}
        self.class2id = {str(label): idx for idx, label in enumerate(classes)}

        self.max_pages = max_pages

    def __len__(self):
        return len(self.inventory)

    def __getitem__(self, idx: int):

        sample_path = self.path / self.inventory[idx]

        ground_truth = json.load(open(sample_path / "ground_truth.json"))

        offset = random.randint(0, max(0, (len(ground_truth) - self.max_pages)))
        batch = ground_truth[offset : offset + self.max_pages]

        doc_id_offset = ground_truth[0]["doc_id"]
        for sample in batch:
            sample["letter_id"] = idx # TODO maybe use str2int mapping
            sample["doc_id"] -= doc_id_offset  # doc_ids should start at 0
            sample["doc_class"] = self.class2id[sample["doc_class"]]        
            src_page = sample["src_page"]
            img = Image.open(sample_path / f"page_{src_page}.png")
            sample["pixel_values"] = self.prepare_function(img)

        return batch


class MosaicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: Path,
        classes,
        prepare_function,
        batch_size=1,
        num_workers: int = 0,
        max_pages: int = 64,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.classes = classes
        self.num_workers = num_workers
        self.max_pages = max_pages

        self.prepare_function = prepare_function

    def setup(self, stage=None):
        self.train_dataset = MosaicDataset(
            self.path,
            Bucket.Training,
            self.classes,
            self.max_pages,
            self.prepare_function,
        )
        self.val_dataset = MosaicDataset(
            self.path,
            Bucket.Validation,
            self.classes,
            self.max_pages,
            self.prepare_function,
        )
        self.test_dataset = MosaicDataset(
            self.path,
            Bucket.Testing,
            self.classes,
            self.max_pages,
            self.prepare_function,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=val_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=val_collate,
        )


def window_shuffle(input_list, window_size=3, shuffle_percent=0.25):
    shuffle_idxs = random.sample(
        range(0, len(input_list), 1), k=int(shuffle_percent * len(input_list))
    )  # Possibly shuffles 25% of the input sequence with their 'window_size' neighbors
    for i in shuffle_idxs:
        sub = input_list[i : i + window_size]
        random.shuffle(sub)
        input_list[i : i + window_size] = sub

    return input_list


def collate(
    samples: list[list[dict[str, Any]]], shuffle_mode="window"
):  # "all", "none"
    assert len(samples) == 1

    sample = samples[0]

    # TODO schuffle
    # if shuffle_mode == "all":
    #     random.shuffle(doc)
    # elif shuffle_mode == "window":
    #     sample = window_shuffle(doc)

    batch = default_collate(sample)

    return batch


def val_collate(samples: list[list[dict[str, Any]]]):
    assert len(samples) == 1

    sample = samples[0]

    batch = default_collate(sample)

    return batch
