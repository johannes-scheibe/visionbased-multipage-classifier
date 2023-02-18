import json
import math
import os

from pathlib import Path
import random
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from PIL import Image

class UCSFDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        labels: List[dict],
        classes: List[str],
        prepare_function
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.labels = labels
        
        self.prepare_function = prepare_function

        self.classes = classes
        self.id2class = {idx: str(label) for idx, label in enumerate(self.classes)}
        self.class2id = {str(label): idx for idx, label in enumerate(self.classes)}
        
        self.dataset_length = len(self.labels)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx: int):
        
        label = self.labels[idx]
        img_folder = label["image_folder"]

        pages = []
        for i, file in enumerate(os.listdir(self.dataset_path / img_folder)):
            path = self.dataset_path / img_folder / file
            pages.append(
                {
                    "pixel_values": self.prepare_function(Image.open(path)),
                    #"letter_id": label["id"], # we can set this to the doc_id because its unique as well 
                    "doc_class":  self.class2id[label["type"]],
                    #"doc_id": label["id"], # set this in collate
                    "page_nr": i
                }
            )
        return pages



class UCSFDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, prepare_function, split: list = [0.8, 0.2], batch_size=8, num_workers: int = 1):
        super().__init__()
        self.dataset_path = dataset_path
        self.prepare_function = prepare_function

        assert math.fsum(split) == 1.0, f"{split} doesn't add up to 1 ({math.fsum(split)})"
        assert len(split) in [2, 3], f"Length of {split} is not in [2, 3] ({len(split)})"

        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        labels = json.load(open(self.dataset_path / "labels.json"))

        random.shuffle(labels)

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
        self.train_dataset = UCSFDataset(
            self.dataset_path, self.train_labels, self.classes, self.prepare_function
        )
        self.val_dataset = UCSFDataset(
            self.dataset_path, self.val_labels, self.classes, self.prepare_function
        )
        self.test_dataset = UCSFDataset(
            self.dataset_path, self.test_labels, self.classes, self.prepare_function
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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


def collate(samples: List[List[dict]], shuffle_mode="window"): # "all", "none"
    prep_samples: List[dict] = []
    for i, sample in enumerate(samples):  
        for page in sample: 
            page["letter_id"] = 0 # always same letter
            page["doc_id"] = i
        if shuffle_mode == "all":
            random.shuffle(sample)
        elif shuffle_mode == "window":
            window_shuffle(sample)
        prep_samples.extend(sample)

    ret = default_collate(prep_samples)
    return ret

def val_collate(samples: List[List[dict]]) -> Dict:
    prep_samples: List[dict] = []
    for i, sample in enumerate(samples): 
        for page in sample: 
            page["letter_id"] = 0 # always same letter
            page["doc_id"] = i
        prep_samples.extend(sample)
    batch = [sample for sample in prep_samples]

    # Using the whole batch could use too much memory. Monitor this.
    ret = default_collate(batch)

    return ret

def window_shuffle(input_list, window_size=3, shuffle_percent=0.25):
    shuffle_idxs = random.sample(
        range(0, len(input_list) - window_size, 1), k=int(shuffle_percent * len(input_list))
    )  # Possibly shuffles 25% of the input sequence with their 'window_size' neighbors
    for i in shuffle_idxs:
        random.shuffle(input_list[i : i + window_size])

    return input_list