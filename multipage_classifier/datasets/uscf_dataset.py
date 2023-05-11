import json
import math
import os

from pathlib import Path
import random
from typing import Dict, List

import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, default_collate

from PIL import Image

class UCSFDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        labels: List[dict],
        classes: List[str],
        prepare_function,
        max_pages: int
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.labels = labels
        
        self.prepare_function = prepare_function

        self.classes = classes
        self.id2class = {idx: str(label) for idx, label in enumerate(self.classes)}
        self.class2id = {str(label): idx for idx, label in enumerate(self.classes)}

        self.max_pages = max_pages

        total_num_pages = sum(item['pages'] for item in self.labels)
        self.dataset_length = int(total_num_pages / self.max_pages)

    def __len__(self):
        return self.dataset_length


    def __getitem__(self, idx: int):
        stack = []
        num_pages = 0

        doc_id = 0

        while True:

            i = random.randint(0, len(self.labels) - 1)

            label = self.labels[i]
            
            if num_pages + label["pages"] > self.max_pages:
                if len(stack) == 0:
                    continue
                break
                
            stack.append(self._prepare_doc(i, doc_id=doc_id))
            
            num_pages = num_pages + label["pages"]
            doc_id += 1
            

        return stack
    
    def _prepare_doc(self, page_idx, doc_id):
        label = self.labels[page_idx]

        img_folder = label["image_folder"]

        doc_pages = []

        for page_idx in range(label["pages"]):
            file = f"page_{page_idx}.jpg"
            path = self.dataset_path / img_folder / file

            doc_pages.append(
                {
                    "pixel_values": self.prepare_function(Image.open(path)),
                    "letter_id": 0,  # always same letter
                    "doc_id": doc_id,
                    "doc_class":  self.class2id[label["type"]],
                    "page_nr": page_idx
                }
            )

        return doc_pages


class UCSFDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, ds_file,  prepare_function, split: list = [0.8, 0.2], max_pages=64, num_workers: int = 1):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.ds_file = ds_file
        self.prepare_function = prepare_function

        assert math.fsum(split) == 1.0, f"{split} doesn't add up to 1 ({math.fsum(split)})"
        assert len(split) in [2, 3], f"Length of {split} is not in [2, 3] ({len(split)})"

        self.split = split
        self.max_pages = max_pages
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
        self.train_dataset = UCSFDataset(
            self.dataset_path, self.train_labels, self.classes, self.prepare_function, self.max_pages
        )
        self.val_dataset = UCSFDataset(
            self.dataset_path, self.val_labels, self.classes, self.prepare_function, self.max_pages
        )
        self.test_dataset = UCSFDataset(
            self.dataset_path, self.test_labels, self.classes, self.prepare_function, self.max_pages
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=val_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=val_collate,
        )
    

def collate(data: List[List[List[dict]]], shuffle_mode="window"):  # "all", "none"
    samples = data[0]
    prep_samples: List[dict] = []
    for i, sample in enumerate(samples):           
        if shuffle_mode == "all":
            random.shuffle(sample)
        elif shuffle_mode == "window":
            sample = window_shuffle(sample)
        prep_samples.extend(sample)

    ret = default_collate(prep_samples)
    return ret


def val_collate(data: List[List[List[dict]]]) -> Dict:
    samples = data[0]
    batch: List[dict] = []
    for i, sample in enumerate(samples):
        batch.extend(sample)

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