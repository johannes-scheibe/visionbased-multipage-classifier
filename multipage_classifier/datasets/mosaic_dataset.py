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

        with (self.path / f"{bucket.value}.txt").open("r") as file:
            self.inventory = [Path(line.rstrip()) for line in file.readlines()]

        self.classes = classes
        self.id2class = {idx: str(label) for idx, label in enumerate(classes)}
        self.class2id = {str(label): idx for idx, label in enumerate(classes)}

        self.max_pages = max_pages

    def __len__(self):
        return len(self.inventory)

    def __getitem__(self, idx: int):
        letter_id = idx

        sample_path = self.path / self.inventory[idx]
        sample_data = {
            path.name: path.read_bytes()
            for path in sample_path.iterdir()
            if path.is_file() and path.name != self.sample_info_file_name
        }

        document = json.loads(sample_data["document.json"].decode())

        best_candidate = max(
            document["prediction"]["candidates"], key=lambda c: c["score"]
        )
        assert len(best_candidate["documents"]) > 0 and len(document["pages"]) > 0

        batch = []

        for doc_id, predicted_doc in enumerate(best_candidate["documents"]):
            class_identifier = str(
                Path(predicted_doc["documentClass"]).relative_to(
                    document["documentClass"]
                )
            )
            if class_identifier not in self.classes:
                raise ValueError(
                    f"Prediction contains invalid class identifier: {class_identifier}"
                )

            pages = predicted_doc["pages"]
            if len(pages) == 0:
                pages = [{"sourcePage": i} for i in range(len(document["pages"]))]

            doc = []
            for dst_page, page in enumerate(pages[: self.max_pages]):
                src_page = page.get("sourcePage", 0)  # NOTE the default value is 0
                page_bytes = sample_data[f"page_{src_page}.png"]
                img = Image.open(io.BytesIO(page_bytes))

                page = {
                    "pixel_values": self.prepare_function(img),
                    "letter_id": letter_id,
                    "doc_class": self.class2id[class_identifier],
                    "doc_id": doc_id,
                    "page_nr": dst_page,
                }
                doc.append(page)
            batch.append(doc)

        return batch[: self.max_pages]


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
            collate_fn=partial(collate, batch_size=self.max_pages),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(val_collate, batch_size=self.max_pages),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(val_collate, batch_size=self.max_pages),
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
    samples: List[List[List[dict]]], batch_size: int, shuffle_mode="window"
):  # "all", "none"
    batch: List[dict] = []
    i = 0
    for sample in samples:
        for doc in sample:
            for p in doc:
                p["doc_id"] = i
            if shuffle_mode == "all":
                random.shuffle(doc)
            elif shuffle_mode == "window":
                sample = window_shuffle(doc)
            batch.extend(doc)
            i += 1

    ret = default_collate(batch[:batch_size])

    return ret


def val_collate(samples: List[List[List[dict]]], batch_size: int) -> Dict:
    batch: List[dict] = []
    i = 0
    for sample in samples:
        for doc in sample:
            for p in doc:
                p["doc_id"] = i

            batch.extend(doc)
            i += 1

    ret = default_collate(batch[:batch_size])

    return ret
