import io
import json
from pathlib import Path
import random
from typing import Any, List, Tuple

from pydantic import BaseModel

from multipage_classifier.datasets.utils import Bucket

from multipage_classifier.multipage_transformer import MultipageTransformer
import torch

from PIL import Image
from torch.utils.data import Dataset


class TransformerSample(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    pixel_values: torch.Tensor
    decoder_input_ids: torch.Tensor
    prompt_end_index: torch.Tensor
    decoder_labels: torch.Tensor
    target_sequence: str
    ground_truth: list[dict]


class TransformerDataset(Dataset):
    sample_info_file_name: str = "sample.json"

    def __init__(
        self,
        path: Path,
        bucket: Bucket,
        model: MultipageTransformer,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s_cls>",
        prompt_end_token: str | None = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.path = path
        self.bucket = bucket

        with (self.path / f"{bucket.value.lower()}.txt").open("r") as file:
            self.inventory = [Path(line.rstrip()) for line in file.readlines()]

        self.model = model

        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key

        self.prompt_end_token_id = model.decoder.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

    def __len__(self):
        return len(self.inventory)

    def __getitem__(self, idx: int) -> TransformerSample:
        sample_path = self.path / self.inventory[idx]

        sample = json.load(open(sample_path / "ground_truth.json"))

        offset = random.randint(0, max(0, (len(sample) - self.model.config.max_pages)))
        batch = sample[offset : offset + self.model.config.max_pages]

        page_tensors: list[torch.Tensor] = []
        ground_truth: list[dict[str, str]] = []

        doc_id_offset = sample[0]["doc_id"]
        for sample in batch:
            src_page = sample["src_page"]
            img = Image.open(sample_path / f"page_{src_page}.png")

            page_tensors.append(
                self.model.encoder.page_encoder.prepare_input(
                    img, self.bucket == Bucket.Training
                ).unsqueeze(0)
            )

            ground_truth.append(
                {
                    "doc_id": str(sample["doc_id"] - doc_id_offset),
                    "doc_class": str(sample["doc_class"]),
                    "page_nr": str(sample["page_nr"]),
                }
            )

        pixel_values = torch.cat(page_tensors)

        target_sequence = (
            self.task_start_token
            + self.model.json2token(
                ground_truth,
                sort_json_key=self.sort_json_key,
                update_special_tokens_for_json_key=False,
            )
            + self.model.decoder.tokenizer.eos_token
        )

        input_ids: torch.Tensor = self.model.decoder.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.model.decoder.model.config.max_position_embeddings,  # type: ignore
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]  # type: ignore
        input_ids = input_ids.squeeze(0)

        labels = input_ids.clone()
        labels[
            labels == self.model.decoder.tokenizer.pad_token_id
        ] = self.ignore_id  # model doesn't need to predict pad token
        labels[
            : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
        ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)

        prompt_end_index = torch.nonzero(input_ids == self.prompt_end_token_id).sum()

        return TransformerSample(
            pixel_values=pixel_values,
            decoder_input_ids=input_ids,
            prompt_end_index=prompt_end_index,
            decoder_labels=labels,
            target_sequence=target_sequence,
            ground_truth=ground_truth,
        )
