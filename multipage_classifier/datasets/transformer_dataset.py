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


added_tokens = []

class Page(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    pixel_values: torch.Tensor
    decoder_input_ids: torch.Tensor
    prompt_end_index: torch.Tensor
    decoder_labels: torch.Tensor
    target_sequence: str

class TransformerDataset(Dataset):
    sample_info_file_name: str = "sample.json"
    def __init__(
        self,
        path: Path, 
        bucket: Bucket,
        model: MultipageTransformer,

        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str | None = None,
        sort_json_key: bool = True,
    ):
        super().__init__()


        self.path = path
        self.bucket = bucket

        with (self.path / f"{bucket.value}.txt").open("r") as file:
            self.inventory = [Path(line.rstrip()) for line in file.readlines()]

        self.model = model

        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        # TODO dont hard code the tokens here
        self.add_tokens([rf"<s_{k}>" for k in ["doc_id", "doc_class", "page_nr"]])
        self.add_tokens([rf"</s_{k}>" for k in ["doc_id", "doc_class", "page_nr"]])
        self.prompt_end_token_id = model.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

    def add_tokens(self, list_of_tokens: List[str]):
        self.model.add_tokens(list_of_tokens)
        added_tokens.extend(list_of_tokens)

    def __len__(self):
        return 50
        return len(self.inventory)

    def __getitem__(self, idx: int) -> list[Page]:
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
            
            pages = predicted_doc["pages"]
            if len(pages) == 0:
                pages = [{"sourcePage": i} for i in range(len(document["pages"]))]

            for dst_page, page in enumerate(pages):
                src_page = page.get("sourcePage", 0)  # NOTE the default value is 0
                page_bytes = sample_data[f"page_{src_page}.png"]
                
                img = Image.open(io.BytesIO(page_bytes))
                
                pixel_values: torch.Tensor = self.model.encoder.page_encoder.prepare_input(
                    img, self.bucket == Bucket.Training
                ).squeeze()

                ground_truth = {
                        "doc_id": doc_id,
                        "doc_class": class_identifier,
                        "page_nr": dst_page,
                    }
                
                target_sequence = (
                    self.task_start_token
                    + self.model.json2token(
                        ground_truth,
                        sort_json_key=self.sort_json_key,
                        update_special_tokens_for_json_key=False
                    )
                    + self.model.tokenizer.eos_token
                )

                input_ids: torch.Tensor = self.model.tokenizer(
                    target_sequence,
                    add_special_tokens=False,
                    max_length=self.model.decoder.model.config.max_length,  # type: ignore
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]  # type: ignore
                input_ids = input_ids.squeeze(0)

                labels = input_ids.clone()
                labels[
                    labels == self.model.tokenizer.pad_token_id
                ] = self.ignore_id  # model doesn't need to predict pad token
                labels[
                    : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
                ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)

                prompt_end_index = torch.nonzero(input_ids == self.prompt_end_token_id).sum()
                
                batch.append(Page(
                    pixel_values=pixel_values,
                    decoder_input_ids=input_ids,
                    prompt_end_index=prompt_end_index,
                    decoder_labels=labels,
                    target_sequence=target_sequence
                ))

        return batch
