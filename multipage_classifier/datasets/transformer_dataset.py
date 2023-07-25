from pathlib import Path
import random
from typing import Any, List, Tuple

from multipage_classifier.multipage_transformer import MultipageTransformer
import torch

from PIL import Image
from torch.utils.data import Dataset


added_tokens = []


class TransformerDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        labels: List[dict],
        model: MultipageTransformer,
        max_pages: int,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str | None = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.model = model

        self.dataset_path = dataset_path
        self.labels = labels

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key

        self.max_pages = max_pages
        total_num_pages = sum(item["pages"] for item in self.labels)
        self.dataset_length = int(total_num_pages / self.max_pages)

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
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """

        pixel_values = []
        ground_truth = []

        doc_id = 0

        while True:
            i = random.randint(0, len(self.labels) - 1)

            label = self.labels[i]

            if len(pixel_values) + label["pages"] > self.max_pages:
                if len(pixel_values) == 0:
                    continue
                break

            img_folder = label["image_folder"]

            for page_idx in range(label["pages"]):
                file = f"page_{page_idx}.jpg"
                page_path = self.dataset_path / img_folder / file

                image = Image.open(page_path)
                img_tensor = self.model.encoder.page_encoder.prepare_input(
                    image, random_padding=self.split == "train"
                ).unsqueeze(0)
                pixel_values.append(img_tensor)

                ground_truth.append(
                    {
                        "doc_id": doc_id,
                        "doc_class": label["type"],
                        "page_nr": page_idx,
                    }
                )

            doc_id += 1

        pixel_values = torch.cat(pixel_values)

        target_sequence = (
            self.task_start_token
            + self.model.json2token(
                ground_truth,
                sort_json_key=self.sort_json_key,
            )
            + self.model.tokenizer.eos_token
        )

        input_ids = self.model.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.model.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return pixel_values, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return pixel_values, input_ids, prompt_end_index, target_sequence
