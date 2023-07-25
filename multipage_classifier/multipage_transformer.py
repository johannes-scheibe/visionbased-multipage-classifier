import os
import re
from typing import Any

import torch
import torch.nn as nn
from multipage_classifier.decoder.bart_decoder import MBartDecoder
from multipage_classifier.encoder.multipage_encoder import MultipageEncoder
from multipage_classifier.encoder.swin_encoder import SwinEncoder, SwinEncoderConfig
from pydantic import BaseModel
from torchvision import transforms
from transformers import MBartConfig, MBartForCausalLM, MBartTokenizer


class MultipageTransformerConfig(BaseModel):
    input_size: tuple[int, int] = (2560, 1920)
    max_pages: int = 64
    max_seq_len: int = 768

    encoder_cfg: SwinEncoderConfig


class MultipageTransformer(nn.Module):
    def __init__(self, config: MultipageTransformerConfig):
        super().__init__()

        self.config = config

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        page_encoder = SwinEncoder(self.config.encoder_cfg)
        self.encoder = MultipageEncoder(page_encoder, self.config.max_pages)

        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")

        self.decoder = MBartDecoder(config=MBartConfig(d_model=768))

    def forward(
        self,
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_labels: torch.Tensor,
    ):
        encoder_outputs = self.encoder(image_tensors)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            labels=decoder_labels,
        )
        return decoder_outputs

    def inference(
        self,
        image_tensors: torch.Tensor,
        prompt_tensors: torch.Tensor,
        return_json: bool = True,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format
        Args:
            image_tensors: (1, num_channels, height, width)
                            convert prompt to tensor if image_tensor is not fed
            prompt: task prompt (string) to guide Donut Decoder generation
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        encoder_last_hidden_state = self.encoder(image_tensors[0])  # TODO see above

        if len(encoder_last_hidden_state.size()) == 1:
            encoder_last_hidden_state = encoder_last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # get decoder output
        assert isinstance(self.decoder.model, MBartForCausalLM)

        decoder_output = self.decoder.model.generate(
            input_ids=prompt_tensors,
            encoder_hidden_states=encoder_last_hidden_state,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
        )

        output = {"predictions": list()}
        for seq in self.tokenizer.batch_decode(decoder_output):
            seq = seq.replace(self.tokenizer.eos_token, "").replace(
                self.tokenizer.pad_token, ""
            )
            seq = re.sub(
                r"<.*?>", "", seq, count=1
            ).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        return output

    def add_tokens(self, list_of_tokens: list[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.decoder.model.resize_token_embeddings(len(self.tokenizer))

    def add_special_tokens(self, list_of_tokens: list[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self.decoder.model.resize_token_embeddings(len(self.tokenizer))

    def json2token(
        self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_special_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(
                            obj[k], update_special_tokens_for_json_key, sort_json_key
                        )
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [
                    self.json2token(
                        item, update_special_tokens_for_json_key, sort_json_key
                    )
                    for item in obj
                ]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.tokenizer.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(rf"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}",
                    tokens,
                    re.IGNORECASE,
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                leaf in self.tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}
