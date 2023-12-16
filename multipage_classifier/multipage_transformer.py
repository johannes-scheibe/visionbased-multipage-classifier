import os
import re
from typing import Any

import torch
import torch.nn as nn
from multipage_classifier.encoder.multipage_encoder import MultipageEncoder
from multipage_classifier.encoder.swin_encoder import SwinEncoder, SwinEncoderConfig
from pydantic import BaseModel
from torchvision import transforms
from multipage_classifier.decoder.donut_decoder import BARTDecoder


class MultipageTransformerConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        
    max_pages: int = 64
    max_seq_len: int = 768

    detached: bool = False

    encoder_cfg: SwinEncoderConfig | None = None
    pretrained_encoder: str | None = None

    max_position_embeddings: int | None = None
    decoder_layer: int = 4
    decoder_name_or_path: str | None = None
    special_tokens: list[str] = []


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

        assert self.config.pretrained_encoder or self.config.encoder_cfg

        if self.config.pretrained_encoder:
            page_encoder = torch.load(self.config.pretrained_encoder)
        else:
            page_encoder = SwinEncoder(
                config.encoder_cfg # type: ignore
            )

        self.encoder = MultipageEncoder(page_encoder, self.config.max_pages, self.config.detached)

        self.decoder = BARTDecoder(
            decoder_layer=self.config.decoder_layer,
            hidden_dim=self.encoder.hidden_dim,
            max_position_embeddings=self.config.max_seq_len if self.config.max_position_embeddings is None else self.config.max_position_embeddings,
            special_tokens=self.config.special_tokens,
            name_or_path=self.config.decoder_name_or_path,
        )
    def forward(
        self,
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_labels: torch.Tensor,
    ):  
        encoder_outputs = self.encoder.forward(image_tensors[0]).unsqueeze(0)
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
        encoder_outputs = self.encoder.forward(image_tensors)

        if len(encoder_outputs.size()) == 1:
            encoder_outputs = encoder_outputs.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)


        # get decoder output
        decoder_output = self.decoder.model.generate(
            input_ids=prompt_tensors,
            encoder_hidden_states=encoder_outputs,
            max_length=self.config.max_seq_len,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        output = {"predictions": list()}
        for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(
                self.decoder.tokenizer.pad_token, ""
            )
            seq = re.sub(
                r"<.*?>", "", seq, count=1
            ).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        return output


    def json2token(
        self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ) -> str:
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
                        self.decoder.add_special_tokens([rf"<s_{k}>", rf"</s_{k}>"])
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
            if f"<{obj}/>" in self.decoder.tokenizer.all_special_tokens:
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
                                leaf in self.decoder.tokenizer.get_added_vocab()
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
