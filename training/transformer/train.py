from pathlib import Path

import torch
from multipage_classifier.encoder.swin_encoder import SwinEncoderConfig
import pytorch_lightning as pl
from multipage_classifier.multipage_transformer import MultipageTransformerConfig
from training.transformer.lightning_module import (
    MultipageTransformerPLModule,
    MultipagePLDataModule,
)
from transformers import AutoConfig, DonutSwinModel, Swinv2Model, SwinModel

from transformers import MBartConfig, MBartForCausalLM, MBartTokenizer

NAME = "multipage_transformer"

DATASET_PATH = "/data/training/master_thesis/datasets/2023-05-23"
CLASS_PATH = "/data/training/master_thesis/datasets/bzuf_classes.json"
LIGHTNING_PATH = "/data/training/master_thesis/lightning_logs"


N_EPOCHS = 30
MAX_PAGES = 16
NUM_WORKERS = 4

TASK_PROMPT = "<s_classification>"

PRETRAINED_ENCODER = "/data/training/master_thesis/models/swin-encoder-pretrained/model.bin"
MAX_LENGTH = 768
DETACHED = True


special_tokens = [TASK_PROMPT]
for k in ["doc_id", "doc_class", "page_nr"]:
    special_tokens.extend([rf"<s_{k}>", rf"</s_{k}>"])

# Define Model
config = MultipageTransformerConfig(
    max_pages=MAX_PAGES,
    pretrained_encoder=PRETRAINED_ENCODER,
    detached=DETACHED,
    special_tokens=special_tokens
)

model = MultipageTransformerPLModule(config)

data_module = MultipagePLDataModule(Path(DATASET_PATH), model.model,  task_prompt=TASK_PROMPT, num_workers=NUM_WORKERS) 

data_module.prepare_data()
data_module.setup() # ensure tokens are configured 

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Configure checkpointing
checkpoint_callback = ModelCheckpoint(
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_metric",
    mode="min",
    save_last=True,
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "checkpoint-{epoch:02d}-{val_loss:.4f}"

logger = TensorBoardLogger(LIGHTNING_PATH, name=NAME)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[1],
    logger=logger,
    max_epochs=N_EPOCHS,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, data_module)
