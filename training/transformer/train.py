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


N_EPOCHS = 2
MAX_PAGES = 8
NUM_WORKERS = 1

IMAGE_SIZE  = (420, 360)
PRETRAINED_ENCODER = "/data/training/master_thesis/models/swin-encoder-tiny/model.bin"

MAX_LENGTH = 768



# Define Model
config = MultipageTransformerConfig(
    
    max_pages=MAX_PAGES,
    input_size=IMAGE_SIZE,
    pretrained_encoder=PRETRAINED_ENCODER,

    decoder_cfg=MBartConfig(d_model=768)
)

model = MultipageTransformerPLModule(config)

data_module = MultipagePLDataModule(Path(DATASET_PATH), model.model, num_workers=NUM_WORKERS)

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
    devices=[0],
    logger=logger,
    max_epochs=N_EPOCHS,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, data_module)

torch.save(model, Path(LIGHTNING_PATH) / "last_model.ckpt")