from multipage_classifier.encoder.swin_encoder import SwinEncoderConfig
import pytorch_lightning as pl
from multipage_classifier.multipage_transformer import MultipageTransformerConfig
from training.transformer.lightning_module import (
    MultipageTransformerPLModule,
    MultipagePLDataModule,
)
from transformers import AutoConfig, DonutSwinModel, Swinv2Model, SwinModel


NAME = "multipage_transformer"

DATASET_PATH = "../dataset/ucsf-idl-resized-without_emails/"
DS_FILE = "IDL-less_20_pages.json"

N_EPOCHS = 50
MAX_PAGES = 16

NUM_WORKERS = 1

MAX_LENGTH = 768
IMAGE_SIZE = (640, 896)
PRETRAINED_ENCODER = ""

# Define Model
config = MultipageTransformerConfig(
    max_pages=MAX_PAGES,
    input_size=IMAGE_SIZE,
    encoder_cfg=SwinEncoderConfig(
        image_size=IMAGE_SIZE,
        pretrained_model_name_or_path=PRETRAINED_ENCODER,
        pretrained_model_type=SwinModel,
    ),
)
model = MultipageTransformerPLModule(config)

data_module = MultipagePLDataModule(DATASET_PATH, DS_FILE, model.model)

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

logger = TensorBoardLogger("lightning_logs", name=NAME)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    logger=logger,
    max_epochs=N_EPOCHS,
    callbacks=[checkpoint_callback],
    num_sanity_val_steps=0,
)

trainer.fit(model, data_module)
