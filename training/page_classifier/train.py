
from pathlib import Path

import pytorch_lightning as pl
from lightning_module import MultipageClassifierPLModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from multipage_classifier.datasets.ucsf_dataset import UCSFDataModule
from multipage_classifier.page_classifier import (MultipageClassifier,
                                                  MultipageClassifierConfig)

NAME = "multipage_classifier"

DATASET_PATH = "../dataset/ucsf-idl-resized-without_emails/"
DS_FILE = "IDL-less_20_pages.json"

N_EPOCHS = 100
MAX_PAGES = 10
NUM_WORKERS = 8

IMAGE_SIZE  = [512, 704] 

# Define Model
config = MultipageClassifierConfig(
    input_size=IMAGE_SIZE,
    max_pages=MAX_PAGES
)
model = MultipageClassifierPLModule(config)

data_module = UCSFDataModule(
    Path(DATASET_PATH),
    DS_FILE,
    # TODO maybe call this in PLModule
    prepare_function=model.classifier.prepare_input,
    split=[0.8, 0.2],
    max_pages=MAX_PAGES,
    num_workers=NUM_WORKERS
)

# Configure checkpointing
checkpoint_callback = ModelCheckpoint(
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
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
    num_sanity_val_steps=0
)

trainer.fit(model, data_module)
